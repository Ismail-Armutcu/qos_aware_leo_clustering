from __future__ import annotations

"""
IDE-friendly parallel sensitivity runner.

Edit the SETTINGS block below and run this file directly from your IDE.
No CLI arguments are required.

This version:
- runs one-factor and interaction sweeps in parallel
- supports process/thread backends through the SETTINGS block
- preserves plain interaction metadata columns (e.g. theta_3db_deg, Ks_max)
  so interaction aggregation/heatmaps work correctly
- includes traffic sensitivity and traffic stress-test sweeps
"""

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import os
import traceback

import pandas as pd
import matplotlib.pyplot as plt

from config import ScenarioConfig
from src.pipeline import run_scenario


# =============================================================================
# SETTINGS — edit these directly and press Run in your IDE
# =============================================================================
SETTINGS: dict[str, Any] = {
    # Parallel backend: "process" is recommended for CPU-heavy workloads.
    "backend": "process",             # "process" | "thread"
    "max_workers": max(1, (os.cpu_count() or 4) - 1),

    # Output directory
    "out_dir": "sensitivity_results",

    # Base scenario
    "region_mode": "turkey",         # "turkey" | "debug"
    "constellation_source": "tle",   # "tle" | "walker"
    "tle_path": "starlink.tle",
    "time_utc_iso": None,             # e.g. "2026-01-29T16:19:56Z"
    "enable_plots": False,
    "verbose": False,
    "enable_fastbp_baselines": False,

    # Common sweep axes
    "n_users_list": [1000, 1500, 2000],
    "seeds": [1, 2],

    # One-factor sweeps (baseline-centered)
    "theta_3db_deg_values": [0.9, 1.0, 1.1],
    "elev_mask_deg_values": [20.0, 25.0, 30.0],
    "Ks_max_values": [192, 256, 320],
    "W_slots_values": [6, 8, 10],
    "rho_safe_values": [0.6, 0.7, 0.8],
    # hotspot_scale multiplies both sigma_min and sigma_max
    "hotspot_scale_values": [0.8, 1.0, 1.2],

    # Traffic sensitivity / stress
    "demand_mbps_median_values": [5.0, 10.0, 20.0],
    "demand_mbps_median_stress_values": [10.0, 50.0, 100.0],

    # One interaction study
    "interaction_theta_3db_deg_values": [0.9, 1.0, 1.1],
    "interaction_Ks_max_values": [192, 256, 320],
}


# =============================================================================
# Config / extraction helpers
# =============================================================================

def _deep_replace(obj: Any, updates: dict[str, Any]) -> Any:
    """Nested dataclass replace using dotted keys, e.g. beam.theta_3db_deg."""
    if not updates:
        return obj

    local = {}
    grouped: dict[str, dict[str, Any]] = {}
    for k, v in updates.items():
        if "." in k:
            head, tail = k.split(".", 1)
            grouped.setdefault(head, {})[tail] = v
        else:
            local[k] = v

    kwargs = {}
    for field_name, child_updates in grouped.items():
        child = getattr(obj, field_name)
        kwargs[field_name] = _deep_replace(child, child_updates)
    kwargs.update(local)
    return replace(obj, **kwargs)


def _make_base_cfg() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg = replace(
        cfg,
        region_mode=SETTINGS["region_mode"],
        run=replace(
            cfg.run,
            enable_plots=bool(SETTINGS["enable_plots"]),
            verbose=bool(SETTINGS["verbose"]),
            enable_fastbp_baselines=bool(SETTINGS["enable_fastbp_baselines"]),
        ),
        multisat=replace(
            cfg.multisat,
            source=str(SETTINGS["constellation_source"]),
            tle_path=str(SETTINGS["tle_path"]),
            time_utc_iso=SETTINGS["time_utc_iso"],
        ),
    )
    return cfg


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_main_metrics(res: dict[str, Any]) -> dict[str, Any]:
    main = res.get("main_ref_lb", {}) or {}
    runtime_total = (
        _safe_float(res.get("time_usergen_s", 0.0))
        + _safe_float(res.get("time_sat_select_s", 0.0))
        + _safe_float(res.get("time_assoc_s", 0.0))
        + _safe_float(res.get("time_split_s", 0.0))
        + _safe_float(res.get("time_ent_ref_s", 0.0))
        + _safe_float(res.get("time_lb_ref_s", 0.0))
    )
    return {
        "payload_feasible": int(bool(res.get("payload_feasible", False))),
        "K": _safe_float(main.get("K")),
        "U_max": _safe_float(main.get("U_max")),
        "ent_edge_pct": _safe_float(main.get("ent_edge_pct")),
        "ms_n_active": _safe_float(res.get("ms_n_active")),
        "payload_n_viol_K": _safe_float(res.get("payload_n_viol_K")),
        "payload_n_viol_T": _safe_float(res.get("payload_n_viol_T")),
        "payload_K_over_sum": _safe_float(res.get("payload_K_over_sum")),
        "payload_T_over_sum": _safe_float(res.get("payload_T_over_sum")),
        "runtime_total_s": runtime_total,
    }


# =============================================================================
# Parallel worker
# =============================================================================

def _worker(task: dict[str, Any]) -> dict[str, Any]:
    seed = int(task["seed"])
    n_users = int(task["n_users"])
    updates = dict(task["updates"])
    tag = str(task["tag"])
    x_name = str(task["x_name"])
    x_value = task["x_value"]

    # Preserve extra task metadata such as theta_3db_deg, Ks_max, hotspot_scale, etc.
    meta = {
        k: v
        for k, v in task.items()
        if k not in {"updates", "tag", "x_name", "x_value", "seed", "n_users"}
    }

    cfg = _make_base_cfg()
    cfg = replace(cfg, run=replace(cfg.run, seed=seed, n_users=n_users))
    cfg = _deep_replace(cfg, updates)

    try:
        res = run_scenario(cfg)
        return {
            "ok": 1,
            "tag": tag,
            "x_name": x_name,
            "x_value": x_value,
            "seed": seed,
            "n_users": n_users,
            **meta,
            **updates,
            **_extract_main_metrics(res),
        }
    except Exception as e:
        return {
            "ok": 0,
            "tag": tag,
            "x_name": x_name,
            "x_value": x_value,
            "seed": seed,
            "n_users": n_users,
            **meta,
            **updates,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "payload_feasible": 0,
            "K": float("nan"),
            "U_max": float("nan"),
            "ent_edge_pct": float("nan"),
            "ms_n_active": float("nan"),
            "payload_n_viol_K": float("nan"),
            "payload_n_viol_T": float("nan"),
            "payload_K_over_sum": float("nan"),
            "payload_T_over_sum": float("nan"),
            "runtime_total_s": float("nan"),
        }


def _run_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backend = str(SETTINGS["backend"]).lower()
    max_workers = int(SETTINGS["max_workers"])
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    results: list[dict[str, Any]] = []
    with Executor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(fut_map):
            results.append(fut.result())

    results.sort(key=lambda r: (
        str(r.get("tag", "")),
        str(r.get("x_name", "")),
        str(r.get("x_value", "")),
        int(r.get("n_users", 0)),
        int(r.get("seed", 0)),
    ))
    return results


# =============================================================================
# Task builders
# =============================================================================

def _build_one_factor_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    sweep_defs = [
        ("theta_3db_deg", "beam.theta_3db_deg", SETTINGS["theta_3db_deg_values"]),
        ("elev_mask_deg", "multisat.elev_mask_deg", SETTINGS["elev_mask_deg_values"]),
        ("Ks_max", "payload.Ks_max", SETTINGS["Ks_max_values"]),
        ("W_slots", "payload.W_slots", SETTINGS["W_slots_values"]),
        ("rho_safe", "ent.rho_safe", SETTINGS["rho_safe_values"]),
        ("hotspot_scale", "hotspot_scale", SETTINGS["hotspot_scale_values"]),
        ("demand_mbps_median", "traffic.demand_mbps_median", SETTINGS["demand_mbps_median_values"]),
        ("demand_mbps_median_stress", "traffic.demand_mbps_median", SETTINGS["demand_mbps_median_stress_values"]),
    ]

    for sweep_name, dotted_key, values in sweep_defs:
        for x in values:
            updates = {}
            meta = {}
            if dotted_key == "hotspot_scale":
                base = _make_base_cfg()
                updates["usergen.hotspot_sigma_m_min"] = float(base.usergen.hotspot_sigma_m_min) * float(x)
                updates["usergen.hotspot_sigma_m_max"] = float(base.usergen.hotspot_sigma_m_max) * float(x)
                meta["hotspot_scale"] = x
            else:
                updates[dotted_key] = x
                meta[sweep_name] = x

            for n_users in SETTINGS["n_users_list"]:
                for seed in SETTINGS["seeds"]:
                    tasks.append({
                        "tag": f"one_factor:{sweep_name}",
                        "x_name": sweep_name,
                        "x_value": x,
                        "n_users": n_users,
                        "seed": seed,
                        "updates": updates,
                        **meta,
                    })
    return tasks


def _build_interaction_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for theta in SETTINGS["interaction_theta_3db_deg_values"]:
        for ks in SETTINGS["interaction_Ks_max_values"]:
            updates = {
                "beam.theta_3db_deg": theta,
                "payload.Ks_max": ks,
            }
            for n_users in SETTINGS["n_users_list"]:
                for seed in SETTINGS["seeds"]:
                    tasks.append({
                        "tag": "interaction:theta_x_Ksmax",
                        "x_name": "theta_x_Ksmax",
                        "x_value": f"theta={theta},Ks={ks}",
                        "theta_3db_deg": theta,
                        "Ks_max": ks,
                        "n_users": n_users,
                        "seed": seed,
                        "updates": updates,
                    })
    return tasks


# =============================================================================
# Aggregation / plots
# =============================================================================

def _aggregate_mean(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{c}_std"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
            row[f"{c}_min"] = float(vals.min()) if len(vals) else float("nan")
            row[f"{c}_max"] = float(vals.max()) if len(vals) else float("nan")
            row[f"{c}_n"] = int(vals.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tables(summary: pd.DataFrame, interaction_summary: pd.DataFrame, out_dir: Path) -> None:
    lines: list[str] = []
    lines.append("SENSITIVITY TABLES")
    lines.append("")

    def add_table(title: str, df: pd.DataFrame):
        lines.append("=" * 100)
        lines.append(title)
        lines.append("")
        if df.empty:
            lines.append("(empty)")
            lines.append("")
            return
        lines.append(df.to_string(index=False))
        lines.append("")

    add_table("ONE-FACTOR SUMMARY", summary)
    add_table("INTERACTION SUMMARY", interaction_summary)
    (out_dir / "sensitivity_tables.txt").write_text("\n".join(lines), encoding="utf-8")


def _plot_one_factor(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return

    metrics = [
        ("payload_feasible_mean", "Feasible rate", False),
        ("K_mean", "K", False),
        ("U_max_mean", "U_max", False),
        ("ent_edge_pct_mean", "Enterprise edge exposure (%)", False),
        ("ms_n_active_mean", "Used satellites", False),
        ("runtime_total_s_mean", "Runtime (s)", True),
    ]

    for tag, g in summary.groupby("tag"):
        sweep_name = str(g["x_name"].iloc[0])
        g = g.copy()
        try:
            g["_xsort"] = pd.to_numeric(g["x_value"])
        except Exception:
            g["_xsort"] = g["x_value"].astype(str)

        for metric, ylabel, ylog in metrics:
            plt.figure(figsize=(7, 4.5))
            for n_users, gg in g.groupby("n_users"):
                gg = gg.sort_values("_xsort")
                plt.plot(gg["x_value"], gg[metric], marker="o", label=f"n={int(n_users)}")
            plt.xlabel(sweep_name)
            plt.ylabel(ylabel)
            if ylog:
                plt.yscale("log")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fname = f"{tag.replace(':','_')}_{metric}.png"
            plt.savefig(out_dir / fname, dpi=160)
            plt.close()


def _plot_interaction(interaction_summary: pd.DataFrame, out_dir: Path) -> None:
    if interaction_summary.empty:
        return

    metrics = [
        ("payload_feasible_mean", "Feasible rate"),
        ("K_mean", "K"),
        ("U_max_mean", "U_max"),
        ("runtime_total_s_mean", "Runtime (s)"),
    ]

    for n_users, g in interaction_summary.groupby("n_users"):
        for metric, title_metric in metrics:
            pivot = g.pivot(index="theta_3db_deg", columns="Ks_max", values=metric)
            plt.figure(figsize=(6, 4.8))
            im = plt.imshow(pivot.values, aspect="auto", origin="lower")
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            plt.xlabel("Ks_max")
            plt.ylabel("theta_3db_deg")
            plt.title(f"{title_metric}, n_users={int(n_users)}")
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(out_dir / f"interaction_theta_x_Ksmax_{metric}_n{int(n_users)}.png", dpi=160)
            plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    out_dir = Path(str(SETTINGS["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    # One-factor sweeps
    one_factor_tasks = _build_one_factor_tasks()
    one_factor_results = _run_tasks(one_factor_tasks)
    df_runs = pd.DataFrame(one_factor_results)
    df_runs.to_csv(out_dir / "sensitivity_runs.csv", index=False)

    value_cols = [
        "payload_feasible",
        "K",
        "U_max",
        "ent_edge_pct",
        "ms_n_active",
        "payload_n_viol_K",
        "payload_n_viol_T",
        "payload_K_over_sum",
        "payload_T_over_sum",
        "runtime_total_s",
    ]
    summary = _aggregate_mean(df_runs, ["tag", "x_name", "x_value", "n_users"], value_cols)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    # Interaction sweep
    interaction_tasks = _build_interaction_tasks()
    interaction_results = _run_tasks(interaction_tasks)
    df_inter = pd.DataFrame(interaction_results)
    df_inter.to_csv(out_dir / "sensitivity_interaction_runs.csv", index=False)

    required_cols = ["tag", "n_users", "theta_3db_deg", "Ks_max"]
    missing = [c for c in required_cols if c not in df_inter.columns]
    if missing:
        raise RuntimeError(
            f"Interaction DataFrame is missing required columns: {missing}. "
            f"Available columns: {list(df_inter.columns)}"
        )

    interaction_summary = _aggregate_mean(
        df_inter,
        ["tag", "n_users", "theta_3db_deg", "Ks_max"],
        value_cols,
    )
    interaction_summary.to_csv(out_dir / "sensitivity_interaction_summary.csv", index=False)

    _write_tables(summary, interaction_summary, out_dir)
    _plot_one_factor(summary, out_dir)
    _plot_interaction(interaction_summary, out_dir)

    print(f"[done] wrote sensitivity outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
