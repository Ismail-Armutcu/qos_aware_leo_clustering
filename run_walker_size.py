from __future__ import annotations

"""
IDE-friendly Walker-delta constellation size study.

This study forces:
- multisat.source = "walker"

and sweeps:
- walker.total_sats
- walker.n_planes
- multiple fixed snapshot times (walker_time_utc_iso_list)

with valid Walker pairs only, i.e. total_sats % n_planes == 0.

Outputs:
- raw run CSV
- summary-by-time CSV
- summary-over-time CSV
- tables txt
- interaction heatmaps by time
- interaction heatmaps aggregated over time
- one-factor line plots aggregated over time for selected plane families

This file is standalone and does NOT modify other scripts.
"""

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import os
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ScenarioConfig
from src.pipeline import run_scenario


# =============================================================================
# SETTINGS
# =============================================================================
SETTINGS: dict[str, Any] = {
    # Parallel execution
    "backend": "process",  # "process" | "thread"
    "max_workers": max(1, (os.cpu_count() or 4) - 1),

    # Output
    "out_dir": "walker_size_results",

    # Base scenario
    "region_mode": "turkey",
    "enable_plots": False,
    "verbose": False,
    "enable_fastbp_baselines": False,

    # Study scope
    "n_users_list": [1000, 2000, 3000],
    "seeds": [1, 2, 3, 4, 5],

    # Walker interaction study
    "total_sats_values": [72, 96, 120, 144, 168, 192],
    "n_planes_values": [6, 8, 12],

    # One-factor presentation study (all selected plane families)
    "fixed_n_planes_for_lines_list": [6, 8, 12],

    # Fixed Walker parameters
    "walker_phasing": 1,
    "walker_inclination_deg": 53.0,
    "walker_altitude_m": 600_000.0,
    "walker_epoch_utc_iso": "2026-01-01T00:00:00Z",

    # Multiple fixed snapshot times for temporal robustness
    "walker_time_utc_iso_list": [
        "2026-01-01T00:00:00Z",
        "2026-01-01T06:00:00Z",
        "2026-01-01T12:00:00Z",
        "2026-01-01T18:00:00Z",
    ],
}


# =============================================================================
# Helpers
# =============================================================================
def _deep_replace(obj: Any, updates: dict[str, Any]) -> Any:
    """Nested dataclass replace using dotted keys."""
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
        region_mode=str(SETTINGS["region_mode"]),
        run=replace(
            cfg.run,
            enable_plots=bool(SETTINGS["enable_plots"]),
            verbose=bool(SETTINGS["verbose"]),
            enable_fastbp_baselines=bool(SETTINGS["enable_fastbp_baselines"]),
        ),
        multisat=replace(
            cfg.multisat,
            source="walker",
            time_utc_iso=None,  # overridden per task
            walker=replace(
                cfg.multisat.walker,
                phasing=int(SETTINGS["walker_phasing"]),
                inclination_deg=float(SETTINGS["walker_inclination_deg"]),
                altitude_m=float(SETTINGS["walker_altitude_m"]),
                epoch_utc_iso=str(SETTINGS["walker_epoch_utc_iso"]),
            ),
        ),
    )
    return cfg


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _valid_pairs(total_sats_values: list[int], n_planes_values: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for total_sats in total_sats_values:
        for n_planes in n_planes_values:
            if int(n_planes) <= 0:
                continue
            if int(total_sats) % int(n_planes) == 0:
                pairs.append((int(total_sats), int(n_planes)))
    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs


def _sanitize_for_filename(s: str) -> str:
    out = s.replace(":", "-").replace("/", "_").replace(" ", "_")
    out = out.replace("T", "_").replace("Z", "")
    return out


def _extract_metrics(res: dict[str, Any]) -> dict[str, Any]:
    main = res.get("main_ref_lb", {}) or {}

    runtime_total = (
        _safe_float(res.get("time_usergen_s", 0.0))
        + _safe_float(res.get("time_sat_select_s", 0.0))
        + _safe_float(res.get("time_assoc_s", 0.0))
        + _safe_float(res.get("time_split_s", 0.0))
        + _safe_float(res.get("time_ent_ref_s", 0.0))
        + _safe_float(res.get("time_lb_ref_s", 0.0))
    )

    best_m = res.get("payload_best_m", res.get("ms_n_active"))
    if best_m is None:
        best_m = res.get("ms_n_active")

    return {
        "payload_feasible": int(bool(res.get("payload_feasible", False))),
        "K": _safe_float(main.get("K")),
        "U_mean": _safe_float(main.get("U_mean")),
        "U_max": _safe_float(main.get("U_max")),
        "ent_edge_pct": _safe_float(main.get("ent_edge_pct")),
        "best_m": _safe_float(best_m),
        "ms_n_active": _safe_float(res.get("ms_n_active")),
        "payload_n_viol_K": _safe_float(res.get("payload_n_viol_K")),
        "payload_n_viol_T": _safe_float(res.get("payload_n_viol_T")),
        "payload_K_over_sum": _safe_float(res.get("payload_K_over_sum")),
        "payload_T_over_sum": _safe_float(res.get("payload_T_over_sum")),
        "runtime_total_s": runtime_total,
    }


# =============================================================================
# Worker
# =============================================================================
def _worker(task: dict[str, Any]) -> dict[str, Any]:
    seed = int(task["seed"])
    n_users = int(task["n_users"])
    total_sats = int(task["total_sats"])
    n_planes = int(task["n_planes"])
    snapshot_time_utc_iso = str(task["snapshot_time_utc_iso"])
    updates = dict(task["updates"])

    cfg = _make_base_cfg()
    cfg = replace(cfg, run=replace(cfg.run, seed=seed, n_users=n_users))
    cfg = _deep_replace(cfg, updates)

    try:
        res = run_scenario(cfg)
        return {
            "ok": 1,
            "seed": seed,
            "n_users": n_users,
            "total_sats": total_sats,
            "n_planes": n_planes,
            "sats_per_plane": int(total_sats // n_planes),
            "snapshot_time_utc_iso": snapshot_time_utc_iso,
            **_extract_metrics(res),
        }
    except Exception as e:
        return {
            "ok": 0,
            "seed": seed,
            "n_users": n_users,
            "total_sats": total_sats,
            "n_planes": n_planes,
            "sats_per_plane": int(total_sats // n_planes),
            "snapshot_time_utc_iso": snapshot_time_utc_iso,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "payload_feasible": 0,
            "K": float("nan"),
            "U_mean": float("nan"),
            "U_max": float("nan"),
            "ent_edge_pct": float("nan"),
            "best_m": float("nan"),
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
        total = len(fut_map)
        done = 0
        for fut in as_completed(fut_map):
            done += 1
            results.append(fut.result())
            print(f"[{done}/{total}] finished")

    results.sort(
        key=lambda r: (
            str(r.get("snapshot_time_utc_iso", "")),
            int(r.get("total_sats", 0)),
            int(r.get("n_planes", 0)),
            int(r.get("n_users", 0)),
            int(r.get("seed", 0)),
        )
    )
    return results


# =============================================================================
# Task building
# =============================================================================
def _build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    pairs = _valid_pairs(
        list(SETTINGS["total_sats_values"]),
        list(SETTINGS["n_planes_values"]),
    )
    time_list = list(SETTINGS["walker_time_utc_iso_list"])
    if not time_list:
        raise RuntimeError("walker_time_utc_iso_list is empty.")

    for snapshot_time_utc_iso in time_list:
        for total_sats, n_planes in pairs:
            updates = {
                "multisat.walker.total_sats": int(total_sats),
                "multisat.walker.n_planes": int(n_planes),
                "multisat.time_utc_iso": str(snapshot_time_utc_iso),
            }
            for n_users in SETTINGS["n_users_list"]:
                for seed in SETTINGS["seeds"]:
                    tasks.append(
                        {
                            "total_sats": int(total_sats),
                            "n_planes": int(n_planes),
                            "n_users": int(n_users),
                            "seed": int(seed),
                            "snapshot_time_utc_iso": str(snapshot_time_utc_iso),
                            "updates": updates,
                        }
                    )
    return tasks


# =============================================================================
# Aggregation
# =============================================================================
def _aggregate_stats(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{c}_mean"] = float("nan")
                row[f"{c}_median"] = float("nan")
                row[f"{c}_std"] = float("nan")
                row[f"{c}_p10"] = float("nan")
                row[f"{c}_p90"] = float("nan")
                row[f"{c}_min"] = float("nan")
                row[f"{c}_max"] = float("nan")
                row[f"{c}_n"] = 0
            else:
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_median"] = float(vals.median())
                row[f"{c}_std"] = float(vals.std(ddof=0))
                row[f"{c}_p10"] = float(vals.quantile(0.10))
                row[f"{c}_p90"] = float(vals.quantile(0.90))
                row[f"{c}_min"] = float(vals.min())
                row[f"{c}_max"] = float(vals.max())
                row[f"{c}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# CSV / tables helpers
# =============================================================================
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    return pd.read_csv(path)


def _write_tables_from_csvs(out_dir: Path, fixed_n_planes_list: list[int]) -> None:
    by_time = _read_csv(out_dir / "walker_size_summary.csv")
    overall = _read_csv(out_dir / "walker_size_summary_over_time.csv")

    lines: list[str] = []
    lines.append("WALKER SIZE STUDY TABLES")
    lines.append("")
    lines.append("Aggregation note:")
    lines.append("- payload_feasible is best interpreted through mean")
    lines.append("- continuous metrics are best interpreted through median and p10/p90")
    lines.append("- by-time summary aggregates over seeds at each fixed snapshot time")
    lines.append("- over-time summary aggregates over all runs across all fixed snapshot times and seeds")
    lines.append("")

    def add_table(title: str, df: pd.DataFrame):
        lines.append("=" * 150)
        lines.append(title)
        lines.append("")
        if df.empty:
            lines.append("(empty)")
            lines.append("")
            return
        lines.append(df.to_string(index=False))
        lines.append("")

    by_time_cols = [
        "snapshot_time_utc_iso",
        "total_sats", "n_planes", "sats_per_plane", "n_users",
        "payload_feasible_mean",
        "K_median", "best_m_median", "U_mean_median", "U_max_median",
        "ent_edge_pct_median", "runtime_total_s_median",
        "payload_T_over_sum_median",
    ]
    by_time_cols = [c for c in by_time_cols if c in by_time.columns]
    add_table("INTERACTION SUMMARY BY TIME", by_time[by_time_cols].copy() if not by_time.empty else by_time)

    overall_cols = [
        "total_sats", "n_planes", "sats_per_plane", "n_users",
        "payload_feasible_mean",
        "K_median", "best_m_median", "U_mean_median", "U_max_median",
        "ent_edge_pct_median", "runtime_total_s_median",
        "payload_T_over_sum_median",
    ]
    overall_cols = [c for c in overall_cols if c in overall.columns]
    add_table("INTERACTION SUMMARY OVER TIME", overall[overall_cols].copy() if not overall.empty else overall)

    for fixed_n_planes in fixed_n_planes_list:
        one_factor = overall[overall["n_planes"] == int(fixed_n_planes)].copy()
        one_factor_cols = [
            "total_sats", "n_users",
            "payload_feasible_mean",
            "K_median", "best_m_median", "U_mean_median", "U_max_median",
            "ent_edge_pct_median", "runtime_total_s_median",
            "payload_T_over_sum_median",
        ]
        one_factor_cols = [c for c in one_factor_cols if c in one_factor.columns]
        add_table(
            f"ONE-FACTOR SUMMARY OVER TIME (n_planes = {fixed_n_planes})",
            one_factor[one_factor_cols].copy() if not one_factor.empty else one_factor,
        )

    (out_dir / "walker_size_tables.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Plot helpers
# =============================================================================
def _annotate_heatmap(ax, data: np.ndarray) -> None:
    if data.size == 0:
        return
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                if abs(v) >= 100 or float(v).is_integer():
                    txt = f"{v:.0f}"
                elif abs(v) >= 10:
                    txt = f"{v:.1f}"
                else:
                    txt = f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)


def _plot_interaction_heatmaps_by_time_from_csvs(out_dir: Path) -> None:
    agg = _read_csv(out_dir / "walker_size_summary.csv")
    if agg.empty:
        return

    metric_specs = [
        ("payload_feasible_mean", "Feasible rate"),
        ("K_median", "K"),
        ("best_m_median", "Minimum feasible prefix"),
        ("U_mean_median", "U_mean"),
        ("ent_edge_pct_median", "Enterprise edge exposure (%)"),
        ("runtime_total_s_median", "Runtime (s)"),
    ]

    plot_dir = out_dir / "interaction_plots" / "by_time"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for snapshot_time_utc_iso, gt in agg.groupby("snapshot_time_utc_iso"):
        time_tag = _sanitize_for_filename(str(snapshot_time_utc_iso))
        for n_users, g in gt.groupby("n_users"):
            g = g.copy()
            g["total_sats"] = pd.to_numeric(g["total_sats"], errors="coerce")
            g["n_planes"] = pd.to_numeric(g["n_planes"], errors="coerce")
            g = g.sort_values(["n_planes", "total_sats"])

            for metric, title_metric in metric_specs:
                if metric not in g.columns:
                    continue

                pivot = g.pivot(index="n_planes", columns="total_sats", values=metric)
                if pivot.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7.0, 5.0))
                im = ax.imshow(pivot.values, aspect="auto", origin="lower")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(int(c)) for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([str(int(v)) for v in pivot.index])
                ax.set_xlabel("total_sats")
                ax.set_ylabel("n_planes")
                ax.set_title(f"{title_metric}, n_users={int(n_users)}, t={snapshot_time_utc_iso}")
                fig.colorbar(im, ax=ax)
                _annotate_heatmap(ax, pivot.values)
                plt.tight_layout()
                plt.savefig(
                    plot_dir / f"walker_interaction_{metric}_n{int(n_users)}_{time_tag}.png",
                    dpi=180,
                )
                plt.close()


def _plot_interaction_heatmaps_over_time_from_csvs(out_dir: Path) -> None:
    agg = _read_csv(out_dir / "walker_size_summary_over_time.csv")
    if agg.empty:
        return

    metric_specs = [
        ("payload_feasible_mean", "Feasible rate"),
        ("K_median", "K"),
        ("best_m_median", "Minimum feasible prefix"),
        ("U_mean_median", "U_mean"),
        ("ent_edge_pct_median", "Enterprise edge exposure (%)"),
        ("runtime_total_s_median", "Runtime (s)"),
    ]

    plot_dir = out_dir / "interaction_plots" / "overall"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for n_users, g in agg.groupby("n_users"):
        g = g.copy()
        g["total_sats"] = pd.to_numeric(g["total_sats"], errors="coerce")
        g["n_planes"] = pd.to_numeric(g["n_planes"], errors="coerce")
        g = g.sort_values(["n_planes", "total_sats"])

        for metric, title_metric in metric_specs:
            if metric not in g.columns:
                continue

            pivot = g.pivot(index="n_planes", columns="total_sats", values=metric)
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(c)) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(int(v)) for v in pivot.index])
            ax.set_xlabel("total_sats")
            ax.set_ylabel("n_planes")
            ax.set_title(f"{title_metric}, n_users={int(n_users)} (aggregated over time)")
            fig.colorbar(im, ax=ax)
            _annotate_heatmap(ax, pivot.values)
            plt.tight_layout()
            plt.savefig(plot_dir / f"walker_interaction_over_time_{metric}_n{int(n_users)}.png", dpi=180)
            plt.close()


def _plot_one_factor_lines_over_time_from_csvs(out_dir: Path, fixed_n_planes_list: list[int]) -> None:
    agg = _read_csv(out_dir / "walker_size_summary_over_time.csv")
    if agg.empty:
        return

    plot_dir = out_dir / "onefactor_plots" / "overall"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("payload_feasible_mean", "Feasible rate", True, False),
        ("K_median", "K", False, False),
        ("best_m_median", "Minimum feasible prefix", False, False),
        ("U_mean_median", "U_mean", False, False),
        ("U_max_median", "U_max", False, False),
        ("ent_edge_pct_median", "Enterprise edge exposure (%)", False, False),
        ("runtime_total_s_median", "Runtime (s)", False, True),
    ]

    for fixed_n_planes in fixed_n_planes_list:
        g = agg[agg["n_planes"] == int(fixed_n_planes)].copy()
        if g.empty:
            continue

        g["total_sats"] = pd.to_numeric(g["total_sats"], errors="coerce")

        for metric, ylabel, use_mean_style, y_log in metric_specs:
            if metric not in g.columns:
                continue

            plt.figure(figsize=(7.4, 4.8))
            for n_users, gg in g.groupby("n_users"):
                gg = gg.sort_values("total_sats")
                x = gg["total_sats"].to_numpy(dtype=float)
                y = pd.to_numeric(gg[metric], errors="coerce").to_numpy(dtype=float)
                plt.plot(x, y, marker="o", label=f"n={int(n_users)}")

                if not use_mean_style:
                    low_col = metric.replace("_median", "_p10")
                    high_col = metric.replace("_median", "_p90")
                    if low_col in gg.columns and high_col in gg.columns:
                        ylo = pd.to_numeric(gg[low_col], errors="coerce").to_numpy(dtype=float)
                        yhi = pd.to_numeric(gg[high_col], errors="coerce").to_numpy(dtype=float)
                        if np.isfinite(ylo).any() and np.isfinite(yhi).any():
                            plt.fill_between(x, ylo, yhi, alpha=0.15)

            plt.xlabel("total_sats")
            plt.ylabel(ylabel)
            if y_log:
                plt.yscale("log")
            plt.title(f"{ylabel} vs total_sats (n_planes={fixed_n_planes}, aggregated over time)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"walker_onefactor_over_time_{metric}_fixedP{fixed_n_planes}.png", dpi=180)
            plt.close()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    out_dir = Path(str(SETTINGS["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _valid_pairs(
        list(SETTINGS["total_sats_values"]),
        list(SETTINGS["n_planes_values"]),
    )

    if not pairs:
        raise RuntimeError("No valid Walker (total_sats, n_planes) pairs found.")

    time_list = list(SETTINGS["walker_time_utc_iso_list"])
    if not time_list:
        raise RuntimeError("walker_time_utc_iso_list is empty.")

    fixed_n_planes_list = [int(x) for x in SETTINGS["fixed_n_planes_for_lines_list"]]
    if not fixed_n_planes_list:
        raise RuntimeError("fixed_n_planes_for_lines_list is empty.")

    tasks = _build_tasks()
    rows = _run_tasks(tasks)

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(out_dir / "walker_size_runs.csv", index=False)

    value_cols = [
        "ok",
        "payload_feasible",
        "K",
        "U_mean",
        "U_max",
        "ent_edge_pct",
        "best_m",
        "ms_n_active",
        "payload_n_viol_K",
        "payload_n_viol_T",
        "payload_K_over_sum",
        "payload_T_over_sum",
        "runtime_total_s",
    ]

    # Summary at each fixed snapshot time (aggregate over seeds)
    summary_by_time = _aggregate_stats(
        df_runs,
        ["snapshot_time_utc_iso", "total_sats", "n_planes", "sats_per_plane", "n_users"],
        value_cols,
    )
    summary_by_time.to_csv(out_dir / "walker_size_summary.csv", index=False)

    # Summary aggregated over all snapshot times and seeds
    summary_over_time = _aggregate_stats(
        df_runs,
        ["total_sats", "n_planes", "sats_per_plane", "n_users"],
        value_cols,
    )
    summary_over_time.to_csv(out_dir / "walker_size_summary_over_time.csv", index=False)

    _write_tables_from_csvs(out_dir, fixed_n_planes_list=fixed_n_planes_list)
    _plot_interaction_heatmaps_by_time_from_csvs(out_dir)
    _plot_interaction_heatmaps_over_time_from_csvs(out_dir)
    _plot_one_factor_lines_over_time_from_csvs(out_dir, fixed_n_planes_list=fixed_n_planes_list)

    print(f"[done] wrote Walker size study outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()