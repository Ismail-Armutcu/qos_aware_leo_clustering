# src/helper.py
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from config import ScenarioConfig
from src.models import Users


def _fmt_value(v: Any, key: str | None = None) -> str:
    """Pretty formatting + some unit-aware helpers based on key names."""
    if isinstance(v, float):
        if key:
            k = key.lower()
            if k.endswith("_hz"):
                if v >= 1e9:
                    return f"{v/1e9:.3f} GHz"
                if v >= 1e6:
                    return f"{v/1e6:.3f} MHz"
                if v >= 1e3:
                    return f"{v/1e3:.3f} kHz"
                return f"{v:.3f} Hz"
            if k.endswith("_m"):
                if abs(v) >= 1000.0:
                    return f"{v/1000.0:.3f} km"
                return f"{v:.3f} m"
            if k.endswith("_km"):
                return f"{v:.3f} km"
            if k.endswith("_dbw") or k.endswith("_db"):
                return f"{v:.3f} dB" if k.endswith("_db") else f"{v:.3f} dBW"
        return f"{v:.6g}"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt_value(x) for x in v) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def _flatten_dict(d: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dicts into [('a.b.c', value), ...]."""
    items: list[tuple[str, Any]] = []
    for k in sorted(d.keys()):
        v = d[k]
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            items.extend(_flatten_dict(v, full))
        else:
            items.append((full, v))
    return items


def _group_by_topkey(flat: Sequence[tuple[str, Any]]) -> dict[str, list[tuple[str, Any]]]:
    groups: dict[str, list[tuple[str, Any]]] = {}
    for path, v in flat:
        top = path.split(".", 1)[0]
        rest = path.split(".", 1)[1] if "." in path else ""
        groups.setdefault(top, []).append((rest, v))
    return groups


def print_config(cfg: Any) -> None:
    """Pretty-print ScenarioConfig (or any nested dataclass) with sections."""
    if not is_dataclass(cfg):
        raise TypeError("print_config expects a dataclass instance (e.g., ScenarioConfig).")

    d = asdict(cfg)

    # Add derived info (if present)
    try:
        d["_derived"] = {
            "bbox": {
                "lat_min": cfg.bbox.lat_min,
                "lat_max": cfg.bbox.lat_max,
                "lon_min": cfg.bbox.lon_min,
                "lon_max": cfg.bbox.lon_max,
            },
            "lat0_deg": getattr(cfg, "lat0_deg", None),
            "lon0_deg": getattr(cfg, "lon0_deg", None),
        }
    except Exception:
        pass

    flat = _flatten_dict(d)
    groups = _group_by_topkey(flat)

    preferred_order = [
        "region_mode",
        "run",
        "phy",
        "beam",
        "traffic",
        "ent",
        "qos_refine",
        "lb_refine",
        "usergen",
        "multisat",
        "_derived",
    ]
    section_names = [s for s in preferred_order if s in groups]
    for s in sorted(groups.keys()):
        if s not in section_names:
            section_names.append(s)

    print("\n" + "=" * 72)
    print("SCENARIO CONFIG")
    print("=" * 72)

    for sec in section_names:
        entries = groups[sec]
        print(f"\n[{sec}]")

        keys = [k for (k, _v) in entries]
        pad = max([len(k) for k in keys] + [1])

        for k, v in sorted(entries, key=lambda x: x[0]):
            shown_key = k if k else sec
            leaf_name = shown_key.split(".")[-1]
            print(f"  {shown_key:<{pad}} : {_fmt_value(v, leaf_name)}")

    print("\n" + "=" * 72 + "\n")


def summarize(users: Users, cfg: ScenarioConfig, clusters, evals) -> dict:
    """
    Compute KPIs for a clustering result using eval outputs.
    Adds sanity checks + stronger enterprise metrics.
    """
    K = len(clusters)
    feasible_rate = float(np.mean([ev["feasible"] for ev in evals])) if K > 0 else 0.0

    U = np.array([ev.get("U", np.nan) for ev in evals], dtype=float)
    U = U[~np.isnan(U)]

    ent_total = int((users.qos_w == 4).sum())
    ent_exposed = 0
    ent_z_all = []

    for S, ev in zip(clusters, evals):
        if ev.get("R_m") is None:
            continue

        z = ev.get("z", None)
        if z is None:
            raise ValueError("evaluate_cluster() did not return 'z'.")
        if len(z) != len(S):
            raise ValueError(f"Mismatch: len(z)={len(z)} but len(cluster)={len(S)}")

        w = users.qos_w[S]
        ent_local = (w == 4)

        if np.any(ent_local):
            z_ent = z[ent_local]
            ent_z_all.append(z_ent)
            ent_exposed += int((z_ent > cfg.ent.rho_safe).sum())

    ent_edge_pct = (100.0 * ent_exposed / ent_total) if ent_total > 0 else 0.0

    if len(ent_z_all) > 0:
        ent_z = np.concatenate(ent_z_all)
        ent_z_mean = float(np.mean(ent_z))
        ent_z_p90 = float(np.quantile(ent_z, 0.90))
        ent_z_max = float(np.max(ent_z))
    else:
        ent_z_mean = ent_z_p90 = ent_z_max = 0.0

    risk_sum = float(np.sum([ev.get("risk", 0.0) for ev in evals])) if K > 0 else 0.0

    return {
        "K": int(K),
        "feasible_rate": float(feasible_rate),
        "U_mean": float(np.mean(U)) if U.size > 0 else 0.0,
        "U_max": float(np.max(U)) if U.size > 0 else 0.0,
        "U_min": float(np.min(U)) if U.size > 0 else 0.0,
        "risk_sum": float(risk_sum),

        "ent_total": int(ent_total),
        "ent_exposed": int(ent_exposed),
        "ent_edge_pct": float(ent_edge_pct),
        "ent_z_mean": float(ent_z_mean),
        "ent_z_p90": float(ent_z_p90),
        "ent_z_max": float(ent_z_max),
    }


def summarize_multisat(
    pieces: list[tuple[Users, list[np.ndarray], list[dict]]],
    cfg: ScenarioConfig,
) -> dict:
    """
    Global summary across multiple satellites.
    Each piece: (users_sat, clusters_sat, evals_sat)
    """
    all_U: list[float] = []
    feasible_flags: list[bool] = []
    risk_sum = 0.0

    ent_total = 0
    ent_exposed = 0
    ent_z_all = []

    total_K = 0

    for users_sat, clusters, evals in pieces:
        total_K += len(clusters)
        if len(clusters) != len(evals):
            raise ValueError("Mismatch: len(clusters) != len(evals) in summarize_multisat().")

        feasible_flags.extend([bool(ev.get("feasible", False)) for ev in evals])
        risk_sum += float(np.sum([ev.get("risk", 0.0) for ev in evals]))

        for ev in evals:
            u = ev.get("U", None)
            if u is not None and not np.isnan(u):
                all_U.append(float(u))

        ent_total += int((users_sat.qos_w == 4).sum())

        for S, ev in zip(clusters, evals):
            if ev.get("R_m") is None:
                continue

            z = ev.get("z", None)
            if z is None:
                raise ValueError("evaluate_cluster() did not return 'z'.")
            if len(z) != len(S):
                raise ValueError(f"Mismatch: len(z)={len(z)} but len(cluster)={len(S)}")

            w = users_sat.qos_w[S]
            ent_local = (w == 4)

            if np.any(ent_local):
                z_ent = z[ent_local]
                ent_z_all.append(z_ent)
                ent_exposed += int((z_ent > cfg.ent.rho_safe).sum())

    feasible_rate = float(np.mean(feasible_flags)) if total_K > 0 else 0.0

    U = np.asarray(all_U, dtype=float)
    U_mean = float(np.mean(U)) if U.size > 0 else 0.0
    U_max = float(np.max(U)) if U.size > 0 else 0.0
    U_min = float(np.min(U)) if U.size > 0 else 0.0

    ent_edge_pct = (100.0 * ent_exposed / ent_total) if ent_total > 0 else 0.0

    if len(ent_z_all) > 0:
        ent_z = np.concatenate(ent_z_all)
        ent_z_mean = float(np.mean(ent_z))
        ent_z_p90 = float(np.quantile(ent_z, 0.90))
        ent_z_max = float(np.max(ent_z))
    else:
        ent_z_mean = ent_z_p90 = ent_z_max = 0.0

    return {
        "K": int(total_K),
        "feasible_rate": float(feasible_rate),
        "U_mean": float(U_mean),
        "U_max": float(U_max),
        "U_min": float(U_min),
        "risk_sum": float(risk_sum),

        "ent_total": int(ent_total),
        "ent_exposed": int(ent_exposed),
        "ent_edge_pct": float(ent_edge_pct),
        "ent_z_mean": float(ent_z_mean),
        "ent_z_p90": float(ent_z_p90),
        "ent_z_max": float(ent_z_max),
    }


def print_summary(title: str, s: dict, cfg: ScenarioConfig):
    print(f"\n=== {title} ===")
    print(f"K: {s['K']}")
    print(f"Feasible cluster rate: {s['feasible_rate']*100:.2f}%")
    print(f"Utilization U: mean={s['U_mean']:.3f}, max={s['U_max']:.3f}, min={s['U_min']:.3f}")
    print(
        f"Enterprise edge exposure: {s['ent_edge_pct']:.2f}% "
        f"({s['ent_exposed']}/{s['ent_total']})  (z > rho={cfg.ent.rho_safe})"
    )
    print(f"Enterprise z: mean={s['ent_z_mean']:.3f}, p90={s['ent_z_p90']:.3f}, max={s['ent_z_max']:.3f}")
    print(f"Total enterprise risk (soft): {s['risk_sum']:.3f}")


# ----------------------------
# Helpers for sweep output
# ----------------------------
def flatten_summary(prefix: str, s: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in s.items()}


def _empty_summary() -> dict[str, Any]:
    return {
        "K": 0,
        "feasible_rate": 0.0,
        "U_mean": 0.0,
        "U_max": 0.0,
        "U_min": 0.0,
        "risk_sum": 0.0,
        "ent_total": 0,
        "ent_exposed": 0,
        "ent_edge_pct": 0.0,
        "ent_z_mean": 0.0,
        "ent_z_p90": 0.0,
        "ent_z_max": 0.0,
    }


def flatten_run_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a run record returned by pipeline.run_scenario() into one CSV row.
    This is now robust (won't crash if an optional key is missing).
    """
    row: dict[str, Any] = {
        "seed": rec["seed"],
        "region_mode": rec["region_mode"],
        "n_users": rec["n_users"],

        "use_hotspots": rec.get("use_hotspots", False),
        "n_hotspots": rec.get("n_hotspots", 0),
        "noise_frac": rec.get("noise_frac", 0.0),
        "sigma_min": rec.get("sigma_min", 0.0),
        "sigma_max": rec.get("sigma_max", 0.0),

        "rho_safe": rec.get("rho_safe", 0.0),
        "eirp_dbw": rec.get("eirp_dbw", 0.0),
        "bandwidth_hz": rec.get("bandwidth_hz", 0.0),
        "radius_modes_km": str(rec.get("radius_modes_km", ())),

        "time_usergen_s": rec.get("time_usergen_s", 0.0),
        "time_sat_select_s": rec.get("time_sat_select_s", 0.0),
        "time_assoc_s": rec.get("time_assoc_s", 0.0),
        "time_split_s": rec.get("time_split_s", 0.0),
        "time_ent_ref_s": rec.get("time_ent_ref_s", 0.0),
        "time_lb_ref_s": rec.get("time_lb_ref_s", 0.0),

        "eval_calls": rec.get("eval_calls", 0),
        "n_splits": rec.get("n_splits", 0),
        "ent_moves_tried": rec.get("ent_moves_tried", 0),
        "ent_moves_accepted": rec.get("ent_moves_accepted", 0),
        "lb_moves_tried": rec.get("lb_moves_tried", 0),
        "lb_moves_accepted": rec.get("lb_moves_accepted", 0),

        "time_baseline_without_qos_s": rec.get("time_baseline_without_qos_s", 0.0),
        "time_baseline_with_qos_s": rec.get("time_baseline_with_qos_s", 0.0),
        "time_baseline_bkmeans_s": rec.get("time_baseline_bkmeans_s", 0.0),
        "time_baseline_tgbp_s": rec.get("time_baseline_tgbp_s", 0.0),
    }

    # Optional multi-sat metadata (if present)
    for k in ["ms_tle_path", "ms_time_utc", "ms_elev_mask_deg", "ms_n_active", "ms_n_unserved", "ms_assoc_moves"]:
        if k in rec:
            row[k] = rec[k]

    # Algorithm KPIs
    row |= flatten_summary("main", rec.get("main", _empty_summary()))
    row |= flatten_summary("main_ref", rec.get("main_ref", _empty_summary()))
    row |= flatten_summary("main_ref_lb", rec.get("main_ref_lb", _empty_summary()))
    row |= flatten_summary("wk_demand_fixed", rec.get("wk_demand_fixed", _empty_summary()))
    row |= flatten_summary("wk_demand_rep", rec.get("wk_demand_rep", _empty_summary()))
    row |= flatten_summary("wk_qos_fixed", rec.get("wk_qos_fixed", _empty_summary()))
    row |= flatten_summary("wk_qos_rep", rec.get("wk_qos_rep", _empty_summary()))
    row |= flatten_summary("bk_fixed", rec.get("bk_fixed", _empty_summary()))
    row |= flatten_summary("bk_rep", rec.get("bk_rep", _empty_summary()))
    row |= flatten_summary("tgbp_fixed", rec.get("tgbp_fixed", _empty_summary()))
    row |= flatten_summary("tgbp_rep", rec.get("tgbp_rep", _empty_summary()))
    return row


def write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return

    # Union of all keys -> stable even if some runs have extra fields
    fieldset = set()
    for r in rows:
        fieldset.update(r.keys())
    fieldnames = sorted(fieldset)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
