from __future__ import annotations
from config import ScenarioConfig
from src.models import Users
import numpy as np
import csv
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence


def _fmt_value(v: Any, key: str | None = None) -> str:
    """Pretty formatting + some unit-aware helpers based on key names."""
    if isinstance(v, float):
        # unit-aware
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
        # default float
        return f"{v:.6g}"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        # show tuples nicely (e.g., radius modes)
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
    """
    Pretty-print ScenarioConfig (or any nested dataclass) with sections.

    Usage:
        from src.config_print import print_config
        print_config(cfg)
    """
    if not is_dataclass(cfg):
        raise TypeError("print_config expects a dataclass instance (e.g., ScenarioConfig).")

    d = asdict(cfg)

    # Add computed properties (nice to see)
    # (won't be in asdict because they're @property)
    try:
        d["_derived"] = {
            "bbox": {
                "lat_min": cfg.bbox.lat_min,
                "lat_max": cfg.bbox.lat_max,
                "lon_min": cfg.bbox.lon_min,
                "lon_max": cfg.bbox.lon_max,
            },
            "lat0_deg": cfg.lat0_deg,
            "lon0_deg": cfg.lon0_deg,
        }
    except Exception:
        pass

    flat = _flatten_dict(d)
    groups = _group_by_topkey(flat)

    # Order sections in a human-friendly way (fallback to alpha for unknown)
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
        "_derived",
    ]
    section_names = []
    for s in preferred_order:
        if s in groups:
            section_names.append(s)
    for s in sorted(groups.keys()):
        if s not in section_names:
            section_names.append(s)

    print("\n" + "=" * 72)
    print("SCENARIO CONFIG")
    print("=" * 72)

    for sec in section_names:
        entries = groups[sec]
        # single scalar like region_mode will have rest == ""
        print(f"\n[{sec}]")

        # compute padding for pretty alignment
        keys = [k for (k, _v) in entries]
        pad = max([len(k) for k in keys] + [1])

        for k, v in sorted(entries, key=lambda x: x[0]):
            shown_key = k if k else sec  # region_mode case
            # show unit-aware formatting
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
    U = U[~np.isnan(U)]  # in case some ev missing U (shouldn't)

    # --- Enterprise metrics ---
    ent_total = int((users.qos_w == 4).sum())
    ent_exposed = 0

    ent_z_all = []  # collect normalized radii z for enterprise users across all clusters

    for S, ev in zip(clusters, evals):
        # if geom infeasible (shouldn't happen after repair), skip safely
        if ev.get("R_m") is None:
            continue

        # Sanity: z must match cluster size
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

    if ent_total > 0:
        ent_edge_pct = 100.0 * ent_exposed / ent_total
    else:
        ent_edge_pct = 0.0

    if len(ent_z_all) > 0:
        ent_z = np.concatenate(ent_z_all)
        ent_z_mean = float(np.mean(ent_z))
        ent_z_p90 = float(np.quantile(ent_z, 0.90))
        ent_z_max = float(np.max(ent_z))
    else:
        ent_z_mean = ent_z_p90 = ent_z_max = 0.0

    risk_sum = float(np.sum([ev.get("risk", 0.0) for ev in evals])) if K > 0 else 0.0

    return {
        "K": K,
        "feasible_rate": feasible_rate,
        "U_mean": float(np.mean(U)) if U.size > 0 else 0.0,
        "U_max": float(np.max(U)) if U.size > 0 else 0.0,
        "U_min": float(np.min(U)) if U.size > 0 else 0.0,
        "risk_sum": risk_sum,

        # enterprise metrics (more defensible than risk_sum alone)
        "ent_total": ent_total,
        "ent_exposed": ent_exposed,
        "ent_edge_pct": float(ent_edge_pct),
        "ent_z_mean": ent_z_mean,
        "ent_z_p90": ent_z_p90,
        "ent_z_max": ent_z_max,
    }

def print_summary(title: str, s: dict, cfg: ScenarioConfig):
    print(f"\n=== {title} ===")
    print(f"K: {s['K']}")
    print(f"Feasible cluster rate: {s['feasible_rate']*100:.2f}%")
    print(f"Utilization U: mean={s['U_mean']:.3f}, max={s['U_max']:.3f}, min={s['U_min']:.3f}")

    # Enterprise stats
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

def flatten_run_record(rec: dict[str, Any]) -> dict[str, Any]:
    # scenario columns
    row: dict[str, Any] = {
        "seed": rec["seed"],
        "region_mode": rec["region_mode"],
        "n_users": rec["n_users"],

        # keep the CSV header stable even if internal name is now usergen.enabled
        "use_hotspots": rec["use_hotspots"],
        "n_hotspots": rec["n_hotspots"],
        "noise_frac": rec["noise_frac"],
        "sigma_min": rec["sigma_min"],
        "sigma_max": rec["sigma_max"],

        "rho_safe": rec["rho_safe"],
        "eirp_dbw": rec["eirp_dbw"],
        "bandwidth_hz": rec["bandwidth_hz"],
        "radius_modes_km": str(rec["radius_modes_km"]),

        "time_usergen_s": rec["time_usergen_s"],
        "time_split_s": rec["time_split_s"],
        "time_ent_ref_s": rec["time_ent_ref_s"],
        "time_lb_ref_s": rec["time_lb_ref_s"],
        "eval_calls": rec["eval_calls"],
        "n_splits": rec["n_splits"],
        "ent_moves_tried": rec["ent_moves_tried"],
        "ent_moves_accepted": rec["ent_moves_accepted"],
        "lb_moves_tried": rec["lb_moves_tried"],
        "lb_moves_accepted": rec["lb_moves_accepted"],
        "time_baseline_without_qos_s": rec["time_baseline_without_qos_s"],
        "time_baseline_with_qos_s": rec["time_baseline_with_qos_s"]
    }

    # algorithm KPIs
    row |= flatten_summary("main", rec["main"])
    row |= flatten_summary("main_ref", rec["main_ref"])
    row |= flatten_summary("main_ref_lb", rec["main_ref_lb"])
    row |= flatten_summary("wk_demand_fixed", rec["wk_demand_fixed"])
    row |= flatten_summary("wk_demand_rep", rec["wk_demand_rep"])
    row |= flatten_summary("wk_qos_fixed", rec["wk_qos_fixed"])
    row |= flatten_summary("wk_qos_rep", rec["wk_qos_rep"])

    return row

def write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)