from config import ScenarioConfig
from src.models import Users
import numpy as np
from typing import Any
import csv

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
    }

    # algorithm KPIs
    row |= flatten_summary("main", rec["main"])
    row |= flatten_summary("main_ref", rec["main_ref"])
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