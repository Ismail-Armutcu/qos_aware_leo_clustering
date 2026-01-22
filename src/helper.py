from config import ScenarioConfig
from src.models import Users
import numpy as np

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
            ent_exposed += int((z_ent > cfg.rho_safe).sum())

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
        f"({s['ent_exposed']}/{s['ent_total']})  (z > rho={cfg.rho_safe})"
    )
    print(f"Enterprise z: mean={s['ent_z_mean']:.3f}, p90={s['ent_z_p90']:.3f}, max={s['ent_z_max']:.3f}")

    print(f"Total enterprise risk (soft): {s['risk_sum']:.3f}")