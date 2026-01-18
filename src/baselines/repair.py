# src/baselines/repair.py
from __future__ import annotations
import numpy as np

from src.evaluator import evaluate_cluster
from src.split import split_farthest


def repair_clusters_split_until_feasible(
    users,
    cfg,
    clusters: list[np.ndarray],
    max_total_clusters: int = 20000,
    max_splits_total: int = 200000,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    STRICT repair:
    Repeatedly split any infeasible cluster until all clusters are feasible.

    - Feasibility uses evaluate_cluster(): discrete footprint modes + Shannon capacity
    - Splitting uses split_farthest()
    - If a singleton user cluster is infeasible (U>1), this function raises:
      that scenario is infeasible under the PHY/demand settings.

    Returns: (clusters_repaired, evals_repaired, stats)
    """
    queue: list[np.ndarray] = [np.asarray(S, dtype=int) for S in clusters if len(S) > 0]
    repaired: list[np.ndarray] = []
    evals: list[dict] = []

    n_splits = 0
    n_geom_fails = 0
    n_cap_fails = 0
    worst_singletons = []  # store debug info if infeasible singleton happens

    while queue:
        if len(queue) + len(repaired) > max_total_clusters:
            raise RuntimeError(
                f"Repair produced too many clusters (> {max_total_clusters}). "
                "Likely geometric infeasibility (beam modes too small for region) or bad generator."
            )
        if n_splits > max_splits_total:
            raise RuntimeError(
                f"Repair exceeded max_splits_total={max_splits_total}. "
                "Likely infeasible scenario or too strict constraints."
            )

        S = queue.pop()
        if S.size == 0:
            continue

        ev = evaluate_cluster(users, S, cfg)

        if ev["feasible"]:
            repaired.append(S)
            evals.append(ev)
            continue

        reason = ev.get("reason", None)
        if reason == "geom":
            n_geom_fails += 1
        elif reason == "cap":
            n_cap_fails += 1

        # If singleton infeasible -> cannot be repaired by splitting
        if S.size == 1:
            uid = int(S[0])
            # Helpful diagnostics
            d = float(users.demand_mbps[uid])
            w = int(users.qos_w[uid])
            # For singleton, z is 0, so infeasible must be capacity
            U = float(ev.get("U", np.nan))
            r = float(ev.get("rate_mbps", np.array([np.nan]))[0]) if ev.get("rate_mbps", None) is not None else np.nan
            worst_singletons.append((uid, w, d, r, U, reason))
            raise RuntimeError(
                "STRICT repair failed: found infeasible SINGLETON cluster. "
                f"user_id={uid}, qos_w={w}, demand_mbps={d:.3f}, rate_mbps={r:.3f}, U={U:.3f}, reason={reason}. "
                "This indicates the scenario is infeasible under current PHY/demand settings."
            )

        # If size==2 and infeasible, splitting will produce singletons, so test if those would be feasible.
        # We can still split, but we want deterministic behavior and clear failure if singletons infeasible.
        seed = cfg.seed + 10_000 + n_splits
        S1, S2 = split_farthest(users.xy_m, S, seed=seed)
        n_splits += 1

        if verbose:
            print(f"[repair] split size={S.size} reason={reason} -> {S1.size}+{S2.size}")

        queue.append(S1)
        queue.append(S2)

    # Final sanity: all feasible
    if not all(ev["feasible"] for ev in evals):
        # This should not happen in strict mode, but keep safety
        bad = sum(not ev["feasible"] for ev in evals)
        raise RuntimeError(f"STRICT repair ended with {bad} infeasible clusters (unexpected).")

    stats = {
        "n_splits": n_splits,
        "n_geom_fails_initially": n_geom_fails,
        "n_cap_fails_initially": n_cap_fails,
        "K_before": len([S for S in clusters if len(S) > 0]),
        "K_after": len(repaired),
    }
    return repaired, evals, stats
