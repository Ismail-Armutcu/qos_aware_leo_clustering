# src/refine_qos_angle.py
from __future__ import annotations

import numpy as np

from src.coords import unit
from src.evaluator import evaluate_cluster


def cluster_boresight_unit_vectors(users, clusters: list[np.ndarray]) -> np.ndarray:
    """
    Compute u_k for each cluster k: unit vector from satellite to cluster-center ECEF.

    Cluster-center ECEF is approximated as mean of member ECEF (consistent with evaluate_cluster()).

    Returns
    -------
    Uc : np.ndarray, shape (K, 3)
        Unit boresight vectors sat -> cluster center.
    """
    K = len(clusters)
    Uc = np.zeros((K, 3), dtype=float)
    for k, S in enumerate(clusters):
        if S.size == 0:
            # Should not happen, but keep safe
            Uc[k] = np.array([1.0, 0.0, 0.0], dtype=float)
            continue
        c_ecef = users.ecef_m[S].mean(axis=0)
        v = c_ecef - users.sat_ecef_m
        Uc[k] = unit(v)
    return Uc


def cluster_boresight(users, S: np.ndarray) -> np.ndarray:
    """
    Compute a single cluster boresight unit vector u = unit(c_ecef - sat_ecef),
    where c_ecef is mean of member ECEF.
    """
    if S.size == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    c_ecef = users.ecef_m[S].mean(axis=0)
    v = c_ecef - users.sat_ecef_m
    return unit(v)


def refine_enterprise_by_angle(
    users,
    cfg,
    clusters: list[np.ndarray],
    evals: list[dict],
    n_rounds: int = 3,
    kcand: int = 6,
    max_moves_per_round: int = 2000,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    QoS refinement: move enterprise users away from cluster edge while preserving feasibility.

    The core idea:
      - Enterprise users should not be near the beam edge (z > rho_safe).
      - Candidate target clusters are chosen by smallest 3D off-axis angle, implemented by
        maximizing dot product u_i · u_k, where:
          u_i = sat->user unit vector (precomputed users.u_sat2user)
          u_k = sat->cluster-center unit vector (computed from member ECEF mean)

    Acceptance rule:
      - Both affected clusters must remain feasible under evaluate_cluster() after the move.
      - Prefer moves that reduce the *number* of exposed enterprise users (z_ent > rho_safe) across the
        two affected clusters; if tied, accept if total risk decreases.

    IMPORTANT:
      - This function COPIES input clusters/evals so the caller's "unrefined" results are not mutated.

    Parameters
    ----------
    users : Users container
    cfg : ScenarioConfig
    clusters : list[np.ndarray]
        List of user-index arrays, one per cluster.
    evals : list[dict]
        evaluate_cluster outputs aligned with clusters.
    n_rounds : int
        Number of refinement passes.
    kcand : int
        Number of candidate clusters to try per risky enterprise user.
    max_moves_per_round : int
        Cap accepted moves per round for runtime control.

    Returns
    -------
    clusters_ref : list[np.ndarray]
    evals_ref : list[dict]
    stats : dict
    """
    # IMPORTANT: copy to avoid mutating caller's clusters/evals (keeps "unrefined" results valid)
    clusters = [np.asarray(S, dtype=int).copy() for S in clusters]
    evals = [dict(ev) for ev in evals]  # shallow copy ok; we overwrite changed ones anyway

    K = len(clusters)
    if K == 0:
        return clusters, evals, {"moves_accepted": 0, "moves_tried": 0, "rounds": 0}

    rho = cfg.ent.rho_safe
    stats = {"moves_accepted": 0, "moves_tried": 0, "rounds": int(n_rounds)}

    # Map user -> current cluster index for fast lookup
    user2cluster = np.empty(users.n, dtype=int)
    for k, S in enumerate(clusters):
        user2cluster[S] = k

    def ent_exposed_count(S: np.ndarray, ev: dict) -> int:
        """Count enterprise users with z > rho in cluster S given its evaluation ev."""
        if ev.get("R_m") is None:
            return 0
        w = users.qos_w[S]
        ent = (w == 4)
        if not np.any(ent):
            return 0
        z = ev["z"]
        return int(np.sum(z[ent] > rho))

    for rnd in range(n_rounds):
        # Recompute boresights for this round (clusters changed across rounds)
        Uc = cluster_boresight_unit_vectors(users, clusters)  # (K,3)

        # Build list of risky enterprise users: enterprise with z > rho in their current cluster
        at_risk_users: list[int] = []
        for k, (S, ev) in enumerate(zip(clusters, evals)):
            if ev.get("R_m") is None:
                continue
            z = ev["z"]
            w = users.qos_w[S]
            ent_local = (w == 4)
            if not np.any(ent_local):
                continue
            # indices within cluster array
            idx_local = np.where(ent_local)[0]
            risky_local = idx_local[z[ent_local] > rho]
            if risky_local.size > 0:
                at_risk_users.extend(S[risky_local].tolist())

        if not at_risk_users:
            break

        # Shuffle to avoid bias
        rng = np.random.default_rng(cfg.run.seed + 12345 + rnd)
        rng.shuffle(at_risk_users)

        moves_this_round = 0

        for uid in at_risk_users:
            if moves_this_round >= max_moves_per_round:
                break

            k_from = int(user2cluster[uid])
            S_from = clusters[k_from]

            # Avoid degeneracy: do not take from very small clusters
            if S_from.size <= 2:
                continue

            u_i = users.u_sat2user[uid]  # (3,)

            # Candidate clusters by smallest angle <=> max dot(u_k, u_i)
            dots = Uc @ u_i  # (K,)
            dots[k_from] = -np.inf  # exclude current cluster

            # Choose top-kcand candidates efficiently
            kc = int(min(max(kcand, 1), K - 1))
            cand_idx = np.argpartition(-dots, kth=kc - 1)[:kc]
            cand_idx = cand_idx[np.argsort(-dots[cand_idx])]

            for k_to in cand_idx:
                stats["moves_tried"] += 1
                k_to = int(k_to)

                S_to = clusters[k_to]

                # Propose new sets
                S_from_new = S_from[S_from != uid]
                if S_from_new.size == 0:
                    continue
                S_to_new = np.append(S_to, uid)

                # Evaluate only affected clusters
                ev_from_new = evaluate_cluster(users, S_from_new, cfg)
                if not ev_from_new["feasible"]:
                    continue
                ev_to_new = evaluate_cluster(users, S_to_new, cfg)
                if not ev_to_new["feasible"]:
                    continue

                # Exposure + risk before/after (local to affected clusters)
                before_exposed = ent_exposed_count(S_from, evals[k_from]) + ent_exposed_count(S_to, evals[k_to])
                after_exposed = ent_exposed_count(S_from_new, ev_from_new) + ent_exposed_count(S_to_new, ev_to_new)

                before_risk = float(evals[k_from].get("risk", 0.0) + evals[k_to].get("risk", 0.0))
                after_risk = float(ev_from_new.get("risk", 0.0) + ev_to_new.get("risk", 0.0))

                accept = False
                if after_exposed < before_exposed:
                    accept = True
                elif after_exposed == before_exposed and after_risk < before_risk - 1e-12:
                    accept = True

                if not accept:
                    continue

                # Commit move
                clusters[k_from] = S_from_new
                clusters[k_to] = S_to_new
                evals[k_from] = ev_from_new
                evals[k_to] = ev_to_new
                user2cluster[uid] = k_to

                # Update boresights for the two changed clusters (keeps candidate selection consistent within round)
                Uc[k_from] = cluster_boresight(users, clusters[k_from])
                Uc[k_to] = cluster_boresight(users, clusters[k_to])

                stats["moves_accepted"] += 1
                moves_this_round += 1
                break  # next user

    # Final safety: alignment check (cheap and prevents silent KPI corruption)
    for k, (S, ev) in enumerate(zip(clusters, evals)):
        if ev.get("R_m") is None:
            continue
        z = ev.get("z", None)
        if z is None or len(z) != len(S):
            raise RuntimeError(f"refine_enterprise_by_angle alignment error at k={k}: len(z)={len(z) if z is not None else None} len(S)={len(S)}")

    return clusters, evals, stats
