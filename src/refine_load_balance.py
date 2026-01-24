# src/refine_load_balance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

from src.evaluator import evaluate_cluster

# ----------------------------
# Helpers
# ----------------------------
def _objective_value(U: np.ndarray, mode: str) -> float:
    if U.size == 0:
        return 0.0
    if mode == "var":
        return float(np.var(U))
    # default: range
    return float(U.max() - U.min())


def _cluster_enterprise_exposed(users, cfg, S: np.ndarray, ev: dict) -> int:
    if ev.get("R_m") is None:
        return 0
    w = users.qos_w[S]
    ent = (w == 4)
    if not np.any(ent):
        return 0
    z = ev["z"]
    return int(np.sum(z[ent] > cfg.ent.rho_safe))


def _build_overlap_adjacency(
    centers_xy: np.ndarray,
    radii_m: np.ndarray,
    margin_m: float,
) -> List[np.ndarray]:
    """
    adjacency[k] = array of cluster indices j that overlap with cluster k
    overlap: dist <= Rk + Rj + margin
    """
    K = centers_xy.shape[0]
    adjacency: List[np.ndarray] = []

    # O(K^2) build — fine for your K (~few hundreds). If K gets huge, we can grid-index it.
    for k in range(K):
        ck = centers_xy[k]
        rk = radii_m[k]
        # if radius missing -> no neighbors
        if not np.isfinite(rk) or rk <= 0:
            adjacency.append(np.array([], dtype=int))
            continue
        d = np.linalg.norm(centers_xy - ck[None, :], axis=1)
        ok = (d <= (rk + radii_m + margin_m))
        ok[k] = False
        adjacency.append(np.where(ok)[0].astype(int))

    return adjacency

def _build_overlap_adjacency_grid(
    centers_xy: np.ndarray,
    radii_m: np.ndarray,
    margin_m: float,
) -> List[np.ndarray]:
    """
    Grid-hash based overlap adjacency.
    adjacency[k] = array of indices j whose circles overlap with k:
        dist(center_k, center_j) <= Rk + Rj + margin

    This avoids O(K^2) by only comparing beams in nearby spatial cells.

    Works best when beams are roughly uniformly distributed geographically.
    """
    K = centers_xy.shape[0]
    adjacency: List[np.ndarray] = [np.array([], dtype=int) for _ in range(K)]

    # Valid beams only
    valid = np.isfinite(radii_m) & (radii_m > 0) & np.isfinite(centers_xy).all(axis=1)
    if not np.any(valid):
        return adjacency

    idx_valid = np.where(valid)[0]
    C = centers_xy[idx_valid]
    R = radii_m[idx_valid]

    # Cell size: choose something that ensures overlapping circles are in same or neighboring cells.
    # Using max radius + margin is a good conservative default.
    rmax = float(np.nanmax(R))
    cell = max(1.0, rmax + float(margin_m))

    # Build grid: cell -> list of valid indices (original k)
    grid: Dict[Tuple[int, int], List[int]] = {}
    inv = 1.0 / cell

    gx = np.floor(C[:, 0] * inv).astype(int)
    gy = np.floor(C[:, 1] * inv).astype(int)

    for kk, cx, cy in zip(idx_valid, gx, gy):
        key = (int(cx), int(cy))
        grid.setdefault(key, []).append(int(kk))

    # Neighbor cell offsets (3x3)
    neigh_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    # For each valid beam, only check candidates in nearby cells.
    # We'll construct adjacency symmetrically to avoid duplicating work.
    for k in idx_valid:
        ck = centers_xy[k]
        rk = float(radii_m[k])
        if not np.isfinite(rk) or rk <= 0:
            continue

        cx = int(np.floor(ck[0] * inv))
        cy = int(np.floor(ck[1] * inv))

        cand: List[int] = []
        for dx, dy in neigh_offsets:
            cand.extend(grid.get((cx + dx, cy + dy), []))

        if not cand:
            continue

        cand_arr = np.array(cand, dtype=int)
        # Only check j>k to build symmetric adjacency once
        cand_arr = cand_arr[cand_arr > k]
        if cand_arr.size == 0:
            continue

        cj = centers_xy[cand_arr]
        rj = radii_m[cand_arr]

        # Overlap test
        d = np.linalg.norm(cj - ck[None, :], axis=1)
        ok = d <= (rk + rj + margin_m)

        js = cand_arr[ok]
        if js.size == 0:
            continue

        # Add both ways
        adjacency[k] = np.concatenate([adjacency[k], js])
        for j in js:
            adjacency[j] = np.concatenate([adjacency[j], np.array([k], dtype=int)])

    # Optional: unique-sort each adjacency list (prevents duplicates)
    for k in range(K):
        if adjacency[k].size > 1:
            adjacency[k] = np.unique(adjacency[k])

    return adjacency


def _select_donor_clusters(U: np.ndarray) -> np.ndarray:
    """
    Donors are high-utilization clusters.
    We simply sort descending and let the move acceptance rule decide.
    """
    return np.argsort(-U).astype(int)


def _receiver_candidates(adjacency_k: np.ndarray, U: np.ndarray, k_receivers: int) -> np.ndarray:
    """
    Choose receiver clusters among neighbors with smallest U.
    """
    if adjacency_k.size == 0:
        return np.array([], dtype=int)
    neigh = adjacency_k
    # sort neighbors by low utilization
    order = np.argsort(U[neigh])
    chosen = neigh[order[: min(k_receivers, order.size)]]
    return chosen.astype(int)


def _donor_user_candidates(
    users,
    cfg,
    S_from: np.ndarray,
    ev_from: dict,
    k_users: int,
    prefer_non_ent: bool,
) -> np.ndarray:
    """
    Candidate users to move out of donor:
      - prioritize non-enterprise if prefer_non_ent
      - then prioritize "edge-ish" users (high z) since they are less central
    """
    if S_from.size == 0 or ev_from.get("R_m") is None:
        return np.array([], dtype=int)

    z = ev_from["z"]
    w = users.qos_w[S_from]

    if prefer_non_ent:
        non_ent_idx = np.where(w != 4)[0]
        ent_idx = np.where(w == 4)[0]

        # Sort each set by descending z (more edge first)
        non_ent_sorted = non_ent_idx[np.argsort(-z[non_ent_idx])] if non_ent_idx.size else np.array([], dtype=int)
        ent_sorted = ent_idx[np.argsort(-z[ent_idx])] if ent_idx.size else np.array([], dtype=int)

        cand_local = np.concatenate([non_ent_sorted, ent_sorted], axis=0)
    else:
        cand_local = np.argsort(-z).astype(int)

    cand_local = cand_local[: min(k_users, cand_local.size)]
    return S_from[cand_local].astype(int)


# ----------------------------
# Main algorithm
# ----------------------------
def _range_after_two_updates(U: np.ndarray, i: int, ui_new: float, j: int, uj_new: float) -> float:
    """
    Compute new (max-min) after changing U[i], U[j] to new values.

    Fast path if we don't hit old extrema; safe fallback does a full recompute.
    """
    umin = float(U.min())
    umax = float(U.max())
    ui = float(U[i])
    uj = float(U[j])

    hits_extrema = (ui == umin) or (ui == umax) or (uj == umin) or (uj == umax)
    if not hits_extrema:
        umin2 = min(umin, ui_new, uj_new)
        umax2 = max(umax, ui_new, uj_new)
        return umax2 - umin2

    # Safe fallback
    U_tmp = U.copy()
    U_tmp[i] = ui_new
    U_tmp[j] = uj_new
    return float(U_tmp.max() - U_tmp.min())


def refine_load_balance_by_overlap(
    users,
    cfg,
    clusters: List[np.ndarray],
    evals: List[dict],
    prof: Any | None = None,
) -> Tuple[List[np.ndarray], List[dict], Dict[str, Any]]:
    """
    Stage-2 refinement: balance loads by moving users between *overlapping beams*.

    Rules:
      - moves allowed only between clusters whose circles overlap
      - accept move only if:
          (a) both affected clusters remain feasible
          (b) load-balance objective improves
          (c) enterprise risk/exposure does not worsen beyond slack
          (d) (optional) receiver not too full

    Performance improvements vs previous version:
      - remove O(m) `uid in S_from` check
      - avoid O(K) U.copy() and full objective recompute per attempt (for objective="range")
      - add cheap pre-filters to reduce expensive eval calls
      - optional profiler counters
    """
    if not cfg.lb_refine.enabled:
        return clusters, evals, {"enabled": False, "reason": "disabled"}

    # Defensive copies (keeps caller's state unchanged)
    clusters = [np.asarray(S, dtype=int).copy() for S in clusters]
    evals = [dict(ev) for ev in evals]

    K = len(clusters)
    stats: Dict[str, Any] = {
        "enabled": True,
        "rounds": int(cfg.lb_refine.rounds),
        "moves_tried": 0,
        "moves_accepted": 0,
        "objective_before": None,
        "objective_after": None,
    }

    # Prepare arrays (centers/radii/U) from evals
    centers_xy = np.zeros((K, 2), dtype=float)
    radii_m = np.full(K, np.nan, dtype=float)
    U = np.zeros(K, dtype=float)

    for k, ev in enumerate(evals):
        c = ev.get("center_xy", None)
        r = ev.get("R_m", None)
        centers_xy[k] = np.asarray(c, dtype=float) if c is not None else np.array([np.nan, np.nan], dtype=float)
        radii_m[k] = float(r) if (r is not None) else np.nan
        U[k] = float(ev.get("U", 0.0))

    obj0 = _objective_value(U, cfg.lb_refine.objective)
    stats["objective_before"] = float(obj0)

    rng = np.random.default_rng(cfg.run.seed + 44444)

    # Optional: only consider donors above this U threshold to cut work
    # (Keep gentle; otherwise can miss improvements.)
    donor_u_min = 0.0  # set e.g. 0.5 if you want aggressive pruning

    rounds = int(cfg.lb_refine.rounds)
    max_moves_round = int(cfg.lb_refine.max_moves_per_round)
    k_receivers = int(cfg.lb_refine.k_receivers)
    k_users = int(cfg.lb_refine.k_users_from_donor)
    margin_m = float(cfg.lb_refine.intersect_margin_m)

    for rnd in range(rounds):
        # refresh arrays from evals (centers/radii/U may change after accepted moves)
        for k, ev in enumerate(evals):
            c = ev.get("center_xy", None)
            r = ev.get("R_m", None)
            if c is not None:
                centers_xy[k] = np.asarray(c, dtype=float)
            if r is not None:
                radii_m[k] = float(r)
            U[k] = float(ev.get("U", U[k]))

        adjacency = _build_overlap_adjacency_grid(centers_xy, radii_m, margin_m)

        donors = _select_donor_clusters(U)  # expects high->low
        moves_this_round = 0

        # Compute objective once per round for fast updates (range mode)
        if cfg.lb_refine.objective == "range":
            obj_before_round = float(U.max() - U.min())
        else:
            obj_before_round = float(_objective_value(U, cfg.lb_refine.objective))

        for k_from in donors:
            if moves_this_round >= max_moves_round:
                break

            if U[k_from] < donor_u_min:
                continue

            neigh = adjacency[k_from]
            if neigh.size == 0:
                continue

            # choose best candidate receivers among neighbors (lowest U)
            receivers = _receiver_candidates(neigh, U, k_receivers)
            if receivers.size == 0:
                continue

            # receiver fullness constraint (optional)
            if not bool(cfg.lb_refine.allow_receiver_close_to_full):
                receivers = receivers[U[receivers] <= float(cfg.lb_refine.receiver_u_max)]
                if receivers.size == 0:
                    continue

            S_from = clusters[k_from]
            ev_from = evals[k_from]

            if S_from.size <= 2:
                continue
            if not ev_from.get("feasible", True):
                continue

            # Candidate users to move out of donor
            cand_users = _donor_user_candidates(
                users, cfg, S_from, ev_from,
                k_users=k_users,
                prefer_non_ent=bool(cfg.lb_refine.prefer_non_enterprise),
            )
            if cand_users.size == 0:
                continue

            rng.shuffle(cand_users)

            # Precompute donor-side enterprise metrics once (used in guards)
            donor_risk0 = float(evals[k_from].get("risk", 0.0))
            donor_exposed0 = _cluster_enterprise_exposed(users, cfg, S_from, evals[k_from])

            # Try moving candidates out
            for uid in cand_users:
                if moves_this_round >= max_moves_round:
                    break

                # S_from contains uid by construction; no O(m) membership check

                for k_to in receivers:
                    stats["moves_tried"] += 1

                    # cheap directionality: only move from higher-U donor to lower-U receiver
                    if U[k_to] >= U[k_from] - 1e-12:
                        continue

                    # optional: receiver near-full skip already handled by receiver_u_max
                    S_to = clusters[k_to]
                    if S_to.size == 0:
                        continue
                    if not evals[k_to].get("feasible", True):
                        continue

                    # Propose new sets
                    S_from_new = S_from[S_from != uid]
                    if S_from_new.size == 0:
                        continue
                    S_to_new = np.append(S_to, uid).astype(int)

                    # Evaluate only affected clusters
                    ev_from_new = evaluate_cluster(users, S_from_new, cfg)
                    if prof:
                        prof.inc("eval_calls")
                    if not ev_from_new.get("feasible", False):
                        continue

                    ev_to_new = evaluate_cluster(users, S_to_new, cfg)
                    if prof:
                        prof.inc("eval_calls")
                    if not ev_to_new.get("feasible", False):
                        continue

                    # Enterprise guards (risk and exposed count) on these two clusters only
                    before_risk = donor_risk0 + float(evals[k_to].get("risk", 0.0))
                    after_risk = float(ev_from_new.get("risk", 0.0) + ev_to_new.get("risk", 0.0))
                    if after_risk > before_risk + float(cfg.lb_refine.risk_slack):
                        continue

                    before_exposed = donor_exposed0 + _cluster_enterprise_exposed(users, cfg, S_to, evals[k_to])
                    after_exposed = _cluster_enterprise_exposed(users, cfg, S_from_new, ev_from_new) + \
                                    _cluster_enterprise_exposed(users, cfg, S_to_new, ev_to_new)
                    if after_exposed > before_exposed + int(cfg.lb_refine.exposure_slack):
                        continue

                    # Load objective improvement check
                    U_from_new = float(ev_from_new.get("U", U[k_from]))
                    U_to_new = float(ev_to_new.get("U", U[k_to]))

                    if cfg.lb_refine.objective == "range":
                        obj_after = _range_after_two_updates(U, k_from, U_from_new, k_to, U_to_new)
                        obj_before = obj_before_round
                    else:
                        # keep correctness for var: compute full objective with two updates
                        # (still avoid copy if you later add a delta formula)
                        U_tmp = U.copy()
                        U_tmp[k_from] = U_from_new
                        U_tmp[k_to] = U_to_new
                        obj_before = float(_objective_value(U, cfg.lb_refine.objective))
                        obj_after = float(_objective_value(U_tmp, cfg.lb_refine.objective))

                    if obj_after >= obj_before - 1e-12:
                        continue

                    # Accept move
                    clusters[k_from] = S_from_new
                    clusters[k_to] = S_to_new
                    evals[k_from] = ev_from_new
                    evals[k_to] = ev_to_new

                    # Update U in-place
                    U[k_from] = U_from_new
                    U[k_to] = U_to_new

                    # Update cached donor references for continued moves from same donor
                    S_from = clusters[k_from]
                    ev_from = evals[k_from]
                    donor_risk0 = float(evals[k_from].get("risk", 0.0))
                    donor_exposed0 = _cluster_enterprise_exposed(users, cfg, S_from, evals[k_from])

                    if cfg.lb_refine.objective == "range":
                        obj_before_round = float(obj_after)

                    stats["moves_accepted"] += 1
                    moves_this_round += 1
                    break  # stop trying receivers for this uid

                # end receivers loop

        # end donors loop

    stats["objective_after"] = float(_objective_value(U, cfg.lb_refine.objective))
    return clusters, evals, stats