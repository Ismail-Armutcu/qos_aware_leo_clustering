# src/refine_load_balance.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from src.evaluator import evaluate_cluster
from src.phy import fspl_db, gain_db_gaussian, shannon_rate_mbps


def _build_overlap_adjacency_grid(
    centers_xy: np.ndarray,
    radii_m: np.ndarray,
    margin_m: float,
) -> List[np.ndarray]:
    """
    Grid-hash overlap adjacency:
      overlap if dist(center_k, center_j) <= Rk + Rj + margin
    """
    K = int(centers_xy.shape[0])
    adjacency: List[np.ndarray] = [np.array([], dtype=int) for _ in range(K)]

    valid = np.isfinite(radii_m) & (radii_m > 0) & np.isfinite(centers_xy).all(axis=1)
    if not np.any(valid):
        return adjacency

    idx_valid = np.where(valid)[0]
    C = centers_xy[idx_valid]
    R = radii_m[idx_valid]

    rmax = float(np.max(R))
    cell = max(1.0, rmax + float(margin_m))
    inv = 1.0 / cell

    gx = np.floor(C[:, 0] * inv).astype(int)
    gy = np.floor(C[:, 1] * inv).astype(int)

    grid: Dict[Tuple[int, int], List[int]] = {}
    for kk, cx, cy in zip(idx_valid, gx, gy):
        key = (int(cx), int(cy))
        grid.setdefault(key, []).append(int(kk))

    neigh_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    for k in idx_valid:
        ck = centers_xy[k]
        rk = float(radii_m[k])

        cx = int(np.floor(ck[0] * inv))
        cy = int(np.floor(ck[1] * inv))

        cand: List[int] = []
        for dx, dy in neigh_offsets:
            cand.extend(grid.get((cx + dx, cy + dy), []))
        if not cand:
            continue

        cand_arr = np.array(cand, dtype=int)
        cand_arr = cand_arr[cand_arr > k]  # build symmetric once
        if cand_arr.size == 0:
            continue

        cj = centers_xy[cand_arr]
        rj = radii_m[cand_arr]
        d = np.linalg.norm(cj - ck[None, :], axis=1)
        ok = d <= (rk + rj + float(margin_m))

        js = cand_arr[ok]
        if js.size == 0:
            continue

        adjacency[k] = np.concatenate([adjacency[k], js])
        for j in js:
            adjacency[j] = np.concatenate([adjacency[j], np.array([k], dtype=int)])

    for k in range(K):
        if adjacency[k].size > 1:
            adjacency[k] = np.unique(adjacency[k])

    return adjacency


def _snr_const_db_all(users, cfg) -> np.ndarray:
    """
    snr_db = snr_const_db + g_db, where:
      snr_const_db = eirp_dbw - fspl_db - loss_misc_db - (noise_psd_dbw_hz + 10log10(B))
    """
    fspl = fspl_db(users.range_m, cfg.phy.carrier_freq_hz)
    noise_total_db = float(cfg.phy.noise_psd_dbw_hz) + 10.0 * np.log10(float(cfg.phy.bandwidth_hz))
    return float(cfg.phy.eirp_dbw) - fspl - float(cfg.phy.loss_misc_db) - noise_total_db


def _rate_mbps_vec(u_sat2user: np.ndarray, u_c: np.ndarray, theta_3db: float, snr_const_db: np.ndarray, cfg) -> np.ndarray:
    cosang = np.clip(u_sat2user @ u_c, -1.0, 1.0)
    theta = np.arccos(cosang)
    g_db = gain_db_gaussian(theta, theta_3db)
    snr = 10.0 ** ((snr_const_db + g_db) / 10.0)
    return shannon_rate_mbps(snr, cfg.phy.bandwidth_hz, cfg.phy.eta)


def _enterprise_penalty(z: float, rho: float) -> float:
    t = z - rho
    return float(t * t) if t > 0.0 else 0.0


def _top2_max(U: np.ndarray) -> Tuple[int, float, float]:
    """Return (idx_max, max1, max2). max2 is second-largest value (or -inf if size<2)."""
    if U.size == 0:
        return (0, -np.inf, -np.inf)
    idx = int(np.argmax(U))
    max1 = float(U[idx])
    if U.size == 1:
        return (idx, max1, -np.inf)
    # second max via partition
    part = np.partition(U, -2)
    max2 = float(part[-2])
    return (idx, max1, max2)


def refine_load_balance_by_overlap(
    users,
    cfg,
    clusters: List[np.ndarray],
    evals: List[dict],
    prof: Any | None = None,
) -> Tuple[List[np.ndarray], List[dict], Dict[str, Any]]:
    """
    Load-balance refinement (FAST):
      - Frozen beams: center_xy_m / center_ecef_m / R_m do NOT change during LB.
      - Incremental feasibility:
          * geom: moved user must be inside receiver R_m (+margin)
          * cap: U_to_new <= 1, and optionally <= receiver_u_max
      - Objective: "max" (minimize U_max) or "range" or "var"

    Naming convention is strict:
      - requires evals[k]["center_xy_m"] and evals[k]["center_ecef_m"]
      - no other center keys are used anywhere.
    """
    if not cfg.lb_refine.enabled:
        return clusters, evals, {"enabled": False, "moves_tried": 0, "moves_accepted": 0}

    clusters = [np.asarray(S, dtype=int).copy() for S in clusters]
    K = int(len(clusters))
    if K == 0:
        return clusters, evals, {"enabled": True, "moves_tried": 0, "moves_accepted": 0}

    # Config
    rounds = int(cfg.lb_refine.rounds)
    max_moves_round = int(cfg.lb_refine.max_moves_per_round)     # ACCEPTED moves cap
    k_receivers = int(cfg.lb_refine.k_receivers)
    k_users = int(cfg.lb_refine.k_users_from_donor)
    margin_m = float(cfg.lb_refine.intersect_margin_m)
    objective = str(cfg.lb_refine.objective)
    prefer_non_ent = bool(cfg.lb_refine.prefer_non_enterprise)
    risk_slack = float(cfg.lb_refine.risk_slack)
    exposure_slack = int(cfg.lb_refine.exposure_slack)
    allow_receiver_close = bool(cfg.lb_refine.allow_receiver_close_to_full)
    receiver_u_max = float(cfg.lb_refine.receiver_u_max)

    rho = float(cfg.ent.rho_safe)

    # Frozen beam geometry from evals (STRICT keys)
    centers_xy = np.zeros((K, 2), dtype=float)
    centers_ecef = np.zeros((K, 3), dtype=float)
    Rm = np.zeros(K, dtype=float)

    for k in range(K):
        centers_xy[k] = np.asarray(evals[k]["center_xy_m"], dtype=float).reshape(2)
        centers_ecef[k] = np.asarray(evals[k]["center_ecef_m"], dtype=float).reshape(3)
        Rm[k] = float(evals[k]["R_m"])

    # Beam pointing derived from frozen center_ecef
    v_c = centers_ecef - users.sat_ecef_m[None, :]
    d_center = np.linalg.norm(v_c, axis=1) + 1e-12
    u_c = v_c / d_center[:, None]
    theta_3db = np.arctan(Rm / d_center)

    # Precompute per-user constants (local, no mutation)
    snr_const_db = _snr_const_db_all(users, cfg)
    wdemand = users.qos_w * users.demand_mbps
    is_ent = (users.qos_w == 4)

    # Initial per-cluster metrics recomputed under frozen beams (consistent with incremental updates)
    U = np.zeros(K, dtype=float)
    risk = np.zeros(K, dtype=float)
    exposed = np.zeros(K, dtype=int)

    for k in range(K):
        S = clusters[k]
        if S.size == 0:
            U[k] = 0.0
            risk[k] = 0.0
            exposed[k] = 0
            continue

        # rates under frozen beam k
        rates = _rate_mbps_vec(users.u_sat2user[S], u_c[k], float(theta_3db[k]), snr_const_db[S], cfg)
        share = wdemand[S] / (rates + 1e-9)
        U[k] = float(share.sum())

        # enterprise penalties under frozen center_xy/Rm
        dist = np.linalg.norm(users.xy_m[S] - centers_xy[k][None, :], axis=1)
        z = dist / (Rm[k] + 1e-9)
        ent_local = is_ent[S]
        if np.any(ent_local):
            z_ent = z[ent_local]
            exposed[k] = int(np.sum(z_ent > rho))
            # risk sum
            t = z_ent - rho
            t = t[t > 0.0]
            risk[k] = float(np.sum(t * t))
        else:
            exposed[k] = 0
            risk[k] = 0.0

    # Adjacency is also frozen (beams fixed)
    adjacency = _build_overlap_adjacency_grid(centers_xy, Rm, margin_m)

    stats: Dict[str, Any] = {
        "enabled": True,
        "moves_tried": 0,
        "moves_accepted": 0,
        "objective_before": None,
        "objective_after": None,
    }

    # Objective init
    if objective == "var":
        sumU = float(U.sum())
        sumU2 = float(np.dot(U, U))
        obj_current = float(np.var(U))
    elif objective == "range":
        sumU = sumU2 = 0.0
        obj_current = float(U.max() - U.min())
    else:  # "max"
        sumU = sumU2 = 0.0
        obj_current = float(U.max())

    stats["objective_before"] = float(obj_current)

    rng = np.random.default_rng(int(cfg.run.seed) + 991)
    donor_top_k = min(12, K)

    for _rnd in range(rounds):
        # donors: highest utilization
        donors = np.argsort(-U)[:donor_top_k]

        moves_this_round = 0

        # for "max" we keep top2 to evaluate obj_after in O(1)
        idx_max, max1, max2 = _top2_max(U)

        for k_from in donors:
            if moves_this_round >= max_moves_round:
                break

            S_from = clusters[k_from]
            if S_from.size <= 1:
                continue

            neigh = adjacency[k_from]
            if neigh.size == 0:
                continue

            # receivers: lowest U among neighbors
            recv_sorted = neigh[np.argsort(U[neigh])]
            receivers = [int(x) for x in recv_sorted if int(x) != k_from]
            if not allow_receiver_close:
                receivers = [k for k in receivers if U[k] <= receiver_u_max]
            if not receivers:
                continue
            receivers = receivers[: min(k_receivers, len(receivers))]

            # Candidate users to move: pick large-share users first (best leverage on U_max)
            # rates/share for donor members under frozen donor beam
            rates_from = _rate_mbps_vec(users.u_sat2user[S_from], u_c[k_from], float(theta_3db[k_from]), snr_const_db[S_from], cfg)
            share_from = wdemand[S_from] / (rates_from + 1e-9)

            # prefer non-enterprise first if requested
            if prefer_non_ent:
                non_ent_idx = np.where(users.qos_w[S_from] != 4)[0]
                ent_idx = np.where(users.qos_w[S_from] == 4)[0]
                # sort each by descending share
                non_ent_sorted = non_ent_idx[np.argsort(-share_from[non_ent_idx])] if non_ent_idx.size else np.array([], dtype=int)
                ent_sorted = ent_idx[np.argsort(-share_from[ent_idx])] if ent_idx.size else np.array([], dtype=int)
                order = np.concatenate([non_ent_sorted, ent_sorted], axis=0)
            else:
                order = np.argsort(-share_from)

            order = order[: min(k_users, order.size)]
            if order.size == 0:
                continue

            # shuffle within top-k to avoid deterministic traps
            order = order.copy()
            rng.shuffle(order)

            for idx_in_from in order:
                if moves_this_round >= max_moves_round:
                    break

                uid = int(S_from[idx_in_from])
                s_from = float(share_from[idx_in_from])

                # donor update (always helps)
                U_from_new = float(U[k_from] - s_from)

                # enterprise donor-side deltas (only if uid is enterprise)
                uid_is_ent = bool(is_ent[uid])
                if uid_is_ent:
                    dist_from = float(np.linalg.norm(users.xy_m[uid] - centers_xy[k_from]))
                    z_from = dist_from / (Rm[k_from] + 1e-9)
                    risk_from_delta = -_enterprise_penalty(z_from, rho)
                    exposed_from_delta = - (1 if (z_from > rho) else 0)
                else:
                    risk_from_delta = 0.0
                    exposed_from_delta = 0

                accepted = False

                for k_to in receivers:
                    stats["moves_tried"] += 1

                    # geom: uid must be inside receiver radius
                    dist_to = float(np.linalg.norm(users.xy_m[uid] - centers_xy[k_to]))
                    if dist_to > (Rm[k_to] + margin_m + 1e-9):
                        continue

                    # receiver share for this uid under frozen receiver beam
                    rate_to = _rate_mbps_vec(
                        users.u_sat2user[np.array([uid], dtype=int)],
                        u_c[k_to],
                        float(theta_3db[k_to]),
                        snr_const_db[np.array([uid], dtype=int)],
                        cfg,
                    )[0]
                    s_to = float(wdemand[uid] / (float(rate_to) + 1e-9))

                    U_to_new = float(U[k_to] + s_to)
                    if U_to_new > 1.0 + 1e-9:
                        continue
                    if (not allow_receiver_close) and (U_to_new > receiver_u_max + 1e-12):
                        continue

                    # enterprise receiver deltas
                    if uid_is_ent:
                        z_to = dist_to / (Rm[k_to] + 1e-9)
                        risk_to_delta = _enterprise_penalty(z_to, rho)
                        exposed_to_delta = (1 if (z_to > rho) else 0)
                    else:
                        risk_to_delta = 0.0
                        exposed_to_delta = 0

                    # Guards: risk/exposure (pairwise)
                    before_risk = float(risk[k_from] + risk[k_to])
                    after_risk = float((risk[k_from] + risk_from_delta) + (risk[k_to] + risk_to_delta))
                    if after_risk > before_risk + risk_slack:
                        continue

                    before_exp = int(exposed[k_from] + exposed[k_to])
                    after_exp = int((exposed[k_from] + exposed_from_delta) + (exposed[k_to] + exposed_to_delta))
                    if after_exp > before_exp + exposure_slack:
                        continue

                    # Objective after updating only k_from, k_to
                    if objective == "var":
                        n = float(U.size)
                        ui = float(U[k_from]); uj = float(U[k_to])
                        sumU_new = sumU + (U_from_new - ui) + (U_to_new - uj)
                        sumU2_new = sumU2 + (U_from_new * U_from_new - ui * ui) + (U_to_new * U_to_new - uj * uj)
                        mu = sumU_new / n
                        obj_after = max(0.0, (sumU2_new / n) - mu * mu)
                    elif objective == "range":
                        umin = float(U.min())
                        umax = float(U.max())
                        # quick check; safe fallback if extrema touched
                        hits = (float(U[k_from]) == umin) or (float(U[k_from]) == umax) or (float(U[k_to]) == umin) or (float(U[k_to]) == umax)
                        if not hits:
                            obj_after = max(umax, U_from_new, U_to_new) - min(umin, U_from_new, U_to_new)
                        else:
                            U_tmp = U.copy()
                            U_tmp[k_from] = U_from_new
                            U_tmp[k_to] = U_to_new
                            obj_after = float(U_tmp.max() - U_tmp.min())
                    else:  # "max"
                        # O(1) max update using top2
                        if idx_max != k_from and idx_max != k_to:
                            obj_after = max(max1, U_from_new, U_to_new)
                        else:
                            obj_after = max(max2, U_from_new, U_to_new)

                    if obj_after >= obj_current - 1e-12:
                        continue

                    # ---- ACCEPT ----
                    # update membership
                    clusters[k_from] = np.delete(clusters[k_from], int(idx_in_from))
                    clusters[k_to] = np.append(clusters[k_to], int(uid)).astype(int)

                    # update metrics
                    U[k_from] = U_from_new
                    U[k_to] = U_to_new

                    risk[k_from] = float(risk[k_from] + risk_from_delta)
                    risk[k_to] = float(risk[k_to] + risk_to_delta)

                    exposed[k_from] = int(exposed[k_from] + exposed_from_delta)
                    exposed[k_to] = int(exposed[k_to] + exposed_to_delta)

                    # update objective bookkeeping
                    if objective == "var":
                        ui = float(U[k_from]); uj = float(U[k_to])  # already updated values
                        # recompute sums cheaply from old sums (we stored pre-update values in ui/uj above)
                        # simplest & safe: update from old sums using deltas
                        # (we can just assign)
                        sumU = sumU_new
                        sumU2 = sumU2_new
                    obj_current = float(obj_after)

                    stats["moves_accepted"] += 1
                    moves_this_round += 1
                    accepted = True

                    # refresh top2 if objective is max (accepted move changes maxima)
                    if objective == "max":
                        idx_max, max1, max2 = _top2_max(U)

                    break  # stop trying receivers for this uid

                if accepted:
                    # donor array changed; move to next donor (keeps per-donor indexing safe and fast)
                    break

    # Rebuild evals once, consistent with frozen beams and strict naming
    evals_out: List[dict] = []
    for k in range(K):
        ev = evaluate_cluster(
            users,
            clusters[k],
            cfg,
            center_xy_override=centers_xy[k],
            center_ecef_override=centers_ecef[k],
            R_m_override=float(Rm[k]),
        )
        evals_out.append(ev)

    stats["objective_after"] = float(obj_current)
    return clusters, evals_out, stats
