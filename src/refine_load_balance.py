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

    Returns:
        adjacency[k] = np.ndarray of neighbor indices overlapping with k.
    """
    K = int(centers_xy.shape[0])
    adjacency: List[np.ndarray] = [np.array([], dtype=int) for _ in range(K)]

    valid = (
        np.isfinite(radii_m) & (radii_m > 0) &
        np.isfinite(centers_xy).all(axis=1)
    )
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
        grid.setdefault((int(cx), int(cy)), []).append(int(kk))

    neigh_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    # Build symmetric adjacency once by considering candidates with id > k
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
        cand_arr = cand_arr[cand_arr > k]
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

    # Deduplicate
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
    noise_total_db = float(cfg.phy.noise_psd_dbw_hz) + 10.0 * np.log10(float(cfg.phy.bandwidth_hz) + 1e-12)
    return float(cfg.phy.eirp_dbw) - fspl - float(cfg.phy.loss_misc_db) - noise_total_db


def _rate_mbps_vec(
    u_sat2user: np.ndarray,
    u_c: np.ndarray,
    theta_3db: float,
    snr_const_db: np.ndarray,
    cfg,
) -> np.ndarray:
    """Vector rate for a set of users served by ONE beam (scalar theta_3db)."""
    cosang = np.clip(u_sat2user @ u_c, -1.0, 1.0)
    theta = np.arccos(cosang)
    g_db = gain_db_gaussian(theta, float(theta_3db))
    snr = 10.0 ** ((snr_const_db + g_db) / 10.0)
    return shannon_rate_mbps(snr, cfg.phy.bandwidth_hz, cfg.phy.eta)


_LN2 = float(np.log(2.0))


def _gain_db_gaussian_multi(theta_rad: np.ndarray, theta_3db_rad: np.ndarray) -> np.ndarray:
    """
    Gaussian mainlobe gain with per-sample theta_3db (vectorized).
    Matches src.phy.gain_db_gaussian() for scalar theta_3db.
    """
    g_lin = np.exp(-_LN2 * (theta_rad / (theta_3db_rad + 1e-12)) ** 2)
    return 10.0 * np.log10(g_lin + 1e-12)


def _rate_uid_to_beams(
    u_uid: np.ndarray,                 # (3,)
    beam_ids: np.ndarray,              # (M,)
    u_c: np.ndarray,                   # (K,3)
    theta_3db: np.ndarray,             # (K,)
    snr_const_db_uid: float,
    cfg,
) -> np.ndarray:
    """
    Rate for a single uid if served by multiple beams (beam_ids). Vectorized over beams.
    """
    cosang = np.clip(u_c[beam_ids] @ u_uid, -1.0, 1.0)   # (M,)
    theta = np.arccos(cosang)                             # (M,)
    g_db = _gain_db_gaussian_multi(theta, theta_3db[beam_ids])
    snr = 10.0 ** ((snr_const_db_uid + g_db) / 10.0)
    return shannon_rate_mbps(snr, cfg.phy.bandwidth_hz, cfg.phy.eta)


def _enterprise_penalty(z: float, rho: float) -> float:
    t = z - rho
    return float(t * t) if t > 0.0 else 0.0


def refine_load_balance_by_overlap(
    users,
    cfg,
    clusters: List[np.ndarray],
    evals: List[dict],
    prof: Any | None = None,
) -> Tuple[List[np.ndarray], List[dict], Dict[str, Any]]:
    """
    Load-balance refinement (FAST, improved U_max quality):

      - Frozen beams during LB: centers/R are taken from evals and never change.
      - Incremental updates for U / enterprise risk / exposed count.
      - Objective:
          * "max": minimize U_max using a peak-aware acceptance rule:
              - if multiple clusters sit at current max, accept "peak-count reducing" moves
                that pull a peak donor below max without pushing receiver to max.
              - if unique peak, accept only moves that reduce the peak value.
          * "range"/"var": keep standard objective improvement checks.

    End-of-stage:
      - Recompute evals ONLY for clusters touched by accepted moves, using evaluate_cluster() with overrides
        to match the frozen beam geometry.
    """
    if not cfg.lb_refine.enabled:
        return clusters, evals, {"enabled": False, "moves_tried": 0, "moves_accepted": 0}

    clusters = [np.asarray(S, dtype=int).copy() for S in clusters]
    K = int(len(clusters))
    if K == 0:
        return clusters, evals, {"enabled": True, "moves_tried": 0, "moves_accepted": 0}

    # Config
    rounds = int(cfg.lb_refine.rounds)
    max_moves_round = int(cfg.lb_refine.max_moves_per_round)     # accepted moves cap per round
    k_receivers = int(cfg.lb_refine.k_receivers)
    k_users = int(cfg.lb_refine.k_users_from_donor)
    margin_m = float(cfg.lb_refine.intersect_margin_m)
    objective = str(cfg.lb_refine.objective).lower().strip()

    prefer_non_ent = bool(cfg.lb_refine.prefer_non_enterprise)
    risk_slack = float(cfg.lb_refine.risk_slack)
    exposure_slack = int(cfg.lb_refine.exposure_slack)
    allow_receiver_close = bool(cfg.lb_refine.allow_receiver_close_to_full)
    receiver_u_max = float(cfg.lb_refine.receiver_u_max)

    rho = float(cfg.ent.rho_safe)

    # Frozen beam geometry from evals
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

    # Per-user constants
    snr_const_db = _snr_const_db_all(users, cfg)  # (N,)
    wdemand = users.qos_w * users.demand_mbps     # (N,)
    is_ent = (users.qos_w == 4)

    # Initial per-cluster metrics under frozen beams
    U = np.zeros(K, dtype=float)
    risk = np.zeros(K, dtype=float)
    exposed = np.zeros(K, dtype=int)

    for k in range(K):
        S = clusters[k]
        if S.size == 0:
            continue

        rates = _rate_mbps_vec(users.u_sat2user[S], u_c[k], float(theta_3db[k]), snr_const_db[S], cfg)
        share = wdemand[S] / (rates + 1e-9)
        U[k] = float(share.sum())

        dist = np.linalg.norm(users.xy_m[S] - centers_xy[k][None, :], axis=1)
        z = dist / (Rm[k] + 1e-9)
        ent_local = is_ent[S]
        if np.any(ent_local):
            z_ent = z[ent_local]
            exposed[k] = int(np.sum(z_ent > rho))
            t = z_ent - rho
            t = t[t > 0.0]
            risk[k] = float(np.sum(t * t))

    adjacency = _build_overlap_adjacency_grid(centers_xy, Rm, margin_m)

    stats: Dict[str, Any] = {
        "enabled": True,
        "moves_tried": 0,
        "moves_accepted": 0,
        "objective_before": None,
        "objective_after": None,
    }

    # Objective init (for bookkeeping)
    if objective == "var":
        n = float(U.size)
        sumU = float(U.sum())
        sumU2 = float(np.dot(U, U))
        mu = sumU / n
        obj_current = max(0.0, (sumU2 / n) - mu * mu)
    elif objective == "range":
        sumU = sumU2 = 0.0
        obj_current = float(U.max() - U.min())
    else:  # "max"
        sumU = sumU2 = 0.0
        obj_current = float(U.max())

    stats["objective_before"] = float(obj_current)

    rng = np.random.default_rng(int(cfg.run.seed) + 991)

    donor_top_k = min(24, K)  # still small; avoids missing new peaks after some moves
    max_moves_per_donor = 3   # bounded, improves peak chipping

    # Peak-aware acceptance tolerances
    max_eps = 1e-12
    peak_drop_tol = 1e-6

    changed = np.zeros(K, dtype=bool)

    for _rnd in range(rounds):
        moves_this_round = 0

        # Keep iterating while we can make progress this round
        while moves_this_round < max_moves_round:
            progressed = False

            if objective == "max":
                max1 = float(U.max())
                count_max = int(np.sum(U >= (max1 - 1e-12)))
            else:
                max1 = 0.0
                count_max = 0

            donors_sorted = np.argsort(-U)[:donor_top_k]

            for k_from in donors_sorted:
                if moves_this_round >= max_moves_round:
                    break

                if objective == "max":
                    # Donors are sorted; once we fall below peak, stop.
                    if float(U[k_from]) < (max1 - 1e-12):
                        break

                S_from = clusters[k_from]
                if S_from.size <= 1:
                    continue

                neigh = adjacency[k_from]
                if neigh.size == 0:
                    continue

                # Receiver shortlist: lowest U among neighbors
                recv_sorted = neigh[np.argsort(U[neigh])]
                receivers = recv_sorted[recv_sorted != k_from]
                if receivers.size == 0:
                    continue

                if not allow_receiver_close:
                    receivers = receivers[U[receivers] <= receiver_u_max]
                    if receivers.size == 0:
                        continue

                receivers = receivers[: min(k_receivers, receivers.size)].astype(int)
                if receivers.size == 0:
                    continue

                # Up to a few accepted moves from this donor (chip the peak)
                for _m in range(max_moves_per_donor):
                    if moves_this_round >= max_moves_round:
                        break

                    S_from = clusters[k_from]
                    if S_from.size <= 1:
                        break

                    # Recompute donor shares under frozen donor beam (after deletions)
                    rates_from = _rate_mbps_vec(
                        users.u_sat2user[S_from],
                        u_c[k_from],
                        float(theta_3db[k_from]),
                        snr_const_db[S_from],
                        cfg,
                    )
                    share_from = wdemand[S_from] / (rates_from + 1e-9)

                    # Candidate users: largest share first
                    if prefer_non_ent:
                        non_ent_idx = np.where(users.qos_w[S_from] != 4)[0]
                        ent_idx = np.where(users.qos_w[S_from] == 4)[0]
                        non_ent_sorted = non_ent_idx[np.argsort(-share_from[non_ent_idx])] if non_ent_idx.size else np.array([], dtype=int)
                        ent_sorted = ent_idx[np.argsort(-share_from[ent_idx])] if ent_idx.size else np.array([], dtype=int)
                        order = np.concatenate([non_ent_sorted, ent_sorted], axis=0)
                    else:
                        order = np.argsort(-share_from)

                    order = order[: min(k_users, order.size)]
                    if order.size == 0:
                        break

                    # Best-first then slight randomness after top few
                    top_det = min(4, int(order.size))
                    order_det = order[:top_det]
                    order_rest = order[top_det:].copy()
                    if order_rest.size > 1:
                        rng.shuffle(order_rest)
                    order = np.concatenate([order_det, order_rest], axis=0)

                    accepted = False

                    # Refresh peak stats before trying (only for max objective)
                    if objective == "max":
                        max1 = float(U.max())
                        count_max = int(np.sum(U >= (max1 - 1e-12)))

                    for idx_in_from in order:
                        if moves_this_round >= max_moves_round:
                            break

                        uid = int(S_from[int(idx_in_from)])
                        s_from = float(share_from[int(idx_in_from)])

                        U_from_new = float(U[k_from] - s_from)

                        # Enterprise donor deltas
                        uid_is_ent = bool(is_ent[uid])
                        if uid_is_ent:
                            dist_from = float(np.linalg.norm(users.xy_m[uid] - centers_xy[k_from]))
                            z_from = dist_from / (Rm[k_from] + 1e-9)
                            risk_from_delta = -_enterprise_penalty(z_from, rho)
                            exposed_from_delta = -(1 if (z_from > rho) else 0)
                        else:
                            risk_from_delta = 0.0
                            exposed_from_delta = 0

                        # ---- Vectorized receiver screening for this uid ----
                        recv_arr = receivers
                        dxy = centers_xy[recv_arr] - users.xy_m[uid][None, :]
                        dist_to = np.linalg.norm(dxy, axis=1)
                        ok_geom = dist_to <= (Rm[recv_arr] + margin_m + 1e-9)
                        if not np.any(ok_geom):
                            continue

                        recv_arr2 = recv_arr[ok_geom]
                        dist_to2 = dist_to[ok_geom]

                        u_uid = users.u_sat2user[uid]
                        rate_to = _rate_uid_to_beams(
                            u_uid=u_uid,
                            beam_ids=recv_arr2,
                            u_c=u_c,
                            theta_3db=theta_3db,
                            snr_const_db_uid=float(snr_const_db[uid]),
                            cfg=cfg,
                        )
                        s_to = float(wdemand[uid]) / (rate_to + 1e-9)
                        U_to_new = U[recv_arr2] + s_to

                        ok_cap = U_to_new <= (1.0 + 1e-9)
                        if not allow_receiver_close:
                            ok_cap &= (U_to_new <= (receiver_u_max + 1e-12))
                        if not np.any(ok_cap):
                            continue

                        recv_arr3 = recv_arr2[ok_cap]
                        dist_to3 = dist_to2[ok_cap]
                        U_to_new3 = U_to_new[ok_cap]

                        # Best-first receivers: smallest resulting receiver utilization
                        ord_r = np.argsort(U_to_new3)
                        recv_arr3 = recv_arr3[ord_r]
                        dist_to3 = dist_to3[ord_r]
                        U_to_new3 = U_to_new3[ord_r]

                        for k_to, dist_to_val, U_to_new_val in zip(recv_arr3, dist_to3, U_to_new3):
                            stats["moves_tried"] += 1

                            # Enterprise receiver deltas
                            if uid_is_ent:
                                z_to = float(dist_to_val) / (Rm[int(k_to)] + 1e-9)
                                risk_to_delta = _enterprise_penalty(z_to, rho)
                                exposed_to_delta = (1 if (z_to > rho) else 0)
                            else:
                                risk_to_delta = 0.0
                                exposed_to_delta = 0

                            # Guards: risk/exposure (pairwise)
                            before_risk = float(risk[k_from] + risk[int(k_to)])
                            after_risk = float((risk[k_from] + risk_from_delta) + (risk[int(k_to)] + risk_to_delta))
                            if after_risk > before_risk + risk_slack:
                                continue

                            before_exp = int(exposed[k_from] + exposed[int(k_to)])
                            after_exp = int((exposed[k_from] + exposed_from_delta) + (exposed[int(k_to)] + exposed_to_delta))
                            if after_exp > before_exp + exposure_slack:
                                continue

                            # Objective acceptance
                            if objective == "var":
                                n = float(U.size)
                                ui = float(U[k_from]); uj = float(U[int(k_to)])
                                sumU_new = sumU + (U_from_new - ui) + (float(U_to_new_val) - uj)
                                sumU2_new = sumU2 + (U_from_new * U_from_new - ui * ui) + (float(U_to_new_val) * float(U_to_new_val) - uj * uj)
                                mu = sumU_new / n
                                obj_after = max(0.0, (sumU2_new / n) - mu * mu)
                                if obj_after >= obj_current - 1e-12:
                                    continue

                            elif objective == "range":
                                umin = float(U.min()); umax = float(U.max())
                                hits = (float(U[k_from]) == umin) or (float(U[k_from]) == umax) or (float(U[int(k_to)]) == umin) or (float(U[int(k_to)]) == umax)
                                if not hits:
                                    obj_after = max(umax, U_from_new, float(U_to_new_val)) - min(umin, U_from_new, float(U_to_new_val))
                                else:
                                    U_tmp = U.copy()
                                    U_tmp[k_from] = U_from_new
                                    U_tmp[int(k_to)] = float(U_to_new_val)
                                    obj_after = float(U_tmp.max() - U_tmp.min())
                                if obj_after >= obj_current - 1e-12:
                                    continue

                            else:  # "max"
                                # Never increase the peak
                                if float(U_to_new_val) > max1 + 1e-9:
                                    continue

                                donor_is_peak = (float(U[k_from]) >= max1 - max_eps)
                                if not donor_is_peak:
                                    continue

                                if count_max > 1:
                                    # Reduce peak count: donor drops below max and receiver stays below max
                                    if not ((U_from_new < max1 - peak_drop_tol) and (float(U_to_new_val) < max1 - peak_drop_tol)):
                                        continue
                                else:
                                    # Unique peak: must reduce peak value itself
                                    new_peak = max(float(U_to_new_val), float(U_from_new))
                                    if not (new_peak < max1 - peak_drop_tol):
                                        continue

                            # ---- ACCEPT ----
                            clusters[k_from] = np.delete(clusters[k_from], int(idx_in_from))
                            clusters[int(k_to)] = np.append(clusters[int(k_to)], int(uid)).astype(int)

                            U[k_from] = float(U_from_new)
                            U[int(k_to)] = float(U_to_new_val)

                            risk[k_from] = float(risk[k_from] + risk_from_delta)
                            risk[int(k_to)] = float(risk[int(k_to)] + risk_to_delta)

                            exposed[k_from] = int(exposed[k_from] + exposed_from_delta)
                            exposed[int(k_to)] = int(exposed[int(k_to)] + exposed_to_delta)

                            changed[k_from] = True
                            changed[int(k_to)] = True

                            if objective == "var":
                                sumU = sumU_new
                                sumU2 = sumU2_new
                                obj_current = float(obj_after)
                            elif objective == "range":
                                obj_current = float(obj_after)
                            else:
                                obj_current = float(U.max())

                            stats["moves_accepted"] += 1
                            moves_this_round += 1
                            accepted = True
                            progressed = True
                            break  # stop trying receivers for this uid

                        if accepted:
                            break  # recompute donor shares after accept

                    if not accepted:
                        break  # no move found for this donor

                # end _m loop

            # end donor loop

            if not progressed:
                break  # no more improvements this round

    # Rebuild evals only for changed clusters (others unchanged, safe to reuse)
    evals_out: List[dict] = [dict(ev) for ev in evals]
    for k in np.where(changed)[0]:
        evs = evaluate_cluster(
            users,
            clusters[k],
            cfg,
            center_xy_override=centers_xy[k],
            center_ecef_override=centers_ecef[k],
            R_m_override=float(Rm[k]),
        )
        if prof:
            prof.inc("eval_calls")
        evals_out[k] = evs

    stats["objective_after"] = float(U.max()) if objective == "max" else float(obj_current)
    return clusters, evals_out, stats
