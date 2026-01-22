# src/evaluator.py
import numpy as np
from src.coords import unit
from src.phy import fspl_db, gain_db_gaussian, snr_lin, shannon_rate_mbps


def choose_radius_mode_m(required_m: float, modes_km: tuple[float, ...]) -> float | None:
    for r_km in modes_km:
        r_m = r_km * 1000.0
        if required_m <= r_m + 1e-9:
            return r_m
    return None


def evaluate_cluster(
    users,
    cluster_ids: np.ndarray,
    cfg,
    center_xy_override: np.ndarray | None = None,
    center_ecef_override: np.ndarray | None = None,
):
    """
    Returns feasibility + metrics for one cluster:
    - discrete radius mode
    - Shannon rate per user (noise-limited)
    - time-share utilization U
    - enterprise edge-risk penalty (soft)

    Optional overrides:
      - center_xy_override: use this XY center (e.g., k-means center)
      - center_ecef_override: use this ECEF center (e.g., weighted mean ECEF)
    """
    S = np.asarray(cluster_ids, dtype=int)
    m = S.size
    if m == 0:
        return {"feasible": False, "reason": "empty", "R_m": None}

    xy = users.xy_m[S]  # (m,2)

    # -----------------------
    # Center in XY (override)
    # -----------------------
    if center_xy_override is not None:
        c_xy = np.asarray(center_xy_override, dtype=float).reshape(2)
    else:
        c_xy = xy.mean(axis=0)

    # Geometric radius requirement
    dist_m = np.linalg.norm(xy - c_xy[None, :], axis=1)
    req_m = float(dist_m.max())
    R_m = choose_radius_mode_m(req_m, cfg.radius_modes_km)
    if R_m is None:
        return {"feasible": False, "reason": "geom", "center_xy": c_xy, "R_m": None, "req_m": req_m}

    # -------------------------
    # Center in ECEF (override)
    # -------------------------
    if center_ecef_override is not None:
        c_ecef = np.asarray(center_ecef_override, dtype=float).reshape(3)
    else:
        # Default: mean of member ECEF
        c_ecef = users.ecef_m[S].mean(axis=0)

    # Satellite->center boresight
    v_c = c_ecef - users.sat_ecef_m
    d_center = float(np.linalg.norm(v_c))
    u_c = unit(v_c)  # (3,)

    # Off-axis angles for members
    u_i = users.u_sat2user[S]  # (m,3)
    cosang = np.clip(u_i @ u_c, -1.0, 1.0)
    theta = np.arccos(cosang)

    # Tie beamwidth to discrete footprint mode:
    theta_3db = float(np.arctan(R_m / (d_center + 1e-9)))

    # PHY: FSPL + gain + SNR + Shannon rate
    fspl = fspl_db(users.range_m[S], cfg.carrier_freq_hz)
    g_db = gain_db_gaussian(theta, theta_3db)
    snr = snr_lin(
        cfg.eirp_dbw,
        g_db,
        fspl,
        cfg.loss_misc_db,
        cfg.noise_psd_dbw_hz,
        cfg.bandwidth_hz,
    )
    rate_mbps = shannon_rate_mbps(snr, cfg.bandwidth_hz, cfg.eta)

    # Capacity via time-share utilization
    d = users.demand_mbps[S]
    wq = users.qos_w[S]
    s_share = (wq * d) / (rate_mbps + 1e-9)
    U = float(s_share.sum())
    cap_ok = U <= 1.0 + 1e-9

    # Enterprise edge-risk (soft penalty)
    rho = cfg.rho_safe
    z = dist_m / (R_m + 1e-9)
    ent = (wq == 4)
    risk = float(np.sum(np.maximum(0.0, z[ent] - rho) ** 2))

    return {
        "feasible": bool(cap_ok),
        "reason": None if cap_ok else "cap",
        "center_xy": c_xy,
        "R_m": float(R_m),
        "req_m": req_m,
        "U": U,
        "risk": risk,
        "rate_mbps": rate_mbps,
        "z": z,
    }
