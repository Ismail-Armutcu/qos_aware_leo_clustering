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


def evaluate_cluster(users, cluster_ids: np.ndarray, cfg):
    """
    Returns feasibility + metrics for one cluster:
    - discrete radius mode
    - Shannon rate per user (noise-limited)
    - time-share utilization U
    - enterprise edge-risk penalty (soft)
    """
    xy = users.xy_m[cluster_ids]  # (m,2)
    c_xy = xy.mean(axis=0)

    # Geometric radius requirement
    dist_m = np.linalg.norm(xy - c_xy[None, :], axis=1)
    req_m = float(dist_m.max())
    R_m = choose_radius_mode_m(req_m, cfg.radius_modes_km)
    if R_m is None:
        return {"feasible": False, "reason": "geom", "center_xy": c_xy, "R_m": None}

    # Approx center ECEF as mean of member ECEF (good enough for regional modeling)
    c_ecef = users.ecef_m[cluster_ids].mean(axis=0)
    v_c = c_ecef - users.sat_ecef_m
    d_center = float(np.linalg.norm(v_c))
    u_c = unit(v_c)  # (3,)

    # Off-axis angles for members
    u_i = users.u_sat2user[cluster_ids]  # (m,3)
    cosang = np.clip(u_i @ u_c, -1.0, 1.0)
    theta = np.arccos(cosang)

    # Tie beamwidth to discrete footprint mode:
    # treat ground edge ~ -3 dB point => theta_3db ~ atan(R / slant_range_to_center)
    theta_3db = float(np.arctan(R_m / (d_center + 1e-9)))

    # PHY: FSPL + gain + SNR + Shannon rate
    fspl = fspl_db(users.range_m[cluster_ids], cfg.carrier_freq_hz)
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
    d = users.demand_mbps[cluster_ids]
    w = users.qos_w[cluster_ids]
    s = (w * d) / (rate_mbps + 1e-9)
    U = float(s.sum())
    cap_ok = U <= 1.0 + 1e-9

    # Enterprise edge-risk (soft penalty)
    rho = cfg.rho_safe
    z = dist_m / (R_m + 1e-9)
    ent = (w == 4)
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
