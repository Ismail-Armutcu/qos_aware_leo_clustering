# src/evaluator.py
import numpy as np

from src.coords import unit
from src.phy import fspl_db, gain_db_gaussian, snr_lin, shannon_rate_mbps


def evaluate_cluster(
    users,
    cluster_ids: np.ndarray,
    cfg,
    center_xy_override: np.ndarray | None = None,
    center_ecef_override: np.ndarray | None = None,
    R_m_override: float | None = None,
):
    """
    Evaluate one cluster under a fixed half-beamwidth-angle model.

    IMPORTANT CONTRACT (for summarize_*):
      - If returned dict has R_m != None, it MUST include 'z' with len(z)==len(cluster).

    Beam model:
      - Fixed angular half-power beamwidth: cfg.beam.theta_3db_deg
      - Beam footprint radius is derived per cluster as:
            R_m = d_center * tan(theta_3db)
        where d_center is the satellite-to-beam-center slant range.

    Optional overrides:
      - center_xy_override   : force XY center (meters)
      - center_ecef_override : force ECEF center (meters)
      - R_m_override         : force footprint radius (meters) for this evaluation only.
                               Useful in local refinements that want to preserve the current
                               footprint while testing reassignment moves. If provided,
                               geometry is feasible iff req_m <= R_m_override.
    """
    S = np.asarray(cluster_ids, dtype=int)
    m = int(S.size)
    if m == 0:
        return {"feasible": False, "reason": "empty", "R_m": None}

    xy = users.xy_m[S]  # (m,2)

    # --- center XY ---
    if center_xy_override is not None:
        c_xy = np.asarray(center_xy_override, dtype=float).reshape(2)
    else:
        c_xy = xy.mean(axis=0)

    dist_m = np.linalg.norm(xy - c_xy[None, :], axis=1)
    req_m = float(dist_m.max())

    # --- center ECEF ---
    if center_ecef_override is not None:
        c_ecef = np.asarray(center_ecef_override, dtype=float).reshape(3)
    else:
        c_ecef = users.ecef_m[S].mean(axis=0)

    # boresight sat->center
    v_c = c_ecef - users.sat_ecef_m
    d_center = float(np.linalg.norm(v_c)) + 1e-12
    u_c = unit(v_c)

    # --- fixed-angle beam model / optional fixed-radius override ---
    if R_m_override is not None:
        R_m = float(R_m_override)
        theta_3db = float(np.arctan(R_m / d_center))
    else:
        theta_3db = float(np.deg2rad(cfg.beam.theta_3db_deg))
        if theta_3db <= 0.0:
            return {
                "feasible": False,
                "reason": "geom",
                "center_xy_m": c_xy,
                "center_ecef_m": c_ecef,
                "R_m": None,
                "req_m": req_m,
                "z": None,
            }
        R_m = float(d_center * np.tan(theta_3db))

    # Always compute z whenever R_m is known (even if infeasible)
    z = dist_m / (R_m + 1e-9)

    # geometry check
    if req_m > R_m + 1e-9:
        rho = float(cfg.ent.rho_safe)
        ent = (users.qos_w[S] == 4)
        risk = float(np.sum(np.maximum(0.0, z[ent] - rho) ** 2))
        return {
            "feasible": False,
            "reason": "geom",
            "center_xy_m": c_xy,
            "center_ecef_m": c_ecef,
            "R_m": float(R_m),
            "req_m": req_m,
            "U": float("nan"),
            "risk": risk,
            "rate_mbps": None,
            "z": z,
            "u_c": u_c,
            "d_center": d_center,
            "theta_3db": theta_3db,
        }

    # off-axis angles
    u_i = users.u_sat2user[S]  # (m,3)
    cosang = np.clip(u_i @ u_c, -1.0, 1.0)
    theta = np.arccos(cosang)

    # PHY
    fspl = fspl_db(users.range_m[S], cfg.phy.carrier_freq_hz)
    g_db = gain_db_gaussian(theta, theta_3db)

    snr = snr_lin(
        cfg.phy.eirp_dbw,
        g_db,
        fspl,
        cfg.phy.loss_misc_db,
        cfg.phy.noise_psd_dbw_hz,
        cfg.phy.bandwidth_hz,
    )
    rate_mbps = shannon_rate_mbps(snr, cfg.phy.bandwidth_hz, cfg.phy.eta)

    # capacity (time-share) -- UNWEIGHTED demand
    d = users.demand_mbps[S]
    s_share = d / (rate_mbps + 1e-9)
    U = float(s_share.sum())
    cap_ok = (U <= cfg.payload.W_slots + 1e-9)

    # enterprise edge-risk
    rho = float(cfg.ent.rho_safe)
    ent = (users.qos_w[S] == 4)
    risk = float(np.sum(np.maximum(0.0, z[ent] - rho) ** 2))

    return {
        "feasible": bool(cap_ok),
        "reason": None if cap_ok else "cap",
        "center_xy_m": c_xy,
        "center_ecef_m": c_ecef,
        "R_m": float(R_m),
        "req_m": req_m,
        "U": U,
        "risk": risk,
        "rate_mbps": rate_mbps,
        "z": z,
        # extra cached stuff (ok to keep)
        "u_c": u_c,
        "d_center": d_center,
        "theta_3db": theta_3db,
    }
