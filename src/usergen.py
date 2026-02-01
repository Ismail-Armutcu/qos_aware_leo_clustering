# src/usergen.py
from __future__ import annotations

import numpy as np

from config import ScenarioConfig, BBox
from src.coords import ll_to_local_xy_m, llh_to_ecef
from src.models import User, UsersRaw, Users


def local_xy_to_ll_deg(xy_m: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Approx inverse of ll_to_local_xy_m (equirectangular).
    Only used for generating random points; accuracy is fine for scenario-gen.
    """
    R = 6371000.0
    x = xy_m[:, 0]
    y = xy_m[:, 1]
    lat0 = np.radians(lat0_deg)
    lat = y / R + lat0
    lon = x / (R * np.cos(lat0)) + np.radians(lon0_deg)
    return np.degrees(lat), np.degrees(lon)


def generate_users(cfg: ScenarioConfig) -> list[User]:
    """
    Generate users in the configured bbox.
    - positions are stored in (lat,lon), local xy wrt bbox center, and ECEF.
    - satellite geometry is NOT baked here (multi-sat uses build_users_for_sat).
    """
    rng = np.random.default_rng(cfg.run.seed)
    b: BBox = cfg.bbox  # <-- FIX: cfg.bbox is a property in your original config

    lat0 = (b.lat_min + b.lat_max) / 2.0
    lon0 = (b.lon_min + b.lon_max) / 2.0

    N = cfg.run.n_users

    # Convert bbox corners to local xy to get approximate rectangle extents
    xy_min = ll_to_local_xy_m(b.lat_min, b.lon_min, lat0, lon0)
    xy_max = ll_to_local_xy_m(b.lat_max, b.lon_max, lat0, lon0)
    xmin, ymin = float(min(xy_min[0], xy_max[0])), float(min(xy_min[1], xy_max[1]))
    xmax, ymax = float(max(xy_min[0], xy_max[0])), float(max(xy_min[1], xy_max[1]))

    # Hotspots + noise
    users_xy = np.zeros((N, 2), dtype=float)

    if cfg.usergen.enabled and cfg.usergen.n_hotspots > 0:
        H = cfg.usergen.n_hotspots

        centers = np.column_stack([
            rng.uniform(xmin, xmax, size=H),
            rng.uniform(ymin, ymax, size=H),
        ])

        sigmas = rng.uniform(cfg.usergen.hotspot_sigma_m_min, cfg.usergen.hotspot_sigma_m_max, size=H)

        n_noise = int(round(cfg.usergen.noise_frac * N))
        n_clustered = N - n_noise

        hotspot_ids = rng.integers(0, H, size=n_clustered)
        users_xy[:n_clustered, 0] = centers[hotspot_ids, 0] + rng.normal(0.0, sigmas[hotspot_ids])
        users_xy[:n_clustered, 1] = centers[hotspot_ids, 1] + rng.normal(0.0, sigmas[hotspot_ids])

        users_xy[n_clustered:, 0] = rng.uniform(xmin, xmax, size=n_noise)
        users_xy[n_clustered:, 1] = rng.uniform(ymin, ymax, size=n_noise)

        users_xy[:, 0] = np.clip(users_xy[:, 0], xmin, xmax)
        users_xy[:, 1] = np.clip(users_xy[:, 1], ymin, ymax)
    else:
        users_xy[:, 0] = rng.uniform(xmin, xmax, size=N)
        users_xy[:, 1] = rng.uniform(ymin, ymax, size=N)

    lat_deg, lon_deg = local_xy_to_ll_deg(users_xy, lat0, lon0)
    ecef_m = llh_to_ecef(lat_deg, lon_deg, np.zeros(N))

    # Demand + QoS from cfg.traffic (so config actually controls generation)
    # lognormal with specified median:
    median = float(cfg.traffic.demand_mbps_median)
    sigma = float(cfg.traffic.demand_logn_sigma)
    mu = np.log(max(median, 1e-9))  # median = exp(mu)
    demand = rng.lognormal(mean=mu, sigma=sigma, size=N).astype(float)

    qos_choices = np.array([1, 2, 4], dtype=int)
    qos_probs = np.array(cfg.traffic.qos_probs, dtype=float)
    qos_probs = qos_probs / qos_probs.sum()
    qos_w = rng.choice(qos_choices, size=N, p=qos_probs).astype(int)

    users: list[User] = []
    for i in range(N):
        users.append(User(
            id=i,
            lat_deg=float(lat_deg[i]),
            lon_deg=float(lon_deg[i]),
            xy_m=users_xy[i].astype(float),
            ecef_m=ecef_m[i].astype(float),
            demand_mbps=float(demand[i]),
            qos_w=int(qos_w[i]),
        ))

    return users


def pack_users_raw(user_list: list[User]) -> UsersRaw:
    N = len(user_list)
    lat = np.zeros(N, dtype=float)
    lon = np.zeros(N, dtype=float)
    xy = np.zeros((N, 2), dtype=float)
    ecef = np.zeros((N, 3), dtype=float)
    demand = np.zeros(N, dtype=float)
    qos = np.zeros(N, dtype=int)

    for i, u in enumerate(user_list):
        lat[i] = u.lat_deg
        lon[i] = u.lon_deg
        xy[i] = u.xy_m
        ecef[i] = u.ecef_m
        demand[i] = u.demand_mbps
        qos[i] = u.qos_w

    return UsersRaw(
        users=user_list,
        lat_deg=lat,
        lon_deg=lon,
        xy_m=xy,
        ecef_m=ecef,
        demand_mbps=demand,
        qos_w=qos,
    )


def build_users_for_sat(users_raw: UsersRaw, user_ids: np.ndarray, sat_ecef_m: np.ndarray) -> Users:
    """
    Per-satellite Users view with cached sat-dependent geometry:
      - range_m
      - u_sat2user
    """
    user_ids = np.asarray(user_ids, dtype=int)
    sub_users = [users_raw.users[int(i)] for i in user_ids]

    lat = users_raw.lat_deg[user_ids]
    lon = users_raw.lon_deg[user_ids]
    xy = users_raw.xy_m[user_ids]
    ecef = users_raw.ecef_m[user_ids]
    demand = users_raw.demand_mbps[user_ids]
    qos = users_raw.qos_w[user_ids]

    sat_ecef_m = np.asarray(sat_ecef_m, dtype=float).reshape(3)

    vec = ecef - sat_ecef_m[None, :]                      # (N,3)
    range_m = np.linalg.norm(vec, axis=1).astype(float)   # (N,)
    u_sat2user = vec / (range_m[:, None] + 1e-12)         # (N,3)

    return Users(
        users=sub_users,
        lat_deg=lat,
        lon_deg=lon,
        xy_m=xy,
        ecef_m=ecef,
        demand_mbps=demand,
        qos_w=qos,
        sat_ecef_m=sat_ecef_m,
        range_m=range_m,
        u_sat2user=u_sat2user
    )
