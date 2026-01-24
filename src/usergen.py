# src/usergen.py
from __future__ import annotations

import numpy as np

from config import ScenarioConfig, BBox
from src.coords import ll_to_local_xy_m, llh_to_ecef
from src.models import User, Users


# -----------------------------
# Helpers: local tangent inverse
# -----------------------------
def local_xy_to_ll_deg(xy_m: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of ll_to_local_xy_m using the same local tangent approximation.
    x: East (m), y: North (m)
    """
    R = 6_371_000.0
    lat0 = np.deg2rad(lat0_deg)

    x = xy_m[:, 0]
    y = xy_m[:, 1]

    dlat = y / R
    dlon = x / (R * np.cos(lat0) + 1e-12)

    lat = np.deg2rad(lat0_deg) + dlat
    lon = np.deg2rad(lon0_deg) + dlon

    return np.rad2deg(lat), np.rad2deg(lon)


def _sample_uniform_latlon_in_bbox(rng: np.random.Generator, bbox: BBox, n: int) -> tuple[np.ndarray, np.ndarray]:
    lat = rng.uniform(bbox.lat_min, bbox.lat_max, size=n)
    lon = rng.uniform(bbox.lon_min, bbox.lon_max, size=n)
    return lat, lon


def _sample_hotspot_centers_xy(
    rng: np.random.Generator,
    bbox: BBox,
    lat0_deg: float,
    lon0_deg: float,
    n_hotspots: int,
) -> np.ndarray:
    """
    Generate hotspot centers uniformly inside bbox, returned in local XY meters.
    """
    latc, lonc = _sample_uniform_latlon_in_bbox(rng, bbox, n_hotspots)
    return ll_to_local_xy_m(latc, lonc, lat0_deg, lon0_deg)


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


# -----------------------------
# Main: user generation
# -----------------------------
def generate_users(cfg: ScenarioConfig) -> list[User]:
    """
    Generates users according to cfg.usergen:
      - If cfg.usergen.enabled is False -> uniform in bbox
      - Else -> Gaussian hotspots + uniform noise in bbox

    Also assigns demand and QoS according to cfg.traffic.
    """
    rng = np.random.default_rng(cfg.run.seed)
    N = int(cfg.run.n_users)
    bbox = cfg.bbox

    # --- Demand and QoS assignment ---
    # (Note: lognormal in numpy uses underlying normal mean/sigma. Keep your existing semantics.)
    median = cfg.traffic.demand_mbps_median
    sigma = cfg.traffic.demand_logn_sigma
    mu = np.log(max(median, 1e-9))
    demand = rng.lognormal(mean=mu, sigma=sigma, size=N)

    qos_probs = np.asarray(cfg.traffic.qos_probs, dtype=float)
    qos_probs = qos_probs / (qos_probs.sum() + 1e-12)
    qos_cls = rng.choice(np.array([1, 2, 4], dtype=int), size=N, p=qos_probs)

    # ---------------------------
    # Case A: uniform generation
    # ---------------------------
    if not cfg.usergen.enabled:
        lat, lon = _sample_uniform_latlon_in_bbox(rng, bbox, N)
        xy = ll_to_local_xy_m(lat, lon, cfg.lat0_deg, cfg.lon0_deg)
        ecef = llh_to_ecef(lat, lon, 0.0)

        users: list[User] = []
        for i in range(N):
            users.append(
                User(
                    id=i,
                    lat_deg=float(lat[i]),
                    lon_deg=float(lon[i]),
                    xy_m=xy[i].astype(float),
                    ecef_m=ecef[i].astype(float),
                    demand_mbps=float(demand[i]),
                    qos_w=int(qos_cls[i]),
                )
            )
        return users

    # --------------------------------------
    # Case B: hotspots + uniform noise in bbox
    # --------------------------------------
    noise_frac = _clip01(float(cfg.usergen.noise_frac))
    n_noise = int(round(N * noise_frac))
    n_hot = N - n_noise

    n_hotspots = max(int(cfg.usergen.n_hotspots), 1)

    # Hotspot centers
    if cfg.usergen.hotspot_centers_random:
        centers_xy = _sample_hotspot_centers_xy(
            rng=rng,
            bbox=bbox,
            lat0_deg=cfg.lat0_deg,
            lon0_deg=cfg.lon0_deg,
            n_hotspots=n_hotspots,
        )
    else:
        centers_latlon = getattr(cfg.usergen, "hotspot_centers_latlon", None)
        if not centers_latlon:
            raise ValueError("usergen.hotspot_centers_random=False but usergen.hotspot_centers_latlon not provided.")
        latc = np.array([p[0] for p in centers_latlon], dtype=float)
        lonc = np.array([p[1] for p in centers_latlon], dtype=float)
        centers_xy = ll_to_local_xy_m(latc, lonc, cfg.lat0_deg, cfg.lon0_deg)
        n_hotspots = int(centers_xy.shape[0])

    # Per-hotspot sigma
    smin = float(cfg.usergen.hotspot_sigma_m_min)
    smax = float(cfg.usergen.hotspot_sigma_m_max)
    if smax < smin:
        smin, smax = smax, smin
    sigmas = rng.uniform(smin, smax, size=n_hotspots).astype(float)

    # Hotspot mixture weights
    w_hotspots = rng.dirichlet(np.ones(n_hotspots, dtype=float))

    # Assign each hotspot-generated user to a hotspot id
    hotspot_ids = rng.choice(np.arange(n_hotspots), size=n_hot, p=w_hotspots)

    # Sample hotspot points with rejection to stay in bbox (via lat/lon check)
    xy_hot = np.zeros((n_hot, 2), dtype=float)
    filled = 0
    max_tries = 60  # keep bounded

    for _ in range(max_tries):
        if filled >= n_hot:
            break

        remaining = n_hot - filled
        hids = hotspot_ids[filled:filled + remaining]

        eps = rng.standard_normal((remaining, 2))
        xy_prop = centers_xy[hids] + eps * sigmas[hids][:, None]

        lat_prop, lon_prop = local_xy_to_ll_deg(xy_prop, cfg.lat0_deg, cfg.lon0_deg)
        ok = (
            (lat_prop >= bbox.lat_min) & (lat_prop <= bbox.lat_max) &
            (lon_prop >= bbox.lon_min) & (lon_prop <= bbox.lon_max)
        )

        accept = np.where(ok)[0]
        if accept.size == 0:
            continue

        take = accept[:remaining]
        xy_hot[filled:filled + take.size] = xy_prop[take]
        filled += take.size

    # Fallback: fill remaining with uniform noise
    if filled < n_hot:
        lat_u, lon_u = _sample_uniform_latlon_in_bbox(rng, bbox, n_hot - filled)
        xy_u = ll_to_local_xy_m(lat_u, lon_u, cfg.lat0_deg, cfg.lon0_deg)
        xy_hot[filled:] = xy_u

    # Noise users
    if n_noise > 0:
        lat_n, lon_n = _sample_uniform_latlon_in_bbox(rng, bbox, n_noise)
        xy_noise = ll_to_local_xy_m(lat_n, lon_n, cfg.lat0_deg, cfg.lon0_deg)
        xy_all = np.vstack([xy_hot, xy_noise])
    else:
        xy_all = xy_hot

    # Convert to lat/lon + ECEF
    lat_all, lon_all = local_xy_to_ll_deg(xy_all, cfg.lat0_deg, cfg.lon0_deg)
    ecef_all = llh_to_ecef(lat_all, lon_all, 0.0)

    # Build User objects
    users: list[User] = []
    for i in range(N):
        users.append(
            User(
                id=i,
                lat_deg=float(lat_all[i]),
                lon_deg=float(lon_all[i]),
                xy_m=xy_all[i].astype(float),
                ecef_m=ecef_all[i].astype(float),
                demand_mbps=float(demand[i]),
                qos_w=int(qos_cls[i]),
            )
        )

    return users


# -----------------------------
# Packing: Users container
# -----------------------------
def build_users_container(user_list: list[User], cfg: ScenarioConfig) -> Users:
    """
    Pack list[User] into vectorized Users container + precompute sat->user geometry.
    """
    lat = np.array([u.lat_deg for u in user_list], dtype=float)
    lon = np.array([u.lon_deg for u in user_list], dtype=float)
    xy = np.stack([u.xy_m for u in user_list], axis=0).astype(float)
    ecef = np.stack([u.ecef_m for u in user_list], axis=0).astype(float)
    demand = np.array([u.demand_mbps for u in user_list], dtype=float)
    qos = np.array([u.qos_w for u in user_list], dtype=int)

    # Satellite ECEF: above bbox center at configured altitude
    sat_ecef = llh_to_ecef(cfg.lat0_deg, cfg.lon0_deg, float(cfg.phy.sat_altitude_m)).astype(float)

    # Precompute sat->user geometry
    vec = ecef - sat_ecef[None, :]
    rng_m = np.linalg.norm(vec, axis=1)
    u_sat2user = vec / (rng_m[:, None] + 1e-12)

    return Users(
        users=user_list,
        lat_deg=lat,
        lon_deg=lon,
        xy_m=xy,
        ecef_m=ecef,
        demand_mbps=demand,
        qos_w=qos,
        sat_ecef_m=sat_ecef,
        range_m=rng_m,
        u_sat2user=u_sat2user,
    )
