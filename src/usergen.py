# src/usergen.py
from __future__ import annotations
import numpy as np

from config import ScenarioConfig
from src.coords import ll_to_local_xy_m, llh_to_ecef
from src.models import User, Users


def local_xy_to_ll_deg(xy_m: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of ll_to_local_xy_m using the same local tangent approximation.
    x: East (m), y: North (m)
    """
    R = 6371000.0
    lat0 = np.deg2rad(lat0_deg)

    x = xy_m[:, 0]
    y = xy_m[:, 1]

    dlat = y / R
    dlon = x / (R * np.cos(lat0) + 1e-12)

    lat = np.deg2rad(lat0_deg) + dlat
    lon = np.deg2rad(lon0_deg) + dlon

    return np.rad2deg(lat), np.rad2deg(lon)


def _sample_uniform_latlon_in_bbox(rng, bbox, n: int) -> tuple[np.ndarray, np.ndarray]:
    lat = rng.uniform(bbox.lat_min, bbox.lat_max, size=n)
    lon = rng.uniform(bbox.lon_min, bbox.lon_max, size=n)
    return lat, lon


def _sample_hotspot_centers_xy(rng, cfg, n_hotspots: int) -> np.ndarray:
    """
    Generate hotspot centers uniformly inside bbox, in local XY coordinates.
    """
    latc, lonc = _sample_uniform_latlon_in_bbox(rng, cfg.bbox, n_hotspots)
    centers_xy = ll_to_local_xy_m(latc, lonc, cfg.lat0_deg, cfg.lon0_deg)
    return centers_xy


def generate_users(cfg) -> list[User]:
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n_users

    # --- Demand and QoS assignment (same as before) ---
    mu = np.log(max(cfg.demand_logn_mean, 1e-6))
    demand = rng.lognormal(mean=mu, sigma=cfg.demand_logn_sigma, size=N)
    qos_cls = rng.choice([1, 2, 4], size=N, p=cfg.qos_probs)

    # --- Choose generation mode ---
    if not getattr(cfg, "use_hotspots", False):
        # Original uniform generator
        lat, lon = _sample_uniform_latlon_in_bbox(rng, cfg.bbox, N)
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

    # --- Hotspots + noise generator ---
    noise_frac = float(getattr(cfg, "noise_frac", 0.15))
    noise_frac = min(max(noise_frac, 0.0), 1.0)

    n_noise = int(round(N * noise_frac))
    n_hot = N - n_noise

    n_hotspots = int(getattr(cfg, "n_hotspots", 6))
    n_hotspots = max(n_hotspots, 1)

    # Hotspot centers in XY
    if getattr(cfg, "hotspot_centers_random", True):
        centers_xy = _sample_hotspot_centers_xy(rng, cfg, n_hotspots)
    else:
        # Optional fixed centers in lat/lon (if you define cfg.hotspot_centers_latlon)
        centers_latlon = getattr(cfg, "hotspot_centers_latlon", None)
        if not centers_latlon:
            raise ValueError("hotspot_centers_random=False but cfg.hotspot_centers_latlon not provided.")
        latc = np.array([p[0] for p in centers_latlon], dtype=float)
        lonc = np.array([p[1] for p in centers_latlon], dtype=float)
        centers_xy = ll_to_local_xy_m(latc, lonc, cfg.lat0_deg, cfg.lon0_deg)
        n_hotspots = centers_xy.shape[0]

    # Hotspot sigmas (per-hotspot)
    smin = float(getattr(cfg, "hotspot_sigma_m_min", 3000.0))
    smax = float(getattr(cfg, "hotspot_sigma_m_max", 12000.0))
    if smax < smin:
        smin, smax = smax, smin
    sigmas = rng.uniform(smin, smax, size=n_hotspots)

    # Hotspot weights (random but deterministic)
    # Dirichlet gives a nice variability of hotspot sizes.
    alpha = np.ones(n_hotspots)
    w_hotspots = rng.dirichlet(alpha)

    # Assign each hotspot-generated user to a hotspot id
    hotspot_ids = rng.choice(np.arange(n_hotspots), size=n_hot, p=w_hotspots)

    # Sample hotspot points in XY with bbox rejection
    xy_hot = np.zeros((n_hot, 2), dtype=float)
    filled = 0
    max_tries = 50

    # Precompute bbox bounds in XY approximately by converting corners
    # (we use rejection in lat/lon space later; this is just to avoid crazy loops)
    for _ in range(max_tries):
        if filled >= n_hot:
            break
        remaining = n_hot - filled

        hids = hotspot_ids[filled:filled + remaining]
        # Sample Gaussian around each selected hotspot
        # Use isotropic sigma per hotspot
        eps = rng.standard_normal((remaining, 2))
        xy_prop = centers_xy[hids] + eps * sigmas[hids][:, None]

        # Convert proposal to lat/lon and accept only those inside bbox
        lat_prop, lon_prop = local_xy_to_ll_deg(xy_prop, cfg.lat0_deg, cfg.lon0_deg)
        ok = (
            (lat_prop >= cfg.bbox.lat_min) & (lat_prop <= cfg.bbox.lat_max) &
            (lon_prop >= cfg.bbox.lon_min) & (lon_prop <= cfg.bbox.lon_max)
        )

        accept = np.where(ok)[0]
        if accept.size == 0:
            continue

        take = accept[:remaining]
        xy_hot[filled:filled + take.size] = xy_prop[take]
        filled += take.size

    if filled < n_hot:
        # fallback: fill remaining with uniform noise in bbox
        lat_u, lon_u = _sample_uniform_latlon_in_bbox(rng, cfg.bbox, n_hot - filled)
        xy_u = ll_to_local_xy_m(lat_u, lon_u, cfg.lat0_deg, cfg.lon0_deg)
        xy_hot[filled:] = xy_u

    # Sample noise users uniformly in bbox
    if n_noise > 0:
        lat_n, lon_n = _sample_uniform_latlon_in_bbox(rng, cfg.bbox, n_noise)
        xy_noise = ll_to_local_xy_m(lat_n, lon_n, cfg.lat0_deg, cfg.lon0_deg)
        xy_all = np.vstack([xy_hot, xy_noise])
    else:
        xy_all = xy_hot

    # Convert to lat/lon, ECEF
    lat_all, lon_all = local_xy_to_ll_deg(xy_all, cfg.lat0_deg, cfg.lon0_deg)
    ecef_all = llh_to_ecef(lat_all, lon_all, 0.0)

    # Build users
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

def build_users_container(user_list, cfg: ScenarioConfig) -> Users:
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
    sat_ecef = llh_to_ecef(cfg.lat0_deg, cfg.lon0_deg, cfg.sat_altitude_m).astype(float)

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