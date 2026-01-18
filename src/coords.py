# src/coords.py
import numpy as np

# WGS84 constants
A = 6378137.0
E2 = 6.69437999014e-3


def llh_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: float | np.ndarray) -> np.ndarray:
    """
    Convert lat/lon/height to ECEF (meters). Works for satellites too (height above ellipsoid).
    lat_deg, lon_deg can be scalars or arrays.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    N = A / np.sqrt(1.0 - E2 * sin_lat**2)

    x = (N + h_m) * cos_lat * np.cos(lon)
    y = (N + h_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - E2) + h_m) * sin_lat

    return np.stack([x, y, z], axis=-1)


def ll_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """
    Simple local tangent plane approximation for regional areas.
    x: East (m), y: North (m)
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    R = 6371000.0  # mean Earth radius
    x = R * dlon * np.cos(lat0)
    y = R * dlat
    return np.stack([x, y], axis=-1)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)
