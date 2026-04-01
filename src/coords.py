# src/coords.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# WGS84 constants
A = 6378137.0
E2 = 6.69437999014e-3
R_MEAN_EARTH_M = 6371000.0


def llh_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: float | np.ndarray) -> np.ndarray:
    """
    Convert latitude / longitude / height to ECEF coordinates in meters.
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
    Simple local tangent-plane approximation for regional areas.
    x: East (m), y: North (m)
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x = R_MEAN_EARTH_M * dlon * np.cos(lat0)
    y = R_MEAN_EARTH_M * dlat
    return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class LocalFrame:
    lat0_deg: float
    lon0_deg: float


def local_xy_to_ll_deg(x_m: np.ndarray | float, y_m: np.ndarray | float, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of ll_to_local_xy_m under the same regional tangent-plane approximation.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)

    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))

    lat = lat0 + y / R_MEAN_EARTH_M
    lon = lon0 + x / (R_MEAN_EARTH_M * np.cos(lat0) + 1e-12)

    return np.rad2deg(lat), np.rad2deg(lon)


def local_xy_to_ecef(x_m: np.ndarray | float, y_m: np.ndarray | float, lat0_deg: float, lon0_deg: float, h_m: float | np.ndarray = 0.0) -> np.ndarray:
    """
    Convert local tangent-plane coordinates back to ECEF by first recovering latitude/longitude.
    """
    lat_deg, lon_deg = local_xy_to_ll_deg(x_m, y_m, lat0_deg, lon0_deg)
    return llh_to_ecef(lat_deg, lon_deg, h_m)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)


def elevation_deg(user_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> np.ndarray:
    """
    Elevation angle of the satellite as seen from the user location(s), in degrees.
    user_ecef_m: shape (N, 3) or (3,)
    sat_ecef_m: shape (3,)
    """
    user = np.asarray(user_ecef_m, dtype=float)
    sat = np.asarray(sat_ecef_m, dtype=float)

    up = unit(user)
    los = unit(sat - user)
    sin_el = np.sum(los * up, axis=-1)
    sin_el = np.clip(sin_el, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_el))
