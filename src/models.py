# src/models.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class User:
    id: int
    lat_deg: float
    lon_deg: float
    xy_m: np.ndarray     # shape (2,)
    ecef_m: np.ndarray   # shape (3,)
    demand_mbps: float
    qos_w: int           # 1,2,4


@dataclass
class Users:
    """
    Container for fast vectorized access while preserving the User class.
    """
    users: list[User]

    # Packed arrays (N, ...)
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    xy_m: np.ndarray          # (N,2)
    ecef_m: np.ndarray        # (N,3)
    demand_mbps: np.ndarray   # (N,)
    qos_w: np.ndarray         # (N,)

    # Cached geometry w.r.t satellite
    sat_ecef_m: np.ndarray    # (3,)
    range_m: np.ndarray       # (N,)
    u_sat2user: np.ndarray    # (N,3)

    @property
    def n(self) -> int:
        return len(self.users)
