# src/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class User:
    id: int
    lat_deg: float
    lon_deg: float
    xy_m: np.ndarray     # (2,)
    ecef_m: np.ndarray   # (3,)
    demand_mbps: float
    qos_w: int           # 1,2,4


@dataclass
class UsersRaw:
    """Vectorized user container WITHOUT satellite-dependent cached geometry."""
    users: list[User]

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    xy_m: np.ndarray         # (N,2)
    ecef_m: np.ndarray       # (N,3)
    demand_mbps: np.ndarray  # (N,)
    qos_w: np.ndarray        # (N,)

    @property
    def n(self) -> int:
        return len(self.users)


@dataclass
class Users(UsersRaw):
    """Vectorized user container WITH satellite-dependent cached geometry."""
    sat_ecef_m: np.ndarray    # (3,)
    range_m: np.ndarray       # (N,)
    u_sat2user: np.ndarray    # (N,3)
