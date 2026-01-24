# config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal


# -----------------------------
# Region definitions
# -----------------------------
@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


TURKEY_BBOX = BBox(36.0, 42.2, 26.0, 45.5)
DEBUG_BBOX  = BBox(39.5, 40.3, 32.3, 33.3)


# -----------------------------
# Grouped config blocks
# -----------------------------
@dataclass(frozen=True)
class PhyConfig:
    sat_altitude_m: float = 600_000.0
    carrier_freq_hz: float = 20e9
    bandwidth_hz: float = 250e6
    eirp_dbw: float = 40.0
    eta: float = 0.7
    loss_misc_db: float = 2.0
    noise_psd_dbw_hz: float = -228.6


@dataclass(frozen=True)
class BeamConfig:
    # Discrete footprint radius modes (km)
    radius_modes_km: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class EnterpriseConfig:
    # Edge threshold for enterprise protection
    rho_safe: float = 0.7


@dataclass(frozen=True)
class TrafficConfig:
    # Demand distribution + QoS class ratios
    demand_logn_mean: float = 5.0
    demand_logn_sigma: float = 0.6
    qos_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)  # eco/std/ent


@dataclass(frozen=True)
class QoSRefineConfig:
    # Stage-1 refinement: enterprise angle moves
    rounds: int = 3
    kcand: int = 6
    max_moves_per_round: int = 2000


@dataclass(frozen=True)
class LoadBalanceRefineConfig:
    """
    Stage-2 refinement: load balancing via moves across intersecting beams.
    """
    enabled: bool = True

    rounds: int = 3
    tau: float = 0.85  # overloaded threshold on U
    max_moves_per_round: int = 2000

    # Candidate limits
    kcand_neighbors: int = 6       # how many intersecting neighbors to try
    user_cand_per_beam: int = 30   # how many users to try moving from an overloaded beam

    # Policy knobs
    protect_enterprise: bool = True       # do not worsen ent exposure/risk
    allow_enterprise_moves: bool = False  # if False, only move non-enterprise users
    improve_global_umax: bool = True      # if True, reject moves that increase global Umax


@dataclass(frozen=True)
class HotspotGenConfig:
    enabled: bool = True
    n_hotspots: int = 10
    hotspot_sigma_m_min: float = 5_000.0
    hotspot_sigma_m_max: float = 30_000.0
    noise_frac: float = 0.15
    hotspot_centers_random: bool = True
    # If not random, you can add:
    # hotspot_centers_latlon: tuple[tuple[float, float], ...] = ((39.93, 32.85), (41.01, 28.98), ...)


@dataclass(frozen=True)
class RunConfig:
    n_users: int = 250
    seed: int = 1

    enable_plots: bool = True
    verbose: bool = True


# -----------------------------
# Top-level scenario config
# -----------------------------
@dataclass(frozen=True)
class ScenarioConfig:
    region_mode: Literal["debug", "turkey"] = "turkey"

    run: RunConfig = RunConfig()
    phy: PhyConfig = PhyConfig()
    beam: BeamConfig = BeamConfig()
    ent: EnterpriseConfig = EnterpriseConfig()
    traffic: TrafficConfig = TrafficConfig()

    qos_refine: QoSRefineConfig = QoSRefineConfig()
    lb_refine: LoadBalanceRefineConfig = LoadBalanceRefineConfig()

    usergen: HotspotGenConfig = HotspotGenConfig()

    @property
    def bbox(self) -> BBox:
        return DEBUG_BBOX if self.region_mode == "debug" else TURKEY_BBOX

    @property
    def lat0_deg(self) -> float:
        b = self.bbox
        return (b.lat_min + b.lat_max) / 2.0

    @property
    def lon0_deg(self) -> float:
        b = self.bbox
        return (b.lon_min + b.lon_max) / 2.0
