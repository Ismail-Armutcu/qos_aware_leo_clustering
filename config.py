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
    demand_mbps_median: float = 0.005
    demand_logn_sigma: float = 0.6
    qos_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)  # eco/std/ent


@dataclass(frozen=True)
class QoSRefineConfig:
    # Stage-1 refinement: enterprise angle moves
    rounds: int = 3
    kcand: int = 6
    max_moves_per_round: int = 2000


# ----------------------------
# Config (nested under cfg.lb_refine)
# ----------------------------
@dataclass(frozen=True)
class LoadBalanceRefineConfig:
    enabled: bool = True
    rounds: int = 3
    # How many attempted moves per round (upper bound)
    max_moves_per_round: int = 3000
    # For each donor cluster, how many receiver clusters to consider (by adjacency+best utility)
    k_receivers: int = 8
    # Within a donor cluster, only consider up to this many candidate users for moving
    k_users_from_donor: int = 30
    # Only allow moves between beams whose circles overlap (intersect)
    # dist(center_i, center_j) <= (R_i + R_j + intersect_margin_m)
    intersect_margin_m: float = 0.0
    # Objective: reduce imbalance; you can choose "range" (max-min) or "var" (variance)
    objective: str = "range"  # "range" or "var"
    # Optional: don’t touch enterprise users unless needed
    prefer_non_enterprise: bool = True
    # Guard against making enterprise risk worse
    # Accept move if (risk_after <= risk_before + risk_slack)
    risk_slack: float = 1e-9
    # Also guard enterprise exposure count (z>rho) from increasing too much
    # Accept move if (exposed_after <= exposed_before + exposure_slack)
    exposure_slack: int = 0
    # If True, allow receiver to be slightly higher utilization as long as imbalance improves.
    allow_receiver_close_to_full: bool = False
    receiver_u_max: float = 0.95  # used if allow_receiver_close_to_full is False


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
    enable_fastbp_baselines: bool = False


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
