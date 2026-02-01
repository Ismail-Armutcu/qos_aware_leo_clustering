# config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional


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
# Grouped config blocks (KEEP YOUR ORIGINAL CONTRACT)
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
    radius_modes_km: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class EnterpriseConfig:
    rho_safe: float = 0.7


@dataclass(frozen=True)
class TrafficConfig:
    demand_mbps_median: float = 5
    demand_logn_sigma: float = 0.6
    qos_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)  # eco/std/ent


@dataclass(frozen=True)
class QoSRefineConfig:
    rounds: int = 3
    kcand: int = 6
    max_moves_per_round: int = 2000


@dataclass(frozen=True)
class LoadBalanceRefineConfig:
    enabled: bool = True
    rounds: int = 3
    max_moves_per_round: int = 3000
    k_receivers: int = 8
    k_users_from_donor: int = 30
    intersect_margin_m: float = 0.0
    objective: str = "range"  # "range" or "var"
    prefer_non_enterprise: bool = True
    risk_slack: float = 1e-9
    exposure_slack: int = 0
    allow_receiver_close_to_full: bool = False
    receiver_u_max: float = 0.95


@dataclass(frozen=True)
class HotspotGenConfig:
    enabled: bool = True
    n_hotspots: int = 10
    hotspot_sigma_m_min: float = 5_000.0
    hotspot_sigma_m_max: float = 30_000.0
    noise_frac: float = 0.15
    hotspot_centers_random: bool = True


@dataclass(frozen=True)
class RunConfig:
    n_users: int = 250
    seed: int = 1
    enable_plots: bool = True
    verbose: bool = True
    enable_fastbp_baselines: bool = False


# -----------------------------
# NEW: Multi-satellite snapshot config (ADDED, NOT REPLACING ANYTHING)
# -----------------------------
@dataclass(frozen=True)
class MultiSatConfig:
    tle_path: str = "starlink.tle"
    elev_mask_deg: float = 25.0
    n_active: int = 30

    # Optional fixed snapshot time for reproducibility (ISO string)
    # Example: "2026-01-29T16:19:56Z"
    time_utc_iso: Optional[str] = None

    # Reference point to pick top-N satellites
    ref_site_mode: Literal["bbox_center", "ankara"] = "bbox_center"

    # Balanced association knobs (if your pipeline uses them)
    assoc_load_mode: Literal["count", "demand", "wq_demand"] = "wq_demand"
    assoc_slack: float = 0.15
    assoc_max_rounds: int = 6
    assoc_max_moves: int = 200000


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

    # NEW
    multisat: MultiSatConfig = MultiSatConfig()

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

    @property
    def ref_site_llh(self) -> tuple[float, float, float]:
        """Reference site used to pick Top-N satellites from TLE file."""
        if self.multisat.ref_site_mode == "ankara":
            return (39.9334, 32.8597, 0.0)
        return (self.lat0_deg, self.lon0_deg, 0.0)
