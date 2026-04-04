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
    # Fixed half-power beamwidth (3 dB half-angle) in degrees.
    # Ground footprint radius is derived per beam as:
    #   R_m = d_center * tan(theta_3db)
    # where d_center is the satellite-to-beam-center slant range.
    theta_3db_deg: float = 1.0


@dataclass(frozen=True)
class EnterpriseConfig:
    rho_safe: float = 0.7


@dataclass(frozen=True)
class TrafficConfig:
    # Baseline traffic model
    demand_mbps_median: float = 5
    demand_logn_sigma: float = 0.6

    # QoS distribution (eco/std/ent)
    qos_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)

    # demand conditional on QoS
    # Multipliers applied to demand_mbps_median for (eco, std, ent)
    demand_median_mult_by_qos: Tuple[float, float, float] = (0.7, 1.0, 2.0)


@dataclass(frozen=True)
class QoSRefineConfig:
    rounds: int = 3
    kcand: int = 6
    max_moves_per_round: int = 2000


@dataclass(frozen=True)
class LoadBalanceRefineConfig:
    enabled: bool = True
    rounds: int = 2
    max_moves_per_round: int = 3000
    k_receivers: int = 6
    k_users_from_donor: int = 20
    intersect_margin_m: float = 0.1
    objective: str = "max"
    prefer_non_enterprise: bool = False
    risk_slack: float = 1e-4
    exposure_slack: int = 0
    allow_receiver_close_to_full: bool = False
    receiver_u_max: float = 0.90


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
# Multi-satellite snapshot config
# -----------------------------
@dataclass(frozen=True)
class WalkerConfig:
    # Walker-delta parameters: T / P / F
    total_sats: int = 144
    n_planes: int = 12
    phasing: int = 1

    # Orbital parameters
    inclination_deg: float = 53.0
    altitude_m: float = 600_000.0

    # Reference epoch at which the Walker mean anomalies are defined.
    # If multisat.time_utc_iso is not provided, this epoch is also used as
    # the default snapshot time for deterministic runs.
    epoch_utc_iso: str = "2026-01-01T00:00:00Z"


@dataclass(frozen=True)
class MultiSatConfig:
    # Satellite source: real TLEs or synthetic Walker-delta constellation.
    source: Literal["tle", "walker"] = "tle"

    # TLE source configuration (used when source == "tle")
    tle_path: str = "starlink.tle"

    # Synthetic Walker-delta configuration (used when source == "walker")
    walker: WalkerConfig = WalkerConfig()

    elev_mask_deg: float = 25.0

    # Optional fixed snapshot time for reproducibility (ISO string)
    # Example: "2026-01-29T16:19:56Z"
    # - TLE mode   : selects the snapshot instant used for propagation.
    # - Walker mode: selects the instant to which the synthetic constellation
    #                is propagated from walker.epoch_utc_iso.
    time_utc_iso: Optional[str] = None

    # Reference point (kept for compatibility)
    ref_site_mode: Literal["bbox_center", "ankara"] = "bbox_center"

    # Balanced association knobs
    # IMPORTANT: default is now physical demand (NOT demand*qos)
    assoc_load_mode: Literal["count", "demand", "wq_demand"] = "demand"
    assoc_slack: float = 0.15
    assoc_max_rounds: int = 6
    assoc_max_moves: int = 200000


# -----------------------------
# Payload feasibility config (UPDATED)
# -----------------------------
@dataclass(frozen=True)
class PayloadConfig:
    enabled: bool = True

    # Beam-hopping / scheduling window model:
    #   per-sat time feasibility: sum_b U_{s,b} <= J_lanes * W_slots
    J_lanes: float = 8.0
    W_slots: int = 8

    # Per-satellite beam count cap:
    #   per-sat beam feasibility: K_s <= Ks_max
    Ks_max: int = 256

    # Inner repair loop limits (cluster-level offloads)
    max_rounds: int = 8
    max_offloads_per_round: int = 12

    # Outer loop (min satellites): linear scan prefixes 1..m.
    # If None: uses all active_sats returned by sort_active_sats.
    max_prefix: Optional[int] = None


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

    multisat: MultiSatConfig = MultiSatConfig()
    payload: PayloadConfig = PayloadConfig()

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
        """Reference site used to pick Top-N satellites from TLE file (legacy)."""
        if self.multisat.ref_site_mode == "ankara":
            return (39.9334, 32.8597, 0.0)
        return (self.lat0_deg, self.lon0_deg, 0.0)
