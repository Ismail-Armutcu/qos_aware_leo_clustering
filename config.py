# config.py
from dataclasses import dataclass
from typing import Tuple, Literal

@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

TURKEY_BBOX = BBox(36.0, 42.2, 26.0, 45.5)

# Debug: smaller region around Ankara-ish (example)
DEBUG_BBOX = BBox(39.5, 40.3, 32.3, 33.3)

@dataclass(frozen=True)
class ScenarioConfig:
    region_mode: Literal["debug", "turkey"] = "turkey"

    n_users: int = 2500
    seed: int = 1

    sat_altitude_m: float = 600_000.0

    radius_modes_km: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)

    bandwidth_hz: float = 250e6
    eirp_dbw: float = 40.0
    carrier_freq_hz: float = 20e9
    eta: float = 0.7
    loss_misc_db: float = 2.0
    noise_psd_dbw_hz: float = -228.6

    rho_safe: float = 0.7

    demand_logn_mean: float = 5.0
    demand_logn_sigma: float = 0.6
    qos_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1)

    # in ScenarioConfig
    qos_refine_rounds: int = 3
    qos_refine_kcand: int = 6
    qos_refine_max_moves_per_round: int = 2000

    # Hotspot + noise user generator
    use_hotspots: bool = True
    n_hotspots: int = 10
    # Hotspot spread in meters (std dev). You can use one sigma for all or random per-hotspot.
    hotspot_sigma_m_min: float = 5_000.0
    hotspot_sigma_m_max: float = 30_000.0
    # Fraction of users generated as uniform noise in bbox (0..1)
    noise_frac: float = 0.15
    # If True: randomly place hotspot centers inside bbox
    hotspot_centers_random: bool = True
    # If False: you can provide fixed hotspot centers in lat/lon (optional)
    # hotspot_centers_latlon: tuple[tuple[float,float], ...] = ((39.93, 32.85), (41.01, 28.98), ...)

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
