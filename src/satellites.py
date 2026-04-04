# src/satellites.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from src.coords import llh_to_ecef, unit

if TYPE_CHECKING:
    from config import ScenarioConfig, BBox


# -----------------------------
# Constants
# -----------------------------
EARTH_RADIUS_M = 6_371_000.0
MU_EARTH_M3_S2 = 3.986004418e14
SIDEREAL_DAY_S = 86164.0905
EARTH_ROT_RATE_RAD_S = 2.0 * np.pi / SIDEREAL_DAY_S


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ActiveSat:
    """Active satellite at a given instant (snapshot)."""
    name: str
    ecef_m: np.ndarray   # (3,)
    elev_ref_deg: float  # max elevation across anchors


@dataclass(frozen=True)
class AssocResult:
    """Result of balanced user->satellite association."""
    assign: np.ndarray                 # (N,), sat index or -1
    sat_user_ids: List[np.ndarray]     # length S, each array of user indices
    loads: np.ndarray                  # (S,), load per sat (count or demand or wq-demand)
    cap: float                         # soft cap used for balancing
    n_unserved: int
    n_moves: int


@dataclass(frozen=True)
class Anchor:
    lat_deg: float
    lon_deg: float
    height_m: float = 0.0


# -----------------------------
# Helper functions
# -----------------------------
def _parse_utc_iso(iso_z: str) -> datetime:
    """
    Parse an ISO time string. Accepts 'Z' suffix for UTC.
    Returns timezone-aware UTC datetime.
    """
    s = iso_z.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _choose_snapshot_time_utc(sats: Sequence[object]) -> datetime:
    """
    Deterministic snapshot selection for TLE mode: midpoint of TLE epoch span (UTC).
    """
    epochs = [sat.epoch.utc_datetime() for sat in sats]
    t_min = min(epochs)
    t_max = max(epochs)
    return _normalize_utc(t_min + (t_max - t_min) / 2)


def _build_anchor_grid(bbox: "BBox", n_lat: int = 3, n_lon: int = 3) -> List[Anchor]:
    """
    Build an n_lat x n_lon anchor grid over the region bounding box.
    Anchor-only selection is fast and does not require user data.
    """
    if n_lat < 2 or n_lon < 2:
        raise ValueError("Anchor grid must be at least 2x2 (use 3x3 by default).")

    lats = np.linspace(float(bbox.lat_min), float(bbox.lat_max), int(n_lat))
    lons = np.linspace(float(bbox.lon_min), float(bbox.lon_max), int(n_lon))
    anchors: List[Anchor] = []
    for lat in lats:
        for lon in lons:
            anchors.append(Anchor(lat_deg=float(lat), lon_deg=float(lon), height_m=0.0))
    return anchors


def _anchors_to_ecef_m(anchors: Sequence[Anchor]) -> np.ndarray:
    lat = np.array([a.lat_deg for a in anchors], dtype=float)
    lon = np.array([a.lon_deg for a in anchors], dtype=float)
    h = np.array([a.height_m for a in anchors], dtype=float)
    return np.asarray(llh_to_ecef(lat, lon, h), dtype=float)


def _quality_from_elev_deg(elev_deg: np.ndarray, emin_deg: float, mode: str = "sin") -> np.ndarray:
    """
    Convert elevation (deg) to a nonnegative quality, with q=0 below mask.
    """
    q = np.zeros_like(elev_deg, dtype=np.float64)
    mask = elev_deg >= float(emin_deg)
    if not np.any(mask):
        return q

    if mode == "sin":
        q[mask] = np.sin(np.deg2rad(elev_deg[mask]))
    elif mode == "sin2":
        s = np.sin(np.deg2rad(elev_deg[mask]))
        q[mask] = s * s
    elif mode == "linear":
        q[mask] = elev_deg[mask]
    else:
        raise ValueError(f"Unknown quality mode: {mode}")
    return q


def _walker_mean_motion_rad_s(altitude_m: float) -> float:
    r_orbit_m = EARTH_RADIUS_M + float(altitude_m)
    return float(np.sqrt(MU_EARTH_M3_S2 / (r_orbit_m ** 3)))


def _walker_delta_elements(total_sats: int, n_planes: int, phasing: int, inc_deg: float) -> list[tuple[str, float, float, float]]:
    """
    Walker-delta constellation definition.

    Returns a list of tuples: (name, RAAN_rad, inc_rad, M0_rad)
    """
    T = int(total_sats)
    P = int(n_planes)
    F = int(phasing)
    if T <= 0 or P <= 0:
        raise ValueError("Walker parameters must satisfy total_sats > 0 and n_planes > 0.")
    if T % P != 0:
        raise ValueError(f"Walker-delta requires total_sats divisible by n_planes, got T={T}, P={P}.")

    sats_per_plane = T // P
    inc = float(np.deg2rad(inc_deg))
    elems: list[tuple[str, float, float, float]] = []
    for p in range(P):
        raan = 2.0 * np.pi * p / P
        for s in range(sats_per_plane):
            M0 = 2.0 * np.pi * s / sats_per_plane + 2.0 * np.pi * F * p / T
            elems.append((f"WALKER-P{p:02d}-S{s:02d}", float(raan), inc, float(M0)))
    return elems


def _walker_sat_ecef_m(raan_rad: float, inc_rad: float, M0_rad: float, dt_s: float, altitude_m: float) -> np.ndarray:
    """
    Circular-orbit Walker satellite propagated from the reference epoch into ECEF.
    """
    r_orbit_m = EARTH_RADIUS_M + float(altitude_m)
    n = _walker_mean_motion_rad_s(altitude_m)
    u = float(M0_rad + n * dt_s)

    cosO = float(np.cos(raan_rad))
    sinO = float(np.sin(raan_rad))
    cosi = float(np.cos(inc_rad))
    sini = float(np.sin(inc_rad))
    cosu = float(np.cos(u))
    sinu = float(np.sin(u))

    x_eci = r_orbit_m * (cosO * cosu - sinO * sinu * cosi)
    y_eci = r_orbit_m * (sinO * cosu + cosO * sinu * cosi)
    z_eci = r_orbit_m * (sinu * sini)

    theta = EARTH_ROT_RATE_RAD_S * float(dt_s)
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    x_ecef = c * x_eci + s * y_eci
    y_ecef = -s * x_eci + c * y_eci
    return np.array([x_ecef, y_ecef, z_eci], dtype=float)


def _rank_visible_sats(
    sat_names: Sequence[str],
    sat_ecef_m: np.ndarray,
    anchor_ecef_m: np.ndarray,
    emin_deg: float,
    quality_mode: str = "sin",
) -> List[ActiveSat]:
    """
    Common ranking logic for both TLE and Walker sources.
    """
    if sat_ecef_m.size == 0:
        return []

    elev = _elevations_deg(anchor_ecef_m, sat_ecef_m).astype(np.float64)  # (P, N)
    cand_mask = (elev >= float(emin_deg)).any(axis=0)
    cand_idx = np.where(cand_mask)[0]
    if cand_idx.size == 0:
        return []

    elev_c = elev[:, cand_idx]
    q = _quality_from_elev_deg(elev_c, emin_deg=float(emin_deg), mode=quality_mode)

    Mc = q.shape[1]
    best = np.zeros(q.shape[0], dtype=np.float64)
    chosen = np.zeros(Mc, dtype=bool)
    order_local: List[int] = []

    while True:
        improvement = np.maximum(q - best[:, None], 0.0)
        delta = improvement.sum(axis=0)
        delta[chosen] = -1.0
        j = int(np.argmax(delta))
        if delta[j] <= 0.0:
            break
        chosen[j] = True
        order_local.append(j)
        best = np.maximum(best, q[:, j])

    remaining = np.where(~chosen)[0]
    if remaining.size > 0:
        rem_score = q[:, remaining].sum(axis=0)
        rem_order = remaining[np.argsort(-rem_score, kind="mergesort")]
        order_local.extend(rem_order.tolist())

    selected_sat_idx = cand_idx[np.array(order_local, dtype=int)]

    active: List[ActiveSat] = []
    for si in selected_sat_idx:
        active.append(
            ActiveSat(
                name=str(sat_names[int(si)]),
                ecef_m=np.asarray(sat_ecef_m[int(si)], dtype=float),
                elev_ref_deg=float(elev[:, int(si)].max()),
            )
        )
    return active


def _sort_active_sats_from_tle(
    scenarioConfig: "ScenarioConfig",
    *,
    t0_utc: Optional[datetime],
    n_lat_anchors: int,
    n_lon_anchors: int,
    quality_mode: str,
) -> tuple[datetime, List[ActiveSat]]:
    from skyfield.api import load
    from skyfield.framelib import itrs

    tle_path = scenarioConfig.multisat.tle_path
    emin_deg = float(scenarioConfig.multisat.elev_mask_deg)
    bbox = scenarioConfig.bbox

    ts = load.timescale()
    sats = load.tle_file(tle_path)
    if len(sats) == 0:
        raise RuntimeError(f"No satellites loaded from TLE file: {tle_path}")

    if t0_utc is not None:
        t0 = _normalize_utc(t0_utc)
    elif scenarioConfig.multisat.time_utc_iso is not None:
        t0 = _parse_utc_iso(scenarioConfig.multisat.time_utc_iso)
    else:
        t0 = _choose_snapshot_time_utc(sats)

    t = ts.from_datetime(t0)
    sat_names = [sat.name for sat in sats]
    sat_ecef_m = np.stack(
        [np.asarray(sat.at(t).frame_xyz(itrs).km, dtype=float) * 1000.0 for sat in sats],
        axis=0,
    )

    anchors = _build_anchor_grid(bbox, n_lat=n_lat_anchors, n_lon=n_lon_anchors)
    anchor_ecef_m = _anchors_to_ecef_m(anchors)
    active = _rank_visible_sats(sat_names, sat_ecef_m, anchor_ecef_m, emin_deg, quality_mode)
    return t0, active


def _sort_active_sats_from_walker(
    scenarioConfig: "ScenarioConfig",
    *,
    t0_utc: Optional[datetime],
    n_lat_anchors: int,
    n_lon_anchors: int,
    quality_mode: str,
) -> tuple[datetime, List[ActiveSat]]:
    ms = scenarioConfig.multisat
    walker = ms.walker
    emin_deg = float(ms.elev_mask_deg)
    bbox = scenarioConfig.bbox

    epoch_utc = _parse_utc_iso(walker.epoch_utc_iso)
    if t0_utc is not None:
        t0 = _normalize_utc(t0_utc)
    elif ms.time_utc_iso is not None:
        t0 = _parse_utc_iso(ms.time_utc_iso)
    else:
        t0 = epoch_utc

    dt_s = float((t0 - epoch_utc).total_seconds())
    elems = _walker_delta_elements(
        total_sats=int(walker.total_sats),
        n_planes=int(walker.n_planes),
        phasing=int(walker.phasing),
        inc_deg=float(walker.inclination_deg),
    )

    sat_names = [name for name, _, _, _ in elems]
    sat_ecef_m = np.stack(
        [
            _walker_sat_ecef_m(raan, inc, M0, dt_s, float(walker.altitude_m))
            for _, raan, inc, M0 in elems
        ],
        axis=0,
    )

    anchors = _build_anchor_grid(bbox, n_lat=n_lat_anchors, n_lon=n_lon_anchors)
    anchor_ecef_m = _anchors_to_ecef_m(anchors)
    active = _rank_visible_sats(sat_names, sat_ecef_m, anchor_ecef_m, emin_deg, quality_mode)
    return t0, active


# -----------------------------
# Satellite selection API
# -----------------------------
def sort_active_sats(
    scenarioConfig: "ScenarioConfig",
    *,
    t0_utc: Optional[datetime] = None,
    n_lat_anchors: int = 3,
    n_lon_anchors: int = 3,
    quality_mode: str = "sin",
) -> tuple[datetime, List[ActiveSat]]:
    """
    Returns a UTC snapshot time and an ordered list of active satellites.

    Supported satellite sources:
      - source == "tle"    : real satellites from a TLE file
      - source == "walker" : synthetic Walker-delta constellation

    Ranking behavior is identical in both modes:
      1) filter satellites visible above the elevation mask at any anchor,
      2) greedily rank them by marginal anchor-quality gain,
      3) append the remaining visible satellites by fallback total quality.
    """
    source = str(getattr(scenarioConfig.multisat, "source", "tle")).lower()
    if source == "tle":
        return _sort_active_sats_from_tle(
            scenarioConfig,
            t0_utc=t0_utc,
            n_lat_anchors=n_lat_anchors,
            n_lon_anchors=n_lon_anchors,
            quality_mode=quality_mode,
        )
    if source == "walker":
        return _sort_active_sats_from_walker(
            scenarioConfig,
            t0_utc=t0_utc,
            n_lat_anchors=n_lat_anchors,
            n_lon_anchors=n_lon_anchors,
            quality_mode=quality_mode,
        )
    raise ValueError(f"Unknown multisat.source={source!r}. Expected 'tle' or 'walker'.")


# -----------------------------
# Association logic (unchanged)
# -----------------------------
def _elevations_deg(user_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> np.ndarray:
    """
    Compute elevation angles for all user-sat pairs.

    user_ecef_m: (N,3)
    sat_ecef_m : (S,3)

    Returns
    -------
    elev_deg: (N,S)
    """
    # Local up approximation: radial unit vector
    up = unit(user_ecef_m)  # (N,3)

    N = user_ecef_m.shape[0]
    S = sat_ecef_m.shape[0]
    elev = np.empty((N, S), dtype=np.float32)

    # For each sat: LOS from user->sat, unit LOS, elevation = asin(dot(LOS_hat, up))
    for s in range(S):
        los = sat_ecef_m[s][None, :] - user_ecef_m          # (N,3)
        los_hat = unit(los)                                 # (N,3)
        sin_el = np.einsum("ij,ij->i", los_hat, up)         # (N,)
        sin_el = np.clip(sin_el, -1.0, 1.0)
        elev[:, s] = np.degrees(np.arcsin(sin_el)).astype(np.float32)

    return elev


def associate_users_balanced(
    user_ecef_m: np.ndarray,
    user_demand_mbps: np.ndarray,
    user_qos_w: np.ndarray,
    sat_ecef_m: np.ndarray,
    elev_mask_deg: float,
    load_mode: str = "demand",   # "count" | "demand" | "wq_demand"
    slack: float = 0.15,
    max_rounds: int = 6,
    max_total_moves: int = 200000,
    seed: int = 1,
) -> AssocResult:
    """
    Balanced association:
      1) Assign each user to its best-elevation visible satellite
      2) Rebalance overloaded sats by moving 'cheap-to-move' users

    Overcrowding model:
      - Compute loads per satellite using load_mode
      - Soft cap = (total_load / n_active_sats) * (1+slack)
      - If a sat exceeds cap, move users to alternatives until under cap or stuck.
    """
    rng = np.random.default_rng(seed)

    N = user_ecef_m.shape[0]
    S = sat_ecef_m.shape[0]

    elev = _elevations_deg(user_ecef_m, sat_ecef_m)  # (N,S)
    valid = elev >= float(elev_mask_deg)

    # Preferences (best to worst elevation)
    pref = np.argsort(-elev, axis=1)  # (N,S)
    pref_valid = np.take_along_axis(valid, pref, axis=1)

    has_valid = pref_valid.any(axis=1)
    first_pos = np.argmax(pref_valid, axis=1)  # 0 even if no valid; fix below

    assign = np.full(N, -1, dtype=np.int32)
    rows = np.arange(N, dtype=np.int32)
    assign[has_valid] = pref[rows[has_valid], first_pos[has_valid]].astype(np.int32)

    # Define per-user load weight
    if load_mode == "count":
        w = np.ones(N, dtype=np.float64)
    elif load_mode == "demand":
        w = user_demand_mbps.astype(np.float64)
    elif load_mode == "wq_demand":
        w = (user_demand_mbps.astype(np.float64) * user_qos_w.astype(np.float64))
    else:
        raise ValueError(f"Unknown load_mode={load_mode}")

    # Initial loads
    loads = np.zeros(S, dtype=np.float64)
    for s in range(S):
        idx = np.where(assign == s)[0]
        if idx.size:
            loads[s] = float(w[idx].sum())

    served_mask = assign >= 0
    total_load = float(w[served_mask].sum())
    cap = (total_load / max(S, 1)) * (1.0 + float(slack))

    # Helper: find next feasible sat for user i given current sat s_cur
    def _next_sat(i: int, s_cur: int) -> int:
        # scan ordered preferences; pick first valid != s_cur
        for k in range(S):
            s2 = int(pref[i, k])
            if s2 == s_cur:
                continue
            if valid[i, s2]:
                return s2
        return -1

    n_moves = 0

    for _round in range(max_rounds):
        overloaded = np.where(loads > cap)[0]
        if overloaded.size == 0:
            break

        # Process overloaded sats in descending overload amount
        overloaded = overloaded[np.argsort(-(loads[overloaded] - cap))]

        moved_this_round = 0

        for s in overloaded:
            if n_moves >= max_total_moves:
                break
            # Users currently assigned to this sat
            users_s = np.where(assign == s)[0]
            if users_s.size == 0:
                continue

            # For each user, compute "switch cost" to next best sat:
            # delta_elev = elev(best) - elev(next)
            # Move smallest delta first.
            alt = np.array([_next_sat(int(i), int(s)) for i in users_s], dtype=np.int32)
            ok = alt >= 0
            if not np.any(ok):
                continue

            cand_users = users_s[ok]
            cand_alt = alt[ok]

            delta = elev[cand_users, s] - elev[cand_users, cand_alt]
            order = np.argsort(delta, kind="mergesort")

            for idx_in_order in order:
                if loads[s] <= cap:
                    break
                if n_moves >= max_total_moves:
                    break

                i = int(cand_users[idx_in_order])
                s2 = int(cand_alt[idx_in_order])

                # If target also overloaded, try to find a better target among valid ones
                # Choose the valid sat with minimum load (excluding s).
                if loads[s2] > cap:
                    valid_sats = np.where(valid[i])[0]
                    valid_sats = valid_sats[valid_sats != s]
                    if valid_sats.size == 0:
                        continue
                    s2 = int(valid_sats[np.argmin(loads[valid_sats])])

                # perform move
                assign[i] = s2
                wi = float(w[i])
                loads[s] -= wi
                loads[s2] += wi
                n_moves += 1
                moved_this_round += 1

        if moved_this_round == 0:
            break

    # Build sat->user lists
    sat_user_ids: List[np.ndarray] = []
    for s in range(S):
        sat_user_ids.append(np.where(assign == s)[0].astype(np.int32))

    n_unserved = int((assign < 0).sum())

    return AssocResult(
        assign=assign,
        sat_user_ids=sat_user_ids,
        loads=loads,
        cap=float(cap),
        n_unserved=n_unserved,
        n_moves=int(n_moves),
    )