# src/satellites.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from src.coords import unit

if TYPE_CHECKING:
    from config import ScenarioConfig, BBox


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ActiveSat:
    """Active satellite at a given instant (snapshot)."""
    name: str
    ecef_m: np.ndarray   # (3,)
    elev_ref_deg: float  # DEBUG: max elevation across anchors (not a single reference site)


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
        # Interpret naive as UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _choose_snapshot_time_utc(sats: Sequence[object]) -> datetime:
    """
    Deterministic snapshot selection: midpoint of TLE epoch span (UTC).
    """
    epochs = [sat.epoch.utc_datetime() for sat in sats]
    t_min = min(epochs)
    t_max = max(epochs)
    t0 = t_min + (t_max - t_min) / 2
    if t0.tzinfo is None:
        return t0.replace(tzinfo=timezone.utc)
    return t0.astimezone(timezone.utc)


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


def _quality_from_elev_deg(elev_deg: np.ndarray, emin_deg: float, mode: str = "sin") -> np.ndarray:
    """
    Convert elevation (deg) to a nonnegative "quality" q, with q=0 below mask.

    mode:
      - "sin":    q = sin(e)
      - "sin2":   q = sin(e)^2 (penalizes low elevation more)
      - "linear": q = e (deg)
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


# -----------------------------
# New satellite selection API
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
    Load a TLE file, choose a snapshot time t0, then select active satellites using:

      (1) Multi-anchor visibility filter:
          keep satellites visible (elev >= mask) at ANY anchor over the region bbox.

      (2) Greedy marginal-gain ordering (anchors only):
          iteratively pick satellites that maximize the additional anchor quality
          not already provided by the selected set.

    Returns
    -------
    (t0_utc, active_sats)
      - t0_utc: UTC timezone-aware datetime
      - active_sats: List[ActiveSat] in greedy order, length <= scenarioConfig.multisat.n_active

    Notes
    -----
    - This fully replaces the old reference-site elevation sorting logic.
    - 'elev_ref_deg' is kept for compatibility but now stores the MAX elevation across anchors.
    """
    # Skyfield import inside to keep the rest of the project usable without it.
    from skyfield.api import load, wgs84
    from skyfield.framelib import itrs

    tle_path = scenarioConfig.multisat.tle_path
    emin_deg = float(scenarioConfig.multisat.elev_mask_deg)
    n_active = int(scenarioConfig.multisat.n_active)
    bbox = scenarioConfig.bbox

    ts = load.timescale()
    sats = load.tle_file(tle_path)
    if len(sats) == 0:
        raise RuntimeError(f"No satellites loaded from TLE file: {tle_path}")

    # Snapshot time
    if t0_utc is not None:
        t0 = t0_utc.astimezone(timezone.utc) if t0_utc.tzinfo else t0_utc.replace(tzinfo=timezone.utc)
    elif scenarioConfig.multisat.time_utc_iso is not None:
        t0 = _parse_utc_iso(scenarioConfig.multisat.time_utc_iso)
    else:
        t0 = _choose_snapshot_time_utc(sats)

    t = ts.from_datetime(t0)

    # Anchors over region
    anchors = _build_anchor_grid(bbox, n_lat=n_lat_anchors, n_lon=n_lon_anchors)
    P = len(anchors)
    N = len(sats)

    # Pre-create anchor observer objects
    obs = [wgs84.latlon(a.lat_deg, a.lon_deg, elevation_m=a.height_m) for a in anchors]

    # Elevation matrix: elev[p, s]
    elev = np.empty((P, N), dtype=np.float32)
    for si, sat in enumerate(sats):
        for pi, o in enumerate(obs):
            alt, _, _ = (sat - o).at(t).altaz()
            elev[pi, si] = float(alt.degrees)

    # (1) Multi-anchor visibility filter: visible at ANY anchor
    cand_mask = (elev >= emin_deg).any(axis=0)
    cand_idx = np.where(cand_mask)[0]
    if cand_idx.size == 0:
        return t0, []

    elev_c = elev[:, cand_idx].astype(np.float64)  # (P, Mc)

    # Anchor quality q[p, j]
    q = _quality_from_elev_deg(elev_c, emin_deg=emin_deg, mode=quality_mode)

    # (2) Greedy marginal-gain ordering
    Mc = q.shape[1]
    best = np.zeros(P, dtype=np.float64)
    chosen = np.zeros(Mc, dtype=bool)
    order_local: List[int] = []

    kmax = min(n_active, Mc)
    for _ in range(kmax):
        improvement = np.maximum(q - best[:, None], 0.0)  # (P,Mc)
        delta = improvement.sum(axis=0)                   # (Mc,)
        delta[chosen] = -1.0

        j = int(np.argmax(delta))
        if delta[j] <= 0.0:
            break  # no further improvement anywhere
        chosen[j] = True
        order_local.append(j)
        best = np.maximum(best, q[:, j])

    selected_sat_idx = cand_idx[np.array(order_local, dtype=int)]

    # Build ActiveSat list in greedy order
    active: List[ActiveSat] = []
    for si in selected_sat_idx:
        sat = sats[int(si)]
        # Satellite ECEF position at t0
        p_km = sat.at(t).frame_xyz(itrs).km  # (3,)
        ecef_m = np.array(p_km, dtype=float) * 1000.0

        # DEBUG: max elevation across anchors
        el_max = float(elev[:, int(si)].max())

        active.append(ActiveSat(name=sat.name, ecef_m=ecef_m, elev_ref_deg=el_max))

    return t0, active


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
    load_mode: str = "wq_demand",   # "count" | "demand" | "wq_demand"
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