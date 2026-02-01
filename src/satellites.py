# src/satellites.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.coords import unit


@dataclass(frozen=True)
class ActiveSat:
    """Active satellite at a given instant (snapshot)."""
    name: str
    ecef_m: np.ndarray   # (3,)
    elev_ref_deg: float  # elevation at reference site (for debug)


@dataclass(frozen=True)
class AssocResult:
    """Result of balanced user->satellite association."""
    assign: np.ndarray                 # (N,), sat index or -1
    sat_user_ids: List[np.ndarray]     # length S, each array of user indices
    loads: np.ndarray                  # (S,), load per sat (count or demand or wq-demand)
    cap: float                         # soft cap used for balancing
    n_unserved: int
    n_moves: int


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


def select_top_n_active_sats(
    tle_path: str,
    n_active: int,
    elev_mask_deg: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_height_m: float = 0.0,
    t0_utc: datetime | None = None,
) -> tuple[datetime, List[ActiveSat]]:
    """
    Load a TLE file, pick a valid snapshot time (t0), then select Top-N sats by
    elevation at a reference site, and return their ECEF positions at t0.

    - If t0_utc is None, choose the midpoint of TLE epoch range.
    - Keep only sats with elevation >= elev_mask_deg at the reference site.
    """
    # Skyfield import inside to keep the rest of the project usable without it.
    from skyfield.api import load, wgs84
    from skyfield.framelib import itrs

    ts = load.timescale()
    sats = load.tle_file(tle_path)

    if len(sats) == 0:
        raise RuntimeError(f"No satellites loaded from TLE file: {tle_path}")

    # Choose t0 inside epoch span if not provided
    if t0_utc is None:
        epochs = [sat.epoch.utc_datetime() for sat in sats]
        t_min = min(epochs)
        t_max = max(epochs)
        t0_utc = t_min + (t_max - t_min) / 2
        if t0_utc.tzinfo is None:
            t0_utc = t0_utc.replace(tzinfo=timezone.utc)
        else:
            t0_utc = t0_utc.astimezone(timezone.utc)

    t = ts.from_datetime(t0_utc)

    ref = wgs84.latlon(ref_lat_deg, ref_lon_deg, elevation_m=ref_height_m)

    visible: list[tuple[float, object]] = []
    for sat in sats:
        alt, az, dist = (sat - ref).at(t).altaz()
        el = float(alt.degrees)
        if el >= elev_mask_deg:
            visible.append((el, sat))

    visible.sort(key=lambda x: x[0], reverse=True)
    visible = visible[:n_active]

    active: List[ActiveSat] = []
    for el, sat in visible:
        # Satellite ECEF position at t0
        p_km = sat.at(t).frame_xyz(itrs).km  # (3,)
        ecef_m = np.array(p_km, dtype=float) * 1000.0
        active.append(ActiveSat(name=sat.name, ecef_m=ecef_m, elev_ref_deg=el))

    return t0_utc, active


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
