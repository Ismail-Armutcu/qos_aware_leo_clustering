from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import ScenarioConfig
from src.coords import elevation_deg, unit
from src.phy import fspl_db, gain_db_gaussian, snr_lin, shannon_rate_mbps
from src.usergen import build_users_for_sat

from .candidates import GridCenter
from .prepare import PreparedMILPInstance


@dataclass(frozen=True)
class BeamCandidate:
    bid: int
    sat_idx: int
    grid_idx: int
    center_xy_m: np.ndarray
    center_ecef_m: np.ndarray
    d_center_m: float
    radius_m: float
    user_ids: np.ndarray
    a_coeff: Dict[int, float]


@dataclass(frozen=True)
class MILPPrecomputedData:
    instance: PreparedMILPInstance
    grid: list[GridCenter]
    candidates: list[BeamCandidate]
    user_to_bids: Dict[int, List[int]]
    sat_to_bids: Dict[int, List[int]]


class CandidateComputer:
    """
    Precompute candidate-beam coefficients for the MILP.

    Fixed-angle beam model:
        R = d_center * tan(theta_3db)

    Payload load abstraction:
        a_{u,b} = demand_u / rate_{u,b}
    """

    def __init__(
        self,
        cfg: ScenarioConfig,
        *,
        min_rate_mbps: float = 1e-6,
        enforce_elev_mask: bool = True,
        max_user_share_per_beam: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.min_rate_mbps = float(min_rate_mbps)
        self.enforce_elev_mask = bool(enforce_elev_mask)
        self.max_user_share_per_beam = (
            float(max_user_share_per_beam)
            if max_user_share_per_beam is not None
            else float(cfg.payload.W_slots)
        )

        theta = np.deg2rad(float(cfg.beam.theta_3db_deg))
        if theta <= 0.0:
            raise ValueError("cfg.beam.theta_3db_deg must be positive for MILP candidate generation.")
        self.theta_3db_rad = float(theta)

    def _build_all_users_for_sat(self, instance: PreparedMILPInstance, sat_idx: int):
        all_user_ids = np.arange(instance.n_users, dtype=int)
        sat_ecef = instance.sat_pool[sat_idx].ecef_m
        return build_users_for_sat(instance.users_raw, all_user_ids, sat_ecef)

    def _compute_candidate_for_sat_grid(self, users_sat, sat_idx: int, grid_center: GridCenter) -> Optional[BeamCandidate]:
        sat_ecef = np.asarray(users_sat.sat_ecef_m, dtype=float)
        center_ecef = np.asarray(grid_center.ecef_m, dtype=float)
        center_xy = np.asarray(grid_center.xy_m, dtype=float)

        v_c = center_ecef - sat_ecef
        d_center = float(np.linalg.norm(v_c))
        if not np.isfinite(d_center) or d_center <= 1e-9:
            return None

        radius_m = float(d_center * np.tan(self.theta_3db_rad))
        if not np.isfinite(radius_m) or radius_m <= 0.0:
            return None

        dist_xy = np.linalg.norm(users_sat.xy_m - center_xy[None, :], axis=1)
        cover_mask = dist_xy <= (radius_m + 1e-9)
        if self.enforce_elev_mask:
            elev = elevation_deg(users_sat.ecef_m, sat_ecef)
            cover_mask &= elev >= float(self.cfg.multisat.elev_mask_deg)

        local_ids = np.where(cover_mask)[0]
        if local_ids.size == 0:
            return None

        u_c = np.asarray(unit(v_c), dtype=float).reshape(3)
        u_i = np.asarray(users_sat.u_sat2user[local_ids], dtype=float)
        cosang = np.clip(u_i @ u_c, -1.0, 1.0)
        theta = np.arccos(cosang)

        ranges = np.asarray(users_sat.range_m[local_ids], dtype=float)
        fspl = fspl_db(ranges, float(self.cfg.phy.carrier_freq_hz))
        g_db = gain_db_gaussian(theta, self.theta_3db_rad)
        snr = snr_lin(
            float(self.cfg.phy.eirp_dbw),
            g_db,
            fspl,
            float(self.cfg.phy.loss_misc_db),
            float(self.cfg.phy.noise_psd_dbw_hz),
            float(self.cfg.phy.bandwidth_hz),
        )
        rate = shannon_rate_mbps(snr, float(self.cfg.phy.bandwidth_hz), float(self.cfg.phy.eta))

        a_coeff: Dict[int, float] = {}
        user_ids: list[int] = []
        for arr_pos, local_idx in enumerate(local_ids.tolist()):
            r = float(rate[arr_pos])
            if not np.isfinite(r) or r <= self.min_rate_mbps:
                continue

            uid = int(local_idx)  # all users were built in global order
            share = float(users_sat.demand_mbps[local_idx] / r)
            if not np.isfinite(share) or share > self.max_user_share_per_beam:
                continue

            a_coeff[uid] = share
            user_ids.append(uid)

        if not a_coeff:
            return None

        return BeamCandidate(
            bid=-1,
            sat_idx=int(sat_idx),
            grid_idx=int(grid_center.gid),
            center_xy_m=center_xy,
            center_ecef_m=center_ecef,
            d_center_m=float(d_center),
            radius_m=float(radius_m),
            user_ids=np.array(sorted(user_ids), dtype=int),
            a_coeff=a_coeff,
        )

    def build(self, instance: PreparedMILPInstance, grid: list[GridCenter]) -> MILPPrecomputedData:
        candidates: list[BeamCandidate] = []
        user_to_bids: Dict[int, List[int]] = {u: [] for u in range(instance.n_users)}
        sat_to_bids: Dict[int, List[int]] = {s: [] for s in range(len(instance.sat_pool))}

        bid = 0
        for s_idx, _sat in enumerate(instance.sat_pool):
            users_sat = self._build_all_users_for_sat(instance, s_idx)
            for gc in grid:
                cand = self._compute_candidate_for_sat_grid(users_sat, s_idx, gc)
                if cand is None:
                    continue

                cand = BeamCandidate(
                    bid=bid,
                    sat_idx=cand.sat_idx,
                    grid_idx=cand.grid_idx,
                    center_xy_m=cand.center_xy_m,
                    center_ecef_m=cand.center_ecef_m,
                    d_center_m=cand.d_center_m,
                    radius_m=cand.radius_m,
                    user_ids=cand.user_ids,
                    a_coeff=cand.a_coeff,
                )
                candidates.append(cand)
                sat_to_bids[s_idx].append(bid)
                for uid in cand.user_ids.tolist():
                    user_to_bids[uid].append(bid)
                bid += 1

        return MILPPrecomputedData(
            instance=instance,
            grid=grid,
            candidates=candidates,
            user_to_bids=user_to_bids,
            sat_to_bids=sat_to_bids,
        )
