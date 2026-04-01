from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from config import ScenarioConfig
from src.satellites import ActiveSat, sort_active_sats
from src.usergen import generate_users, pack_users_raw


@dataclass(frozen=True)
class PreparedMILPInstance:
    cfg: ScenarioConfig
    users_raw: Any
    n_users: int
    t0_utc: datetime
    sat_pool: list[ActiveSat]


class MILPPreparation:
    """Prepare one snapshot using the same user generation and satellite ordering logic as the main pipeline."""

    def __init__(
        self,
        cfg: ScenarioConfig,
        *,
        n_candidate_sats: int = 10,
        t0_utc: Optional[datetime] = None,
        n_lat_anchors: int = 3,
        n_lon_anchors: int = 3,
        quality_mode: str = "sin",
    ) -> None:
        self.cfg = cfg
        self.n_candidate_sats = int(n_candidate_sats)
        self.t0_utc = t0_utc
        self.n_lat_anchors = int(n_lat_anchors)
        self.n_lon_anchors = int(n_lon_anchors)
        self.quality_mode = str(quality_mode)

    def run(self) -> PreparedMILPInstance:
        user_list = generate_users(self.cfg)
        users_raw = pack_users_raw(user_list)

        t0_utc, active_sats = sort_active_sats(
            self.cfg,
            t0_utc=self.t0_utc,
            n_lat_anchors=self.n_lat_anchors,
            n_lon_anchors=self.n_lon_anchors,
            quality_mode=self.quality_mode,
        )
        if not active_sats:
            raise RuntimeError("No visible candidate satellites were returned by sort_active_sats().")

        sat_pool = list(active_sats[: self.n_candidate_sats])
        if not sat_pool:
            raise RuntimeError("Candidate satellite pool is empty. Increase n_candidate_sats or inspect elevation mask / TLE.")

        return PreparedMILPInstance(
            cfg=self.cfg,
            users_raw=users_raw,
            n_users=len(user_list),
            t0_utc=t0_utc,
            sat_pool=sat_pool,
        )


def prepare_snapshot(
    cfg: ScenarioConfig,
    *,
    n_candidate_sats: int = 10,
    t0_utc: Optional[datetime] = None,
    n_lat_anchors: int = 3,
    n_lon_anchors: int = 3,
    quality_mode: str = "sin",
) -> PreparedMILPInstance:
    return MILPPreparation(
        cfg,
        n_candidate_sats=n_candidate_sats,
        t0_utc=t0_utc,
        n_lat_anchors=n_lat_anchors,
        n_lon_anchors=n_lon_anchors,
        quality_mode=quality_mode,
    ).run()
