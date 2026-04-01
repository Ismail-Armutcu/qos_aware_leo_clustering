from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from config import ScenarioConfig
from .prepare import MILPPreparation, PreparedMILPInstance
from .candidates import GridCandidateGenerator
from .compute import CandidateComputer, MILPPrecomputedData
from .model import BeamPlacementMILP, MILPSolution, MILPSolveConfig


@dataclass(frozen=True)
class MILPRunnerConfig:
    n_candidate_sats: int = 10
    grid_spacing_m: float = 5_000.0
    grid_margin_m: float = 0.0
    n_lat_anchors: int = 3
    n_lon_anchors: int = 3
    quality_mode: str = "sin"
    min_rate_mbps: float = 1e-6
    enforce_elev_mask: bool = True
    max_user_share_per_beam: Optional[float] = None
    time_limit_s: float = 300.0
    mip_gap: float = 0.0
    threads: Optional[int] = None
    log_to_console: bool = True
    objective_mode: str = "beam_only"   # beam_only | weighted_sat_beam
    satellite_weight: Optional[float] = None
    print_diagnostics: bool = True


class MILPExperimentRunner:
    def __init__(self, cfg: ScenarioConfig, run_cfg: MILPRunnerConfig | None = None) -> None:
        self.cfg = cfg
        self.run_cfg = run_cfg or MILPRunnerConfig()

    def prepare(self) -> PreparedMILPInstance:
        prep = MILPPreparation(
            self.cfg,
            n_candidate_sats=self.run_cfg.n_candidate_sats,
            n_lat_anchors=self.run_cfg.n_lat_anchors,
            n_lon_anchors=self.run_cfg.n_lon_anchors,
            quality_mode=self.run_cfg.quality_mode,
        )
        return prep.run()

    def build_grid(self, instance: PreparedMILPInstance):
        from src.usergen import build_users_for_sat

        all_ids = np.arange(instance.n_users, dtype=int)
        users_sat0 = build_users_for_sat(instance.users_raw, all_ids, instance.sat_pool[0].ecef_m)
        gen = GridCandidateGenerator(
            lat0_deg=float(self.cfg.lat0_deg),
            lon0_deg=float(self.cfg.lon0_deg),
            spacing_m=self.run_cfg.grid_spacing_m,
            margin_m=self.run_cfg.grid_margin_m,
        )
        return gen.build(users_sat0.xy_m)

    def precompute(self, instance: PreparedMILPInstance, grid) -> MILPPrecomputedData:
        comp = CandidateComputer(
            self.cfg,
            min_rate_mbps=self.run_cfg.min_rate_mbps,
            enforce_elev_mask=self.run_cfg.enforce_elev_mask,
            max_user_share_per_beam=self.run_cfg.max_user_share_per_beam,
        )
        return comp.build(instance, grid)

    def solve(self, data: MILPPrecomputedData) -> MILPSolution:
        solve_cfg = MILPSolveConfig(
            time_limit_s=self.run_cfg.time_limit_s,
            mip_gap=self.run_cfg.mip_gap,
            threads=self.run_cfg.threads,
            log_to_console=self.run_cfg.log_to_console,
            objective_mode=self.run_cfg.objective_mode,
            satellite_weight=self.run_cfg.satellite_weight,
        )
        model = BeamPlacementMILP(self.cfg, data, solve_cfg=solve_cfg)
        return model.solve()

    def _diagnostics(self, data: MILPPrecomputedData) -> dict[str, float]:
        n_users = data.instance.n_users
        n_cands = len(data.candidates)
        user_counts = np.array([len(data.user_to_bids.get(u, [])) for u in range(n_users)], dtype=float)
        beam_counts = np.array([len(c.user_ids) for c in data.candidates], dtype=float) if data.candidates else np.zeros(0, dtype=float)
        return {
            "n_users": float(n_users),
            "n_candidates": float(n_cands),
            "avg_feasible_beams_per_user": float(user_counts.mean()) if user_counts.size else 0.0,
            "median_feasible_beams_per_user": float(np.median(user_counts)) if user_counts.size else 0.0,
            "min_feasible_beams_per_user": float(user_counts.min()) if user_counts.size else 0.0,
            "avg_users_per_candidate_beam": float(beam_counts.mean()) if beam_counts.size else 0.0,
            "median_users_per_candidate_beam": float(np.median(beam_counts)) if beam_counts.size else 0.0,
            "max_users_per_candidate_beam": float(beam_counts.max()) if beam_counts.size else 0.0,
        }

    def run(self) -> dict[str, Any]:
        instance = self.prepare()
        grid = self.build_grid(instance)
        data = self.precompute(instance, grid)
        diagnostics = self._diagnostics(data)
        if self.run_cfg.print_diagnostics:
            print("=== MILP CANDIDATE DIAGNOSTICS ===")
            for k, v in diagnostics.items():
                print(f"{k}: {v}")
        sol = self.solve(data)
        return {
            "instance": instance,
            "grid": grid,
            "data": data,
            "diagnostics": diagnostics,
            "solution": sol,
        }


def run_milp_experiment(cfg: ScenarioConfig, run_cfg: MILPRunnerConfig | None = None) -> dict[str, Any]:
    return MILPExperimentRunner(cfg, run_cfg=run_cfg).run()
