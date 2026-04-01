from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import gurobipy as gp
from gurobipy import GRB

from config import ScenarioConfig
from .compute import MILPPrecomputedData


@dataclass(frozen=True)
class MILPSolveConfig:
    time_limit_s: float = 300.0
    mip_gap: float = 0.0
    threads: Optional[int] = None
    log_to_console: bool = True
    objective_mode: str = "beam_only"   # beam_only | weighted_sat_beam
    satellite_weight: Optional[float] = None
    mip_focus: int = 1
    presolve: int = -1


@dataclass(frozen=True)
class MILPSolution:
    feasible: bool
    status: int
    status_name: str
    objective_value: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    solve_time_s: float
    used_sat_indices: list[int]
    active_beam_ids: list[int]
    assignment: dict[int, int]                  # user -> beam id
    K_total: int
    n_used_sats: int
    sat_beam_count: dict[int, int]
    sat_time_load: dict[int, float]
    beam_time_load: dict[int, float]


class BeamPlacementMILP:
    def __init__(self, cfg: ScenarioConfig, data: MILPPrecomputedData, solve_cfg: MILPSolveConfig | None = None) -> None:
        self.cfg = cfg
        self.data = data
        self.solve_cfg = solve_cfg or MILPSolveConfig()

    def _status_name(self, status: int) -> str:
        mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }
        return mapping.get(status, f"STATUS_{status}")

    def _set_objective(self, model: gp.Model, x: dict[int, gp.Var], z: dict[int, gp.Var]) -> None:
        mode = str(self.solve_cfg.objective_mode).lower().strip()

        if mode == "beam_only":
            model.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)
            return

        if mode == "weighted_sat_beam":
            alpha = (
                float(self.solve_cfg.satellite_weight)
                if self.solve_cfg.satellite_weight is not None
                else float(len(self.data.candidates) + 1)
            )
            model.setObjective(
                alpha * gp.quicksum(z.values()) + gp.quicksum(x.values()),
                GRB.MINIMIZE,
            )
            return

        raise ValueError(
            f"Unknown objective_mode={self.solve_cfg.objective_mode!r}. "
            f"Use 'beam_only' or 'weighted_sat_beam'."
        )

    def solve(self) -> MILPSolution:
        data = self.data
        n_sats = len(data.instance.sat_pool)
        model = gp.Model("leo_grid_beam_milp")
        model.Params.TimeLimit = float(self.solve_cfg.time_limit_s)
        model.Params.MIPGap = float(self.solve_cfg.mip_gap)
        model.Params.OutputFlag = 1 if self.solve_cfg.log_to_console else 0
        model.Params.MIPFocus = int(self.solve_cfg.mip_focus)
        model.Params.Presolve = int(self.solve_cfg.presolve)
        if self.solve_cfg.threads is not None:
            model.Params.Threads = int(self.solve_cfg.threads)

        candidates = data.candidates
        if not candidates:
            raise RuntimeError(
                "No candidate beams generated. Increase satellite pool or relax grid spacing / coverage assumptions."
            )

        x = {c.bid: model.addVar(vtype=GRB.BINARY, name=f"x_{c.bid}") for c in candidates}
        z = {s: model.addVar(vtype=GRB.BINARY, name=f"z_{s}") for s in range(n_sats)}
        y: dict[tuple[int, int], gp.Var] = {}
        for c in candidates:
            for uid in c.user_ids.tolist():
                y[(int(uid), c.bid)] = model.addVar(vtype=GRB.BINARY, name=f"y_{uid}_{c.bid}")
        model.update()

        self._set_objective(model, x, z)

        # Each user must be assigned exactly once.
        for uid, bids in data.user_to_bids.items():
            if not bids:
                raise RuntimeError(
                    f"User {uid} has no feasible candidate beam. Reduce grid spacing, enlarge satellite pool, or relax assumptions."
                )
            model.addConstr(gp.quicksum(y[(uid, b)] for b in bids) == 1, name=f"assign_{uid}")

        # Activation consistency
        for (uid, bid), var in y.items():
            model.addConstr(var <= x[bid], name=f"yx_{uid}_{bid}")

        # Satellite activation consistency
        for c in candidates:
            model.addConstr(x[c.bid] <= z[c.sat_idx], name=f"xz_{c.bid}")

        W = float(self.cfg.payload.W_slots)
        J = float(self.cfg.payload.J_lanes)
        Ks_max = int(self.cfg.payload.Ks_max)

        # Beam-level load
        for c in candidates:
            expr = gp.quicksum(float(c.a_coeff[int(uid)]) * y[(int(uid), c.bid)] for uid in c.user_ids.tolist())
            model.addConstr(expr <= W * x[c.bid], name=f"beamcap_{c.bid}")

        # Satellite beam count and total time budget
        bid_to_cand = {c.bid: c for c in candidates}
        for s in range(n_sats):
            bids = data.sat_to_bids.get(s, [])
            if not bids:
                model.addConstr(z[s] == 0, name=f"z_zero_{s}")
                continue
            model.addConstr(gp.quicksum(x[b] for b in bids) <= Ks_max * z[s], name=f"ks_{s}")
            expr = gp.LinExpr()
            for b in bids:
                cand = bid_to_cand[b]
                for uid in cand.user_ids.tolist():
                    expr += float(cand.a_coeff[int(uid)]) * y[(int(uid), b)]
            model.addConstr(expr <= (J * W) * z[s], name=f"time_{s}")

        tic = time.perf_counter()
        model.optimize()
        toc = time.perf_counter()

        status = int(model.Status)
        status_name = self._status_name(status)
        if status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
            return MILPSolution(
                feasible=False,
                status=status,
                status_name=status_name,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                solve_time_s=float(toc - tic),
                used_sat_indices=[],
                active_beam_ids=[],
                assignment={},
                K_total=0,
                n_used_sats=0,
                sat_beam_count={},
                sat_time_load={},
                beam_time_load={},
            )

        used_sat_indices = [s for s, var in z.items() if var.X > 0.5]
        active_beam_ids = [c.bid for c in candidates if x[c.bid].X > 0.5]
        assignment: dict[int, int] = {}
        for (uid, bid), var in y.items():
            if var.X > 0.5:
                assignment[int(uid)] = int(bid)

        sat_beam_count = {s: 0 for s in used_sat_indices}
        sat_time_load = {s: 0.0 for s in used_sat_indices}
        beam_time_load: dict[int, float] = {}
        for bid in active_beam_ids:
            cand = bid_to_cand[bid]
            sat_beam_count[cand.sat_idx] = sat_beam_count.get(cand.sat_idx, 0) + 1
            load = 0.0
            for uid in cand.user_ids.tolist():
                if assignment.get(int(uid), None) == bid:
                    load += float(cand.a_coeff[int(uid)])
            beam_time_load[bid] = float(load)
            sat_time_load[cand.sat_idx] = sat_time_load.get(cand.sat_idx, 0.0) + float(load)

        obj = float(model.ObjVal) if model.SolCount > 0 else None
        best_bound = float(model.ObjBound) if model.SolCount > 0 else None
        gap = float(model.MIPGap) if model.SolCount > 0 else None
        feasible = model.SolCount > 0
        return MILPSolution(
            feasible=bool(feasible),
            status=status,
            status_name=status_name,
            objective_value=obj,
            best_bound=best_bound,
            mip_gap=gap,
            solve_time_s=float(toc - tic),
            used_sat_indices=used_sat_indices,
            active_beam_ids=active_beam_ids,
            assignment=assignment,
            K_total=len(active_beam_ids),
            n_used_sats=len(used_sat_indices),
            sat_beam_count=sat_beam_count,
            sat_time_load=sat_time_load,
            beam_time_load=beam_time_load,
        )
