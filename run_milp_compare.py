
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import ScenarioConfig
from src.pipeline import run_scenario
from src.milp.runner import run_milp_experiment, MILPRunnerConfig

try:
    from src.helper import flatten_run_record  # preferred: project-native flattening
except Exception:  # pragma: no cover
    flatten_run_record = None


class SimpleProfiler:
    """Lightweight wall-time profiler for the comparison script."""

    def __init__(self) -> None:
        self._tic: dict[str, float] = {}
        self.timings: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._tic[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        t0 = self._tic.pop(name)
        dt = time.perf_counter() - t0
        self.timings[name] = self.timings.get(name, 0.0) + dt
        return dt


@dataclass(frozen=True)
class CompareSweepConfig:
    output_csv: str = "milp_compare.csv"
    region_mode: str = "turkey"
    n_users_list: tuple[int, ...] = (50, 75, 100)
    seeds: tuple[int, ...] = (1, 2, 3)

    # Reduced MILP instances
    milp_n_candidate_sats: int = 6
    milp_grid_spacing_m: float = 15_000.0
    milp_grid_margin_m: float = 0.0
    milp_time_limit_s: float = 300.0
    milp_mip_gap: float = 0.0
    milp_threads: int | None = None
    milp_log_to_console: bool = True
    milp_objective_mode: str = "beam_only"  # beam_only | weighted_sat_beam
    milp_satellite_weight: float | None = None
    milp_print_diagnostics: bool = True

    # Heuristic settings
    heuristic_enable_lb: bool = True
    heuristic_disable_baselines: bool = True

    # User generation knobs
    hotspot_enabled: bool = True
    n_hotspots: int = 5
    hotspot_sigma_m_min: float = 5_000.0
    hotspot_sigma_m_max: float = 30_000.0
    noise_frac: float = 0.0


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _recursive_flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
            out.update(_recursive_flatten(v, key))
    else:
        out[prefix] = obj
    return out


def _flatten_record(rec: dict[str, Any]) -> dict[str, Any]:
    if flatten_run_record is not None:
        try:
            return dict(flatten_run_record(rec))
        except Exception:
            pass
    return _recursive_flatten(rec)


def _first_present(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _find_prefixed_metric(
    flat: dict[str, Any],
    prefixes: Sequence[str],
    exact_suffixes: Sequence[str],
    fuzzy_contains: Sequence[str] = (),
) -> tuple[Any, str | None]:
    """
    Try exact keys first: {prefix}_{suffix}
    Then try any key starting with prefix and containing all fuzzy tokens.
    Returns (value, matched_key).
    """
    for prefix in prefixes:
        for suffix in exact_suffixes:
            key = f"{prefix}_{suffix}"
            if key in flat and flat[key] is not None:
                return flat[key], key

    fuzzy_tokens = tuple(tok.lower() for tok in fuzzy_contains if tok)
    for prefix in prefixes:
        pfx = f"{prefix}_"
        for key, value in flat.items():
            if not key.startswith(pfx):
                continue
            kl = key.lower()
            if all(tok in kl for tok in fuzzy_tokens):
                return value, key

    return None, None


def _build_compare_cfg(base: ScenarioConfig, sweep: CompareSweepConfig, *, n_users: int, seed: int) -> ScenarioConfig:
    return replace(
        base,
        region_mode=sweep.region_mode,
        run=replace(
            base.run,
            seed=seed,
            n_users=n_users,
            enable_plots=False,
            verbose=False,
            enable_fastbp_baselines=(not sweep.heuristic_disable_baselines),
        ),
        usergen=replace(
            base.usergen,
            enabled=sweep.hotspot_enabled,
            n_hotspots=sweep.n_hotspots,
            hotspot_sigma_m_min=sweep.hotspot_sigma_m_min,
            hotspot_sigma_m_max=sweep.hotspot_sigma_m_max,
            noise_frac=sweep.noise_frac,
        ),
        lb_refine=replace(base.lb_refine, enabled=sweep.heuristic_enable_lb),
        payload=replace(base.payload, max_prefix=sweep.milp_n_candidate_sats),
    )


def _milp_runner_cfg(sweep: CompareSweepConfig) -> MILPRunnerConfig:
    return MILPRunnerConfig(
        n_candidate_sats=sweep.milp_n_candidate_sats,
        grid_spacing_m=sweep.milp_grid_spacing_m,
        grid_margin_m=sweep.milp_grid_margin_m,
        n_lat_anchors=3,
        n_lon_anchors=3,
        quality_mode="sin",
        min_rate_mbps=1e-6,
        enforce_elev_mask=True,
        max_user_share_per_beam=None,
        time_limit_s=sweep.milp_time_limit_s,
        mip_gap=sweep.milp_mip_gap,
        threads=sweep.milp_threads,
        log_to_console=sweep.milp_log_to_console,
        objective_mode=sweep.milp_objective_mode,
        satellite_weight=sweep.milp_satellite_weight,
        print_diagnostics=sweep.milp_print_diagnostics,
    )


def compare_one(base: ScenarioConfig, sweep: CompareSweepConfig, *, n_users: int, seed: int) -> dict[str, Any]:
    cfg = _build_compare_cfg(base, sweep, n_users=n_users, seed=seed)
    prof = SimpleProfiler()

    row: dict[str, Any] = {
        "seed": seed,
        "n_users": n_users,
        "region_mode": cfg.region_mode,
        "milp_n_candidate_sats": sweep.milp_n_candidate_sats,
        "milp_grid_spacing_m": sweep.milp_grid_spacing_m,
        "milp_objective_mode": sweep.milp_objective_mode,
        "milp_time_limit_s": sweep.milp_time_limit_s,
        "cfg_theta_3db_deg": getattr(cfg.beam, "theta_3db_deg", None),
        "cfg_W_slots": getattr(cfg.payload, "W_slots", None),
        "cfg_J_lanes": getattr(cfg.payload, "J_lanes", None),
        "cfg_Ks_max": getattr(cfg.payload, "Ks_max", None),
    }

    # Heuristic side
    prof.start("heuristic_wall_s")
    rec = run_scenario(cfg)
    prof.stop("heuristic_wall_s")
    flat = _flatten_record(rec)

    prefixes = ("main_ref_lb", "main_ref", "main")
    row["heur_payload_feasible"] = _first_present(flat, ("payload_feasible", "heur_payload_feasible"))
    row["heur_used_sats"] = _first_present(flat, ("payload_best_m", "best_m"))
    row["heur_wall_s"] = prof.timings["heuristic_wall_s"]

    # Robust metric discovery
    heur_K, heur_K_key = _find_prefixed_metric(flat, prefixes, ("K",), ("_k",))
    heur_U_max, heur_U_key = _find_prefixed_metric(flat, prefixes, ("U_max",), ("u_max",))
    heur_edge, heur_edge_key = _find_prefixed_metric(
        flat,
        prefixes,
        ("ent_edge_pct", "enterprise_edge_pct"),
        ("edge", "pct"),
    )
    heur_risk, heur_risk_key = _find_prefixed_metric(
        flat,
        prefixes,
        ("ent_risk", "risk", "enterprise_risk"),
        ("risk",),
    )
    heur_r_mean, heur_r_mean_key = _find_prefixed_metric(
        flat,
        prefixes,
        ("beam_r_mean_km", "beam_radius_mean_km", "r_mean_km"),
        ("beam", "mean"),
    )

    row["heur_K"] = heur_K
    row["heur_U_max"] = heur_U_max
    row["heur_ent_edge_pct"] = heur_edge
    row["heur_ent_risk"] = heur_risk
    row["heur_beam_r_mean"] = heur_r_mean
    row["heur_K_key"] = heur_K_key
    row["heur_U_max_key"] = heur_U_key
    row["heur_ent_edge_pct_key"] = heur_edge_key
    row["heur_ent_risk_key"] = heur_risk_key
    row["heur_beam_r_mean_key"] = heur_r_mean_key

    # Existing internal timings from pipeline
    for k, v in flat.items():
        if k.startswith("time_"):
            row[f"heur_{k}"] = v

    # MILP side
    milp_cfg = _milp_runner_cfg(sweep)
    prof.start("milp_wall_s")
    milp_out = run_milp_experiment(cfg, run_cfg=milp_cfg)
    prof.stop("milp_wall_s")

    sol = milp_out["solution"]
    diag = milp_out.get("diagnostics", {})
    data = milp_out.get("data", None)
    grid = milp_out.get("grid", None)

    row["milp_feasible"] = bool(sol.feasible)
    row["milp_status"] = sol.status_name
    row["milp_objective"] = sol.objective_value
    row["milp_best_bound"] = sol.best_bound
    row["milp_mip_gap"] = sol.mip_gap
    row["milp_solve_time_s"] = sol.solve_time_s
    row["milp_wall_s"] = prof.timings["milp_wall_s"]
    row["milp_used_sats"] = sol.n_used_sats
    row["milp_used_sat_indices"] = " ".join(map(str, sol.used_sat_indices))
    row["milp_K"] = sol.K_total
    row["milp_n_active_beams"] = len(sol.active_beam_ids)
    row["milp_active_beam_ids"] = " ".join(map(str, sol.active_beam_ids))
    row["milp_n_grid_centers"] = len(grid) if grid is not None else None
    row["milp_n_candidates"] = len(data.candidates) if data is not None else None

    for k, v in diag.items():
        row[f"milp_diag_{k}"] = v

    if getattr(sol, "sat_time_load", None):
        sat_loads = list(sol.sat_time_load.values())
        row["milp_sat_time_load_max"] = max(sat_loads) if sat_loads else None
        row["milp_sat_time_load_mean"] = sum(sat_loads) / len(sat_loads) if sat_loads else None
    if getattr(sol, "beam_time_load", None):
        beam_loads = list(sol.beam_time_load.values())
        row["milp_beam_time_load_max"] = max(beam_loads) if beam_loads else None
        row["milp_beam_time_load_mean"] = sum(beam_loads) / len(beam_loads) if beam_loads else None

    # Comparison deltas
    heur_K_f = _safe_float(row.get("heur_K"))
    milp_K_f = _safe_float(row.get("milp_K"))
    if heur_K_f is not None and milp_K_f is not None and milp_K_f > 0.0:
        row["gap_K_abs"] = heur_K_f - milp_K_f
        row["gap_K_pct"] = 100.0 * (heur_K_f - milp_K_f) / milp_K_f

    heur_used_sats_f = _safe_float(row.get("heur_used_sats"))
    milp_used_sats_f = _safe_float(row.get("milp_used_sats"))
    if heur_used_sats_f is not None and milp_used_sats_f is not None:
        row["gap_used_sats_abs"] = heur_used_sats_f - milp_used_sats_f

    return row


def run_compare_sweep(base: ScenarioConfig, sweep: CompareSweepConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(sweep.n_users_list) * len(sweep.seeds)
    done = 0
    for n_users in sweep.n_users_list:
        for seed in sweep.seeds:
            done += 1
            print(f"[{done}/{total}] Comparing MILP vs heuristic for n_users={n_users}, seed={seed} ...")
            rows.append(compare_one(base, sweep, n_users=n_users, seed=seed))
    return rows


def write_rows_to_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    base = ScenarioConfig()
    sweep = CompareSweepConfig()
    rows = run_compare_sweep(base, sweep)
    out_path = write_rows_to_csv(rows, sweep.output_csv)
    print(f"Saved comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
