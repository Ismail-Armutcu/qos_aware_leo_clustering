from __future__ import annotations

import csv
import math
import os
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import ScenarioConfig
from src.pipeline import run_scenario
from src.milp.runner import MILPRunnerConfig, run_milp_experiment

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
    aggregate_csv: str = "milp_compare_aggregate.csv"
    tables_txt: str = "plots/milp/milp_compare_tables.txt"
    output_dir: str = "plots/milp"

    # Outer comparison parallelism. Use processes for real CPU-bound speedup.
    parallel_enabled: bool = True
    max_workers: int = max(1, (os.cpu_count() or 4) - 1)

    region_mode: str = "turkey"
    n_users_list: tuple[int, ...] = (50, 75, 100, 125, 150, 175, 200, 500, 1000, 2000, 5000)
    seeds: tuple[int, ...] = (1, 2, 3)

    # Reduced MILP instances / fixed pool
    milp_n_candidate_sats: int = 150
    milp_grid_spacing_m: float = 15_000.0
    milp_grid_margin_m: float = 0.0
    milp_time_limit_s: float = 300.0
    milp_mip_gap: float = 0.0
    milp_threads: int | None = None
    milp_log_to_console: bool = True
    milp_objective_mode: str = "weighted_sat_beam"  # beam_only | weighted_sat_beam
    milp_satellite_weight: float | None = None
    milp_print_diagnostics: bool = True

    # Small MILP grid-spacing sensitivity (MILP-only)
    grid_sensitivity_enabled: bool = True
    grid_sensitivity_n_users_list: tuple[int, ...] = (100, 150)
    grid_sensitivity_grid_spacing_m_list: tuple[float, ...] = (10_000.0, 15_000.0, 20_000.0)
    grid_sensitivity_csv: str = "milp_grid_sensitivity.csv"
    grid_sensitivity_aggregate_csv: str = "milp_grid_sensitivity_aggregate.csv"
    grid_sensitivity_tables_txt: str = "plots/milp/milp_grid_sensitivity_tables.txt"
    grid_sensitivity_output_dir: str = "plots/milp/grid_sensitivity"

    # Heuristic settings
    heuristic_enable_lb: bool = True
    heuristic_disable_baselines: bool = True

    # User generation knobs
    hotspot_enabled: bool = True
    n_hotspots: int = 5
    hotspot_sigma_m_min: float = 5_000.0
    hotspot_sigma_m_max: float = 30_000.0
    noise_frac: float = 0.0

    # Heuristic metric source preference
    heuristic_prefixes: tuple[str, ...] = ("main_ref_lb", "ab_A3", "main_ref", "main")


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


def _milp_runner_cfg(sweep: CompareSweepConfig, *, grid_spacing_m: float | None = None) -> MILPRunnerConfig:
    milp_threads = sweep.milp_threads
    # When the outer compare runs in parallel, keep each MILP solve single-threaded unless
    # the user explicitly set milp_threads, to avoid severe oversubscription.
    if sweep.parallel_enabled and sweep.max_workers > 1 and milp_threads is None:
        milp_threads = 1

    return MILPRunnerConfig(
        n_candidate_sats=sweep.milp_n_candidate_sats,
        grid_spacing_m=(sweep.milp_grid_spacing_m if grid_spacing_m is None else float(grid_spacing_m)),
        grid_margin_m=sweep.milp_grid_margin_m,
        n_lat_anchors=3,
        n_lon_anchors=3,
        quality_mode="sin",
        min_rate_mbps=1e-6,
        enforce_elev_mask=True,
        max_user_share_per_beam=None,
        time_limit_s=sweep.milp_time_limit_s,
        mip_gap=sweep.milp_mip_gap,
        threads=milp_threads,
        log_to_console=sweep.milp_log_to_console,
        objective_mode=sweep.milp_objective_mode,
        satellite_weight=sweep.milp_satellite_weight,
        print_diagnostics=sweep.milp_print_diagnostics,
    )


def _heuristic_metrics_from_flat(flat: dict[str, Any], prefixes: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out["heur_payload_feasible"] = _first_present(
        flat,
        (
            "payload_feasible",
            "main_ref_lb_payload_feasible",
            "ab_A3_payload_feasible",
            "main_ref_payload_feasible",
            "main_payload_feasible",
        ),
    )
    out["heur_used_sats"] = _first_present(
        flat,
        (
            "payload_best_m",
            "main_ref_lb_best_m",
            "ablation_fixed_prefix_m",
            "main_ref_best_m",
            "main_best_m",
        ),
    )

    heur_K, _ = _find_prefixed_metric(flat, prefixes, ("K",), ("_k",))
    heur_U_max, _ = _find_prefixed_metric(flat, prefixes, ("U_max",), ("u_max",))
    heur_U_mean, _ = _find_prefixed_metric(flat, prefixes, ("U_mean",), ("u_mean",))
    heur_edge, _ = _find_prefixed_metric(flat, prefixes, ("ent_edge_pct", "enterprise_edge_pct"), ("edge", "pct"))
    heur_risk, _ = _find_prefixed_metric(flat, prefixes, ("ent_risk", "risk", "enterprise_risk"), ("risk",))
    heur_r_mean, _ = _find_prefixed_metric(
        flat,
        prefixes,
        ("beam_r_mean_km", "beam_radius_mean_km", "radius_mean_km", "r_mean_km"),
        ("radius", "mean"),
    )

    out.update(
        {
            "heur_K": heur_K,
            "heur_U_max": heur_U_max,
            "heur_U_mean": heur_U_mean,
            "heur_ent_edge_pct": heur_edge,
            "heur_ent_risk": heur_risk,
            "heur_beam_r_mean": heur_r_mean,
        }
    )

    out["heur_payload_n_viol_K"] = _first_present(flat, ("payload_n_viol_K",))
    out["heur_payload_n_viol_T"] = _first_present(flat, ("payload_n_viol_T",))
    out["heur_payload_K_over_sum"] = _first_present(flat, ("payload_K_over_sum",))
    out["heur_payload_T_over_sum"] = _first_present(flat, ("payload_T_over_sum",))
    out["heur_payload_K_max"] = _first_present(flat, ("payload_K_max",))
    out["heur_payload_T_max"] = _first_present(flat, ("payload_T_max",))
    return out


def _is_milp_optimal(status_name: Any, mip_gap: Any, feasible: bool) -> bool:
    if not feasible:
        return False
    s = str(status_name).strip().upper()
    if s == "OPTIMAL":
        return True
    g = _safe_float(mip_gap)
    return g is not None and g <= 1e-9


def _set_default_error_fields(row: dict[str, Any]) -> None:
    row.setdefault("compare_ok", 1)
    row.setdefault("heur_error", "")
    row.setdefault("milp_error", "")


def _fill_heur_nan_fields(row: dict[str, Any]) -> None:
    nan = float("nan")
    defaults = {
        "heur_payload_feasible": nan,
        "heur_used_sats": nan,
        "heur_K": nan,
        "heur_U_max": nan,
        "heur_U_mean": nan,
        "heur_ent_edge_pct": nan,
        "heur_ent_risk": nan,
        "heur_beam_r_mean": nan,
        "heur_payload_n_viol_K": nan,
        "heur_payload_n_viol_T": nan,
        "heur_payload_K_over_sum": nan,
        "heur_payload_T_over_sum": nan,
        "heur_payload_K_max": nan,
        "heur_payload_T_max": nan,
        "heur_wall_s": nan,
    }
    for k, v in defaults.items():
        row.setdefault(k, v)


def _fill_milp_nan_fields(row: dict[str, Any]) -> None:
    nan = float("nan")
    defaults = {
        "milp_feasible": nan,
        "milp_status": "ERROR",
        "milp_objective": nan,
        "milp_best_bound": nan,
        "milp_mip_gap": nan,
        "milp_solve_time_s": nan,
        "milp_wall_s": nan,
        "milp_used_sats": nan,
        "milp_used_sat_indices": "",
        "milp_K": nan,
        "milp_n_active_beams": nan,
        "milp_active_beam_ids": "",
        "milp_n_grid_centers": nan,
        "milp_n_candidates": nan,
        "milp_sat_beam_count_max": nan,
        "milp_sat_beam_count_mean": nan,
        "milp_optimal": nan,
        "milp_sat_time_load_max": nan,
        "milp_sat_time_load_mean": nan,
        "milp_beam_time_load_max": nan,
        "milp_beam_time_load_mean": nan,
    }
    for k, v in defaults.items():
        row.setdefault(k, v)


def compare_one(
    base: ScenarioConfig,
    sweep: CompareSweepConfig,
    *,
    n_users: int,
    seed: int,
    milp_grid_spacing_m: float | None = None,
) -> dict[str, Any]:
    cfg = _build_compare_cfg(base, sweep, n_users=n_users, seed=seed)
    prof = SimpleProfiler()
    row: dict[str, Any] = {
        "seed": seed,
        "n_users": n_users,
        "region_mode": cfg.region_mode,
        "milp_n_candidate_sats": sweep.milp_n_candidate_sats,
        "milp_grid_spacing_m": (sweep.milp_grid_spacing_m if milp_grid_spacing_m is None else float(milp_grid_spacing_m)),
        "milp_objective_mode": sweep.milp_objective_mode,
        "milp_time_limit_s": sweep.milp_time_limit_s,
        "cfg_theta_3db_deg": getattr(cfg.beam, "theta_3db_deg", None),
        "cfg_W_slots": getattr(cfg.payload, "W_slots", None),
        "cfg_J_lanes": getattr(cfg.payload, "J_lanes", None),
        "cfg_Ks_max": getattr(cfg.payload, "Ks_max", None),
    }
    _set_default_error_fields(row)

    # Heuristic side
    try:
        prof.start("heuristic_wall_s")
        rec = run_scenario(cfg)
        prof.stop("heuristic_wall_s")
        flat = _flatten_record(rec)
        row["heur_wall_s"] = prof.timings.get("heuristic_wall_s", float("nan"))
        row.update(_heuristic_metrics_from_flat(flat, sweep.heuristic_prefixes))
        for k, v in flat.items():
            if k.startswith("time_"):
                row[f"heur_{k}"] = v
    except Exception as e:
        if "heuristic_wall_s" in prof._tic:
            try:
                prof.stop("heuristic_wall_s")
            except Exception:
                pass
        row["compare_ok"] = 0
        row["heur_error"] = f"{type(e).__name__}: {e}"
        row["heur_traceback"] = traceback.format_exc()
        _fill_heur_nan_fields(row)

    # MILP side
    try:
        milp_cfg = _milp_runner_cfg(sweep, grid_spacing_m=milp_grid_spacing_m)
        prof.start("milp_wall_s")
        milp_out = run_milp_experiment(cfg, run_cfg=milp_cfg)
        prof.stop("milp_wall_s")
        sol = milp_out["solution"]
        diag = milp_out.get("diagnostics", {})
        data = milp_out.get("data", None)
        grid = milp_out.get("grid", None)

        row["milp_feasible"] = int(bool(sol.feasible))
        row["milp_status"] = sol.status_name
        row["milp_objective"] = sol.objective_value
        row["milp_best_bound"] = sol.best_bound
        row["milp_mip_gap"] = sol.mip_gap
        row["milp_solve_time_s"] = sol.solve_time_s
        row["milp_wall_s"] = prof.timings.get("milp_wall_s", float("nan"))
        row["milp_used_sats"] = sol.n_used_sats
        row["milp_used_sat_indices"] = " ".join(map(str, sol.used_sat_indices))
        row["milp_K"] = sol.K_total
        row["milp_n_active_beams"] = len(sol.active_beam_ids)
        row["milp_active_beam_ids"] = " ".join(map(str, sol.active_beam_ids))
        row["milp_n_grid_centers"] = len(grid) if grid is not None else None
        row["milp_n_candidates"] = len(data.candidates) if data is not None else None
        row["milp_sat_beam_count_max"] = max(sol.sat_beam_count.values()) if getattr(sol, "sat_beam_count", None) else None
        row["milp_sat_beam_count_mean"] = (
            sum(sol.sat_beam_count.values()) / len(sol.sat_beam_count) if getattr(sol, "sat_beam_count", None) else None
        )
        row["milp_optimal"] = int(_is_milp_optimal(sol.status_name, sol.mip_gap, bool(sol.feasible)))

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

    except Exception as e:
        if "milp_wall_s" in prof._tic:
            try:
                prof.stop("milp_wall_s")
            except Exception:
                pass
        row["compare_ok"] = 0
        row["milp_error"] = f"{type(e).__name__}: {e}"
        row["milp_traceback"] = traceback.format_exc()
        _fill_milp_nan_fields(row)

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

    heur_U_max_f = _safe_float(row.get("heur_U_max"))
    milp_beam_time_load_max_f = _safe_float(row.get("milp_beam_time_load_max"))
    if heur_U_max_f is not None and milp_beam_time_load_max_f is not None:
        row["gap_peak_load_abs"] = heur_U_max_f - milp_beam_time_load_max_f

    heur_U_mean_f = _safe_float(row.get("heur_U_mean"))
    milp_beam_time_load_mean_f = _safe_float(row.get("milp_beam_time_load_mean"))
    if heur_U_mean_f is not None and milp_beam_time_load_mean_f is not None:
        row["gap_mean_load_abs"] = heur_U_mean_f - milp_beam_time_load_mean_f

    heur_wall_f = _safe_float(row.get("heur_wall_s"))
    milp_wall_f = _safe_float(row.get("milp_wall_s"))
    if heur_wall_f is not None and milp_wall_f is not None and milp_wall_f > 0.0:
        row["runtime_ratio_heur_over_milp"] = heur_wall_f / milp_wall_f

    return row


def _run_parallel_jobs(jobs: list[tuple[str, dict[str, Any]]], fn, max_workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(fn, **kwargs): (label, kwargs) for label, kwargs in jobs}
        for fut in as_completed(fut_map):
            done += 1
            label, kwargs = fut_map[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "seed": kwargs.get("seed"),
                    "n_users": kwargs.get("n_users"),
                    "milp_grid_spacing_m": kwargs.get("milp_grid_spacing_m"),
                    "compare_ok": 0,
                    "heur_error": "",
                    "milp_error": f"WorkerCrash: {type(e).__name__}: {e}",
                    "milp_traceback": traceback.format_exc(),
                }
                _fill_heur_nan_fields(row)
                _fill_milp_nan_fields(row)
            rows.append(row)
            print(f"[{done}/{total}] {label} done")
    return rows


def run_compare_sweep(base: ScenarioConfig, sweep: CompareSweepConfig) -> list[dict[str, Any]]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for n_users in sweep.n_users_list:
        for seed in sweep.seeds:
            label = f"Comparing MILP vs heuristic for n_users={n_users}, seed={seed}"
            jobs.append((label, {"base": base, "sweep": sweep, "n_users": n_users, "seed": seed}))

    if sweep.parallel_enabled and sweep.max_workers > 1:
        rows = _run_parallel_jobs(jobs, compare_one, sweep.max_workers)
        rows.sort(key=lambda r: (int(r.get("n_users", 0)), int(r.get("seed", 0))))
        return rows

    rows: list[dict[str, Any]] = []
    total = len(jobs)
    done = 0
    for label, kwargs in jobs:
        done += 1
        print(f"[{done}/{total}] {label} ...")
        rows.append(compare_one(**kwargs))
    return rows


def run_grid_sensitivity(base: ScenarioConfig, sweep: CompareSweepConfig) -> list[dict[str, Any]]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for grid_spacing_m in sweep.grid_sensitivity_grid_spacing_m_list:
        for n_users in sweep.grid_sensitivity_n_users_list:
            for seed in sweep.seeds:
                label = f"grid sensitivity spacing={grid_spacing_m:.0f} m, n_users={n_users}, seed={seed}"
                jobs.append(
                    (
                        label,
                        {
                            "base": base,
                            "sweep": sweep,
                            "n_users": n_users,
                            "seed": seed,
                            "milp_grid_spacing_m": grid_spacing_m,
                        },
                    )
                )

    if sweep.parallel_enabled and sweep.max_workers > 1:
        rows = _run_parallel_jobs(jobs, compare_one, sweep.max_workers)
        for row in rows:
            row["study"] = "grid_sensitivity"
        rows.sort(
            key=lambda r: (
                float(r.get("milp_grid_spacing_m", 0.0)),
                int(r.get("n_users", 0)),
                int(r.get("seed", 0)),
            )
        )
        return rows

    rows: list[dict[str, Any]] = []
    total = len(jobs)
    done = 0
    for label, kwargs in jobs:
        done += 1
        print(f"[grid {done}/{total}] {label} ...")
        row = compare_one(**kwargs)
        row["study"] = "grid_sensitivity"
        rows.append(row)
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


def _agg_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "median": v, "std": 0.0, "p10": v, "p90": v, "min": v, "max": v, "n": 1}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _aggregate_metric(
    rows: list[dict[str, Any]],
    group_values: Sequence[int | float],
    group_key: str,
    series_to_key: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_val in group_values:
        rows_g = [r for r in rows if r.get(group_key) == group_val]
        for series, key in series_to_key.items():
            vals = [_safe_float(r.get(key)) for r in rows_g]
            vals = [v for v in vals if v is not None]
            st = _agg_stats(vals)
            out.append({group_key: group_val, "series": series, **st})
    return out


def write_aggregate_csv(rows: list[dict[str, Any]], sweep: CompareSweepConfig) -> Path:
    blocks = {
        "feasible_rate": {
            "Heuristic": "heur_payload_feasible",
            "MILP": "milp_feasible",
        },
        "optimal_solve_rate": {
            "MILP optimal": "milp_optimal",
        },
        "K": {
            "Heuristic": "heur_K",
            "MILP": "milp_K",
        },
        "used_sats": {
            "Heuristic": "heur_used_sats",
            "MILP": "milp_used_sats",
        },
        "mean_beam_load": {
            "Heuristic": "heur_U_mean",
            "MILP": "milp_beam_time_load_mean",
        },
        "peak_beam_load": {
            "Heuristic": "heur_U_max",
            "MILP": "milp_beam_time_load_max",
        },
        "runtime_s": {
            "Heuristic": "heur_wall_s",
            "MILP": "milp_wall_s",
        },
        "mip_gap": {
            "MILP": "milp_mip_gap",
        },
        "K_gap_pct": {
            "Heuristic minus MILP": "gap_K_pct",
        },
        "used_sats_gap_abs": {
            "Heuristic minus MILP": "gap_used_sats_abs",
        },
    }

    agg_rows: list[dict[str, Any]] = []
    for metric_name, series_to_key in blocks.items():
        block_rows = _aggregate_metric(rows, sweep.n_users_list, "n_users", series_to_key)
        for br in block_rows:
            agg_rows.append({"metric": metric_name, **br})

    out_path = Path(sweep.aggregate_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "n_users", "series", "mean", "median", "std", "p10", "p90", "min", "max", "n"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    return out_path


def write_grid_sensitivity_aggregate_csv(rows: list[dict[str, Any]], sweep: CompareSweepConfig) -> Path:
    blocks = {
        "optimal_solve_rate": {"MILP optimal": "milp_optimal"},
        "K": {"MILP": "milp_K"},
        "runtime_s": {"MILP": "milp_wall_s"},
        "mip_gap": {"MILP": "milp_mip_gap"},
        "n_grid_centers": {"MILP": "milp_n_grid_centers"},
        "n_candidates": {"MILP": "milp_n_candidates"},
    }

    agg_rows: list[dict[str, Any]] = []
    for n_users in sweep.grid_sensitivity_n_users_list:
        rows_n = [r for r in rows if int(r["n_users"]) == int(n_users)]
        for metric_name, series_to_key in blocks.items():
            for grid_spacing_m in sweep.grid_sensitivity_grid_spacing_m_list:
                rows_ng = [r for r in rows_n if float(r["milp_grid_spacing_m"]) == float(grid_spacing_m)]
                for series, key in series_to_key.items():
                    vals = [_safe_float(r.get(key)) for r in rows_ng]
                    vals = [v for v in vals if v is not None]
                    st = _agg_stats(vals)
                    agg_rows.append(
                        {
                            "metric": metric_name,
                            "n_users": n_users,
                            "milp_grid_spacing_m": grid_spacing_m,
                            "series": series,
                            **st,
                        }
                    )

    out_path = Path(sweep.grid_sensitivity_aggregate_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "n_users",
        "milp_grid_spacing_m",
        "series",
        "mean",
        "median",
        "std",
        "p10",
        "p90",
        "min",
        "max",
        "n",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    return out_path


# -----------------------------------------------------------------------------
# Tables / plots now read the CSVs created by this script
# -----------------------------------------------------------------------------

def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected CSV not found: {p}")
    return pd.read_csv(p)


def _write_tables_from_aggregate_csv(sweep: CompareSweepConfig) -> Path:
    agg = _read_csv(sweep.aggregate_csv)

    out_path = Path(sweep.tables_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "feasible_rate",
        "optimal_solve_rate",
        "K",
        "used_sats",
        "mean_beam_load",
        "peak_beam_load",
        "runtime_s",
        "mip_gap",
        "K_gap_pct",
        "used_sats_gap_abs",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("MILP VS HEURISTIC TABLES (datapoints used in plots)\n")
        f.write(f"AGGREGATE CSV: {sweep.aggregate_csv}\n\n")
        f.write("Aggregation note:\n")
        f.write("- feasible/optimal rates are interpreted through mean\n")
        f.write("- continuous metrics are best interpreted through median and p10/p90\n\n")

        for metric in metric_order:
            dfm = agg[agg["metric"] == metric].copy()
            if dfm.empty:
                continue

            f.write("=" * 110 + "\n")
            f.write(f"TABLE: {metric}\n\n")
            header = f"{'n_users':<8} | {'series':<24} | {'mean':<12} | {'median':<12} | {'p10':<12} | {'p90':<12} | {'n':<3}\n"
            f.write(header)
            f.write("-" * (len(header) - 1) + "\n")

            for _, r in dfm.iterrows():
                f.write(
                    f"{int(r['n_users']):<8} | {str(r['series']):<24} | "
                    f"{str(r['mean']):<12} | {str(r['median']):<12} | "
                    f"{str(r['p10']):<12} | {str(r['p90']):<12} | {int(r['n']):<3}\n"
                )
            f.write("\n")

    return out_path


def _write_grid_sensitivity_tables_from_aggregate_csv(sweep: CompareSweepConfig) -> Path:
    agg = _read_csv(sweep.grid_sensitivity_aggregate_csv)

    out_path = Path(sweep.grid_sensitivity_tables_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "optimal_solve_rate",
        "K",
        "runtime_s",
        "mip_gap",
        "n_grid_centers",
        "n_candidates",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("MILP GRID-SPACING SENSITIVITY TABLES\n")
        f.write(f"AGGREGATE CSV: {sweep.grid_sensitivity_aggregate_csv}\n\n")
        f.write("Aggregation note:\n")
        f.write("- rates are interpreted through mean\n")
        f.write("- continuous metrics are best interpreted through median and p10/p90\n\n")

        for n_users in sweep.grid_sensitivity_n_users_list:
            dfn = agg[agg["n_users"] == n_users].copy()
            f.write("#" * 110 + "\n")
            f.write(f"n_users = {n_users}\n\n")

            for metric in metric_order:
                dfm = dfn[dfn["metric"] == metric].copy()
                if dfm.empty:
                    continue

                f.write("=" * 110 + "\n")
                f.write(f"TABLE: {metric}\n\n")
                header = f"{'grid_m':<10} | {'series':<16} | {'mean':<12} | {'median':<12} | {'p10':<12} | {'p90':<12} | {'n':<3}\n"
                f.write(header)
                f.write("-" * (len(header) - 1) + "\n")

                for _, r in dfm.iterrows():
                    f.write(
                        f"{int(r['milp_grid_spacing_m']):<10} | {str(r['series']):<16} | "
                        f"{str(r['mean']):<12} | {str(r['median']):<12} | "
                        f"{str(r['p10']):<12} | {str(r['p90']):<12} | {int(r['n']):<3}\n"
                    )
                f.write("\n")

    return out_path


def _plot_metric_from_aggregate(
    agg: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    y_log: bool = False,
    use_mean: bool = False,
    x_key: str = "n_users",
) -> None:
    dfm = agg[agg["metric"] == metric].copy()
    if dfm.empty:
        return

    center_col = "mean" if use_mean else "median"
    low_col = None if use_mean else "p10"
    high_col = None if use_mean else "p90"

    plt.figure(figsize=(7.4, 4.8))
    for series, g in dfm.groupby("series"):
        g = g.sort_values(x_key)
        x = g[x_key].to_numpy(dtype=float)
        y = g[center_col].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.8, markersize=5, label=str(series))
        if low_col is not None and high_col is not None:
            ylo = g[low_col].to_numpy(dtype=float)
            yhi = g[high_col].to_numpy(dtype=float)
            if np.isfinite(ylo).any() and np.isfinite(yhi).any():
                plt.fill_between(x, ylo, yhi, alpha=0.15)

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    if y_log:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_plots_from_aggregate_csv(sweep: CompareSweepConfig) -> Path:
    out_dir = Path(sweep.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _read_csv(sweep.aggregate_csv)

    specs = [
        ("feasible_rate", "Feasible rate vs n_users", "feasible rate", False, True),
        ("optimal_solve_rate", "MILP optimal solve rate vs n_users", "optimal solve rate", False, True),
        ("K", "K vs n_users", "K", False, False),
        ("used_sats", "Used satellites vs n_users", "used satellites", False, False),
        ("mean_beam_load", "Mean beam load vs n_users", "mean beam load", False, False),
        ("peak_beam_load", "Peak beam load vs n_users", "peak beam load", False, False),
        ("runtime_s", "Runtime vs n_users", "runtime (s)", True, False),
        ("mip_gap", "MILP MIP gap vs n_users", "MIP gap", False, False),
        ("K_gap_pct", "K gap (%) vs n_users", "gap_K_pct", False, False),
        ("used_sats_gap_abs", "Used satellites gap vs n_users", "gap_used_sats_abs", False, False),
    ]

    for metric, title, ylabel, y_log, use_mean in specs:
        _plot_metric_from_aggregate(
            agg,
            metric=metric,
            title=title,
            ylabel=ylabel,
            out_path=out_dir / f"{metric}_vs_nusers.png",
            y_log=y_log,
            use_mean=use_mean,
            x_key="n_users",
        )
    return out_dir


def make_grid_sensitivity_plots_from_aggregate_csv(sweep: CompareSweepConfig) -> Path:
    out_dir = Path(sweep.grid_sensitivity_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _read_csv(sweep.grid_sensitivity_aggregate_csv)

    specs = [
        ("optimal_solve_rate", "MILP optimal solve rate vs grid spacing", "optimal solve rate", False, True),
        ("K", "MILP K vs grid spacing", "K", False, False),
        ("runtime_s", "MILP runtime vs grid spacing", "runtime (s)", True, False),
        ("mip_gap", "MILP MIP gap vs grid spacing", "MIP gap", False, False),
        ("n_grid_centers", "Grid centers vs grid spacing", "# grid centers", False, False),
        ("n_candidates", "Candidate beams vs grid spacing", "# candidates", False, False),
    ]

    for n_users in sweep.grid_sensitivity_n_users_list:
        dfn = agg[agg["n_users"] == n_users].copy()
        if dfn.empty:
            continue
        for metric, title, ylabel, y_log, use_mean in specs:
            _plot_metric_from_aggregate(
                dfn,
                metric=metric,
                title=f"{title} (n_users={n_users})",
                ylabel=ylabel,
                out_path=out_dir / f"n{n_users}_{metric}_vs_grid_spacing.png",
                y_log=y_log,
                use_mean=use_mean,
                x_key="milp_grid_spacing_m",
            )
    return out_dir


def main() -> None:
    base = ScenarioConfig()
    sweep = CompareSweepConfig()

    rows = run_compare_sweep(base, sweep)
    out_path = write_rows_to_csv(rows, sweep.output_csv)
    agg_path = write_aggregate_csv(rows, sweep)
    tables_path = _write_tables_from_aggregate_csv(sweep)
    plot_dir = make_plots_from_aggregate_csv(sweep)

    print(f"Saved comparison CSV to: {out_path}")
    print(f"Saved aggregate CSV to: {agg_path}")
    print(f"Saved tables to: {tables_path}")
    print(f"Saved plots to: {plot_dir}")

    if sweep.grid_sensitivity_enabled:
        grid_rows = run_grid_sensitivity(base, sweep)
        grid_csv = write_rows_to_csv(grid_rows, sweep.grid_sensitivity_csv)
        grid_agg = write_grid_sensitivity_aggregate_csv(grid_rows, sweep)
        grid_tables = _write_grid_sensitivity_tables_from_aggregate_csv(sweep)
        grid_plots = make_grid_sensitivity_plots_from_aggregate_csv(sweep)
        print(f"Saved grid-sensitivity CSV to: {grid_csv}")
        print(f"Saved grid-sensitivity aggregate CSV to: {grid_agg}")
        print(f"Saved grid-sensitivity tables to: {grid_tables}")
        print(f"Saved grid-sensitivity plots to: {grid_plots}")


if __name__ == "__main__":
    main()