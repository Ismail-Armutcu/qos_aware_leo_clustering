from __future__ import annotations

import csv
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt

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
    aggregate_csv: str = "milp_compare_aggregate.csv"
    tables_txt: str = "plots/milp_compare_tables.txt"
    output_dir: str = "plots/milp_compare_plots"

    region_mode: str = "turkey"
    n_users_list: tuple[int, ...] = (50, 75, 100)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)

    # Reduced MILP instances / fixed pool
    milp_n_candidate_sats: int = 6
    milp_grid_spacing_m: float = 15_000.0
    milp_grid_margin_m: float = 0.0
    milp_time_limit_s: float = 300.0
    milp_mip_gap: float = 0.0
    milp_threads: int | None = None
    milp_log_to_console: bool = True
    milp_objective_mode: str = "weighted_sat_beam"  # beam_only | weighted_sat_beam
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

    heur_K, heur_K_key = _find_prefixed_metric(flat, prefixes, ("K",), ("_k",))
    heur_U_max, heur_U_key = _find_prefixed_metric(flat, prefixes, ("U_max",), ("u_max",))
    heur_edge, heur_edge_key = _find_prefixed_metric(
        flat, prefixes, ("ent_edge_pct", "enterprise_edge_pct"), ("edge", "pct")
    )
    heur_risk, heur_risk_key = _find_prefixed_metric(
        flat, prefixes, ("ent_risk", "risk", "enterprise_risk"), ("risk",)
    )
    heur_r_mean, heur_r_mean_key = _find_prefixed_metric(
        flat,
        prefixes,
        ("beam_r_mean_km", "beam_radius_mean_km", "radius_mean_km", "r_mean_km"),
        ("radius", "mean"),
    )
    heur_U_mean, heur_U_mean_key = _find_prefixed_metric(flat, prefixes, ("U_mean",), ("u_mean",))

    out.update(
        {
            "heur_K": heur_K,
            "heur_U_max": heur_U_max,
            "heur_U_mean": heur_U_mean,
            "heur_ent_edge_pct": heur_edge,
            "heur_ent_risk": heur_risk,
            "heur_beam_r_mean": heur_r_mean,
            "heur_K_key": heur_K_key,
            "heur_U_max_key": heur_U_key,
            "heur_U_mean_key": heur_U_mean_key,
            "heur_ent_edge_pct_key": heur_edge_key,
            "heur_ent_risk_key": heur_risk_key,
            "heur_beam_r_mean_key": heur_r_mean_key,
        }
    )

    # Helpful load / payload diagnostics if present
    out["heur_payload_n_viol_K"] = _first_present(flat, ("payload_n_viol_K",))
    out["heur_payload_n_viol_T"] = _first_present(flat, ("payload_n_viol_T",))
    out["heur_payload_K_over_sum"] = _first_present(flat, ("payload_K_over_sum",))
    out["heur_payload_T_over_sum"] = _first_present(flat, ("payload_T_over_sum",))
    out["heur_payload_K_max"] = _first_present(flat, ("payload_K_max",))
    out["heur_payload_T_max"] = _first_present(flat, ("payload_T_max",))
    return out


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
    row["heur_wall_s"] = prof.timings["heuristic_wall_s"]
    row.update(_heuristic_metrics_from_flat(flat, sweep.heuristic_prefixes))

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
    row["milp_sat_beam_count_max"] = max(sol.sat_beam_count.values()) if getattr(sol, "sat_beam_count", None) else None
    row["milp_sat_beam_count_mean"] = (
        sum(sol.sat_beam_count.values()) / len(sol.sat_beam_count) if getattr(sol, "sat_beam_count", None) else None
    )

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

    heur_U_max_f = _safe_float(row.get("heur_U_max"))
    milp_beam_time_load_max_f = _safe_float(row.get("milp_beam_time_load_max"))
    if heur_U_max_f is not None and milp_beam_time_load_max_f is not None:
        row["gap_peak_load_abs"] = heur_U_max_f - milp_beam_time_load_max_f

    heur_wall_f = _safe_float(row.get("heur_wall_s"))
    milp_wall_f = _safe_float(row.get("milp_wall_s"))
    if heur_wall_f is not None and milp_wall_f is not None and milp_wall_f > 0.0:
        row["runtime_ratio_heur_over_milp"] = heur_wall_f / milp_wall_f

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


def _agg_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0], "n": 1}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _aggregate_metric(rows: list[dict[str, Any]], n_users_list: Sequence[int], series_to_key: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n_users in n_users_list:
        rows_n = [r for r in rows if int(r["n_users"]) == int(n_users)]
        for series, key in series_to_key.items():
            vals = [_safe_float(r.get(key)) for r in rows_n]
            vals = [v for v in vals if v is not None]
            st = _agg_stats(vals)
            out.append({"n_users": n_users, "series": series, **st})
    return out


def write_aggregate_csv(rows: list[dict[str, Any]], sweep: CompareSweepConfig) -> Path:
    # long-form aggregate rows across the core comparison metrics
    blocks = {
        "feasible_rate": {
            "Heuristic": "heur_payload_feasible",
            "MILP": "milp_feasible",
        },
        "K": {
            "Heuristic": "heur_K",
            "MILP": "milp_K",
        },
        "used_sats": {
            "Heuristic": "heur_used_sats",
            "MILP": "milp_used_sats",
        },
        "Umax_or_peak_load": {
            "Heuristic": "heur_U_max",
            "MILP": "milp_beam_time_load_max",
        },
        "runtime_s": {
            "Heuristic": "heur_wall_s",
            "MILP": "milp_wall_s",
        },
        "K_gap_pct": {
            "Heuristic minus MILP": "gap_K_pct",
        },
        "sat_time_load_max": {
            "MILP": "milp_sat_time_load_max",
            "Heuristic(payload T_max)": "heur_payload_T_max",
        },
        "sat_beam_count_max": {
            "MILP": "milp_sat_beam_count_max",
            "Heuristic(payload K_max)": "heur_payload_K_max",
        },
    }

    agg_rows: list[dict[str, Any]] = []
    for metric_name, series_to_key in blocks.items():
        block_rows = _aggregate_metric(rows, sweep.n_users_list, series_to_key)
        for br in block_rows:
            agg_rows.append({"metric": metric_name, **br})

    out_path = Path(sweep.aggregate_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "n_users", "series", "mean", "std", "min", "max", "n"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    return out_path


def _write_tables(rows: list[dict[str, Any]], sweep: CompareSweepConfig) -> Path:
    out_path = Path(sweep.tables_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        (
            "feasible_rate_vs_nusers",
            {"Heuristic": "heur_payload_feasible", "MILP": "milp_feasible"},
        ),
        (
            "K_vs_nusers",
            {"Heuristic": "heur_K", "MILP": "milp_K"},
        ),
        (
            "used_sats_vs_nusers",
            {"Heuristic": "heur_used_sats", "MILP": "milp_used_sats"},
        ),
        (
            "Umax_or_peak_load_vs_nusers",
            {"Heuristic": "heur_U_max", "MILP": "milp_beam_time_load_max"},
        ),
        (
            "runtime_vs_nusers",
            {"Heuristic": "heur_wall_s", "MILP": "milp_wall_s"},
        ),
        (
            "K_gap_pct_vs_nusers",
            {"Heuristic minus MILP": "gap_K_pct"},
        ),
        (
            "sat_time_load_max_vs_nusers",
            {"Heuristic(payload T_max)": "heur_payload_T_max", "MILP": "milp_sat_time_load_max"},
        ),
        (
            "sat_beam_count_max_vs_nusers",
            {"Heuristic(payload K_max)": "heur_payload_K_max", "MILP": "milp_sat_beam_count_max"},
        ),
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("MILP VS HEURISTIC TABLES (datapoints used in plots)\n")
        f.write(f"CSV: {sweep.output_csv}\n\n")
        f.write("Aggregation note: NaN values are excluded from means/min/max and from n.\n\n")
        for title, mapping in blocks:
            f.write("=" * 90 + "\n")
            f.write(f"TABLE: {title}\n\n")
            rows_block = _aggregate_metric(rows, sweep.n_users_list, mapping)
            header = f"{'n_users':<8} | {'series':<22} | {'mean':<12} | {'std':<12} | {'min':<10} | {'max':<10} | {'n':<3}\n"
            f.write(header)
            f.write("-" * (len(header) - 1) + "\n")
            for r in rows_block:
                f.write(
                    f"{int(r['n_users']):<8} | {str(r['series']):<22} | {str(r['mean']):<12} | {str(r['std']):<12} | "
                    f"{str(r['min']):<10} | {str(r['max']):<10} | {int(r['n']):<3}\n"
                )
            f.write("\n")
    return out_path


def _series_points(rows: list[dict[str, Any]], n_users_list: Sequence[int], series_to_key: dict[str, str]) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {name: [] for name in series_to_key}
    for name, key in series_to_key.items():
        agg_rows = _aggregate_metric(rows, n_users_list, {name: key})
        for r in agg_rows:
            v = _safe_float(r["mean"])
            if v is not None:
                out[name].append((int(r["n_users"]), float(v)))
    return out


def _plot_metric(
    rows: list[dict[str, Any]],
    n_users_list: Sequence[int],
    series_to_key: dict[str, str],
    title: str,
    ylabel: str,
    out_path: Path,
    *,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=(7.4, 4.8))
    series_pts = _series_points(rows, n_users_list, series_to_key)
    for label, pts in series_pts.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=label)
    plt.title(title)
    plt.xlabel("n_users")
    plt.ylabel(ylabel)
    if y_log:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_plots(rows: list[dict[str, Any]], sweep: CompareSweepConfig) -> Path:
    out_dir = Path(sweep.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "feasible_rate_vs_nusers.png",
            {"Heuristic": "heur_payload_feasible", "MILP": "milp_feasible"},
            "Feasible rate vs n_users",
            "feasible rate",
            False,
        ),
        (
            "K_vs_nusers.png",
            {"Heuristic": "heur_K", "MILP": "milp_K"},
            "K vs n_users",
            "K",
            False,
        ),
        (
            "used_sats_vs_nusers.png",
            {"Heuristic": "heur_used_sats", "MILP": "milp_used_sats"},
            "Used satellites vs n_users",
            "used satellites",
            False,
        ),
        (
            "Umax_or_peak_load_vs_nusers.png",
            {"Heuristic": "heur_U_max", "MILP": "milp_beam_time_load_max"},
            "Peak beam load vs n_users",
            "peak load",
            False,
        ),
        (
            "runtime_vs_nusers.png",
            {"Heuristic": "heur_wall_s", "MILP": "milp_wall_s"},
            "Runtime vs n_users",
            "runtime (s)",
            True,
        ),
        (
            "K_gap_pct_vs_nusers.png",
            {"Heuristic minus MILP": "gap_K_pct"},
            "K gap (%) vs n_users",
            "gap_K_pct",
            False,
        ),
        (
            "sat_time_load_max_vs_nusers.png",
            {"Heuristic(payload T_max)": "heur_payload_T_max", "MILP": "milp_sat_time_load_max"},
            "Max satellite time load vs n_users",
            "max sat time load",
            False,
        ),
        (
            "sat_beam_count_max_vs_nusers.png",
            {"Heuristic(payload K_max)": "heur_payload_K_max", "MILP": "milp_sat_beam_count_max"},
            "Max satellite beam count vs n_users",
            "max sat beam count",
            False,
        ),
    ]
    for filename, mapping, title, ylabel, y_log in specs:
        _plot_metric(rows, sweep.n_users_list, mapping, title, ylabel, out_dir / filename, y_log=y_log)
    return out_dir


def main() -> None:
    base = ScenarioConfig()
    sweep = CompareSweepConfig()
    rows = run_compare_sweep(base, sweep)
    out_path = write_rows_to_csv(rows, sweep.output_csv)
    agg_path = write_aggregate_csv(rows, sweep)
    tables_path = _write_tables(rows, sweep)
    plot_dir = make_plots(rows, sweep)

    print(f"Saved comparison CSV to: {out_path}")
    print(f"Saved aggregate CSV to: {agg_path}")
    print(f"Saved tables to: {tables_path}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
