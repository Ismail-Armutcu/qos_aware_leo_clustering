# src/sweep_plots.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ----------------------------
# CSV loading
# ----------------------------
def _try_float(x: str) -> Any:
    s = str(x).strip()
    if s == "":
        return np.nan
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except Exception:
        return s


def load_sweep_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    if pd is not None:
        return pd.read_csv(path)

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        return {}

    cols = list(rows[0].keys())
    data: Dict[str, List[Any]] = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            data[c].append(_try_float(row.get(c, "")))

    out: Dict[str, np.ndarray] = {}
    for c, vals in data.items():
        # best-effort numeric conversion
        try:
            out[c] = np.asarray(vals, dtype=float)
        except Exception:
            out[c] = np.asarray(vals, dtype=object)
    return out


def _is_pandas_df(df) -> bool:
    return pd is not None and hasattr(df, "columns")


def _has_col(df, col: str) -> bool:
    if _is_pandas_df(df):
        return col in df.columns
    return isinstance(df, dict) and col in df


def _col(df, col: str) -> np.ndarray:
    if _is_pandas_df(df):
        return df[col].to_numpy()
    return np.asarray(df[col])


def _nrows(df) -> int:
    if _is_pandas_df(df):
        return int(df.shape[0])
    if isinstance(df, dict) and df:
        k0 = next(iter(df.keys()))
        return int(len(df[k0]))
    return 0


# ----------------------------
# Aggregation
# ----------------------------
@dataclass(frozen=True)
class AggSeries:
    x: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    n: np.ndarray


_DOWNSTREAM_INVALID_IF_PAYLOAD_INFEASIBLE = {
    # refinement/main downstream summaries
    "main_ref_K", "main_ref_feasible_rate", "main_ref_U_mean", "main_ref_U_max", "main_ref_U_min",
    "main_ref_risk_sum", "main_ref_ent_total", "main_ref_ent_exposed", "main_ref_ent_edge_pct",
    "main_ref_ent_z_mean", "main_ref_ent_z_p90", "main_ref_ent_z_max",
    "main_ref_radius_mean_km", "main_ref_radius_min_km", "main_ref_radius_p10_km", "main_ref_radius_p50_km",
    "main_ref_radius_p90_km", "main_ref_radius_max_km",
    "main_ref_sat_mean_radius_mean_km", "main_ref_sat_mean_radius_min_km", "main_ref_sat_mean_radius_max_km",

    "main_ref_lb_K", "main_ref_lb_feasible_rate", "main_ref_lb_U_mean", "main_ref_lb_U_max", "main_ref_lb_U_min",
    "main_ref_lb_risk_sum", "main_ref_lb_ent_total", "main_ref_lb_ent_exposed", "main_ref_lb_ent_edge_pct",
    "main_ref_lb_ent_z_mean", "main_ref_lb_ent_z_p90", "main_ref_lb_ent_z_max",
    "main_ref_lb_radius_mean_km", "main_ref_lb_radius_min_km", "main_ref_lb_radius_p10_km", "main_ref_lb_radius_p50_km",
    "main_ref_lb_radius_p90_km", "main_ref_lb_radius_max_km",
    "main_ref_lb_sat_mean_radius_mean_km", "main_ref_lb_sat_mean_radius_min_km", "main_ref_lb_sat_mean_radius_max_km",

    # baselines
    "wk_demand_fixed_K", "wk_demand_fixed_feasible_rate", "wk_demand_fixed_U_mean", "wk_demand_fixed_U_max", "wk_demand_fixed_U_min",
    "wk_demand_fixed_risk_sum", "wk_demand_fixed_ent_total", "wk_demand_fixed_ent_exposed", "wk_demand_fixed_ent_edge_pct",
    "wk_demand_fixed_ent_z_mean", "wk_demand_fixed_ent_z_p90", "wk_demand_fixed_ent_z_max",
    "wk_demand_rep_K", "wk_demand_rep_feasible_rate", "wk_demand_rep_U_mean", "wk_demand_rep_U_max", "wk_demand_rep_U_min",
    "wk_demand_rep_risk_sum", "wk_demand_rep_ent_total", "wk_demand_rep_ent_exposed", "wk_demand_rep_ent_edge_pct",
    "wk_demand_rep_ent_z_mean", "wk_demand_rep_ent_z_p90", "wk_demand_rep_ent_z_max",

    "wk_qos_fixed_K", "wk_qos_fixed_feasible_rate", "wk_qos_fixed_U_mean", "wk_qos_fixed_U_max", "wk_qos_fixed_U_min",
    "wk_qos_fixed_risk_sum", "wk_qos_fixed_ent_total", "wk_qos_fixed_ent_exposed", "wk_qos_fixed_ent_edge_pct",
    "wk_qos_fixed_ent_z_mean", "wk_qos_fixed_ent_z_p90", "wk_qos_fixed_ent_z_max",
    "wk_qos_rep_K", "wk_qos_rep_feasible_rate", "wk_qos_rep_U_mean", "wk_qos_rep_U_max", "wk_qos_rep_U_min",
    "wk_qos_rep_risk_sum", "wk_qos_rep_ent_total", "wk_qos_rep_ent_exposed", "wk_qos_rep_ent_edge_pct",
    "wk_qos_rep_ent_z_mean", "wk_qos_rep_ent_z_p90", "wk_qos_rep_ent_z_max",

    "bk_fixed_K", "bk_fixed_feasible_rate", "bk_fixed_U_mean", "bk_fixed_U_max", "bk_fixed_U_min",
    "bk_fixed_risk_sum", "bk_fixed_ent_total", "bk_fixed_ent_exposed", "bk_fixed_ent_edge_pct",
    "bk_fixed_ent_z_mean", "bk_fixed_ent_z_p90", "bk_fixed_ent_z_max",
    "bk_rep_K", "bk_rep_feasible_rate", "bk_rep_U_mean", "bk_rep_U_max", "bk_rep_U_min",
    "bk_rep_risk_sum", "bk_rep_ent_total", "bk_rep_ent_exposed", "bk_rep_ent_edge_pct",
    "bk_rep_ent_z_mean", "bk_rep_ent_z_p90", "bk_rep_ent_z_max",

    "tgbp_fixed_K", "tgbp_fixed_feasible_rate", "tgbp_fixed_U_mean", "tgbp_fixed_U_max", "tgbp_fixed_U_min",
    "tgbp_fixed_risk_sum", "tgbp_fixed_ent_total", "tgbp_fixed_ent_exposed", "tgbp_fixed_ent_edge_pct",
    "tgbp_fixed_ent_z_mean", "tgbp_fixed_ent_z_p90", "tgbp_fixed_ent_z_max",
    "tgbp_rep_K", "tgbp_rep_feasible_rate", "tgbp_rep_U_mean", "tgbp_rep_U_max", "tgbp_rep_U_min",
    "tgbp_rep_risk_sum", "tgbp_rep_ent_total", "tgbp_rep_ent_exposed", "tgbp_rep_ent_edge_pct",
    "tgbp_rep_ent_z_mean", "tgbp_rep_ent_z_p90", "tgbp_rep_ent_z_max",

    # timings for stages that are not run on payload-infeasible cases
    "time_ent_ref_s", "time_lb_ref_s",
    "time_baseline_without_qos_s", "time_baseline_with_qos_s", "time_baseline_bkmeans_s", "time_baseline_tgbp_s",
}


def _mask_payload_infeasible_values(df, ycol: str, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    if ycol not in _DOWNSTREAM_INVALID_IF_PAYLOAD_INFEASIBLE:
        return y
    if not _has_col(df, "payload_feasible"):
        return y
    feasible = _col(df, "payload_feasible").astype(float, copy=False)
    bad = np.isnan(feasible) | (feasible < 0.5)
    if bad.shape[0] == y.shape[0]:
        y[bad] = np.nan
    return y


def aggregate_by_group_mean_std(df, xcol: str, ycol: str) -> AggSeries:
    if not (_has_col(df, xcol) and _has_col(df, ycol)):
        return AggSeries(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    x = _col(df, xcol).astype(float, copy=False)
    y = _col(df, ycol).astype(float, copy=False)
    y = _mask_payload_infeasible_values(df, ycol, y)

    xs = np.unique(x[~np.isnan(x)])
    xs = np.sort(xs)

    y_mean = np.zeros_like(xs, dtype=float)
    y_std = np.zeros_like(xs, dtype=float)
    y_min = np.zeros_like(xs, dtype=float)
    y_max = np.zeros_like(xs, dtype=float)
    n = np.zeros_like(xs, dtype=int)

    for i, xv in enumerate(xs):
        idx = np.where(x == xv)[0]
        yy = y[idx]
        yy = yy[~np.isnan(yy)]
        if yy.size == 0:
            y_mean[i] = np.nan
            y_std[i] = np.nan
            y_min[i] = np.nan
            y_max[i] = np.nan
            n[i] = 0
        else:
            y_mean[i] = float(np.mean(yy))
            y_std[i] = float(np.std(yy))
            y_min[i] = float(np.min(yy))
            y_max[i] = float(np.max(yy))
            n[i] = int(yy.size)

    return AggSeries(xs, y_mean, y_std, y_min, y_max, n)


# ----------------------------
# Plot utils
# ----------------------------
def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _save_or_show(fig, out_path: str | None, show: bool) -> None:
    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    elif show:
        fig.tight_layout()
        plt.show()
    else:
        plt.close(fig)


def plot_lines_vs_nusers(
    df,
    series: Dict[str, str],
    *,
    title: str,
    ylabel: str,
    out_path: str | None = None,
    show: bool = False,
    include_errorbars: bool = False,
):
    if not series:
        return

    fig = plt.figure()
    ax = plt.gca()

    for label, col in series.items():
        if not _has_col(df, col):
            continue
        agg = aggregate_by_group_mean_std(df, "n_users", col)
        if agg.x.size == 0:
            continue

        if include_errorbars:
            ax.errorbar(
                agg.x, agg.y_mean,
                yerr=agg.y_std,
                marker="o",
                linestyle="-",
                capsize=3,
                label=label,
            )
        else:
            ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label=label)

    ax.set_title(title)
    ax.set_xlabel("n_users")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(ncol=1, fontsize=9)
    _save_or_show(fig, out_path, show)


def plot_payload_ratio_vs_nusers(df, *, title: str, out_path: str | None = None, show: bool = False):
    if not (_has_col(df, "payload_T_sum") and _has_col(df, "payload_global_cap")):
        return

    T_sum = _col(df, "payload_T_sum").astype(float, copy=False)
    cap = _col(df, "payload_global_cap").astype(float, copy=False)
    ratio = T_sum / np.maximum(cap, 1e-12)

    tmp = {"n_users": _col(df, "n_users"), "ratio": ratio}

    fig = plt.figure()
    ax = plt.gca()
    agg = aggregate_by_group_mean_std(tmp, "n_users", "ratio")
    ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel("n_users")
    ax.set_ylabel("ratio")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    _save_or_show(fig, out_path, show)


# ----------------------------
# Phase B tables
# ----------------------------
def _augment_with_k_theoretical_min(df):
    """
    Add a derived global beam-count lower bound:
        K_lb = ceil(T_sum / W_slots)
    This is a global time-based lower bound, not a per-satellite packing bound.
    Returned object matches the input style (pandas DataFrame or dict of arrays).
    """
    if not (_has_col(df, "payload_T_sum") and _has_col(df, "payload_W_slots") and _has_col(df, "n_users")):
        return df

    T_sum = _col(df, "payload_T_sum").astype(float, copy=False)
    W = _col(df, "payload_W_slots").astype(float, copy=False)
    K_lb = np.ceil(T_sum / np.maximum(W, 1e-12))

    if _is_pandas_df(df):
        out = df.copy()
        out["K_theoretical_min"] = K_lb
        return out

    out = dict(df)
    out["K_theoretical_min"] = np.asarray(K_lb, dtype=float)
    return out


def _format_table_block(
    name: str,
    rows: List[Tuple[Any, ...]],
    header: Tuple[str, ...],
    col_widths: List[int],
) -> str:
    out = []
    out.append("=" * 90)
    out.append(f"TABLE: {name}")
    out.append("")
    head_line = " | ".join([f"{h:<{w}}" for h, w in zip(header, col_widths)])
    out.append(head_line)
    out.append("-" * len(head_line))
    for r in rows:
        out.append(" | ".join([f"{str(x):<{w}}" for x, w in zip(r, col_widths)]))
    out.append("")
    return "\n".join(out)


def _write_phaseB_tables_txt(dfB, out_dir: str, *, csv_path: str):
    path = os.path.join(out_dir, "phaseB_tables.txt")
    lines: List[str] = []
    lines.append(f"PHASE B TABLES (datapoints used in plots)\nCSV: {os.path.basename(csv_path)}\n")
    lines.append("Aggregation note: NaN values are excluded from means/min/max and from n. Downstream metrics for payload-infeasible runs are intentionally stored as NaN, not zero.")
    lines.append("")
    if _has_col(dfB, "theta_3db_deg"):
        theta_vals = np.unique(_col(dfB, "theta_3db_deg").astype(float, copy=False))
        theta_vals = theta_vals[~np.isnan(theta_vals)]
        if theta_vals.size > 0:
            vals = ", ".join([f"{v:.3f}" for v in theta_vals])
            lines.append(f"Beam model note: fixed half-beamwidth angle theta_3db_deg = [{vals}] deg")
            lines.append("Actual footprint radius is geometry-dependent: R_m = d_center * tan(theta_3db).")
            lines.append("")

    def add_table(name: str, series: Dict[str, str], ylabel: str):
        rows: List[Tuple[Any, ...]] = []
        for label, col in series.items():
            if not _has_col(dfB, col):
                continue
            agg = aggregate_by_group_mean_std(dfB, "n_users", col)
            for x, mu, sd, mn, mx, nn in zip(agg.x, agg.y_mean, agg.y_std, agg.y_min, agg.y_max, agg.n):
                rows.append((int(x), label, float(mu), float(sd), float(mn), float(mx), int(nn)))

        rows.sort(key=lambda r: (r[0], r[1]))
        header = ("n_users", "series", "mean", "std", "min", "max", "n")
        col_widths = [8, 14, 10, 10, 8, 8, 3]
        lines.append(_format_table_block(f"{name}  (ylabel={ylabel})", rows, header, col_widths))

    k_table_series = {
        "main+qos+lb": "main_ref_lb_K",
        "wk demand rep": "wk_demand_rep_K",
        "wk qos rep": "wk_qos_rep_K",
        "bk rep": "bk_rep_K",
        "tgbp rep": "tgbp_rep_K",
    }
    if _has_col(dfB, "payload_T_sum") and _has_col(dfB, "payload_W_slots"):
        dfB = _augment_with_k_theoretical_min(dfB)
        k_table_series = {"K theoretical min": "K_theoretical_min", **k_table_series}

    add_table(
        "K_vs_nusers",
        k_table_series,
        ylabel="K",
    )

    add_table(
        "ent_edge_pct_vs_nusers",
        {
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
        },
        ylabel="enterprise edge exposure (%)",
    )

    add_table(
        "Umax_vs_nusers",
        {
            "main": "main_U_max",
            "main+qos": "main_ref_U_max",
            "main+qos+lb": "main_ref_lb_U_max",
        },
        ylabel="U_max",
    )

    if _has_col(dfB, "payload_W_min_req"):
        add_table(
            "payload_Wmin_req_vs_nusers",
            {"W_min_req": "payload_W_min_req"},
            ylabel="payload_W_min_req",
        )

    if _has_col(dfB, "payload_T_sum") and _has_col(dfB, "payload_global_cap"):
        add_table(
            "payload_Tsum_vs_nusers",
            {"T_sum": "payload_T_sum", "global_cap": "payload_global_cap"},
            ylabel="payload_T_sum",
        )

    if _has_col(dfB, "payload_T_over_sum") or _has_col(dfB, "payload_K_over_sum"):
        series = {}
        if _has_col(dfB, "payload_K_over_sum"):
            series["K_over_sum"] = "payload_K_over_sum"
        if _has_col(dfB, "payload_T_over_sum"):
            series["T_over_sum"] = "payload_T_over_sum"
        add_table("payload_overflow_vs_nusers", series, ylabel="overflow (sum)")

    if _has_col(dfB, "main_ref_lb_radius_min_km"):
        add_table(
            "beam_radius_km_vs_nusers",
            {
                "beam r_min": "main_ref_lb_radius_min_km",
                "beam r_mean": "main_ref_lb_radius_mean_km",
                "beam r_p50": "main_ref_lb_radius_p50_km",
                "beam r_p90": "main_ref_lb_radius_p90_km",
                "beam r_max": "main_ref_lb_radius_max_km",
            },
            ylabel="beam radius (km)",
        )

    if _has_col(dfB, "main_ref_lb_sat_mean_radius_min_km"):
        add_table(
            "sat_mean_beam_radius_km_vs_nusers",
            {
                "sat mean r_min": "main_ref_lb_sat_mean_radius_min_km",
                "sat mean r_mean": "main_ref_lb_sat_mean_radius_mean_km",
                "sat mean r_max": "main_ref_lb_sat_mean_radius_max_km",
            },
            ylabel="satellite mean beam radius (km)",
        )

    if _has_col(dfB, "payload_n_viol_T") or _has_col(dfB, "payload_n_viol_K"):
        series = {}
        if _has_col(dfB, "payload_n_viol_T"):
            series["n_viol_T"] = "payload_n_viol_T"
        if _has_col(dfB, "payload_n_viol_K"):
            series["n_viol_K"] = "payload_n_viol_K"
        add_table("payload_nviol_vs_nusers", series, ylabel="#violating sats")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# Plots
# ----------------------------
def plot_total_runtime_vs_nusers(df, out_dir: str, *, show: bool = False, use_logy: bool = False) -> None:
    if not _has_col(df, "n_users"):
        raise ValueError("CSV must contain n_users column.")

    out_path = os.path.join(out_dir, "runtime_total_vs_nusers.png")

    my_parts = ["time_split_s", "time_ent_ref_s", "time_lb_ref_s"]
    present_parts = [c for c in my_parts if _has_col(df, c)]
    if not present_parts:
        raise ValueError("No internal timing columns found for main algorithm (time_*).")

    n = _nrows(df)
    my_total = np.zeros(n, dtype=float)
    for c in present_parts:
        my_total += _col(df, c).astype(float, copy=False)

    baselines = {
        "WKMeans++ (demand) total": "time_baseline_without_qos_s",
        "WKMeans++ (demand*qos) total": "time_baseline_with_qos_s",
        "BKMeans total": "time_baseline_bkmeans_s",
        "TGBP total": "time_baseline_tgbp_s",
    }

    fig = plt.figure()
    ax = plt.gca()

    if _has_col(df, "payload_feasible"):
        feasible = _col(df, "payload_feasible").astype(float, copy=False)
        bad = np.isnan(feasible) | (feasible < 0.5)
        if bad.shape[0] == my_total.shape[0]:
            my_total = my_total.astype(float, copy=False)
            my_total[bad] = np.nan

    tmp_my = {"n_users": _col(df, "n_users"), "my_total_s": my_total}
    agg = aggregate_by_group_mean_std(tmp_my, "n_users", "my_total_s")
    ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label="MY algorithm total")

    for label, colname in baselines.items():
        if not _has_col(df, colname):
            continue
        agg_b = aggregate_by_group_mean_std(df, "n_users", colname)
        if agg_b.x.size == 0:
            continue
        ax.plot(agg_b.x, agg_b.y_mean, marker="o", linestyle="-", label=label)

    ax.set_title("Total runtime vs n_users (MY algorithm vs baselines)")
    ax.set_xlabel("n_users")
    ax.set_ylabel("seconds")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if use_logy:
        ax.set_yscale("log")
    ax.legend(ncol=1, fontsize=9)
    _save_or_show(fig, out_path, show)


# ----------------------------
# Public API
# ----------------------------
def plot_phaseB(phaseB_csv: str, out_dir: str, *, show: bool = False) -> None:
    dfB = load_sweep_csv(phaseB_csv)
    _ensure_out_dir(out_dir)

    dfB = _augment_with_k_theoretical_min(dfB)
    _write_phaseB_tables_txt(dfB, out_dir, csv_path=phaseB_csv)

    k_plot_series = {
        "K theoretical min": "K_theoretical_min",
        "main+qos+lb": "main_ref_lb_K",
        "wk demand rep": "wk_demand_rep_K",
        "wk qos rep": "wk_qos_rep_K",
        "bk rep": "bk_rep_K",
        "tgbp rep": "tgbp_rep_K",
    } if _has_col(dfB, "K_theoretical_min") else {
        "main+qos+lb": "main_ref_lb_K",
        "wk demand rep": "wk_demand_rep_K",
        "wk qos rep": "wk_qos_rep_K",
        "bk rep": "bk_rep_K",
        "tgbp rep": "tgbp_rep_K",
    }

    plot_lines_vs_nusers(
        dfB,
        series=k_plot_series,
        title="Beams K vs n_users",
        ylabel="K",
        out_path=os.path.join(out_dir, "K_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
        },
        title="Enterprise edge exposure (%) vs n_users",
        ylabel="exposed enterprise (%)",
        out_path=os.path.join(out_dir, "ent_exposure_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB,
        series={
            "main": "main_U_max",
            "main+qos": "main_ref_U_max",
            "main+qos+lb": "main_ref_lb_U_max",
        },
        title="Peak utilization U_max vs n_users",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "Umax_vs_nusers.png"),
        show=show,
    )

    if _has_col(dfB, "main_ref_lb_radius_min_km"):
        plot_lines_vs_nusers(
            dfB,
            series={
                "beam r_min": "main_ref_lb_radius_min_km",
                "beam r_mean": "main_ref_lb_radius_mean_km",
                "beam r_max": "main_ref_lb_radius_max_km",
            },
            title="Beam radius min / mean / max vs n_users",
            ylabel="beam radius (km)",
            out_path=os.path.join(out_dir, "beam_radius_min_mean_max_vs_nusers.png"),
            show=show,
        )

        plot_lines_vs_nusers(
            dfB,
            series={
                "beam r_p10": "main_ref_lb_radius_p10_km",
                "beam r_p50": "main_ref_lb_radius_p50_km",
                "beam r_p90": "main_ref_lb_radius_p90_km",
            },
            title="Beam radius percentiles vs n_users",
            ylabel="beam radius (km)",
            out_path=os.path.join(out_dir, "beam_radius_percentiles_vs_nusers.png"),
            show=show,
        )

    if _has_col(dfB, "main_ref_lb_sat_mean_radius_min_km"):
        plot_lines_vs_nusers(
            dfB,
            series={
                "sat mean r_min": "main_ref_lb_sat_mean_radius_min_km",
                "sat mean r_mean": "main_ref_lb_sat_mean_radius_mean_km",
                "sat mean r_max": "main_ref_lb_sat_mean_radius_max_km",
            },
            title="Across-satellite mean beam radius range vs n_users",
            ylabel="satellite mean beam radius (km)",
            out_path=os.path.join(out_dir, "sat_mean_beam_radius_range_vs_nusers.png"),
            show=show,
        )

    plot_total_runtime_vs_nusers(dfB, out_dir, show=show, use_logy=False)

    # Payload plots
    if _has_col(dfB, "payload_feasible"):
        plot_lines_vs_nusers(
            dfB,
            series={"payload_feasible_rate": "payload_feasible"},
            title="Payload feasibility rate vs n_users",
            ylabel="feasible (0/1)",
            out_path=os.path.join(out_dir, "payload_feasible_rate_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_best_m"):
        plot_lines_vs_nusers(
            dfB,
            series={"best_m": "payload_best_m"},
            title="Minimum satellites needed (best prefix m) vs n_users",
            ylabel="m",
            out_path=os.path.join(out_dir, "payload_best_m_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

        plot_lines_vs_nusers(
            dfB,
            series={"avg satellites used": "payload_best_m"},
            title="Average satellites used vs n_users",
            ylabel="satellites",
            out_path=os.path.join(out_dir, "avg_satellites_used_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_W_min_req"):
        plot_lines_vs_nusers(
            dfB,
            series={"W_min_req": "payload_W_min_req"},
            title="Required hopping window (W_min_req) vs n_users",
            ylabel="W_min_req (slots)",
            out_path=os.path.join(out_dir, "payload_Wmin_req_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_T_sum") and _has_col(dfB, "payload_global_cap"):
        plot_lines_vs_nusers(
            dfB,
            series={"T_sum": "payload_T_sum", "global_cap": "payload_global_cap"},
            title="Total required time vs total payload capacity",
            ylabel="time units (normalized)",
            out_path=os.path.join(out_dir, "payload_Tsum_vs_globalcap.png"),
            show=show,
            include_errorbars=False,
        )
        plot_payload_ratio_vs_nusers(
            dfB,
            title="Global load ratio (T_sum / global_cap) vs n_users",
            out_path=os.path.join(out_dir, "payload_load_ratio_vs_nusers.png"),
            show=show,
        )

    if _has_col(dfB, "payload_T_max") and _has_col(dfB, "payload_T_cap"):
        tmp = {
            "n_users": _col(dfB, "n_users"),
            "Tmax_over_Tcap": (
                _col(dfB, "payload_T_max").astype(float, copy=False)
                / np.maximum(_col(dfB, "payload_T_cap").astype(float, copy=False), 1e-12)
            ),
        }
        plot_lines_vs_nusers(
            tmp,
            series={"T_max / T_cap": "Tmax_over_Tcap"},
            title="Normalized time bottleneck (T_max / T_cap) vs n_users",
            ylabel="ratio",
            out_path=os.path.join(out_dir, "payload_Tmax_over_Tcap_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_K_max") and _has_col(dfB, "payload_K_cap"):
        tmp = {
            "n_users": _col(dfB, "n_users"),
            "Kmax_over_Kcap": (
                _col(dfB, "payload_K_max").astype(float, copy=False)
                / np.maximum(_col(dfB, "payload_K_cap").astype(float, copy=False), 1e-12)
            ),
        }
        plot_lines_vs_nusers(
            tmp,
            series={"K_max / K_cap": "Kmax_over_Kcap"},
            title="Normalized beam-count bottleneck (K_max / K_cap) vs n_users",
            ylabel="ratio",
            out_path=os.path.join(out_dir, "payload_Kmax_over_Kcap_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_W_min_req") and _has_col(dfB, "payload_W_slots"):
        tmp = {
            "n_users": _col(dfB, "n_users"),
            "W_gap": (
                _col(dfB, "payload_W_min_req").astype(float, copy=False)
                - _col(dfB, "payload_W_slots").astype(float, copy=False)
            ),
        }
        plot_lines_vs_nusers(
            tmp,
            series={"W_min_req - W": "W_gap"},
            title="Extra slots needed (W_min_req - W) vs n_users",
            ylabel="slots",
            out_path=os.path.join(out_dir, "payload_Wgap_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    # Repair effort curves
    if _has_col(dfB, "payload_moves_tried") and _has_col(dfB, "payload_moves_accepted"):
        plot_lines_vs_nusers(
            dfB,
            series={"moves_tried": "payload_moves_tried", "moves_accepted": "payload_moves_accepted"},
            title="Payload repair effort vs n_users",
            ylabel="moves",
            out_path=os.path.join(out_dir, "payload_repair_moves_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_moves_accepted_K") or _has_col(dfB, "payload_moves_accepted_T") or _has_col(dfB, "payload_moves_accepted_smooth"):
        series = {}
        if _has_col(dfB, "payload_moves_accepted_K"):
            series["accepted_K"] = "payload_moves_accepted_K"
        if _has_col(dfB, "payload_moves_accepted_T"):
            series["accepted_T"] = "payload_moves_accepted_T"
        if _has_col(dfB, "payload_moves_accepted_smooth"):
            series["accepted_smooth"] = "payload_moves_accepted_smooth"
        plot_lines_vs_nusers(
            dfB,
            series=series,
            title="Accepted payload repair moves by type vs n_users",
            ylabel="accepted moves",
            out_path=os.path.join(out_dir, "payload_repair_moves_by_type_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_T_over_sum") or _has_col(dfB, "payload_K_over_sum"):
        series = {}
        if _has_col(dfB, "payload_T_over_sum"):
            series["T_over_sum"] = "payload_T_over_sum"
        if _has_col(dfB, "payload_K_over_sum"):
            series["K_over_sum"] = "payload_K_over_sum"
        plot_lines_vs_nusers(
            dfB,
            series=series,
            title="Payload overflow severity (sum) vs n_users",
            ylabel="overflow (sum across sats)",
            out_path=os.path.join(out_dir, "payload_overflow_sum_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    if _has_col(dfB, "payload_n_viol_T") or _has_col(dfB, "payload_n_viol_K"):
        series = {}
        if _has_col(dfB, "payload_n_viol_T"):
            series["n_viol_T"] = "payload_n_viol_T"
        if _has_col(dfB, "payload_n_viol_K"):
            series["n_viol_K"] = "payload_n_viol_K"
        plot_lines_vs_nusers(
            dfB,
            series=series,
            title="Count of violating satellites vs n_users",
            ylabel="violating sats (mean)",
            out_path=os.path.join(out_dir, "payload_viol_counts_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )


def plot_phaseA(phaseA_csv: str, out_dir: str, *, show: bool = False) -> None:
    dfA = load_sweep_csv(phaseA_csv)
    _ensure_out_dir(out_dir)

    metrics = [
        "main_K", "main_ref_K", "main_ref_lb_K",
        "main_ent_edge_pct", "main_ref_ent_edge_pct", "main_ref_lb_ent_edge_pct",
        "main_U_max", "main_ref_U_max", "main_ref_lb_U_max",
        "main_U_mean", "main_ref_U_mean", "main_ref_lb_U_mean",
    ]

    for m in metrics:
        if not _has_col(dfA, m):
            continue

        plot_lines_vs_nusers(
            dfA,
            series={m: m},
            title=f"{m} vs n_users (Phase A)",
            ylabel=m,
            out_path=os.path.join(out_dir, f"{m}_vs_nusers.png"),
            show=show,
        )


def plot_all(phaseA_csv: str, phaseB_csv: str, out_dir: str = "plots", *, show: bool = False) -> None:
    """
    Compatibility helper: main.py imports plot_all().
    """
    plot_phaseA(phaseA_csv, os.path.join(out_dir, "phaseA"), show=show)
    plot_phaseB(phaseB_csv, os.path.join(out_dir, "phaseB"), show=show)