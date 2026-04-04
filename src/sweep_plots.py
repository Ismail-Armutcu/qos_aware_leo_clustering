
# src/sweep_plots.py
from __future__ import annotations

import argparse
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


def aggregate_by_group_mean_std(df, xcol: str, ycol: str) -> AggSeries:
    if not (_has_col(df, xcol) and _has_col(df, ycol)):
        return AggSeries(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    x = _col(df, xcol).astype(float, copy=False)
    y = _col(df, ycol).astype(float, copy=False)

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
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    elif show:
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
    y_log: bool = False,
) -> None:
    plotted = False
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()

    for label, col in series.items():
        if not _has_col(df, col):
            continue
        agg = aggregate_by_group_mean_std(df, "n_users", col)
        if agg.x.size == 0:
            continue
        plotted = True
        if include_errorbars:
            ax.errorbar(
                agg.x, agg.y_mean, yerr=agg.y_std,
                marker="o", linestyle="-", capsize=3, label=label
            )
        else:
            ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label=label)

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("n_users")
    ax.set_ylabel(ylabel)
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9)
    _save_or_show(fig, out_path, show)


def plot_grouped_bars_vs_nusers(
    df,
    series: Dict[str, str],
    *,
    title: str,
    ylabel: str,
    out_path: str | None = None,
    show: bool = False,
) -> None:
    available = [(label, col) for label, col in series.items() if _has_col(df, col)]
    if not available:
        return

    xs = None
    aggs: List[Tuple[str, AggSeries]] = []
    for label, col in available:
        agg = aggregate_by_group_mean_std(df, "n_users", col)
        if agg.x.size == 0:
            continue
        aggs.append((label, agg))
        if xs is None:
            xs = agg.x
    if not aggs or xs is None:
        return

    fig = plt.figure(figsize=(7.5, 4.8))
    ax = plt.gca()

    idx = np.arange(len(xs))
    width = 0.8 / max(1, len(aggs))

    for i, (label, agg) in enumerate(aggs):
        ax.bar(idx + i * width - 0.4 + width / 2.0, agg.y_mean, width=width, label=label)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_xlabel("n_users")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9)
    _save_or_show(fig, out_path, show)


def _series_table_rows(df, table_name: str, series: Dict[str, str]) -> List[Tuple[Any, ...]]:
    rows: List[Tuple[Any, ...]] = []
    for label, col in series.items():
        if not _has_col(df, col):
            continue
        agg = aggregate_by_group_mean_std(df, "n_users", col)
        for i, xv in enumerate(agg.x):
            rows.append(
                (
                    int(xv),
                    label,
                    float(agg.y_mean[i]) if np.isfinite(agg.y_mean[i]) else "nan",
                    float(agg.y_std[i]) if np.isfinite(agg.y_std[i]) else "nan",
                    float(agg.y_min[i]) if np.isfinite(agg.y_min[i]) else "nan",
                    float(agg.y_max[i]) if np.isfinite(agg.y_max[i]) else "nan",
                    int(agg.n[i]),
                )
            )
    rows.sort(key=lambda r: (r[0], str(r[1])))
    return rows


def _format_table_block(
    name: str,
    rows: List[Tuple[Any, ...]],
    header: Tuple[str, ...] = ("n_users", "series", "mean", "std", "min", "max", "n"),
) -> str:
    widths = [8, 20, 12, 12, 10, 10, 3]
    out: List[str] = []
    out.append("=" * 90)
    out.append(f"TABLE: {name}")
    out.append("")
    head = " | ".join(f"{h:<{w}}" for h, w in zip(header, widths))
    out.append(head)
    out.append("-" * len(head))
    for r in rows:
        out.append(" | ".join(f"{str(v):<{w}}" for v, w in zip(r, widths)))
    out.append("")
    return "\n".join(out)


def _augment_with_k_theoretical_min(df):
    if not (_has_col(df, "payload_T_sum") and _has_col(df, "payload_W_slots")):
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


def _main_total_runtime_col(df) -> str | None:
    parts = ["time_assoc_s", "time_split_s", "time_ent_ref_s", "time_lb_ref_s"]
    if not all(_has_col(df, c) for c in parts):
        return None
    vals = sum(_col(df, c).astype(float, copy=False) for c in parts)
    name = "_derived_time_main_total_s"
    if _is_pandas_df(df):
        df[name] = vals
    else:
        df[name] = np.asarray(vals, dtype=float)
    return name


def _write_phaseB_tables_txt(dfB, out_dir: str, *, csv_path: str) -> None:
    path = os.path.join(out_dir, "phaseB_tables.txt")
    lines: List[str] = []
    lines.append(f"PHASE B TABLES (datapoints used in plots)\nCSV: {os.path.basename(csv_path)}\n")
    lines.append("Aggregation note: NaN values are excluded from means/min/max and from n.\n")
    lines.append("This version keeps only the thesis-useful tables and removes low-value/redundant plot summaries.\n")

    # External beam-placement comparisons
    k_series = {
        "K theoretical min": "K_theoretical_min",
        "main+qos+lb": "main_ref_lb_K",
        "bk rep": "bk_rep_K",
        "tgbp rep": "tgbp_rep_K",
        "wk demand rep": "wk_demand_rep_K",
        "wk qos rep": "wk_qos_rep_K",
    }
    lines.append(_format_table_block("K_vs_nusers", _series_table_rows(dfB, "K_vs_nusers", k_series)))

    # Rigid fixed-prefix ablation tables
    ab_feas = {
        "A0 pure+split": "ab_A0_payload_feasible",
        "A1 bal+split": "ab_A1_payload_feasible",
        "A2 bal+split+qos": "ab_A2_payload_feasible",
        "A3 bal+split+qos+lb": "ab_A3_payload_feasible",
    }
    lines.append(_format_table_block("ablation_feasible_rate_vs_nusers", _series_table_rows(dfB, "ablation_feasible_rate_vs_nusers", ab_feas)))

    ab_k = {
        "A0 pure+split": "ab_A0_pure_split_K",
        "A1 bal+split": "ab_A1_bal_split_K",
        "A2 bal+split+qos": "ab_A2_bal_split_qos_K",
        "A3 bal+split+qos+lb": "ab_A3_bal_split_qos_lb_K",
    }
    lines.append(_format_table_block("ablation_K_vs_nusers", _series_table_rows(dfB, "ablation_K_vs_nusers", ab_k)))

    ab_umax = {
        "A0 pure+split": "ab_A0_pure_split_U_max",
        "A1 bal+split": "ab_A1_bal_split_U_max",
        "A2 bal+split+qos": "ab_A2_bal_split_qos_U_max",
        "A3 bal+split+qos+lb": "ab_A3_bal_split_qos_lb_U_max",
    }
    lines.append(_format_table_block("ablation_Umax_vs_nusers", _series_table_rows(dfB, "ablation_Umax_vs_nusers", ab_umax)))

    ab_rt = {
        "A0 pure+split": "time_ab_A0_s",
        "A1 bal+split": "time_ab_A1_s",
        "A2 bal+split+qos": "time_ab_A2_s",
        "A3 bal+split+qos+lb": "time_ab_A3_s",
    }
    lines.append(_format_table_block("ablation_runtime_vs_nusers", _series_table_rows(dfB, "ablation_runtime_vs_nusers", ab_rt)))

    # Enterprise exposure
    ent_series = {
        "main+qos+lb": "main_ref_lb_ent_edge_pct",
        "wk demand rep": "wk_demand_rep_ent_edge_pct",
        "wk qos rep": "wk_qos_rep_ent_edge_pct",
    }
    lines.append(_format_table_block("ent_edge_pct_vs_nusers", _series_table_rows(dfB, "ent_edge_pct_vs_nusers", ent_series)))

    # System-level association comparison
    sys_k = {
        "pure max elev": "sys_pure_max_elev_K",
        "balanced max elev": "sys_balanced_max_elev_K",
        "max service time": "sys_max_service_time_K",
    }
    lines.append(_format_table_block("system_assoc_K_vs_nusers", _series_table_rows(dfB, "system_assoc_K_vs_nusers", sys_k)))

    sys_umax = {
        "pure max elev": "sys_pure_max_elev_U_max",
        "balanced max elev": "sys_balanced_max_elev_U_max",
        "max service time": "sys_max_service_time_U_max",
    }
    lines.append(_format_table_block("system_assoc_Umax_vs_nusers", _series_table_rows(dfB, "system_assoc_Umax_vs_nusers", sys_umax)))

    sys_feas = {
        "pure max elev": "sys_pure_max_elev_payload_feasible",
        "balanced max elev": "sys_balanced_max_elev_payload_feasible",
        "max service time": "sys_max_service_time_payload_feasible",
    }
    lines.append(_format_table_block("system_assoc_feasible_rate_vs_nusers", _series_table_rows(dfB, "system_assoc_feasible_rate_vs_nusers", sys_feas)))

    # Payload
    payload_over = {
        "K_over_sum": "payload_K_over_sum",
        "T_over_sum": "payload_T_over_sum",
    }
    lines.append(_format_table_block("payload_overflow_vs_nusers", _series_table_rows(dfB, "payload_overflow_vs_nusers", payload_over)))

    payload_nviol = {
        "n_viol_K": "payload_n_viol_K",
        "n_viol_T": "payload_n_viol_T",
    }
    lines.append(_format_table_block("payload_nviol_vs_nusers", _series_table_rows(dfB, "payload_nviol_vs_nusers", payload_nviol)))

    beam_r = {
        "beam r_mean": "main_ref_lb_radius_mean_km",
        "beam r_p90": "main_ref_lb_radius_p90_km",
        "sat mean r_mean": "main_ref_lb_sat_mean_radius_mean_km",
    }
    lines.append(_format_table_block("beam_radius_km_vs_nusers", _series_table_rows(dfB, "beam_radius_km_vs_nusers", beam_r)))

    rt_series = {
        "main total": "_derived_time_main_total_s",
        "wk demand rep": "time_baseline_without_qos_s",
        "wk qos rep": "time_baseline_with_qos_s",
        "bk rep": "time_baseline_bkmeans_s",
        "tgbp rep": "time_baseline_tgbp_s",
    }
    lines.append(_format_table_block("runtime_vs_nusers", _series_table_rows(dfB, "runtime_vs_nusers", rt_series)))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_phaseB_plots(csv_path: str, out_dir: str, *, show: bool = False) -> None:
    _ensure_out_dir(out_dir)
    df = load_sweep_csv(csv_path)
    df = _augment_with_k_theoretical_min(df)
    _main_total_runtime_col(df)

    # 1) Main external comparison: beam count
    plot_lines_vs_nusers(
        df,
        {
            "main+qos+lb": "main_ref_lb_K",
            "bk rep": "bk_rep_K",
            "tgbp rep": "tgbp_rep_K",
            "wk demand rep": "wk_demand_rep_K",
            "wk qos rep": "wk_qos_rep_K",
        },
        title="Beam count vs user count",
        ylabel="K",
        out_path=os.path.join(out_dir, "K_vs_nusers.png"),
        show=show,
    )

    # 2) External comparison: Umax
    plot_lines_vs_nusers(
        df,
        {
            "main+qos+lb": "main_ref_lb_U_max",
            "bk rep": "bk_rep_U_max",
            "tgbp rep": "tgbp_rep_U_max",
            "wk demand rep": "wk_demand_rep_U_max",
            "wk qos rep": "wk_qos_rep_U_max",
        },
        title="Peak utilization vs user count",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "Umax_vs_nusers.png"),
        show=show,
    )

    # 3) External comparison: enterprise edge exposure
    plot_lines_vs_nusers(
        df,
        {
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
        },
        title="Enterprise edge exposure vs user count",
        ylabel="enterprise edge exposure (%)",
        out_path=os.path.join(out_dir, "ent_edge_pct_vs_nusers.png"),
        show=show,
    )

    # 4) Rigid fixed-prefix ablation: feasible rate
    plot_lines_vs_nusers(
        df,
        {
            "A0 pure+split": "ab_A0_payload_feasible",
            "A1 bal+split": "ab_A1_payload_feasible",
            "A2 bal+split+qos": "ab_A2_payload_feasible",
            "A3 bal+split+qos+lb": "ab_A3_payload_feasible",
        },
        title="Ablation: feasible rate under fixed prefix",
        ylabel="feasible rate",
        out_path=os.path.join(out_dir, "ablation_feasible_rate_vs_nusers.png"),
        show=show,
    )

    # 5) Rigid fixed-prefix ablation: beam count
    plot_lines_vs_nusers(
        df,
        {
            "A0 pure+split": "ab_A0_pure_split_K",
            "A1 bal+split": "ab_A1_bal_split_K",
            "A2 bal+split+qos": "ab_A2_bal_split_qos_K",
            "A3 bal+split+qos+lb": "ab_A3_bal_split_qos_lb_K",
        },
        title="Ablation: beam count under fixed prefix",
        ylabel="K",
        out_path=os.path.join(out_dir, "ablation_K_vs_nusers.png"),
        show=show
    )

    # 6) Rigid fixed-prefix ablation: Umax
    plot_lines_vs_nusers(
        df,
        {
            "A0 pure+split": "ab_A0_pure_split_U_max",
            "A1 bal+split": "ab_A1_bal_split_U_max",
            "A2 bal+split+qos": "ab_A2_bal_split_qos_U_max",
            "A3 bal+split+qos+lb": "ab_A3_bal_split_qos_lb_U_max",
        },
        title="Ablation: peak utilization under fixed prefix",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "ablation_Umax_vs_nusers.png"),
        show=show,
    )

    # 7) Rigid fixed-prefix ablation: runtime
    plot_lines_vs_nusers(
        df,
        {
            "A0 pure+split": "time_ab_A0_s",
            "A1 bal+split": "time_ab_A1_s",
            "A2 bal+split+qos": "time_ab_A2_s",
            "A3 bal+split+qos+lb": "time_ab_A3_s",
        },
        title="Ablation: runtime under fixed prefix",
        ylabel="runtime (s)",
        out_path=os.path.join(out_dir, "ablation_runtime_vs_nusers.png"),
        show=show,
        y_log=True,
    )

    # 8) System-level association comparison: K
    plot_lines_vs_nusers(
        df,
        {
            "pure max elev": "sys_pure_max_elev_K",
            "balanced max elev": "sys_balanced_max_elev_K",
            "max service time": "sys_max_service_time_K",
        },
        title="System-level comparison: beam count under fixed prefix",
        ylabel="K",
        out_path=os.path.join(out_dir, "system_assoc_K_vs_nusers.png"),
        show=show,
    )

    # 6) System-level association comparison: Umax
    plot_lines_vs_nusers(
        df,
        {
            "pure max elev": "sys_pure_max_elev_U_max",
            "balanced max elev": "sys_balanced_max_elev_U_max",
            "max service time": "sys_max_service_time_U_max",
        },
        title="System-level comparison: peak utilization under fixed prefix",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "system_assoc_Umax_vs_nusers.png"),
        show=show,
    )

    # 7) System-level association comparison: feasible rate
    plot_lines_vs_nusers(
        df,
        {
            "pure max elev": "sys_pure_max_elev_payload_feasible",
            "balanced max elev": "sys_balanced_max_elev_payload_feasible",
            "max service time": "sys_max_service_time_payload_feasible",
        },
        title="System-level comparison: feasible rate under fixed prefix",
        ylabel="feasible rate",
        out_path=os.path.join(out_dir, "system_assoc_feasible_rate_vs_nusers.png"),
        show=show,
    )

    # 8) Payload violations
    plot_grouped_bars_vs_nusers(
        df,
        {
            "# violating sats (K)": "payload_n_viol_K",
            "# violating sats (T)": "payload_n_viol_T",
        },
        title="Payload-violating satellites",
        ylabel="# violating satellites",
        out_path=os.path.join(out_dir, "payload_nviol_vs_nusers.png"),
        show=show,
    )

    # 8) Payload overflow
    plot_lines_vs_nusers(
        df,
        {
            "K_over_sum": "payload_K_over_sum",
            "T_over_sum": "payload_T_over_sum",
        },
        title="Payload overflow vs user count",
        ylabel="overflow (sum)",
        out_path=os.path.join(out_dir, "payload_overflow_vs_nusers.png"),
        show=show,
    )

    # 9) Beam radius summary
    plot_lines_vs_nusers(
        df,
        {
            "beam r_mean": "main_ref_lb_radius_mean_km",
            "beam r_p90": "main_ref_lb_radius_p90_km",
            "sat mean r_mean": "main_ref_lb_sat_mean_radius_mean_km",
        },
        title="Beam radius summary",
        ylabel="beam radius (km)",
        out_path=os.path.join(out_dir, "beam_radius_summary_vs_nusers.png"),
        show=show,
    )

    # 10) Runtime
    plot_lines_vs_nusers(
        df,
        {
            "main total": "_derived_time_main_total_s",
            "wk demand rep": "time_baseline_without_qos_s",
            "wk qos rep": "time_baseline_with_qos_s",
            "bk rep": "time_baseline_bkmeans_s",
            "tgbp rep": "time_baseline_tgbp_s",
        },
        title="Runtime vs user count",
        ylabel="runtime (s)",
        out_path=os.path.join(out_dir, "runtime_vs_nusers.png"),
        show=show,
        y_log=True,
    )

    _write_phaseB_tables_txt(df, out_dir, csv_path=csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concise sweep plots/tables for Phase B.")
    parser.add_argument("--csv", default="sweep_phaseB.csv", help="Input CSV path")
    parser.add_argument("--out", default="plots_phaseB", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plots instead of only saving")
    args = parser.parse_args()

    make_phaseB_plots(args.csv, args.out, show=bool(args.show))
    print(f"Wrote concise Phase B plots/tables to: {args.out}")


if __name__ == "__main__":
    main()
