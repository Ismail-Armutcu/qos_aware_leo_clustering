# src/sweep_plots.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional pandas (nice-to-have). If unavailable, we fall back to csv module.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ----------------------------
# CSV loading (pandas or fallback)
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
    """
    Returns either:
      - pandas.DataFrame (if pandas installed), or
      - dict[str, np.ndarray] fallback
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    if pd is not None:
        return pd.read_csv(path)

    # Fallback
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
        if all((isinstance(v, (int, float, np.floating, bool)) or (v is np.nan)) for v in vals):
            # bools become 0/1 if cast to float downstream
            out[c] = np.asarray(vals, dtype=float)
        else:
            out[c] = np.asarray(vals, dtype=object)
    return out


def _has_col(df, col: str) -> bool:
    if pd is not None and hasattr(df, "columns"):
        return col in df.columns
    return isinstance(df, dict) and col in df


def _col(df, col: str) -> np.ndarray:
    if pd is not None and hasattr(df, "columns"):
        return df[col].to_numpy()
    return np.asarray(df[col])


def _nrows(df) -> int:
    if pd is not None and hasattr(df, "shape"):
        return int(df.shape[0])
    if isinstance(df, dict) and df:
        k0 = next(iter(df.keys()))
        return int(len(df[k0]))
    return 0


def _is_pandas_df(df) -> bool:
    return pd is not None and hasattr(df, "columns") and hasattr(df, "to_numpy")


# ----------------------------
# Aggregation helpers
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
    """
    Group by xcol and compute mean/std/min/max for ycol.
    Ignores NaNs in ycol.
    """
    if not _has_col(df, xcol) or not _has_col(df, ycol):
        return AggSeries(
            x=np.asarray([], dtype=float),
            y_mean=np.asarray([], dtype=float),
            y_std=np.asarray([], dtype=float),
            y_min=np.asarray([], dtype=float),
            y_max=np.asarray([], dtype=float),
            n=np.asarray([], dtype=int),
        )

    x = _col(df, xcol)
    y = _col(df, ycol)

    # Force numeric when possible; booleans become 0/1
    try:
        y = y.astype(float, copy=False)
    except Exception:
        # Non-numeric -> no series
        return AggSeries(
            x=np.asarray([], dtype=float),
            y_mean=np.asarray([], dtype=float),
            y_std=np.asarray([], dtype=float),
            y_min=np.asarray([], dtype=float),
            y_max=np.asarray([], dtype=float),
            n=np.asarray([], dtype=int),
        )

    mask = ~np.isnan(y)
    # if x numeric, also drop NaN x
    try:
        x_num = x.astype(float, copy=False)
        mask &= ~np.isnan(x_num)
        x = x_num
    except Exception:
        pass

    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return AggSeries(
            x=np.asarray([], dtype=float),
            y_mean=np.asarray([], dtype=float),
            y_std=np.asarray([], dtype=float),
            y_min=np.asarray([], dtype=float),
            y_max=np.asarray([], dtype=float),
            n=np.asarray([], dtype=int),
        )

    xuniq = sorted(set(x.tolist()))
    means, stds, mins, maxs, ns = [], [], [], [], []

    for xv in xuniq:
        m = (x == xv)
        yy = y[m]
        if yy.size == 0:
            continue
        means.append(float(np.mean(yy)))
        stds.append(float(np.std(yy)))
        mins.append(float(np.min(yy)))
        maxs.append(float(np.max(yy)))
        ns.append(int(yy.size))

    return AggSeries(
        x=np.asarray([float(v) for v in xuniq], dtype=float),
        y_mean=np.asarray(means, dtype=float),
        y_std=np.asarray(stds, dtype=float),
        y_min=np.asarray(mins, dtype=float),
        y_max=np.asarray(maxs, dtype=float),
        n=np.asarray(ns, dtype=int),
    )


def _unique_sorted_nusers(df) -> np.ndarray:
    """Return sorted unique n_users values as ints."""
    if not _has_col(df, "n_users"):
        return np.asarray([], dtype=int)
    x = _col(df, "n_users").astype(float, copy=False)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.asarray([], dtype=int)
    xi = np.asarray([int(round(v)) for v in x.tolist()], dtype=int)
    return np.unique(xi)


# ----------------------------
# Payload masking utilities
# ----------------------------
def _payload_feasible_mask(df, feas_col: str = "payload_feasible") -> np.ndarray | None:
    if not _has_col(df, feas_col):
        return None
    v = _col(df, feas_col)
    # Accept bool, int, float; treat >0.5 as True
    try:
        vv = v.astype(float, copy=False)
        return vv > 0.5
    except Exception:
        # object: try string "True"
        vv = np.array([str(x).strip().lower() for x in v], dtype=object)
        return vv == "true"


def mask_infeasible_as_nan(df, cols: Sequence[str], feas_col: str = "payload_feasible"):
    """
    For columns that are only meaningful when payload is feasible (refinements/baselines),
    set values to NaN when payload_feasible == False, so aggregations ignore them.
    """
    m = _payload_feasible_mask(df, feas_col=feas_col)
    if m is None:
        return df  # nothing to do

    if _is_pandas_df(df):
        out = df.copy()
        inv = ~m
        for c in cols:
            if c in out.columns:
                try:
                    arr = out[c].to_numpy()
                    arr = arr.astype(float, copy=True)
                    arr[inv] = np.nan
                    out[c] = arr
                except Exception:
                    pass
        return out

    # dict fallback
    if isinstance(df, dict):
        out: Dict[str, np.ndarray] = dict(df)
        inv = ~m
        for c in cols:
            if c in out:
                try:
                    arr = np.asarray(out[c]).astype(float, copy=True)
                    arr[inv] = np.nan
                    out[c] = arr
                except Exception:
                    pass
        return out

    return df


def filter_payload_feasible_only(df, feas_col: str = "payload_feasible"):
    """
    Return a filtered df with only payload-feasible rows.
    Useful for runtime comparisons (baselines not run on infeasible).
    """
    m = _payload_feasible_mask(df, feas_col=feas_col)
    if m is None:
        return df

    if _is_pandas_df(df):
        return df.loc[m].copy()

    if isinstance(df, dict):
        out: Dict[str, np.ndarray] = {}
        for k, v in df.items():
            out[k] = np.asarray(v)[m]
        return out

    return df


# ----------------------------
# Text table helpers
# ----------------------------
def _fmt_num(v: Any, *, digits: int = 6) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return f"{float(v):.{digits}g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)
    except Exception:
        return str(v)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Simple fixed-width ASCII table."""
    cols = len(headers)
    str_rows: List[List[str]] = []
    for r in rows:
        rr = list(r)
        if len(rr) != cols:
            raise ValueError("Row length does not match headers.")
        str_rows.append([str(x) for x in rr])

    widths = [len(h) for h in headers]
    for r in str_rows:
        for j in range(cols):
            widths[j] = max(widths[j], len(r[j]))

    def fmt_row(r: Sequence[str]) -> str:
        return " | ".join(r[j].ljust(widths[j]) for j in range(cols))

    sep = "-+-".join("-" * w for w in widths)
    out = []
    out.append(fmt_row(list(headers)))
    out.append(sep)
    for r in str_rows:
        out.append(fmt_row(r))
    return "\n".join(out) + "\n"


def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _write_phaseB_tables_txt(dfB, out_dir: str, *, csv_path: str) -> str:
    """
    Writes all Phase-B plot datapoints (mean-per-n_users-per-series) to one text file.
    Returns output path.
    """
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "phaseB_tables.txt")

    if not _has_col(dfB, "n_users"):
        raise ValueError("CSV missing 'n_users'.")

    nusers = _unique_sorted_nusers(dfB)
    if nusers.size == 0:
        raise ValueError("No valid n_users in CSV.")

    # Columns that are invalid when payload is infeasible (refinements + baselines)
    masked_cols = [
        # main refined
        "main_ref_K", "main_ref_lb_K",
        "main_ref_ent_edge_pct", "main_ref_lb_ent_edge_pct",
        "main_ref_U_mean", "main_ref_lb_U_mean",
        "main_ref_U_max", "main_ref_lb_U_max",
        "main_ref_risk_sum", "main_ref_lb_risk_sum",
        # baselines
        "wk_demand_rep_K", "wk_qos_rep_K", "bk_rep_K", "tgbp_rep_K",
        "wk_demand_rep_ent_edge_pct", "wk_qos_rep_ent_edge_pct", "bk_rep_ent_edge_pct", "tgbp_rep_ent_edge_pct",
        "wk_demand_rep_U_mean", "wk_qos_rep_U_mean", "bk_rep_U_mean", "tgbp_rep_U_mean",
        "wk_demand_rep_U_max", "wk_qos_rep_U_max", "bk_rep_U_max", "tgbp_rep_U_max",
        "wk_demand_rep_risk_sum", "wk_qos_rep_risk_sum", "bk_rep_risk_sum", "tgbp_rep_risk_sum",
        # runtime baselines
        "time_baseline_without_qos_s", "time_baseline_with_qos_s", "time_baseline_bkmeans_s", "time_baseline_tgbp_s",
        "time_ent_ref_s", "time_lb_ref_s",
    ]
    dfB_masked = mask_infeasible_as_nan(dfB, masked_cols)

    # Define Phase-B plotted series (must match plot_phaseB)
    metric_series: List[Tuple[str, str, Dict[str, str]]] = [
        ("K_vs_nusers", "K", {
            "main": "main_K",
            "main+qos": "main_ref_K",
            "main+qos+lb": "main_ref_lb_K",
            "wk demand rep": "wk_demand_rep_K",
            "wk qos rep": "wk_qos_rep_K",
            "bk rep": "bk_rep_K",
            "tgbp rep": "tgbp_rep_K",
        }),
        ("ent_edge_pct_vs_nusers", "enterprise edge exposure (%)", {
            "main": "main_ent_edge_pct",
            "main+qos": "main_ref_ent_edge_pct",
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
            "bk rep": "bk_rep_ent_edge_pct",
            "tgbp rep": "tgbp_rep_ent_edge_pct",
        }),
        ("Umean_vs_nusers", "U_mean", {
            "main": "main_U_mean",
            "main+qos": "main_ref_U_mean",
            "main+qos+lb": "main_ref_lb_U_mean",
            "wk demand rep": "wk_demand_rep_U_mean",
            "wk qos rep": "wk_qos_rep_U_mean",
            "bk rep": "bk_rep_U_mean",
            "tgbp rep": "tgbp_rep_U_mean",
        }),
        ("Umax_vs_nusers", "U_max", {
            "main": "main_U_max",
            "main+qos": "main_ref_U_max",
            "main+qos+lb": "main_ref_lb_U_max",
            "wk demand rep": "wk_demand_rep_U_max",
            "wk qos rep": "wk_qos_rep_U_max",
            "bk rep": "bk_rep_U_max",
            "tgbp rep": "tgbp_rep_U_max",
        }),
        ("risk_sum_vs_nusers", "risk_sum", {
            "main": "main_risk_sum",
            "main+qos": "main_ref_risk_sum",
            "main+qos+lb": "main_ref_lb_risk_sum",
            "wk demand rep": "wk_demand_rep_risk_sum",
            "wk qos rep": "wk_qos_rep_risk_sum",
            "bk rep": "bk_rep_risk_sum",
            "tgbp rep": "tgbp_rep_risk_sum",
        }),
    ]

    # Payload feasibility tables (if present)
    payload_tables: List[Tuple[str, str, Dict[str, str]]] = [
        ("payload_feasible_rate_vs_nusers", "payload_feasible (mean over seeds)", {
            "payload feasible rate": "payload_feasible",
        }),
        ("payload_best_m_vs_nusers", "payload_best_m", {
            "best_m": "payload_best_m",
        }),
        ("payload_W_min_req_vs_nusers", "payload_W_min_req", {
            "W_min_req": "payload_W_min_req",
        }),
        ("payload_Tsum_vs_nusers", "payload_T_sum", {
            "T_sum": "payload_T_sum",
            "global_cap": "payload_global_cap",
        }),
        ("payload_overflow_vs_nusers", "overflow (sum)", {
            "T_over_sum": "payload_T_over_sum",
            "K_over_sum": "payload_K_over_sum",
        }),
        ("payload_viol_counts_vs_nusers", "violating satellites (mean)", {
            "n_viol_T": "payload_n_viol_T",
            "n_viol_K": "payload_n_viol_K",
        }),
        ("payload_bottlenecks_vs_nusers", "bottleneck", {
            "T_max": "payload_T_max",
            "K_max": "payload_K_max",
        }),
    ]

    # Runtime plot datapoints (use feasible-only df)
    dfB_rt = filter_payload_feasible_only(dfB_masked)
    my_parts = ["time_split_s", "time_ent_ref_s", "time_lb_ref_s"]
    present_parts = [c for c in my_parts if _has_col(dfB_rt, c)]
    runtime_series: Dict[str, str] = {}
    my_total = None

    if present_parts:
        runtime_series = {
            "MY algorithm total": "__my_total_s__",
            "WKMeans++ (demand) total": "time_baseline_without_qos_s",
            "WKMeans++ (demand*qos) total": "time_baseline_with_qos_s",
            "BKMeans total": "time_baseline_bkmeans_s",
            "TGBP total": "time_baseline_tgbp_s",
        }
        n = _nrows(dfB_rt)
        my_total = np.zeros(n, dtype=float)
        for c in present_parts:
            my_total += _col(dfB_rt, c).astype(float, copy=False)

    # Write file
    lines: List[str] = []
    lines.append("PHASE B TABLES (datapoints used in plots)\n")
    lines.append(f"CSV: {csv_path}\n")
    lines.append("Each row corresponds to a mean-over-seeds datapoint at a given n_users.\n")
    lines.append("Columns: mean/std/min/max/n are computed over seeds (ignoring NaNs).\n\n")

    # Main metric tables (use masked df so infeasible runs do not pollute)
    for table_name, ylabel, series in metric_series:
        lines.append("=" * 90 + "\n")
        lines.append(f"TABLE: {table_name}  (ylabel={ylabel})\n\n")

        headers = ["n_users", "series", "mean", "std", "min", "max", "n"]
        rows: List[List[str]] = []

        for label, colname in series.items():
            if not _has_col(dfB_masked, colname):
                continue
            agg = aggregate_by_group_mean_std(dfB_masked, "n_users", colname)
            if agg.x.size == 0:
                continue

            x_int = np.asarray([int(round(v)) for v in agg.x.tolist()], dtype=int)
            mm = {int(xi): i for i, xi in enumerate(x_int)}

            for nu in nusers.tolist():
                if int(nu) not in mm:
                    continue
                i = mm[int(nu)]
                rows.append([
                    str(int(nu)),
                    str(label),
                    _fmt_num(agg.y_mean[i]),
                    _fmt_num(agg.y_std[i]),
                    _fmt_num(agg.y_min[i]),
                    _fmt_num(agg.y_max[i]),
                    str(int(agg.n[i])),
                ])

        rows_sorted = sorted(rows, key=lambda r: (int(r[0]), r[1]))
        lines.append(_format_table(headers, rows_sorted))
        lines.append("\n")

    # Payload tables (use raw df, because feasibility itself is what we are plotting)
    for table_name, ylabel, series in payload_tables:
        keep = any(_has_col(dfB, c) for c in series.values())
        if not keep:
            continue

        lines.append("=" * 90 + "\n")
        lines.append(f"TABLE: {table_name}  (ylabel={ylabel})\n\n")

        headers = ["n_users", "series", "mean", "std", "min", "max", "n"]
        rows: List[List[str]] = []

        for label, colname in series.items():
            if not _has_col(dfB, colname):
                continue
            agg = aggregate_by_group_mean_std(dfB, "n_users", colname)
            if agg.x.size == 0:
                continue

            x_int = np.asarray([int(round(v)) for v in agg.x.tolist()], dtype=int)
            mm = {int(xi): i for i, xi in enumerate(x_int)}

            for nu in nusers.tolist():
                if int(nu) not in mm:
                    continue
                i = mm[int(nu)]
                rows.append([
                    str(int(nu)),
                    str(label),
                    _fmt_num(agg.y_mean[i]),
                    _fmt_num(agg.y_std[i]),
                    _fmt_num(agg.y_min[i]),
                    _fmt_num(agg.y_max[i]),
                    str(int(agg.n[i])),
                ])

        rows_sorted = sorted(rows, key=lambda r: (int(r[0]), r[1]))
        lines.append(_format_table(headers, rows_sorted))
        lines.append("\n")

    if runtime_series and my_total is not None:
        lines.append("=" * 90 + "\n")
        lines.append("TABLE: runtime_methods_vs_nusers  (ylabel=seconds)  (payload-feasible runs only)\n\n")

        tmp_my = {"n_users": _col(dfB_rt, "n_users"), "__my_total_s__": my_total}

        headers = ["n_users", "series", "mean_s", "std_s", "min_s", "max_s", "n"]
        rows: List[List[str]] = []

        for label, colname in runtime_series.items():
            if colname == "__my_total_s__":
                agg = aggregate_by_group_mean_std(tmp_my, "n_users", "__my_total_s__")
            else:
                if not _has_col(dfB_rt, colname):
                    continue
                agg = aggregate_by_group_mean_std(dfB_rt, "n_users", colname)

            if agg.x.size == 0:
                continue

            x_int = np.asarray([int(round(v)) for v in agg.x.tolist()], dtype=int)
            mm = {int(xi): i for i, xi in enumerate(x_int)}

            for nu in nusers.tolist():
                if int(nu) not in mm:
                    continue
                i = mm[int(nu)]
                rows.append([
                    str(int(nu)),
                    str(label),
                    _fmt_num(agg.y_mean[i]),
                    _fmt_num(agg.y_std[i]),
                    _fmt_num(agg.y_min[i]),
                    _fmt_num(agg.y_max[i]),
                    str(int(agg.n[i])),
                ])

        rows_sorted = sorted(rows, key=lambda r: (int(r[0]), r[1]))
        lines.append(_format_table(headers, rows_sorted))
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    return out_path


# ----------------------------
# Plot helpers
# ----------------------------
def _save_or_show(fig, out_path: str, show: bool) -> None:
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_lines_vs_nusers(
    df,
    series: Dict[str, str],
    *,
    title: str,
    ylabel: str,
    out_path: str,
    show: bool = False,
    include_errorbars: bool = False,
    use_logy: bool = False,
) -> None:
    if not _has_col(df, "n_users"):
        raise ValueError("CSV does not contain 'n_users' column.")

    fig = plt.figure()
    ax = plt.gca()

    for label, colname in series.items():
        if not _has_col(df, colname):
            continue
        agg = aggregate_by_group_mean_std(df, "n_users", colname)
        if agg.x.size == 0:
            continue

        if include_errorbars:
            ax.errorbar(agg.x, agg.y_mean, yerr=agg.y_std, marker="o", linestyle="-", label=label)
        else:
            ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label=label)

    ax.set_title(title)
    ax.set_xlabel("n_users")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if use_logy:
        ax.set_yscale("log")
    ax.legend()
    _save_or_show(fig, out_path, show)


def plot_payload_ratio_vs_nusers(
    df,
    *,
    title: str,
    out_path: str,
    show: bool = False,
) -> None:
    """
    Plot load ratio = payload_T_sum / payload_global_cap vs n_users.
    """
    if not (_has_col(df, "n_users") and _has_col(df, "payload_T_sum") and _has_col(df, "payload_global_cap")):
        return

    T = _col(df, "payload_T_sum").astype(float, copy=False)
    C = _col(df, "payload_global_cap").astype(float, copy=False)
    ratio = T / (C + 1e-12)

    tmp = {"n_users": _col(df, "n_users"), "ratio": ratio}
    plot_lines_vs_nusers(
        tmp,
        series={"T_sum / global_cap": "ratio"},
        title=title,
        ylabel="load ratio",
        out_path=out_path,
        show=show,
        include_errorbars=False,
        use_logy=False,
    )


def plot_phaseA_robustness_scatter(df, metric_col: str, *, out_path: str, show: bool = False) -> None:
    if not _has_col(df, "seed"):
        raise ValueError("Phase A CSV missing 'seed'.")
    if not _has_col(df, metric_col):
        raise ValueError(f"Phase A CSV missing '{metric_col}'.")

    seed = _col(df, "seed").astype(float, copy=False)
    y = _col(df, metric_col).astype(float, copy=False)

    mask = ~np.isnan(seed) & ~np.isnan(y)
    seed = seed[mask]
    y = y[mask]

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(seed, y, label=metric_col)

    tmp = {"seed": seed, metric_col: y}
    agg = aggregate_by_group_mean_std(tmp, "seed", metric_col)
    if agg.x.size > 0:
        ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label="mean")

    ax.set_title(f"Phase A robustness: {metric_col} vs seed")
    ax.set_xlabel("seed")
    ax.set_ylabel(metric_col)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    _save_or_show(fig, out_path, show)


def plot_runtime_methods_vs_nusers(
    df,
    *,
    out_path: str,
    show: bool = False,
    use_logy: bool = False,
) -> None:
    """
    Phase B: Plot total runtime of:
      - your full pipeline (sum of internal stages)
      - each baseline (already measured as one number per run)
    All curves are mean-over-seeds per n_users.

    IMPORTANT: If payload_feasible exists, uses ONLY feasible runs
    (because baselines/refinements are typically skipped when infeasible).
    """
    if not _has_col(df, "n_users"):
        raise ValueError("CSV missing 'n_users'.")

    df_rt = filter_payload_feasible_only(df)

    my_parts = ["time_split_s", "time_ent_ref_s", "time_lb_ref_s"]
    present_parts = [c for c in my_parts if _has_col(df_rt, c)]
    if not present_parts:
        raise ValueError("No internal timing columns found for main algorithm (time_*).")

    n = _nrows(df_rt)
    my_total = np.zeros(n, dtype=float)
    for c in present_parts:
        my_total += _col(df_rt, c).astype(float, copy=False)

    baselines = {
        "WKMeans++ (demand) total": "time_baseline_without_qos_s",
        "WKMeans++ (demand*qos) total": "time_baseline_with_qos_s",
        "BKMeans total": "time_baseline_bkmeans_s",
        "TGBP total": "time_baseline_tgbp_s",
    }

    fig = plt.figure()
    ax = plt.gca()

    tmp_my = {"n_users": _col(df_rt, "n_users"), "my_total_s": my_total}
    agg = aggregate_by_group_mean_std(tmp_my, "n_users", "my_total_s")
    ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label="MY algorithm total")

    for label, colname in baselines.items():
        if not _has_col(df_rt, colname):
            continue
        agg_b = aggregate_by_group_mean_std(df_rt, "n_users", colname)
        if agg_b.x.size == 0:
            continue
        ax.plot(agg_b.x, agg_b.y_mean, marker="o", linestyle="-", label=label)

    ax.set_title("Total runtime vs n_users (payload-feasible runs only)")
    ax.set_xlabel("n_users")
    ax.set_ylabel("seconds")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if use_logy:
        ax.set_yscale("log")
    ax.legend(ncol=1, fontsize=9)
    _save_or_show(fig, out_path, show)


# ----------------------------
# Public API (call from main.py)
# ----------------------------
def plot_phaseB(phaseB_csv: str, out_dir: str, *, show: bool = False) -> None:
    dfB = load_sweep_csv(phaseB_csv)
    _ensure_out_dir(out_dir)

    # Write all plot datapoints to a txt table file
    _write_phaseB_tables_txt(dfB, out_dir, csv_path=phaseB_csv)

    # Mask refinements/baselines when payload infeasible to avoid "fake zeros"
    masked_cols = [
        "main_ref_K", "main_ref_lb_K",
        "main_ref_ent_edge_pct", "main_ref_lb_ent_edge_pct",
        "main_ref_U_mean", "main_ref_lb_U_mean",
        "main_ref_U_max", "main_ref_lb_U_max",
        "main_ref_risk_sum", "main_ref_lb_risk_sum",
        "wk_demand_rep_K", "wk_qos_rep_K", "bk_rep_K", "tgbp_rep_K",
        "wk_demand_rep_ent_edge_pct", "wk_qos_rep_ent_edge_pct", "bk_rep_ent_edge_pct", "tgbp_rep_ent_edge_pct",
        "wk_demand_rep_U_mean", "wk_qos_rep_U_mean", "bk_rep_U_mean", "tgbp_rep_U_mean",
        "wk_demand_rep_U_max", "wk_qos_rep_U_max", "bk_rep_U_max", "tgbp_rep_U_max",
        "wk_demand_rep_risk_sum", "wk_qos_rep_risk_sum", "bk_rep_risk_sum", "tgbp_rep_risk_sum",
        "time_baseline_without_qos_s", "time_baseline_with_qos_s", "time_baseline_bkmeans_s", "time_baseline_tgbp_s",
        "time_ent_ref_s", "time_lb_ref_s",
    ]
    dfB_masked = mask_infeasible_as_nan(dfB, masked_cols)

    # ----------------------------
    # Existing algorithm plots (comparisons)
    # ----------------------------
    plot_lines_vs_nusers(
        dfB_masked,
        series={
            "main+qos+lb": "main_ref_lb_K",
            "wk demand rep": "wk_demand_rep_K",
            "wk qos rep": "wk_qos_rep_K",
            "bk rep": "bk_rep_K",
            "tgbp rep": "tgbp_rep_K",
        },
        title="Beams K vs n_users (payload-feasible runs only)",
        ylabel="K",
        out_path=os.path.join(out_dir, "K_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB_masked,
        series={
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
            "bk rep": "bk_rep_ent_edge_pct",
            "tgbp rep": "tgbp_rep_ent_edge_pct",
        },
        title="Enterprise edge exposure (%) vs n_users (payload-feasible runs only)",
        ylabel="enterprise edge exposure (%)",
        out_path=os.path.join(out_dir, "ent_edge_pct_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB_masked,
        series={
            "main+qos+lb": "main_ref_lb_U_mean",
            "wk demand rep": "wk_demand_rep_U_mean",
            "wk qos rep": "wk_qos_rep_U_mean",
            "bk rep": "bk_rep_U_mean",
            "tgbp rep": "tgbp_rep_U_mean",
        },
        title="Utilization U_mean vs n_users (payload-feasible runs only)",
        ylabel="U_mean",
        out_path=os.path.join(out_dir, "Umean_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB_masked,
        series={
            "main+qos+lb": "main_ref_lb_U_max",
            "wk demand rep": "wk_demand_rep_U_max",
            "wk qos rep": "wk_qos_rep_U_max",
            "bk rep": "bk_rep_U_max",
            "tgbp rep": "tgbp_rep_U_max",
        },
        title="Utilization U_max vs n_users (payload-feasible runs only)",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "Umax_vs_nusers.png"),
        show=show,
    )

    plot_lines_vs_nusers(
        dfB_masked,
        series={
            "main+qos+lb": "main_ref_lb_risk_sum",
            "wk demand rep": "wk_demand_rep_risk_sum",
            "wk qos rep": "wk_qos_rep_risk_sum",
            "bk rep": "bk_rep_risk_sum",
            "tgbp rep": "tgbp_rep_risk_sum",
        },
        title="Enterprise risk_sum vs n_users (payload-feasible runs only)",
        ylabel="risk_sum",
        out_path=os.path.join(out_dir, "risk_sum_vs_nusers.png"),
        show=show,
    )

    # Runtime comparisons (feasible-only)
    try:
        plot_runtime_methods_vs_nusers(
            dfB_masked,
            out_path=os.path.join(out_dir, "runtime_methods_vs_nusers.png"),
            show=show,
            use_logy=False,
        )
    except Exception:
        # If timing columns missing, skip
        pass

    # ----------------------------
    # NEW: Satellite payload feasibility plots
    # ----------------------------
    # Feasibility rate vs n_users
    if _has_col(dfB, "payload_feasible"):
        plot_lines_vs_nusers(
            dfB,
            series={"payload feasible rate": "payload_feasible"},
            title="Payload feasibility rate vs n_users",
            ylabel="P(feasible) over seeds",
            out_path=os.path.join(out_dir, "payload_feasible_rate_vs_nusers.png"),
            show=show,
            include_errorbars=False,
        )

    # Best prefix size m vs n_users
    if _has_col(dfB, "payload_best_m"):
        plot_lines_vs_nusers(
            dfB,
            series={"best m (mean)": "payload_best_m"},
            title="Minimal satellites used (best_m) vs n_users",
            ylabel="best_m",
            out_path=os.path.join(out_dir, "payload_best_m_vs_nusers.png"),
            show=show,
            include_errorbars=True,
        )

    # Required W_min (diagnostic) vs n_users
    if _has_col(dfB, "payload_W_min_req"):
        plot_lines_vs_nusers(
            dfB,
            series={"W_min_req": "payload_W_min_req"},
            title="Required schedule window W_min_req vs n_users",
            ylabel="W_min_req (slots)",
            out_path=os.path.join(out_dir, "payload_Wmin_req_vs_nusers.png"),
            show=show,
            include_errorbars=True,
        )

    # Time demand vs capacity (T_sum and global_cap)
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

    # Bottleneck satellites: T_max and K_max
    if _has_col(dfB, "payload_T_max") or _has_col(dfB, "payload_K_max"):
        series = {}
        if _has_col(dfB, "payload_T_max"):
            series["T_max"] = "payload_T_max"
        if _has_col(dfB, "payload_K_max"):
            series["K_max"] = "payload_K_max"
        plot_lines_vs_nusers(
            dfB,
            series=series,
            title="Payload bottlenecks (T_max, K_max) vs n_users",
            ylabel="value",
            out_path=os.path.join(out_dir, "payload_bottlenecks_vs_nusers.png"),
            show=show,
            include_errorbars=True,
        )

    # Overflow severity (sums)
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
            include_errorbars=True,
        )

    # Violation counts (mean number of violating satellites)
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
            include_errorbars=True,
        )


def plot_phaseA(phaseA_csv: str, out_dir: str, *, show: bool = False) -> None:
    dfA = load_sweep_csv(phaseA_csv)
    _ensure_out_dir(out_dir)

    metrics = [
        "main_K", "main_ref_K", "main_ref_lb_K",
        "main_ent_edge_pct", "main_ref_ent_edge_pct", "main_ref_lb_ent_edge_pct",
        "main_U_mean", "main_ref_U_mean", "main_ref_lb_U_mean",
        "main_U_max", "main_ref_U_max", "main_ref_lb_U_max",
        "main_risk_sum", "main_ref_risk_sum", "main_ref_lb_risk_sum",
        "time_split_s", "time_ent_ref_s", "time_lb_ref_s",
        "time_baseline_bkmeans_s", "time_baseline_tgbp_s",
    ]

    for m in metrics:
        if _has_col(dfA, m):
            plot_phaseA_robustness_scatter(
                dfA, m,
                out_path=os.path.join(out_dir, f"{m}_vs_seed.png"),
                show=show,
            )


def plot_all(phaseA_csv: str, phaseB_csv: str, out_dir: str = "plots", *, show: bool = False) -> None:
    plot_phaseA(phaseA_csv, os.path.join(out_dir, "phaseA"), show=show)
    plot_phaseB(phaseB_csv, os.path.join(out_dir, "phaseB"), show=show)