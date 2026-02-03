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
        if all((isinstance(v, (int, float, np.floating)) or (v is np.nan)) for v in vals):
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
    y = _col(df, ycol).astype(float, copy=False)

    mask = ~np.isnan(y)
    # if x numeric, also drop NaN x
    if np.issubdtype(np.asarray(x).dtype, np.number):
        mask &= ~np.isnan(x.astype(float, copy=False))

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

    # convert x to float for plotting where possible
    try:
        xplot = np.asarray([float(v) for v in xuniq], dtype=float)
    except Exception:
        xplot = np.arange(len(xuniq), dtype=float)

    return AggSeries(
        x=xplot,
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
    # round-close to int
    xi = np.asarray([int(round(v)) for v in x.tolist()], dtype=int)
    return np.unique(xi)


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


def _write_phaseB_tables_txt(
    dfB,
    out_dir: str,
    *,
    csv_path: str,
) -> str:
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

    # Define Phase-B plotted series (must match plot_phaseB)
    metric_series: List[Tuple[str, str, Dict[str, str]]] = [
        ("K_vs_nusers", "K", {
            "main" : "main_K",
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

    # Runtime plot datapoints
    my_parts = ["time_split_s", "time_ent_ref_s", "time_lb_ref_s"]
    present_parts = [c for c in my_parts if _has_col(dfB, c)]
    if present_parts:
        runtime_series: Dict[str, str] = {
            "MY algorithm total": "__my_total_s__",
            "WKMeans++ (demand) total": "time_baseline_without_qos_s",
            "WKMeans++ (demand*qos) total": "time_baseline_with_qos_s",
            "BKMeans total": "time_baseline_bkmeans_s",
            "TGBP total": "time_baseline_tgbp_s",
        }
    else:
        runtime_series = {}

    # Precompute my_total if possible
    if runtime_series:
        n = _nrows(dfB)
        my_total = np.zeros(n, dtype=float)
        for c in present_parts:
            my_total += _col(dfB, c).astype(float, copy=False)

    # Write file
    lines: List[str] = []
    lines.append("PHASE B TABLES (datapoints used in plots)\n")
    lines.append(f"CSV: {csv_path}\n")
    lines.append("Each row corresponds to a mean-over-seeds datapoint at a given n_users.\n")
    lines.append("Columns: mean/std/min/max/n are computed over seeds (ignoring NaNs).\n")
    lines.append("\n")

    for table_name, ylabel, series in metric_series:
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

            # Map float x back to int keys
            x_int = np.asarray([int(round(v)) for v in agg.x.tolist()], dtype=int)
            m = {int(xi): i for i, xi in enumerate(x_int)}

            for nu in nusers.tolist():
                if int(nu) not in m:
                    continue
                i = m[int(nu)]
                rows.append([
                    str(int(nu)),
                    str(label),
                    _fmt_num(agg.y_mean[i]),
                    _fmt_num(agg.y_std[i]),
                    _fmt_num(agg.y_min[i]),
                    _fmt_num(agg.y_max[i]),
                    str(int(agg.n[i])),
                ])

        # Sort rows by n_users then label for stable output
        rows_sorted = sorted(rows, key=lambda r: (int(r[0]), r[1]))
        lines.append(_format_table(headers, rows_sorted))
        lines.append("\n")

    if runtime_series:
        lines.append("=" * 90 + "\n")
        lines.append("TABLE: runtime_methods_vs_nusers  (ylabel=seconds)\n\n")

        # Build a temp df-like view for my_total
        tmp_my = {"n_users": _col(dfB, "n_users"), "__my_total_s__": my_total}

        headers = ["n_users", "series", "mean_s", "std_s", "min_s", "max_s", "n"]
        rows: List[List[str]] = []

        for label, colname in runtime_series.items():
            if colname == "__my_total_s__":
                agg = aggregate_by_group_mean_std(tmp_my, "n_users", "__my_total_s__")
            else:
                if not _has_col(dfB, colname):
                    continue
                agg = aggregate_by_group_mean_std(dfB, "n_users", colname)

            if agg.x.size == 0:
                continue
            x_int = np.asarray([int(round(v)) for v in agg.x.tolist()], dtype=int)
            m = {int(xi): i for i, xi in enumerate(x_int)}

            for nu in nusers.tolist():
                if int(nu) not in m:
                    continue
                i = m[int(nu)]
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
def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


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
    ax.legend()

    _save_or_show(fig, out_path, show)


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
      - your full pipeline (sum of your internal stages)
      - each baseline (already measured as one number per run)
    All curves are mean-over-seeds per n_users.
    """
    if not _has_col(df, "n_users"):
        raise ValueError("CSV missing 'n_users'.")

    # ----- build "my total runtime" as sum of internal parts -----
    my_parts = [
        "time_split_s",
        "time_ent_ref_s",
        "time_lb_ref_s",
    ]
    present_parts = [c for c in my_parts if _has_col(df, c)]
    if not present_parts:
        raise ValueError("No internal timing columns found for main algorithm (time_*).")

    n = _nrows(df)
    my_total = np.zeros(n, dtype=float)
    for c in present_parts:
        my_total += _col(df, c).astype(float, copy=False)

    # ----- baseline totals (already measured per run) -----
    baselines = {
        "WKMeans++ (demand) total": "time_baseline_without_qos_s",
        "WKMeans++ (demand*qos) total": "time_baseline_with_qos_s",
        "BKMeans total": "time_baseline_bkmeans_s",
        "TGBP total": "time_baseline_tgbp_s",
    }

    fig = plt.figure()
    ax = plt.gca()

    # My algorithm curve
    tmp_my = {"n_users": _col(df, "n_users"), "my_total_s": my_total}
    agg = aggregate_by_group_mean_std(tmp_my, "n_users", "my_total_s")
    ax.plot(agg.x, agg.y_mean, marker="o", linestyle="-", label="MY algorithm total")

    # Baseline curves
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
# Public API (call from main.py)
# ----------------------------
def plot_phaseB(phaseB_csv: str, out_dir: str, *, show: bool = False) -> None:
    dfB = load_sweep_csv(phaseB_csv)
    _ensure_out_dir(out_dir)

    # --- NEW: write all plot datapoints to a txt table file ---
    _write_phaseB_tables_txt(dfB, out_dir, csv_path=phaseB_csv)

    # K scaling
    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_K",
            "wk demand rep": "wk_demand_rep_K",
            "wk qos rep": "wk_qos_rep_K",
            "bk rep": "bk_rep_K",
            "tgbp rep": "tgbp_rep_K",
        },
        title="Beams K vs n_users",
        ylabel="K",
        out_path=os.path.join(out_dir, "K_vs_nusers.png"),
        show=show,
    )

    # Enterprise exposure
    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_ent_edge_pct",
            "wk demand rep": "wk_demand_rep_ent_edge_pct",
            "wk qos rep": "wk_qos_rep_ent_edge_pct",
            "bk rep": "bk_rep_ent_edge_pct",
            "tgbp rep": "tgbp_rep_ent_edge_pct",
        },
        title="Enterprise edge exposure (%) vs n_users",
        ylabel="enterprise edge exposure (%)",
        out_path=os.path.join(out_dir, "ent_edge_pct_vs_nusers.png"),
        show=show,
    )

    # U_mean
    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_U_mean",
            "wk demand rep": "wk_demand_rep_U_mean",
            "wk qos rep": "wk_qos_rep_U_mean",
            "bk rep": "bk_rep_U_mean",
            "tgbp rep": "tgbp_rep_U_mean",
        },
        title="Utilization U_mean vs n_users",
        ylabel="U_mean",
        out_path=os.path.join(out_dir, "Umean_vs_nusers.png"),
        show=show,
    )

    # U_max
    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_U_max",
            "wk demand rep": "wk_demand_rep_U_max",
            "wk qos rep": "wk_qos_rep_U_max",
            "bk rep": "bk_rep_U_max",
            "tgbp rep": "tgbp_rep_U_max",
        },
        title="Utilization U_max vs n_users",
        ylabel="U_max",
        out_path=os.path.join(out_dir, "Umax_vs_nusers.png"),
        show=show,
    )

    # Risk_sum
    plot_lines_vs_nusers(
        dfB,
        series={
            "main+qos+lb": "main_ref_lb_risk_sum",
            "wk demand rep": "wk_demand_rep_risk_sum",
            "wk qos rep": "wk_qos_rep_risk_sum",
            "bk rep": "bk_rep_risk_sum",
            "tgbp rep": "tgbp_rep_risk_sum",
        },
        title="Enterprise risk_sum vs n_users",
        ylabel="risk_sum",
        out_path=os.path.join(out_dir, "risk_sum_vs_nusers.png"),
        show=show,
    )

    # All runtime components in a single plot
    plot_runtime_methods_vs_nusers(
        dfB,
        out_path=os.path.join(out_dir, "runtime_methods_vs_nusers.png"),
        show=show,
        use_logy=False,
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
