# src/pipeline.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import ScenarioConfig
from src.baselines.weighted_kmeans import run_weighted_kmeans_baseline
from src.baselines.fast_beam_placement import run_bkmeans_baseline, run_tgbp_baseline
from src.evaluator import evaluate_cluster
from src.helper import summarize_multisat, print_summary, print_config
from src.plot import plot_clusters_overlay
from src.refine_qos_angle import refine_enterprise_by_angle
from src.refine_load_balance import refine_load_balance_by_overlap
from src.split import split_farthest
from src.profiling import Profiler
from src.usergen import generate_users, pack_users_raw, build_users_for_sat

# NEW satellites API
from src.satellites import sort_active_sats, associate_users_balanced


def _parse_time_utc_iso(s: str) -> datetime:
    """Parse an ISO time string into an aware UTC datetime."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _empty_summary() -> dict[str, Any]:
    # Must match helper.summarize() keys exactly (so flatten_summary works).
    return {
        "K": 0,
        "feasible_rate": 0.0,
        "U_mean": 0.0,
        "U_max": 0.0,
        "U_min": 0.0,
        "risk_sum": 0.0,
        "ent_total": 0,
        "ent_exposed": 0,
        "ent_edge_pct": 0.0,
        "ent_z_mean": 0.0,
        "ent_z_p90": 0.0,
        "ent_z_max": 0.0,
    }


def split_to_feasible(users, cfg: ScenarioConfig, prof: Profiler | None = None, max_clusters: int = 5000):
    """
    Queue-based split-to-feasible:
    - Evaluate only the cluster being processed.
    - If infeasible, split it and push children.
    - If feasible, store it.
    """
    pending = deque([np.arange(users.n, dtype=int)])
    clusters_feas: list[np.ndarray] = []
    evals_feas: list[dict] = []
    n_splits = 0

    while pending:
        if (len(pending) + len(clusters_feas)) > max_clusters:
            raise RuntimeError("Too many clusters. Check demands/PHY parameters.")

        S = pending.pop()
        if S.size == 0:
            continue

        ev = evaluate_cluster(users, S, cfg)
        if prof:
            prof.inc("eval_calls")

        if ev["feasible"]:
            clusters_feas.append(S)
            evals_feas.append(ev)
            continue

        seed = int(cfg.run.seed) + n_splits + 1
        S1, S2 = split_farthest(users.xy_m, S, seed=seed)
        S1 = np.asarray(S1, dtype=int)
        S2 = np.asarray(S2, dtype=int)

        if S1.size == 0 or S2.size == 0:
            raise RuntimeError(f"split_farthest produced empty split: |S|={S.size} -> {S1.size},{S2.size}")

        pending.append(S1)
        pending.append(S2)
        n_splits += 1

    if prof:
        prof.c["n_splits"] = prof.c.get("n_splits", 0) + n_splits

    return clusters_feas, evals_feas


def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    """
    CLEAN multi-satellite snapshot scenario.
    Restores BK-Means + TGBP baselines and returns ALL keys required by helper.flatten_run_record().
    """
    prof = Profiler()
    ms = cfg.multisat  # no wrappers/fallbacks

    # 1) Generate users (sat-agnostic)
    prof.tic("usergen")
    user_list = generate_users(cfg)
    users_raw = pack_users_raw(user_list)
    prof.toc("usergen")

    verbose = bool(getattr(cfg.run, "verbose", True))
    enable_plots = bool(getattr(cfg.run, "enable_plots", False))

    if verbose:
        print(f"Generated {users_raw.n} users in {cfg.region_mode} bbox.")
        print(f"Beam modes (km): {cfg.beam.radius_modes_km}, B={cfg.phy.bandwidth_hz/1e6:.0f} MHz, EIRP={cfg.phy.eirp_dbw:.1f} dBW")

    print_config(cfg)

    # 2) Select active satellites (NEW: multi-anchor + greedy marginal gain)
    # Optional fixed snapshot time (UTC) for reproducibility
    t0_utc = _parse_time_utc_iso(ms.time_utc_iso) if ms.time_utc_iso else None

    # Anchor grid knobs (fast, minimal data). Keep hard-coded unless you add cfg fields.
    n_lat_anchors = 3
    n_lon_anchors = 3
    quality_mode = "sin"

    prof.tic("sat_select")
    t0_utc, active_sats = sort_active_sats(
        cfg,
        t0_utc=t0_utc,
        n_lat_anchors=n_lat_anchors,
        n_lon_anchors=n_lon_anchors,
        quality_mode=quality_mode,
    )
    prof.toc("sat_select")

    if len(active_sats) == 0:
        raise RuntimeError("No active satellites found above elev mask at any anchor in the region bbox.")

    sat_ecef_m = np.stack([s.ecef_m for s in active_sats], axis=0)

    if verbose:
        print(
            f"\nSelected {len(active_sats)} active sats @ {t0_utc.isoformat()} (UTC), "
            f"anchors={n_lat_anchors}x{n_lon_anchors}, mask={ms.elev_mask_deg:.1f}°"
        )
        print(
            f"Top sat max-elev across anchors: {active_sats[0].elev_ref_deg:.1f}°, "
            f"bottom: {active_sats[-1].elev_ref_deg:.1f}°"
        )

    # 3) Balanced association users -> satellite
    prof.tic("assoc")
    assoc = associate_users_balanced(
        user_ecef_m=users_raw.ecef_m,
        user_demand_mbps=users_raw.demand_mbps,
        user_qos_w=users_raw.qos_w,
        sat_ecef_m=sat_ecef_m,
        elev_mask_deg=ms.elev_mask_deg,
        load_mode=ms.assoc_load_mode,
        slack=ms.assoc_slack,
        max_rounds=ms.assoc_max_rounds,
        max_total_moves=ms.assoc_max_moves,
        seed=int(cfg.run.seed),
    )
    prof.toc("assoc")

    if verbose:
        print(f"\nAssociation: unserved={assoc.n_unserved}/{users_raw.n} ({100*assoc.n_unserved/max(users_raw.n,1):.2f}%), moves={assoc.n_moves}")

    # 4) Per-satellite runs + baselines (restored)
    pieces_main: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_ref: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_lb: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    pieces_wk_demand_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_demand_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_qos_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_qos_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    pieces_bk_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_bk_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_tgbp_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_tgbp_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    busiest_sat_idx = int(np.argmax([len(x) for x in assoc.sat_user_ids])) if len(assoc.sat_user_ids) else 0
    busiest_users = None
    busiest_clusters = busiest_evals = None
    busiest_clusters_ref = busiest_evals_ref = None
    busiest_clusters_lb = busiest_evals_lb = None

    # Baselines toggle (backward compatible): default True
    enable_fastbp = bool(getattr(cfg.run, "enable_fastbp_baselines", True))

    for s_idx, user_ids in enumerate(assoc.sat_user_ids):
        if user_ids.size == 0:
            continue

        users_sat = build_users_for_sat(users_raw, user_ids, sat_ecef_m[s_idx])
        if users_sat.n == 0:
            continue

        # --- Main (split-to-feasible)
        prof.tic("split")
        clusters, evals = split_to_feasible(users_sat, cfg, prof=prof)
        prof.toc("split")
        pieces_main.append((users_sat, clusters, evals))

        # --- Enterprise refinement
        prof.tic("ent_ref")
        clusters_ref, evals_ref, ref_stats = refine_enterprise_by_angle(
            users_sat, cfg, clusters, evals,
            n_rounds=cfg.qos_refine.rounds,
            kcand=cfg.qos_refine.kcand,
            max_moves_per_round=cfg.qos_refine.max_moves_per_round,
        )
        prof.toc("ent_ref")
        prof.c["ent_moves_tried"] = prof.c.get("ent_moves_tried", 0) + int(ref_stats.get("moves_tried", 0))
        prof.c["ent_moves_accepted"] = prof.c.get("ent_moves_accepted", 0) + int(ref_stats.get("moves_accepted", 0))
        pieces_ref.append((users_sat, clusters_ref, evals_ref))

        # --- Load-balance refinement
        prof.tic("lb_ref")
        clusters_lb, evals_lb, lb_stats = refine_load_balance_by_overlap(users_sat, cfg, clusters_ref, evals_ref)
        prof.toc("lb_ref")
        prof.c["lb_moves_tried"] = prof.c.get("lb_moves_tried", 0) + int(lb_stats.get("moves_tried", 0))
        prof.c["lb_moves_accepted"] = prof.c.get("lb_moves_accepted", 0) + int(lb_stats.get("moves_accepted", 0))
        pieces_lb.append((users_sat, clusters_lb, evals_lb))

        # Reference K for fixed-K baselines
        K_ref = len(clusters)

        # --- Weighted KMeans++ baselines
        prof.tic("baseline_without_qos")
        base_wo = run_weighted_kmeans_baseline(users_sat, cfg, K_ref=K_ref, use_qos_weight=False)
        prof.toc("baseline_without_qos")
        pieces_wk_demand_fixed.append((users_sat, base_wo["fixedK"]["clusters"], base_wo["fixedK"]["evals"]))
        pieces_wk_demand_rep.append((users_sat, base_wo["repaired"]["clusters"], base_wo["repaired"]["evals"]))

        prof.tic("baseline_with_qos")
        base_wq = run_weighted_kmeans_baseline(users_sat, cfg, K_ref=K_ref, use_qos_weight=True)
        prof.toc("baseline_with_qos")
        pieces_wk_qos_fixed.append((users_sat, base_wq["fixedK"]["clusters"], base_wq["fixedK"]["evals"]))
        pieces_wk_qos_rep.append((users_sat, base_wq["repaired"]["clusters"], base_wq["repaired"]["evals"]))

        # --- FastBP paper baselines (BK-Means + TGBP)
        if enable_fastbp:
            prof.tic("baseline_bkmeans")
            out_bk = run_bkmeans_baseline(users_sat, cfg, K_hint=K_ref)
            prof.toc("baseline_bkmeans")
            pieces_bk_fixed.append((users_sat, out_bk["fixedK"]["clusters"], out_bk["fixedK"]["evals"]))
            pieces_bk_rep.append((users_sat, out_bk["repaired"]["clusters"], out_bk["repaired"]["evals"]))

            prof.tic("baseline_tgbp")
            out_tg = run_tgbp_baseline(users_sat, cfg)
            prof.toc("baseline_tgbp")
            pieces_tgbp_fixed.append((users_sat, out_tg["fixedK"]["clusters"], out_tg["fixedK"]["evals"]))
            pieces_tgbp_rep.append((users_sat, out_tg["repaired"]["clusters"], out_tg["repaired"]["evals"]))

        # For plotting: store busiest sat result
        if s_idx == busiest_sat_idx:
            busiest_users = users_sat
            busiest_clusters, busiest_evals = clusters, evals
            busiest_clusters_ref, busiest_evals_ref = clusters_ref, evals_ref
            busiest_clusters_lb, busiest_evals_lb = clusters_lb, evals_lb

    # 5) Global summaries
    main_summary = summarize_multisat(pieces_main, cfg) if pieces_main else _empty_summary()
    ref_summary = summarize_multisat(pieces_ref, cfg) if pieces_ref else _empty_summary()
    lb_summary = summarize_multisat(pieces_lb, cfg) if pieces_lb else _empty_summary()

    wk_demand_fixed = summarize_multisat(pieces_wk_demand_fixed, cfg) if pieces_wk_demand_fixed else _empty_summary()
    wk_demand_rep = summarize_multisat(pieces_wk_demand_rep, cfg) if pieces_wk_demand_rep else _empty_summary()
    wk_qos_fixed = summarize_multisat(pieces_wk_qos_fixed, cfg) if pieces_wk_qos_fixed else _empty_summary()
    wk_qos_rep = summarize_multisat(pieces_wk_qos_rep, cfg) if pieces_wk_qos_rep else _empty_summary()

    bk_fixed = summarize_multisat(pieces_bk_fixed, cfg) if pieces_bk_fixed else _empty_summary()
    bk_rep = summarize_multisat(pieces_bk_rep, cfg) if pieces_bk_rep else _empty_summary()
    tgbp_fixed = summarize_multisat(pieces_tgbp_fixed, cfg) if pieces_tgbp_fixed else _empty_summary()
    tgbp_rep = summarize_multisat(pieces_tgbp_rep, cfg) if pieces_tgbp_rep else _empty_summary()

    if verbose:
        print_summary("Main algorithm (split-to-feasible) [global]", main_summary, cfg)
        print_summary("Main + enterprise angle refinement [global]", ref_summary, cfg)
        print_summary("Main + enterprise + load-balance refine [global]", lb_summary, cfg)

        print_summary("WKMeans++ baseline demand (fixed-K per sat) [global]", wk_demand_fixed, cfg)
        print_summary("WKMeans++ baseline demand (after repair) [global]", wk_demand_rep, cfg)
        print_summary("WKMeans++ baseline demand*qos (fixed-K per sat) [global]", wk_qos_fixed, cfg)
        print_summary("WKMeans++ baseline demand*qos (after repair) [global]", wk_qos_rep, cfg)

        if enable_fastbp:
            print_summary("BK-Means baseline (fixed-K) [global]", bk_fixed, cfg)
            print_summary("BK-Means baseline (after repair) [global]", bk_rep, cfg)
            print_summary("TGBP baseline (fixed-K) [global]", tgbp_fixed, cfg)
            print_summary("TGBP baseline (after repair) [global]", tgbp_rep, cfg)

    # 6) Optional plots (only busiest sat)
    if enable_plots and busiest_users is not None and busiest_clusters is not None:
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters,
            evals=busiest_evals,
            title=f"Busiest sat: main (K={len(busiest_clusters)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters_ref,
            evals=busiest_evals_ref,
            title=f"Busiest sat: main+QoS refine (K={len(busiest_clusters_ref)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters_lb,
            evals=busiest_evals_lb,
            title=f"Busiest sat: main+QoS+LB refine (K={len(busiest_clusters_lb)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )

    # 7) Return record: MUST include keys expected by helper.flatten_run_record()
    return {
        "seed": cfg.run.seed,
        "region_mode": cfg.region_mode,
        "n_users": cfg.run.n_users,

        "use_hotspots": cfg.usergen.enabled,
        "n_hotspots": cfg.usergen.n_hotspots,
        "noise_frac": cfg.usergen.noise_frac,
        "sigma_min": cfg.usergen.hotspot_sigma_m_min,
        "sigma_max": cfg.usergen.hotspot_sigma_m_max,

        "rho_safe": cfg.ent.rho_safe,
        "eirp_dbw": cfg.phy.eirp_dbw,
        "bandwidth_hz": cfg.phy.bandwidth_hz,
        "radius_modes_km": cfg.beam.radius_modes_km,

        # Multi-sat metadata (nice to keep in CSV)
        "ms_tle_path": ms.tle_path,
        "ms_time_utc": t0_utc.isoformat(),
        "ms_elev_mask_deg": ms.elev_mask_deg,
        "ms_n_active": int(len(active_sats)),
        "ms_n_unserved": int(assoc.n_unserved),
        "ms_assoc_moves": int(assoc.n_moves),

        # Summaries (REQUIRED by helper.flatten_run_record)
        "main": main_summary,
        "main_ref": ref_summary,
        "main_ref_lb": lb_summary,

        "wk_demand_fixed": wk_demand_fixed,
        "wk_demand_rep": wk_demand_rep,
        "wk_qos_fixed": wk_qos_fixed,
        "wk_qos_rep": wk_qos_rep,

        "bk_fixed": bk_fixed,
        "bk_rep": bk_rep,
        "tgbp_fixed": tgbp_fixed,
        "tgbp_rep": tgbp_rep,

        # Timings (REQUIRED by helper.flatten_run_record)
        "time_usergen_s": prof.t.get("usergen", 0.0),
        "time_sat_select_s": prof.t.get("sat_select", 0.0),
        "time_assoc_s": prof.t.get("assoc", 0.0),
        "time_split_s": prof.t.get("split", 0.0),
        "time_ent_ref_s": prof.t.get("ent_ref", 0.0),
        "time_lb_ref_s": prof.t.get("lb_ref", 0.0),

        "eval_calls": prof.c.get("eval_calls", 0),
        "n_splits": prof.c.get("n_splits", 0),
        "ent_moves_tried": prof.c.get("ent_moves_tried", 0),
        "ent_moves_accepted": prof.c.get("ent_moves_accepted", 0),
        "lb_moves_tried": prof.c.get("lb_moves_tried", 0),
        "lb_moves_accepted": prof.c.get("lb_moves_accepted", 0),

        "time_baseline_without_qos_s": prof.t.get("baseline_without_qos", 0.0),
        "time_baseline_with_qos_s": prof.t.get("baseline_with_qos", 0.0),
        "time_baseline_bkmeans_s": prof.t.get("baseline_bkmeans", 0.0),
        "time_baseline_tgbp_s": prof.t.get("baseline_tgbp", 0.0),
    }