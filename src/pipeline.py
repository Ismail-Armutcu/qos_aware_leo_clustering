# src/pipeline.py
from collections import deque

import numpy as np

from config import ScenarioConfig
from src.baselines.weighted_kmeans import run_weighted_kmeans_baseline
from src.baselines.fast_beam_placement import run_bkmeans_baseline, run_tgbp_baseline
from src.evaluator import evaluate_cluster
from src.helper import summarize, print_summary, print_config
from src.plot import plot_clusters_overlay
from src.refine_qos_angle import refine_enterprise_by_angle
from src.refine_load_balance import refine_load_balance_by_overlap
from src.split import split_farthest
from src.profiling import Profiler
from typing import Any

from src.usergen import generate_users, build_users_container


def split_to_feasible(users, cfg, prof=None, max_clusters: int = 5000):
    """
    Queue-based split-to-feasible:
    - Evaluate only the cluster being processed.
    - If infeasible, split it and push children.
    - If feasible, store it.
    This is much faster than re-evaluating all clusters each loop.

    Returns:
      clusters_feas: list[np.ndarray]
      evals_feas: list[dict] aligned
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

        # # Infeasible -> split
        # if S.size <= 2:
        #     # should only happen in truly infeasible scenario (e.g. singleton cap infeasible)
        #     raise RuntimeError(
        #         f"Infeasible tiny cluster |S|={S.size}. reason={ev.get('reason')} U={ev.get('U')}."
        #     )

        seed = cfg.run.seed + n_splits + 1
        S1, S2 = split_farthest(users.xy_m, S, seed=seed)
        S1 = np.asarray(S1, dtype=int)
        S2 = np.asarray(S2, dtype=int)

        # Safety (worth keeping during dev; can remove later)
        if S1.size == 0 or S2.size == 0:
            raise RuntimeError(f"split_farthest produced empty split: |S|={S.size} -> {S1.size},{S2.size}")

        pending.append(S1)
        pending.append(S2)
        n_splits += 1

    if prof:
        prof.c["n_splits"] = n_splits

    return clusters_feas, evals_feas



def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    # ---------------------------
    # 1) Generate and pack users
    # ---------------------------
    prof = Profiler()

    prof.tic("usergen")
    user_list = generate_users(cfg)
    users = build_users_container(user_list, cfg)
    prof.toc("usergen")
    if cfg.run.verbose:
        print(f"Generated {users.n} users in {cfg.region_mode} bbox.")
        print(f"Satellite altitude: {cfg.phy.sat_altitude_m / 1000:.0f} km")
        print(
            f"Beam modes (km): {cfg.beam.radius_modes_km}, B={cfg.phy.bandwidth_hz / 1e6:.0f} MHz, EIRP={cfg.phy.eirp_dbw:.1f} dBW")
    # ------------------------------------
    # 2) Main algorithm: split-to-feasible
    # ------------------------------------
    print_config(cfg)
    prof.tic("split")
    clusters, evals = split_to_feasible(users, cfg, prof=prof)
    prof.toc("split")
    main_summary = summarize(users, cfg, clusters, evals)
    if cfg.run.verbose:
        print_summary("Main algorithm (split-to-feasible)", main_summary, cfg)
    if cfg.run.enable_plots:
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=clusters,
            evals=evals,
            title=f"Main algorithm (K={len(clusters)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
    # -----------------------------
    # Refinement: enterprise by angle
    # -----------------------------
    prof.tic("ent_ref")
    clusters_ref, evals_ref, ref_stats = refine_enterprise_by_angle(
        users, cfg, clusters, evals,
        n_rounds=cfg.qos_refine.rounds,
        kcand=cfg.qos_refine.kcand,
        max_moves_per_round=cfg.qos_refine.max_moves_per_round,
    )
    prof.toc("ent_ref")
    prof.c["ent_moves_tried"] = int(ref_stats.get("moves_tried", 0))
    prof.c["ent_moves_accepted"] = int(ref_stats.get("moves_accepted", 0))

    ref_summary = summarize(users, cfg, clusters_ref, evals_ref)
    if cfg.run.verbose:
        print("\nQoS refinement stats:", ref_stats)
        print_summary("Main algorithm + enterprise angle refinement", ref_summary, cfg)
    if cfg.run.enable_plots:
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=clusters_ref,
            evals=evals_ref,
            title=f"Main + QoS refine (K={len(clusters_ref)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
    # -----------------------------
    # Refinement: load balance
    # -----------------------------
    prof.tic("lb_ref")
    clusters_lb, evals_lb, lb_stats = refine_load_balance_by_overlap(users, cfg, clusters_ref, evals_ref)
    prof.toc("lb_ref")
    prof.c["lb_moves_tried"] = int(lb_stats.get("moves_tried", 0))
    prof.c["lb_moves_accepted"] = int(lb_stats.get("moves_accepted", 0))
    lb_summary = summarize(users, cfg, clusters_lb, evals_lb)
    if cfg.run.verbose:
        print("\nLoad-balance refinement stats:", lb_stats)
        print_summary("Main + enterprise refine + load-balance refine", lb_summary, cfg)
    if cfg.run.enable_plots:
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=clusters_lb,
            evals=evals_lb,
            title=f"Main + QoS refine + LB refine (K={len(clusters_lb)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )

    # ---------------------------------------------
    # 3) Baselines: WKMeans++ fixed-K then repair
    # ---------------------------------------------
    K_ref = len(clusters)
    prof.tic("baseline_without_qos")
    baseline_without_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=False)
    prof.toc("baseline_without_qos")
    prof.tic("baseline_with_qos")
    baseline_with_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=True)
    prof.toc("baseline_with_qos")
    if cfg.run.verbose:
        print_summary(f"Baseline without QOS {baseline_without_qos['name']} (fixed K={K_ref})",
                      baseline_without_qos["fixedK"]["summary"], cfg)
        rep_stats = baseline_without_qos["repair_stats"]
        print(
            f"\nRepair stats: K_before={rep_stats['K_before']}, K_after={rep_stats['K_after']}, splits={rep_stats['n_splits']}")
        print_summary(f"Baseline without QOS {baseline_without_qos['name']} (after repair)",
                      baseline_without_qos["repaired"]["summary"], cfg)
        print_summary(f"Baseline with QOS {baseline_with_qos['name']} (fixed K={K_ref})",
                      baseline_with_qos["fixedK"]["summary"], cfg)
        rep_stats = baseline_with_qos["repair_stats"]
        print(
            f"\nRepair stats: K_before={rep_stats['K_before']}, K_after={rep_stats['K_after']}, splits={rep_stats['n_splits']}")
        print_summary(f"Baseline with QOS {baseline_with_qos['name']} (after repair)",
                      baseline_with_qos["repaired"]["summary"], cfg)
    if cfg.run.enable_plots:
        # demand baseline plots
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=baseline_without_qos["fixedK"]["clusters"],
            evals=baseline_without_qos["fixedK"]["evals"],
            title=f"Baseline demand fixed-K (K={len(baseline_without_qos['fixedK']['clusters'])})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=baseline_without_qos["repaired"]["clusters"],
            evals=baseline_without_qos["repaired"]["evals"],
            title=f"Baseline demand repaired (K={len(baseline_without_qos['repaired']['clusters'])})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        # qos baseline plots
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=baseline_with_qos["fixedK"]["clusters"],
            evals=baseline_with_qos["fixedK"]["evals"],
            title=f"Baseline demand*qos fixed-K (K={len(baseline_with_qos['fixedK']['clusters'])})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=users.xy_m,
            qos_w=users.qos_w,
            clusters=baseline_with_qos["repaired"]["clusters"],
            evals=baseline_with_qos["repaired"]["evals"],
            title=f"Baseline demand*qos repaired (K={len(baseline_with_qos['repaired']['clusters'])})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )

    # ---------------------------------------------
    # 4) Baselines from "Fast Beam Placement" paper
    # ---------------------------------------------
    EMPTY_SUMMARY = {
        "K": float("nan"),
        "feasible_rate": float("nan"),
        "U_mean": float("nan"),
        "U_max": float("nan"),
        "U_min": float("nan"),
        "risk_sum": float("nan"),
        "ent_total": float("nan"),
        "ent_exposed": float("nan"),
        "ent_edge_pct": float("nan"),
        "ent_z_mean": float("nan"),
        "ent_z_p90": float("nan"),
        "ent_z_max": float("nan"),
    }

    if cfg.run.enable_fastbp_baselines:
        prof.tic("baseline_bkmeans")
        baseline_bkmeans = run_bkmeans_baseline(users, cfg, K_hint=K_ref, mu_restarts=10, max_iter=40)
        prof.toc("baseline_bkmeans")

        prof.tic("baseline_tgbp")
        baseline_tgbp = run_tgbp_baseline(users, cfg, do_phase2=True, max_rounds=10)
        prof.toc("baseline_tgbp")

        if cfg.run.verbose:
            print_summary(f"Baseline BKMeans (fixed K={baseline_bkmeans['fixedK']['summary']['K']})",
                          baseline_bkmeans["fixedK"]["summary"], cfg)
            print_summary("Baseline BKMeans (after repair)",
                          baseline_bkmeans["repaired"]["summary"], cfg)

            print_summary(f"Baseline TGBP (fixed K={baseline_tgbp['fixedK']['summary']['K']})",
                          baseline_tgbp["fixedK"]["summary"], cfg)
            print_summary("Baseline TGBP (after repair)",
                          baseline_tgbp["repaired"]["summary"], cfg)
    else:
        baseline_bkmeans = {"fixedK": {"summary": EMPTY_SUMMARY}, "repaired": {"summary": EMPTY_SUMMARY}}
        baseline_tgbp = {"fixedK": {"summary": EMPTY_SUMMARY}, "repaired": {"summary": EMPTY_SUMMARY}}
        # keep profiler keys stable
        prof.t["baseline_bkmeans"] = 0.0
        prof.t["baseline_tgbp"] = 0.0
    # ---------------------------
    # Return record for sweeps
    # ---------------------------
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
        "main": main_summary,
        "main_ref": ref_summary,
        "main_ref_lb": lb_summary,
        "wk_demand_fixed": baseline_without_qos["fixedK"]["summary"],
        "wk_demand_rep": baseline_without_qos["repaired"]["summary"],
        "wk_qos_fixed": baseline_with_qos["fixedK"]["summary"],
        "wk_qos_rep": baseline_with_qos["repaired"]["summary"],
        "time_usergen_s": prof.t.get("usergen", 0.0),
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
        "bk_fixed": baseline_bkmeans["fixedK"]["summary"],
        "bk_rep": baseline_bkmeans["repaired"]["summary"],
        "tgbp_fixed": baseline_tgbp["fixedK"]["summary"],
        "tgbp_rep": baseline_tgbp["repaired"]["summary"],

        "time_baseline_bkmeans_s": prof.t.get("baseline_bkmeans", 0.0),
        "time_baseline_tgbp_s": prof.t.get("baseline_tgbp", 0.0),
    }