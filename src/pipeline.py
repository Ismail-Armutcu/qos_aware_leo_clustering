# src/pipeline.py
import numpy as np

from config import ScenarioConfig
from src.baselines.weighted_kmeans import run_weighted_kmeans_baseline
from src.evaluator import evaluate_cluster
from src.helper import summarize, print_summary
from src.plot import plot_clusters_overlay
from src.refine_qos_angle import refine_enterprise_by_angle
from src.split import split_farthest
from typing import Any

from src.usergen import generate_users, build_users_container


def split_to_feasible(users, cfg, max_clusters: int = 5000):
    """
    Phase 1: Start with 1 cluster. Split only when infeasible (geom or capacity).
    This naturally drives toward minimal K without guessing K.
    """
    clusters: list[np.ndarray] = [np.arange(users.n, dtype=int)]

    while True:
        if len(clusters) > max_clusters:
            raise RuntimeError("Too many clusters. Check demands/PHY parameters.")

        evals = [evaluate_cluster(users, S, cfg) for S in clusters]

        # Find first infeasible cluster by priority: geom -> cap
        bad_idx = None
        for i, ev in enumerate(evals):
            if (not ev["feasible"]) and ev["reason"] == "geom":
                bad_idx = i
                break
        if bad_idx is None:
            for i, ev in enumerate(evals):
                if (not ev["feasible"]) and ev["reason"] == "cap":
                    bad_idx = i
                    break

        if bad_idx is None:
            return clusters, evals

        # Split the bad cluster
        S = clusters.pop(bad_idx)
        S1, S2 = split_farthest(users.xy_m, S, seed=cfg.seed + len(clusters) + 1)
        clusters.append(S1)
        clusters.append(S2)



def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    # ---------------------------
    # 1) Generate and pack users
    # ---------------------------
    user_list = generate_users(cfg)
    users = build_users_container(user_list, cfg)
    if cfg.verbose:
        print(f"Generated {users.n} users in {cfg.region_mode} bbox.")
        print(f"Satellite altitude: {cfg.sat_altitude_m / 1000:.0f} km")
        print(
            f"Beam modes (km): {cfg.radius_modes_km}, B={cfg.bandwidth_hz / 1e6:.0f} MHz, EIRP={cfg.eirp_dbw:.1f} dBW")
    # ------------------------------------
    # 2) Main algorithm: split-to-feasible
    # ------------------------------------
    clusters, evals = split_to_feasible(users, cfg)
    main_summary = summarize(users, cfg, clusters, evals)
    if cfg.verbose:
        print_summary("Main algorithm (split-to-feasible)", main_summary, cfg)
    if cfg.enable_plots:
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
    clusters_ref, evals_ref, ref_stats = refine_enterprise_by_angle(
        users, cfg, clusters, evals,
        n_rounds=cfg.qos_refine_rounds,
        kcand=cfg.qos_refine_kcand,
        max_moves_per_round=cfg.qos_refine_max_moves_per_round,
    )
    ref_summary = summarize(users, cfg, clusters_ref, evals_ref)
    if cfg.verbose:
        print("\nQoS refinement stats:", ref_stats)
        print_summary("Main algorithm + enterprise angle refinement", ref_summary, cfg)
    if cfg.enable_plots:
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
    # ---------------------------------------------
    # 3) Baselines: WKMeans++ fixed-K then repair
    # ---------------------------------------------
    K_ref = len(clusters)
    baseline_without_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=False)
    baseline_with_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=True)
    if cfg.verbose:
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
    if cfg.enable_plots:
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
    # ---------------------------
    # Return record for sweeps
    # ---------------------------
    return {
        "seed": cfg.seed,
        "region_mode": cfg.region_mode,
        "n_users": cfg.n_users,
        "use_hotspots": cfg.use_hotspots,
        "n_hotspots": cfg.n_hotspots,
        "noise_frac": cfg.noise_frac,
        "sigma_min": cfg.hotspot_sigma_m_min,
        "sigma_max": cfg.hotspot_sigma_m_max,
        "rho_safe": cfg.rho_safe,
        "eirp_dbw": cfg.eirp_dbw,
        "bandwidth_hz": cfg.bandwidth_hz,
        "radius_modes_km": cfg.radius_modes_km,
        "main": main_summary,
        "main_ref": ref_summary,
        "wk_demand_fixed": baseline_without_qos["fixedK"]["summary"],
        "wk_demand_rep": baseline_without_qos["repaired"]["summary"],
        "wk_qos_fixed": baseline_with_qos["fixedK"]["summary"],
        "wk_qos_rep": baseline_with_qos["repaired"]["summary"],
    }