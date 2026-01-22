# main.py
from __future__ import annotations

import numpy as np

from config import ScenarioConfig
from src.helper import summarize, print_summary
from src.usergen import generate_users, build_users_container
from src.coords import llh_to_ecef
from src.models import Users
from src.pipeline import split_to_feasible
from src.evaluator import evaluate_cluster
from src.refine_qos_angle import refine_enterprise_by_angle
from src.plot import plot_clusters_overlay

from src.baselines.weighted_kmeans import weighted_kmeans, labels_to_clusters, run_weighted_kmeans_baseline
from src.baselines.repair import repair_clusters_split_until_feasible








def main():
    cfg = ScenarioConfig()

    # ---------------------------
    # 1) Generate and pack users
    # ---------------------------
    user_list = generate_users(cfg)
    users = build_users_container(user_list, cfg)

    print(f"Generated {users.n} users in {cfg.region_mode} bbox.")
    print(f"Satellite altitude: {cfg.sat_altitude_m/1000:.0f} km")
    print(f"Beam modes (km): {cfg.radius_modes_km}, B={cfg.bandwidth_hz/1e6:.0f} MHz, EIRP={cfg.eirp_dbw:.1f} dBW")

    # ------------------------------------
    # 2) Main algorithm: split-to-feasible
    # ------------------------------------
    clusters, evals = split_to_feasible(users, cfg)
    main_summary = summarize(users, cfg, clusters, evals)
    print_summary("Main algorithm (split-to-feasible)", main_summary, cfg)

    # Plot main clustering (circles may be skipped if K is high)
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

    clusters_ref, evals_ref, ref_stats = refine_enterprise_by_angle(
        users, cfg, clusters, evals, n_rounds=cfg.qos_refine_rounds, kcand=cfg.qos_refine_kcand,
        max_moves_per_round=cfg.qos_refine_max_moves_per_round
    )

    print("\nQoS refinement stats:", ref_stats)

    ref_summary = summarize(users, cfg, clusters_ref, evals_ref)
    print_summary("Main algorithm + enterprise angle refinement", ref_summary, cfg)


    # (optional) plot refined
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
    # 3) Baseline: weighted k-means++ with fixed K
    #    then repair to enforce feasibility
    # ---------------------------------------------
    K_ref = len(clusters)

    baseline_without_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=False)
    print_summary(f"Baseline without QOS {baseline_without_qos['name']} (fixed K={K_ref})", baseline_without_qos["fixedK"]["summary"], cfg)

    rep_stats = baseline_without_qos["repair_stats"]
    print(f"\nRepair stats: K_before={rep_stats['K_before']}, K_after={rep_stats['K_after']}, splits={rep_stats['n_splits']}")

    print_summary(f"Baseline without QOS {baseline_without_qos['name']} (after repair)", baseline_without_qos["repaired"]["summary"], cfg)

    # Plot baseline fixed-K clustering
    plot_clusters_overlay(
        users_xy_m=users.xy_m,
        qos_w=users.qos_w,
        clusters=baseline_without_qos["fixedK"]["clusters"],
        evals=baseline_without_qos["fixedK"]["evals"],
        title=f"Baseline without QOS fixed-K (K={len(baseline_without_qos['fixedK']['clusters'])})",
        draw_circles=True,
        draw_centers=True,
        max_circles=250,
    )

    # Plot baseline repaired clustering
    plot_clusters_overlay(
        users_xy_m=users.xy_m,
        qos_w=users.qos_w,
        clusters=baseline_without_qos["repaired"]["clusters"],
        evals=baseline_without_qos["repaired"]["evals"],
        title=f"Baseline without QOS after repair (K={len(baseline_without_qos['repaired']['clusters'])})",
        draw_circles=True,
        draw_centers=True,
        max_circles=250,
    )

    baseline_with_qos = run_weighted_kmeans_baseline(users, cfg, K_ref=K_ref, use_qos_weight=True)
    print_summary(f"Baseline with QOS {baseline_with_qos['name']} (fixed K={K_ref})",
                  baseline_with_qos["fixedK"]["summary"], cfg)

    rep_stats = baseline_with_qos["repair_stats"]
    print(
        f"\nRepair stats: K_before={rep_stats['K_before']}, K_after={rep_stats['K_after']}, splits={rep_stats['n_splits']}")

    print_summary(f"Baseline with QOS {baseline_with_qos['name']} (after repair)",
                  baseline_with_qos["repaired"]["summary"], cfg)


    # Plot baseline fixed-K clustering
    plot_clusters_overlay(
        users_xy_m=users.xy_m,
        qos_w=users.qos_w,
        clusters=baseline_with_qos["fixedK"]["clusters"],
        evals=baseline_with_qos["fixedK"]["evals"],
        title=f"Baseline with QOS fixed-K (K={len(baseline_with_qos['fixedK']['clusters'])})",
        draw_circles=True,
        draw_centers=True,
        max_circles=250,
    )

    # Plot baseline repaired clustering
    plot_clusters_overlay(
        users_xy_m=users.xy_m,
        qos_w=users.qos_w,
        clusters=baseline_with_qos["repaired"]["clusters"],
        evals=baseline_with_qos["repaired"]["evals"],
        title=f"Baseline with QOS after repair (K={len(baseline_with_qos['repaired']['clusters'])})",
        draw_circles=True,
        draw_centers=True,
        max_circles=250,
    )


if __name__ == "__main__":
    main()
