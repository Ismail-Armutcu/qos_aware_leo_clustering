# main.py
from __future__ import annotations

import numpy as np

from config import ScenarioConfig
from src.usergen import generate_users, build_users_container
from src.coords import llh_to_ecef
from src.models import Users
from src.pipeline import split_to_feasible
from src.evaluator import evaluate_cluster
from src.refine_qos_angle import refine_enterprise_by_angle
from src.plot import plot_clusters_overlay

from src.baselines.weighted_kmeans import weighted_kmeans, labels_to_clusters
from src.baselines.repair import repair_clusters_split_until_feasible


def summarize(users: Users, cfg: ScenarioConfig, clusters, evals) -> dict:
    """
    Compute KPIs for a clustering result using eval outputs.
    Adds sanity checks + stronger enterprise metrics.
    """
    K = len(clusters)
    feasible_rate = float(np.mean([ev["feasible"] for ev in evals])) if K > 0 else 0.0

    U = np.array([ev.get("U", np.nan) for ev in evals], dtype=float)
    U = U[~np.isnan(U)]  # in case some ev missing U (shouldn't)

    # --- Enterprise metrics ---
    ent_total = int((users.qos_w == 4).sum())
    ent_exposed = 0

    ent_z_all = []  # collect normalized radii z for enterprise users across all clusters

    for S, ev in zip(clusters, evals):
        # if geom infeasible (shouldn't happen after repair), skip safely
        if ev.get("R_m") is None:
            continue

        # Sanity: z must match cluster size
        z = ev.get("z", None)
        if z is None:
            raise ValueError("evaluate_cluster() did not return 'z'.")
        if len(z) != len(S):
            raise ValueError(f"Mismatch: len(z)={len(z)} but len(cluster)={len(S)}")

        w = users.qos_w[S]
        ent_local = (w == 4)

        if np.any(ent_local):
            z_ent = z[ent_local]
            ent_z_all.append(z_ent)
            ent_exposed += int((z_ent > cfg.rho_safe).sum())

    if ent_total > 0:
        ent_edge_pct = 100.0 * ent_exposed / ent_total
    else:
        ent_edge_pct = 0.0

    if len(ent_z_all) > 0:
        ent_z = np.concatenate(ent_z_all)
        ent_z_mean = float(np.mean(ent_z))
        ent_z_p90 = float(np.quantile(ent_z, 0.90))
        ent_z_max = float(np.max(ent_z))
    else:
        ent_z_mean = ent_z_p90 = ent_z_max = 0.0

    risk_sum = float(np.sum([ev.get("risk", 0.0) for ev in evals])) if K > 0 else 0.0

    return {
        "K": K,
        "feasible_rate": feasible_rate,
        "U_mean": float(np.mean(U)) if U.size > 0 else 0.0,
        "U_max": float(np.max(U)) if U.size > 0 else 0.0,
        "U_min": float(np.min(U)) if U.size > 0 else 0.0,
        "risk_sum": risk_sum,

        # enterprise metrics (more defensible than risk_sum alone)
        "ent_total": ent_total,
        "ent_exposed": ent_exposed,
        "ent_edge_pct": float(ent_edge_pct),
        "ent_z_mean": ent_z_mean,
        "ent_z_p90": ent_z_p90,
        "ent_z_max": ent_z_max,
    }

def print_summary(title: str, s: dict, cfg: ScenarioConfig):
    print(f"\n=== {title} ===")
    print(f"K: {s['K']}")
    print(f"Feasible cluster rate: {s['feasible_rate']*100:.2f}%")
    print(f"Utilization U: mean={s['U_mean']:.3f}, max={s['U_max']:.3f}, min={s['U_min']:.3f}")

    # Enterprise stats
    print(
        f"Enterprise edge exposure: {s['ent_edge_pct']:.2f}% "
        f"({s['ent_exposed']}/{s['ent_total']})  (z > rho={cfg.rho_safe})"
    )
    print(f"Enterprise z: mean={s['ent_z_mean']:.3f}, p90={s['ent_z_p90']:.3f}, max={s['ent_z_max']:.3f}")

    print(f"Total enterprise risk (soft): {s['risk_sum']:.3f}")


def run_weighted_kmeans_baseline(users: Users, cfg: ScenarioConfig, K_ref: int, use_qos_weight: bool):
    """
    Baseline: weighted k-means++ with fixed K, then repair by splitting infeasible clusters.

    IMPORTANT: Evaluate fixed-K clusters using k-means centers:
      - center_xy_override = k-means center (weighted centroid in XY)
      - center_ecef_override = weighted mean in ECEF using same sample_w
    """
    if use_qos_weight:
        sample_w = users.demand_mbps * users.qos_w
        name = "WKMeans++ (weights=demand*qos)"
    else:
        sample_w = users.demand_mbps
        name = "WKMeans++ (weights=demand)"

    labels, centers = weighted_kmeans(
        X=users.xy_m,
        K=K_ref,
        sample_w=sample_w,
        n_iter=50,
        seed=cfg.seed + 999,
    )
    clusters = labels_to_clusters(labels, K_ref)

    # Evaluate fixed-K using baseline-true centers
    evals = []
    for k, S in enumerate(clusters):
        c_xy = centers[k]

        wk = np.maximum(sample_w[S].astype(float), 0.0)
        sw = float(wk.sum())
        if sw > 0:
            c_ecef = (users.ecef_m[S] * wk[:, None]).sum(axis=0) / sw
        else:
            c_ecef = users.ecef_m[S].mean(axis=0)

        ev = evaluate_cluster(
            users, S, cfg,
            center_xy_override=c_xy,
            center_ecef_override=c_ecef
        )
        evals.append(ev)

    fixed_summary = summarize(users, cfg, clusters, evals)

    # Repair (kept as-is; after splitting, k-means centers don't apply anymore)
    clusters_rep, evals_rep, rep_stats = repair_clusters_split_until_feasible(
        users=users,
        cfg=cfg,
        clusters=clusters,
        max_total_clusters=8000,
    )
    rep_summary = summarize(users, cfg, clusters_rep, evals_rep)

    return {
        "name": name,
        "fixedK": {"clusters": clusters, "evals": evals, "summary": fixed_summary},
        "repaired": {"clusters": clusters_rep, "evals": evals_rep, "summary": rep_summary},
        "repair_stats": rep_stats,
    }


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
