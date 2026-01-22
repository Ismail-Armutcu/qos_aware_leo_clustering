# src/baselines/weighted_kmeans.py
from __future__ import annotations
import numpy as np

from config import ScenarioConfig
from src.baselines.repair import repair_clusters_split_until_feasible
from src.evaluator import evaluate_cluster
from src.helper import summarize
from src.models import Users


def _weighted_choice(rng: np.random.Generator, probs: np.ndarray) -> int:
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s <= 0:
        return int(rng.integers(0, len(probs)))
    probs = probs / s
    return int(rng.choice(len(probs), p=probs))


def weighted_kmeans_pp_init(
    X: np.ndarray,
    K: int,
    sample_w: np.ndarray,
    seed: int = 1
) -> np.ndarray:
    """
    Weighted k-means++ initialization.
    X: (N,2) points
    sample_w: (N,) nonnegative weights (influence seed selection)
    Returns centers: (K,2)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    w = np.maximum(sample_w.astype(float), 0.0)

    centers = np.empty((K, X.shape[1]), dtype=float)

    # 1) first center sampled proportional to weight
    idx0 = _weighted_choice(rng, w)
    centers[0] = X[idx0]

    # distances to nearest center
    d2 = np.sum((X - centers[0])**2, axis=1)

    for k in range(1, K):
        # k-means++ uses prob ~ D(x)^2; weighted variant: prob ~ w_i * D_i^2
        probs = w * d2
        idx = _weighted_choice(rng, probs)
        centers[k] = X[idx]

        # update nearest-center distances
        new_d2 = np.sum((X - centers[k])**2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return centers


def weighted_kmeans(
    X: np.ndarray,
    K: int,
    sample_w: np.ndarray,
    n_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted k-means (Lloyd) with weighted k-means++ init.
    sample_w: weights used in centroid update and init.
    Returns:
      labels: (N,) int
      centers: (K,2)
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    w = np.maximum(np.asarray(sample_w, dtype=float), 0.0)

    centers = weighted_kmeans_pp_init(X, K, w, seed=seed)
    labels = np.zeros(N, dtype=int)

    for it in range(n_iter):
        # assign
        # (N,K) squared distances
        d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        new_labels = np.argmin(d2, axis=1)

        if it > 0 and np.all(new_labels == labels):
            break
        labels = new_labels

        # update centers with weights
        new_centers = np.copy(centers)
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                # re-seed empty cluster: pick a heavy far point
                # choose from points with large distance to its center
                far_scores = w * d2[np.arange(N), labels]
                j = int(np.argmax(far_scores))
                new_centers[k] = X[j]
                continue

            wk = w[idx]
            sw = wk.sum()
            if sw <= 0:
                new_centers[k] = X[idx].mean(axis=0)
            else:
                new_centers[k] = (X[idx] * wk[:, None]).sum(axis=0) / sw

        # convergence
        shift = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + 1e-12)
        centers = new_centers
        if shift < tol:
            break

    return labels, centers


def labels_to_clusters(labels: np.ndarray, K: int) -> list[np.ndarray]:
    clusters = []
    for k in range(K):
        clusters.append(np.where(labels == k)[0].astype(int))
    return clusters

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
