# src/baselines/fast_beam_placement.py
# Implementation of Fast Beam Placement for Ultra-Dense LEO Networks Paper
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from config import ScenarioConfig
from src.baselines.repair import repair_clusters_split_until_feasible
from src.evaluator import evaluate_cluster
from src.helper import summarize
from src.models import Users

# Optional fast deps (recommended)
try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover
    ConvexHull = None  # type: ignore

try:
    from sklearn.cluster import KMeans as _SKKMeans
except Exception:  # pragma: no cover
    _SKKMeans = None  # type: ignore


def _rmax_m_from_cfg(cfg: ScenarioConfig) -> float:
    # Uses the largest footprint radius mode as r_max.
    return float(max(cfg.beam.radius_modes_km)) * 1000.0


def _pairwise_diameter_sq(points_xy: np.ndarray) -> float:
    """
    Exact (or hull-reduced) max squared pairwise distance.
    Uses convex hull reduction for large sets when available.
    """
    m = points_xy.shape[0]
    if m <= 1:
        return 0.0

    pts = points_xy
    if m > 512 and ConvexHull is not None:
        try:
            hull = ConvexHull(points_xy)
            pts = points_xy[hull.vertices]
        except Exception:
            pts = points_xy

    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(d2.max())


def _cluster_is_clique(points_xy: np.ndarray, dist_thresh_m: float) -> bool:
    """
    Clique under threshold graph <=> cluster diameter <= dist_thresh.
    Uses bounding-box diagonal fast reject + exact/hull diameter.
    """
    if points_xy.shape[0] <= 1:
        return True

    mins = points_xy.min(axis=0)
    maxs = points_xy.max(axis=0)
    diag2 = float(np.sum((maxs - mins) ** 2))
    if diag2 > dist_thresh_m * dist_thresh_m:
        return False

    diam2 = _pairwise_diameter_sq(points_xy)
    return diam2 <= dist_thresh_m * dist_thresh_m + 1e-9


def _labels_to_clusters(labels: np.ndarray, K: int) -> list[np.ndarray]:
    clusters: list[np.ndarray] = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size > 0:
            clusters.append(idx.astype(int, copy=False))
    return clusters


# =============================================================================
# TGBP (Two-Phase Graph-Based Placement)
# =============================================================================

@dataclass
class TGBPStats:
    phase1_groups: int
    phase2_rounds: int
    phase2_moves: int
    degrees_mean: float
    degrees_p95: float
    degrees_max: int


def _compute_degrees(xy: np.ndarray, dist_thresh_m: float) -> np.ndarray:
    """
    Degree in the threshold graph. Uses cKDTree when available.
    """
    N = xy.shape[0]
    if cKDTree is None:
        d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        deg = (d2 <= dist_thresh_m * dist_thresh_m).sum(axis=1) - 1
        return deg.astype(int)

    tree = cKDTree(xy)
    deg = tree.query_ball_point(xy, r=dist_thresh_m, return_length=True) - 1
    return np.asarray(deg, dtype=int)


def _tgbp_phase1_greedy_cover(
    xy: np.ndarray,
    dist_thresh_m: float,
    degrees: np.ndarray,
) -> list[np.ndarray]:
    """
    Phase-1: Greedy clique cover:
      - order users by descending degree
      - for each uncovered k: clique = {k}
      - candidates = N(k) ordered by degree desc
      - add u if dist(u, all clique members) <= dist_thresh
    """
    N = xy.shape[0]
    tree = cKDTree(xy) if cKDTree is not None else None

    order = np.argsort(-degrees, kind="mergesort")
    covered = np.zeros(N, dtype=bool)

    groups: list[np.ndarray] = []

    for k in order:
        if covered[k]:
            continue
        covered[k] = True
        group: list[int] = [int(k)]

        if tree is None:
            d2 = np.sum((xy - xy[k]) ** 2, axis=1)
            cand = np.where((d2 <= dist_thresh_m * dist_thresh_m) & (~covered))[0]
        else:
            cand = np.asarray(tree.query_ball_point(xy[k], r=dist_thresh_m), dtype=int)
            cand = cand[~covered[cand]]
            cand = cand[cand != k]

        if cand.size:
            cand = cand[np.argsort(-degrees[cand], kind="mergesort")]

        dist2 = dist_thresh_m * dist_thresh_m
        for u in cand:
            if covered[u]:
                continue
            gxy = xy[np.asarray(group, dtype=int)]
            if np.all(np.sum((gxy - xy[u]) ** 2, axis=1) <= dist2 + 1e-9):
                group.append(int(u))
                covered[u] = True

        groups.append(np.asarray(group, dtype=int))

    return groups


def _tgbp_phase2_balance(
    xy: np.ndarray,
    groups: list[np.ndarray],
    dist_thresh_m: float,
    max_rounds: int = 10,
    max_moves_per_round: int = 50_000,
) -> Tuple[list[np.ndarray], int, int]:
    """
    Phase-2: Reduce imbalance by moving users from large cliques to small cliques
    while preserving clique property (receiver compatibility check).
    Removing from a clique is always safe; only check receiver adjacency.
    """
    if not groups:
        return groups, 0, 0

    dist2 = dist_thresh_m * dist_thresh_m
    moves_total = 0
    rounds_used = 0

    glists: list[list[int]] = [g.tolist() for g in groups]

    for _rnd in range(max_rounds):
        sizes = np.array([len(g) for g in glists], dtype=int)
        if sizes.size == 0:
            break

        order_small = np.argsort(sizes, kind="mergesort")
        order_large = order_small[::-1]

        moved_this_round = 0

        n_pick = min(12, len(glists))
        receivers = order_small[:n_pick]
        donors = order_large[:n_pick]

        for d in donors:
            if moved_this_round >= max_moves_per_round:
                break
            if len(glists[d]) <= 1:
                continue

            for r in receivers:
                if moved_this_round >= max_moves_per_round:
                    break
                if d == r:
                    continue
                if len(glists[d]) - len(glists[r]) <= 1:
                    continue

                recv = glists[r]
                donor = glists[d]

                if len(recv) == 0:
                    recv.append(donor.pop())
                    moved_this_round += 1
                    continue

                recv_xy = xy[np.asarray(recv, dtype=int)]

                step = max(1, len(donor) // 32)
                cand_iter = donor[::step]

                for k in cand_iter:
                    if np.all(np.sum((recv_xy - xy[k]) ** 2, axis=1) <= dist2 + 1e-9):
                        donor.remove(k)
                        recv.append(k)
                        moved_this_round += 1
                        break

        moves_total += moved_this_round
        rounds_used += 1
        if moved_this_round == 0:
            break

    groups_out = [np.asarray(g, dtype=int) for g in glists if len(g) > 0]
    return groups_out, rounds_used, moves_total


def run_tgbp_baseline(
    users: Users,
    cfg: ScenarioConfig,
    *,
    rmax_m: Optional[float] = None,
    do_phase2: bool = True,
    max_rounds: int = 10,
    max_moves_per_round: int = 50_000,
) -> dict[str, Any]:
    """
    TGBP baseline (paper), adapted to your beam model.

    Edge if dist <= 2*r_max.
    Clique cover -> initial clusters, then your repair for capacity feasibility.
    """
    xy = users.xy_m
    rmax = _rmax_m_from_cfg(cfg) if rmax_m is None else float(rmax_m)
    dist_thresh = 2.0 * rmax

    degrees = _compute_degrees(xy, dist_thresh)
    groups = _tgbp_phase1_greedy_cover(xy, dist_thresh, degrees)

    rounds_used = 0
    moves_used = 0
    if do_phase2 and len(groups) > 1:
        groups, rounds_used, moves_used = _tgbp_phase2_balance(
            xy, groups, dist_thresh,
            max_rounds=max_rounds,
            max_moves_per_round=max_moves_per_round,
        )

    evals = [evaluate_cluster(users, S, cfg) for S in groups]
    fixed_summary = summarize(users, cfg, groups, evals)

    clusters_rep, evals_rep, rep_stats = repair_clusters_split_until_feasible(users, cfg, groups)
    rep_summary = summarize(users, cfg, clusters_rep, evals_rep)

    stats = TGBPStats(
        phase1_groups=len(groups),
        phase2_rounds=rounds_used,
        phase2_moves=moves_used,
        degrees_mean=float(np.mean(degrees)) if degrees.size else 0.0,
        degrees_p95=float(np.percentile(degrees, 95)) if degrees.size else 0.0,
        degrees_max=int(np.max(degrees)) if degrees.size else 0,
    )

    return {
        "name": "TGBP",
        "fixedK": {"clusters": groups, "evals": evals, "summary": fixed_summary},
        "repaired": {"clusters": clusters_rep, "evals": evals_rep, "summary": rep_summary},
        "repair_stats": rep_stats,
        "tgbp_stats": stats.__dict__,
    }


# =============================================================================
# BK-Means (Binary search K + K-means + clique feasibility)
# =============================================================================

@dataclass
class BKMeansStats:
    K_found: int
    K_lo_final: int
    K_hi_final: int
    restarts_used: int
    kmeans_calls: int
    clique_checks: int


def _kmeans_fit_predict(
    xy: np.ndarray,
    K: int,
    *,
    seed: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, centers). Uses sklearn if available; otherwise numpy fallback.
    """
    if _SKKMeans is not None:
        km = _SKKMeans(
            n_clusters=K,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = km.fit_predict(xy)
        centers = km.cluster_centers_
        return labels.astype(int, copy=False), centers.astype(float, copy=False)

    # Fallback Lloyd (chunked assignment)
    rng = np.random.default_rng(seed)

    # k-means++ init
    centers = np.empty((K, 2), dtype=float)
    centers[0] = xy[int(rng.integers(0, xy.shape[0]))]
    d2_min = np.sum((xy - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        probs = d2_min / max(d2_min.sum(), 1e-12)
        idx = int(rng.choice(xy.shape[0], p=probs))
        centers[k] = xy[idx]
        d2_new = np.sum((xy - centers[k]) ** 2, axis=1)
        d2_min = np.minimum(d2_min, d2_new)

    labels = np.full(xy.shape[0], -1, dtype=int)

    for _ in range(max_iter):
        new_labels = np.empty_like(labels)

        # assign in chunks
        for i0 in range(0, xy.shape[0], 4096):
            Xc = xy[i0:i0 + 4096]
            d2 = np.sum((Xc[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels[i0:i0 + Xc.shape[0]] = np.argmin(d2, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update
        for k in range(K):
            idx = np.where(labels == k)[0]
            if idx.size > 0:
                centers[k] = xy[idx].mean(axis=0)
            else:
                # re-seed empty cluster
                centers[k] = xy[int(rng.integers(0, xy.shape[0]))]

    return labels, centers


def _all_clusters_clique(
    xy: np.ndarray,
    clusters: list[np.ndarray],
    dist_thresh_m: float,
) -> Tuple[bool, int]:
    checks = 0
    for S in clusters:
        checks += 1
        if not _cluster_is_clique(xy[S], dist_thresh_m):
            return False, checks
    return True, checks


def run_bkmeans_baseline(
    users: Users,
    cfg: ScenarioConfig,
    *,
    K_hint: Optional[int] = None,
    mu_restarts: int = 10,
    max_iter: int = 40,
    seed_offset: int = 9000,
) -> dict[str, Any]:
    """
    BK-Means baseline (paper), adapted to your beam model.

    Finds smallest K such that a K-means partition yields all clusters as cliques
    under threshold dist <= 2*r_max. Then runs your repair for feasibility.
    """
    xy = users.xy_m
    rmax = _rmax_m_from_cfg(cfg)
    dist_thresh = 2.0 * rmax

    N = xy.shape[0]
    rng = np.random.default_rng(int(cfg.run.seed) + int(seed_offset))

    K_hi = 1 if K_hint is None else int(max(1, min(N, K_hint)))

    kmeans_calls = 0
    clique_checks_total = 0
    restarts_used_total = 0

    def feasible_for_K(K: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        nonlocal kmeans_calls, clique_checks_total, restarts_used_total
        for _rr in range(int(mu_restarts)):
            seed = int(rng.integers(0, 2**31 - 1))
            labels, centers = _kmeans_fit_predict(xy, K, seed=seed, max_iter=int(max_iter))
            kmeans_calls += 1

            clusters = _labels_to_clusters(labels, K)

            # IMPORTANT: if some clusters are empty, treat as NOT feasible for this K
            # (otherwise binary-search can “cheat” by collapsing K effectively)
            if len(clusters) < K:
                restarts_used_total += 1
                continue

            ok, checks = _all_clusters_clique(xy, clusters, dist_thresh)
            clique_checks_total += checks
            restarts_used_total += 1
            if ok:
                return True, labels, centers
        return False, None, None

    ok, labels_hi, centers_hi = feasible_for_K(K_hi)
    while (not ok) and (K_hi < N):
        K_hi = min(N, max(K_hi + 1, int(K_hi * 2)))
        ok, labels_hi, centers_hi = feasible_for_K(K_hi)

    if (not ok) or (labels_hi is None) or (centers_hi is None):
        raise RuntimeError("BKMeans: failed to find a feasible K up to N.")

    K_lo = 0
    best_labels = labels_hi
    best_centers = centers_hi

    while K_lo + 1 < K_hi:
        K_mid = (K_lo + K_hi) // 2
        ok_mid, labels_mid, centers_mid = feasible_for_K(K_mid)
        if ok_mid and (labels_mid is not None) and (centers_mid is not None):
            K_hi = K_mid
            best_labels = labels_mid
            best_centers = centers_mid
        else:
            K_lo = K_mid

    clusters_fixed = _labels_to_clusters(best_labels, K_hi)

    # Evaluate fixed clusters using KMeans centers as beam centers (override)
    evals_fixed: list[dict] = []
    for S in clusters_fixed:
        k = int(best_labels[S[0]])
        center_xy = best_centers[k]
        center_ecef = users.ecef_m[S].mean(axis=0)
        ev = evaluate_cluster(
            users, S, cfg,
            center_xy_override=center_xy,
            center_ecef_override=center_ecef,
        )
        evals_fixed.append(ev)

    fixed_summary = summarize(users, cfg, clusters_fixed, evals_fixed)

    clusters_rep, evals_rep, rep_stats = repair_clusters_split_until_feasible(users, cfg, clusters_fixed)
    rep_summary = summarize(users, cfg, clusters_rep, evals_rep)

    stats = BKMeansStats(
        K_found=int(K_hi),
        K_lo_final=int(K_lo),
        K_hi_final=int(K_hi),
        restarts_used=int(restarts_used_total),
        kmeans_calls=int(kmeans_calls),
        clique_checks=int(clique_checks_total),
    )

    return {
        "name": "BKMeans",
        "fixedK": {"clusters": clusters_fixed, "evals": evals_fixed, "summary": fixed_summary},
        "repaired": {"clusters": clusters_rep, "evals": evals_rep, "summary": rep_summary},
        "repair_stats": rep_stats,
        "bkmeans_stats": stats.__dict__,
    }
