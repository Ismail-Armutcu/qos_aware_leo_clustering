# src/pipeline.py
import numpy as np
from src.evaluator import evaluate_cluster
from src.split import split_farthest


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
