# src/split.py
import numpy as np


def split_farthest(users_xy_m: np.ndarray, cluster_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast bisection:
    pick a point -> farthest A -> farthest B from A, then split by proximity to A/B.
    """
    rng = np.random.default_rng(seed)
    xy = users_xy_m[cluster_ids]

    i0 = int(rng.integers(0, len(cluster_ids)))
    a = int(np.argmax(np.linalg.norm(xy - xy[i0], axis=1)))
    b = int(np.argmax(np.linalg.norm(xy - xy[a], axis=1)))

    A = xy[a]
    B = xy[b]

    dA = np.linalg.norm(xy - A[None, :], axis=1)
    dB = np.linalg.norm(xy - B[None, :], axis=1)
    mask = dA <= dB

    S1 = cluster_ids[mask]
    S2 = cluster_ids[~mask]

    # Safety
    if len(S1) == 0 or len(S2) == 0:
        mid = len(cluster_ids) // 2
        return cluster_ids[:mid], cluster_ids[mid:]
    return S1, S2
