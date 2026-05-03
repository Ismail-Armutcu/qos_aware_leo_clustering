# src/split.py
from __future__ import annotations

import numpy as np


def split_farthest(users_xy_m: np.ndarray, cluster_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Original fast geometry-only bisection.

    Pick a point -> farthest A -> farthest B from A, then split by proximity to A/B.
    Kept unchanged for baselines and rollback.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids, dtype=int)

    if cluster_ids.size <= 1:
        return cluster_ids.copy(), np.array([], dtype=int)

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
        return cluster_ids[:mid].astype(int), cluster_ids[mid:].astype(int)
    return S1.astype(int), S2.astype(int)


def _choose_enterprise_anchors(users, S: np.ndarray, seed: int) -> tuple[int, int] | None:
    """
    Choose split anchors with enterprise priority.

    - If >=2 enterprise users exist, both anchors are enterprise users.
    - If exactly 1 enterprise user exists, one anchor is that enterprise user.
    - If no enterprise exists, caller falls back to split_farthest().
    """
    S = np.asarray(S, dtype=int)
    q = users.qos_w[S]
    ent_ids = S[q == 4]

    if ent_ids.size == 0:
        return None

    if ent_ids.size >= 2:
        ent_xy = users.xy_m[ent_ids]
        c_ent = ent_xy.mean(axis=0)

        a_local = int(np.argmax(np.linalg.norm(ent_xy - c_ent[None, :], axis=1)))
        a_id = int(ent_ids[a_local])

        d = np.linalg.norm(ent_xy - users.xy_m[a_id][None, :], axis=1)
        b_local = int(np.argmax(d))
        b_id = int(ent_ids[b_local])

        if a_id != b_id:
            return a_id, b_id

    a_id = int(ent_ids[0])
    d = np.linalg.norm(users.xy_m[S] - users.xy_m[a_id][None, :], axis=1)
    b_id = int(S[np.argmax(d)])

    if a_id == b_id:
        return None
    return a_id, b_id


def split_enterprise_first(users, cluster_ids: np.ndarray, cfg, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    QoS-aware bisection for the proposed method.

    Enterprise users are used as split anchors when available. If the cluster has no
    enterprise users, the original split_farthest() is used. This keeps behavior stable
    in purely eco/standard clusters.
    """
    S = np.asarray(cluster_ids, dtype=int)
    if S.size <= 1:
        return S.copy(), np.array([], dtype=int)

    anchors = _choose_enterprise_anchors(users, S, seed=seed)
    if anchors is None:
        return split_farthest(users.xy_m, S, seed=seed)

    a_id, b_id = anchors
    A = users.xy_m[a_id]
    B = users.xy_m[b_id]

    dA = np.linalg.norm(users.xy_m[S] - A[None, :], axis=1)
    dB = np.linalg.norm(users.xy_m[S] - B[None, :], axis=1)
    mask = dA <= dB

    S1 = S[mask]
    S2 = S[~mask]

    if S1.size == 0 or S2.size == 0:
        return split_farthest(users.xy_m, S, seed=seed)

    return S1.astype(int), S2.astype(int)


def split_cluster(
    users,
    cluster_ids: np.ndarray,
    cfg,
    seed: int,
    *,
    split_mode_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split dispatcher.

    Proposed method uses cfg.split.mode.
    Baselines can force original behavior using split_mode_override="farthest".
    """
    if split_mode_override is not None:
        mode = str(split_mode_override).lower().strip()
    else:
        mode = "farthest"
        if hasattr(cfg, "split"):
            mode = str(getattr(cfg.split, "mode", "farthest")).lower().strip()

    if mode == "enterprise_first":
        return split_enterprise_first(users, cluster_ids, cfg, seed=seed)
    if mode in {"farthest", "geometric", "original"}:
        return split_farthest(users.xy_m, np.asarray(cluster_ids, dtype=int), seed=seed)

    raise ValueError(f"Unknown split mode: {mode!r}. Expected 'farthest' or 'enterprise_first'.")
