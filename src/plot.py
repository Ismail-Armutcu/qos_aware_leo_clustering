# src/plot.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_users_by_qos(users_xy_m: np.ndarray, qos_w: np.ndarray, title: str = ""):
    """
    Scatter plot of users colored by QoS class:
    eco=1, std=2, ent=4
    """
    xy = users_xy_m
    qos = qos_w

    # masks
    m_eco = (qos == 1)
    m_std = (qos == 2)
    m_ent = (qos == 4)

    plt.figure()
    plt.scatter(xy[m_eco, 0], xy[m_eco, 1], s=6, c="tab:blue",  label="Eco (w=1)", alpha=0.6)
    plt.scatter(xy[m_std, 0], xy[m_std, 1], s=6, c="tab:orange",label="Std (w=2)", alpha=0.6)
    plt.scatter(xy[m_ent, 0], xy[m_ent, 1], s=10, c="tab:red",  label="Ent (w=4)", alpha=0.8)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title or "Users by QoS")
    plt.xlabel("x East (m)")
    plt.ylabel("y North (m)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.2)
    plt.show()


def plot_clusters_overlay(
    users_xy_m: np.ndarray,
    qos_w: np.ndarray,
    clusters: list[np.ndarray],
    evals: list[dict],
    title: str = "",
    draw_circles: bool = True,
    draw_centers: bool = True,
    max_circles: int = 300,
):
    """
    Plot users by QoS class and overlay clustering result.

    - draw_circles: draws footprint circles using eval['center_xy_m'] and eval['R_m']
    - draw_centers: draws cluster centers
    - max_circles: if K exceeds this, circles are skipped automatically (centers still drawn)
    """
    xy = users_xy_m
    qos = qos_w

    # masks
    m_eco = (qos == 1)
    m_std = (qos == 2)
    m_ent = (qos == 4)

    K = len(clusters)

    plt.figure()
    plt.scatter(xy[m_eco, 0], xy[m_eco, 1], s=6,  c="tab:blue",   label="Eco (w=1)", alpha=0.6)
    plt.scatter(xy[m_std, 0], xy[m_std, 1], s=6,  c="tab:orange", label="Std (w=2)", alpha=0.6)
    plt.scatter(xy[m_ent, 0], xy[m_ent, 1], s=10, c="tab:red",    label="Ent (w=4)", alpha=0.85)

    ax = plt.gca()

    circles_ok = draw_circles and (K <= max_circles)

    for ev in evals:
        c = ev.get("center_xy_m", None)
        R = ev.get("R_m", None)
        if c is None:
            continue

        if draw_centers:
            ax.scatter([c[0]], [c[1]], s=15, c="black", marker="x", alpha=0.8)

        if circles_ok and (R is not None):
            circ = plt.Circle((c[0], c[1]), float(R), fill=False, linewidth=1, alpha=0.7)
            ax.add_patch(circ)

    if draw_circles and not circles_ok:
        plt.text(
            0.01, 0.01,
            f"Circles skipped (K={K} > max_circles={max_circles})",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.8
        )

    ax.set_aspect("equal", adjustable="box")
    plt.title(title or f"Clusters overlay (K={K})")
    plt.xlabel("x East (m)")
    plt.ylabel("y North (m)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.2)
    plt.show()


def plot_payload_sat_stats(
    T_by_sat: np.ndarray,
    K_by_sat: np.ndarray,
    T_cap: float,
    K_cap: int,
    J_lanes: float,
    W_slots: int,
    title: str = "",
) -> None:
    """
    Debug visualization for payload feasibility on a chosen satellite prefix.

    Inputs:
      - T_by_sat: (S,) per-satellite time load, T_s = sum_b U_{s,b}
      - K_by_sat: (S,) per-satellite beam count, K_s = |B_s|
      - T_cap   : per-satellite time cap, J_lanes * W_slots
      - K_cap   : per-satellite beam cap, Ks_max
      - J_lanes : #parallel beam "lanes"
      - W_slots : #slots in beam-hopping window

    Shows:
      1) Bar plot of T_s with cap line
      2) Bar plot of K_s with cap line
      3) Bar plot of per-sat W_min_req_s = ceil(T_s / J_lanes) with W_slots line
    """
    T = np.asarray(T_by_sat, dtype=float).copy()
    K = np.asarray(K_by_sat, dtype=float).copy()

    if T.size == 0:
        return

    S = int(T.size)
    x = np.arange(S, dtype=int)

    # --- T_s bars ---
    plt.figure()
    plt.bar(x, T)
    plt.axhline(float(T_cap), linestyle="--", linewidth=1.5)
    plt.title((title + " - Time load") if title else "Payload per-satellite time load")
    plt.xlabel("satellite index in prefix")
    plt.ylabel("T_s = sum_b U_{s,b}")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

    # --- K_s bars ---
    plt.figure()
    plt.bar(x, K)
    plt.axhline(float(K_cap), linestyle="--", linewidth=1.5)
    plt.title((title + " - Beam count") if title else "Payload per-satellite beam count")
    plt.xlabel("satellite index in prefix")
    plt.ylabel("K_s = |B_s|")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

    # --- required window per satellite ---
    plt.figure()
    Wreq = np.ceil(T / max(float(J_lanes), 1e-12))
    plt.bar(x, Wreq)
    plt.axhline(float(W_slots), linestyle="--", linewidth=1.5)
    plt.title((title + " - Window requirement") if title else "Required beam-hopping window per satellite")
    plt.xlabel("satellite index in prefix")
    plt.ylabel("W_min_req_s = ceil(T_s / J_lanes)")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()