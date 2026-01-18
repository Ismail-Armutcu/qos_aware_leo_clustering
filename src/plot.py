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

    - draw_circles: draws footprint circles using eval['center_xy'] and eval['R_m']
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

    # Decide whether to draw circles
    circles_ok = draw_circles and (K <= max_circles)

    for ev in evals:
        c = ev.get("center_xy", None)
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
