from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.coords import local_xy_to_ecef


@dataclass(frozen=True)
class GridCenter:
    gid: int
    xy_m: np.ndarray
    ecef_m: np.ndarray


class GridCandidateGenerator:
    """Generate grid centers over the realized user footprint."""

    def __init__(
        self,
        *,
        lat0_deg: float,
        lon0_deg: float,
        spacing_m: float = 5_000.0,
        margin_m: float = 0.0,
    ) -> None:
        self.lat0_deg = float(lat0_deg)
        self.lon0_deg = float(lon0_deg)
        self.spacing_m = float(spacing_m)
        self.margin_m = float(margin_m)
        if self.spacing_m <= 0.0:
            raise ValueError("spacing_m must be positive.")

    def build(self, xy_m: np.ndarray) -> list[GridCenter]:
        xy = np.asarray(xy_m, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy_m must have shape (N, 2).")

        xmin = float(np.min(xy[:, 0])) - self.margin_m
        xmax = float(np.max(xy[:, 0])) + self.margin_m
        ymin = float(np.min(xy[:, 1])) - self.margin_m
        ymax = float(np.max(xy[:, 1])) + self.margin_m

        xs = np.arange(xmin, xmax + 0.5 * self.spacing_m, self.spacing_m, dtype=float)
        ys = np.arange(ymin, ymax + 0.5 * self.spacing_m, self.spacing_m, dtype=float)

        out: list[GridCenter] = []
        gid = 0
        for xx in xs:
            for yy in ys:
                xy_c = np.array([xx, yy], dtype=float)
                ecef = np.asarray(local_xy_to_ecef(xx, yy, self.lat0_deg, self.lon0_deg, 0.0), dtype=float)
                out.append(GridCenter(gid=gid, xy_m=xy_c, ecef_m=ecef))
                gid += 1
        return out
