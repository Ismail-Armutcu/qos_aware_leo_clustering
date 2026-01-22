# main.py
from __future__ import annotations
from config import ScenarioConfig
from src.helper import flatten_run_record, write_csv
from src.pipeline import run_scenario
import os
import multiprocessing as mp
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def worker(cfg: ScenarioConfig) -> dict[str, Any]:
    rec = run_scenario(cfg)
    return flatten_run_record(rec)


def run_parallel(configs: list[ScenarioConfig], max_workers: int | None = None) -> list[dict[str, Any]]:
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, cfg) for cfg in configs]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows



def main():
    base = ScenarioConfig()
    # Example sweep: 20 seeds robustness
    seeds = list(range(1, 21))
    configs = [
        replace(
            base,
            seed=s,
            enable_plots=False,
            verbose=False,  # requires you added verbose field
            n_users=2500,
            region_mode="turkey",
            use_hotspots=True,
            n_hotspots=10,
            hotspot_sigma_m_min=5_000.0,
            hotspot_sigma_m_max=30_000.0,
            noise_frac=0.15,
        )
        for s in seeds
    ]
    rows = run_parallel(configs, max_workers=None)
    rows.sort(key=lambda r: r["seed"])
    write_csv("sweep_phaseA_parallel.csv", rows)
    print(f"Wrote {len(rows)} runs to sweep_phaseA_parallel.csv")


    # -------------------------
    # Phase B: scaling sweep
    # -------------------------
    n_users_list = [1000, 2500, 5000, 10000]
    seeds_b = list(range(1, 11))

    configs_b = [
        replace(
            base,
            seed=s,
            n_users=n,
            enable_plots=False,
            verbose=False,  # if you added verbose to ScenarioConfig
        )
        for n in n_users_list
        for s in seeds_b
    ]

    rows_b = run_parallel(configs_b, max_workers=None)

    # optional: stable sort for nicer CSV
    rows_b.sort(key=lambda r: (r["n_users"], r["seed"]))

    write_csv("sweep_phaseB_scaling.csv", rows_b)
    print(f"Wrote {len(rows_b)} runs to sweep_phaseB_scaling.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()

