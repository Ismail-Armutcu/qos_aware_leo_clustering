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
import time


def worker(cfg: ScenarioConfig) -> dict[str, Any]:
    rec = run_scenario(cfg)
    return flatten_run_record(rec)


# def run_parallel(configs: list[ScenarioConfig], max_workers: int | None = None) -> list[dict[str, Any]]:
#     if max_workers is None:
#         max_workers = max(1, (os.cpu_count() or 4) - 1)
#
#     rows: list[dict[str, Any]] = []
#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         futs = [ex.submit(worker, cfg) for cfg in configs]
#         for fut in as_completed(futs):
#             rows.append(fut.result())
#     return rows

def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"

def run_parallel(configs: list[ScenarioConfig], max_workers: int | None = None) -> list[dict[str, Any]]:
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    total = len(configs)
    if total == 0:
        return []

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    last_print = 0.0
    print_every_sec = 1.0  # throttle prints

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, cfg) for cfg in configs]

        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1

            now = time.time()
            if (now - last_print) >= print_every_sec or done == total:
                elapsed = now - t0
                rate = done / max(elapsed, 1e-9)
                eta = (total - done) / max(rate, 1e-9)

                pct = 100.0 * done / total
                print(f"[{done:>4d}/{total}] {pct:6.2f}% | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta)}")
                last_print = now

    return rows




def main():
    base = ScenarioConfig()

    # -------------------------
    # Phase A: robustness sweep
    # -------------------------
    seeds = list(range(1, 11))

    configs = []
    for s in seeds:
        cfg = replace(
            base,
            region_mode="turkey",
            run=replace(
                base.run,
                seed=s,
                n_users=2500,
                enable_plots=False,
                verbose=False,
                enable_fastbp_baselines=True
            ),
            usergen=replace(
                base.usergen,
                enabled=True,
                n_hotspots=10,
                hotspot_sigma_m_min=5_000.0,
                hotspot_sigma_m_max=30_000.0,
                noise_frac=0.15,
            ),
            lb_refine=replace(
                base.lb_refine,
                enabled=True
            )
        )
        configs.append(cfg)

    rows = run_parallel(configs, max_workers=None)

    # Your flatten_run_record should already output seed; but now seed lives in cfg.run.seed
    # Make sure your record uses cfg.run.seed (see note below).
    rows.sort(key=lambda r: r["seed"])
    write_csv("sweep_phaseA_low_demand.csv", rows)
    print(f"Wrote {len(rows)} runs to sweep_phaseA_parallel.csv")

    # -------------------------
    # Phase B: scaling sweep
    # -------------------------
    n_users_list = [1000, 2500, 5000, 10000, 15000, 20000, 25000, 50000]
    seeds_b = list(range(1, 5))

    configs_b = []
    for n in n_users_list:
        for s in seeds_b:
            cfg = replace(
                base,
                region_mode="turkey",
                run=replace(
                    base.run,
                    seed=s,
                    n_users=n,
                    enable_plots=False,
                    verbose=False,
                    enable_fastbp_baselines=True
                ),
                # keep the same user generator settings as base (or set explicitly if you want)
                # usergen=replace(base.usergen, enabled=True, ...)
            )
            configs_b.append(cfg)

    rows_b = run_parallel(configs_b, max_workers=None)
    rows_b.sort(key=lambda r: (r["n_users"], r["seed"]))
    write_csv("sweep_phaseB_scaling.csv", rows_b)
    print(f"Wrote {len(rows_b)} runs to sweep_phaseB_scaling.csv")

if __name__ == "__main__":
    mp.freeze_support()
    main()

