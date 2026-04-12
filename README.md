# QoS-Aware User Clustering for LEO Satellite Networks

This repository is a Python research prototype for LEO satellite user association,
beam placement, QoS-aware refinement, payload feasibility checks, and MILP-based
benchmarking.

The current code models one static user/satellite snapshot. It generates users
over a Turkey or debug bounding box, selects visible satellites from either a
synthetic Walker-delta constellation or a TLE file, assigns users to satellites,
clusters users into beams, refines beam membership, and checks satellite payload
limits. It also includes scripts for scaling sweeps, sensitivity studies, and a
Gurobi MILP comparison.

## What Is In The Repo

```text
.
|-- config.py                  # Scenario, PHY, user, satellite, and payload config
|-- main.py                    # Current Phase B scaling sweep runner
|-- run_sensitivity.py         # Editable sensitivity-study runner
|-- run_milp_smoke.py          # Small MILP smoke run
|-- run_milp_compare.py        # Heuristic vs MILP comparison and grid sensitivity
|-- starlink.tle               # TLE data for multisat.source="tle" experiments
|-- sweep_phaseA.csv           # Existing generated sweep output
|-- sweep_phaseB.csv           # Existing/current Phase B sweep output
|-- milp_compare.csv           # Existing/generated MILP comparison output
|-- milp_compare_aggregate.csv # Existing/generated MILP aggregate output
|-- plots/                     # Generated plots and tables
|-- sensitivity_results/       # Generated sensitivity-study outputs
`-- src/
    |-- pipeline.py            # Main heuristic pipeline and comparisons
    |-- satellites.py          # Walker/TLE satellite ranking and user association
    |-- evaluator.py           # Beam geometry, PHY rate, and capacity checks
    |-- usergen.py             # Hotspot/uniform user generation
    |-- models.py              # User containers
    |-- coords.py              # LLH, ECEF, and local XY transforms
    |-- phy.py                 # FSPL, antenna gain, SNR, Shannon rate
    |-- split.py               # Farthest-point bisection
    |-- refine_qos_angle.py    # Enterprise edge-risk refinement
    |-- refine_load_balance.py # Overlap-based load-balance refinement
    |-- helper.py              # Summary, flattening, CSV helpers
    |-- plot.py                # Plotting helpers for scenario visualizations
    |-- sweep_plots.py         # Phase B aggregate plot/table generation
    |-- baselines/
    |   |-- weighted_kmeans.py
    |   |-- fast_beam_placement.py
    |   `-- repair.py
    `-- milp/
        |-- prepare.py
        |-- candidates.py
        |-- compute.py
        |-- model.py
        `-- runner.py
```

## Main Model

The heuristic pipeline in `src/pipeline.py` runs several comparable variants of
the same system:

1. Generate users with QoS classes and demand.
2. Rank visible satellites from `multisat.source`.
3. Search over prefixes of the ranked satellite list to find the smallest
   payload-feasible active set.
4. Associate users to satellites using `balanced_max_elev` for the proposed
   pipeline.
5. Cluster each satellite's users with recursive farthest-point splitting until
   every cluster is feasible.
6. Optionally move enterprise users away from beam edges.
7. Optionally move users between overlapping beams to reduce load imbalance.
8. Repair or offload whole beams between satellites to satisfy payload limits.
9. Run system-level association comparisons, ablations, and baselines.

The beam model is fixed-angle, not fixed-radius. For each beam center:

```text
R_m = d_center * tan(theta_3db)
```

where `d_center` is the satellite-to-beam-center slant range and
`theta_3db` comes from `cfg.beam.theta_3db_deg`.

Beam capacity uses a time-share quantity:

```text
U = sum(demand_mbps / rate_mbps)
```

A beam is feasible when geometry holds and `U <= cfg.payload.W_slots`.
Satellite payload feasibility then checks:

```text
sum_beam U <= cfg.payload.J_lanes * cfg.payload.W_slots
K_sat <= cfg.payload.Ks_max
```

QoS classes are generated as weights `1`, `2`, and `4`. QoS is used for demand
conditioning, enterprise edge-risk accounting, and optional association load
weighting. The default association load mode is physical demand, not
`demand * qos`.

## Defaults Worth Knowing

Important defaults live in `config.py`:

| Block | Default |
| --- | --- |
| Region | `region_mode="turkey"` |
| Users | `n_users=250`, `seed=1` |
| User distribution | 10 hotspots, 15 percent uniform noise |
| Traffic | lognormal demand, 5 Mbps base median |
| QoS probabilities | eco/std/enterprise = `(0.6, 0.3, 0.1)` |
| PHY | 20 GHz, 250 MHz, 40 dBW EIRP, `eta=0.7` |
| Beam | `theta_3db_deg=1.0` |
| Satellite source | `multisat.source="walker"` |
| Walker constellation | 144 sats, 12 planes, 53 deg inclination, 600 km altitude |
| Elevation mask | 25 deg |
| Association load | `demand` |
| Payload | enabled, `J_lanes=8`, `W_slots=8`, `Ks_max=256` |
| FastBP baselines | disabled unless `run.enable_fastbp_baselines=True` |

TLE mode is also supported:

```python
from dataclasses import replace
from config import ScenarioConfig

base = ScenarioConfig()
cfg = replace(
    base,
    multisat=replace(
        base.multisat,
        source="tle",
        tle_path="starlink.tle",
    ),
)
```

## Installation

There is currently no `requirements.txt` or `pyproject.toml`. Install the
runtime dependencies manually in your environment.

Core heuristic runs:

```bash
python -m pip install numpy matplotlib
```

Recommended optional packages:

```bash
python -m pip install scipy scikit-learn pandas skyfield
```

- `scipy` and `scikit-learn` accelerate the Fast Beam Placement baselines.
  The code has fallbacks for some baseline paths if they are absent.
- `pandas` is used by `run_sensitivity.py`.
- `skyfield` is required when `multisat.source="tle"`.
- `gurobipy` plus a working Gurobi license is required for `src/milp/*`,
  `run_milp_smoke.py`, and `run_milp_compare.py`.

The code uses modern Python type syntax, so use Python 3.10 or newer.

## Quick Start

Run one heuristic scenario from Python:

```python
from dataclasses import replace
from config import ScenarioConfig
from src.pipeline import run_scenario

base = ScenarioConfig()
cfg = replace(
    base,
    run=replace(
        base.run,
        n_users=250,
        seed=1,
        verbose=False,
        enable_plots=False,
        enable_fastbp_baselines=False,
    ),
)

result = run_scenario(cfg)

print(result["payload_feasible"])
print(result["payload_best_m"])
print(result["main_ref_lb"]["K"])
print(result["main_ref_lb"]["U_max"])
print(result["main_ref_lb"]["ent_edge_pct"])
```

Run the current scaling sweep:

```bash
python main.py
```

The current `main.py` is configured for Phase B with:

```text
n_users = [1000, 1500, 2000, 2500]
seeds = [1, 2, 3, 4, 5]
region = turkey
enable_fastbp_baselines = True
```

It writes `sweep_phaseB.csv` and regenerates plots/tables under `plots/phaseB/`.
It also calls the Phase B plotter once at startup, so the checked-in
`sweep_phaseB.csv` should be present when running the script as written. Edit
`main.py` directly if you want a different sweep.

Regenerate Phase B plots from an existing CSV:

```bash
python -m src.sweep_plots --csv sweep_phaseB.csv --out plots/phaseB
```

## Sensitivity Studies

`run_sensitivity.py` is an IDE-friendly script with an editable `SETTINGS`
dictionary at the top. It runs one-factor sweeps plus a
`theta_3db_deg x Ks_max` interaction study.

Run it with:

```bash
python run_sensitivity.py
```

It writes:

```text
sensitivity_results/sensitivity_runs.csv
sensitivity_results/sensitivity_summary.csv
sensitivity_results/sensitivity_interaction_runs.csv
sensitivity_results/sensitivity_interaction_summary.csv
sensitivity_results/sensitivity_tables.txt
sensitivity_results/*.png
```

Note that this script currently defaults to `constellation_source="tle"`, so it
needs `skyfield` and a usable `starlink.tle` file unless you change the setting
to `walker`.

## MILP Tools

The MILP code in `src/milp/` builds a grid-candidate mixed-integer model using
the same user generation and satellite ranking as the heuristic pipeline.

MILP preparation:

- generate users
- rank the candidate satellite pool
- build grid centers over the realized user footprint
- create feasible satellite/grid beam candidates
- precompute `demand / rate` coefficients for user-to-candidate assignments

MILP solve variables:

- `x_b`: candidate beam activation
- `z_s`: satellite activation
- `y_u_b`: user assignment to an active candidate beam

MILP constraints include exactly-one user assignment, beam load,
satellite beam count, satellite time budget, and optional prefix-ordered
satellite activation. Objectives are `beam_only` or `weighted_sat_beam`.

Small smoke run:

```bash
python run_milp_smoke.py
```

Full comparison sweep:

```bash
python run_milp_compare.py
```

The comparison script is configured in `CompareSweepConfig` inside
`run_milp_compare.py`. The current defaults are large and can take a long time:
many user counts, five seeds, up to 150 candidate satellites, Gurobi time limits,
and a grid-spacing sensitivity study.

Generated MILP outputs include:

```text
milp_compare.csv
milp_compare_aggregate.csv
milp_grid_sensitivity.csv
milp_grid_sensitivity_aggregate.csv
plots/milp/
plots/milp/grid_sensitivity/
```

## Baselines And Comparisons

The heuristic run records include:

- `main`: split-to-feasible only
- `main_ref`: split plus enterprise edge refinement
- `main_ref_lb`: split plus enterprise refinement plus load-balance refinement
- `sys_pure_max_elev`: fixed-prefix pure max-elevation association comparison
- `sys_balanced_max_elev`: fixed-prefix balanced max-elevation comparison
- `sys_max_service_time`: fixed-prefix service-time proxy comparison
- `ab_A0_pure_split`: pure max elevation plus split
- `ab_A1_bal_split`: balanced max elevation plus split
- `ab_A2_bal_split_qos`: balanced max elevation plus split plus QoS refinement
- `ab_A3_bal_split_qos_lb`: balanced max elevation plus split plus QoS plus load balance
- `wk_demand_rep`: weighted k-means baseline using demand weights and repair
- `wk_qos_rep`: weighted k-means baseline using demand times QoS weights and repair
- `bk_rep`: BK-Means baseline, when FastBP baselines are enabled
- `tgbp_rep`: TGBP baseline, when FastBP baselines are enabled

Common summary metrics:

- `K`: total beams
- `feasible_rate`: cluster feasibility rate
- `U_mean`, `U_max`, `U_min`: beam utilization summaries
- `risk_sum`: enterprise edge-risk penalty
- `ent_edge_pct`: percent of enterprise users with `z > rho_safe`
- `ent_z_mean`, `ent_z_p90`, `ent_z_max`: enterprise radial position summaries
- `radius_*_km`: beam radius summaries
- `sat_mean_radius_*_km`: per-satellite mean beam radius summaries
- `payload_*`: payload feasibility and overflow diagnostics
- `time_*`: runtime components

## Generated Files

This repository contains generated CSVs, plots, and sensitivity outputs. The
runner scripts may overwrite them. Before committing results, check the diff and
decide whether the generated artifacts are intended to be updated.

Useful outputs:

```text
sweep_phaseA.csv
sweep_phaseB.csv
plots/phaseB/phaseB_tables.txt
milp_compare.csv
milp_compare_aggregate.csv
plots/milp/milp_compare_tables.txt
sensitivity_results/sensitivity_tables.txt
```

## Current Limitations

- There is no packaged CLI; the scripts are edited and run directly.
- There is no dependency lockfile in this checkout.
- There is no `tests/` directory in this checkout.
- `run_scenario()` performs several comparisons per call, so even a single
  scenario can be more expensive than a minimal clustering-only run.
- Gurobi is mandatory for MILP code paths.
- No `LICENSE` file is present in this checkout.
