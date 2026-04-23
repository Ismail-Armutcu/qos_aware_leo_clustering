# Thesis Project Technical Report

Project: QoS-Aware User Clustering for LEO Satellite Networks  
Repository analyzed: `qos_aware_user_clustering`  
Report date: 2026-04-15  
Prepared for: thesis supervision discussion

## 1. Executive Summary

This repository is a Python research prototype for QoS-aware beam placement and
user clustering in Low Earth Orbit (LEO) satellite networks. It models a static
snapshot of users over a geographic region, selects visible satellites, assigns
users to satellites, clusters users into satellite beams, refines beam
membership according to QoS edge-risk and load-balance criteria, checks payload
constraints, and compares the heuristic against baselines and a MILP benchmark.

The central research idea is:

> In a LEO downlink snapshot, user clustering should not only minimize the
> number of beams. It should jointly respect beam geometry, link budget,
> demand-driven time-share capacity, enterprise-user QoS exposure, satellite
> beam-count limits, and satellite payload time-budget limits.

The implemented system combines:

- LEO visibility and satellite ranking using either a synthetic Walker-delta
  constellation or TLE data.
- QoS-conditioned traffic generation with heterogeneous demand classes.
- Fixed-angle beam modeling where ground footprint radius depends on slant
  range.
- Shannon-rate capacity evaluation using FSPL, antenna gain, SNR, and
  bandwidth.
- Recursive farthest-point splitting to guarantee per-beam feasibility.
- Enterprise-user refinement to reduce beam-edge exposure.
- Overlap-based load balancing to reduce peak beam utilization.
- Payload-level feasibility checks and repair across satellites.
- Baselines including weighted k-means, BK-Means, TGBP-style graph placement,
  and a Gurobi MILP benchmark.

Based on the current generated Phase B outputs in `sweep_phaseB.csv`, all
20 scaling runs from 1000 to 2500 users are payload feasible with zero payload
time or beam-count violations. The proposed full method (`main_ref_lb`) scales
from about 112.6 beams at 1000 users to 230.6 beams at 2500 users. The
load-balance refinement reduces peak beam utilization by about 23 percent to
39 percent compared with split-only clustering in the Phase B data. Existing
MILP comparison results show that the MILP is useful as a small to medium scale
benchmark, but becomes computationally expensive and often time-limited at
larger user counts.

## 2. Repository Purpose

The repository addresses the beam placement and user association problem for
LEO satellite downlinks. The motivating network scenario is a dense set of
ground users with heterogeneous service requirements. The network must decide:

1. Which visible satellites should serve the snapshot.
2. Which users should be associated with each selected satellite.
3. How users assigned to a satellite should be grouped into beams.
4. Whether each beam is feasible in geometry and capacity.
5. Whether the set of beams on each satellite fits payload limits.
6. Whether high-priority enterprise users are placed safely away from beam
   edges.

The project is therefore a hybrid of satellite geometry, wireless physical
layer modeling, clustering, combinatorial optimization, and performance
evaluation.

## 3. Codebase Map

Main files and generated artifacts:

| Path | Role |
| --- | --- |
| `config.py` | Dataclass configuration for region, PHY, traffic, beam, QoS refinement, payload, satellite source, and run settings. |
| `main.py` | Current Phase B scaling-sweep runner. It runs multiple scenarios in parallel, writes `sweep_phaseB.csv`, and regenerates Phase B plots. |
| `run_sensitivity.py` | Editable sensitivity-study runner with one-factor and interaction sweeps. |
| `run_milp_smoke.py` | Small MILP test script. |
| `run_milp_compare.py` | Heuristic vs MILP comparison sweep plus grid sensitivity analysis. |
| `starlink.tle` | TLE data for real-constellation experiments. |
| `sweep_phaseA.csv`, `sweep_phaseB.csv` | Existing generated scaling results. |
| `milp_compare.csv`, `milp_compare_aggregate.csv` | Existing generated MILP comparison results. |
| `plots/phaseB/` | Generated Phase B figures and tables. |
| `plots/milp/` | Generated MILP comparison figures and tables. |
| `sensitivity_results/` | Generated sensitivity CSVs, figures, and tables. |

Core source modules:

| Module | Main responsibility |
| --- | --- |
| `src/usergen.py` | Generates users over Turkey or a debug bounding box with hotspot/noise spatial distribution, QoS labels, and lognormal demand. |
| `src/models.py` | Defines scalar and vectorized user containers. |
| `src/coords.py` | Converts between latitude/longitude, local XY, and ECEF coordinates; computes elevation. |
| `src/satellites.py` | Builds and ranks visible satellites from Walker or TLE sources and implements association rules. |
| `src/phy.py` | Implements FSPL, Gaussian antenna gain, SNR, and Shannon rate. |
| `src/evaluator.py` | Evaluates one beam/cluster for geometry, rate, utilization, and enterprise edge risk. |
| `src/split.py` | Implements farthest-point bisection. |
| `src/refine_qos_angle.py` | Moves enterprise users toward better beam boresight positions when feasible. |
| `src/refine_load_balance.py` | Moves users between overlapping beams to reduce peak/variance/range utilization while preserving feasibility and QoS risk constraints. |
| `src/pipeline.py` | Main orchestration: user generation, satellite selection, prefix search, association, clustering, refinement, payload repair, baselines, summaries. |
| `src/helper.py` | Summary, flattening, printing, and CSV writing helpers. |
| `src/sweep_plots.py` | Aggregates sweep CSVs and generates Phase B plots/tables. |
| `src/baselines/` | Weighted k-means, BK-Means, TGBP-style graph placement, and strict repair. |
| `src/milp/` | Grid-candidate MILP preparation, candidate precomputation, Gurobi model, and experiment runner. |

## 4. Default Scenario Configuration

Important defaults from `config.py`:

| Block | Default |
| --- | --- |
| Region | `region_mode="turkey"` |
| Turkey bounding box | Latitude 36.0 to 42.2 deg, longitude 26.0 to 45.5 deg |
| Users | `n_users=250`, `seed=1` |
| Spatial model | 10 hotspots plus 15 percent uniform noise |
| QoS class probabilities | eco/std/enterprise = 0.6/0.3/0.1 |
| QoS weights | eco = 1, standard = 2, enterprise = 4 |
| Demand model | Lognormal, median 5 Mbps baseline, sigma 0.6 |
| Demand by QoS | Multipliers 0.7, 1.0, 2.0 for eco/std/enterprise |
| Carrier | 20 GHz |
| Bandwidth | 250 MHz |
| EIRP | 40 dBW |
| Spectral efficiency factor | `eta=0.7` |
| Miscellaneous loss | 2 dB |
| Beam half-power angle | `theta_3db_deg=1.0` |
| Satellite source | Synthetic Walker constellation by default |
| Walker constellation | 144 satellites, 12 planes, 53 deg inclination, 600 km altitude |
| Elevation mask | 25 deg |
| User association load mode | Physical demand |
| Payload time budget | `J_lanes=8`, `W_slots=8`, so per-satellite time cap is 64 |
| Payload beam-count cap | `Ks_max=256` beams per satellite |
| Enterprise safe radius | `rho_safe=0.7` |

The code also supports TLE mode by setting `cfg.multisat.source = "tle"` and
`cfg.multisat.tle_path = "starlink.tle"`.

## 5. System Model

### 5.1 Users

Each user has a geodetic position, local tangent-plane coordinates, an ECEF
position, a traffic demand in Mbps, and a QoS weight in `{1, 2, 4}`. The user
generator samples positions from either a uniform distribution or a hotspot
mixture. In the default hotspot setting, most users are generated around random
hotspot centers and a smaller fraction is uniformly spread across the region.

Demand is generated after QoS class assignment. Enterprise users therefore tend
to have higher median demand, which makes QoS visible in both traffic load and
edge-risk statistics.

### 5.2 Satellites and Visibility

The code supports two satellite sources:

1. `walker`: a deterministic synthetic Walker-delta constellation.
2. `tle`: real TLE propagation through Skyfield.

For both sources, the pipeline builds an anchor grid across the region, computes
satellite elevation angles at the anchors, filters satellites above the
elevation mask, and ranks visible satellites by greedy marginal anchor-quality
gain. This creates an ordered satellite pool. A user can be assigned to a
satellite only if that satellite's elevation angle is above
`cfg.multisat.elev_mask_deg`.

### 5.3 Fixed-Angle Beam Geometry

The beam model is fixed-angle, not fixed-radius. For a cluster/beam center:

```text
R_m = d_center * tan(theta_3db)
```

where:

- `R_m` is the beam footprint radius on the local plane.
- `d_center` is the satellite-to-beam-center slant range.
- `theta_3db` is the configured 3 dB half-angle.

A cluster is geometrically feasible if every member lies within `R_m` of the
beam center in local XY coordinates.

The normalized radial position of user `i` inside a beam is:

```text
z_i = distance(user_i, beam_center) / R_m
```

For enterprise users, the project tracks exposure:

```text
enterprise exposed if z_i > rho_safe
```

and a soft risk:

```text
risk = sum_enterprise max(0, z_i - rho_safe)^2
```

### 5.4 Physical Layer

For each user in a beam, the evaluator computes:

1. Off-axis angle between satellite-to-user direction and beam boresight.
2. Free-space path loss:

```text
FSPL(dB) = 20 log10(4 pi d f / c)
```

3. Gaussian-like antenna gain with -3 dB at `theta_3db`.
4. SNR from EIRP, antenna gain, path loss, miscellaneous loss, noise PSD, and
   bandwidth.
5. Shannon-rate approximation:

```text
rate_mbps = eta * bandwidth_hz * log2(1 + SNR) / 1e6
```

### 5.5 Beam and Payload Capacity

The code uses a time-share utilization model:

```text
U_b = sum_i demand_i_mbps / rate_i_b_mbps
```

A beam is capacity feasible when:

```text
U_b <= W_slots
```

Satellite payload feasibility is checked by:

```text
sum_b U_s_b <= J_lanes * W_slots
K_s <= Ks_max
```

With defaults:

```text
J_lanes * W_slots = 8 * 8 = 64
Ks_max = 256
```

## 6. Main Heuristic Pipeline

The main scenario function is `run_scenario(cfg)` in `src/pipeline.py`.

The pipeline is:

1. Generate a user snapshot.
2. Select and rank visible satellites.
3. Search over prefixes of the ranked satellite list.
4. Associate users to satellites.
5. Build feasible beams per satellite.
6. Optionally refine enterprise users away from beam edges.
7. Optionally refine load balance between overlapping beams.
8. Repair payload violations by moving whole beams between satellites.
9. Compare against ablations, association rules, baselines, and MILP outputs.
10. Flatten metrics for CSV/plots.

### 6.1 Prefix Search

The main method tries satellite prefixes:

```text
m = 1, 2, ..., max_prefix
```

For each prefix, the method runs association, clustering, refinement, and
payload repair. It stops at the first feasible prefix. If no feasible prefix is
found, it keeps the best failed candidate based on violation severity. This is
a practical way to approximate a minimum active satellite set without solving a
full global optimization problem.

### 6.2 Association Rules

The code supports three association rules:

| Rule | Description |
| --- | --- |
| `pure_max_elev` | Assign each user to the visible satellite with maximum elevation. |
| `balanced_max_elev` | Start with max-elevation assignment, then move users away from overloaded satellites while minimizing elevation loss. |
| `max_service_time` | Use a finite-difference proxy for remaining visibility time, with elevation as a small tie-breaker. |

The proposed pipeline uses `balanced_max_elev` by default.

### 6.3 Recursive Split-to-Feasible Clustering

For a satellite's assigned users, the main clustering primitive starts with one
cluster containing all users. It repeatedly evaluates a cluster. If the cluster
is feasible, it is accepted. If it is infeasible, it is split by farthest-point
bisection:

1. Pick a random point.
2. Find the farthest point A.
3. Find the farthest point B from A.
4. Split all users according to whether they are closer to A or B.

This recursive splitting continues until all beams are feasible. It is simple,
deterministic given the seed, and produces feasibility by construction unless
single-user feasibility itself is impossible under the PHY/demand settings.

### 6.4 Enterprise QoS Refinement

The enterprise refinement targets enterprise users near beam edges. A user is
considered risky if:

```text
qos_w == 4 and z > rho_safe
```

The algorithm considers moving risky enterprise users to candidate clusters
whose boresight direction is closer in 3D angle. A move is accepted only if:

- The donor and receiver beams remain feasible.
- The number of exposed enterprise users decreases, or exposure count is tied
  and total enterprise risk decreases.

This makes QoS refinement conservative. It improves the placement of high
priority users without breaking geometry or capacity constraints.

### 6.5 Load-Balance Refinement

The load-balance module builds an overlap graph between beams:

```text
beams overlap if distance(center_a, center_b) <= R_a + R_b + margin
```

It then tries to move users from high-utilization donor beams to lower-utility
overlapping receiver beams. The default objective is to reduce peak utilization
`U_max`. Moves must preserve receiver geometry, receiver capacity, enterprise
risk/exposure limits, and optional preference to move non-enterprise users first.

### 6.6 Payload Repair

After beams are built for each satellite, payload repair checks per-satellite
total time load and beam count. If a satellite violates beam count or time
budget, the repair loop tries to offload whole beams to other visible satellites
with payload slack. Candidate receiver satellites must be able to see all users
in the offloaded beam.

This is an important system-level feature: the pipeline is not merely local
clustering per satellite. It includes cross-satellite repair.

## 7. Baselines and MILP Benchmark

### 7.1 Ablation Chain

| Label | Method |
| --- | --- |
| A0 | Pure max-elevation association plus split-to-feasible. |
| A1 | Balanced max-elevation association plus split-to-feasible. |
| A2 | Balanced association plus split plus enterprise QoS refinement. |
| A3 | Balanced association plus split plus QoS refinement plus load-balance refinement. |

This isolates which components affect feasibility, beam count, enterprise edge
exposure, and utilization.

### 7.2 Clustering Baselines

The repository implements weighted k-means baselines with weights equal to
demand and demand times QoS. These baselines use weighted k-means++ and Lloyd
iterations, then strict repair by recursive splitting.

The `fast_beam_placement.py` module implements paper-inspired fast beam
placement baselines:

- BK-Means: binary search over `K` with k-means and clique feasibility checks.
- TGBP: two-phase graph-based placement using a greedy clique cover and optional
  load balancing.

Because the main project uses slant-range-dependent beam footprint radius,
these graph baselines use a conservative reference radius:

```text
r_ref = tan(theta_3db) * max_user_range
```

This keeps them runnable, but it is an approximation relative to the main
per-beam slant-range model.

### 7.3 MILP Benchmark

The MILP benchmark creates grid candidate beam centers over the realized user
footprint, then generates feasible satellite/grid beam candidates. The MILP uses
binary variables:

| Variable | Meaning |
| --- | --- |
| `x_b` | Whether beam candidate `b` is active. |
| `z_s` | Whether satellite `s` is active. |
| `y_u_b` | Whether user `u` is assigned to candidate beam `b`. |

Key constraints:

- Each user must be assigned exactly once.
- A user can be assigned only to an active beam.
- A beam can be active only if its satellite is active.
- Beam load cannot exceed `W_slots`.
- Satellite beam count cannot exceed `Ks_max`.
- Satellite time load cannot exceed `J_lanes * W_slots`.
- Optional satellite activation prefix ordering.

The MILP supports `beam_only` and `weighted_sat_beam` objectives. It is best
used as a validation and benchmarking tool for small to medium instances, while
the heuristic is the scalable operating method.

## 8. Generated Results in the Repository

This section summarizes the generated artifacts currently present in the
repository. I used the checked-in CSV and table files because Python execution
is not available in this shell environment.

Execution note:

- `python`, `python3`, and `py` are not available as working Python commands in
  the current shell. `python.exe` and `python3.exe` resolve to Windows Store
  stubs, not a real interpreter.
- Therefore, I did not regenerate experiments in this session.
- The quantitative analysis below is based on existing files:
  - `sweep_phaseB.csv`
  - `plots/phaseB/phaseB_tables.txt`
  - `milp_compare.csv`
  - `milp_compare_aggregate.csv`
  - `plots/milp/milp_compare_tables.txt`
  - `sensitivity_results/sensitivity_summary.csv`

### 8.1 Phase B Scaling Results

The current `main.py` Phase B sweep is configured for:

```text
n_users = [1000, 1500, 2000, 2500]
seeds = [1, 2, 3, 4, 5]
region = turkey
enable_fastbp_baselines = True
satellite source = walker
```

Summary of proposed full method (`main_ref_lb` / A3):

| Users | Runs | Payload feasible | Mean active satellites | Mean beams K | Split-only U_max | Full U_max | U_max reduction | Enterprise edge % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 5 | 100% | 1.0 | 112.6 | 0.845 | 0.518 | 38.7% | 7.00% |
| 1500 | 5 | 100% | 1.0 | 148.8 | 1.257 | 0.897 | 28.6% | 6.88% |
| 2000 | 5 | 100% | 1.0 | 177.0 | 1.500 | 1.155 | 23.0% | 6.23% |
| 2500 | 5 | 100% | 1.2 | 230.6 | 2.442 | 1.495 | 38.8% | 6.57% |

Payload summary:

| Users | Mean active satellites | Max active satellites | Mean payload T_max | Mean payload K_max | Time cap | Beam cap | Mean T violations | Mean K violations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1.0 | 1 | 7.682 | 112.6 | 64 | 256 | 0 | 0 |
| 1500 | 1.0 | 1 | 11.800 | 148.8 | 64 | 256 | 0 | 0 |
| 2000 | 1.0 | 1 | 15.645 | 177.0 | 64 | 256 | 0 | 0 |
| 2500 | 1.2 | 2 | 18.772 | 200.2 | 64 | 256 | 0 | 0 |

Interpretation:

- The proposed method maintained payload feasibility in all Phase B runs.
- There were no satellite time-budget or beam-count violations after payload
  repair.
- Peak beam utilization increased with user count, as expected, but load-balance
  refinement significantly reduced peak utilization compared with split-only
  clustering.
- Active satellite count stayed very low in this Walker snapshot. Most cases
  used one satellite; some 2500-user seeds required two.
- Mean beam radius stayed approximately 16 to 17 km across the sweep, which is
  consistent with the fixed-angle beam model and similar slant ranges.

### 8.2 Phase B Baseline Comparison

Mean beam counts and runtimes from `sweep_phaseB.csv`:

| Users | Proposed K | BK K | TGBP K | WK demand K | WK demand*QoS K | Proposed runtime s | BK runtime s | TGBP runtime s | WK demand runtime s | WK QoS runtime s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 112.6 | 105.0 | 119.6 | 123.8 | 123.6 | 0.470 | 4.907 | 0.820 | 0.817 | 0.588 |
| 1500 | 148.8 | 141.0 | 159.0 | 163.0 | 169.2 | 1.130 | 7.676 | 1.199 | 0.915 | 0.969 |
| 2000 | 177.0 | 170.0 | 181.0 | 196.4 | 199.6 | 1.675 | 12.227 | 1.087 | 1.560 | 0.800 |
| 2500 | 230.6 | 198.2 | 242.2 | 254.4 | 314.8 | 1.498 | 7.194 | 0.590 | 1.197 | 2.212 |

Interpretation:

- BK-Means often uses fewer beams than the proposed method in the current
  Phase B data, but it is substantially slower.
- TGBP is competitive in runtime and beam count, especially at larger user
  counts, but it is based on a conservative graph-radius approximation rather
  than the full per-beam slant-range geometry.
- Weighted k-means variants generally produce more beams after strict repair.
- The proposed method is attractive because it directly optimizes the specific
  feasibility and QoS criteria in the project, rather than relying on generic
  clustering geometry and then repairing afterwards.

### 8.3 Ablation Findings

The ablation tables in `plots/phaseB/phaseB_tables.txt` show:

- Feasible cluster rate is 100 percent for A0, A1, A2, and A3 across all Phase B
  user counts.
- Beam count is almost unchanged across A0 to A3. This is expected because QoS
  and load-balance refinements move users among existing feasible clusters; they
  do not primarily add or remove beams.
- The main benefit of A3 is reduced `U_max`, not reduced `K`.

Mean `U_max` by ablation:

| Users | A0 pure+split | A1 balanced+split | A2 + QoS | A3 + QoS + load balance |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.845 | 0.845 | 0.836 | 0.518 |
| 1500 | 1.257 | 1.257 | 1.251 | 0.897 |
| 2000 | 1.500 | 1.500 | 1.500 | 1.155 |
| 2500 | 2.442 | 2.614 | 2.607 | 1.495 |

Interpretation:

- Enterprise refinement alone has small effect on peak load because its goal is
  edge-risk reduction, not utilization balancing.
- Load-balance refinement is the component responsible for the large reduction
  in peak utilization.
- Association variants are nearly identical in this Phase B sweep because the
  chosen payload-feasible prefix often contains only one active satellite.

### 8.4 MILP Comparison Results

Selected results from `milp_compare_aggregate.csv`:

| Users | MILP optimal rate | Heuristic K | MILP K | K gap, heuristic minus MILP | Heuristic runtime s | MILP runtime s | MILP candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 100% | 22.4 | 25.4 | -11.5% | 15.5 | 85.4 | 4765 |
| 100 | 100% | 39.0 | 36.0 | 9.0% | 16.4 | 78.7 | 7348 |
| 500 | 100% | 90.6 | 77.6 | 16.3% | 19.7 | 163.1 | 13475 |
| 700 | 100% | 105.2 | 88.2 | 19.1% | 18.5 | 213.5 | 14862 |
| 1000 | 80% | 128.6 | 2487.8 | high variance | 16.8 | 601.6 | 16074 |
| 2500 | 20% | 180.4 | 11103.2 | not reliable | 18.2 | 1857.6 | 19835 |
| 5000 | 0% | 256.4 | 15613.6 | not reliable | 26.2 | 2427.5 | 22388 |

Interpretation:

- Up to about 700 users, MILP solve rate is 100 percent optimal in the stored
  comparison data. In this range, the heuristic uses roughly 6 to 25 percent
  more beams than the grid-candidate MILP, except at 50 users where the
  heuristic appears better because the MILP is restricted to discrete grid
  candidates.
- At 1000 users and above, MILP optimal solve rate drops. The large MILP K
  values at high user counts are symptoms of time-limited or non-ideal
  solutions, so they should not be treated as true optimal beam counts.
- MILP runtime is one to two orders of magnitude larger than heuristic runtime
  in the stored comparisons.

### 8.5 Sensitivity Results

The existing `sensitivity_results/` artifacts appear to come from a TLE-based
sensitivity run and may not exactly match the current editable `SETTINGS` block
in `run_sensitivity.py`. They are still useful as evidence of parameter trends.

Selected 5000-user sensitivity rows:

| Factor | Value | Payload feasible | K mean | U_max mean | Enterprise edge % | Active sats mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Demand median Mbps | 5 | 100% | 973.6 | 1.089 | 6.89 | 6.0 |
| Demand median Mbps | 10 | 100% | 973.6 | 2.179 | 6.89 | 6.0 |
| Demand median Mbps | 20 | 100% | 974.4 | 4.105 | 6.89 | 6.0 |
| Demand stress Mbps | 50 | 100% | 1056.4 | 5.302 | 6.66 | 10.4 |
| Demand stress Mbps | 100 | 100% | 1203.2 | 6.472 | 7.86 | 20.0 |
| Demand stress Mbps | 150 | 100% | 1362.4 | 7.119 | 9.16 | 29.0 |
| Elevation mask deg | 20 | 100% | 977.0 | 1.095 | 7.00 | 6.0 |
| Elevation mask deg | 25 | 100% | 973.6 | 1.089 | 6.89 | 6.0 |
| Elevation mask deg | 30 | 100% | 994.4 | 1.301 | 6.08 | 6.4 |
| Beam angle deg | 1 | 100% | 973.6 | 1.089 | 6.89 | 6.0 |
| Beam angle deg | 2 | 100% | 503.6 | 2.408 | 4.22 | 2.6 |
| Beam angle deg | 3 | 100% | 321.8 | 4.188 | 2.69 | 2.0 |
| `Ks_max` | 128 | 100% | 1059.8 | 0.691 | 6.86 | 12.4 |
| `Ks_max` | 256 | 100% | 973.6 | 1.089 | 6.89 | 6.0 |
| `Ks_max` | 312 | 100% | 962.4 | 2.332 | 6.41 | 4.6 |
| `rho_safe` | 0.6 | 100% | 973.6 | 1.067 | 14.75 | 6.0 |
| `rho_safe` | 0.7 | 100% | 973.6 | 1.089 | 6.89 | 6.0 |
| `rho_safe` | 0.8 | 100% | 973.6 | 1.101 | 2.75 | 6.0 |

Interpretation:

- Payload feasibility is robust across the shown sensitivity points.
- Increasing demand mostly increases `U_max`; under stress demand it also
  increases beam count and required active satellites.
- Increasing beam angle reduces beam count because each beam footprint grows,
  but it can increase `U_max` because larger beams collect more aggregate
  demand.
- Tightening per-satellite beam-count cap (`Ks_max=128`) forces the use of more
  active satellites and increases runtime.
- Increasing `rho_safe` makes the enterprise edge-exposure definition less
  strict in the stored outputs, so the reported enterprise edge percentage
  decreases.

## 9. Main Technical Contributions

The project can be presented as having the following contributions:

1. A LEO snapshot simulation pipeline that integrates user generation, satellite
   visibility, user association, beam clustering, QoS risk, and payload
   feasibility.
2. A fixed-angle beam feasibility model coupled to a physical link budget and
   time-share capacity abstraction.
3. A scalable recursive split-to-feasible heuristic that produces geometry and
   capacity feasible beams.
4. An enterprise-aware refinement step that explicitly reduces high-priority
   user edge exposure.
5. An overlap-based load-balancing refinement that reduces peak beam utilization
   while preserving feasibility and enterprise-risk constraints.
6. A payload repair loop that moves whole beams between satellites to satisfy
   satellite-level time and beam-count budgets.
7. A comparative framework including association-rule ablations, weighted
   k-means baselines, graph/k-means beam-placement baselines, and a Gurobi MILP
   benchmark.
8. Generated scaling, sensitivity, and MILP comparison artifacts suitable for
   thesis figures and discussion.

## 10. Strengths of the Current Work

The strongest parts of the repository are:

- It is end-to-end. The pipeline does not stop at clustering; it evaluates
  physical rates, QoS exposure, and payload feasibility.
- It has an explicit payload model. The satellite-level constraints make the
  problem more realistic than isolated beam clustering.
- It separates algorithmic components cleanly enough for ablation studies.
- It includes multiple baselines and a MILP benchmark, which helps position the
  heuristic academically.
- It supports both synthetic Walker and TLE-based satellite snapshots.
- It stores generated CSVs, aggregate tables, and plots for reproducibility of
  the current experimental narrative.

## 11. Limitations and Risks

These are the main limitations to discuss transparently with the professor:

1. Static snapshot only.
   The project currently models one instant in time. It does not yet model
   mobility across multiple time slots, handover, or temporal continuity of beam
   assignments.

2. Simplified physical layer.
   The PHY model uses FSPL, a Gaussian antenna pattern, simple losses, and
   Shannon-rate approximation. It does not model inter-beam interference,
   rain fading, adaptive coding/modulation tables, polarization, or adjacent
   satellite interference.

3. Simplified payload abstraction.
   Payload feasibility is represented through time-share load and beam-count
   limits. This is useful, but it does not model detailed satellite hardware
   scheduling, feeder links, gateway constraints, frequency reuse, or power
   amplifier constraints.

4. QoS is partly indirect.
   QoS affects demand generation and enterprise edge-risk metrics. Capacity
   itself uses physical demand, not demand multiplied by QoS. This is a design
   choice, but it should be explained.

5. Local tangent-plane approximation.
   Local XY coordinates are adequate for regional experiments, but Turkey is a
   large enough region that higher-fidelity map projection could be considered
   for final work.

6. MILP is grid-based.
   The MILP optimizes over candidate grid centers, not continuous beam centers.
   Its optimum is therefore an optimum of the discretized benchmark, not
   necessarily the true continuous problem.

7. MILP scalability.
   At large user counts, the stored MILP results become time-limited and should
   not be interpreted as reliable optimal beam counts.

8. Missing packaging and tests.
   There is no `requirements.txt`, `pyproject.toml`, or `tests/` directory in
   the repository. This makes reproducibility and regression checking harder.

9. Generated artifacts are tracked.
   The repository includes generated plots, CSVs, and cache files. This is useful
   for review, but before publication or submission it should be cleaned and
   organized.

10. Environment not fully reproducible from this checkout.
    In this shell session, Python is not available on PATH, so I could inspect
    and analyze the code and generated data but could not rerun experiments.

## 12. Recommended Improvements Before Thesis Defense

High priority:

1. Add `requirements.txt` or `pyproject.toml`.
   Include at least `numpy`, `matplotlib`, and optional groups for `pandas`,
   `scipy`, `scikit-learn`, `skyfield`, and `gurobipy`.

2. Add a reproducibility section.
   Document the exact command sequence for Phase B, sensitivity, and MILP
   outputs.

3. Add smoke tests.
   A small no-Gurobi test should run user generation, Walker satellite
   selection, one small `run_scenario`, and summary flattening.

4. Add a result manifest.
   For each generated CSV, record the config used, date generated, Python
   version, and dependency versions.

5. Separate generated artifacts from source.
   Consider `results/` or `artifacts/` directories with clear subfolders.

Medium priority:

1. Add a CLI or YAML configuration file.
   This avoids editing scripts directly for each experiment.

2. Add time-series experiments.
   Extend from static snapshot to multiple snapshots and study handover,
   reassignment stability, and temporal fairness.

3. Add interference/frequency reuse modeling.
   Even a simplified reuse-factor model would strengthen the communication
   systems realism.

4. Add more QoS metrics.
   Examples: outage probability per QoS class, mean achieved rate, Jain
   fairness, enterprise p95 edge exposure, and demand satisfaction.

5. Add confidence intervals.
   Existing tables use mean/std. Thesis plots could add 95 percent confidence
   intervals over seeds.

6. Clarify TLE vs Walker result sets.
   Phase B uses Walker by default, while sensitivity outputs appear TLE-based.
   This distinction should be explicit in figure captions.

Research extension ideas:

1. Multi-objective optimization:
   jointly minimize beams, active satellites, peak utilization, and enterprise
   risk.

2. Learning-assisted heuristics:
   use historical snapshots to predict good initial satellite prefixes or split
   decisions.

3. Robustness to traffic uncertainty:
   evaluate under stochastic demand perturbations or traffic bursts.

4. Dynamic beam hopping:
   model scheduling over time slots instead of only aggregate time-share load.

5. Gateway/backhaul constraints:
   add feeder-link or gateway assignment limits.

## 13. Suggested Thesis Presentation Structure

Recommended slide flow:

1. Problem motivation
   - LEO satellites serve dense heterogeneous users.
   - Beam placement must account for geometry, rate, QoS, and payload limits.

2. Research question
   - How can we cluster users into LEO beams while preserving QoS and payload
     feasibility at scale?

3. System model
   - Users, satellites, visibility, fixed-angle beams, link budget, time-share
     capacity, enterprise edge risk.

4. Proposed pipeline
   - Satellite ranking and prefix search.
   - Balanced association.
   - Split-to-feasible beam construction.
   - Enterprise refinement.
   - Load-balance refinement.
   - Payload repair.

5. Mathematical feasibility criteria
   - Beam geometry.
   - Beam capacity.
   - Satellite payload constraints.
   - Enterprise edge-risk metric.

6. Baselines
   - Weighted k-means.
   - BK-Means.
   - TGBP.
   - MILP benchmark.

7. Phase B scaling results
   - Use `plots/phaseB/K_vs_nusers.png`.
   - Use `plots/phaseB/runtime_vs_nusers.png`.
   - Use `plots/phaseB/ablation_Umax_vs_nusers.png`.
   - Use `plots/phaseB/payload_overflow_vs_nusers.png`.

8. Main result message
   - Payload feasibility is maintained.
   - Load-balance refinement reduces peak utilization significantly.
   - Proposed method is much faster than MILP at scale.

9. Sensitivity results
   - Discuss demand, beam angle, elevation mask, `Ks_max`, and `rho_safe`.

10. Limitations and future work
    - Static snapshot, simplified PHY, no interference, grid MILP, missing tests.

11. Conclusion
    - The project provides a complete experimental framework for QoS-aware,
      payload-aware LEO beam clustering and a clear path toward dynamic network
      optimization.

## 14. Recommended Figures to Show

Good figures already present in the repository:

| Figure path | Why it is useful |
| --- | --- |
| `plots/phaseB/K_vs_nusers.png` | Shows beam-count scaling and comparison against baselines. |
| `plots/phaseB/runtime_vs_nusers.png` | Shows computational scalability. |
| `plots/phaseB/ablation_Umax_vs_nusers.png` | Best evidence that load balancing improves peak utilization. |
| `plots/phaseB/system_assoc_Umax_vs_nusers.png` | Shows association-rule comparison. |
| `plots/phaseB/payload_overflow_vs_nusers.png` | Shows payload violations are eliminated. |
| `plots/phaseB/beam_radius_summary_vs_nusers.png` | Supports the fixed-angle beam footprint discussion. |
| `plots/milp/K_vs_nusers.png` | Heuristic vs MILP beam-count benchmark. |
| `plots/milp/runtime_vs_nusers.png` | MILP scalability limitation. |
| `plots/milp/optimal_solve_rate_vs_nusers.png` | Shows where MILP remains reliable. |
| `sensitivity_results/one_factor_theta_3db_deg_K_mean.png` | Shows beam angle tradeoff. |
| `sensitivity_results/one_factor_demand_mbps_median_U_max_mean.png` | Shows traffic-load sensitivity. |

## 15. Answers to Likely Professor Questions

### What is novel here?

The project is not only applying k-means to users. It formulates a practical
LEO beam placement pipeline that combines satellite visibility, physical link
capacity, QoS edge-risk, load balancing, and satellite payload limits. The
pipeline is also benchmarked against classical clustering baselines and a MILP
formulation.

### Why use recursive splitting instead of directly optimizing all beam centers?

Direct global optimization is expensive, especially with satellite assignment
and payload constraints. Recursive splitting is scalable and produces feasible
clusters by construction. The MILP results show why exact optimization becomes
costly as the problem size grows.

### Why is beam radius not fixed?

The model uses a fixed beam half-angle, which is more natural for antenna
pointing. The ground footprint grows with slant range:

```text
R_m = d_center * tan(theta_3db)
```

This makes beam size depend on satellite-user geometry.

### How is QoS represented?

QoS is represented in three ways:

1. Higher QoS users have higher median demand during traffic generation.
2. Enterprise users are tracked for beam-edge exposure.
3. Optional weighted baselines use `demand * qos` as clustering weight.

The main capacity calculation itself uses physical demand, not artificially
weighted demand.

### Why does load balancing matter if all beams are already feasible?

Feasibility only says each beam is under the capacity threshold. Load balancing
reduces the peak beam utilization, creating more operational margin against
traffic uncertainty, demand bursts, and modeling errors.

### Why can the heuristic sometimes appear better than the MILP?

The MILP is a discretized grid-candidate problem. The heuristic can place beam
centers continuously as cluster means, while the MILP is limited to candidate
grid centers. Therefore, the MILP is a benchmark for its candidate set, not
always a strict lower bound on the continuous heuristic problem.

### Why do high-user MILP results become strange?

At large user counts, the MILP often hits time limits. Non-optimal or
time-limited incumbent solutions can have very large beam counts. Those values
should be interpreted as evidence of MILP scalability limits, not as true
optimal results.

## 16. Reproducibility Notes

Typical commands from the README:

```bash
python main.py
python -m src.sweep_plots --csv sweep_phaseB.csv --out plots/phaseB
python run_sensitivity.py
python run_milp_smoke.py
python run_milp_compare.py
```

Dependency expectations:

```bash
python -m pip install numpy matplotlib
python -m pip install scipy scikit-learn pandas skyfield
```

Additional MILP dependency:

```bash
python -m pip install gurobipy
```

A working Gurobi license is required for MILP experiments.

## 17. Overall Assessment

This is a strong master thesis prototype. It has a clear engineering problem, a
nontrivial system model, a scalable heuristic, meaningful QoS and payload
constraints, multiple baselines, and generated quantitative evidence.

The main work before final presentation is not to invent a completely new
algorithm from scratch. The most valuable next step is to make the experimental
story cleaner and more reproducible:

- Lock dependencies.
- Add small smoke tests.
- Clearly separate Walker and TLE result sets.
- Regenerate final figures from a documented configuration.
- Be explicit about the limits of the static snapshot and MILP benchmark.

The thesis argument can be summarized as:

> A QoS-aware and payload-aware heuristic can produce feasible LEO beam
> assignments at scales where MILP benchmarking becomes expensive, while
> preserving physical beam feasibility and reducing peak utilization through
> targeted refinement.
