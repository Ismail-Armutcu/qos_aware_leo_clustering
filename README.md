# QoS-Aware User Clustering for LEO Satellite Networks

A high-performance Python framework for QoS-aware beam placement and user clustering in Low Earth Orbit (LEO) satellite networks, with support for multi-satellite constellations, heterogeneous QoS requirements, and real-time TLE-based satellite tracking.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Algorithm Pipeline](#algorithm-pipeline)
- [Baseline Comparisons](#baseline-comparisons)
- [Configuration](#configuration)
- [Output & Visualization](#output--visualization)
- [Performance Optimization](#performance-optimization)
- [Research Applications](#research-applications)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This framework addresses the challenging problem of **dynamic beam placement** in LEO satellite networks where users have **heterogeneous Quality of Service (QoS) requirements**. It provides:

1. **QoS-differentiated clustering**: Eco (w=1), Standard (w=2), and Enterprise (w=4) user tiers
2. **Multi-satellite constellation support**: Real-time TLE-based satellite selection and tracking
3. **Balanced load distribution**: Advanced user-to-satellite association with load balancing
4. **Enterprise edge protection**: Special refinement to keep high-priority users away from beam edges
5. **Capacity-aware beam sizing**: Discrete footprint radius modes with Shannon capacity constraints
6. **Comprehensive baselines**: Includes implementations of Weighted K-Means, BK-Means, and TGBP algorithms

The system models realistic physical layer characteristics including free-space path loss, Gaussian antenna patterns, and Shannon capacity under thermal noise constraints.

---

## Key Features

### 🌍 **Geographic Coverage**
- **Configurable regions**: Turkey (default), debug regions, or custom bounding boxes
- **Hotspot generation**: Realistic user distributions with Gaussian hotspots and noise overlay
- **Coordinate systems**: Seamless conversion between geodetic (lat/lon), local tangent plane (XY), and ECEF coordinates

### 🛰️ **Multi-Satellite Support**
- **TLE-based tracking**: Load real satellite constellations from TLE files
- **Elevation masking**: Configurable minimum elevation angles (default: 25°)
- **Smart selection**: Top-N satellites by elevation at reference site
- **Balanced association**: Minimize load imbalance across satellites using demand-weighted or QoS-weighted metrics

### 📡 **Physical Layer Modeling**
- **Frequency**: 20 GHz Ka-band (configurable)
- **Bandwidth**: 250 MHz (configurable)
- **EIRP**: 40 dBW (configurable)
- **Beam patterns**: Gaussian antenna model with -3 dB beamwidth
- **Footprint modes**: Discrete radius options (default: 5, 10, 15, 20 km)
- **Shannon capacity**: Time-division multiple access with weighted demand sharing

### 🎯 **QoS Differentiation**
- **Eco (w=1)**: 60% probability, basic service
- **Standard (w=2)**: 30% probability, medium priority
- **Enterprise (w=4)**: 10% probability, mission-critical service with edge-risk penalties

### ⚡ **Performance Optimizations**
- **Parallel execution**: Multi-core scenario sweeps using ProcessPoolExecutor
- **Incremental evaluation**: Frozen beam geometry during load-balance refinement
- **Grid-based adjacency**: Fast overlap detection for refinement
- **Efficient data structures**: NumPy vectorization throughout
- **Profiling support**: Built-in timing and operation counters

---

## Architecture

```
qos_aware_user_clustering/
│
├── main.py                          # Entry point: parallel sweeps, plotting
├── config.py                        # Comprehensive configuration dataclasses
│
├── src/
│   ├── pipeline.py                  # Main orchestration: multi-sat workflow
│   ├── models.py                    # Data structures: User, Users, UsersRaw
│   ├── usergen.py                   # User generation with hotspots
│   ├── satellites.py                # TLE loading, selection, association
│   ├── coords.py                    # Coordinate transformations (geodetic/ECEF/XY)
│   ├── phy.py                       # Physical layer: FSPL, gain, SNR, Shannon
│   ├── evaluator.py                 # Cluster feasibility evaluation
│   ├── split.py                     # Farthest-point bisection
│   ├── refine_qos_angle.py          # Enterprise edge refinement (angle-based)
│   ├── refine_load_balance.py       # Load-balance refinement (overlap-based)
│   ├── helper.py                    # Summary KPIs, CSV I/O, config printing
│   ├── plot.py                      # Visualization: user scatter, beam overlays
│   ├── sweep_plots.py               # Aggregate plot generation (Phase A/B)
│   ├── profiling.py                 # Timing and counter profiler
│   │
│   └── baselines/
│       ├── weighted_kmeans.py       # Weighted K-Means++ (demand, demand*QoS)
│       ├── fast_beam_placement.py   # BK-Means & TGBP from FastBP paper
│       └── repair.py                # Strict split-until-feasible repair
│
├── plots/                           # Generated visualizations (Phase A/B)
├── sweep_phaseA.csv                 # Robustness results (10 seeds)
├── sweep_phaseB.csv                 # Scaling results (1k-10k users)
├── .gitignore                       # Excludes plots/, .idea/, __pycache__
└── README.md                        # This file
```

---

## Installation

### Prerequisites

- **Python 3.9+** (tested with 3.12)
- **NumPy** (core operations)
- **Matplotlib** (visualization)
- **Skyfield** (TLE parsing and satellite ephemeris)
- **SciPy** (optional: spatial accelerators for baselines)
- **scikit-learn** (optional: K-Means baseline acceleration)
- **Pandas** (optional: CSV loading for plots)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd qos_aware_user_clustering

# Install dependencies
pip install numpy matplotlib skyfield scipy scikit-learn pandas

# Verify TLE file exists (or download your own)
# Example: starlink.tle (not included, download from celestrak.org)
```

---

## Quick Start

### 1. Single Scenario Run

```python
from config import ScenarioConfig
from src.pipeline import run_scenario

# Default configuration: Turkey region, 250 users, seed=1
cfg = ScenarioConfig()
result = run_scenario(cfg)

print(f"Total beams: {result['main_ref_lb']['K']}")
print(f"U_max: {result['main_ref_lb']['U_max']:.3f}")
print(f"Enterprise edge exposure: {result['main_ref_lb']['ent_edge_pct']:.1f}%")
```

### 2. Parallel Sweep (Phase B: Scaling)

```python
from main import main
main()  # Runs Phase A (robustness) + Phase B (scaling) + generates plots
```

This will:
- Generate users with hotspots across 10 seeds (Phase A: 2500 users)
- Scale from 1k to 10k users across 5 seeds (Phase B)
- Output CSVs: `sweep_phaseA.csv`, `sweep_phaseB.csv`
- Generate plots in `plots/phaseA/` and `plots/phaseB/`

---

## Core Modules

### `config.py` - Configuration System

Hierarchical frozen dataclasses for reproducible experiments:

```python
@dataclass(frozen=True)
class ScenarioConfig:
    region_mode: Literal["debug", "turkey"] = "turkey"
    run: RunConfig = RunConfig()
    phy: PhyConfig = PhyConfig()
    beam: BeamConfig = BeamConfig()
    ent: EnterpriseConfig = EnterpriseConfig()
    traffic: TrafficConfig = TrafficConfig()
    qos_refine: QoSRefineConfig = QoSRefineConfig()
    lb_refine: LoadBalanceRefineConfig = LoadBalanceRefineConfig()
    usergen: HotspotGenConfig = HotspotGenConfig()
    multisat: MultiSatConfig = MultiSatConfig()
```

**Key Config Blocks:**
- **PhyConfig**: Satellite altitude, carrier frequency, bandwidth, EIRP, spectral efficiency
- **BeamConfig**: Discrete footprint radius modes (km)
- **TrafficConfig**: Demand distribution (lognormal median + sigma), QoS probabilities
- **EnterpriseConfig**: `rho_safe` threshold (enterprise users must stay z ≤ 0.7 within beam)
- **QoSRefineConfig**: Enterprise refinement rounds, candidate count, move limits
- **LoadBalanceRefineConfig**: LB refinement objective (max/range/var), overlap margin, receiver utilization cap

### `pipeline.py` - Main Orchestration

**Workflow:**
1. **User generation** (`usergen.py`): Hotspots + noise in configured bbox
2. **Satellite selection** (`satellites.py`): TLE → Top-N by elevation
3. **User-to-satellite association** (`satellites.py`): Balanced load distribution
4. **Per-satellite clustering**:
   - **Split-to-feasible**: Recursive bisection until all clusters satisfy geometry + capacity
   - **QoS refinement**: Move enterprise users away from beam edges (angle-based)
   - **Load-balance refinement**: Minimize U_max across overlapping beams (frozen geometry)
5. **Baseline runs**: Weighted K-Means, BK-Means, TGBP (with repair)
6. **Global aggregation**: Summarize KPIs across all satellites
7. **Plotting**: Visualize busiest satellite result

### `evaluator.py` - Feasibility Checking

**Cluster evaluation returns:**
- `feasible`: Boolean (geometry + capacity constraints)
- `reason`: "geom" (users exceed largest footprint mode), "cap" (U > 1), or None
- `R_m`: Selected footprint radius (meters)
- `req_m`: Required radius to cover all users (meters)
- `U`: Utilization = Σ(qos_w × demand / rate)
- `risk`: Σ(max(0, z_ent - rho_safe)²) for enterprise users
- `rate_mbps`: Per-user Shannon rates (array)
- `z`: Normalized radial distances from beam center (array)

**Physics:**
```python
# Free-space path loss (dB)
FSPL = 20 log₁₀(4πdf/c)

# Gaussian antenna gain (dB)
G = -3 (θ / θ_3dB)² log(2)

# SNR (linear)
SNR = 10^((EIRP + G - FSPL - L_misc - N_total) / 10)

# Shannon rate (Mbps)
R = η × B × log₂(1 + SNR) / 10⁶
```

### `refine_qos_angle.py` - Enterprise Edge Protection

**Algorithm:**
- Identify "at-risk" enterprise users: z > rho_safe in their current cluster
- Candidate target clusters: smallest 3D off-axis angle (max dot product of boresight vectors)
- Move user if both clusters remain feasible and:
  - Number of exposed enterprise users decreases, OR
  - Exposure unchanged but total risk decreases

**Implementation details:**
- Uses `u_sat2user` (precomputed unit vectors) for fast angle computation
- Recomputes cluster boresights after each move within a round
- Conservative: never moves from clusters with ≤2 users

### `refine_load_balance.py` - Load Balancing

**Key innovation: Frozen beam geometry**
- Beam centers and radii are **fixed** after split-to-feasible
- Only cluster membership changes
- Incremental feasibility checks:
  - **Geometry**: User must be inside receiver's frozen footprint (+ margin)
  - **Capacity**: U_to_new ≤ 1.0 (and optionally ≤ receiver_u_max)
- Enterprise guards: Exposure and risk must not increase (+ slack)

**Objective functions:**
- `max`: Minimize max(U) across all beams (default)
- `range`: Minimize max(U) - min(U)
- `var`: Minimize variance of U

**Adjacency:** Grid-hash spatial index for fast overlap queries (beams overlap if dist(centers) ≤ R1 + R2 + margin)

### `baselines/` - Comparison Algorithms

#### Weighted K-Means++ (`weighted_kmeans.py`)
- **Initialization**: k-means++ with weights (demand or demand×QoS)
- **Lloyd iterations**: Weighted centroid updates
- **Fixed-K evaluation**: Uses k-means centers as beam centers (frozen)
- **Repair**: Split infeasible clusters until all feasible

#### BK-Means (`fast_beam_placement.py`)
- **Binary search K**: Find smallest K where k-means yields all clusters as "cliques" (diameter ≤ 2×r_max)
- **Multi-restart**: μ=10 random restarts per K to avoid local minima
- **Fixed-K evaluation**: Uses k-means centers (frozen)
- **Repair**: Split infeasible clusters

#### TGBP - Two-Phase Graph-Based Placement (`fast_beam_placement.py`)
- **Phase 1**: Greedy clique cover on threshold graph (edge if dist ≤ 2×r_max)
  - Order users by descending degree
  - For each uncovered user: form maximal clique by adding compatible neighbors
- **Phase 2**: Balance clique sizes by moving users between adjacent cliques
- **Repair**: Split infeasible clusters

---

## Algorithm Pipeline

### Step 1: User Generation
```python
users = generate_users(cfg)  # Returns List[User]
users_raw = pack_users_raw(users)  # Vectorized container (N,) arrays
```
- Hotspots: Gaussian clusters with random centers and variances
- Noise: Uniform background users (default 15%)
- Demand: Lognormal (median=5 Mbps, sigma=0.6)
- QoS: Multinomial (60% eco, 30% std, 10% ent)

### Step 2: Satellite Selection
```python
t0_utc, active_sats = select_top_n_active_sats(
    tle_path="starlink.tle",
    n_active=10,
    elev_mask_deg=25.0,
    ref_lat_deg=39.0, ref_lon_deg=35.0
)
```
- Loads TLE file using Skyfield
- Computes satellite positions at t0 (ECEF)
- Filters by elevation ≥ mask at reference site
- Returns top-N by elevation (descending)

### Step 3: User-to-Satellite Association
```python
assoc = associate_users_balanced(
    user_ecef_m, user_demand_mbps, user_qos_w,
    sat_ecef_m, elev_mask_deg=25.0,
    load_mode="wq_demand",  # "count" | "demand" | "wq_demand"
    slack=0.15, max_rounds=6
)
```
- Initial assignment: Each user → highest-elevation visible satellite
- Rebalancing: Move "cheap-to-move" users from overloaded satellites
- Soft cap: (total_load / n_sats) × (1 + slack)
- Load metrics:
  - `count`: Number of users
  - `demand`: Σ demand_mbps
  - `wq_demand`: Σ (demand_mbps × qos_w) ← **default**

### Step 4: Per-Satellite Clustering

For each satellite:

#### 4a. Split-to-Feasible
```python
clusters, evals = split_to_feasible(users_sat, cfg)
```
- Queue-based: Start with all users in one cluster
- Pop cluster → evaluate → if infeasible, split and push children
- Split method: Farthest-point bisection (fast, balanced)
- Stop when all clusters feasible

#### 4b. QoS Refinement (Enterprise Edge Protection)
```python
clusters_ref, evals_ref, stats = refine_enterprise_by_angle(
    users_sat, cfg, clusters, evals,
    n_rounds=3, kcand=6, max_moves_per_round=2000
)
```
- Reduces enterprise users at beam edges
- Angle-based candidate selection (3D geometry)
- Decreases `ent_edge_pct` (% of enterprise users with z > 0.7)

#### 4c. Load-Balance Refinement
```python
clusters_lb, evals_lb, stats = refine_load_balance_by_overlap(
    users_sat, cfg, clusters_ref, evals_ref
)
```
- Minimizes U_max (or range/variance)
- Frozen beam geometry (centers + radii fixed)
- Only moves users between overlapping beams
- Enterprise guards: Exposure/risk cannot increase

### Step 5: Baseline Runs

For each satellite, run all baselines with **same seed** for fair comparison:
- Weighted K-Means (demand)
- Weighted K-Means (demand×QoS)
- BK-Means (optional: `enable_fastbp_baselines=True`)
- TGBP (optional)

Each baseline produces:
- **Fixed-K results**: Using baseline-computed centers (frozen)
- **Repaired results**: After split-until-feasible repair

### Step 6: Global Aggregation
```python
global_summary = summarize_multisat(pieces, cfg)
```
Aggregates across all satellites:
- `K`: Total beams
- `U_mean`, `U_max`, `U_min`: Utilization statistics
- `ent_total`, `ent_exposed`, `ent_edge_pct`: Enterprise edge metrics
- `ent_z_mean`, `ent_z_p90`, `ent_z_max`: Radial distance percentiles
- `risk_sum`: Total enterprise edge-risk penalty

---

## Baseline Comparisons

### Metrics Tracked

| Metric | Description | Goal |
|--------|-------------|------|
| **K** | Number of beams | Minimize (fewer beams = lower cost) |
| **U_max** | Maximum beam utilization | Minimize (balanced load) |
| **U_mean** | Average beam utilization | Maximize efficiency |
| **ent_edge_pct** | % Enterprise users at edge (z>0.7) | Minimize (QoS compliance) |
| **risk_sum** | Σ(max(0, z_ent-0.7)²) | Minimize (edge penalty) |
| **Runtime** | Total execution time (seconds) | Minimize (scalability) |

### Expected Performance

**Phase A (Robustness)**: 2500 users, 10 seeds, Turkey region with hotspots
- **K**: 60-80 beams (typical)
- **U_max**: 0.75-0.85 (main+QoS+LB)
- **ent_edge_pct**: 2-5% (after refinements)
- **Runtime**: ~10-30s per seed (main algorithm)

**Phase B (Scaling)**: 1k-10k users, 5 seeds
- **K scaling**: Roughly linear with user count
- **U_max stability**: Refinements maintain <0.9 even at 10k users
- **Runtime scaling**: Sub-linear due to efficient spatial indexing

---

## Configuration

### Example: Custom Region

```python
from config import ScenarioConfig, BBox, RunConfig
from dataclasses import replace

custom_bbox = BBox(lat_min=30.0, lat_max=35.0, lon_min=120.0, lon_max=125.0)

cfg = replace(
    ScenarioConfig(),
    region_mode="custom",  # Add custom mode to config.py
    run=replace(RunConfig(), n_users=5000, seed=42, verbose=True)
)
```

### Example: High-Throughput Scenario

```python
from config import ScenarioConfig, PhyConfig, BeamConfig
from dataclasses import replace

high_throughput_cfg = replace(
    ScenarioConfig(),
    phy=replace(
        PhyConfig(),
        bandwidth_hz=500e6,  # 500 MHz (2× default)
        eirp_dbw=45.0,        # 5 dB more power
    ),
    beam=replace(
        BeamConfig(),
        radius_modes_km=(3.0, 7.0, 12.0, 18.0)  # Smaller beams for higher gain
    )
)
```

### Example: Disable Load-Balance Refinement

```python
from config import ScenarioConfig, LoadBalanceRefineConfig
from dataclasses import replace

cfg = replace(
    ScenarioConfig(),
    lb_refine=replace(LoadBalanceRefineConfig(), enabled=False)
)
```

---

## Output & Visualization

### CSV Output

`sweep_phaseA.csv` and `sweep_phaseB.csv` contain flattened records with columns:

**Config fields:**
- `seed`, `region_mode`, `n_users`, `use_hotspots`, `n_hotspots`, ...

**Algorithm summaries (per algorithm):**
- `main_K`, `main_U_max`, `main_ent_edge_pct`, ...
- `main_ref_K`, `main_ref_U_max`, ...
- `main_ref_lb_K`, `main_ref_lb_U_max`, ...

**Baseline summaries:**
- `wk_demand_rep_K`, `wk_qos_rep_K`, `bk_rep_K`, `tgbp_rep_K`, ...

**Timing:**
- `time_split_s`, `time_ent_ref_s`, `time_lb_ref_s`
- `time_baseline_without_qos_s`, `time_baseline_with_qos_s`, ...

**Multi-sat metadata:**
- `ms_tle_path`, `ms_time_utc`, `ms_n_active`, `ms_n_unserved`, `ms_assoc_moves`

### Plots

**Phase A (Robustness):** `plots/phaseA/`
- `<metric>_vs_seed.png`: Scatter + mean line for each metric

**Phase B (Scaling):** `plots/phaseB/`
- `K_vs_nusers.png`: Beam count scaling
- `Umax_vs_nusers.png`: Max utilization vs users
- `ent_edge_pct_vs_nusers.png`: Enterprise edge exposure
- `runtime_methods_vs_nusers.png`: Total runtime comparison (main vs baselines)

**Example plot generation:**
```python
from src.sweep_plots import plot_phaseA, plot_phaseB

plot_phaseA("sweep_phaseA.csv", out_dir="plots/phaseA", show=False)
plot_phaseB("sweep_phaseB.csv", out_dir="plots/phaseB", show=False)
```

**Interactive visualization (single scenario):**
```python
from src.plot import plot_clusters_overlay

plot_clusters_overlay(
    users_xy_m=users.xy_m,
    qos_w=users.qos_w,
    clusters=clusters,
    evals=evals,
    title="Main Algorithm (K=80)",
    draw_circles=True,
    draw_centers=True,
    max_circles=300
)
```

---

## Performance Optimization

### Parallel Execution

```python
from main import run_parallel

configs = [...]  # List of ScenarioConfig instances
rows = run_parallel(configs, max_workers=None)  # Uses CPU count - 1
```

**Progress tracking:**
```
[  45/100]  45.00% | elapsed=2m 34s | eta=3m 12s
```

### Memory Optimization

- **Users container**: Separate `UsersRaw` (satellite-agnostic) and `Users` (per-satellite cached geometry)
- **Frozen dataclasses**: Immutable configs prevent accidental mutations
- **NumPy views**: Avoid unnecessary array copies

### Algorithmic Efficiency

- **Grid-based adjacency** (`refine_load_balance.py`): O(K) overlap queries instead of O(K²)
- **Incremental feasibility** (`refine_load_balance.py`): No re-evaluation of unaffected clusters
- **Top-K selection** (`refine_qos_angle.py`): `argpartition` instead of full sort
- **Early termination**: Refinement rounds stop when no moves accepted

### Profiling

```python
from src.profiling import Profiler

prof = Profiler()
prof.tic("operation")
# ... code ...
prof.toc("operation")

print(f"Time: {prof.t['operation']:.3f}s")
print(f"Calls: {prof.c.get('operation_count', 0)}")
```

---

## Research Applications

### 1. QoS Differentiation Studies
- Compare algorithms with/without QoS refinement
- Vary QoS probabilities and weights
- Analyze trade-offs: beam count vs. enterprise edge protection

### 2. Constellation Design
- Sweep satellite altitude (600-1200 km)
- Vary active satellite count (5-50)
- Study elevation mask impact on coverage

### 3. Beam Technology Assessment
- Compare discrete footprint modes vs. continuous optimization
- Analyze trade-offs: larger beams (fewer, higher capacity) vs. smaller beams (more, lower U)

### 4. Load-Balancing Strategies
- Compare objectives: max, range, variance
- Analyze inter-beam handoff frequency
- Study convergence properties (rounds, moves)

### 5. Demand Modeling
- Vary demand distributions (lognormal parameters)
- Hotspot vs. uniform user distributions
- Time-of-day traffic patterns (not yet implemented: would need temporal snapshots)

### 6. Baseline Benchmarking
- Reproduce results from Fast Beam Placement paper (BK-Means, TGBP)
- Compare weighted vs. unweighted clustering
- Analyze repair cost (K_before vs. K_after)

---

## Advanced Usage

### Custom User Generation

```python
from src.usergen import generate_users
from src.models import User
import numpy as np

def custom_user_generator(cfg: ScenarioConfig) -> list[User]:
    # Your custom logic here
    # Example: Grid placement
    N = cfg.run.n_users
    grid_size = int(np.ceil(np.sqrt(N)))
    xy = np.stack(np.meshgrid(
        np.linspace(0, 100_000, grid_size),
        np.linspace(0, 100_000, grid_size)
    ), axis=-1).reshape(-1, 2)[:N]

    # ... convert to lat/lon, assign demands, QoS, etc.
    return users
```

### Custom Objective Function

Add a new objective to `refine_load_balance.py`:

```python
elif objective == "p95":
    # Minimize 95th percentile utilization
    obj_current = float(np.percentile(U, 95))
    # ... incremental update logic ...
```

### Multi-Objective Optimization

```python
def pareto_dominates(a: dict, b: dict) -> bool:
    """Return True if a dominates b (minimize U_max, ent_edge_pct, K)."""
    return (a['U_max'] <= b['U_max'] and
            a['ent_edge_pct'] <= b['ent_edge_pct'] and
            a['K'] <= b['K'] and
            (a['U_max'] < b['U_max'] or
             a['ent_edge_pct'] < b['ent_edge_pct'] or
             a['K'] < b['K']))

def find_pareto_front(results: list[dict]) -> list[dict]:
    front = []
    for candidate in results:
        dominated = False
        for other in results:
            if pareto_dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front
```

---

## Troubleshooting

### Common Issues

**1. `RuntimeError: No active satellites found above elev mask`**
- **Cause**: TLE file path incorrect, or elev_mask_deg too high
- **Fix**: Check `cfg.multisat.tle_path`, reduce `elev_mask_deg` (try 15°)

**2. `RuntimeError: Too many clusters. Check demands/PHY parameters.`**
- **Cause**: Scenario is infeasible (demands too high, bandwidth too low)
- **Fix**: Increase `cfg.phy.bandwidth_hz`, decrease `cfg.traffic.demand_mbps_median`, or add larger footprint modes

**3. `ValueError: Mismatch: len(z)={...} but len(cluster)={...}`**
- **Cause**: Cluster/eval alignment corruption (should never happen with frozen dataclasses)
- **Fix**: Report as bug with config + seed

**4. Load-balance refinement very slow**
- **Cause**: Too many overlapping beams, high `max_moves_per_round`
- **Fix**: Reduce `cfg.lb_refine.max_moves_per_round` (try 1000), increase `intersect_margin_m` (try 1000.0)

**5. Baselines (BK-Means, TGBP) disabled**
- **Cause**: `enable_fastbp_baselines=False` in RunConfig
- **Fix**: Set `enable_fastbp_baselines=True` in main.py

---

## Testing

### Unit Tests (Example: src/coords.py)

```python
import numpy as np
from src.coords import llh_to_ecef, ll_to_local_xy_m

def test_ecef_roundtrip():
    lat, lon, h = 39.9334, 32.8597, 0.0  # Ankara
    ecef = llh_to_ecef(np.array([lat]), np.array([lon]), h)
    assert ecef.shape == (1, 3)
    # Known value from online calculators (~±1 m)
    expected = np.array([[4156894.0, 2767896.0, 4054878.0]])
    np.testing.assert_allclose(ecef, expected, atol=2.0)

def test_local_xy_zero():
    lat0, lon0 = 40.0, 33.0
    xy = ll_to_local_xy_m(np.array([lat0]), np.array([lon0]), lat0, lon0)
    np.testing.assert_allclose(xy, [[0.0, 0.0]], atol=1e-6)
```

### Integration Test (Example: Full pipeline)

```python
def test_pipeline_turkey_default():
    cfg = ScenarioConfig()
    result = run_scenario(cfg)

    assert result['seed'] == 1
    assert result['n_users'] == 250
    assert result['main_ref_lb']['K'] > 0
    assert 0.0 <= result['main_ref_lb']['U_max'] <= 1.0
    assert result['main_ref_lb']['feasible_rate'] == 1.0
```

---

## Performance Benchmarks

**Hardware:** 12-core Intel i7, 32 GB RAM

| Scenario | n_users | n_sats | K (beams) | Runtime (main) | Runtime (baselines) |
|----------|---------|--------|-----------|----------------|---------------------|
| Turkey, 1 seed | 250 | 10 | 15 | 0.8s | 1.2s |
| Turkey, 1 seed | 2500 | 10 | 80 | 12s | 35s |
| Turkey, 1 seed | 5000 | 10 | 150 | 30s | 85s |
| Turkey, 1 seed | 10000 | 10 | 280 | 85s | 210s |

**Parallel sweep (Phase B: 4 sizes × 5 seeds = 20 runs):** ~25 minutes (10 workers)

---

## Contributing

Contributions are welcome! Areas for improvement:

1. **Temporal dynamics**: Multi-snapshot scenarios with beam handoffs
2. **Inter-satellite links**: Extend to mesh networks with ISL constraints
3. **Advanced PHY**: Implement DVB-S2X MODCOD adaptation, rain fading
4. **GPU acceleration**: Port bottlenecks (evaluate_cluster, association) to CUDA
5. **ML-based initialization**: Replace split-to-feasible with learned clustering
6. **Web interface**: Real-time visualization with Plotly/Dash

### Development Setup

```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Run formatter
black src/ main.py config.py

# Run linter
flake8 src/ main.py config.py --max-line-length=120

# Run type checker
mypy src/ main.py config.py --ignore-missing-imports

# Run tests
pytest tests/
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{qos_aware_clustering_2024,
  title = {QoS-Aware User Clustering for LEO Satellite Networks},
  author = {[Your Name/Organization]},
  year = {2024},
  url = {[Repository URL]},
  note = {Python framework for multi-satellite beam placement with QoS differentiation}
}
```

**Related papers:**
- Fast Beam Placement algorithms (BK-Means, TGBP): [Add reference]
- Weighted K-Means for satellite networks: [Add reference]

---

## License

[Specify your license: MIT, Apache 2.0, GPL, etc.]

**Dependencies:**
- NumPy: BSD-3-Clause
- Matplotlib: PSF-based
- Skyfield: MIT
- SciPy: BSD-3-Clause
- scikit-learn: BSD-3-Clause

---

## Contact

For questions, bug reports, or collaboration inquiries:
- **Issues:** [GitHub Issues Link]
- **Email:** [Your Email]
- **Discussions:** [GitHub Discussions Link]

---

## Acknowledgments

- Skyfield team for excellent TLE parsing and ephemeris computation
- Fast Beam Placement paper authors for baseline algorithm specifications
- [Any funding agencies, collaborators, or institutions]

---

**Last Updated:** 2025-02-01
**Version:** 1.0.0
**Status:** Research prototype (active development)
