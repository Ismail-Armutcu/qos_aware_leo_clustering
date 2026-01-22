# qos_aware_user_clustering

QoS-aware user clustering for satellite beam planning. The code simulates a
regional user population, evaluates beam feasibility using a simplified PHY
model, and builds clusters that satisfy geometry and capacity constraints. It
compares a split-to-feasible heuristic against weighted k-means baselines and
reports QoS risk metrics for enterprise users.

## What this repository does

The pipeline:
- Generates synthetic users in a geographic bounding box (uniform or hotspots),
  each with a demand and QoS class (1, 2, or 4).
- Converts user locations to local XY and ECEF to compute distances and angles.
- Evaluates candidate clusters with discrete beam radius modes and a simplified
  link budget (FSPL, Gaussian mainlobe, Shannon rate).
- Builds clusters by splitting until all are feasible.
- Optionally refines clusters to move enterprise users away from beam edges.
- Runs weighted k-means baselines and repairs infeasible clusters by splitting.
- Plots user distributions and cluster footprints.

## Key modules

- `main.py`: Entry point that runs the full experiment and prints/plots results.
- `config.py`: Scenario configuration (region, PHY parameters, QoS settings).
- `src/usergen.py`: Synthetic user generator (uniform or hotspot+noise).
- `src/coords.py`: Lat/lon <-> local XY and ECEF conversions.
- `src/phy.py`: Link budget helpers (FSPL, antenna gain, SNR, Shannon rate).
- `src/evaluator.py`: Cluster feasibility evaluation and QoS risk metrics.
- `src/pipeline.py`: Split-to-feasible clustering algorithm.
- `src/refine_qos_angle.py`: QoS refinement via angular reassignment.
- `src/baselines/weighted_kmeans.py`: Weighted k-means++ baseline.
- `src/baselines/repair.py`: Strict repair by splitting infeasible clusters.
- `src/plot.py`: Visualization utilities.

## Algorithms

### Split-to-feasible (main)

Start with a single cluster containing all users. Evaluate feasibility:
- Geometry: cluster must fit within a discrete beam radius mode.
- Capacity: time-share utilization U <= 1 based on weighted demand / rate.

If a cluster is infeasible, split it using a farthest-point bisection and
re-evaluate. The process continues until all clusters are feasible.

### QoS refinement (enterprise risk reduction)

Enterprise users (QoS=4) should avoid beam edges. The refinement step moves
enterprise users to candidate clusters with smaller off-axis angles when:
- Both affected clusters remain feasible.
- The count of enterprise users near the edge decreases, or risk drops.

### Baselines

Weighted k-means++ is run with:
- weights = demand
- weights = demand * QoS

Each baseline is also repaired by repeatedly splitting infeasible clusters.

## Metrics reported

- Number of clusters (K)
- Feasible cluster rate
- Utilization U (mean, max, min)
- Enterprise edge exposure (% with normalized radius z > rho_safe)
- Aggregate enterprise risk (soft penalty)

## Running

From the repo root:

```powershell
python main.py
```

Plots are shown using matplotlib.

