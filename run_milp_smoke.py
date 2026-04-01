from dataclasses import replace
from config import ScenarioConfig
from src.milp.runner import run_milp_experiment, MILPRunnerConfig

# Replace this with however you normally build your config in your project
base = ScenarioConfig()

cfg = replace(
            base,
            region_mode="turkey",
            run=replace(
                base.run,
                seed=2,
                n_users=50,
                enable_plots=True,
                verbose=False,
                enable_fastbp_baselines = True
            ),
            usergen=replace(
                base.usergen,
                enabled=True,
                n_hotspots=5,
                hotspot_sigma_m_min=5_000.0,
                hotspot_sigma_m_max=30_000.0,
                noise_frac=0.15,
            ),
            lb_refine=replace(
                base.lb_refine,
                enabled=True
            )
        )

run_cfg = MILPRunnerConfig(
    n_candidate_sats=6,
    grid_spacing_m=15000.0,
    time_limit_s=300.0,
    mip_gap=0.0,
    log_to_console=True,
    objective_mode="weighted_sat_beam",
    print_diagnostics=True,
)

out = run_milp_experiment(cfg, run_cfg=run_cfg)

sol = out["solution"]
data = out["data"]
grid = out["grid"]

print("=== MILP RESULT ===")
print("feasible:", sol.feasible)
print("status:", sol.status_name)
print("objective:", sol.objective_value)
print("mip_gap:", sol.mip_gap)
print("solve_time_s:", sol.solve_time_s)
print("used_sat_indices:", sol.used_sat_indices)
print("n_used_sats:", sol.n_used_sats)
print("K_total:", sol.K_total)
print("n_grid_centers:", len(grid))
print("n_candidates:", len(data.candidates))