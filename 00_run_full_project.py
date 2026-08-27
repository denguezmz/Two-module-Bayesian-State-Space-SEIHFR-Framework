from __future__ import annotations
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_pipeline import build_harmonised_data
from src.state_space import run_state_space_model
from src.seihfr import calibrate_seihfr, run_scenarios
from src.sensitivity import run_global_sensitivity
from src.effective_population_sensitivity import run_effective_population_sensitivity
from src.utils import ensure_project_dirs, load_config


def _write_run_manifest(config: dict, elapsed_seconds: float) -> None:
    """Record the resolved computation profile and fingerprints of key outputs."""
    key_outputs = [
        "results/state_space_latent_rt.csv",
        "results/state_space_rt_posterior_draws.npz",
        "results/seihfr_cut_posterior_particles.csv",
        "results/seihfr_smc_diagnostics.csv",
        "results/seihfr_fit_metrics.csv",
        "results/seihfr_calibration_observations.csv",
        "results/seihfr_scenario_summary.csv",
        "results/seihfr_global_sensitivity.csv",
        "results/seihfr_effective_population_sensitivity.csv",
        "figures/Figure_1_Data_reconstruction.png",
        "figures/Figure_2_Transmission_state.png",
        "figures/Figure_3_Mechanistic_reconstruction.png",
        "figures/Figure_4_Pathway_contributions_and_identifiability.png",
        "figures/Figure_04B_Mechanistic_fit_diagnostics.png",
        "figures/Figure_5_Policy_scenarios.png",
        "figures/Figure_6_SEIHFR_policy_sensitivity.png",
        "figures/Figure_S4_Hospital_stock_calibration.png",
    ]
    fingerprints = {}
    for relative in key_outputs:
        path = ROOT / relative
        if path.exists():
            fingerprints[relative] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": config["run_mode"],
        "random_seed": int(config["random_seed"]),
        "analysis_date": config["analysis_date"],
        "data_cutoff": config["data_cutoff"],
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "resolved_computation_profile": {
            "rt_posterior_paths": int(config["state_space"]["posterior_path_draws"]),
            "smc_particles": int(config["seihfr"]["module2_prior_draws"]),
            "smc_rejuvenation_steps": int(config["seihfr"]["smc_rejuvenation_steps"]),
            "posterior_resample_size": int(config["seihfr"]["posterior_resample_size"]),
            "paths_per_scenario": int(config["forecast"]["paths_per_scenario"]),
            "sobol_base_size": int(config["sensitivity"]["sobol_base_size"]),
            "prcc_size": int(config["sensitivity"]["prcc_size"]),
            "effective_population_values": config["sensitivity"].get("effective_population_values", []),
        },
        "output_fingerprints": fingerprints,
    }
    path = ROOT / "results" / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    started = time.perf_counter()
    ensure_project_dirs(); config=load_config(); print(f"Run mode: {config['run_mode']}")
    print("[1/6] Harmonising aggregate anchors and reconstructing event-time cases..."); data=build_harmonised_data(config,save=True)
    print("[2/6] Module 1: renewal-informed state-space Rt inference..."); ss=run_state_space_model(data.daily,config,save=True,backfill_sensitivity=data.case_backfill_sensitivity)
    print("[3/6] Module 2: SEIHFR modular cut-posterior calibration..."); cal=calibrate_seihfr(ss,data.anchors,config,save=True)
    print("[4/6] Running 90-day policy scenarios and delayed-response analysis..."); scen=run_scenarios(cal,config,save=True)
    print("[5/6] Running Sobol/PRCC global sensitivity analysis..."); sens=run_global_sensitivity(cal,config,save=True)
    print("[6/6] Running effective-population sensitivity analysis..."); neff=run_effective_population_sensitivity(cal,config,save=True)
    from src.figures import generate_all_figures
    generate_all_figures(data,ss,cal,scen,sens,config,neff)
    _write_run_manifest(config, time.perf_counter() - started)
    print("\nDone. Workflow: event-time reconstruction -> latent Rt posterior -> SEIHFR cut posterior -> pathway decomposition -> 90-day policy scenarios -> global sensitivity -> effective-population sensitivity.")
if __name__=='__main__': main()
