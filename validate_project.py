from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty: {path.relative_to(ROOT)}")


def main() -> None:
    required = [
        ROOT / "data" / "official_aggregate_anchors.csv",
        ROOT / "data" / "backfill_case_sensitivity_series.csv",
        ROOT / "results" / "state_space_parameter_summary.csv",
        ROOT / "results" / "state_space_latent_rt.csv",
        ROOT / "results" / "state_space_rt_posterior_draws.npz",
        ROOT / "results" / "state_space_backfill_sensitivity_summary.csv",
        ROOT / "results" / "seihfr_cut_posterior_particles.csv",
        ROOT / "results" / "seihfr_smc_diagnostics.csv",
        ROOT / "results" / "seihfr_parameter_summary.csv",
        ROOT / "results" / "seihfr_prior_posterior_learning.csv",
        ROOT / "results" / "seihfr_parameter_correlation.csv",
        ROOT / "results" / "seihfr_posterior_predictive_fit.csv",
        ROOT / "results" / "seihfr_fit_metrics.csv",
        ROOT / "results" / "seihfr_calibration_observations.csv",
        ROOT / "results" / "seihfr_mechanism_decomposition_daily.csv",
        ROOT / "results" / "seihfr_bridge_diagnostics.csv",
        ROOT / "results" / "seihfr_scenario_summary.csv",
        ROOT / "results" / "seihfr_intervention_delay.csv",
        ROOT / "results" / "seihfr_global_sensitivity.csv",
        ROOT / "results" / "seihfr_effective_population_sensitivity.csv",
        ROOT / "results" / "run_manifest.json",
        ROOT / "results" / "publication_seed_robustness.csv",
        ROOT / "figures" / "Figure_1_Data_reconstruction.png",
        ROOT / "figures" / "Figure_2_Transmission_state.png",
        ROOT / "figures" / "Figure_3_Mechanistic_reconstruction.png",
        ROOT / "figures" / "Figure_4_Pathway_contributions_and_identifiability.png",
        ROOT / "figures" / "Figure_5_Policy_scenarios.png",
        ROOT / "figures" / "Figure_6_SEIHFR_policy_sensitivity.png",
        ROOT / "figures" / "Figure_S4_Hospital_stock_calibration.png",
    ]
    for path in required:
        require(path)

    anchors = pd.read_csv(ROOT / "data" / "official_aggregate_anchors.csv").sort_values("date")
    latest = anchors.iloc[-1]
    assert int(latest["confirmed_cases"]) == 5290
    assert int(latest["confirmed_deaths"]) == 2516

    backfill = pd.read_csv(ROOT / "data" / "backfill_case_sensitivity_series.csv")
    assert backfill["scenario"].nunique() == 6
    for _, group in backfill.groupby("scenario"):
        assert np.isclose(group["notification_backfill_removed"].sum(), 272.0)
        assert np.isclose(group["historical_backfill_added"].sum(), 272.0)
        assert np.isclose(group["reconstructed_incidence_cases"].sum(), 5290.0)

    rt = pd.read_csv(ROOT / "results" / "state_space_latent_rt.csv")
    assert {"rt_smoothed_median", "rt_smoothed_lo", "rt_smoothed_hi", "prob_rt_gt_1"}.issubset(rt.columns)
    assert ((rt["prob_rt_gt_1"] >= 0) & (rt["prob_rt_gt_1"] <= 1)).all()
    draws = np.load(ROOT / "results" / "state_space_rt_posterior_draws.npz")
    assert len(draws.files) > 0

    smc = pd.read_csv(ROOT / "results" / "seihfr_smc_diagnostics.csv")
    assert len(smc) >= 2 and np.isclose(float(smc.iloc[-1]["gamma_end"]), 1.0)
    assert (smc.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).notna().all().all())
    assert float(smc["mutation_acceptance"].iloc[-1]) >= 0.02

    particles = pd.read_csv(ROOT / "results" / "seihfr_cut_posterior_particles.csv")
    assert len(particles) >= 50

    fit = pd.read_csv(ROOT / "results" / "seihfr_fit_metrics.csv")
    assert {"cases", "deaths", "hospital"}.issubset(set(fit["target"]))
    assert np.isfinite(fit[["MAE", "RMSE", "WAPE", "coverage_95"]].to_numpy()).all()
    fit_by_target = fit.set_index("target")
    assert float(fit_by_target.loc["cases", "WAPE"]) < 0.10
    assert float(fit_by_target.loc["deaths", "WAPE"]) < 0.15
    assert float(fit_by_target.loc["hospital", "WAPE"]) < 0.15
    assert float(fit_by_target.loc["hospital", "coverage_95"]) >= 0.85
    assert (fit["coverage_95"] >= 0.70).all()

    stock_obs = pd.read_csv(ROOT / "results" / "seihfr_calibration_observations.csv")
    hospital_obs = stock_obs[stock_obs["target"] == "hospital"]
    assert len(hospital_obs) >= 8
    assert hospital_obs["inside_95"].astype(bool).mean() >= 0.85

    mech = pd.read_csv(ROOT / "results" / "seihfr_mechanism_decomposition_daily.csv")
    needed = {
        "R_community_median", "R_hospital_median", "R_postdeath_median", "R_total_median",
        "community_share_median", "hospital_share_median", "postdeath_share_median"
    }
    assert needed.issubset(mech.columns)
    assert np.isfinite(mech[list(needed)].to_numpy()).all()

    learning = pd.read_csv(ROOT / "results" / "seihfr_prior_posterior_learning.csv")
    assert {"parameter", "interval_contraction_ratio"}.issubset(learning.columns)
    corr = pd.read_csv(ROOT / "results" / "seihfr_parameter_correlation.csv")
    assert {"parameter_a", "parameter_b", "weighted_correlation"}.issubset(corr.columns)

    scenarios = pd.read_csv(ROOT / "results" / "seihfr_scenario_summary.csv")
    assert set(scenarios["scenario"]) >= {
        "Accelerated integrated response", "Current scale-up", "Stalled response", "Operational disruption"
    }
    assert "cases90_median" in scenarios.columns and "rt90_median" in scenarios.columns
    assert np.isfinite(scenarios.select_dtypes(include=[np.number]).to_numpy()).all()
    for prefix in ["cases30", "cases60", "cases90", "deaths90", "peak_hospital90", "rt90"]:
        qcols = [f"{prefix}_q2_5", f"{prefix}_median", f"{prefix}_q97_5"]
        assert (scenarios[qcols[0]] <= scenarios[qcols[1]]).all()
        assert (scenarios[qcols[1]] <= scenarios[qcols[2]]).all()
    scenario_cases = scenarios.set_index("scenario")["cases90_median"]
    assert scenario_cases["Accelerated integrated response"] < scenario_cases["Current scale-up"]
    assert scenario_cases["Current scale-up"] < scenario_cases["Stalled response"]
    assert scenario_cases["Stalled response"] < scenario_cases["Operational disruption"]

    bridge = pd.read_csv(ROOT / "results" / "seihfr_bridge_diagnostics.csv")
    assert float(bridge.iloc[0]["state_space_median_inside_seihfr_95"]) >= 0.50

    delay = pd.read_csv(ROOT / "results" / "seihfr_intervention_delay.csv")
    assert {0, 7, 14, 28}.issubset(set(delay["delay_days"]))

    sensitivity = pd.read_csv(ROOT / "results" / "seihfr_global_sensitivity.csv")
    assert {"parameter", "total_order_sobol_cases90", "PRCC_cases90"}.issubset(sensitivity.columns)
    assert set(sensitivity["parameter"]) == {
        "Community transmission multiplier", "Known-contact follow-up", "Symptom-to-isolation delay",
        "Safe burial coverage", "Health-facility IPC", "Response disruption severity"
    }

    neff = pd.read_csv(ROOT / "results" / "seihfr_effective_population_sensitivity.csv")
    assert {2500000, 5000000, 10000000}.issubset(set(neff["effective_population"]))
    assert {"scenario", "cases90_percent_change_vs_primary", "susceptible_depletion90_percent"}.issubset(neff.columns)
    assert np.isfinite(neff.select_dtypes(include=[np.number]).to_numpy()).all()

    manifest = json.loads((ROOT / "results" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_mode"] in {"smoke", "standard", "publication"}
    assert int(manifest["random_seed"]) > 0
    assert float(manifest["elapsed_seconds"]) > 0
    assert len(manifest["output_fingerprints"]) >= 8

    robustness = pd.read_csv(ROOT / "results" / "publication_seed_robustness.csv")
    assert robustness["seed"].nunique() >= 3
    assert (robustness[["cases_wape", "deaths_wape", "hospital_wape"]].to_numpy() < np.array([0.12, 0.15, 0.16])).all()
    assert (robustness["bridge_coverage_95"] >= 0.50).all()
    assert (robustness["accelerated_case_reduction_fraction"] > 0).all()
    assert (robustness["accelerated_death_reduction_fraction"] > 0).all()

    print(
        "Validation passed: event-time reconstruction, Module 1 joint Rt posterior, "
        "Module 2 tempered SMC cut posterior, posterior predictive fits, pathway decomposition, "
        "90-day SEIHFR policy scenarios, delayed-response analysis, Sobol/PRCC, and "
        "effective-population sensitivity are internally consistent."
    )


if __name__ == "__main__":
    main()
