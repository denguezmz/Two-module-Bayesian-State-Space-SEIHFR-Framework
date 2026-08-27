from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("RUN_MODE", "publication")

from src.data_pipeline import build_harmonised_data
from src.sensitivity import run_global_sensitivity
from src.seihfr import calibrate_seihfr, run_scenarios
from src.state_space import run_state_space_model
from src.utils import load_config


def _scenario_value(summary: pd.DataFrame, scenario: str, column: str) -> float:
    return float(summary.set_index("scenario").loc[scenario, column])


def run_seed(base_config: dict, seed: int) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    config["random_seed"] = int(seed)
    data = build_harmonised_data(config, save=False)
    state = run_state_space_model(
        data.daily,
        config,
        save=False,
        backfill_sensitivity=data.case_backfill_sensitivity,
    )
    calibration = calibrate_seihfr(state, data.anchors, config, save=False)
    scenarios = run_scenarios(calibration, config, save=False)
    sensitivity = run_global_sensitivity(calibration, config, save=False)

    fit = calibration.fit_metrics.set_index("target")
    scen = scenarios.summary
    accelerated_cases = _scenario_value(scen, "Accelerated integrated response", "cases90_median")
    current_cases = _scenario_value(scen, "Current scale-up", "cases90_median")
    accelerated_deaths = _scenario_value(scen, "Accelerated integrated response", "deaths90_median")
    current_deaths = _scenario_value(scen, "Current scale-up", "deaths90_median")
    delay14 = float(
        scenarios.delay.set_index("delay_days").loc[14, "additional_cases90_median"]
    )
    ranked = sensitivity.indices.sort_values(
        "total_order_sobol_cases90", ascending=False
    )["parameter"].tolist()
    bridge = calibration.bridge_diagnostics.iloc[0]
    latest_rt = state.fit.iloc[-1]
    latest_mechanism = calibration.mechanism_decomposition.iloc[-1]
    return {
        "seed": int(seed),
        "module1_latest_rt_median": float(latest_rt["rt_smoothed_median"]),
        "module1_latest_prob_rt_gt_1": float(latest_rt["prob_rt_gt_1"]),
        "module2_latest_rt_median": float(latest_mechanism["R_total_median"]),
        "bridge_coverage_95": float(bridge["state_space_median_inside_seihfr_95"]),
        "cases_wape": float(fit.loc["cases", "WAPE"]),
        "deaths_wape": float(fit.loc["deaths", "WAPE"]),
        "hospital_wape": float(fit.loc["hospital", "WAPE"]),
        "accelerated_cases90_median": accelerated_cases,
        "current_cases90_median": current_cases,
        "accelerated_case_reduction_fraction": 1.0 - accelerated_cases / current_cases,
        "accelerated_death_reduction_fraction": 1.0 - accelerated_deaths / current_deaths,
        "delay14_additional_cases90_median": delay14,
        "smc_final_acceptance": float(calibration.smc_diagnostics.iloc[-1]["mutation_acceptance"]),
        "sobol_rank_1": ranked[0],
        "sobol_rank_2": ranked[1],
        "sobol_rank_3": ranked[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed publication-mode robustness audit.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seeds to run. Default: configured seed and two deterministic offsets.",
    )
    args = parser.parse_args()
    config = load_config()
    base_seed = int(config["random_seed"])
    seeds = args.seeds or [base_seed, base_seed + 101, base_seed + 202]
    rows = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] publication robustness seed={seed}")
        rows.append(run_seed(config, seed))

    result = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "publication_seed_robustness.csv", index=False)

    numeric = result.select_dtypes(include=[np.number]).drop(columns=["seed"])
    summary = {
        "run_mode": "publication",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "numeric_ranges": {
            column: {
                "min": float(numeric[column].min()),
                "median": float(numeric[column].median()),
                "max": float(numeric[column].max()),
            }
            for column in numeric.columns
        },
        "sobol_top_three_by_seed": result[
            ["seed", "sobol_rank_1", "sobol_rank_2", "sobol_rank_3"]
        ].to_dict(orient="records"),
    }
    (out_dir / "publication_seed_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_dir / "publication_seed_robustness.csv")


if __name__ == "__main__":
    main()
