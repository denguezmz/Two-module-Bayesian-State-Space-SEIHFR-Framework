from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .seihfr import SEIHFRCalibrationResults, run_scenarios
from .utils import PROJECT_ROOT


def run_effective_population_sensitivity(
    calibration: SEIHFRCalibrationResults,
    config: dict,
    save: bool = True,
) -> pd.DataFrame:
    """Conditional sensitivity to the fixed effective connected population.

    The SEIHFR posterior is held fixed and only the forward susceptible pool is
    changed.  This targets whether 90-day policy conclusions hinge on the
    assumed N_eff, not a full re-calibration under each population value.
    """
    values = config.get("sensitivity", {}).get(
        "effective_population_values",
        [2_500_000, 5_000_000, 10_000_000],
    )
    base_n = float(config["seihfr"]["effective_population"])
    rows = []
    for value in values:
        cfg = deepcopy(config)
        n_eff = float(value)
        cfg["seihfr"]["effective_population"] = n_eff
        cfg["random_seed"] = int(config["random_seed"]) + int(round(n_eff / 100_000))
        scenarios = run_scenarios(calibration, cfg, save=False)
        summary = scenarios.summary.copy()
        for _, row in summary.iterrows():
            cases90 = float(row["cases90_median"])
            deaths90 = float(row["deaths90_median"])
            peak_hospital90 = float(row["peak_hospital90_median"])
            rows.append(
                {
                    "effective_population": int(round(n_eff)),
                    "relative_to_primary": n_eff / base_n,
                    "scenario": row["scenario"],
                    "cases90_median": cases90,
                    "cases90_q2_5": float(row["cases90_q2_5"]),
                    "cases90_q97_5": float(row["cases90_q97_5"]),
                    "deaths90_median": deaths90,
                    "deaths90_q2_5": float(row["deaths90_q2_5"]),
                    "deaths90_q97_5": float(row["deaths90_q97_5"]),
                    "peak_hospital90_median": peak_hospital90,
                    "peak_hospital90_q2_5": float(row["peak_hospital90_q2_5"]),
                    "peak_hospital90_q97_5": float(row["peak_hospital90_q97_5"]),
                    "rt90_median": float(row["rt90_median"]),
                    "rt90_q2_5": float(row["rt90_q2_5"]),
                    "rt90_q97_5": float(row["rt90_q97_5"]),
                    "prob_rt90_below_1": float(row["prob_rt90_below_1"]),
                    "susceptible_depletion90_percent": 100.0 * cases90 / max(n_eff, 1.0),
                }
            )
    out = pd.DataFrame(rows)
    primary = out[out["effective_population"] == int(round(base_n))][
        ["scenario", "cases90_median", "deaths90_median", "peak_hospital90_median", "rt90_median"]
    ].rename(
        columns={
            "cases90_median": "primary_cases90_median",
            "deaths90_median": "primary_deaths90_median",
            "peak_hospital90_median": "primary_peak_hospital90_median",
            "rt90_median": "primary_rt90_median",
        }
    )
    out = out.merge(primary, on="scenario", how="left")
    out["cases90_percent_change_vs_primary"] = 100.0 * (
        out["cases90_median"] / out["primary_cases90_median"] - 1.0
    )
    out["deaths90_percent_change_vs_primary"] = 100.0 * (
        out["deaths90_median"] / out["primary_deaths90_median"] - 1.0
    )
    out["peak_hospital90_percent_change_vs_primary"] = 100.0 * (
        out["peak_hospital90_median"] / out["primary_peak_hospital90_median"] - 1.0
    )
    out["rt90_percent_change_vs_primary"] = 100.0 * (
        out["rt90_median"] / out["primary_rt90_median"] - 1.0
    )
    if save:
        out.to_csv(PROJECT_ROOT / "results" / "seihfr_effective_population_sensitivity.csv", index=False)
    return out
