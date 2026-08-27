from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .transmission import gamma_discrete_weights, infectiousness_series
from .utils import PROJECT_ROOT


@dataclass
class StateSpaceResults:
    fit: pd.DataFrame
    parameter_summary: pd.DataFrame
    full_daily: pd.DataFrame
    latent_log_rt_mean: np.ndarray
    latent_log_rt_var: np.ndarray
    generation_interval_weights: np.ndarray
    backfill_sensitivity_daily: pd.DataFrame
    backfill_sensitivity_summary: pd.DataFrame
    bridge_targets: pd.DataFrame
    posterior_log_rt_draws: np.ndarray


def _kalman_filter_local_level(
    y: np.ndarray,
    obs_var: np.ndarray,
    process_var: float,
    initial_mean: float = 0.0,
    initial_var: float = 1.0,
) -> dict[str, np.ndarray | float]:
    n = len(y)
    m_pred = np.zeros(n)
    p_pred = np.zeros(n)
    m_filt = np.zeros(n)
    p_filt = np.zeros(n)
    innovations = np.zeros(n)
    innovation_var = np.zeros(n)
    loglik = 0.0

    m_prev = initial_mean
    p_prev = initial_var
    for t in range(n):
        m_pred[t] = m_prev
        p_pred[t] = p_prev + process_var
        innovation = y[t] - m_pred[t]
        f = max(float(p_pred[t] + obs_var[t]), 1e-10)
        k = p_pred[t] / f
        m_filt[t] = m_pred[t] + k * innovation
        p_filt[t] = max((1.0 - k) * p_pred[t], 1e-10)
        innovations[t] = innovation
        innovation_var[t] = f
        loglik += -0.5 * (math.log(2 * math.pi) + math.log(f) + innovation**2 / f)
        m_prev = m_filt[t]
        p_prev = p_filt[t]

    return {
        "m_pred": m_pred,
        "p_pred": p_pred,
        "m_filt": m_filt,
        "p_filt": p_filt,
        "innovations": innovations,
        "innovation_var": innovation_var,
        "loglik": float(loglik),
    }


def _rts_smoother(filter_result: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray]:
    m_pred = np.asarray(filter_result["m_pred"], dtype=float)
    p_pred = np.asarray(filter_result["p_pred"], dtype=float)
    m_filt = np.asarray(filter_result["m_filt"], dtype=float)
    p_filt = np.asarray(filter_result["p_filt"], dtype=float)
    n = len(m_filt)
    m_smooth = m_filt.copy()
    p_smooth = p_filt.copy()
    for t in range(n - 2, -1, -1):
        denom = max(p_pred[t + 1], 1e-12)
        gain = p_filt[t] / denom
        m_smooth[t] = m_filt[t] + gain * (m_smooth[t + 1] - m_pred[t + 1])
        p_smooth[t] = max(
            p_filt[t] + gain**2 * (p_smooth[t + 1] - p_pred[t + 1]),
            1e-10,
        )
    return m_smooth, p_smooth




def _ffbs_sample_local_level(
    filter_result: dict[str, np.ndarray | float],
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Joint posterior latent log-Rt paths via forward-filter backward-sampling."""
    m_pred=np.asarray(filter_result["m_pred"],float); p_pred=np.asarray(filter_result["p_pred"],float)
    m_filt=np.asarray(filter_result["m_filt"],float); p_filt=np.asarray(filter_result["p_filt"],float)
    n=len(m_filt); draws=np.zeros((int(n_draws),n),float)
    draws[:,-1]=rng.normal(m_filt[-1],math.sqrt(max(p_filt[-1],1e-12)),size=n_draws)
    for t in range(n-2,-1,-1):
        gain=p_filt[t]/max(p_pred[t+1],1e-12); mean=m_filt[t]+gain*(draws[:,t+1]-m_pred[t+1]); var=max(p_filt[t]-gain*gain*p_pred[t+1],1e-12)
        draws[:,t]=rng.normal(mean,math.sqrt(var),size=n_draws)
    return draws


def _fit_local_level(
    y: np.ndarray,
    count_var: np.ndarray,
) -> tuple[float, float, dict, np.ndarray, np.ndarray]:
    def objective(theta: np.ndarray) -> float:
        process_var = float(np.exp(theta[0]))
        base_obs_var = float(np.exp(theta[1]))
        obs_var = base_obs_var + count_var
        result = _kalman_filter_local_level(y, obs_var, process_var)
        penalty = 0.01 * (theta[0] ** 2 + theta[1] ** 2)
        return -float(result["loglik"]) + penalty

    starts = [
        np.log([0.01, 0.08]),
        np.log([0.03, 0.15]),
        np.log([0.005, 0.30]),
    ]
    best = None
    for x0 in starts:
        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(-12.0, 0.0), (-12.0, 2.0)],
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not best.success:
        raise RuntimeError("半机理时变传播状态空间模型超参数优化失败。")

    process_var = float(np.exp(best.x[0]))
    base_obs_var = float(np.exp(best.x[1]))
    obs_var = base_obs_var + count_var
    filt = _kalman_filter_local_level(y, obs_var, process_var)
    smooth_mean, smooth_var = _rts_smoother(filt)
    return process_var, base_obs_var, filt, smooth_mean, smooth_var


def _fit_incidence_state(
    incidence_full: np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
    config: dict,
    weights: np.ndarray,
) -> dict[str, np.ndarray | float]:
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    fit_start = pd.Timestamp(config["fit_start"])
    fit_mask = (dates >= fit_start).to_numpy()
    fit_indices = np.where(fit_mask)[0]

    infectiousness_full = infectiousness_series(incidence_full, weights)
    incidence = np.asarray(incidence_full, dtype=float)[fit_indices]
    infectiousness = infectiousness_full[fit_indices]
    c = float(config["state_space"]["continuity_correction"])

    y = np.log((incidence + c) / (infectiousness + c))
    count_var = 1.0 / (incidence + c) + 1.0 / (infectiousness + c)
    count_var = np.clip(count_var, 0.005, 3.0)

    process_var, base_obs_var, filt, m_smooth, p_smooth = _fit_local_level(y, count_var)
    obs_var = base_obs_var + count_var

    m_filt = np.asarray(filt["m_filt"], dtype=float)
    p_filt = np.asarray(filt["p_filt"], dtype=float)
    m_pred = np.asarray(filt["m_pred"], dtype=float)

    return {
        "fit_mask": fit_mask,
        "fit_indices": fit_indices,
        "infectiousness_full": infectiousness_full,
        "incidence_fit": incidence,
        "infectiousness_fit": infectiousness,
        "log_ratio": y,
        "count_var": count_var,
        "obs_var": obs_var,
        "process_var": process_var,
        "base_obs_var": base_obs_var,
        "filter": filt,
        "filtered_mean": m_filt,
        "filtered_var": p_filt,
        "predicted_mean": m_pred,
        "smoothed_mean": m_smooth,
        "smoothed_var": p_smooth,
    }


def _bridge_targets(
    dates_fit: pd.Series,
    smooth_mean: np.ndarray,
    smooth_var: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Return the full daily Rt trajectory used to guide time-varying beta(t).

    v5 no longer compresses the state-space output to four beta knots.  Every
    fitted day is retained so that SEIHFR can construct a smooth beta(t) prior
    from the complete Rt trajectory.
    """
    m = np.asarray(smooth_mean, dtype=float)
    v = np.asarray(smooth_var, dtype=float)
    sd = np.sqrt(np.maximum(v, 1e-12))
    return pd.DataFrame(
        {
            "fit_day_index": np.arange(len(dates_fit), dtype=int),
            "date": pd.to_datetime(dates_fit).to_numpy(),
            "log_rt_mean": m,
            "log_rt_variance": v,
            "rt_median": np.exp(m),
            "rt_lower_95": np.exp(m - 1.96 * sd),
            "rt_upper_95": np.exp(m + 1.96 * sd),
            "prob_rt_gt_1": 0.5 * np.vectorize(math.erfc)((0.0 - m) / np.sqrt(2.0 * np.maximum(v, 1e-12))),
        }
    )

def _backfill_state_space_sensitivity(
    sensitivity: pd.DataFrame | None,
    config: dict,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sensitivity is None or sensitivity.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for _, scenario in sensitivity.groupby("scenario", sort=False):
        scenario = scenario.sort_values("date").reset_index(drop=True)
        incidence = scenario["reconstructed_incidence_cases"].to_numpy(float)
        fit = _fit_incidence_state(incidence, scenario["date"], config, weights)
        dates_fit = pd.to_datetime(scenario.loc[fit["fit_mask"], "date"]).reset_index(drop=True)
        m = np.asarray(fit["smoothed_mean"], dtype=float)
        v = np.asarray(fit["smoothed_var"], dtype=float)
        sd = np.sqrt(v)
        out = pd.DataFrame(
            {
                "date": dates_fit,
                "scenario": scenario["scenario"].iloc[0],
                "scenario_cn": scenario["scenario_cn"].iloc[0],
                "is_primary": bool(scenario["is_primary"].iloc[0]),
                "allocation_start": scenario["allocation_start"].iloc[0],
                "allocation_end": scenario["allocation_end"].iloc[0],
                "rt_median": np.exp(m),
                "rt_lower_95": np.exp(m - 1.96 * sd),
                "rt_upper_95": np.exp(m + 1.96 * sd),
            }
        )
        daily_rows.append(out)
        latest = out.iloc[-1]
        summary_rows.append(
            {
                "scenario": latest["scenario"],
                "scenario_cn": latest["scenario_cn"],
                "is_primary": latest["is_primary"],
                "allocation_start": latest["allocation_start"],
                "allocation_end": latest["allocation_end"],
                "latest_date": latest["date"],
                "rt_median": float(latest["rt_median"]),
                "rt_lower_95": float(latest["rt_lower_95"]),
                "rt_upper_95": float(latest["rt_upper_95"]),
                "median_above_one": bool(latest["rt_median"] > 1.0),
                "lower_95_above_one": bool(latest["rt_lower_95"] > 1.0),
            }
        )
    return pd.concat(daily_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_state_space_model(
    daily: pd.DataFrame,
    config: dict,
    save: bool = True,
    backfill_sensitivity: pd.DataFrame | None = None,
) -> StateSpaceResults:
    cfg = config["state_space"]
    gi = config["generation_interval"]
    weights = gamma_discrete_weights(
        float(gi["mean_days"]),
        float(gi["sd_days"]),
        int(gi["max_lag_days"]),
    )

    incidence_full = daily["event_time_incidence_cases"].to_numpy(float)
    fitted = _fit_incidence_state(incidence_full, daily["date"], config, weights)
    fit_mask = np.asarray(fitted["fit_mask"], dtype=bool)
    fit_indices = np.asarray(fitted["fit_indices"], dtype=int)
    dates_fit = pd.to_datetime(daily.loc[fit_mask, "date"]).reset_index(drop=True)
    infectiousness = np.asarray(fitted["infectiousness_fit"], dtype=float)
    m_filt = np.asarray(fitted["filtered_mean"], dtype=float)
    p_filt = np.asarray(fitted["filtered_var"], dtype=float)
    m_smooth = np.asarray(fitted["smoothed_mean"], dtype=float)
    p_smooth = np.asarray(fitted["smoothed_var"], dtype=float)
    m_pred = np.asarray(fitted["predicted_mean"], dtype=float)

    filtered_sd = np.sqrt(p_filt)
    smooth_sd = np.sqrt(p_smooth)
    rt_filtered = np.exp(m_filt)
    rt_filtered_lo = np.exp(m_filt - 1.96 * filtered_sd)
    rt_filtered_hi = np.exp(m_filt + 1.96 * filtered_sd)
    rt_smoothed = np.exp(m_smooth)
    rt_smoothed_lo = np.exp(m_smooth - 1.96 * smooth_sd)
    rt_smoothed_hi = np.exp(m_smooth + 1.96 * smooth_sd)

    expected_incidence = rt_smoothed * infectiousness
    expected_incidence_lo = rt_smoothed_lo * infectiousness
    expected_incidence_hi = rt_smoothed_hi * infectiousness
    one_step_incidence = np.exp(m_pred) * infectiousness
    n_posterior_paths=int(cfg.get("posterior_path_draws",512)); rng_paths=np.random.default_rng(int(config["random_seed"])+177)
    posterior_log_rt_draws=_ffbs_sample_local_level(fitted["filter"],n_posterior_paths,rng_paths)

    full = daily.copy()
    full["state_space_infectiousness"] = np.asarray(fitted["infectiousness_full"], dtype=float)
    for col in [
        "state_space_rt_filtered_median",
        "state_space_rt_filtered_lo",
        "state_space_rt_filtered_hi",
        "state_space_rt_smoothed_median",
        "state_space_rt_smoothed_lo",
        "state_space_rt_smoothed_hi",
        "state_space_expected_incidence",
        "state_space_expected_incidence_lo",
        "state_space_expected_incidence_hi",
        "state_space_one_step_incidence",
        "state_space_prob_rt_gt_1",
    ]:
        full[col] = np.nan
    full.loc[fit_mask, "state_space_rt_filtered_median"] = rt_filtered
    full.loc[fit_mask, "state_space_rt_filtered_lo"] = rt_filtered_lo
    full.loc[fit_mask, "state_space_rt_filtered_hi"] = rt_filtered_hi
    full.loc[fit_mask, "state_space_rt_smoothed_median"] = rt_smoothed
    full.loc[fit_mask, "state_space_rt_smoothed_lo"] = rt_smoothed_lo
    full.loc[fit_mask, "state_space_rt_smoothed_hi"] = rt_smoothed_hi
    full.loc[fit_mask, "state_space_expected_incidence"] = expected_incidence
    full.loc[fit_mask, "state_space_expected_incidence_lo"] = expected_incidence_lo
    full.loc[fit_mask, "state_space_expected_incidence_hi"] = expected_incidence_hi
    full.loc[fit_mask, "state_space_one_step_incidence"] = one_step_incidence
    full.loc[fit_mask, "state_space_prob_rt_gt_1"] = 0.5 * np.vectorize(math.erfc)((0.0-m_smooth)/np.sqrt(2.0*np.maximum(p_smooth,1e-12)))

    fit = pd.DataFrame(
        {
            "date": dates_fit,
            "observed_incidence": np.asarray(fitted["incidence_fit"], dtype=float),
            "infectiousness": infectiousness,
            "log_rt_observation": np.asarray(fitted["log_ratio"], dtype=float),
            "filtered_log_rt": m_filt,
            "filtered_log_rt_variance": p_filt,
            "smoothed_log_rt": m_smooth,
            "smoothed_log_rt_variance": p_smooth,
            "rt_filtered_median": rt_filtered,
            "rt_filtered_lo": rt_filtered_lo,
            "rt_filtered_hi": rt_filtered_hi,
            "rt_smoothed_median": rt_smoothed,
            "rt_smoothed_lo": rt_smoothed_lo,
            "rt_smoothed_hi": rt_smoothed_hi,
            "expected_incidence": expected_incidence,
            "expected_incidence_lo": expected_incidence_lo,
            "expected_incidence_hi": expected_incidence_hi,
            "one_step_expected_incidence": one_step_incidence,
            "prob_rt_gt_1": 0.5 * np.vectorize(math.erfc)((0.0-m_smooth)/np.sqrt(2.0*np.maximum(p_smooth,1e-12))),
        }
    )

    parameter_summary = pd.DataFrame(
        [
            (
                "state_process_sd_log_rt",
                math.sqrt(float(fitted["process_var"])),
                "estimated",
                "Day-to-day evolution of latent log Rt",
            ),
            (
                "base_observation_sd_log_ratio",
                math.sqrt(float(fitted["base_obs_var"])),
                "estimated",
                "Additional reporting variation beyond count variance",
            ),
            (
                "generation_interval_mean_days",
                float(gi["mean_days"]),
                "fixed",
                "Gamma generation-interval distribution",
            ),
            (
                "generation_interval_sd_days",
                float(gi["sd_days"]),
                "fixed",
                "Gamma generation-interval distribution",
            ),
        ],
        columns=["parameter", "value", "estimation_role", "interpretation"],
    )

    bridge_targets = _bridge_targets(dates_fit, m_smooth, p_smooth, config)
    backfill_daily, backfill_summary = _backfill_state_space_sensitivity(
        backfill_sensitivity, config, weights
    )

    if save:
        fit.to_csv(
            PROJECT_ROOT / "results" / "state_space_latent_rt.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        parameter_summary.to_csv(
            PROJECT_ROOT / "results" / "state_space_parameter_summary.csv", index=False
        )
        full.to_csv(
            PROJECT_ROOT / "results" / "state_space_fit_daily.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        pd.DataFrame(
            {
                "lag_days": np.arange(1, len(weights) + 1),
                "weight": weights,
            }
        ).to_csv(PROJECT_ROOT / "results" / "generation_interval_weights.csv", index=False)
        bridge_targets.to_csv(
            PROJECT_ROOT / "results" / "state_space_seihfr_rt_bridge_targets.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        np.savez_compressed(PROJECT_ROOT / "results" / "state_space_rt_posterior_draws.npz", log_rt=posterior_log_rt_draws, dates=np.asarray(dates_fit.astype(str)))
        if not backfill_daily.empty:
            backfill_daily.to_csv(
                PROJECT_ROOT / "results" / "state_space_backfill_sensitivity_daily.csv",
                index=False,
                date_format="%Y-%m-%d",
            )
            backfill_summary.to_csv(
                PROJECT_ROOT / "results" / "state_space_backfill_sensitivity_summary.csv",
                index=False,
                date_format="%Y-%m-%d",
            )

    return StateSpaceResults(
        fit=fit,
        parameter_summary=parameter_summary,
        full_daily=full,
        latent_log_rt_mean=m_smooth,
        latent_log_rt_var=p_smooth,
        generation_interval_weights=weights,
        backfill_sensitivity_daily=backfill_daily,
        backfill_sensitivity_summary=backfill_summary,
        bridge_targets=bridge_targets,
        posterior_log_rt_draws=posterior_log_rt_draws,
    )
