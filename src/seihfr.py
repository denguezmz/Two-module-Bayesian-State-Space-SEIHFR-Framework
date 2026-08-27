from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .state_space import StateSpaceResults
from .utils import PROJECT_ROOT, normalized_weights_from_loss, weighted_quantile


BETA_BASIS_COUNT = 9
BETA_COEF_NAMES = [f"beta_spline_coef_{i+1}" for i in range(BETA_BASIS_COUNT)]
STATIC_PARAMETER_NAMES = [
    "isolation_delay_base",
    "followup_delay_reduction",
    "hospital_fatality_probability",
    "hospital_outcome_days",
    "hospital_relative_infectiousness",
    "funeral_relative_infectiousness",
    "initial_exposed",
    "initial_infectious",
]
PARAMETER_NAMES = BETA_COEF_NAMES + STATIC_PARAMETER_NAMES
IDX = {name: i for i, name in enumerate(PARAMETER_NAMES)}


@dataclass
class SEIHFRCalibrationResults:
    ensemble: pd.DataFrame
    fit_summary: pd.DataFrame
    parameter_summary: pd.DataFrame
    parameter_provenance: pd.DataFrame
    fit_metrics: pd.DataFrame
    calibration_observations: pd.DataFrame
    parameter_correlation: pd.DataFrame
    bridge_diagnostics: pd.DataFrame
    final_states: np.ndarray
    final_beta: np.ndarray
    weights: np.ndarray
    bridge_summary: pd.DataFrame
    beta_trajectory: pd.DataFrame
    smc_diagnostics: pd.DataFrame
    prior_posterior_learning: pd.DataFrame
    mechanism_decomposition: pd.DataFrame


@dataclass
class ScenarioResults:
    daily: pd.DataFrame
    summary: pd.DataFrame
    delay: pd.DataFrame
    scenario_definitions: pd.DataFrame


def _static_bounds_from_config(config: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = config["seihfr"]["parameter_bounds"]
    lower = np.array([bounds[name][0] for name in STATIC_PARAMETER_NAMES], dtype=float)
    upper = np.array([bounds[name][1] for name in STATIC_PARAMETER_NAMES], dtype=float)
    return lower, upper


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def _bspline_design(n_days: int, config: dict) -> np.ndarray:
    cfg = config["seihfr"]
    n_basis = int(cfg.get("beta_spline_basis_count", BETA_BASIS_COUNT))
    degree = int(cfg.get("beta_spline_degree", 3))
    if n_basis != BETA_BASIS_COUNT:
        raise ValueError(f"当前实现要求 beta_spline_basis_count={BETA_BASIS_COUNT}。")
    if n_basis <= degree:
        raise ValueError("beta_spline_basis_count 必须大于 beta_spline_degree。")
    x = np.linspace(0.0, 1.0, n_days)
    n_internal = n_basis - degree - 1
    internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
    knots = np.r_[np.repeat(0.0, degree + 1), internal, np.repeat(1.0, degree + 1)]
    return BSpline.design_matrix(x, knots, degree, extrapolate=True).toarray()


def _spline_projection(
    basis: np.ndarray,
    log_rt_variance: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Projection matrix for weighted, curvature-penalised spline fitting.

    For a row vector y(t), coefficients are y @ projection.  State-space
    uncertainty therefore controls how strongly each date shapes beta(t).
    """
    cfg = config["seihfr"]
    tau = float(cfg.get("rt_beta_coupling_extra_log_sd", 0.20))
    smooth_lambda = float(cfg.get("beta_spline_smoothing_lambda", 8.0))
    w = 1.0 / np.maximum(np.asarray(log_rt_variance, float) + tau**2, 1e-8)
    d2 = np.diff(np.eye(basis.shape[1]), n=2, axis=0)
    a = basis.T @ (w[:, None] * basis) + smooth_lambda * (d2.T @ d2)
    return (w[:, None] * basis) @ np.linalg.inv(a)


def _mechanistic_rt_factor_daily(
    params: np.ndarray,
    followup: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Daily R/beta factor implied by the SEIHFR infectious pathway.

    The factor contains community, hospital/isolation and post-death infectious
    time plus the same follow-up contact modifier used by the ODE.  Susceptible
    depletion is applied separately when the fitted mechanistic Rt is reported.
    """
    cfg = config["seihfr"]
    p = np.asarray(params, dtype=float)
    followup = np.asarray(followup, dtype=float)
    community_exit = 1.0 / float(cfg["community_outcome_days"])
    p_community_death = float(cfg["community_fatality_probability"])
    mu_i = p_community_death * community_exit
    gamma_i = (1.0 - p_community_death) * community_exit
    kappa = 1.0 / float(cfg["unsafe_funeral_duration_days"])

    iso_base = p[:, IDX["isolation_delay_base"]][:, None]
    followup_reduction = p[:, IDX["followup_delay_reduction"]][:, None]
    p_h = p[:, IDX["hospital_fatality_probability"]][:, None]
    hospital_exit = 1.0 / p[:, IDX["hospital_outcome_days"]][:, None]
    eta_h = p[:, IDX["hospital_relative_infectiousness"]][:, None]
    eta_f = p[:, IDX["funeral_relative_infectiousness"]][:, None]

    trace_norm = np.clip((followup[None, :] - 0.55) / 0.40, 0.0, 1.0)
    iso_delay = np.maximum(1.5, iso_base - followup_reduction * trace_norm)
    delta = 1.0 / iso_delay
    r_i = delta + gamma_i + mu_i
    mu_h = p_h * hospital_exit
    prob_h = delta / np.maximum(r_i, 1e-12)
    prob_f = mu_i / np.maximum(r_i, 1e-12) + prob_h * mu_h / np.maximum(hospital_exit, 1e-12)
    infectious_time = (
        1.0 / np.maximum(r_i, 1e-12)
        + eta_h * prob_h / np.maximum(hospital_exit, 1e-12)
        + eta_f * prob_f / max(kappa, 1e-12)
    )
    return np.maximum((1.0 - 0.25 * trace_norm) * infectious_time, 1e-6)


def _derive_beta_prior_coefficients(
    params_with_static: np.ndarray, followup: np.ndarray, state_space: StateSpaceResults,
    basis: np.ndarray, config: dict, log_rt_paths: np.ndarray | None=None,
) -> np.ndarray:
    """Project joint Module-1 Rt paths into smooth beta(t) prior centres."""
    bridge=state_space.bridge_targets.sort_values("fit_day_index").reset_index(drop=True)
    g=_mechanistic_rt_factor_daily(params_with_static,followup,config)
    if log_rt_paths is None:
        log_rt=np.broadcast_to(bridge["log_rt_mean"].to_numpy(float)[None,:],g.shape)
    else:
        log_rt=np.asarray(log_rt_paths,float)
        if log_rt.shape!=g.shape: raise ValueError("log_rt_paths shape mismatch")
    target=log_rt-np.log(g); projection=_spline_projection(basis,bridge["log_rt_variance"].to_numpy(float),config)
    return target@projection

def _beta_prior_geometry(state_space:StateSpaceResults,basis:np.ndarray,config:dict):
    bridge=state_space.bridge_targets.sort_values("fit_day_index").reset_index(drop=True); projection=_spline_projection(basis,bridge["log_rt_variance"].to_numpy(float),config); draws=np.asarray(state_space.posterior_log_rt_draws,float); coeff=draws@projection; extra=float(config["seihfr"].get("beta_spline_prior_extra_sd",.16)); cov=np.cov(coeff,rowvar=False)+np.eye(BETA_BASIS_COUNT)*extra**2; cov=(cov+cov.T)/2+np.eye(BETA_BASIS_COUNT)*1e-8; inv=np.linalg.inv(cov); logdet=float(np.linalg.slogdet(cov)[1]); return projection,cov,inv,logdet


def _beta_prior_centres(params,followup,state_space,basis,config,projection=None):
    br=state_space.bridge_targets.sort_values("fit_day_index").reset_index(drop=True); g=_mechanistic_rt_factor_daily(params,followup,config); projection=_spline_projection(basis,br["log_rt_variance"].to_numpy(float),config) if projection is None else projection; return (br["log_rt_mean"].to_numpy(float)[None,:]-np.log(g))@projection


def _sample_parameters_from_full_rt_bridge(n,rng,static_lower,static_upper,state_space,followup,basis,config,seed_params=None):
    """Sample the evaluable cut prior p(static)*p(beta coefficients|Module-1 Rt,static)."""
    p=np.zeros((n,len(PARAMETER_NAMES)),float); unit=_latin_hypercube(n,len(STATIC_PARAMETER_NAMES),rng); p[:,BETA_BASIS_COUNT:]=static_lower+(static_upper-static_lower)*unit; projection,cov,_,_=_beta_prior_geometry(state_space,basis,config); centre=_beta_prior_centres(p,followup,state_space,basis,config,projection); noise=rng.multivariate_normal(np.zeros(BETA_BASIS_COUNT),cov,size=n); coef=centre+noise; lo=math.log(float(config["seihfr"].get("beta_min",.03))); hi=math.log(float(config["seihfr"].get("beta_max",2.5))); p[:,:BETA_BASIS_COUNT]=np.clip(coef,lo,hi); return p


def _cut_logprior(params,static_lower,static_upper,state_space,followup,basis,config,geometry=None):
    p=np.asarray(params,float); geometry=_beta_prior_geometry(state_space,basis,config) if geometry is None else geometry; projection,cov,inv,logdet=geometry; inside=np.all((p[:,BETA_BASIS_COUNT:]>=static_lower)&(p[:,BETA_BASIS_COUNT:]<=static_upper),axis=1); blo=math.log(float(config["seihfr"].get("beta_min",.03))); bhi=math.log(float(config["seihfr"].get("beta_max",2.5))); inside &= np.all((p[:,:BETA_BASIS_COUNT]>=blo)&(p[:,:BETA_BASIS_COUNT]<=bhi),axis=1); centre=_beta_prior_centres(p,followup,state_space,basis,config,projection); d=p[:,:BETA_BASIS_COUNT]-centre; lp=-.5*np.einsum('ni,ij,nj->n',d,inv,d)-.5*logdet-.5*BETA_BASIS_COUNT*np.log(2*np.pi); lp[~inside]=-np.inf; return lp

def _beta_daily_from_params(params: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(params[:, :BETA_BASIS_COUNT], float) @ basis.T)


def simulate_historical_batch(
    params: np.ndarray,
    fit_followup: np.ndarray,
    initial_cases: float,
    initial_deaths: float,
    config: dict,
    basis: np.ndarray | None = None,
    return_full: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    cfg = config["seihfr"]
    n_models = params.shape[0]
    n_days = len(fit_followup)
    n_eff = float(cfg["effective_population"])
    dt = float(cfg["integration_step_days"])
    basis = _bspline_design(n_days, config) if basis is None else basis
    beta_daily = _beta_daily_from_params(params, basis)

    y = np.zeros((n_models, 8), dtype=float)  # S,E,I,H,F,R,D,C
    y[:, 1] = params[:, IDX["initial_exposed"]]
    y[:, 2] = params[:, IDX["initial_infectious"]]
    y[:, 6] = initial_deaths
    y[:, 7] = initial_cases
    y[:, 0] = np.maximum(n_eff - y[:, 1] - y[:, 2], 0.0)

    cases = np.zeros((n_models, n_days), dtype=float)
    deaths = np.zeros_like(cases)
    hospital = np.zeros_like(cases)
    cases[:, 0], deaths[:, 0], hospital[:, 0] = y[:, 7], y[:, 6], y[:, 3]
    full = np.zeros((n_models, n_days, 8), dtype=float) if return_full else None
    if return_full:
        full[:, 0, :] = y

    sigma = 1.0 / float(cfg["latent_period_days"])
    community_exit = 1.0 / float(cfg["community_outcome_days"])
    p_community_death = float(cfg["community_fatality_probability"])
    mu_i = p_community_death * community_exit
    gamma_i = (1.0 - p_community_death) * community_exit
    hospital_exit = 1.0 / params[:, IDX["hospital_outcome_days"]]
    kappa = 1.0 / float(cfg["unsafe_funeral_duration_days"])

    day_index = 1
    n_steps = int(math.ceil((n_days - 1) / dt))
    for step in range(n_steps):
        t = step * dt
        lo = min(int(math.floor(t)), n_days - 1)
        hi = min(lo + 1, n_days - 1)
        frac = t - lo
        beta = (1.0 - frac) * beta_daily[:, lo] + frac * beta_daily[:, hi]
        fu = float(np.interp(min(t, n_days - 1), np.arange(n_days), fit_followup))
        trace_norm = np.clip((fu - 0.55) / 0.40, 0.0, 1.0)
        iso_delay = np.maximum(
            1.5,
            params[:, IDX["isolation_delay_base"]]
            - params[:, IDX["followup_delay_reduction"]] * trace_norm,
        )
        delta = 1.0 / iso_delay
        p_h = params[:, IDX["hospital_fatality_probability"]]
        eta_h = params[:, IDX["hospital_relative_infectiousness"]]
        eta_f = params[:, IDX["funeral_relative_infectiousness"]]
        mu_h = p_h * hospital_exit
        gamma_h = (1.0 - p_h) * hospital_exit

        S, E, I, H, F, R, D, C = [y[:, j] for j in range(8)]
        beta_eff = beta * (1.0 - 0.25 * trace_norm)
        force = beta_eff * (I + eta_h * H + eta_f * F) / n_eff
        new_exposed = np.minimum(force * S, S / max(dt, 1e-9))
        onset = sigma * E
        dy = np.column_stack(
            [
                -new_exposed,
                new_exposed - onset,
                onset - (delta + gamma_i + mu_i) * I,
                delta * I - (gamma_h + mu_h) * H,
                mu_i * I + mu_h * H - kappa * F,
                gamma_i * I + gamma_h * H,
                mu_i * I + mu_h * H,
                onset,
            ]
        )
        y = np.maximum(y + dt * dy, 0.0)
        if (step + 1) * dt + 1e-9 >= day_index:
            if day_index < n_days:
                cases[:, day_index] = y[:, 7]
                deaths[:, day_index] = y[:, 6]
                hospital[:, day_index] = y[:, 3]
                if return_full:
                    full[:, day_index, :] = y
            day_index += 1
            if day_index >= n_days:
                break
    return cases, deaths, hospital, full, beta_daily


def _calibration_loss(
    params: np.ndarray,
    cases: np.ndarray,
    deaths: np.ndarray,
    hospital: np.ndarray,
    observed_cases: np.ndarray,
    observed_deaths: np.ndarray,
    target_incidence: np.ndarray,
    selected_indices: np.ndarray,
    hospital_indices: np.ndarray,
    hospital_values: np.ndarray,
    latent_period_days: float,
    final_exposed: np.ndarray | None = None,
    final_infectious: np.ndarray | None = None,
) -> np.ndarray:
    case_error = np.mean(
        (np.log1p(cases[:, selected_indices]) - np.log1p(observed_cases[selected_indices])[None, :]) ** 2,
        axis=1,
    )
    death_error = np.mean(
        (np.log1p(deaths[:, selected_indices]) - np.log1p(observed_deaths[selected_indices])[None, :]) ** 2,
        axis=1,
    )
    model_incidence = np.diff(cases, axis=1, prepend=cases[:, [0]])
    weekly_model, weekly_target = [], []
    end = len(target_incidence)
    for stop in range(end, max(0, end - 56), -7):
        start = max(0, stop - 7)
        weekly_model.append(model_incidence[:, start:stop].sum(axis=1))
        weekly_target.append(float(np.sum(target_incidence[start:stop])))
    weekly_model = np.column_stack(weekly_model)
    weekly_target = np.asarray(weekly_target, float)
    weekly_error = np.mean((np.log1p(weekly_model) - np.log1p(weekly_target)[None, :]) ** 2, axis=1)
    endpoint_case_error = ((cases[:, -1] - observed_cases[-1]) / max(observed_cases[-1], 1.0)) ** 2
    endpoint_death_error = ((deaths[:, -1] - observed_deaths[-1]) / max(observed_deaths[-1], 1.0)) ** 2
    recent_target = max(float(np.mean(target_incidence[-7:])), 1.0)
    endpoint_flow_error = ((model_incidence[:, -7:].mean(axis=1) - recent_target) / recent_target) ** 2
    if len(hospital_indices):
        hscale = np.maximum(hospital_values, 100.0)
        hospital_error = np.mean(((hospital[:, hospital_indices] - hospital_values[None, :]) / hscale[None, :]) ** 2, axis=1)
    else:
        hospital_error = np.zeros(len(cases))
    hidden_pool_penalty = np.zeros(len(cases))
    if final_exposed is not None and final_infectious is not None:
        expected_exposed = recent_target * latent_period_days
        expected_infectious = recent_target * 3.5
        ratio_e = final_exposed / max(expected_exposed, 1.0)
        ratio_i = final_infectious / max(expected_infectious, 1.0)
        hidden_pool_penalty = (
            np.maximum(np.log(np.maximum(ratio_e, 1e-6) / 2.5), 0.0) ** 2
            + np.maximum(np.log(0.35 / np.maximum(ratio_e, 1e-6)), 0.0) ** 2
            + np.maximum(np.log(np.maximum(ratio_i, 1e-6) / 3.0), 0.0) ** 2
            + np.maximum(np.log(0.20 / np.maximum(ratio_i, 1e-6)), 0.0) ** 2
        )
    beta_coef = params[:, :BETA_BASIS_COUNT]
    curvature = np.diff(beta_coef, n=2, axis=1)
    beta_smooth_penalty = np.mean(curvature**2, axis=1)
    return (
        case_error
        + 0.90 * death_error
        + 1.60 * weekly_error
        + 3.00 * endpoint_case_error
        + 2.00 * endpoint_death_error
        + 1.50 * endpoint_flow_error
        + 1.15 * hospital_error
        + 0.30 * hidden_pool_penalty
        + 0.02 * beta_smooth_penalty
    )


def _diagnostic_rows(
    fit_summary: pd.DataFrame,
    selected_indices: np.ndarray,
    hospital_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [
        ("cases", "observed_cases", "cases_median", "cases_q2_5", "cases_q97_5", np.arange(len(fit_summary))),
        ("deaths", "observed_deaths", "deaths_median", "deaths_q2_5", "deaths_q97_5", np.arange(len(fit_summary))),
        ("hospital", "observed_hospital", "hospital_median", "hospital_q2_5", "hospital_q97_5", hospital_indices),
    ]
    metric_rows = []
    observation_rows = []
    selected_lookup = set(int(i) for i in selected_indices)
    for target, obs_col, med_col, lo_col, hi_col, indices in targets:
        sub = fit_summary.iloc[np.asarray(indices, dtype=int)].copy()
        sub = sub[np.isfinite(sub[obs_col].to_numpy(float))]
        if sub.empty:
            metric_rows.append((target, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        observed = sub[obs_col].to_numpy(float)
        median = sub[med_col].to_numpy(float)
        lower = sub[lo_col].to_numpy(float)
        upper = sub[hi_col].to_numpy(float)
        error = median - observed
        scale = np.maximum(observed, 1.0)
        coverage = ((observed >= lower) & (observed <= upper)).mean()
        wape = np.sum(np.abs(error)) / max(np.sum(np.abs(observed)), 1.0)
        endpoint_error = error[-1] / max(observed[-1], 1.0)
        metric_rows.append((
            target,
            len(sub),
            float(np.mean(np.abs(error))),
            float(np.sqrt(np.mean(error**2))),
            float(wape),
            float(np.median(np.abs(error) / scale)),
            float(coverage),
            float(observed[-1]),
            float(median[-1]),
            float(endpoint_error),
            float(np.mean(np.log(np.maximum(median, 1e-9)) - np.log(np.maximum(observed, 1e-9)))),
        ))
        plot_indices = hospital_indices if target == "hospital" else selected_indices
        for idx in np.asarray(plot_indices, dtype=int):
            row = fit_summary.iloc[int(idx)]
            obs = float(row[obs_col])
            if not np.isfinite(obs):
                continue
            med = float(row[med_col])
            lo = float(row[lo_col])
            hi = float(row[hi_col])
            observation_rows.append((
                target,
                row["date"],
                int(idx),
                obs,
                med,
                lo,
                hi,
                med - obs,
                (med - obs) / max(obs, 1.0),
                bool(lo <= obs <= hi),
                bool(int(idx) in selected_lookup),
            ))
    metrics = pd.DataFrame(
        metric_rows,
        columns=[
            "target", "n_points", "MAE", "RMSE", "WAPE", "median_absolute_percentage_error",
            "coverage_95", "latest_observed", "latest_median", "latest_relative_error",
            "mean_log_bias",
        ],
    )
    observations = pd.DataFrame(
        observation_rows,
        columns=[
            "target", "date", "fit_day_index", "observed", "median", "q2_5", "q97_5",
            "absolute_error", "relative_error", "inside_95", "official_anchor_used",
        ],
    )
    return metrics, observations


def _weighted_parameter_correlations(ensemble: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    cols = STATIC_PARAMETER_NAMES
    x = ensemble[cols].to_numpy(float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    mu = np.sum(w[:, None] * x, axis=0)
    centered = x - mu
    cov = centered.T @ (w[:, None] * centered)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sd, sd)
    rows = []
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            rows.append((left, cols[j], float(corr[i, j]), float(abs(corr[i, j]))))
    return pd.DataFrame(rows, columns=["parameter_a", "parameter_b", "weighted_correlation", "abs_correlation"]).sort_values(
        "abs_correlation", ascending=False
    ).reset_index(drop=True)


def _hospital_stock_loglikelihood(
    hospital: np.ndarray,
    hospital_indices: np.ndarray,
    hospital_values: np.ndarray,
    lcfg: dict,
) -> dict[str, np.ndarray]:
    """Observation model for sparse hospital/isolation stock anchors.

    Stock is a dynamic state, so matching only point levels can miss short-run
    changes.  The likelihood combines anchor levels, the latest official stock,
    and changes between consecutive stock anchors.
    """
    n = hospital.shape[0]
    if len(hospital_indices) == 0:
        z = np.zeros(n)
        return {"level": z, "endpoint": z, "change": z, "total": z}

    sh = float(lcfg.get("hospital_log_sd", .32))
    endpoint_sd = float(lcfg.get("hospital_endpoint_log_sd", max(.18, .75 * sh)))
    change_sd = float(lcfg.get("hospital_change_relative_sd", .85))
    recent_weight = float(lcfg.get("hospital_recent_anchor_weight", .0))

    model = np.maximum(hospital[:, hospital_indices], 0.0)
    observed = np.asarray(hospital_values, float)
    if recent_weight:
        anchor_pos = np.linspace(0.0, 1.0, len(observed))
        weights = 1.0 + recent_weight * anchor_pos
    else:
        weights = np.ones(len(observed), float)
    z_level = (np.log1p(model) - np.log1p(observed)[None, :]) / sh
    level = -.5 * np.sum(weights[None, :] * z_level * z_level, axis=1)

    z_endpoint = (np.log1p(model[:, -1]) - np.log1p(observed[-1])) / endpoint_sd
    endpoint = -.5 * z_endpoint * z_endpoint

    if len(observed) > 1:
        model_delta = np.diff(model, axis=1)
        observed_delta = np.diff(observed)
        scale = np.maximum(np.abs(observed_delta), float(lcfg.get("hospital_change_min_scale", 75.0)))
        z_change = (model_delta - observed_delta[None, :]) / (change_sd * scale[None, :])
        change = -.5 * np.sum(z_change * z_change, axis=1)
    else:
        change = np.zeros(n)

    total = level + endpoint + change
    return {"level": level, "endpoint": endpoint, "change": change, "total": total}


def _anchor_interval_increment_loglikelihood(
    cumulative: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_values: np.ndarray,
    log_sd: float,
) -> np.ndarray:
    """Compare cumulative trajectories only through increments between real anchors."""
    n = cumulative.shape[0]
    if len(anchor_indices) <= 1:
        return np.zeros(n)

    model_inc = np.diff(cumulative[:, anchor_indices], axis=1)
    observed_inc = np.diff(np.asarray(anchor_values, float))
    z = (np.log1p(np.maximum(model_inc, 0.0)) - np.log1p(np.maximum(observed_inc, 0.0))[None, :]) / log_sd
    return -0.5 * np.sum(z * z, axis=1)


def _modular_loglikelihood(
    params,
    cases,
    deaths,
    hospital,
    case_anchor_indices,
    case_anchor_values,
    death_anchor_indices,
    death_anchor_values,
    hospital_indices,
    hospital_values,
    followup,
    state_space,
    basis,
    config,
):
    """Module-2 evidence conditional on Module 1 using only real-anchor increments."""
    lcfg = config["seihfr"].get("module2_likelihood", {})
    case_sd = float(lcfg.get("case_anchor_increment_log_sd", lcfg.get("weekly_incidence_log_sd", .40)))
    death_sd = float(lcfg.get("death_increment_log_sd", .30))
    lli = _anchor_interval_increment_loglikelihood(cases, case_anchor_indices, case_anchor_values, case_sd)
    lld = _anchor_interval_increment_loglikelihood(deaths, death_anchor_indices, death_anchor_values, death_sd)
    hospital_terms = _hospital_stock_loglikelihood(hospital, hospital_indices, hospital_values, lcfg)
    llh = hospital_terms["total"]
    return lli + lld + llh, {
        "ll_cases_anchor_increment": lli,
        "ll_deaths_anchor_increment": lld,
        "ll_hospital": llh,
        "ll_hospital_level": hospital_terms["level"],
        "ll_hospital_endpoint": hospital_terms["endpoint"],
        "ll_hospital_change": hospital_terms["change"],
    }


def _normalised_incremental_weights(log_increment):
    z=np.asarray(log_increment,float)-float(np.max(log_increment)); w=np.exp(np.clip(z,-745,0)); return w/w.sum()


def _next_temperature(gamma,loglik,target_fraction):
    if gamma>=1-1e-10: return 1.0
    n=len(loglik); target=max(2.0,target_fraction*n)
    def ess(g):
        w=_normalised_incremental_weights((g-gamma)*loglik); return 1/np.sum(w*w)
    if ess(1.0)>=target: return 1.0
    lo,hi=gamma,1.0
    for _ in range(35):
        mid=(lo+hi)/2
        if ess(mid)<target: hi=mid
        else: lo=mid
    return max(lo,gamma+1e-4)


def _systematic_resample(weights,rng):
    n=len(weights); u=(rng.random()+np.arange(n))/n; c=np.cumsum(weights); return np.searchsorted(c,u,side="right")

def _mechanism_components(params,followup,beta_daily,susceptible,config):
    cfg=config["seihfr"]; p=np.asarray(params,float); fu=np.asarray(followup,float); ce=1/float(cfg["community_outcome_days"]); pcd=float(cfg["community_fatality_probability"]); mui=pcd*ce; gammai=(1-pcd)*ce; kappa=1/float(cfg["unsafe_funeral_duration_days"])
    trace=np.clip((fu[None,:]-0.55)/.40,0,1); delta=1/np.maximum(1.5,p[:,IDX["isolation_delay_base"]][:,None]-p[:,IDX["followup_delay_reduction"]][:,None]*trace); hexit=1/p[:,IDX["hospital_outcome_days"]][:,None]; ph=p[:,IDX["hospital_fatality_probability"]][:,None]; etah=p[:,IDX["hospital_relative_infectiousness"]][:,None]; etaf=p[:,IDX["funeral_relative_infectiousness"]][:,None]; ri=delta+gammai+mui; probh=delta/ri; muh=ph*hexit; probf=mui/ri+probh*muh/hexit; cf=1-.25*trace; sf=np.asarray(susceptible,float)/float(cfg["effective_population"])
    rc=beta_daily*sf*cf/ri; rh=beta_daily*sf*cf*etah*probh/hexit; rf=beta_daily*sf*cf*etaf*probf/kappa; return rc,rh,rf,rc+rh+rf

def calibrate_seihfr(state_space:StateSpaceResults,anchors:pd.DataFrame,config:dict,save:bool=True)->SEIHFRCalibrationResults:
    """Tempered SMC approximation to the modular Bayesian cut posterior."""
    daily=state_space.full_daily.copy(); cfg=config["seihfr"]; rng=np.random.default_rng(int(config["random_seed"])+202); fit_start=pd.Timestamp(config["fit_start"]); mask=daily["date"]>=fit_start; dates=pd.to_datetime(daily.loc[mask,"date"]).reset_index(drop=True); obs_cases=daily.loc[mask,"cumulative_cases_event_time"].to_numpy(float); obs_deaths_daily=daily.loc[mask,"cumulative_deaths_event_time"].to_numpy(float); follow=daily.loc[mask,"contact_followup"].to_numpy(float); lo,hi=_static_bounds_from_config(config); basis=_bspline_design(len(dates),config)
    a=anchors.copy(); a["date"]=pd.to_datetime(a["date"]); a=a[a["date"]>=fit_start].sort_values("date")
    ca=a[np.isfinite(a["confirmed_cases"])]; cidx=np.array([(d-fit_start).days for d in ca["date"]],int); v=(cidx>=0)&(cidx<len(dates)); cidx=cidx[v]; cvals=obs_cases[cidx]
    da=a[np.isfinite(a["confirmed_deaths"])]; didx=np.array([(d-fit_start).days for d in da["date"]],int); v=(didx>=0)&(didx<len(dates)); didx=didx[v]; dvals=obs_deaths_daily[didx]
    ha=a[np.isfinite(a["hospital_isolation"])]; hidx=np.array([(d-fit_start).days for d in ha["date"]],int); v=(hidx>=0)&(hidx<len(dates)); hidx=hidx[v]; hvals=ha["hospital_isolation"].to_numpy(float)[v]
    n=int(cfg.get("module2_prior_draws",900)); geometry=_beta_prior_geometry(state_space,basis,config); particles=_sample_parameters_from_full_rt_bridge(n,rng,lo,hi,state_space,follow,basis,config); logprior=_cut_logprior(particles,lo,hi,state_space,follow,basis,config,geometry); c,d,h,_,_=simulate_historical_batch(particles,follow,obs_cases[0],obs_deaths_daily[0],config,basis=basis,return_full=False); loglik,parts=_modular_loglikelihood(particles,c,d,h,cidx,cvals,didx,dvals,hidx,hvals,follow,state_space,basis,config)
    gamma=0.; diagnostics=[]; target_ess=float(cfg.get("smc_target_ess_fraction",.70)); steps=int(cfg.get("smc_rejuvenation_steps",2)); rw=float(cfg.get("smc_rw_scale",.035)); beta_scale=float(cfg.get("smc_beta_rw_scale",.055)); max_stages=int(cfg.get("smc_max_stages",35)); stage=0
    while gamma<1-1e-9 and stage<max_stages:
        g2=_next_temperature(gamma,loglik,target_ess); w=_normalised_incremental_weights((g2-gamma)*loglik); ess=float(1/np.sum(w*w)); ids=_systematic_resample(w,rng); particles=particles[ids].copy(); loglik=loglik[ids].copy(); logprior=logprior[ids].copy(); accs=[]
        scales=np.r_[np.full(BETA_BASIS_COUNT,beta_scale),rw*(hi-lo)]
        for _ in range(steps):
            prop=particles+rng.normal(0,scales,size=particles.shape); lp=_cut_logprior(prop,lo,hi,state_space,follow,basis,config,geometry); valid=np.isfinite(lp); llp=np.full(n,-np.inf)
            if np.any(valid):
                cc,dd,hh,_,_=simulate_historical_batch(prop[valid],follow,obs_cases[0],obs_deaths_daily[0],config,basis=basis,return_full=False); llv,_=_modular_loglikelihood(prop[valid],cc,dd,hh,cidx,cvals,didx,dvals,hidx,hvals,follow,state_space,basis,config); llp[valid]=llv
            loga=lp+g2*llp-(logprior+g2*loglik); accept=np.log(rng.random(n))<np.minimum(loga,0); particles[accept]=prop[accept]; logprior[accept]=lp[accept]; loglik[accept]=llp[accept]; accs.append(float(np.mean(accept)))
        ar=float(np.mean(accs)); rw=float(np.clip(rw*np.exp(.35*(ar-.25)),.008,.12)); beta_scale=float(np.clip(beta_scale*np.exp(.35*(ar-.25)),.012,.16)); diagnostics.append((stage+1,gamma,g2,ess,ess/n,ar,rw,beta_scale,float(np.min(loglik)),float(np.median(loglik)),float(np.max(loglik)))); gamma=g2; stage+=1
    if gamma<.999: raise RuntimeError(f"SMC failed to reach gamma=1 (reached {gamma:.3f})")
    # Equal-weight SMC particles are posterior particles after final resampling/mutation.
    npost=int(cfg.get("posterior_resample_size",n)); ids=rng.choice(n,size=npost,replace=True); post=particles[ids]; weights=np.full(npost,1/npost); cases,deaths,hospital,full,beta=simulate_historical_batch(post,follow,obs_cases[0],obs_deaths_daily[0],config,basis=basis,return_full=True)
    ensemble=pd.DataFrame(post,columns=PARAMETER_NAMES); ensemble.insert(0,"posterior_weight",weights); qv=[.025,.25,.5,.75,.975]; ps=[]; learn=[]
    # Fresh prior reference for prior-to-posterior learning, independent of final SMC particles.
    prior_ref=_sample_parameters_from_full_rt_bridge(max(1000,min(4000,4*n)),np.random.default_rng(int(config["random_seed"])+299),lo,hi,state_space,follow,basis,config)
    for j,name in enumerate(PARAMETER_NAMES):
        qp=np.quantile(prior_ref[:,j],qv); q=np.quantile(post[:,j],qv); ps.append((name,*q)); pw=max(qp[4]-qp[0],1e-12); learn.append((name,*qp,*q,pw/max(q[4]-q[0],1e-12),(q[2]-qp[2])/pw))
    parameter_summary=pd.DataFrame(ps,columns=["parameter","q2_5","q25","median","q75","q97_5"]); learning=pd.DataFrame(learn,columns=["parameter","prior_q2_5","prior_q25","prior_median","prior_q75","prior_q97_5","posterior_q2_5","posterior_q25","posterior_median","posterior_q75","posterior_q97_5","interval_contraction_ratio","median_shift_prior_widths"])
    def dq(m): return np.quantile(m,qv,axis=0).T
    qc,qd,qh,qb=map(dq,[cases,deaths,hospital,beta])
    stock_ppc_sd=float(cfg.get("module2_likelihood",{}).get("hospital_reporting_log_sd_for_ppc",0.0))
    if stock_ppc_sd>0:
        qh[:,0]=np.maximum(0.0,qh[:,0]*np.exp(-1.96*stock_ppc_sd))
        qh[:,1]=np.maximum(0.0,qh[:,1]*np.exp(-0.674*stock_ppc_sd))
        qh[:,3]=qh[:,3]*np.exp(0.674*stock_ppc_sd)
        qh[:,4]=qh[:,4]*np.exp(1.96*stock_ppc_sd)
    fit=pd.DataFrame({"date":dates,"observed_cases":obs_cases,"cases_q2_5":qc[:,0],"cases_q25":qc[:,1],"cases_median":qc[:,2],"cases_q75":qc[:,3],"cases_q97_5":qc[:,4],"observed_deaths":obs_deaths_daily,"deaths_q2_5":qd[:,0],"deaths_median":qd[:,2],"deaths_q97_5":qd[:,4],"hospital_q2_5":qh[:,0],"hospital_median":qh[:,2],"hospital_q97_5":qh[:,4]}); fit["observed_hospital"]=np.nan
    for _,r in ha.iterrows(): fit.loc[fit["date"]==r["date"],"observed_hospital"]=r["hospital_isolation"]
    rc,rh,rf,rt=_mechanism_components(post,follow,beta,full[:,:,0],config); qrc,qrh,qrf,qrt=map(dq,[rc,rh,rf,rt]); br=state_space.bridge_targets.sort_values("fit_day_index").reset_index(drop=True); bridge=pd.DataFrame({"date":dates,"state_space_rt_median":br["rt_median"].to_numpy(float),"state_space_rt_lower_95":br["rt_lower_95"].to_numpy(float),"state_space_rt_upper_95":br["rt_upper_95"].to_numpy(float),"state_space_prob_rt_gt_1":br["prob_rt_gt_1"].to_numpy(float),"seihfr_implied_rt_q2_5":qrt[:,0],"seihfr_implied_rt_median":qrt[:,2],"seihfr_implied_rt_q97_5":qrt[:,4],"beta_q2_5":qb[:,0],"beta_q25":qb[:,1],"beta_median":qb[:,2],"beta_q75":qb[:,3],"beta_q97_5":qb[:,4]}); bridge["log_rt_gap_median"]=np.log(np.maximum(bridge["seihfr_implied_rt_median"],1e-9))-np.log(np.maximum(bridge["state_space_rt_median"],1e-9)); beta_tr=bridge[["date","beta_q2_5","beta_q25","beta_median","beta_q75","beta_q97_5"]].copy(); total=np.maximum(rt,1e-12); qsc,qsh,qsf=map(dq,[rc/total,rh/total,rf/total]); mech=pd.DataFrame({"date":dates,"R_community_q2_5":qrc[:,0],"R_community_median":qrc[:,2],"R_community_q97_5":qrc[:,4],"R_hospital_q2_5":qrh[:,0],"R_hospital_median":qrh[:,2],"R_hospital_q97_5":qrh[:,4],"R_postdeath_q2_5":qrf[:,0],"R_postdeath_median":qrf[:,2],"R_postdeath_q97_5":qrf[:,4],"R_total_q2_5":qrt[:,0],"R_total_median":qrt[:,2],"R_total_q97_5":qrt[:,4],"community_share_q2_5":qsc[:,0],"community_share_median":qsc[:,2],"community_share_q97_5":qsc[:,4],"hospital_share_q2_5":qsh[:,0],"hospital_share_median":qsh[:,2],"hospital_share_q97_5":qsh[:,4],"postdeath_share_q2_5":qsf[:,0],"postdeath_share_median":qsf[:,2],"postdeath_share_q97_5":qsf[:,4]})
    od=a["date"].drop_duplicates().sort_values(); selected=np.array([(d-fit_start).days for d in od],int); selected=selected[(selected>=0)&(selected<len(dates))]; fit_metrics,obs_rows=_diagnostic_rows(fit,selected,hidx); corr=_weighted_parameter_correlations(ensemble,weights); inside=(bridge.state_space_rt_median>=bridge.seihfr_implied_rt_q2_5)&(bridge.state_space_rt_median<=bridge.seihfr_implied_rt_q97_5); bd=pd.DataFrame([("log_rt_gap_median",len(bridge),float(np.mean(np.abs(bridge.log_rt_gap_median))),float(np.sqrt(np.mean(bridge.log_rt_gap_median**2))),float(np.median(np.abs(bridge.log_rt_gap_median))),float(inside.mean()),float(bridge.state_space_rt_median.iloc[-1]),float(bridge.seihfr_implied_rt_median.iloc[-1]),float(bridge.log_rt_gap_median.iloc[-1]))],columns=["diagnostic","n_days","MAE","RMSE","median_absolute_error","state_space_median_inside_seihfr_95","latest_state_space_rt_median","latest_seihfr_rt_median","latest_log_rt_gap"]); smc=pd.DataFrame(diagnostics,columns=["stage","gamma_start","gamma_end","pre_resample_ESS","ESS_fraction","mutation_acceptance","rw_scale_static","rw_scale_beta","loglik_min","loglik_median","loglik_max"]); prov=pd.DataFrame([("latent Rt trajectory","state-space FFBS posterior","Module 1","Event-time cases identify total transmission state; Module 2 cannot feed back"),("beta(t)","nine cubic B-spline coefficients","conditional Gaussian prior from full Rt posterior","Rt temporal uncertainty is projected to beta-spline covariance"),("mechanism parameters","broad bounded priors","tempered SMC cut posterior","Real-anchor case/death interval increments and reported hospital stock update the mechanism distribution"),("hospital stock observation layer",f"log SD {stock_ppc_sd:.2f} for posterior predictive intervals","reported-observation uncertainty","Median stock remains latent H(t); intervals include sparse stock reporting error"),("future operational levers","scenario profiles","scenario controlled","Forward 90-day intervention analysis only")],columns=["parameter_or_group","value_or_location","role","how_obtained"])
    if save:
        out=PROJECT_ROOT/"results"; ensemble.to_csv(out/"seihfr_cut_posterior_particles.csv",index=False); parameter_summary.to_csv(out/"seihfr_parameter_summary.csv",index=False); learning.to_csv(out/"seihfr_prior_posterior_learning.csv",index=False); prov.to_csv(out/"seihfr_parameter_provenance.csv",index=False); smc.to_csv(out/"seihfr_smc_diagnostics.csv",index=False); fit_metrics.to_csv(out/"seihfr_fit_metrics.csv",index=False); obs_rows.to_csv(out/"seihfr_calibration_observations.csv",index=False,date_format="%Y-%m-%d"); corr.to_csv(out/"seihfr_parameter_correlation.csv",index=False); bd.to_csv(out/"seihfr_bridge_diagnostics.csv",index=False); fit.to_csv(out/"seihfr_posterior_predictive_fit.csv",index=False,date_format="%Y-%m-%d"); bridge.to_csv(out/"state_space_seihfr_daily_bridge.csv",index=False,date_format="%Y-%m-%d"); beta_tr.to_csv(out/"seihfr_beta_trajectory.csv",index=False,date_format="%Y-%m-%d"); mech.to_csv(out/"seihfr_mechanism_decomposition_daily.csv",index=False,date_format="%Y-%m-%d")
    return SEIHFRCalibrationResults(ensemble=ensemble,fit_summary=fit,parameter_summary=parameter_summary,parameter_provenance=prov,fit_metrics=fit_metrics,calibration_observations=obs_rows,parameter_correlation=corr,bridge_diagnostics=bd,final_states=full[:,-1,:],final_beta=beta[:,-1],weights=weights,bridge_summary=bridge,beta_trajectory=beta_tr,smc_diagnostics=smc,prior_posterior_learning=learning,mechanism_decomposition=mech)

def _interpolate_knots(knots: list[tuple[int, float]], horizon: int) -> np.ndarray:
    # Clip long-horizon planning knots to the requested horizon and keep the
    # last value for duplicate clipped locations.
    merged = {}
    for day, value in knots:
        merged[int(np.clip(day, 0, horizon - 1))] = float(value)
    x = np.array(sorted(merged), dtype=float)
    y = np.array([merged[int(k)] for k in x], dtype=float)
    return np.interp(np.arange(horizon, dtype=float), x, y)

def _scenario_profiles(horizon: int) -> dict[str, dict[str, np.ndarray | str]]:
    return {
        "Accelerated integrated response": {
            "followup": _interpolate_knots([(0, 0.83), (14, 0.95), (45, 0.98), (horizon - 1, 0.98)], horizon),
            "isolation_target": _interpolate_knots([(0, 3.5), (21, 2.0), (horizon - 1, 1.8)], horizon),
            "safe_burial": _interpolate_knots([(0, 0.82), (30, 0.96), (horizon - 1, 0.98)], horizon),
            "ipc": _interpolate_knots([(0, 0.75), (30, 0.95), (horizon - 1, 0.97)], horizon),
            "contact_multiplier": _interpolate_knots([(0, 1.00), (21, 0.82), (60, 0.68), (horizon - 1, 0.62)], horizon),
            "description": "Rapid closure of tracing/isolation/IPC/burial gaps with lower effective community contact.",
        },
        "Current scale-up": {
            "followup": _interpolate_knots([(0, 0.83), (30, 0.88), (90, 0.92), (horizon - 1, 0.94)], horizon),
            "isolation_target": _interpolate_knots([(0, 3.5), (60, 2.8), (horizon - 1, 2.5)], horizon),
            "safe_burial": _interpolate_knots([(0, 0.82), (90, 0.90), (horizon - 1, 0.94)], horizon),
            "ipc": _interpolate_knots([(0, 0.75), (90, 0.86), (horizon - 1, 0.90)], horizon),
            "contact_multiplier": _interpolate_knots([(0, 1.00), (60, 0.91), (180, 0.85), (horizon - 1, 0.82)], horizon),
            "description": "Observed scale-up continues gradually without immediate closure of operational gaps.",
        },
        "Stalled response": {
            "followup": np.full(horizon, 0.83),
            "isolation_target": np.full(horizon, 3.5),
            "safe_burial": np.full(horizon, 0.82),
            "ipc": np.full(horizon, 0.75),
            "contact_multiplier": _interpolate_knots([(0, 1.00), (120, 1.02), (horizon - 1, 1.05)], horizon),
            "description": "Response coverage remains near the current level and transmission declines slowly or not at all.",
        },
        "Operational disruption": {
            "followup": _interpolate_knots([(0, 0.83), (30, 0.70), (90, 0.64), (horizon - 1, 0.68)], horizon),
            "isolation_target": _interpolate_knots([(0, 3.5), (30, 5.2), (90, 6.0), (horizon - 1, 5.5)], horizon),
            "safe_burial": _interpolate_knots([(0, 0.82), (45, 0.68), (horizon - 1, 0.72)], horizon),
            "ipc": _interpolate_knots([(0, 0.75), (45, 0.58), (horizon - 1, 0.64)], horizon),
            "contact_multiplier": _interpolate_knots([(0, 1.00), (30, 1.20), (90, 1.33), (horizon - 1, 1.25)], horizon),
            "description": "Security/logistics/staffing disruption reduces tracing, delays isolation and raises effective contact.",
        },
    }


def _future_simulation(params,base_beta,start_states,profile,config,rng):
    cfg=config["seihfr"]; fcfg=config["forecast"]; n=len(params); horizon=int(fcfg["horizon_days"]); dt=float(cfg["integration_step_days"]); n_eff=float(cfg["effective_population"]); y=np.asarray(start_states,float).copy(); y[:,7],y[:,6],y[:,3]=5290.,2516.,837.; y[:,0]=np.maximum(n_eff-y[:,1:6].sum(axis=1),0.)
    sigma=1/float(cfg["latent_period_days"]); ce=1/float(cfg["community_outcome_days"]); pcd=float(cfg["community_fatality_probability"]); mui=pcd*ce; gammai=(1-pcd)*ce; he=1/params[:,IDX["hospital_outcome_days"]]; kappa=1/float(cfg["unsafe_funeral_duration_days"])
    cases=np.zeros((n,horizon)); deaths=np.zeros_like(cases); hosp=np.zeros_like(cases); inc=np.zeros_like(cases); rt=np.zeros_like(cases); prev=y[:,7].copy(); ar=np.zeros(n); steps=int(round(1/dt))
    for day in range(horizon):
        ar=float(fcfg.get("process_noise_ar",.85))*ar+rng.normal(0,float(fcfg.get("process_noise_sd",.035)),n)
        for _ in range(steps):
            fu=float(np.asarray(profile["followup"])[day]); tr=np.clip((fu-.55)/.40,0,1); iso=np.minimum(np.maximum(1.5,params[:,IDX["isolation_delay_base"]]-params[:,IDX["followup_delay_reduction"]]*tr),float(np.asarray(profile["isolation_target"])[day])); delta=1/np.maximum(iso,1.); ph=params[:,IDX["hospital_fatality_probability"]]; muh=ph*he; gammah=(1-ph)*he; ipc=float(np.asarray(profile["ipc"])[day]); safe=float(np.asarray(profile["safe_burial"])[day]); etah=params[:,IDX["hospital_relative_infectiousness"]]*(1-ipc)/.25; etaf=params[:,IDX["funeral_relative_infectiousness"]]*(1-safe)/.18; contact=float(np.asarray(profile["contact_multiplier"])[day]); braw=base_beta*contact*np.exp(ar); beff=braw*(1-.25*tr)
            S,E,I,H,F,R,D,C=[y[:,j] for j in range(8)]; force=beff*(I+etah*H+etaf*F)/n_eff; ne=np.minimum(force*S,S/max(dt,1e-9)); onset=sigma*E; dy=np.column_stack([-ne,ne-onset,onset-(delta+gammai+mui)*I,delta*I-(gammah+muh)*H,mui*I+muh*H-kappa*F,gammai*I+gammah*H,mui*I+muh*H,onset]); y=np.maximum(y+dt*dy,0)
        cases[:,day],deaths[:,day],hosp[:,day]=y[:,7],y[:,6],y[:,3]; inc[:,day]=np.maximum(y[:,7]-prev,0); prev=y[:,7].copy(); ri=delta+gammai+mui; probh=delta/ri; probf=mui/ri+probh*muh/he; teff=1/ri+etah*probh/he+etaf*probf/kappa; rt[:,day]=braw*(1-.25*tr)*teff*(y[:,0]/n_eff)
    return cases,deaths,hosp,inc,rt


def _scenario_summary_rows(name,cases,deaths,hospital,incidence,rt,description):
    horizon=cases.shape[1]; rows=[]
    for day in range(horizon):
        qc=np.quantile(cases[:,day],[.025,.25,.5,.75,.975]); qd=np.quantile(deaths[:,day],[.025,.5,.975]); qh=np.quantile(hospital[:,day],[.025,.5,.975]); qi=np.quantile(incidence[:,day],[.025,.5,.975]); qr=np.quantile(rt[:,day],[.025,.5,.975]); rows.append((name,day+1,*qc,*qd,*qh,*qi,*qr,float(np.mean(rt[:,day]<1))))
    vals=[]
    for d in [30,60,90]: vals.extend(np.quantile(cases[:,min(d,horizon)-1],[.025,.5,.975]))
    vals.extend(np.quantile(deaths[:,min(89,horizon-1)],[.025,.5,.975])); vals.extend(np.quantile(np.max(hospital[:,:min(90,horizon)],axis=1),[.025,.5,.975])); vals.extend(np.quantile(rt[:,min(89,horizon-1)],[.025,.5,.975])); vals.append(float(np.mean(rt[:,min(89,horizon-1)]<1)))
    return rows,(name,*vals,description)


def run_scenarios(calibration:SEIHFRCalibrationResults,config:dict,save:bool=True)->ScenarioResults:
    fcfg=config["forecast"]; horizon=int(fcfg["horizon_days"]); n_paths=int(fcfg["paths_per_scenario"]); profiles=_scenario_profiles(horizon); params=calibration.ensemble[PARAMETER_NAMES].to_numpy(float); w=calibration.weights; master=np.random.default_rng(int(config["random_seed"])+303); daily_rows=[]; sum_rows=[]; def_rows=[]
    for ix,(name,profile) in enumerate(profiles.items()):
        rng=np.random.default_rng(int(config["random_seed"])+1000+ix); ids=master.choice(len(params),size=n_paths,replace=True,p=w); cases,deaths,hosp,inc,rt=_future_simulation(params[ids],calibration.final_beta[ids],calibration.final_states[ids],profile,config,rng); rr,ss=_scenario_summary_rows(name,cases,deaths,hosp,inc,rt,str(profile["description"])); daily_rows.extend(rr); sum_rows.append(ss); def_rows.append((name,float(np.asarray(profile["followup"])[min(6,horizon-1)]),float(np.asarray(profile["followup"])[min(89,horizon-1)]),float(np.asarray(profile["isolation_target"])[min(20,horizon-1)]),float(np.asarray(profile["safe_burial"])[min(29,horizon-1)]),float(np.asarray(profile["ipc"])[min(29,horizon-1)]),float(np.asarray(profile["contact_multiplier"])[min(59,horizon-1)]),str(profile["description"])))
    daily=pd.DataFrame(daily_rows,columns=["scenario","forecast_day","cases_q2_5","cases_q25","cases_median","cases_q75","cases_q97_5","deaths_q2_5","deaths_median","deaths_q97_5","hospital_q2_5","hospital_median","hospital_q97_5","incidence_q2_5","incidence_median","incidence_q97_5","rt_q2_5","rt_median","rt_q97_5","prob_rt_below_1"]); start=pd.Timestamp(config["data_cutoff"])+pd.Timedelta(days=1); daily["date"]=start+pd.to_timedelta(daily["forecast_day"]-1,unit="D")
    cols=["scenario"]
    for d in [30,60,90]: cols += [f"cases{d}_q2_5",f"cases{d}_median",f"cases{d}_q97_5"]
    cols += ["deaths90_q2_5","deaths90_median","deaths90_q97_5","peak_hospital90_q2_5","peak_hospital90_median","peak_hospital90_q97_5","rt90_q2_5","rt90_median","rt90_q97_5","prob_rt90_below_1","scenario_definition"]
    summary=pd.DataFrame(sum_rows,columns=cols); defs=pd.DataFrame(def_rows,columns=["scenario","followup_day7","followup_day90","isolation_target_day21","safe_burial_day30","ipc_day30","contact_multiplier_day60","interpretation"])
    accel=profiles["Accelerated integrated response"]; dr=[]
    for delay in [0,3,7,14,21,28]:
        pp={"description":""}
        for key in ["followup","isolation_target","safe_burial","ipc","contact_multiplier"]:
            arr=np.asarray(accel[key],float); dd=np.empty_like(arr); dd[:delay]=arr[0]; dd[delay:]=arr[:len(arr)-delay] if delay else arr; pp[key]=dd
        rng=np.random.default_rng(int(config["random_seed"])+5000+delay); ids=master.choice(len(params),size=min(900,n_paths),replace=True,p=w); cc,_,_,_,_=_future_simulation(params[ids],calibration.final_beta[ids],calibration.final_states[ids],pp,config,rng); q=np.quantile(cc[:,min(89,horizon-1)],[.025,.5,.975]); dr.append((delay,*q))
    delay=pd.DataFrame(dr,columns=["delay_days","cases90_q2_5","cases90_median","cases90_q97_5"]); base=float(delay.loc[delay.delay_days==0,"cases90_median"].iloc[0]); delay["additional_cases90_median"]=delay["cases90_median"]-base
    if save:
        out=PROJECT_ROOT/"results"; daily.to_csv(out/"seihfr_scenario_daily.csv",index=False,date_format="%Y-%m-%d"); summary.to_csv(out/"seihfr_scenario_summary.csv",index=False); delay.to_csv(out/"seihfr_intervention_delay.csv",index=False); defs.to_csv(out/"seihfr_scenario_definitions.csv",index=False)
    return ScenarioResults(daily=daily,summary=summary,delay=delay,scenario_definitions=defs)
