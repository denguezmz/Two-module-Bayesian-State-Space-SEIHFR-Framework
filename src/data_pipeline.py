from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .utils import PROJECT_ROOT


@dataclass
class HarmonisedData:
    anchors: pd.DataFrame
    daily: pd.DataFrame
    province: pd.DataFrame
    case_backfill_sensitivity: pd.DataFrame


def _interpolate_cumulative(
    anchors: pd.DataFrame,
    dates: pd.DatetimeIndex,
    column: str,
    initial: float,
    outbreak_start: pd.Timestamp,
) -> np.ndarray:
    x_daily = (dates - outbreak_start).days.to_numpy()
    x_anchor = (anchors["date"] - outbreak_start).dt.days.to_numpy()
    y = anchors[column].to_numpy(float)
    mask = np.isfinite(y)
    xa = x_anchor[mask]
    ya = y[mask]
    if len(xa) < 2:
        return np.full(len(dates), ya[0] if len(ya) else initial, dtype=float)

    interpolator = PchipInterpolator(xa, np.log1p(ya), extrapolate=False)
    out = np.empty(len(dates), dtype=float)
    inside = (x_daily >= xa[0]) & (x_daily <= xa[-1])
    out[inside] = np.expm1(interpolator(x_daily[inside]))
    before = x_daily < xa[0]
    out[before] = np.expm1(
        np.interp(
            x_daily[before],
            [x_daily[0], xa[0]],
            [np.log1p(initial), np.log1p(ya[0])],
        )
    )
    after = x_daily > xa[-1]
    out[after] = ya[-1]
    out = np.maximum.accumulate(np.maximum(out, 0.0))
    for _, row in anchors.loc[mask].iterrows():
        idx = int((row["date"] - outbreak_start).days)
        if 0 <= idx < len(out):
            out[idx] = float(row[column])
    return np.maximum.accumulate(out)


def _interpolate_indicator(
    anchors: pd.DataFrame,
    dates: pd.DatetimeIndex,
    column: str,
    default: float,
    outbreak_start: pd.Timestamp,
) -> np.ndarray:
    x_daily = (dates - outbreak_start).days.to_numpy()
    x_anchor = (anchors["date"] - outbreak_start).dt.days.to_numpy()
    y = anchors[column].to_numpy(float)
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return np.full(len(dates), default, dtype=float)
    if mask.sum() == 1:
        return np.full(len(dates), y[mask][0], dtype=float)
    return np.interp(
        x_daily,
        x_anchor[mask],
        y[mask],
        left=y[mask][0],
        right=y[mask][-1],
    )


def _remove_notification_backfill(
    incidence: np.ndarray,
    dates: pd.DatetimeIndex,
    total: float,
    notification_start: str,
    notification_end: str,
    mode: str = "anomaly_excess",
    baseline_days: int = 7,
    anomaly_floor: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove an official backfill total using an auditable notification rule.

    ``anomaly_excess`` treats the excess above the robust pre-notification
    baseline as the primary backfill candidate, so a reconciliation spike can
    absorb more of the historical total than ordinary notification days.  The
    proportional rule is retained for sensitivity analyses.
    """
    base = np.asarray(incidence, dtype=float).copy()
    removed = np.zeros_like(base)
    notification = np.where(
        (dates >= pd.Timestamp(notification_start))
        & (dates <= pd.Timestamp(notification_end))
    )[0]
    if notification.size == 0:
        raise ValueError("The configured notification window does not overlap the daily series")

    total = float(total)
    removable = float(np.maximum(base[notification], 0.0).sum())
    if total < 0.0 or total > removable + 1e-9:
        raise ValueError(
            f"Official backfill total {total:g} cannot be removed from "
            f"notification-window incidence {removable:g}"
        )
    available = np.maximum(base[notification], 0.0)
    mode = str(mode).lower()
    if mode == "proportional":
        scores = available.copy()
    elif mode == "anomaly_excess":
        pre = np.where(dates < pd.Timestamp(notification_start))[0]
        if pre.size == 0:
            raise ValueError("Anomaly allocation requires pre-notification observations")
        ref = float(np.median(np.maximum(base[pre[-int(max(1, baseline_days)):]], 0.0)))
        # A small floor preserves a non-zero chance of ordinary reporting delay
        # on every notification day while concentrating mass on excess counts.
        scores = np.maximum(available - ref, 0.0) + float(anomaly_floor) * max(ref, 1.0)
    else:
        raise ValueError(f"Unknown notification backfill allocation mode: {mode}")

    scores = np.maximum(scores, 0.0)
    if float(scores.sum()) <= 0.0:
        scores = available.copy()
    # Allocate with caps so no day is assigned more backfill than it reported;
    # redistribute any remainder to the still-available notification days.
    remaining = float(total)
    active = available > 0.0
    while remaining > 1e-10 and np.any(active):
        local_scores = np.where(active, scores, 0.0)
        if local_scores.sum() <= 0.0:
            local_scores = np.where(active, available, 0.0)
        proposed = remaining * local_scores / local_scores.sum()
        cap = np.minimum(proposed, available - removed[notification])
        removed[notification] += cap
        remaining -= float(cap.sum())
        active = active & ((available - removed[notification]) > 1e-10)
    if remaining > 1e-8:
        raise ValueError("Backfill allocation could not be capped within notification incidence")
    base[notification] -= removed[notification]
    base[np.abs(base) < 1e-12] = 0.0
    if np.any(base < -1e-9):
        raise AssertionError("Exact notification-window removal produced negative incidence")
    return np.maximum(base, 0.0), removed


def _trailing_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=int(window), min_periods=1)
        .mean()
        .to_numpy(float)
    )


def _allocate_historical_backfill(
    base_incidence: np.ndarray,
    dates: pd.DatetimeIndex,
    total: float,
    historical_end: str,
    smoothing_window_days: int,
    epsilon: float,
    lookback_days: int | None,
    uniform: bool,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Allocate a fixed total over an eligible historical interval."""
    end = pd.Timestamp(historical_end)
    start = dates[0] if lookback_days is None else end - pd.Timedelta(days=int(lookback_days) - 1)
    eligible = np.where((dates >= start) & (dates <= end))[0]
    if eligible.size == 0:
        raise ValueError("The configured historical allocation window is empty")

    if uniform:
        weights = np.ones(eligible.size, dtype=float)
    else:
        smoothed = _trailing_rolling_mean(base_incidence, smoothing_window_days)
        weights = np.maximum(smoothed[eligible], 0.0) + float(epsilon)
    weights /= weights.sum()

    added = np.zeros_like(base_incidence, dtype=float)
    added[eligible] = float(total) * weights
    adjusted = np.asarray(base_incidence, dtype=float) + added
    return adjusted, added, pd.Timestamp(start)


def _build_case_backfill_scenarios(
    incidence: np.ndarray,
    dates: pd.DatetimeIndex,
    backfill_config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    total = float(backfill_config["cases"])
    rows: list[pd.DataFrame] = []
    main_name = ""
    for scenario in backfill_config["case_scenarios"]:
        name = str(scenario["name"])
        if bool(scenario.get("primary", False)):
            if main_name:
                raise ValueError("Exactly one case backfill scenario must be marked primary")
            main_name = name
        base, removed = _remove_notification_backfill(
            incidence, dates, total,
            backfill_config["notification_start"],
            backfill_config["notification_end"],
            mode=scenario.get("notification_mode", backfill_config.get("notification_allocation_mode", "anomaly_excess")),
            baseline_days=int(backfill_config.get("notification_baseline_days", 7)),
            anomaly_floor=float(backfill_config.get("notification_anomaly_floor", 0.05)),
        )
        adjusted, added, allocation_start = _allocate_historical_backfill(
            base,
            dates,
            total,
            backfill_config["historical_end"],
            int(backfill_config["smoothing_window_days"]),
            float(backfill_config["epsilon"]),
            scenario.get("lookback_days"),
            bool(scenario.get("uniform", False)),
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "scenario": name,
                    "scenario_cn": str(scenario["label_cn"]),
                    "is_primary": bool(scenario.get("primary", False)),
                    "allocation_start": allocation_start,
                    "allocation_end": pd.Timestamp(backfill_config["historical_end"]),
                    "base_incidence_after_notification_removal": base,
                    "notification_backfill_removed": removed,
                    "historical_backfill_added": added,
                    "reconstructed_incidence_cases": adjusted,
                }
            )
        )
    if not main_name:
        raise ValueError("Exactly one case backfill scenario must be marked primary")
    sensitivity = pd.concat(rows, ignore_index=True)
    for _, group in sensitivity.groupby("scenario", sort=False):
        if not np.isclose(group["notification_backfill_removed"].sum(), total, atol=1e-8):
            raise AssertionError("A scenario did not remove the exact official case total")
        if not np.isclose(group["historical_backfill_added"].sum(), total, atol=1e-8):
            raise AssertionError("A scenario did not reallocate the exact official case total")
        if not np.isclose(group["reconstructed_incidence_cases"].sum(), incidence.sum(), atol=1e-8):
            raise AssertionError("A scenario did not preserve the cumulative case endpoint")
    return sensitivity, main_name


def build_harmonised_data(config: dict, save: bool = True) -> HarmonisedData:
    anchors = pd.read_csv(PROJECT_ROOT / "data" / "official_aggregate_anchors.csv")
    anchors["date"] = pd.to_datetime(anchors["date"])
    province = pd.read_csv(PROJECT_ROOT / "data" / "province_burden_2026-08-19.csv")

    outbreak_start = pd.Timestamp(config["outbreak_start"])
    data_cutoff = pd.Timestamp(config["data_cutoff"])
    dates = pd.date_range(outbreak_start, data_cutoff, freq="D")

    cumulative_cases_reported = _interpolate_cumulative(
        anchors, dates, "confirmed_cases", 1.0, outbreak_start
    )
    cumulative_deaths_reported = _interpolate_cumulative(
        anchors, dates, "confirmed_deaths", 0.0, outbreak_start
    )
    incidence_cases_reported = np.diff(np.r_[0.0, cumulative_cases_reported])
    incidence_deaths_reported = np.diff(np.r_[0.0, cumulative_deaths_reported])

    bf = config["backfill"]
    case_backfill_sensitivity, main_scenario = _build_case_backfill_scenarios(
        incidence_cases_reported, dates, bf
    )
    main_case = case_backfill_sensitivity.loc[
        case_backfill_sensitivity["scenario"] == main_scenario
    ].copy()
    incidence_cases_event = main_case["reconstructed_incidence_cases"].to_numpy(float)
    case_backfill_component = (
        main_case["notification_backfill_removed"].to_numpy(float)
        - main_case["historical_backfill_added"].to_numpy(float)
    )

    deaths_base, deaths_removed = _remove_notification_backfill(
        incidence_deaths_reported,
        dates,
        bf["deaths"],
        bf["notification_start"],
        bf["notification_end"],
        mode=bf.get("notification_allocation_mode", "anomaly_excess"),
        baseline_days=int(bf.get("notification_baseline_days", 7)),
        anomaly_floor=float(bf.get("notification_anomaly_floor", 0.05)),
    )
    incidence_deaths_event, deaths_added, _ = _allocate_historical_backfill(
        deaths_base,
        dates,
        bf["deaths"],
        bf["historical_end"],
        int(bf["smoothing_window_days"]),
        float(bf["epsilon"]),
        lookback_days=None,
        uniform=False,
    )
    death_backfill_component = deaths_removed - deaths_added

    cumulative_cases_event = np.cumsum(incidence_cases_event)
    cumulative_deaths_event = np.cumsum(incidence_deaths_event)

    followup = _interpolate_indicator(
        anchors, dates, "contact_followup", 0.75, outbreak_start
    )
    affected_hz = _interpolate_indicator(
        anchors, dates, "affected_health_zones", 20.0, outbreak_start
    )
    hospital = _interpolate_indicator(
        anchors, dates, "hospital_isolation", np.nan, outbreak_start
    )
    recovered = _interpolate_indicator(
        anchors, dates, "recovered", np.nan, outbreak_start
    )

    daily = pd.DataFrame(
        {
            "date": dates,
            "cumulative_cases_reported": cumulative_cases_reported,
            "cumulative_deaths_reported": cumulative_deaths_reported,
            "reported_incidence_cases": incidence_cases_reported,
            "reported_incidence_deaths": incidence_deaths_reported,
            "cumulative_cases_event_time": cumulative_cases_event,
            "cumulative_deaths_event_time": cumulative_deaths_event,
            "event_time_incidence_cases": incidence_cases_event,
            "event_time_incidence_deaths": incidence_deaths_event,
            "case_backfill_component": case_backfill_component,
            "death_backfill_component": death_backfill_component,
            "contact_followup": followup,
            "affected_health_zones": affected_hz,
            "hospital_isolation_interpolated": hospital,
            "recovered_interpolated": recovered,
        }
    )
    daily["crude_cfr"] = daily["cumulative_deaths_reported"] / np.maximum(
        daily["cumulative_cases_reported"], 1.0
    )

    if save:
        daily.to_csv(
            PROJECT_ROOT / "data" / "daily_harmonised_series.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        case_backfill_sensitivity.to_csv(
            PROJECT_ROOT / "data" / "backfill_case_sensitivity_series.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
    return HarmonisedData(
        anchors=anchors,
        daily=daily,
        province=province,
        case_backfill_sensitivity=case_backfill_sensitivity,
    )
