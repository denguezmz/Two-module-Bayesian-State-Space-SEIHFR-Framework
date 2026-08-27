from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    mode = os.environ.get("RUN_MODE", config.get("run_mode", "standard")).lower().strip()
    if mode not in {"smoke", "standard", "publication"}:
        raise ValueError("run_mode must be smoke, standard or publication")
    config["run_mode"] = mode
    profiles = {
        "smoke": {"rt_paths": 96, "module2": 160, "posterior": 160, "scenario": 70, "sobol": 30, "prcc": 70, "rejuvenation": 1},
        "standard": {"rt_paths": 512, "module2": 900, "posterior": 1200, "scenario": 1400, "sobol": 1200, "prcc": 2500, "rejuvenation": 2},
        "publication": {"rt_paths": 1600, "module2": 3200, "posterior": 4200, "scenario": 5000, "sobol": 9000, "prcc": 15000, "rejuvenation": 4},
    }
    r = profiles[mode]
    config["state_space"]["posterior_path_draws"] = r["rt_paths"]
    config["seihfr"]["module2_prior_draws"] = r["module2"]
    config["seihfr"]["posterior_resample_size"] = r["posterior"]
    config["forecast"]["paths_per_scenario"] = r["scenario"]
    config["sensitivity"]["sobol_base_size"] = r["sobol"]
    config["sensitivity"]["prcc_size"] = r["prcc"]
    config["seihfr"]["smc_rejuvenation_steps"] = r["rejuvenation"]
    return config


def ensure_project_dirs() -> None:
    for name in ["data", "results", "figures", "manuscript", "docs", "submission"]:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def weighted_quantile(
    values: np.ndarray,
    quantiles: Sequence[float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted quantiles for a one-dimensional array."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)[mask]
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        return np.quantile(values, quantiles)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def normalized_weights_from_loss(loss: np.ndarray, retained_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert calibration loss to stable relative weights.

    The scale is the retained loss interquartile range, so the weights express
    relative compatibility inside the accepted ensemble rather than a formal
    posterior probability.
    """
    loss = np.asarray(loss, dtype=float)
    if retained_mask is not None:
        ref = loss[retained_mask]
    else:
        ref = loss
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.ones_like(loss) / max(len(loss), 1)
    q25, q75 = np.quantile(ref, [0.25, 0.75])
    scale = max(q75 - q25, np.std(ref) * 0.25, 1e-8)
    shifted = loss - np.nanmin(loss)
    logw = -0.5 * shifted / scale
    logw -= np.nanmax(logw)
    w = np.exp(np.clip(logw, -700, 0))
    w[~np.isfinite(w)] = 0.0
    total = w.sum()
    return w / total if total > 0 else np.ones_like(loss) / len(loss)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    out = np.convolve(values, kernel, mode="same")
    # Edge correction: use the available number of values rather than zeros.
    denom = np.convolve(np.ones_like(values), np.ones(window), mode="same")
    return out * window / np.maximum(denom, 1)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
