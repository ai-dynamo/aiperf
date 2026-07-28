# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lognormal fitting and mixture delay sampling for Agentic Code session synthesis."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.random import Generator

from aiperf.dataset.agentic_code_gen.models import (
    LognormalParams,
    MixtureDelayConfig,
    WeibullParams,
)


def lognormal_from_mean_median(mean: float, median: float) -> LognormalParams:
    """Derive lognormal mu/sigma from real-space mean and median.

    mu = ln(median)
    sigma = sqrt(2 * ln(mean / median))
    """
    if mean <= 0 or median <= 0:
        raise ValueError(f"mean ({mean}) and median ({median}) must be positive")
    if mean < median:
        raise ValueError(f"mean ({mean}) must be >= median ({median}) for lognormal")

    mu = math.log(median)
    ratio = mean / median
    sigma = math.sqrt(2.0 * math.log(ratio)) if ratio > 1.0 else 0.0

    return LognormalParams(mu=mu, sigma=sigma, mean=mean, median=median)


def fit_from_samples(samples: np.ndarray) -> LognormalParams:
    """Fit lognormal parameters from raw samples using MLE.

    Takes log of positive samples, computes MLE mean/std in log-space.
    """
    positive = samples[samples > 0]
    if len(positive) < 2:
        raise ValueError(f"Need at least 2 positive samples, got {len(positive)}")

    log_samples = np.log(positive)
    mu = float(np.mean(log_samples))
    sigma = float(np.std(log_samples, ddof=0))

    real_mean = math.exp(mu + sigma**2 / 2.0)
    real_median = math.exp(mu)

    return LognormalParams(mu=mu, sigma=sigma, mean=real_mean, median=real_median)


def _rejection_sample(
    draw: Callable[[int], np.ndarray],
    *,
    lo: float | None,
    hi: float | None,
    clip_min: float | None,
    max_attempts: int,
    size: int,
) -> np.ndarray:
    """Draw *size* samples from *draw*, resampling any that fall outside [lo, hi].

    Values still out of range after max_attempts are clamped, so the bounds hold
    even when the distribution puts little mass inside them. clip_min is a hard
    floor applied last.
    """
    samples = draw(size)
    if lo is not None or hi is not None:
        for _ in range(max_attempts):
            mask = np.zeros(len(samples), dtype=bool)
            if lo is not None:
                mask |= samples < lo
            if hi is not None:
                mask |= samples > hi
            if not mask.any():
                break
            samples[mask] = draw(int(mask.sum()))
        if lo is not None:
            samples = np.maximum(samples, lo)
        if hi is not None:
            samples = np.minimum(samples, hi)
    if clip_min is not None:
        samples = np.maximum(samples, clip_min)
    return samples


def sample_lognormal(
    params: LognormalParams,
    rng: Generator,
    *,
    size: int = 1,
    clip_min: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Draw samples from a lognormal distribution.

    Uses rejection sampling for params.min and params.max (resample out-of-range
    values). clip_min is a hard floor applied after rejection sampling.
    """
    return _rejection_sample(
        lambda n: rng.lognormal(mean=params.mu, sigma=params.sigma, size=n),
        lo=params.min,
        hi=params.max,
        clip_min=clip_min,
        max_attempts=max_attempts,
        size=size,
    )


def sample_weibull(
    params: WeibullParams,
    rng: Generator,
    *,
    size: int = 1,
    clip_min: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Draw samples from a Weibull distribution.

    Uses rejection sampling for params.min and params.max (resample out-of-range
    values). clip_min is a hard floor applied after rejection sampling.
    """
    return _rejection_sample(
        lambda n: rng.weibull(params.shape, size=n) * params.scale,
        lo=params.min,
        hi=params.max,
        clip_min=clip_min,
        max_attempts=max_attempts,
        size=size,
    )


def sample_delay_component(
    params: LognormalParams | WeibullParams, rng: Generator, *, size: int = 1
) -> np.ndarray:
    """Sample a mixture delay component using its own distribution family."""
    if isinstance(params, WeibullParams):
        return sample_weibull(params, rng, size=size)
    return sample_lognormal(params, rng, size=size)


def sample_mixture_delay(
    config: MixtureDelayConfig, rng: Generator, size: int = 1
) -> np.ndarray:
    """Sample from the two-component mixture delay model.

    For each sample, a Bernoulli draw selects agentic (fast) vs human (slow),
    then the corresponding component distribution is sampled.
    """
    is_agentic = rng.random(size=size) < config.agentic_fraction
    agentic_samples = sample_delay_component(config.agentic_delay, rng, size=size)
    human_samples = sample_delay_component(config.human_delay, rng, size=size)
    samples = np.where(is_agentic, agentic_samples, human_samples)
    if config.max is not None:
        samples = np.minimum(samples, config.max)
    return samples
