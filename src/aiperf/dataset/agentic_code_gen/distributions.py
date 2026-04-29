# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lognormal fitting and mixture delay sampling for Agentic Code session synthesis."""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from aiperf.dataset.agentic_code_gen.models import LognormalParams, MixtureDelayConfig


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
    lo = params.min
    hi = params.max
    samples = rng.lognormal(mean=params.mu, sigma=params.sigma, size=size)
    if lo is not None or hi is not None:
        for _ in range(max_attempts):
            mask = np.zeros(len(samples), dtype=bool)
            if lo is not None:
                mask |= samples < lo
            if hi is not None:
                mask |= samples > hi
            if not mask.any():
                break
            samples[mask] = rng.lognormal(
                mean=params.mu, sigma=params.sigma, size=int(mask.sum())
            )
        if lo is not None:
            samples = np.maximum(samples, lo)
        if hi is not None:
            samples = np.minimum(samples, hi)
    if clip_min is not None:
        samples = np.maximum(samples, clip_min)
    return samples


def sample_mixture_delay(
    config: MixtureDelayConfig, rng: Generator, size: int = 1
) -> np.ndarray:
    """Sample from the two-component mixture delay model.

    For each sample, a Bernoulli draw selects agentic (fast) vs human (slow),
    then the corresponding lognormal is sampled.
    """
    is_agentic = rng.random(size=size) < config.agentic_fraction
    agentic_samples = sample_lognormal(config.agentic_delay, rng, size=size)
    human_samples = sample_lognormal(config.human_delay, rng, size=size)
    samples = np.where(is_agentic, agentic_samples, human_samples)
    if config.max is not None:
        samples = np.minimum(samples, config.max)
    return samples
