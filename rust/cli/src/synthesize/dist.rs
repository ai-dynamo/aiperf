// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Lognormal and mixture-delay samplers for Agentic Code synthesis.
//!
//! Mixture sampling always draws the selector and both component values before
//! selecting, preserving the seeded random stream.

use aiperf_runtime::rng::numpy_generator::NumpyGenerator;

use crate::synthesize::config::{LognormalParams, MixtureDelayConfig};

/// Draw one bounded lognormal sample with a hard `clip_min` floor.
pub fn sample_lognormal(
    params: &LognormalParams,
    rng: &mut NumpyGenerator,
    clip_min: Option<f64>,
    max_attempts: u32,
) -> f64 {
    let lo = params.min;
    let hi = params.max;
    let mut sample = rng.lognormal(params.mu, params.sigma);
    if lo.is_some() || hi.is_some() {
        for _ in 0..max_attempts {
            let out_of_range = lo.is_some_and(|l| sample < l) || hi.is_some_and(|h| sample > h);
            if !out_of_range {
                break;
            }
            sample = rng.lognormal(params.mu, params.sigma);
        }
        if let Some(l) = lo {
            sample = sample.max(l);
        }
        if let Some(h) = hi {
            sample = sample.min(h);
        }
    }
    if let Some(c) = clip_min {
        sample = sample.max(c);
    }
    sample
}

/// Draw one sample from the two-component mixture delay model.
///
/// Draw order (must not change): `is_agentic = random() < agentic_fraction`,
/// then `agentic = sample_lognormal(...)`, then `human = sample_lognormal(...)`,
/// then select, then optionally clip to `config.max`.
pub fn sample_mixture_delay(config: &MixtureDelayConfig, rng: &mut NumpyGenerator) -> f64 {
    let is_agentic = rng.random() < config.agentic_fraction;
    let agentic = sample_lognormal(&config.agentic_delay, rng, None, 100);
    let human = sample_lognormal(&config.human_delay, rng, None, 100);
    let mut sample = if is_agentic { agentic } else { human };
    if let Some(m) = config.max {
        sample = sample.min(m);
    }
    sample
}
