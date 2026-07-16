// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Pure-Rust port of the lognormal / mixture-delay samplers
//! (`src/aiperf/dataset/agentic_code_gen/distributions.py`).
//!
//! The synthesizer only ever calls these with `size=1`, so this file implements
//! the `size=1` path exactly: one draw, then rejection-resample any out-of-range
//! value (up to `max_attempts`) before clamping. The critical numpy draw-order
//! for `sample_mixture_delay` (`distributions.py:98-101`) is preserved: draw the
//! Bernoulli selector FIRST, then ALWAYS draw both the agentic and human
//! lognormals, then select — matching numpy's `np.where` over full arrays.

use aiperf_runtime::rng::numpy_generator::NumpyGenerator;

use crate::synthesize::config::{LognormalParams, MixtureDelayConfig};

/// Draw one sample from a lognormal distribution with rejection sampling for
/// `[min, max]` and a hard `clip_min` floor (`distributions.py:53-87`, size=1).
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

/// Draw one sample from the two-component mixture delay model
/// (`distributions.py:90-104`, size=1).
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
