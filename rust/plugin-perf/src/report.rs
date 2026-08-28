// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The serialized parity result document.
//!
//! One experiment produces exactly one of these. It is deliberately
//! self-contained: the frozen identity, every measurement that fed the bound
//! (warmups included, so a reader can see what was discarded), the bound
//! itself, the stability statistics that admitted the sample, and the seed the
//! bootstrap was drawn with. A reader who disagrees with the verdict can
//! recompute it from this document alone.

use serde::{Deserialize, Serialize};

use crate::experiment::ExperimentIdentity;
use crate::stats::{
    MAX_COEFFICIENT_OF_VARIATION, PARITY_LOWER_BOUND_THRESHOLD, PairedSamples,
    coefficient_of_variation,
};

/// One AB or BA measurement pair, in the units it was measured in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedSample {
    /// Position in the schedule. Warmup and retained pairs are numbered
    /// separately, each from zero.
    pub pair_index: u32,
    /// Whether the static build ran first in this pair.
    pub is_ab: bool,
    /// Nanoseconds observed for the statically linked build.
    pub static_value_ns: u64,
    /// Nanoseconds observed for the dynamically loading build.
    pub dynamic_value_ns: u64,
    /// Metric this pair measured, such as `ttft_p50`.
    pub metric: String,
    /// Whether this pair was discarded as warmup.
    pub is_warmup: bool,
}

/// The complete outcome of one parity experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityResult {
    /// Frozen, content-addressed identity of the experiment.
    pub identity: ExperimentIdentity,
    /// Discarded warmup pairs, retained in the document for auditability.
    pub warmup_pairs: Vec<PairedSample>,
    /// Pairs the bound was computed from.
    pub retained_pairs: Vec<PairedSample>,
    /// One-sided lower confidence bound on the retention ratio.
    pub bootstrap_lower_bound: f64,
    /// Mean retention ratio before resampling.
    pub point_estimate: f64,
    /// Resamples the bound was drawn from.
    pub bootstrap_resamples: usize,
    /// Seed the bootstrap was drawn with.
    pub bootstrap_seed: u64,
    /// Coefficient of variation of the retained static-side measurements.
    pub cv_static: f64,
    /// Coefficient of variation of the retained dynamic-side measurements.
    pub cv_dynamic: f64,
    /// Dynamic-side allocations minus static-side allocations. The dynamic
    /// build must not allocate more than the static build, so this must not be
    /// positive.
    pub allocation_delta: i64,
    /// Whether the experiment cleared every gate condition.
    pub is_zero_loss: bool,
}

impl ParityResult {
    /// Whether the retained sample was stable enough to have measured anything.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.cv_static <= MAX_COEFFICIENT_OF_VARIATION
            && self.cv_dynamic <= MAX_COEFFICIENT_OF_VARIATION
    }

    /// Evaluates every gate condition: a stable sample, a retention bound at or
    /// above the threshold, and no allocation growth.
    #[must_use]
    pub fn evaluate_zero_loss(&self) -> bool {
        self.is_stable()
            && self.bootstrap_lower_bound >= PARITY_LOWER_BOUND_THRESHOLD
            && self.allocation_delta <= 0
    }
}

/// Splits retained pairs into the [`PairedSamples`] the bootstrap consumes.
///
/// Execution order is preserved: an AB pair contributes `(static, dynamic)` and
/// a BA pair contributes `(dynamic, static)`, matching how each was measured.
#[must_use]
pub fn to_paired_samples(retained: &[PairedSample]) -> PairedSamples {
    let mut samples = PairedSamples::default();
    for pair in retained {
        let static_ns = pair.static_value_ns as f64;
        let dynamic_ns = pair.dynamic_value_ns as f64;
        if pair.is_ab {
            samples.ab.push((static_ns, dynamic_ns));
        } else {
            samples.ba.push((dynamic_ns, static_ns));
        }
    }
    samples
}

/// Coefficients of variation of the static and dynamic sides of the retained
/// pairs, in that order.
#[must_use]
pub fn side_coefficients_of_variation(retained: &[PairedSample]) -> (f64, f64) {
    let static_values: Vec<f64> = retained
        .iter()
        .map(|pair| pair.static_value_ns as f64)
        .collect();
    let dynamic_values: Vec<f64> = retained
        .iter()
        .map(|pair| pair.dynamic_value_ns as f64)
        .collect();
    (
        coefficient_of_variation(&static_values),
        coefficient_of_variation(&dynamic_values),
    )
}
