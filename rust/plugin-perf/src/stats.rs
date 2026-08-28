// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Statistical engine for the native-plugin performance-parity gate.
//!
//! The gate asks one question: did moving a capability from a statically linked
//! implementation leaf into a dynamically loaded plugin cost measurable
//! performance? Answering it credibly needs three pieces, all of which live
//! here and none of which depend on a live measurement:
//!
//! - A quantile estimator that agrees with R, NumPy, SciPy, and Excel, so a
//!   reported percentile means the same thing to every reader. That is
//!   Hyndman-Fan type 7.
//! - A dispersion statistic that rejects an unstable machine before its noise
//!   is mistaken for a regression. That is the coefficient of variation.
//! - A paired bootstrap that converts retained AB/BA pairs into a one-sided
//!   lower confidence bound on the *retention ratio* `static / dynamic`, so a
//!   slower dynamic build lands below 1.0.
//!
//! Randomness is a seeded linear congruential generator rather than an external
//! crate so a published bound is reproducible from the result document alone,
//! with no dependency-version caveat attached to it.

use std::fmt;

/// Resamples a production parity run must draw before its bound is publishable.
///
/// Tests may bootstrap with fewer to stay fast; the gate binary may not.
pub const MINIMUM_BOOTSTRAP_RESAMPLES: usize = 100_000;

/// Retained pairs, per orientation, a production run must collect.
pub const MINIMUM_RETAINED_PAIRS: usize = 30;

/// Warmup pairs discarded before any pair is retained.
pub const WARMUP_ITERATIONS: usize = 5;

/// Largest coefficient of variation a retained sample may exhibit before the
/// machine is treated as too noisy to have measured anything.
pub const MAX_COEFFICIENT_OF_VARIATION: f64 = 0.02;

/// Smallest retention-ratio lower bound that still counts as zero loss.
pub const PARITY_LOWER_BOUND_THRESHOLD: f64 = 0.99;

/// Quantile definitions this crate can evaluate.
///
/// Only type 7 is implemented: it is the default in R, NumPy, SciPy, and Excel,
/// so it is the one a reader will reproduce without being told which convention
/// was used. The enum exists so a future addition is an additive change rather
/// than a silent reinterpretation of already-published numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantileType {
    /// Linear interpolation of the order statistics at `h = (n - 1) * p`.
    Type7,
}

/// Errors the statistical engine reports instead of publishing a bound.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Too few retained pairs in one orientation to bootstrap.
    InsufficientPairs {
        /// Pairs supplied for the failing orientation.
        found: usize,
        /// Pairs the gate requires.
        required: usize,
    },
    /// The AB and BA orientations disagree on pair count, so they cannot be
    /// treated as one balanced design.
    UnbalancedOrientations {
        /// Static-first pairs supplied.
        ab: usize,
        /// Dynamic-first pairs supplied.
        ba: usize,
    },
    /// A resample count that cannot produce a percentile.
    InsufficientResamples {
        /// Resamples requested.
        found: usize,
    },
    /// A confidence level outside the open interval `(0, 1)`.
    InvalidConfidence {
        /// Confidence requested.
        found: f64,
    },
    /// A measurement that cannot participate in a ratio: non-finite, negative,
    /// or a zero denominator.
    InvalidMeasurement {
        /// Static-side nanoseconds observed.
        static_ns: f64,
        /// Dynamic-side nanoseconds observed.
        dynamic_ns: f64,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientPairs { found, required } => write!(
                f,
                "expected at least {required} retained pairs, found {found}"
            ),
            Self::UnbalancedOrientations { ab, ba } => {
                write!(f, "unbalanced design: {ab} AB pairs against {ba} BA pairs")
            }
            Self::InsufficientResamples { found } => {
                write!(f, "expected at least 2 bootstrap resamples, found {found}")
            }
            Self::InvalidConfidence { found } => {
                write!(f, "confidence must lie in (0, 1), found {found}")
            }
            Self::InvalidMeasurement {
                static_ns,
                dynamic_ns,
            } => write!(
                f,
                "measurement pair is not a usable ratio: static {static_ns} ns, dynamic {dynamic_ns} ns"
            ),
        }
    }
}

impl std::error::Error for StatsError {}

/// Retained measurements for one metric, split by execution order.
///
/// Each tuple is stored in the order the two builds actually ran, so `ab` holds
/// `(static_ns, dynamic_ns)` and `ba` holds `(dynamic_ns, static_ns)`. Keeping
/// the stored form in execution order means a recorded pair cannot be silently
/// reinterpreted as the other orientation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairedSamples {
    /// Static-first pairs, stored as `(static_ns, dynamic_ns)`.
    pub ab: Vec<(f64, f64)>,
    /// Dynamic-first pairs, stored as `(dynamic_ns, static_ns)`.
    pub ba: Vec<(f64, f64)>,
}

impl PairedSamples {
    /// Retention ratios `static / dynamic`, AB pairs first then BA pairs.
    ///
    /// A dynamic build slower than the static build produces a ratio below 1.0,
    /// which is the direction [`PARITY_LOWER_BOUND_THRESHOLD`] is written in.
    /// Drawing from both orientations cancels first-order carryover between the
    /// two halves of a pair.
    pub fn retention_ratios(&self) -> Result<Vec<f64>, StatsError> {
        let mut ratios = Vec::with_capacity(self.ab.len() + self.ba.len());
        for &(static_ns, dynamic_ns) in &self.ab {
            ratios.push(retention_ratio(static_ns, dynamic_ns)?);
        }
        for &(dynamic_ns, static_ns) in &self.ba {
            ratios.push(retention_ratio(static_ns, dynamic_ns)?);
        }
        Ok(ratios)
    }
}

/// One retention ratio, refusing anything that would produce a non-finite or
/// meaningless value rather than propagating an infinity into the bootstrap.
fn retention_ratio(static_ns: f64, dynamic_ns: f64) -> Result<f64, StatsError> {
    let invalid = StatsError::InvalidMeasurement {
        static_ns,
        dynamic_ns,
    };
    if !static_ns.is_finite() || !dynamic_ns.is_finite() {
        return Err(invalid);
    }
    if static_ns < 0.0 || dynamic_ns <= 0.0 {
        return Err(invalid);
    }
    Ok(static_ns / dynamic_ns)
}

/// How a bootstrap is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapConfig {
    /// Resamples to draw. Production runs use [`MINIMUM_BOOTSTRAP_RESAMPLES`].
    pub resamples: usize,
    /// Two-sided confidence level; the one-sided lower bound is taken at the
    /// `1 - confidence` quantile.
    pub confidence: f64,
    /// Seed for the deterministic generator, so the bound is reproducible.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            resamples: MINIMUM_BOOTSTRAP_RESAMPLES,
            confidence: 0.95,
            seed: 0x5EED_0000_0000_0001,
        }
    }
}

/// The published outcome of one bootstrap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapResult {
    /// One-sided lower confidence bound on the mean retention ratio. At or
    /// above [`PARITY_LOWER_BOUND_THRESHOLD`] is zero loss.
    pub lower_bound: f64,
    /// Mean retention ratio over the observed pairs, before resampling.
    pub point_estimate: f64,
    /// Resamples actually drawn.
    pub resamples: usize,
    /// Confidence level the bound was taken at.
    pub confidence: f64,
    /// Retained pairs the bound was drawn from, across both orientations.
    pub pairs: usize,
}

/// Hyndman-Fan type-7 quantile of `samples` at probability `p`.
///
/// Returns NaN for an empty sample, since no order statistic exists. `p` is
/// clamped into `[0, 1]`. NaN values in `samples` sort to the end and will
/// contaminate the result: a NaN measurement is rejected upstream by
/// [`PairedSamples::retention_ratios`] rather than quietly absorbed here.
#[must_use]
pub fn hyndman_fan_quantile(samples: &[f64], p: f64, kind: QuantileType) -> f64 {
    let QuantileType::Type7 = kind;
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p = p.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    // `h` indexes the (n - 1)-length span between the first and last order
    // statistic, so `h.floor()` is always a valid index, and `lower_index + 1`
    // is only reached when the fractional part is non-zero, which excludes the
    // final index.
    let h = (n - 1) as f64 * p;
    let lower = h.floor();
    let frac = h - lower;
    let lower_index = lower as usize;
    if frac == 0.0 {
        return sorted[lower_index];
    }
    let low = sorted[lower_index];
    let high = sorted[lower_index + 1];
    low + frac * (high - low)
}

/// Coefficient of variation: sample standard deviation over the mean.
///
/// Uses the `n - 1` denominator, the convention for a sample drawn from the
/// larger population of runs the rig could have produced. Returns NaN when
/// fewer than two samples are supplied or the mean is zero, because dispersion
/// is undefined in both cases rather than zero — reporting zero would let a
/// degenerate sample pass the stability check.
#[must_use]
pub fn coefficient_of_variation(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return f64::NAN;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if mean == 0.0 {
        return f64::NAN;
    }
    let variance = samples
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1.0);
    variance.sqrt() / mean.abs()
}

/// Deterministic 64-bit linear congruential generator.
///
/// Chosen over an external RNG crate so a bound published in a parity document
/// can be recomputed from the seed alone by any reader, without pinning a
/// dependency version whose stream may change between releases.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seeds the generator, mixing once so a low-entropy seed such as zero does
    /// not spend its first draws near the origin.
    fn new(seed: u64) -> Self {
        let mut lcg = Self { state: seed };
        lcg.next_u64();
        lcg
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform index in `0..bound`, using the high bits where an LCG's
    /// randomness is best, via a widening multiply rather than a biased modulo.
    fn index(&mut self, bound: usize) -> usize {
        let draw = u128::from(self.next_u64());
        ((draw * bound as u128) >> 64) as usize
    }
}

/// Paired bootstrap lower bound on the retention ratio.
///
/// Draws `config.resamples` resamples of the combined AB/BA retention ratios
/// with replacement, takes each resample's mean, and reports the
/// `1 - confidence` type-7 quantile of those means as the one-sided lower
/// bound.
pub fn try_bootstrap_paired_max_degradation(
    samples: &PairedSamples,
    config: &BootstrapConfig,
) -> Result<BootstrapResult, StatsError> {
    if samples.ab.len() != samples.ba.len() {
        return Err(StatsError::UnbalancedOrientations {
            ab: samples.ab.len(),
            ba: samples.ba.len(),
        });
    }
    if samples.ab.len() < MINIMUM_RETAINED_PAIRS {
        return Err(StatsError::InsufficientPairs {
            found: samples.ab.len(),
            required: MINIMUM_RETAINED_PAIRS,
        });
    }
    if config.resamples < 2 {
        return Err(StatsError::InsufficientResamples {
            found: config.resamples,
        });
    }
    if !(config.confidence > 0.0 && config.confidence < 1.0) {
        return Err(StatsError::InvalidConfidence {
            found: config.confidence,
        });
    }

    let ratios = samples.retention_ratios()?;
    let n = ratios.len();
    let point_estimate = ratios.iter().sum::<f64>() / n as f64;

    let mut rng = Lcg::new(config.seed);
    let mut means = Vec::with_capacity(config.resamples);
    for _ in 0..config.resamples {
        let mut total = 0.0;
        for _ in 0..n {
            total += ratios[rng.index(n)];
        }
        means.push(total / n as f64);
    }

    let lower_bound = hyndman_fan_quantile(&means, 1.0 - config.confidence, QuantileType::Type7);
    Ok(BootstrapResult {
        lower_bound,
        point_estimate,
        resamples: config.resamples,
        confidence: config.confidence,
        pairs: n,
    })
}

/// Panicking wrapper over [`try_bootstrap_paired_max_degradation`].
///
/// # Panics
///
/// Panics when the design is not bootstrappable — too few or unbalanced pairs,
/// a degenerate resample count or confidence level, or a measurement that is
/// not a usable ratio. Each of those is a harness-construction error that must
/// abort the gate rather than yield a bound nobody should trust; callers that
/// want to recover use the fallible form.
#[must_use]
pub fn bootstrap_paired_max_degradation(
    samples: &PairedSamples,
    config: &BootstrapConfig,
) -> BootstrapResult {
    match try_bootstrap_paired_max_degradation(samples, config) {
        Ok(result) => result,
        Err(error) => panic!("parity bootstrap is not well posed: {error}"),
    }
}
