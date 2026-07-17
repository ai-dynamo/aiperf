// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling distributions backed by [`RandomGenerator`].
//!
//! These types implement the distribution control flow: post-draw clamping and
//! integer ceiling; normal, log-normal, multimodal, and empirical raw sampling;
//! and cumulative probabilities, right-side search, and batch validation.

use crate::rng::error::{Result, RngError};
use crate::rng::generator::{RandomGenerator, positive_integer_from_f64};

const PROBABILITY_SUM_REL_TOLERANCE: f64 = 1.0e-6;
const PROBABILITY_SUM_ABS_TOLERANCE: f64 = 1.0e-6;

/// Random operations required by workload distributions.
///
/// [`RandomGenerator`] and deterministic test sources use this generic seam
/// without dynamic dispatch on the hot path.
pub trait SamplingRng {
    /// Draw a uniform value from `[0, 1)`.
    fn random(&mut self) -> f64;

    /// Draw `size` uniform values from `[0, 1)`.
    fn random_batch(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.random()).collect()
    }

    /// Draw from a bounded normal distribution.
    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64>;

    /// Draw from a normal distribution truncated at zero.
    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64>;

    /// Draw a positive integer from a normal distribution truncated at zero.
    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64>;
}

impl SamplingRng for RandomGenerator {
    fn random(&mut self) -> f64 {
        Self::random(self)
    }

    fn random_batch(&mut self, size: usize) -> Vec<f64> {
        Self::random_batch(self, size)
    }

    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
        Self::sample_normal(self, mean, stddev, lower, upper)
    }

    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64> {
        Self::sample_positive_normal(self, mean, stddev)
    }

    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64> {
        Self::sample_positive_normal_integer(self, mean, stddev)
    }
}

/// Sampling contract implemented by every configured value distribution.
///
/// `R` is generic because sampling is a hot path. The closed five-way
/// [`SamplingDistribution`] enum is the configuration adapter; consumers that
/// provide another distribution can program against this trait directly.
pub trait DistributionSampler<R: SamplingRng + ?Sized = RandomGenerator> {
    /// Draw one bounded sample.
    fn sample(&self, rng: &mut R) -> Result<f64>;

    /// Draw one sample and return `max(1, ceil(sample))`.
    fn sample_int(&self, rng: &mut R) -> Result<i64> {
        positive_integer_from_f64(self.sample(rng)?.ceil(), "distribution integer sample")
    }

    /// Return the unclamped analytic expected value.
    fn expected_value(&self) -> f64;
}

/// Sampling contract implemented by sequence-length sources.
///
/// The trait leaves room for dataset-backed or graph-backed sequence sources
/// while [`SequenceLengthDistribution`] remains the concrete percentage model.
pub trait SequenceSampler<R: SamplingRng + ?Sized = RandomGenerator> {
    /// Draw one `(ISL, OSL)` pair.
    fn sample(&self, rng: &mut R) -> Result<(i64, i64)>;

    /// Draw `batch_size` `(ISL, OSL)` pairs.
    fn sample_batch(&self, rng: &mut R, batch_size: usize) -> Result<Vec<(i64, i64)>>;
}

/// A weighted component in a [`MultimodalDistribution`].
#[derive(Clone, Debug, PartialEq)]
pub struct PeakEntry {
    /// Distribution sampled when this peak is selected.
    distribution: SamplingDistribution,
    /// Relative non-negative weight.
    weight: f64,
}

impl PeakEntry {
    /// Construct a peak entry.
    pub fn new(distribution: SamplingDistribution, weight: f64) -> Result<Self> {
        validate_weight(weight, "peak weight")?;
        Ok(Self {
            distribution,
            weight,
        })
    }

    /// Distribution sampled when this peak is selected.
    pub const fn distribution(&self) -> &SamplingDistribution {
        &self.distribution
    }

    /// Relative non-negative weight.
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// A weighted value in an [`EmpiricalDistribution`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalPoint {
    /// The discrete sampled value.
    value: f64,
    /// Relative positive weight.
    weight: f64,
}

impl EmpiricalPoint {
    /// Construct an empirical point.
    pub fn new(value: f64, weight: f64) -> Result<Self> {
        validate_finite(value, "empirical value")?;
        validate_weight(weight, "empirical weight")?;
        if weight <= 0.0 {
            return Err(RngError::InvalidWeights {
                reason: "empirical weights must be positive",
            });
        }
        Ok(Self { value, weight })
    }

    /// Discrete sampled value.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Relative positive weight.
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// A constant-valued distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedDistribution {
    /// Constant value returned by every raw sample.
    pub value: f64,
    /// Optional inclusive lower clamp.
    pub min: Option<f64>,
    /// Optional inclusive upper clamp.
    pub max: Option<f64>,
}

impl FixedDistribution {
    /// Construct a fixed distribution.
    pub fn new(value: f64) -> Result<Self> {
        validate_finite(value, "fixed value")?;
        Ok(Self {
            value,
            min: None,
            max: None,
        })
    }
}

/// Positive normal distribution parameterized by mean and stddev.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalDistribution {
    /// Mean of the normal distribution.
    pub mean: f64,
    /// Standard deviation. Zero is deterministic.
    pub stddev: f64,
    /// Optional inclusive lower clamp.
    pub min: Option<f64>,
    /// Optional inclusive upper clamp.
    pub max: Option<f64>,
}

impl NormalDistribution {
    /// Construct a normal distribution.
    pub fn new(mean: f64, stddev: f64) -> Result<Self> {
        if mean < 0.0 {
            return Err(RngError::InvalidParameter {
                what: "normal mean",
                value: mean,
            });
        }
        validate_finite(mean, "normal mean")?;
        if stddev < 0.0 {
            return Err(RngError::InvalidParameter {
                what: "normal stddev",
                value: stddev,
            });
        }
        validate_finite(stddev, "normal stddev")?;
        Ok(Self {
            mean,
            stddev,
            min: None,
            max: None,
        })
    }
}

/// Log-normal distribution parameterized by real-space mean and median.
#[derive(Clone, Debug, PartialEq)]
pub struct LogNormalDistribution {
    /// Desired real-space mean.
    pub mean: f64,
    /// Desired median; must be `<= mean`.
    pub median: f64,
    /// Optional inclusive lower clamp.
    pub min: Option<f64>,
    /// Optional inclusive upper clamp.
    pub max: Option<f64>,
}

impl LogNormalDistribution {
    /// Construct a log-normal distribution.
    pub fn new(mean: f64, median: f64) -> Result<Self> {
        if mean <= 0.0 {
            return Err(RngError::InvalidParameter {
                what: "lognormal mean",
                value: mean,
            });
        }
        if median <= 0.0 {
            return Err(RngError::InvalidParameter {
                what: "lognormal median",
                value: median,
            });
        }
        validate_finite(mean, "lognormal mean")?;
        validate_finite(median, "lognormal median")?;
        if median > mean {
            return Err(RngError::InvalidParameter {
                what: "lognormal median",
                value: median,
            });
        }
        Ok(Self {
            mean,
            median,
            min: None,
            max: None,
        })
    }

    fn sigma(&self) -> f64 {
        if self.median >= self.mean {
            0.0
        } else {
            (2.0 * (self.mean / self.median).ln()).sqrt()
        }
    }
}

/// Weighted mixture of two or more distributions.
#[derive(Clone, Debug, PartialEq)]
pub struct MultimodalDistribution {
    /// Weighted peak list.
    peaks: Vec<PeakEntry>,
    cumulative_weights: Vec<f64>,
    total_weight: f64,
    /// Optional inclusive lower clamp.
    pub min: Option<f64>,
    /// Optional inclusive upper clamp.
    pub max: Option<f64>,
}

impl MultimodalDistribution {
    /// Construct a multimodal distribution from at least two peaks.
    pub fn new(peaks: Vec<PeakEntry>) -> Result<Self> {
        if peaks.len() < 2 {
            return Err(RngError::EmptySequence { what: "peaks" });
        }
        let (cumulative_weights, total_weight) =
            cumulative_weights(peaks.iter().map(|peak| peak.weight).collect())?;
        Ok(Self {
            peaks,
            cumulative_weights,
            total_weight,
            min: None,
            max: None,
        })
    }

    /// Immutable weighted peaks.
    pub fn peaks(&self) -> &[PeakEntry] {
        &self.peaks
    }
}

/// Discrete weighted empirical distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalDistribution {
    /// Weighted points to sample.
    points: Vec<EmpiricalPoint>,
    cumulative_weights: Vec<f64>,
    total_weight: f64,
    /// Optional inclusive lower clamp.
    pub min: Option<f64>,
    /// Optional inclusive upper clamp.
    pub max: Option<f64>,
}

impl EmpiricalDistribution {
    /// Construct an empirical distribution from one or more weighted points.
    pub fn new(points: Vec<EmpiricalPoint>) -> Result<Self> {
        if points.is_empty() {
            return Err(RngError::EmptySequence { what: "points" });
        }
        let (cumulative_weights, total_weight) =
            cumulative_weights(points.iter().map(|point| point.weight).collect())?;
        Ok(Self {
            points,
            cumulative_weights,
            total_weight,
            min: None,
            max: None,
        })
    }

    /// Immutable empirical points.
    pub fn points(&self) -> &[EmpiricalPoint] {
        &self.points
    }
}

/// Five-way sampling distribution used by AIPerf configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingDistribution {
    /// Fixed value.
    Fixed(FixedDistribution),
    /// Positive normal.
    Normal(NormalDistribution),
    /// Log-normal.
    LogNormal(LogNormalDistribution),
    /// Weighted mixture of distributions.
    Multimodal(MultimodalDistribution),
    /// Discrete empirical points.
    Empirical(EmpiricalDistribution),
}

impl SamplingDistribution {
    /// Construct a fixed distribution.
    pub fn fixed(value: f64) -> Result<Self> {
        Ok(Self::Fixed(FixedDistribution::new(value)?))
    }

    /// Construct a normal distribution.
    pub fn normal(mean: f64, stddev: f64) -> Result<Self> {
        Ok(Self::Normal(NormalDistribution::new(mean, stddev)?))
    }

    /// Construct a log-normal distribution.
    pub fn lognormal(mean: f64, median: f64) -> Result<Self> {
        Ok(Self::LogNormal(LogNormalDistribution::new(mean, median)?))
    }

    /// Construct a multimodal distribution.
    pub fn multimodal(peaks: Vec<PeakEntry>) -> Result<Self> {
        Ok(Self::Multimodal(MultimodalDistribution::new(peaks)?))
    }

    /// Construct an empirical distribution.
    pub fn empirical(points: Vec<EmpiricalPoint>) -> Result<Self> {
        Ok(Self::Empirical(EmpiricalDistribution::new(points)?))
    }

    /// Return a copy with optional inclusive bounds.
    pub fn with_bounds(mut self, min: Option<f64>, max: Option<f64>) -> Result<Self> {
        validate_bounds(min, max)?;
        match &mut self {
            Self::Fixed(d) => {
                d.min = min;
                d.max = max;
            }
            Self::Normal(d) => {
                d.min = min;
                d.max = max;
            }
            Self::LogNormal(d) => {
                d.min = min;
                d.max = max;
            }
            Self::Multimodal(d) => {
                d.min = min;
                d.max = max;
            }
            Self::Empirical(d) => {
                d.min = min;
                d.max = max;
            }
        }
        Ok(self)
    }

    /// Draw one sample, applying distribution-level bounds after the raw draw.
    pub fn sample(&self, rng: &mut RandomGenerator) -> Result<f64> {
        DistributionSampler::sample(self, rng)
    }

    /// Draw one sample and return `max(1, ceil(sample))`.
    pub fn sample_int(&self, rng: &mut RandomGenerator) -> Result<i64> {
        DistributionSampler::sample_int(self, rng)
    }

    /// Return the unclamped analytic expected value.
    pub fn expected_value(&self) -> f64 {
        DistributionSampler::<RandomGenerator>::expected_value(self)
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for FixedDistribution {
    fn sample(&self, _rng: &mut R) -> Result<f64> {
        Ok(clamp(self.value, (self.min, self.max)))
    }

    fn expected_value(&self) -> f64 {
        self.value
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for NormalDistribution {
    fn sample(&self, rng: &mut R) -> Result<f64> {
        let raw = if self.stddev <= 0.0 {
            self.mean
        } else {
            rng.sample_positive_normal(self.mean, self.stddev)?
        };
        Ok(clamp(raw, (self.min, self.max)))
    }

    fn expected_value(&self) -> f64 {
        self.mean
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for LogNormalDistribution {
    fn sample(&self, rng: &mut R) -> Result<f64> {
        let sigma = self.sigma();
        let raw = if sigma <= 0.0 {
            self.mean
        } else {
            rng.sample_normal(self.median.ln(), sigma, f64::NEG_INFINITY, f64::INFINITY)?
                .exp()
        };
        Ok(clamp(raw, (self.min, self.max)))
    }

    fn expected_value(&self) -> f64 {
        self.mean
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for MultimodalDistribution {
    fn sample(&self, rng: &mut R) -> Result<f64> {
        let idx =
            weighted_index_for_random(&self.cumulative_weights, self.total_weight, rng.random());
        let raw = DistributionSampler::<R>::sample(&self.peaks[idx].distribution, rng)?;
        Ok(clamp(raw, (self.min, self.max)))
    }

    fn expected_value(&self) -> f64 {
        self.peaks
            .iter()
            .map(|peak| {
                peak.weight / self.total_weight
                    * DistributionSampler::<RandomGenerator>::expected_value(&peak.distribution)
            })
            .sum()
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for EmpiricalDistribution {
    fn sample(&self, rng: &mut R) -> Result<f64> {
        let idx =
            weighted_index_for_random(&self.cumulative_weights, self.total_weight, rng.random());
        Ok(clamp(self.points[idx].value, (self.min, self.max)))
    }

    fn expected_value(&self) -> f64 {
        self.points
            .iter()
            .map(|point| point.weight / self.total_weight * point.value)
            .sum()
    }
}

impl<R: SamplingRng + ?Sized> DistributionSampler<R> for SamplingDistribution {
    fn sample(&self, rng: &mut R) -> Result<f64> {
        match self {
            Self::Fixed(distribution) => DistributionSampler::<R>::sample(distribution, rng),
            Self::Normal(distribution) => DistributionSampler::<R>::sample(distribution, rng),
            Self::LogNormal(distribution) => DistributionSampler::<R>::sample(distribution, rng),
            Self::Multimodal(distribution) => DistributionSampler::<R>::sample(distribution, rng),
            Self::Empirical(distribution) => DistributionSampler::<R>::sample(distribution, rng),
        }
    }

    fn expected_value(&self) -> f64 {
        match self {
            Self::Fixed(distribution) => {
                DistributionSampler::<RandomGenerator>::expected_value(distribution)
            }
            Self::Normal(distribution) => {
                DistributionSampler::<RandomGenerator>::expected_value(distribution)
            }
            Self::LogNormal(distribution) => {
                DistributionSampler::<RandomGenerator>::expected_value(distribution)
            }
            Self::Multimodal(distribution) => {
                DistributionSampler::<RandomGenerator>::expected_value(distribution)
            }
            Self::Empirical(distribution) => {
                DistributionSampler::<RandomGenerator>::expected_value(distribution)
            }
        }
    }
}

/// One ISL/OSL pair with probability and optional normal stddevs.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceLengthPair {
    /// Input sequence length mean.
    pub input_seq_len: i64,
    /// Output sequence length mean.
    pub output_seq_len: i64,
    /// Probability in percent, `[0, 100]`.
    pub probability: f64,
    /// Optional input sequence length stddev.
    pub input_seq_len_stddev: f64,
    /// Optional output sequence length stddev.
    pub output_seq_len_stddev: f64,
}

impl SequenceLengthPair {
    /// Construct and validate one sequence-length pair.
    pub fn new(input_seq_len: i64, output_seq_len: i64, probability: f64) -> Result<Self> {
        Self::new_with_stddev(input_seq_len, 0.0, output_seq_len, 0.0, probability)
    }

    /// Construct and validate one sequence-length pair with stddevs.
    pub fn new_with_stddev(
        input_seq_len: i64,
        input_seq_len_stddev: f64,
        output_seq_len: i64,
        output_seq_len_stddev: f64,
        probability: f64,
    ) -> Result<Self> {
        if input_seq_len <= 0 {
            return Err(RngError::InvalidParameter {
                what: "input_seq_len",
                value: input_seq_len as f64,
            });
        }
        if output_seq_len <= 0 {
            return Err(RngError::InvalidParameter {
                what: "output_seq_len",
                value: output_seq_len as f64,
            });
        }
        if !(0.0..=100.0).contains(&probability) || !probability.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "probability",
                value: probability,
            });
        }
        if input_seq_len_stddev < 0.0 || !input_seq_len_stddev.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "input_seq_len_stddev",
                value: input_seq_len_stddev,
            });
        }
        if output_seq_len_stddev < 0.0 || !output_seq_len_stddev.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "output_seq_len_stddev",
                value: output_seq_len_stddev,
            });
        }
        Ok(Self {
            input_seq_len,
            output_seq_len,
            probability,
            input_seq_len_stddev,
            output_seq_len_stddev,
        })
    }
}

/// Probability distribution over sequence-length pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceLengthDistribution {
    pairs: Vec<SequenceLengthPair>,
    cumulative_probs: Vec<f64>,
}

impl SequenceLengthDistribution {
    /// Construct a distribution. Probabilities must be close to 100.0 with
    /// `rtol=1e-6, atol=1e-6`.
    pub fn new(pairs: Vec<SequenceLengthPair>) -> Result<Self> {
        if pairs.is_empty() {
            return Err(RngError::EmptySequence {
                what: "sequence pairs",
            });
        }
        let total: f64 = pairs.iter().map(|p| p.probability).sum();
        if !probability_sum_is_close(total) {
            return Err(RngError::InvalidProbabilitySum { total });
        }
        let mut cumulative = Vec::with_capacity(pairs.len());
        let mut acc = 0.0;
        for pair in &pairs {
            acc += pair.probability / 100.0;
            cumulative.push(acc);
        }
        Ok(Self {
            pairs,
            cumulative_probs: cumulative,
        })
    }

    /// Immutable sequence pairs.
    pub fn pairs(&self) -> &[SequenceLengthPair] {
        &self.pairs
    }

    /// Draw one `(ISL, OSL)` pair.
    pub fn sample(&self, rng: &mut RandomGenerator) -> Result<(i64, i64)> {
        SequenceSampler::sample(self, rng)
    }

    /// Draw `batch_size` samples.
    pub fn sample_batch(
        &self,
        rng: &mut RandomGenerator,
        batch_size: usize,
    ) -> Result<Vec<(i64, i64)>> {
        SequenceSampler::sample_batch(self, rng, batch_size)
    }

    fn sample_pair_at<R: SamplingRng + ?Sized>(
        &self,
        idx: usize,
        rng: &mut R,
    ) -> Result<(i64, i64)> {
        let pair = &self.pairs[idx];
        let isl = if pair.input_seq_len_stddev > 0.0 {
            rng.sample_positive_normal_integer(
                pair.input_seq_len as f64,
                pair.input_seq_len_stddev,
            )?
        } else {
            pair.input_seq_len
        };
        let osl = if pair.output_seq_len_stddev > 0.0 {
            rng.sample_positive_normal_integer(
                pair.output_seq_len as f64,
                pair.output_seq_len_stddev,
            )?
        } else {
            pair.output_seq_len
        };
        Ok((isl, osl))
    }

    fn index_for_random(&self, r: f64) -> usize {
        let idx = self.cumulative_probs.partition_point(|p| *p <= r);
        usize::min(idx, self.pairs.len() - 1)
    }
}

impl<R: SamplingRng + ?Sized> SequenceSampler<R> for SequenceLengthDistribution {
    fn sample(&self, rng: &mut R) -> Result<(i64, i64)> {
        self.sample_pair_at(self.index_for_random(rng.random()), rng)
    }

    fn sample_batch(&self, rng: &mut R, batch_size: usize) -> Result<Vec<(i64, i64)>> {
        if batch_size == 0 {
            return Err(RngError::InvalidParameter {
                what: "batch_size",
                value: 0.0,
            });
        }
        let indices: Vec<_> = rng
            .random_batch(batch_size)
            .into_iter()
            .map(|r| self.index_for_random(r))
            .collect();
        indices
            .into_iter()
            .map(|idx| self.sample_pair_at(idx, rng))
            .collect()
    }
}

fn probability_sum_is_close(total: f64) -> bool {
    (total - 100.0).abs() <= PROBABILITY_SUM_ABS_TOLERANCE + PROBABILITY_SUM_REL_TOLERANCE * 100.0
}

fn weighted_index_for_random(cumulative_weights: &[f64], total: f64, random: f64) -> usize {
    let r = random * total;
    let idx = cumulative_weights.partition_point(|weight| *weight <= r);
    usize::min(idx, cumulative_weights.len() - 1)
}

fn validate_weight(weight: f64, what: &'static str) -> Result<()> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(RngError::InvalidParameter {
            what,
            value: weight,
        });
    }
    Ok(())
}

fn cumulative_weights(mut weights: Vec<f64>) -> Result<(Vec<f64>, f64)> {
    let mut total = 0.0;
    for weight in &mut weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(RngError::InvalidWeights {
                reason: "weights must be finite and non-negative",
            });
        }
        total += *weight;
        if !total.is_finite() {
            return Err(RngError::InvalidWeights {
                reason: "weights must have a finite sum",
            });
        }
        *weight = total;
    }
    if total <= 0.0 {
        return Err(RngError::InvalidWeights {
            reason: "weights must sum to a positive value",
        });
    }
    Ok((weights, total))
}

fn validate_finite(value: f64, what: &'static str) -> Result<()> {
    if !value.is_finite() {
        return Err(RngError::InvalidParameter { what, value });
    }
    Ok(())
}

fn validate_bounds(min: Option<f64>, max: Option<f64>) -> Result<()> {
    if let Some(min) = min {
        validate_finite(min, "min")?;
    }
    if let Some(max) = max {
        validate_finite(max, "max")?;
    }
    if let (Some(lower), Some(upper)) = (min, max)
        && lower > upper
    {
        return Err(RngError::InvalidBounds { lower, upper });
    }
    Ok(())
}

fn clamp(value: f64, bounds: (Option<f64>, Option<f64>)) -> f64 {
    let (min, max) = bounds;
    let mut out = value;
    if let Some(min) = min
        && out < min
    {
        out = min;
    }
    if let Some(max) = max
        && out > max
    {
        out = max;
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn variance(values: &[f64]) -> f64 {
        let mean = mean(values);
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64
    }

    struct StubRng {
        randoms: VecDeque<f64>,
        normal: f64,
        positive_normal: f64,
        positive_integer: i64,
        normal_calls: usize,
        positive_calls: usize,
        integer_calls: usize,
        fail_normal: bool,
        fail_positive: bool,
        fail_integer: bool,
    }

    impl StubRng {
        fn new(randoms: impl IntoIterator<Item = f64>) -> Self {
            Self {
                randoms: randoms.into_iter().collect(),
                normal: 0.0,
                positive_normal: 0.0,
                positive_integer: 1,
                normal_calls: 0,
                positive_calls: 0,
                integer_calls: 0,
                fail_normal: false,
                fail_positive: false,
                fail_integer: false,
            }
        }
    }

    impl SamplingRng for StubRng {
        fn random(&mut self) -> f64 {
            self.randoms.pop_front().expect("scripted uniform draw")
        }

        fn sample_normal(
            &mut self,
            _mean: f64,
            _stddev: f64,
            _lower: f64,
            _upper: f64,
        ) -> Result<f64> {
            self.normal_calls += 1;
            if self.fail_normal {
                return Err(RngError::InvalidParameter {
                    what: "stub normal",
                    value: 0.0,
                });
            }
            Ok(self.normal)
        }

        fn sample_positive_normal(&mut self, _mean: f64, _stddev: f64) -> Result<f64> {
            self.positive_calls += 1;
            if self.fail_positive {
                return Err(RngError::InvalidParameter {
                    what: "stub positive normal",
                    value: 0.0,
                });
            }
            Ok(self.positive_normal)
        }

        fn sample_positive_normal_integer(&mut self, _mean: f64, _stddev: f64) -> Result<i64> {
            self.integer_calls += 1;
            if self.fail_integer {
                return Err(RngError::InvalidParameter {
                    what: "stub integer",
                    value: 0.0,
                });
            }
            Ok(self.positive_integer)
        }
    }

    #[test]
    fn fixed_distribution_samples_constant_and_clamps() {
        let mut rng = RandomGenerator::from_seed(Some(1));
        let dist = SamplingDistribution::fixed(10.0)
            .unwrap()
            .with_bounds(Some(0.0), Some(5.0))
            .unwrap();
        assert_eq!(dist.sample(&mut rng).unwrap(), 5.0);
        assert_eq!(dist.expected_value(), 10.0);
    }

    #[test]
    fn all_distribution_constructors_reject_invalid_models() {
        assert!(FixedDistribution::new(f64::NAN).is_err());
        assert!(NormalDistribution::new(-1.0, 1.0).is_err());
        assert!(NormalDistribution::new(f64::INFINITY, 1.0).is_err());
        assert!(NormalDistribution::new(1.0, -1.0).is_err());
        assert!(NormalDistribution::new(1.0, f64::NAN).is_err());
        assert!(LogNormalDistribution::new(0.0, 1.0).is_err());
        assert!(LogNormalDistribution::new(1.0, 0.0).is_err());
        assert!(LogNormalDistribution::new(f64::INFINITY, 1.0).is_err());
        assert!(LogNormalDistribution::new(1.0, f64::INFINITY).is_err());
        assert!(LogNormalDistribution::new(1.0, 2.0).is_err());
        assert!(SamplingDistribution::fixed(f64::NAN).is_err());
        assert!(SamplingDistribution::normal(-1.0, 1.0).is_err());
        assert!(SamplingDistribution::lognormal(1.0, 2.0).is_err());
        assert!(SamplingDistribution::multimodal(vec![]).is_err());

        let fixed = SamplingDistribution::fixed(1.0).unwrap();
        assert!(PeakEntry::new(fixed.clone(), -1.0).is_err());
        assert!(PeakEntry::new(fixed.clone(), f64::NAN).is_err());
        assert!(MultimodalDistribution::new(vec![]).is_err());
        assert!(
            MultimodalDistribution::new(vec![PeakEntry::new(fixed.clone(), 1.0).unwrap()]).is_err()
        );
        assert!(
            MultimodalDistribution::new(vec![
                PeakEntry::new(fixed.clone(), 0.0).unwrap(),
                PeakEntry::new(fixed.clone(), 0.0).unwrap(),
            ])
            .is_err()
        );
        assert!(
            MultimodalDistribution::new(vec![
                PeakEntry::new(fixed.clone(), f64::MAX).unwrap(),
                PeakEntry::new(fixed, f64::MAX).unwrap(),
            ])
            .is_err()
        );

        assert!(EmpiricalPoint::new(f64::NAN, 1.0).is_err());
        assert!(EmpiricalPoint::new(1.0, 0.0).is_err());
        assert!(EmpiricalPoint::new(1.0, -1.0).is_err());
        assert!(EmpiricalPoint::new(1.0, f64::INFINITY).is_err());
        assert!(EmpiricalDistribution::new(vec![]).is_err());
        assert!(
            EmpiricalDistribution::new(vec![
                EmpiricalPoint::new(1.0, f64::MAX).unwrap(),
                EmpiricalPoint::new(2.0, f64::MAX).unwrap(),
            ])
            .is_err()
        );
        assert!(cumulative_weights(vec![]).is_err());
        assert!(cumulative_weights(vec![-1.0, 2.0]).is_err());
        assert!(cumulative_weights(vec![f64::NAN]).is_err());
        assert!(cumulative_weights(vec![f64::MAX, f64::MAX]).is_err());
    }

    #[test]
    fn bounds_apply_to_every_closed_union_variant() {
        let variants = vec![
            SamplingDistribution::fixed(10.0).unwrap(),
            SamplingDistribution::normal(10.0, 0.0).unwrap(),
            SamplingDistribution::lognormal(10.0, 10.0).unwrap(),
            SamplingDistribution::multimodal(vec![
                PeakEntry::new(SamplingDistribution::fixed(10.0).unwrap(), 1.0).unwrap(),
                PeakEntry::new(SamplingDistribution::fixed(11.0).unwrap(), 0.0).unwrap(),
            ])
            .unwrap(),
            SamplingDistribution::empirical(vec![EmpiricalPoint::new(10.0, 1.0).unwrap()]).unwrap(),
        ];
        let mut rng = RandomGenerator::from_seed(Some(1));
        for distribution in variants {
            let bounded = distribution.with_bounds(Some(2.0), Some(4.0)).unwrap();
            assert_eq!(bounded.sample(&mut rng).unwrap(), 4.0);
        }

        let lower_only = SamplingDistribution::fixed(-5.0)
            .unwrap()
            .with_bounds(Some(-2.0), None)
            .unwrap();
        let upper_only = SamplingDistribution::fixed(5.0)
            .unwrap()
            .with_bounds(None, Some(2.0))
            .unwrap();
        assert_eq!(lower_only.sample(&mut rng).unwrap(), -2.0);
        assert_eq!(upper_only.sample(&mut rng).unwrap(), 2.0);
    }

    #[test]
    fn sampler_trait_accepts_external_rng_and_distribution_implementations() {
        struct ExternalDistribution {
            fail: bool,
        }

        impl<R: SamplingRng + ?Sized> DistributionSampler<R> for ExternalDistribution {
            fn sample(&self, rng: &mut R) -> Result<f64> {
                if self.fail {
                    return Err(RngError::InvalidParameter {
                        what: "external sample",
                        value: 0.0,
                    });
                }
                Ok(10.0 + rng.random())
            }

            fn expected_value(&self) -> f64 {
                10.5
            }
        }

        let mut rng = StubRng::new([0.25]);
        assert_eq!(
            DistributionSampler::sample(&ExternalDistribution { fail: false }, &mut rng).unwrap(),
            10.25
        );
        assert_eq!(
            DistributionSampler::<StubRng>::expected_value(&ExternalDistribution { fail: false }),
            10.5
        );
        assert_eq!(
            DistributionSampler::sample_int(
                &ExternalDistribution { fail: false },
                &mut StubRng::new([0.25]),
            )
            .unwrap(),
            11
        );
        let mut failing_rng = StubRng::new([]);
        assert!(
            DistributionSampler::sample(&ExternalDistribution { fail: true }, &mut failing_rng)
                .is_err()
        );
        assert!(
            DistributionSampler::<StubRng>::sample_int(
                &ExternalDistribution { fail: true },
                &mut failing_rng,
            )
            .is_err()
        );
    }

    #[test]
    fn configured_distributions_propagate_sampling_source_errors() {
        let normal = NormalDistribution::new(10.0, 1.0).unwrap();
        let mut failing_positive = StubRng::new([]);
        failing_positive.fail_positive = true;
        assert!(DistributionSampler::<StubRng>::sample(&normal, &mut failing_positive).is_err());

        let mut invalid_normal = normal.clone();
        invalid_normal.mean = -1.0;
        assert!(
            DistributionSampler::<RandomGenerator>::sample(
                &invalid_normal,
                &mut RandomGenerator::from_seed(Some(1)),
            )
            .is_err()
        );
        assert!(
            SamplingDistribution::Normal(invalid_normal.clone())
                .sample_int(&mut RandomGenerator::from_seed(Some(10)))
                .is_err()
        );

        let lognormal = LogNormalDistribution::new(10.0, 5.0).unwrap();
        let mut failing_normal = StubRng::new([]);
        failing_normal.fail_normal = true;
        assert!(DistributionSampler::<StubRng>::sample(&lognormal, &mut failing_normal).is_err());

        let mut invalid_lognormal = lognormal;
        invalid_lognormal.median = f64::NAN;
        assert!(
            DistributionSampler::<RandomGenerator>::sample(
                &invalid_lognormal,
                &mut RandomGenerator::from_seed(Some(2)),
            )
            .is_err()
        );

        let multimodal = MultimodalDistribution::new(vec![
            PeakEntry::new(SamplingDistribution::normal(10.0, 1.0).unwrap(), 1.0).unwrap(),
            PeakEntry::new(SamplingDistribution::fixed(1.0).unwrap(), 0.0).unwrap(),
        ])
        .unwrap();
        let mut failing_peak = StubRng::new([0.0]);
        failing_peak.fail_positive = true;
        assert!(DistributionSampler::<StubRng>::sample(&multimodal, &mut failing_peak).is_err());

        let mut invalid_multimodal = multimodal;
        invalid_multimodal.peaks[0].distribution = SamplingDistribution::Normal(invalid_normal);
        assert!(
            DistributionSampler::<RandomGenerator>::sample(
                &invalid_multimodal,
                &mut RandomGenerator::from_seed(Some(3)),
            )
            .is_err()
        );
    }

    #[test]
    fn normal_distribution_uses_positive_normal_semantics() {
        let mut rng = RandomGenerator::from_seed(Some(2));
        let dist = SamplingDistribution::normal(100.0, 10.0).unwrap();
        let samples: Vec<_> = (0..20_000)
            .map(|_| dist.sample(&mut rng).unwrap())
            .collect();
        let sample_mean = mean(&samples);
        assert!((sample_mean - 100.0).abs() / 100.0 < 0.02, "{sample_mean}");
        let sample_variance = variance(&samples);
        assert!(
            (sample_variance - 100.0).abs() / 100.0 < 0.05,
            "{sample_variance}"
        );
        assert!(samples.iter().all(|v| *v >= 0.0));

        let mut stub = StubRng::new([]);
        stub.positive_normal = 7.5;
        let direct = NormalDistribution::new(100.0, 10.0).unwrap();
        assert_eq!(
            DistributionSampler::<StubRng>::sample(&direct, &mut stub).unwrap(),
            7.5
        );
        assert_eq!(stub.positive_calls, 1);
        assert_eq!(
            DistributionSampler::<StubRng>::expected_value(&direct),
            100.0
        );
        assert_eq!(dist.expected_value(), 100.0);
        let deterministic = NormalDistribution::new(12.0, 0.0).unwrap();
        assert_eq!(
            DistributionSampler::<StubRng>::sample(&deterministic, &mut StubRng::new([])).unwrap(),
            12.0
        );
        let enum_normal = SamplingDistribution::Normal(direct);
        let mut enum_stub = StubRng::new([]);
        enum_stub.positive_normal = 8.0;
        assert_eq!(
            DistributionSampler::sample(&enum_normal, &mut enum_stub).unwrap(),
            8.0
        );
    }

    #[test]
    fn lognormal_distribution_derives_sigma_from_mean_and_median() {
        let mut rng = RandomGenerator::from_seed(Some(3));
        let deterministic = SamplingDistribution::lognormal(5.0, 5.0).unwrap();
        assert_eq!(deterministic.sample(&mut rng).unwrap(), 5.0);

        let skewed = SamplingDistribution::lognormal(10.0, 5.0).unwrap();
        let samples: Vec<_> = (0..80_000)
            .map(|_| skewed.sample(&mut rng).unwrap())
            .collect();
        let sample_mean = mean(&samples);
        assert!((sample_mean - 10.0).abs() / 10.0 < 0.05, "{sample_mean}");
        let sample_variance = variance(&samples);
        assert!(
            (sample_variance - 300.0).abs() / 300.0 < 0.12,
            "{sample_variance}"
        );

        let mut stub = StubRng::new([]);
        stub.normal = 7.0_f64.ln();
        let direct = LogNormalDistribution::new(10.0, 5.0).unwrap();
        assert!(
            (DistributionSampler::<StubRng>::sample(&direct, &mut stub).unwrap() - 7.0).abs()
                < 1.0e-12
        );
        assert_eq!(stub.normal_calls, 1);
        assert_eq!(
            DistributionSampler::<StubRng>::expected_value(&direct),
            10.0
        );
        assert_eq!(skewed.expected_value(), 10.0);
        let deterministic_direct = LogNormalDistribution::new(6.0, 6.0).unwrap();
        assert_eq!(
            DistributionSampler::<StubRng>::sample(&deterministic_direct, &mut StubRng::new([]),)
                .unwrap(),
            6.0
        );
        let enum_lognormal = SamplingDistribution::LogNormal(direct);
        let mut enum_stub = StubRng::new([]);
        enum_stub.normal = 9.0_f64.ln();
        assert!(
            (DistributionSampler::sample(&enum_lognormal, &mut enum_stub).unwrap() - 9.0).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn multimodal_distribution_uses_weighted_cumulative_walk() {
        let mut rng = RandomGenerator::from_seed(Some(4));
        let dist = SamplingDistribution::multimodal(vec![
            PeakEntry::new(SamplingDistribution::fixed(1.0).unwrap(), 0.0).unwrap(),
            PeakEntry::new(SamplingDistribution::fixed(9.0).unwrap(), 5.0).unwrap(),
        ])
        .unwrap();
        for _ in 0..100 {
            assert_eq!(dist.sample(&mut rng).unwrap(), 9.0);
        }
        assert_eq!(dist.expected_value(), 9.0);

        let balanced_inner = MultimodalDistribution::new(vec![
            PeakEntry::new(SamplingDistribution::fixed(1.0).unwrap(), 1.0).unwrap(),
            PeakEntry::new(SamplingDistribution::fixed(2.0).unwrap(), 1.0).unwrap(),
        ])
        .unwrap();
        let balanced = SamplingDistribution::Multimodal(balanced_inner.clone());
        let mut first = StubRng::new([0.0]);
        let mut boundary = StubRng::new([0.5]);
        let mut defensive_fallback = StubRng::new([1.0]);
        assert_eq!(
            DistributionSampler::sample(&balanced, &mut first).unwrap(),
            1.0
        );
        assert_eq!(
            DistributionSampler::sample(&balanced, &mut boundary).unwrap(),
            2.0
        );
        assert_eq!(
            DistributionSampler::sample(&balanced, &mut defensive_fallback).unwrap(),
            2.0
        );
        assert_eq!(balanced_inner.peaks().len(), 2);
        assert_eq!(balanced_inner.peaks()[0].weight(), 1.0);
        assert_eq!(
            balanced_inner.peaks()[0].distribution().expected_value(),
            1.0
        );
        assert_eq!(weighted_index_for_random(&[1.0, 2.0], 2.0, 0.0), 0);
        assert_eq!(weighted_index_for_random(&[1.0, 2.0], 2.0, 0.5), 1);
        assert_eq!(weighted_index_for_random(&[1.0, 2.0], 2.0, 1.0), 1);
    }

    #[test]
    fn empirical_distribution_uses_weighted_values() {
        let mut rng = RandomGenerator::from_seed(Some(5));
        assert!(EmpiricalPoint::new(1.0, 0.0).is_err());
        assert!(SamplingDistribution::empirical(Vec::new()).is_err());

        let dist = SamplingDistribution::empirical(vec![
            EmpiricalPoint::new(1.0, 1.0).unwrap(),
            EmpiricalPoint::new(3.0, 3.0).unwrap(),
        ])
        .unwrap();
        assert_eq!(dist.expected_value(), 2.5);
        let samples: Vec<_> = (0..20_000)
            .map(|_| dist.sample(&mut rng).unwrap())
            .collect();
        let sample_mean = mean(&samples);
        assert!((sample_mean - 2.5).abs() / 2.5 < 0.03, "{sample_mean}");
        assert!((variance(&samples) - 0.75).abs() / 0.75 < 0.05);

        let inspected = EmpiricalDistribution::new(vec![
            EmpiricalPoint::new(1.0, 1.0).unwrap(),
            EmpiricalPoint::new(2.0, 1.0).unwrap(),
        ])
        .unwrap();
        assert_eq!(inspected.points().len(), 2);
        assert_eq!(inspected.points()[0].value(), 1.0);
        assert_eq!(inspected.points()[0].weight(), 1.0);

        let enum_empirical = SamplingDistribution::Empirical(
            EmpiricalDistribution::new(vec![EmpiricalPoint::new(7.0, 1.0).unwrap()]).unwrap(),
        );
        assert_eq!(
            DistributionSampler::sample(&enum_empirical, &mut StubRng::new([0.0])).unwrap(),
            7.0
        );
    }

    #[test]
    fn distribution_integer_sampling_uses_ceil_minimum_and_checked_range() {
        let mut rng = RandomGenerator::from_seed(Some(33));
        assert_eq!(
            SamplingDistribution::fixed(-2.0)
                .unwrap()
                .sample_int(&mut rng)
                .unwrap(),
            1
        );
        assert_eq!(
            SamplingDistribution::fixed(1.2)
                .unwrap()
                .sample_int(&mut rng)
                .unwrap(),
            2
        );
        assert!(
            SamplingDistribution::fixed(9_223_372_036_854_775_808.0)
                .unwrap()
                .sample_int(&mut rng)
                .is_err()
        );
    }

    #[test]
    fn distribution_bounds_reject_non_finite_or_inverted_values() {
        assert!(
            SamplingDistribution::fixed(1.0)
                .unwrap()
                .with_bounds(Some(2.0), Some(1.0))
                .is_err()
        );
        assert!(
            SamplingDistribution::fixed(1.0)
                .unwrap()
                .with_bounds(Some(f64::NAN), None)
                .is_err()
        );
        assert!(
            SamplingDistribution::fixed(1.0)
                .unwrap()
                .with_bounds(None, Some(f64::INFINITY))
                .is_err()
        );
    }

    #[test]
    fn sequence_distribution_validates_probability_sum() {
        assert!(SequenceLengthDistribution::new(vec![]).is_err());
        let bad =
            SequenceLengthDistribution::new(vec![SequenceLengthPair::new(10, 20, 90.0).unwrap()]);
        assert!(bad.is_err());
    }

    #[test]
    fn sequence_pairs_validate_every_field() {
        assert!(SequenceLengthPair::new(0, 1, 100.0).is_err());
        assert!(SequenceLengthPair::new(-1, 1, 100.0).is_err());
        assert!(SequenceLengthPair::new(1, 0, 100.0).is_err());
        assert!(SequenceLengthPair::new(1, -1, 100.0).is_err());
        for probability in [-1.0, 101.0, f64::NAN, f64::INFINITY] {
            assert!(SequenceLengthPair::new(1, 1, probability).is_err());
        }
        assert!(SequenceLengthPair::new_with_stddev(1, -1.0, 1, 0.0, 100.0).is_err());
        assert!(SequenceLengthPair::new_with_stddev(1, f64::NAN, 1, 0.0, 100.0).is_err());
        assert!(SequenceLengthPair::new_with_stddev(1, 0.0, 1, -1.0, 100.0).is_err());
        assert!(SequenceLengthPair::new_with_stddev(1, 0.0, 1, f64::INFINITY, 100.0).is_err());
        assert!(SequenceLengthPair::new(1, 1, 0.0).is_ok());
        assert!(SequenceLengthPair::new(1, 1, 100.0).is_ok());
    }

    #[test]
    fn sequence_distribution_probability_sum_uses_isclose_tolerance() {
        let accepted = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new(10, 20, 50.0).unwrap(),
            SequenceLengthPair::new(30, 40, 50.000_05).unwrap(),
        ]);
        assert!(accepted.is_ok());

        let rejected = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new(10, 20, 50.0).unwrap(),
            SequenceLengthPair::new(30, 40, 50.001).unwrap(),
        ]);
        assert!(rejected.is_err());
    }

    #[test]
    fn sequence_batch_draws_all_routing_uniforms_before_stddev_samples() {
        let dist = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(100, 10.0, 50, 5.0, 40.0).unwrap(),
            SequenceLengthPair::new_with_stddev(200, 10.0, 80, 5.0, 60.0).unwrap(),
        ])
        .unwrap();

        let mut batch_rng = RandomGenerator::from_seed(Some(7));
        let batch = dist.sample_batch(&mut batch_rng, 16).unwrap();

        let mut expected_rng = RandomGenerator::from_seed(Some(7));
        let indices: Vec<_> = expected_rng
            .random_batch(16)
            .into_iter()
            .map(|r| dist.index_for_random(r))
            .collect();
        let expected: Vec<_> = indices
            .into_iter()
            .map(|idx| dist.sample_pair_at(idx, &mut expected_rng).unwrap())
            .collect();

        assert_eq!(batch, expected);
    }

    #[test]
    fn sequence_sampler_trait_uses_right_side_routing_and_default_batch_draws() {
        let dist = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new(10, 20, 50.0).unwrap(),
            SequenceLengthPair::new(30, 40, 50.0).unwrap(),
        ])
        .unwrap();
        assert_eq!(dist.pairs().len(), 2);

        let mut scalar = StubRng::new([0.5]);
        assert_eq!(
            SequenceSampler::sample(&dist, &mut scalar).unwrap(),
            (30, 40)
        );
        let mut inherent = RandomGenerator::from_seed(Some(100));
        assert!([(10, 20), (30, 40)].contains(&dist.sample(&mut inherent).unwrap()));

        // StubRng intentionally uses SamplingRng's default random_batch method.
        let mut batch = StubRng::new([0.0, 0.499_999, 0.5, 0.999_999]);
        assert_eq!(
            SequenceSampler::sample_batch(&dist, &mut batch, 4).unwrap(),
            vec![(10, 20), (10, 20), (30, 40), (30, 40)]
        );
        assert!(SequenceSampler::sample_batch(&dist, &mut StubRng::new([]), 0).is_err());
    }

    #[test]
    fn sequence_sampler_trait_leaves_room_for_external_sources() {
        struct ExternalSequence;

        impl<R: SamplingRng + ?Sized> SequenceSampler<R> for ExternalSequence {
            fn sample(&self, _rng: &mut R) -> Result<(i64, i64)> {
                Ok((8, 9))
            }

            fn sample_batch(&self, rng: &mut R, batch_size: usize) -> Result<Vec<(i64, i64)>> {
                (0..batch_size).map(|_| self.sample(rng)).collect()
            }
        }

        fn draw<S: SequenceSampler<StubRng>>(
            source: &S,
            rng: &mut StubRng,
        ) -> Result<Vec<(i64, i64)>> {
            source.sample_batch(rng, 2)
        }

        assert_eq!(
            draw(&ExternalSequence, &mut StubRng::new([])).unwrap(),
            vec![(8, 9), (8, 9)]
        );
    }

    #[test]
    fn sequence_distribution_uses_right_side_search_and_clamp() {
        let dist = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new(10, 20, 50.0).unwrap(),
            SequenceLengthPair::new(30, 40, 50.0).unwrap(),
        ])
        .unwrap();
        assert_eq!(dist.index_for_random(0.0), 0);
        assert_eq!(dist.index_for_random(0.499_999), 0);
        assert_eq!(dist.index_for_random(0.5), 1);
        assert_eq!(dist.index_for_random(1.0), 1);
    }

    #[test]
    fn sequence_distribution_samples_stddev_pairs() {
        let mut rng = RandomGenerator::from_seed(Some(6));
        let dist = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(100, 10.0, 50, 5.0, 100.0).unwrap(),
        ])
        .unwrap();
        assert!(dist.sample_batch(&mut rng, 0).is_err());
        let samples = dist.sample_batch(&mut rng, 100).unwrap();
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|(isl, osl)| *isl >= 1 && *osl >= 1));
        assert!(samples.iter().any(|(isl, _)| *isl != 100));
    }

    #[test]
    fn sequence_stddev_sampling_covers_each_optional_side() {
        let input_only = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(100, 1.0, 50, 0.0, 100.0).unwrap(),
        ])
        .unwrap();
        let mut input_rng = StubRng::new([0.0]);
        input_rng.positive_integer = 101;
        assert_eq!(
            SequenceSampler::sample(&input_only, &mut input_rng).unwrap(),
            (101, 50)
        );
        assert_eq!(input_rng.integer_calls, 1);

        let output_only = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(100, 0.0, 50, 1.0, 100.0).unwrap(),
        ])
        .unwrap();
        let mut output_rng = StubRng::new([0.0]);
        output_rng.positive_integer = 51;
        assert_eq!(
            SequenceSampler::sample(&output_only, &mut output_rng).unwrap(),
            (100, 51)
        );
        assert_eq!(output_rng.integer_calls, 1);

        let mut failed_input = StubRng::new([0.0]);
        failed_input.fail_integer = true;
        assert!(SequenceSampler::sample(&input_only, &mut failed_input).is_err());
        let mut failed_output = StubRng::new([0.0]);
        failed_output.fail_integer = true;
        assert!(SequenceSampler::sample(&output_only, &mut failed_output).is_err());

        let huge_input = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(i64::MAX, 1.0, 1, 0.0, 100.0).unwrap(),
        ])
        .unwrap();
        let huge_output = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new_with_stddev(1, 0.0, i64::MAX, 1.0, 100.0).unwrap(),
        ])
        .unwrap();
        let mut real = RandomGenerator::from_seed(Some(5));
        assert!(huge_input.sample(&mut real).is_err());
        assert!(huge_output.sample(&mut real).is_err());
    }

    #[test]
    fn sequence_pair_frequencies_match_configured_probabilities() {
        let dist = SequenceLengthDistribution::new(vec![
            SequenceLengthPair::new(10, 20, 20.0).unwrap(),
            SequenceLengthPair::new(30, 40, 30.0).unwrap(),
            SequenceLengthPair::new(50, 60, 50.0).unwrap(),
        ])
        .unwrap();
        let mut rng = RandomGenerator::from_seed(Some(919));
        let samples = dist.sample_batch(&mut rng, 100_000).unwrap();
        let expected = [((10, 20), 0.2), ((30, 40), 0.3), ((50, 60), 0.5)];
        for (pair, probability) in expected {
            let observed = samples.iter().filter(|sample| **sample == pair).count() as f64
                / samples.len() as f64;
            assert!(
                (observed - probability).abs() < 0.01,
                "{pair:?}: {observed}"
            );
        }
    }
}
