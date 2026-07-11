// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling distributions backed by [`RandomGenerator`](crate::RandomGenerator).
//!
//! These types port the Python distribution control flow from
//! `src/aiperf/config/distributions.py:109` (post-draw clamping and integer
//! ceiling), `src/aiperf/config/distributions.py:219` / `:269` / `:355` / `:417`
//! (normal, log-normal, multimodal, and empirical raw sampling), and
//! `src/aiperf/common/models/sequence_distribution.py:150` / `:156` / `:188`
//! (cumulative probabilities, right-side search, and batch validation).

use crate::error::{Result, RngError};
use crate::generator::RandomGenerator;

const PROBABILITY_SUM_REL_TOLERANCE: f64 = 1.0e-6;
const PROBABILITY_SUM_ABS_TOLERANCE: f64 = 1.0e-6;

/// A weighted component in a [`MultimodalDistribution`].
#[derive(Clone, Debug, PartialEq)]
pub struct PeakEntry {
    /// Distribution sampled when this peak is selected.
    pub distribution: SamplingDistribution,
    /// Relative non-negative weight.
    pub weight: f64,
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
}

/// A weighted value in an [`EmpiricalDistribution`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalPoint {
    /// The discrete sampled value.
    pub value: f64,
    /// Relative positive weight.
    pub weight: f64,
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
    pub peaks: Vec<PeakEntry>,
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
        validate_weights(peaks.iter().map(|p| p.weight))?;
        Ok(Self {
            peaks,
            min: None,
            max: None,
        })
    }
}

/// Discrete weighted empirical distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalDistribution {
    /// Weighted points to sample.
    pub points: Vec<EmpiricalPoint>,
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
        validate_weights(points.iter().map(|p| p.weight))?;
        Ok(Self {
            points,
            min: None,
            max: None,
        })
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
        let raw = self.sample_raw(rng)?;
        Ok(clamp(raw, self.bounds()))
    }

    /// Draw one sample and return `max(1, ceil(sample))`.
    pub fn sample_int(&self, rng: &mut RandomGenerator) -> Result<i64> {
        Ok(i64::max(1, self.sample(rng)?.ceil() as i64))
    }

    /// Return the unclamped analytic expected value.
    pub fn expected_value(&self) -> f64 {
        match self {
            Self::Fixed(d) => d.value,
            Self::Normal(d) => d.mean,
            Self::LogNormal(d) => d.mean,
            Self::Multimodal(d) => {
                let total: f64 = d.peaks.iter().map(|p| p.weight).sum();
                d.peaks
                    .iter()
                    .map(|p| p.weight / total * p.distribution.expected_value())
                    .sum()
            }
            Self::Empirical(d) => {
                let total: f64 = d.points.iter().map(|p| p.weight).sum();
                d.points.iter().map(|p| p.weight / total * p.value).sum()
            }
        }
    }

    fn sample_raw(&self, rng: &mut RandomGenerator) -> Result<f64> {
        match self {
            Self::Fixed(d) => Ok(d.value),
            Self::Normal(d) => {
                if d.stddev <= 0.0 {
                    Ok(d.mean)
                } else {
                    rng.sample_positive_normal(d.mean, d.stddev)
                }
            }
            Self::LogNormal(d) => {
                let sigma = d.sigma();
                if sigma <= 0.0 {
                    Ok(d.mean)
                } else {
                    Ok(rng
                        .sample_normal(d.median.ln(), sigma, f64::NEG_INFINITY, f64::INFINITY)?
                        .exp())
                }
            }
            Self::Multimodal(d) => {
                let weights: Vec<_> = d.peaks.iter().map(|p| p.weight).collect();
                let idx = weighted_index(rng, &weights)?;
                d.peaks[idx].distribution.sample(rng)
            }
            Self::Empirical(d) => {
                let weights: Vec<_> = d.points.iter().map(|p| p.weight).collect();
                let idx = weighted_index(rng, &weights)?;
                Ok(d.points[idx].value)
            }
        }
    }

    fn bounds(&self) -> (Option<f64>, Option<f64>) {
        match self {
            Self::Fixed(d) => (d.min, d.max),
            Self::Normal(d) => (d.min, d.max),
            Self::LogNormal(d) => (d.min, d.max),
            Self::Multimodal(d) => (d.min, d.max),
            Self::Empirical(d) => (d.min, d.max),
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
    /// Construct a distribution. Probabilities must match Python's `np.isclose`
    /// check against 100.0 with `rtol=1e-6, atol=1e-6`.
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
        self.sample_pair_at(self.index_for_random(rng.random()), rng)
    }

    /// Draw `batch_size` samples.
    pub fn sample_batch(
        &self,
        rng: &mut RandomGenerator,
        batch_size: usize,
    ) -> Result<Vec<(i64, i64)>> {
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

    fn sample_pair_at(&self, idx: usize, rng: &mut RandomGenerator) -> Result<(i64, i64)> {
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

fn probability_sum_is_close(total: f64) -> bool {
    (total - 100.0).abs() <= PROBABILITY_SUM_ABS_TOLERANCE + PROBABILITY_SUM_REL_TOLERANCE * 100.0
}

fn weighted_index(rng: &mut RandomGenerator, weights: &[f64]) -> Result<usize> {
    validate_weights(weights.iter().copied())?;
    let total: f64 = weights.iter().sum();
    let r = rng.random() * total;
    let mut cumulative = 0.0;
    for (idx, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if r < cumulative {
            return Ok(idx);
        }
    }
    Ok(weights.len() - 1)
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

fn validate_weights(weights: impl IntoIterator<Item = f64>) -> Result<()> {
    let mut total = 0.0;
    let mut any = false;
    for weight in weights {
        any = true;
        if !weight.is_finite() || weight < 0.0 {
            return Err(RngError::InvalidWeights {
                reason: "weights must be finite and non-negative",
            });
        }
        total += weight;
    }
    if !any || total <= 0.0 {
        return Err(RngError::InvalidWeights {
            reason: "weights must sum to a positive value",
        });
    }
    Ok(())
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
    if let (Some(lower), Some(upper)) = (min, max) {
        if lower > upper {
            return Err(RngError::InvalidBounds { lower, upper });
        }
    }
    Ok(())
}

fn clamp(value: f64, bounds: (Option<f64>, Option<f64>)) -> f64 {
    let (min, max) = bounds;
    let mut out = value;
    if let Some(min) = min {
        if out < min {
            out = min;
        }
    }
    if let Some(max) = max {
        if out > max {
            out = max;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
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
    fn normal_distribution_uses_positive_normal_semantics() {
        let mut rng = RandomGenerator::from_seed(Some(2));
        let dist = SamplingDistribution::normal(100.0, 10.0).unwrap();
        let samples: Vec<_> = (0..20_000)
            .map(|_| dist.sample(&mut rng).unwrap())
            .collect();
        let sample_mean = mean(&samples);
        assert!((sample_mean - 100.0).abs() / 100.0 < 0.02, "{sample_mean}");
        assert!(samples.iter().all(|v| *v >= 0.0));
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
    }

    #[test]
    fn sequence_distribution_validates_probability_sum() {
        assert!(SequenceLengthDistribution::new(vec![]).is_err());
        let bad =
            SequenceLengthDistribution::new(vec![SequenceLengthPair::new(10, 20, 90.0).unwrap()]);
        assert!(bad.is_err());
    }

    #[test]
    fn sequence_distribution_probability_sum_matches_python_isclose_tolerance() {
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
}
