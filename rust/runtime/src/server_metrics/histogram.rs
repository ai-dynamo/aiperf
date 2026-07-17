// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Polynomial percentile estimation for classic Prometheus histograms.
//!
//! Continuous scrape intervals learn single-bucket means/variance, phase
//! boundaries supply total cumulative-bucket deltas, and generated observations
//! use the exact histogram sum before NumPy-compatible linear percentiles are
//! evaluated.

use std::collections::{BTreeMap, BTreeSet};

use crate::metrics_core::PERCENTILES;

const MIN_VARIANCE_OBSERVATIONS: usize = 3;
const MAX_OBSERVATIONS: f64 = 100_000.0;

/// One cumulative histogram captured by a scrape.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSnapshot {
    /// Scrape timestamp in Clock nanoseconds.
    pub timestamp_ns: i64,
    /// Cumulative histogram sum.
    pub sum: f64,
    /// Cumulative observation count.
    pub count: f64,
    /// Cumulative buckets keyed by upper bound.
    pub buckets: BTreeMap<String, f64>,
}

/// Learned statistics for one finite or `+Inf` bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketStatistics {
    bucket_le: String,
    observation_count: u64,
    weighted_mean_sum: f64,
    sample_count: usize,
    observed_means: Vec<f64>,
}

impl BucketStatistics {
    /// Builds empty statistics for one `le` bucket.
    pub fn new(bucket_le: impl Into<String>) -> Self {
        Self {
            bucket_le: bucket_le.into(),
            observation_count: 0,
            weighted_mean_sum: 0.0,
            sample_count: 0,
            observed_means: Vec::new(),
        }
    }

    /// Bucket upper-bound string.
    pub fn bucket_le(&self) -> &str {
        &self.bucket_le
    }

    /// Total observations contributing to the weighted mean.
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Number of single-bucket intervals learned.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Weighted mean position across learned intervals.
    pub fn estimated_mean(&self) -> Option<f64> {
        (self.observation_count > 0)
            .then_some(self.weighted_mean_sum / self.observation_count as f64)
    }

    /// Sample variance (`ddof=1`) after at least three learned intervals.
    pub fn estimated_variance(&self) -> Option<f64> {
        if self.observed_means.len() < MIN_VARIANCE_OBSERVATIONS {
            return None;
        }
        let mean = self.observed_means.iter().sum::<f64>() / self.observed_means.len() as f64;
        Some(
            self.observed_means
                .iter()
                .map(|value| {
                    let delta = *value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (self.observed_means.len() - 1) as f64,
        )
    }

    /// Records an exact mean from a single-bucket scrape interval.
    pub fn record(&mut self, mean: f64, count: u64) {
        self.observation_count = self.observation_count.saturating_add(count);
        self.weighted_mean_sum += mean * count as f64;
        self.sample_count += 1;
        self.observed_means.push(mean);
    }
}

/// Learns per-bucket means from adjacent scrapes with exactly one active bucket.
///
/// Intervals missing a bucket on either side are skipped rather than treating a
/// partial exposition as zero.
pub fn accumulate_bucket_statistics(
    snapshots: &[HistogramSnapshot],
) -> BTreeMap<String, BucketStatistics> {
    let mut statistics = BTreeMap::<String, BucketStatistics>::new();
    for pair in snapshots.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        let count_delta = (current.count - previous.count) as i64;
        if count_delta <= 0 {
            continue;
        }
        let bucket_names = previous
            .buckets
            .keys()
            .chain(current.buckets.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        if bucket_names
            .iter()
            .any(|name| !previous.buckets.contains_key(name) || !current.buckets.contains_key(name))
        {
            continue;
        }
        let Some(sorted) = sorted_bucket_names(bucket_names) else {
            continue;
        };
        let cumulative = sorted
            .iter()
            .map(|name| (current.buckets[name] - previous.buckets[name]).max(0.0))
            .collect::<Vec<_>>();
        let mut per_bucket = Vec::with_capacity(cumulative.len());
        for (index, value) in cumulative.iter().enumerate() {
            let previous = index
                .checked_sub(1)
                .map_or(0.0, |previous| cumulative[previous]);
            per_bucket.push((*value - previous).max(0.0));
        }
        let active = per_bucket
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > 0.0)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if let [index] = active.as_slice() {
            let mean = (current.sum - previous.sum) / count_delta as f64;
            statistics
                .entry(sorted[*index].clone())
                .or_insert_with(|| BucketStatistics::new(&sorted[*index]))
                .record(mean, count_delta as u64);
        }
    }
    statistics
}

/// Computes the nine AIPerf percentiles using Prometheus linear interpolation.
pub fn compute_prometheus_percentiles(
    cumulative_buckets: &BTreeMap<String, f64>,
    total_count: Option<f64>,
) -> Option<BTreeMap<u32, f64>> {
    let sorted = sorted_bucket_names(cumulative_buckets.keys().cloned())?;
    if sorted.is_empty() {
        return None;
    }
    let total_count = total_count.unwrap_or_else(|| {
        cumulative_buckets
            .get("+Inf")
            .copied()
            .unwrap_or(cumulative_buckets[&sorted[sorted.len() - 1]])
    });
    if total_count == 0.0 || !total_count.is_finite() {
        return None;
    }
    Some(
        PERCENTILES
            .into_iter()
            .map(|percentile| {
                (
                    percentile,
                    prometheus_quantile(
                        percentile as f64 / 100.0,
                        cumulative_buckets,
                        &sorted,
                        total_count,
                    ),
                )
            })
            .collect(),
    )
}

/// Computes variance-aware polynomial percentile estimates.
pub fn compute_estimated_percentiles(
    cumulative_bucket_deltas: &BTreeMap<String, f64>,
    bucket_stats: &BTreeMap<String, BucketStatistics>,
    mut total_sum: f64,
    total_count: u64,
) -> Option<BTreeMap<u32, f64>> {
    if total_count == 0
        || cumulative_bucket_deltas.is_empty()
        || !total_sum.is_finite()
        || total_sum < 0.0
    {
        return None;
    }
    if total_sum == 0.0 {
        return Some(
            PERCENTILES
                .into_iter()
                .map(|percentile| (percentile, 0.0))
                .collect(),
        );
    }
    let finite = sorted_finite_bucket_names(cumulative_bucket_deltas.keys().cloned())?;
    let max_finite = finite.last().and_then(|name| name.parse::<f64>().ok())?;
    let mut per_bucket = cumulative_to_per_bucket(cumulative_bucket_deltas)?;
    let observation_count = per_bucket.values().sum::<f64>();
    if observation_count > MAX_OBSERVATIONS {
        let ratio = MAX_OBSERVATIONS / observation_count;
        for count in per_bucket.values_mut() {
            *count *= ratio;
        }
        total_sum *= ratio;
    }
    let raw_inf_count = per_bucket.get("+Inf").copied().unwrap_or(0.0);
    let inf_count = if raw_inf_count > 0.0 {
        raw_inf_count.ceil() as usize
    } else {
        0
    };
    let estimated_finite_sum = estimate_bucket_sums(&per_bucket, bucket_stats)?
        .values()
        .sum::<f64>();
    let inf_observations =
        estimate_inf_bucket_observations(total_sum, estimated_finite_sum, inf_count, max_finite);
    let actual_finite_sum = total_sum - inf_observations.iter().sum::<f64>();
    let mut observations =
        generate_observations_with_sum_constraint(&per_bucket, actual_finite_sum, bucket_stats)?;
    observations.extend(inf_observations);
    if observations.is_empty() {
        return None;
    }
    observations.sort_by(f64::total_cmp);
    Some(
        PERCENTILES
            .into_iter()
            .map(|percentile| {
                (
                    percentile,
                    linear_percentile_sorted(&observations, percentile as f64),
                )
            })
            .collect(),
    )
}

fn prometheus_quantile(
    quantile: f64,
    cumulative: &BTreeMap<String, f64>,
    sorted: &[String],
    total_count: f64,
) -> f64 {
    let target = quantile * total_count;
    let mut previous_bound = 0.0;
    let mut previous_count = 0.0;
    for name in sorted {
        let current_count = cumulative[name];
        if name == "+Inf" {
            return previous_bound;
        }
        let current_bound = name.parse::<f64>().unwrap_or(previous_bound);
        if current_count >= target {
            let bucket_count = current_count - previous_count;
            if bucket_count == 0.0 {
                return previous_bound;
            }
            let fraction = (target - previous_count) / bucket_count;
            return previous_bound + (current_bound - previous_bound) * fraction;
        }
        previous_bound = current_bound;
        previous_count = current_count;
    }
    previous_bound
}

fn cumulative_to_per_bucket(cumulative: &BTreeMap<String, f64>) -> Option<BTreeMap<String, f64>> {
    let finite = sorted_finite_bucket_names(cumulative.keys().cloned())?;
    let mut output = BTreeMap::new();
    let mut previous = 0.0;
    for name in finite {
        let value = cumulative[&name];
        output.insert(name, value - previous);
        previous = value;
    }
    if let Some(infinite) = cumulative.get("+Inf") {
        output.insert("+Inf".to_string(), *infinite - previous);
    }
    Some(output)
}

fn estimate_bucket_sums(
    per_bucket: &BTreeMap<String, f64>,
    statistics: &BTreeMap<String, BucketStatistics>,
) -> Option<BTreeMap<String, f64>> {
    let finite = sorted_finite_bucket_names(per_bucket.keys().cloned())?;
    let mut sums = BTreeMap::new();
    for name in &finite {
        let count = per_bucket[name];
        if count <= 0.0 {
            continue;
        }
        let (lower, upper) = bucket_bounds(name, &finite)?;
        let midpoint = (lower + upper) / 2.0;
        let mean = statistics
            .get(name)
            .and_then(BucketStatistics::estimated_mean)
            .filter(|mean| *mean > lower && *mean < upper)
            .unwrap_or(midpoint);
        sums.insert(name.clone(), count * mean);
    }
    Some(sums)
}

fn estimate_inf_bucket_observations(
    total_sum: f64,
    estimated_finite_sum: f64,
    count: usize,
    max_finite: f64,
) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let infinite_sum = total_sum - estimated_finite_sum;
    let mut average = if infinite_sum <= 0.0 {
        max_finite * 1.5
    } else {
        infinite_sum / count as f64
    };
    if average <= max_finite {
        average = max_finite * 1.5;
    }
    let mut upper = 2.0 * average - max_finite;
    if upper <= max_finite {
        upper = max_finite * 2.0;
    }
    if count == 1 {
        return vec![average];
    }
    (0..count)
        .map(|index| max_finite + (upper - max_finite) * index as f64 / (count - 1) as f64)
        .collect()
}

fn generate_f3(count: usize, lower: f64, upper: f64, mean: f64, variance: f64) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let second = if mean - lower > 0.0 {
        mean + variance / (mean - lower)
    } else {
        upper
    }
    .clamp(lower, upper);
    let denominator = variance + (mean - lower).powi(2);
    let probability = if denominator > 0.0 {
        variance / denominator
    } else {
        0.5
    }
    .clamp(0.0, 1.0);
    let lower_count = (count as f64 * probability) as usize;
    (0..count)
        .map(|index| if index < lower_count { lower } else { second })
        .collect()
}

fn generate_variance_aware(count: usize, lower: f64, upper: f64, mean: f64, std: f64) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let lower_stds = if std > 0.0 {
        ((mean - lower) / std).min(3.0)
    } else {
        3.0
    };
    let upper_stds = if std > 0.0 {
        ((upper - mean) / std).min(3.0)
    } else {
        3.0
    };
    (0..count)
        .map(|index| {
            let fraction = (index as f64 + 0.5) / count as f64;
            let position = if fraction < 0.5 {
                mean - lower_stds * std * (1.0 - 2.0 * fraction)
            } else {
                mean + upper_stds * std * (2.0 * fraction - 1.0)
            };
            position.clamp(lower, upper)
        })
        .collect()
}

fn generate_blended(count: usize, lower: f64, upper: f64, mean: f64, std: f64) -> Vec<f64> {
    let width = upper - lower;
    let midpoint = (lower + upper) / 2.0;
    let shift = mean - midpoint;
    let variance = generate_variance_aware(count, lower, upper, mean, std);
    variance
        .into_iter()
        .enumerate()
        .map(|(index, variance_value)| {
            let fraction = (index as f64 + 0.5) / count as f64;
            let uniform = (lower + width * fraction + shift).clamp(lower, upper);
            (0.5 * uniform + 0.5 * variance_value).clamp(lower, upper)
        })
        .collect()
}

fn generate_observations_with_sum_constraint(
    source_counts: &BTreeMap<String, f64>,
    mut target_sum: f64,
    statistics: &BTreeMap<String, BucketStatistics>,
) -> Option<Vec<f64>> {
    let finite = sorted_finite_bucket_names(source_counts.keys().cloned())?;
    let mut counts = source_counts.clone();
    let mut total_count = finite.iter().map(|name| counts[name]).sum::<f64>();
    if total_count > MAX_OBSERVATIONS {
        let ratio = MAX_OBSERVATIONS / total_count;
        for count in counts.values_mut() {
            *count *= ratio;
        }
        target_sum *= ratio;
        total_count = finite.iter().map(|name| counts[name]).sum();
    }
    let average = if total_count > 0.0 {
        target_sum / total_count
    } else {
        0.0
    };
    let dominant = if total_count > 0.0 {
        let max_count = finite
            .iter()
            .map(|name| counts[name])
            .max_by(f64::total_cmp)?;
        (max_count / total_count >= 0.95)
            .then(|| {
                finite
                    .iter()
                    .find(|name| counts[*name] == max_count)
                    .cloned()
            })
            .flatten()
    } else {
        None
    };
    let integer_counts = finite
        .iter()
        .map(|name| (name.clone(), counts[name].max(0.0) as usize))
        .collect::<BTreeMap<_, _>>();
    let capacity = integer_counts.values().sum::<usize>();
    if capacity == 0 {
        return Some(Vec::new());
    }
    let mut observations = Vec::with_capacity(capacity);
    let mut ranges = Vec::<(usize, usize, f64, f64)>::new();
    for name in &finite {
        let count = integer_counts[name].min(capacity - observations.len());
        if count == 0 {
            continue;
        }
        let (lower, upper) = bucket_bounds(name, &finite)?;
        let width = upper - lower;
        let midpoint = (lower + upper) / 2.0;
        let learned_mean = statistics
            .get(name)
            .and_then(BucketStatistics::estimated_mean)
            .filter(|mean| *mean > lower && *mean < upper);
        let learned_variance = statistics
            .get(name)
            .and_then(BucketStatistics::estimated_variance);
        let generated = if let (Some(mean), Some(variance)) = (learned_mean, learned_variance)
            && variance > 0.0
        {
            let std = variance.sqrt();
            let spread_coverage = 4.0 * std / width;
            let mean_offset = (mean - midpoint).abs() / width;
            if spread_coverage < 0.01 {
                generate_f3(count, lower, upper, mean, variance)
            } else if spread_coverage < 0.2 && mean_offset < 0.3 {
                generate_blended(count, lower, upper, mean, std)
            } else {
                generate_variance_aware(count, lower, upper, mean, std)
            }
        } else {
            let mut center = learned_mean.unwrap_or(midpoint);
            if dominant.as_ref() == Some(name) && average > lower && average < upper {
                center = average;
            }
            let shift = center - midpoint;
            (0..count)
                .map(|index| {
                    let fraction = (index as f64 + 0.5) / count as f64;
                    (lower + width * fraction + shift).clamp(lower, upper)
                })
                .collect()
        };
        let start = observations.len();
        observations.extend(generated);
        ranges.push((start, count, lower, upper));
    }

    let generated_sum = observations.iter().sum::<f64>();
    if generated_sum <= 0.0 || target_sum <= 0.0 {
        return Some(observations);
    }
    let discrepancy = target_sum - generated_sum;
    if discrepancy.abs() / target_sum < 0.001 {
        return Some(observations);
    }
    let range_count = ranges.len() as f64;
    for (start, count, lower, upper) in ranges {
        let bucket_sum = observations[start..start + count].iter().sum::<f64>();
        let weight = if generated_sum > 0.0 {
            bucket_sum / generated_sum
        } else {
            1.0 / range_count
        };
        let shift = (discrepancy * weight / count as f64)
            .clamp(-(upper - lower) * 0.4, (upper - lower) * 0.4);
        for value in &mut observations[start..start + count] {
            *value = (*value + shift).clamp(lower, upper);
        }
    }
    Some(observations)
}

fn bucket_bounds(name: &str, sorted_finite: &[String]) -> Option<(f64, f64)> {
    let index = sorted_finite
        .iter()
        .position(|candidate| candidate == name)?;
    let upper = name.parse::<f64>().ok()?;
    let lower = if index == 0 {
        0.0
    } else {
        sorted_finite[index - 1].parse::<f64>().ok()?
    };
    Some((lower, upper))
}

fn sorted_bucket_names(names: impl IntoIterator<Item = String>) -> Option<Vec<String>> {
    let mut names = names.into_iter().collect::<Vec<_>>();
    if names
        .iter()
        .any(|name| name != "+Inf" && name.parse::<f64>().is_err())
    {
        return None;
    }
    names.sort_by(|left, right| bucket_sort_value(left).total_cmp(&bucket_sort_value(right)));
    Some(names)
}

fn sorted_finite_bucket_names(names: impl IntoIterator<Item = String>) -> Option<Vec<String>> {
    let mut names = names
        .into_iter()
        .filter(|name| name != "+Inf")
        .collect::<Vec<_>>();
    if names.iter().any(|name| name.parse::<f64>().is_err()) || names.is_empty() {
        return None;
    }
    names.sort_by(|left, right| {
        left.parse::<f64>()
            .unwrap()
            .total_cmp(&right.parse::<f64>().unwrap())
    });
    Some(names)
}

fn bucket_sort_value(name: &str) -> f64 {
    if name == "+Inf" {
        f64::INFINITY
    } else {
        name.parse::<f64>().unwrap_or(f64::NAN)
    }
}

fn linear_percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    let virtual_index = percentile / 100.0 * (values.len() - 1) as f64;
    let lower = virtual_index.floor() as usize;
    let upper = (lower + 1).min(values.len() - 1);
    values[lower] + (virtual_index - lower as f64) * (values[upper] - values[lower])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_string(), *value))
            .collect()
    }

    #[test]
    fn learner_requires_single_bucket_intervals_and_three_samples_for_variance() {
        let snapshots = (0..=3)
            .map(|index| HistogramSnapshot {
                timestamp_ns: index,
                sum: index as f64 * 2.0,
                count: index as f64 * 4.0,
                buckets: map(&[
                    ("0.5", index as f64 * 4.0),
                    ("1.0", index as f64 * 4.0),
                    ("+Inf", index as f64 * 4.0),
                ]),
            })
            .collect::<Vec<_>>();
        let statistics = accumulate_bucket_statistics(&snapshots);
        let first = &statistics["0.5"];
        assert_eq!(first.observation_count(), 12);
        assert_eq!(first.sample_count(), 3);
        assert_eq!(first.estimated_mean(), Some(0.5));
        assert_eq!(first.estimated_variance(), Some(0.0));
    }

    #[test]
    fn learned_schema_gaps_do_not_create_false_bucket_activity() {
        let snapshots = vec![
            HistogramSnapshot {
                timestamp_ns: 0,
                sum: 0.0,
                count: 0.0,
                buckets: map(&[("0.5", 0.0), ("+Inf", 0.0)]),
            },
            HistogramSnapshot {
                timestamp_ns: 1,
                sum: 1.0,
                count: 1.0,
                buckets: map(&[("0.5", 1.0), ("1.0", 1.0), ("+Inf", 1.0)]),
            },
        ];
        assert!(accumulate_bucket_statistics(&snapshots).is_empty());
    }

    #[test]
    fn estimator_matches_numpy_linear_golden_vector() {
        let buckets = map(&[("0.1", 20.0), ("0.5", 60.0), ("1.0", 90.0), ("+Inf", 100.0)]);
        let result = compute_estimated_percentiles(&buckets, &BTreeMap::new(), 45.0, 100).unwrap();
        let expected = [
            (1, 0.0),
            (5, 0.019_503_521_126_760_567),
            (10, 0.044_253_521_126_760_564),
            (25, 0.106_390_845_070_422_57),
            (50, 0.353_521_126_760_563_4),
            (75, 0.629_636_150_234_741_7),
            (90, 0.887_922_535_211_268_6),
            (95, 1.449_999_999_999_999_7),
            (99, 1.890_000_000_000_000_6),
        ];
        for (percentile, expected) in expected {
            assert!(
                (result[&percentile] - expected).abs() < 1e-12,
                "p{percentile}: {} != {expected}",
                result[&percentile]
            );
        }
    }

    #[test]
    fn one_inf_observation_absorbs_the_back_calculated_sum() {
        let result = compute_estimated_percentiles(
            &map(&[("0.001", 1.0), ("+Inf", 2.0)]),
            &BTreeMap::new(),
            100_000.0,
            2,
        )
        .unwrap();
        assert!(result[&99] > 1_000.0);
    }

    #[test]
    fn variance_strategy_paths_match_numpy_golden_vector() {
        let mut statistics = BTreeMap::new();
        for (bucket, means) in [
            ("0.1", [0.05, 0.0501, 0.0499]),
            ("1.0", [0.54, 0.55, 0.56]),
            ("10.0", [5.0, 7.0, 9.0]),
        ] {
            let mut stats = BucketStatistics::new(bucket);
            for mean in means {
                stats.record(mean, 10);
            }
            statistics.insert(bucket.to_string(), stats);
        }
        let result = compute_estimated_percentiles(
            &map(&[
                ("0.1", 30.0),
                ("1.0", 70.0),
                ("10.0", 95.0),
                ("+Inf", 100.0),
            ]),
            &statistics,
            180.0,
            100,
        )
        .unwrap();
        let expected = [
            (1, 0.029_202_474_272_917_767),
            (5, 0.029_202_474_272_917_767),
            (10, 0.029_202_474_272_917_767),
            (25, 0.029_202_474_272_917_767),
            (50, 0.321_225_932_098_367_1),
            (75, 1.009_948_884_878_778),
            (90, 6.103_795_539_515_114),
            (95, 7.415_805_762_539_349),
            (99, 17.525_000_000_000_013),
        ];
        for (percentile, expected) in expected {
            assert!(
                (result[&percentile] - expected).abs() < 1e-12,
                "p{percentile}: {} != {expected}",
                result[&percentile]
            );
        }
    }

    #[test]
    fn zero_sum_and_invalid_inputs_are_explicit() {
        let buckets = map(&[("1.0", 100.0), ("+Inf", 100.0)]);
        assert!(compute_estimated_percentiles(&buckets, &BTreeMap::new(), f64::NAN, 100).is_none());
        assert!(compute_estimated_percentiles(&buckets, &BTreeMap::new(), -1.0, 100).is_none());
        assert!(compute_estimated_percentiles(&buckets, &BTreeMap::new(), 1.0, 0).is_none());
        assert!(
            compute_estimated_percentiles(&buckets, &BTreeMap::new(), 0.0, 100)
                .unwrap()
                .values()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn estimator_caps_materialized_observations() {
        let result = compute_estimated_percentiles(
            &map(&[("1.0", 10_000_000.0), ("+Inf", 10_000_000.0)]),
            &BTreeMap::new(),
            5_000_000.0,
            10_000_000,
        );
        assert!(result.is_some());
    }
}
