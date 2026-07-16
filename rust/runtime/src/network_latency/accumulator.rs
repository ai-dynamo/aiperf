// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flat per-sample accumulation and NumPy-compatible distribution statistics.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::metrics_core::{Accumulator, ExportContext, Phase};

use crate::network_latency::model::{
    NetworkLatencyErrorDetails, NetworkLatencyErrorDetailsCount, NetworkLatencyResults,
    NetworkLatencySample, NetworkLatencyStats, NetworkLatencyTargetSummary,
};

type ErrorKey = (Option<i32>, Option<String>, String);
type ErrorCount = (NetworkLatencyErrorDetails, usize);

#[derive(Clone, Debug, Default, PartialEq)]
struct TargetState {
    target_url: String,
    target_host: String,
    target_port: u16,
    count: usize,
    failure_count: usize,
    successful_rtts: Vec<i64>,
}

/// Run-local accumulator for all probe targets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NetworkLatencyAccumulator {
    benchmark_id: Option<String>,
    samples: Vec<NetworkLatencySample>,
    targets: BTreeMap<String, TargetState>,
    errors: BTreeMap<ErrorKey, ErrorCount>,
}

/// Incompatibility detected while merging independently probed network state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetworkLatencyMergeError {
    /// Probe workers were associated with different benchmark identities.
    BenchmarkIdConflict {
        /// Existing benchmark identifier.
        existing: String,
        /// Incoming benchmark identifier.
        incoming: String,
    },
}

impl Display for NetworkLatencyMergeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::BenchmarkIdConflict { existing, incoming } => write!(
                formatter,
                "cannot merge network latency for benchmark {incoming:?} into {existing:?}"
            ),
        }
    }
}

impl std::error::Error for NetworkLatencyMergeError {}

impl NetworkLatencyAccumulator {
    /// Build an empty accumulator with optional run identity.
    pub fn new(benchmark_id: Option<String>) -> Self {
        Self {
            benchmark_id,
            ..Self::default()
        }
    }

    /// Retain one success or failure sample.
    pub fn add_sample(&mut self, sample: NetworkLatencySample) {
        Self::tally(&mut self.targets, &mut self.errors, &sample);
        self.samples.push(sample);
    }

    /// Fold one borrowed sample into per-target and per-error tallies.
    ///
    /// Shared by live accumulation and the filtered `export_results` pass so the
    /// latter can aggregate in-range samples by reference without cloning whole
    /// samples into a throwaway accumulator.
    fn tally(
        targets: &mut BTreeMap<String, TargetState>,
        errors: &mut BTreeMap<ErrorKey, ErrorCount>,
        sample: &NetworkLatencySample,
    ) {
        let key = format!("{}:{}", sample.target_host, sample.target_port);
        let target = targets.entry(key).or_insert_with(|| TargetState {
            target_url: sample.target_url.clone(),
            target_host: sample.target_host.clone(),
            target_port: sample.target_port,
            ..TargetState::default()
        });
        target.count += 1;
        if sample.success {
            if let Some(rtt_ns) = sample.rtt_ns {
                target.successful_rtts.push(rtt_ns);
            } else {
                target.failure_count += 1;
            }
        } else {
            target.failure_count += 1;
            if let Some(error) = &sample.error {
                let key = (error.code, error.error_type.clone(), error.message.clone());
                errors
                    .entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert_with(|| (error.clone(), 1));
            }
        }
    }

    /// Every retained sample in issuance/completion order.
    pub fn samples(&self) -> &[NetworkLatencySample] {
        &self.samples
    }

    /// Flat mean over all successful samples across every target.
    pub fn mean_rtt_ns(&self) -> Option<f64> {
        let (sum, count) = self
            .targets
            .values()
            .flat_map(|target| target.successful_rtts.iter().copied())
            .fold((0.0, 0usize), |(sum, count), value| {
                (sum + value as f64, count + 1)
            });
        (count > 0).then_some(sum / count as f64)
    }

    /// Successful sample count across every target.
    pub fn successful_sample_count(&self) -> usize {
        self.targets
            .values()
            .map(|target| target.successful_rtts.len())
            .sum()
    }

    /// Successful sample count for one stable `host:port` key.
    pub fn successful_samples_for(&self, target_key: &str) -> usize {
        self.targets
            .get(target_key)
            .map_or(0, |target| target.successful_rtts.len())
    }

    /// Final per-target and aggregate result.
    pub fn export_results(&self) -> NetworkLatencyResults {
        let target_summaries = self
            .targets
            .iter()
            .map(|(key, target)| {
                (
                    key.clone(),
                    NetworkLatencyTargetSummary {
                        target_url: target.target_url.clone(),
                        target_host: target.target_host.clone(),
                        target_port: target.target_port,
                        count: target.count,
                        success_count: target.successful_rtts.len(),
                        failure_count: target.failure_count,
                        stats: stats(&target.successful_rtts),
                    },
                )
            })
            .collect();
        let successful = self
            .targets
            .values()
            .flat_map(|target| target.successful_rtts.iter().copied())
            .collect::<Vec<_>>();
        NetworkLatencyResults {
            benchmark_id: self.benchmark_id.clone(),
            target_summaries,
            count: self.targets.values().map(|target| target.count).sum(),
            success_count: successful.len(),
            failure_count: self
                .targets
                .values()
                .map(|target| target.failure_count)
                .sum(),
            stats: stats(&successful),
            error_summary: self
                .errors
                .values()
                .map(|(error_details, count)| NetworkLatencyErrorDetailsCount {
                    error_details: error_details.clone(),
                    count: *count,
                })
                .collect(),
        }
    }
}

impl Accumulator<NetworkLatencySample> for NetworkLatencyAccumulator {
    type Summary = NetworkLatencyResults;
    type MergeError = NetworkLatencyMergeError;

    fn process_record(&mut self, record: &NetworkLatencySample) {
        self.add_sample(record.clone());
    }

    fn query_time_range(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.samples
            .iter()
            .map(|sample| sample.timestamp_ns >= start_ns && sample.timestamp_ns < end_ns)
            .collect()
    }

    fn export_results(&self, context: &ExportContext) -> Self::Summary {
        let include_phase = context.phase.is_none_or(|phase| phase == Phase::Profiling);
        let mut filtered = Self::new(self.benchmark_id.clone());
        if include_phase {
            for sample in &self.samples {
                if context
                    .start_ns
                    .is_none_or(|start_ns| sample.timestamp_ns >= start_ns)
                    && context
                        .end_ns
                        .is_none_or(|end_ns| sample.timestamp_ns < end_ns)
                {
                    // Aggregate the borrowed sample directly; the summary reads
                    // only the target/error tallies, so no sample clone or
                    // samples-Vec growth is needed on this hot export path.
                    Self::tally(&mut filtered.targets, &mut filtered.errors, sample);
                }
            }
        }
        NetworkLatencyAccumulator::export_results(&filtered)
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        if let (Some(existing), Some(incoming)) = (&self.benchmark_id, &other.benchmark_id)
            && existing != incoming
        {
            return Err(NetworkLatencyMergeError::BenchmarkIdConflict {
                existing: existing.clone(),
                incoming: incoming.clone(),
            });
        }
        if self.benchmark_id.is_none() {
            self.benchmark_id.clone_from(&other.benchmark_id);
        }
        for sample in &other.samples {
            self.add_sample(sample.clone());
        }
        Ok(())
    }
}

fn stats(values: &[i64]) -> NetworkLatencyStats {
    if values.is_empty() {
        return NetworkLatencyStats::default();
    }
    let mut sorted = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / count;
    let variance = sorted
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    NetworkLatencyStats {
        min_ns: sorted.first().copied(),
        mean_ns: Some(mean),
        median_ns: Some(linear_percentile(&sorted, 50.0)),
        p90_ns: Some(linear_percentile(&sorted, 90.0)),
        p99_ns: Some(linear_percentile(&sorted, 99.0)),
        stddev_ns: Some(variance.sqrt()),
    }
}

fn linear_percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = percentile / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(rtt_ns: Option<i64>, success: bool) -> NetworkLatencySample {
        NetworkLatencySample {
            timestamp_ns: 1,
            target_url: "http://localhost:8000/v1".to_string(),
            target_host: "localhost".to_string(),
            target_port: 8000,
            probe_type: "tcp_connect",
            rtt_ns,
            success,
            error: (!success).then(|| NetworkLatencyErrorDetails {
                code: Some(111),
                error_type: Some("ConnectionRefusedError".to_string()),
                message: "refused".to_string(),
                cause: None,
                cause_chain: Some(vec!["ConnectionRefusedError".to_string()]),
            }),
        }
    }

    #[test]
    fn stats_match_numpy_linear_and_population_rules() {
        let mut accumulator = NetworkLatencyAccumulator::new(Some("benchmark".to_string()));
        for value in 1..=10 {
            accumulator.add_sample(sample(Some(value * 100), true));
        }
        accumulator.add_sample(sample(None, false));
        let results = accumulator.export_results();
        assert_eq!(results.success_count, 10);
        assert_eq!(results.failure_count, 1);
        assert_eq!(results.stats.mean_ns, Some(550.0));
        assert_eq!(results.stats.median_ns, Some(550.0));
        assert_eq!(results.stats.p90_ns, Some(910.0));
        assert_eq!(results.stats.p99_ns, Some(991.0));
        assert!((results.stats.stddev_ns.unwrap() - 287.228_132_326_901_46).abs() < 1e-9);
        assert_eq!(results.error_summary[0].count, 1);
    }

    #[test]
    fn empty_or_failed_only_accumulator_has_no_mean() {
        let mut accumulator = NetworkLatencyAccumulator::new(None);
        assert_eq!(accumulator.mean_rtt_ns(), None);
        accumulator.add_sample(sample(None, false));
        assert_eq!(accumulator.mean_rtt_ns(), None);
    }

    #[test]
    fn shared_accumulator_seam_filters_half_open_ranges_and_phase() {
        let mut first = sample(Some(100), true);
        first.timestamp_ns = 10;
        let mut second = sample(Some(300), true);
        second.timestamp_ns = 20;
        let mut accumulator = NetworkLatencyAccumulator::new(Some("benchmark".to_string()));
        accumulator.process_record(&first);
        accumulator.process_record(&second);

        assert_eq!(accumulator.query_time_range(10, 20), vec![true, false]);
        let range = Accumulator::export_results(&accumulator, &ExportContext::time_range(10, 20));
        assert_eq!(range.success_count, 1);
        assert_eq!(range.stats.mean_ns, Some(100.0));
        let warmup =
            Accumulator::export_results(&accumulator, &ExportContext::phase(Phase::Warmup));
        assert_eq!(warmup.count, 0);
    }

    #[test]
    fn shared_accumulator_merge_preserves_samples_and_rejects_run_mismatch() {
        let mut left = NetworkLatencyAccumulator::new(Some("benchmark".to_string()));
        left.add_sample(sample(Some(100), true));
        let mut right = NetworkLatencyAccumulator::new(Some("benchmark".to_string()));
        right.add_sample(sample(Some(300), true));

        left.merge(&right).unwrap();
        assert_eq!(left.export_results().success_count, 2);
        assert_eq!(left.mean_rtt_ns(), Some(200.0));

        let mismatch = NetworkLatencyAccumulator::new(Some("other".to_string()));
        assert_eq!(
            left.merge(&mismatch),
            Err(NetworkLatencyMergeError::BenchmarkIdConflict {
                existing: "benchmark".to_string(),
                incoming: "other".to_string(),
            })
        );
    }
}
