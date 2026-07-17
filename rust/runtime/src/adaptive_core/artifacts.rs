// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed adaptive events and summary artifacts.
//!
//! The schema carries event, candidate, and summary fields. File I/O is isolated
//! behind [`AdaptiveArtifactSink`].

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::metrics_core::linear_distribution;
use serde::Serialize as DeriveSerialize;
use serde::ser::{Serialize, Serializer};

use crate::adaptive_core::actuator::ControlSnapshot;
use crate::adaptive_core::controller::{
    AdaptiveStatus, CandidateObservation, ControllerEvent, ControllerEventKind, ControllerPhase,
    ControllerSnapshot,
};
use crate::adaptive_core::error::AdaptiveError;
use crate::adaptive_core::sla::{SlaFilter, SlaOp, SlaStat};
use crate::adaptive_core::step::StepPolicySnapshot;

/// Adaptive event and summary schema version.
pub const ADAPTIVE_SCHEMA_VERSION: u32 = 2;

/// Render a nanosecond instant carried on an adaptive event as an ISO-8601 UTC
/// string with microsecond precision and a trailing `Z`
/// (e.g. `2026-07-13T17:02:30.123456Z`).
///
/// The value is derived from the same `timestamp_ns` already recorded on the
/// event — never from an ambient `SystemTime::now()`/`Utc::now()` read — so the
/// rendered string and the raw nanosecond field stay on one timeline. Returns
/// `None` only when the instant falls outside the representable calendar range.
pub(crate) fn format_epoch_ns_utc(epoch_ns: i64) -> Option<String> {
    let secs = epoch_ns.div_euclid(1_000_000_000);
    let nanos = epoch_ns.rem_euclid(1_000_000_000) as u32;
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nanos)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string())
}

/// Artifact-safe numeric value; non-finite observations serialize as JSON
/// `null` instead of leaking NaN/Infinity across a report boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtifactValue(pub f64);

impl ArtifactValue {
    /// Return the finite inner value, or `None` for a non-finite observation.
    pub fn finite(self) -> Option<f64> {
        self.0.is_finite().then_some(self.0)
    }
}

impl Serialize for ArtifactValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.finite() {
            Some(value) => serializer.serialize_f64(value),
            None => serializer.serialize_none(),
        }
    }
}

/// Correlation fields shared by every adaptive event.
#[derive(Clone, Debug, Default)]
pub struct CorrelationContext {
    /// Benchmark run identifier.
    pub run_id: Option<String>,
    /// Stable phase identifier.
    pub phase_id: String,
    /// Human-readable phase name.
    pub phase_name: Option<String>,
    /// Optional wall-clock phase-start representation supplied by an exporter.
    pub phase_start_ts: Option<String>,
    /// Optional wall-clock phase-end representation supplied by an exporter.
    pub phase_end_ts: Option<String>,
    /// Optional fault-injection window identifier.
    pub fault_window_id: Option<String>,
}

/// One line in `adaptive_scale_events.jsonl`.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveEvent {
    /// Schema version.
    pub schema_version: u32,
    /// Clock timestamp retained for Python-schema compatibility.
    pub timestamp: i64,
    /// Clock timestamp in nanoseconds.
    pub timestamp_ns: i64,
    /// ISO-8601 UTC rendering (microsecond precision, trailing `Z`) derived
    /// from `timestamp_ns`; `None` only when that instant is uncalendarable.
    pub timestamp_utc: Option<String>,
    /// Event type.
    pub event: ControllerEventKind,
    /// Controller phase.
    pub phase: ControllerPhase,
    /// Controlled load variable.
    pub control_variable: &'static str,
    /// Control value before the event's decision.
    pub control_value_before: f64,
    /// Control value after the event's decision.
    pub control_value_after: f64,
    /// Alias for the post-decision value.
    pub control_value: f64,
    /// Target/actual control snapshot.
    pub control: ControlSnapshot,
    /// Conservative last-passing boundary.
    pub boundary_value: Option<f64>,
    /// Most recent passing value.
    pub last_passing_value: Option<f64>,
    /// First failing discovery value.
    pub first_failing_value: Option<f64>,
    /// Primary SLA metric tag.
    pub sla_metric: String,
    /// Primary SLA statistic.
    pub sla_stat: SlaStat,
    /// Primary SLA operator.
    pub sla_op: SlaOp,
    /// Primary observed SLA value.
    pub sla_value: Option<ArtifactValue>,
    /// Primary SLA threshold.
    pub sla_bound: f64,
    /// All observed SLA values.
    pub sla_values: BTreeMap<String, ArtifactValue>,
    /// Tightest normalized-margin SLA key.
    pub binding_sla: Option<String>,
    /// Successful request throughput for the window.
    pub throughput: f64,
    /// Successful request count.
    pub sample_count: usize,
    /// Successful request count under the exporter vocabulary.
    pub completed: usize,
    /// Returned request attempts in the window.
    pub sent: usize,
    /// In-flight request count when an issuer supplies it.
    pub in_flight: Option<usize>,
    /// Cancelled request count.
    pub cancelled: usize,
    /// Failed request count under the exporter vocabulary.
    pub errored: usize,
    /// Failed request count.
    pub error_count: usize,
    /// Window pass/fail/inconclusive tri-state.
    pub sla_passed: Option<bool>,
    /// Controller strategy name.
    pub strategy_type: &'static str,
    /// Step-policy name.
    pub step_policy: &'static str,
    /// Absolute control delta.
    pub step_size: Option<f64>,
    /// Human-readable event reason.
    pub reason: String,
    /// Benchmark run identifier.
    pub run_id: Option<String>,
    /// Stable phase identifier.
    pub phase_id: String,
    /// Human-readable phase name.
    pub phase_name: Option<String>,
    /// Optional phase-start correlation timestamp.
    pub phase_start_ts: Option<String>,
    /// Optional phase-end correlation timestamp.
    pub phase_end_ts: Option<String>,
    /// Adaptive iteration.
    pub adaptive_iteration: u64,
    /// Control value evaluated by the window/decision.
    pub candidate_value: f64,
    /// Control value accepted after the event.
    pub accepted_value: f64,
    /// Optional fault-window correlation identifier.
    pub fault_window_id: Option<String>,
}

impl AdaptiveEvent {
    pub(crate) fn from_controller(
        event: &ControllerEvent,
        timestamp_ns: i64,
        control_variable: &'static str,
        primary_sla: &SlaFilter,
        policy: &StepPolicySnapshot,
        correlation: &CorrelationContext,
        adaptive_iteration: u64,
    ) -> Self {
        Self {
            schema_version: ADAPTIVE_SCHEMA_VERSION,
            timestamp: timestamp_ns,
            timestamp_ns,
            timestamp_utc: format_epoch_ns_utc(timestamp_ns),
            event: event.kind,
            phase: event.phase,
            control_variable,
            control_value_before: event.control_before,
            control_value_after: event.control_after,
            control_value: event.control_after,
            control: event.control_snapshot.clone(),
            boundary_value: event.boundary_value,
            last_passing_value: event.last_passing_value,
            first_failing_value: event.first_failing_value,
            sla_metric: primary_sla.metric_tag.clone(),
            sla_stat: primary_sla.stat,
            sla_op: primary_sla.op,
            sla_value: event.sla_value.map(ArtifactValue),
            sla_bound: primary_sla.threshold,
            sla_values: event
                .sla_values
                .iter()
                .map(|(key, value)| (key.clone(), ArtifactValue(*value)))
                .collect(),
            binding_sla: event.binding_sla.clone(),
            throughput: event.throughput,
            sample_count: event.sample_count,
            completed: event.sample_count,
            sent: event.sample_count + event.error_count + event.cancelled_count,
            in_flight: None,
            cancelled: event.cancelled_count,
            errored: event.error_count,
            error_count: event.error_count,
            sla_passed: event.passed,
            strategy_type: "ramp_until_fail",
            step_policy: policy.name,
            step_size: event.step_size,
            reason: event.reason.clone(),
            run_id: correlation.run_id.clone(),
            phase_id: correlation.phase_id.clone(),
            phase_name: correlation.phase_name.clone(),
            phase_start_ts: correlation.phase_start_ts.clone(),
            phase_end_ts: correlation.phase_end_ts.clone(),
            adaptive_iteration,
            candidate_value: event.control_before,
            accepted_value: event.control_after,
            fault_window_id: correlation.fault_window_id.clone(),
        }
    }
}

/// Per-window candidate entry in the terminal summary.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveCandidate {
    /// Adaptive iteration.
    pub adaptive_iteration: u64,
    /// Control value assessed by the window.
    pub candidate_value: f64,
    /// Alias for the assessed control value.
    pub control_value: f64,
    /// Optional wall-clock start representation.
    pub start_ts: Option<String>,
    /// Optional wall-clock end representation.
    pub end_ts: Option<String>,
    /// Clock start in nanoseconds.
    pub start_ns: i64,
    /// Clock end in nanoseconds.
    pub end_ns: i64,
    /// Window duration in seconds.
    pub duration_s: f64,
    /// Returned attempt count.
    pub request_count: usize,
    /// Failed attempt count.
    pub error_count: usize,
    /// Cancelled attempt count.
    pub cancelled: usize,
    /// Successful attempt count.
    pub success_count: usize,
    /// Successful attempts divided by all returns.
    pub success_rate: f64,
    /// Completed-request throughput.
    pub throughput_rps: f64,
    /// Median request latency in milliseconds.
    pub latency_p50_ms: f64,
    /// P95 request latency in milliseconds.
    pub latency_p95_ms: f64,
    /// P99 request latency in milliseconds.
    pub latency_p99_ms: f64,
    /// Median TTFT in milliseconds.
    pub ttft_p50_ms: f64,
    /// P95 TTFT in milliseconds.
    pub ttft_p95_ms: f64,
    /// P99 TTFT in milliseconds.
    pub ttft_p99_ms: f64,
    /// Median per-request mean ITL in milliseconds.
    pub itl_p50_ms: f64,
    /// P95 per-request mean ITL in milliseconds.
    pub itl_p95_ms: f64,
    /// P99 per-request mean ITL in milliseconds.
    pub itl_p99_ms: f64,
    /// Whether every SLA filter passed.
    pub accepted: bool,
    /// Rejection category for a non-accepted candidate.
    pub rejection_reason: Option<&'static str>,
    /// Average request latency in milliseconds.
    pub latency_avg_ms: f64,
}

impl AdaptiveCandidate {
    pub(crate) fn from_observation(observation: CandidateObservation) -> Self {
        let latency = observation.stats.latency_samples();
        let ttft = observation.stats.ttft_samples();
        let itl = observation.stats.itl_samples();
        let request_count = observation.stats.total();
        let success_count = observation.stats.completed();
        Self {
            adaptive_iteration: observation.adaptive_iteration,
            candidate_value: observation.candidate_value,
            control_value: observation.candidate_value,
            start_ts: None,
            end_ts: None,
            start_ns: observation.stats.start_ns,
            end_ns: observation.stats.end_ns,
            duration_s: observation.stats.elapsed_sec,
            request_count,
            error_count: observation.stats.errors,
            cancelled: observation.stats.cancelled,
            success_count,
            success_rate: if request_count == 0 {
                0.0
            } else {
                success_count as f64 / request_count as f64
            },
            throughput_rps: observation.stats.throughput(),
            latency_p50_ms: percentile_ms(&latency, 50),
            latency_p95_ms: percentile_ms(&latency, 95),
            latency_p99_ms: percentile_ms(&latency, 99),
            ttft_p50_ms: percentile_ms(&ttft, 50),
            ttft_p95_ms: percentile_ms(&ttft, 95),
            ttft_p99_ms: percentile_ms(&ttft, 99),
            itl_p50_ms: percentile_ms(&itl, 50),
            itl_p95_ms: percentile_ms(&itl, 95),
            itl_p99_ms: percentile_ms(&itl, 99),
            accepted: observation.accepted,
            rejection_reason: observation.rejection_reason,
            latency_avg_ms: if latency.is_empty() {
                0.0
            } else {
                latency.iter().sum::<f64>() / latency.len() as f64 / 1_000_000.0
            },
        }
    }
}

fn percentile_ms(values_ns: &[f64], percentile: u32) -> f64 {
    if values_ns.is_empty() {
        return 0.0;
    }
    let Some(stats) = linear_distribution(
        "adaptive_candidate",
        values_ns.to_vec(),
        values_ns.iter().sum(),
        0,
    ) else {
        return 0.0;
    };
    stats
        .percentiles
        .get(&percentile)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        / 1_000_000.0
}

/// Conservative discovery result block.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveResult {
    /// Most recent passing control value.
    pub last_passing_value: Option<f64>,
    /// First failing discovery value.
    pub first_failing_value: Option<f64>,
    /// Held boundary value.
    pub boundary_value: Option<f64>,
}

/// Return totals from the terminal assessment window.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveTotals {
    /// All returned attempts.
    pub sent: usize,
    /// Successful requests.
    pub completed: usize,
    /// Failed requests.
    pub errored: usize,
    /// Cancelled requests.
    pub cancelled: usize,
}

/// Compact primary SLA block in the summary.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveSlaSummary {
    /// Primary metric tag.
    pub metric: String,
    /// Primary statistic.
    pub stat: SlaStat,
    /// Primary operator.
    pub op: SlaOp,
    /// Primary threshold.
    pub bound: f64,
}

/// Final `adaptive_scale_summary.json` payload.
#[derive(Clone, Debug, DeriveSerialize)]
pub struct AdaptiveSummary {
    /// Schema version.
    pub schema_version: u32,
    /// Completed/incomplete/failed classification.
    pub status: AdaptiveStatus,
    /// Controlled load variable.
    pub control_variable: &'static str,
    /// Final control value.
    pub control_value: f64,
    /// Final target/actual control snapshot.
    pub control: ControlSnapshot,
    /// Held boundary.
    pub boundary_value: Option<f64>,
    /// Most recent passing value.
    pub last_passing_value: Option<f64>,
    /// First failing discovery value.
    pub first_failing_value: Option<f64>,
    /// Nested discovery result.
    pub result: AdaptiveResult,
    /// Sustain-entry clock timestamp in nanoseconds.
    pub sustain_started_at: Option<i64>,
    /// Configured sustain duration in seconds.
    pub sustain_duration_seconds: f64,
    /// Idempotent terminal reason.
    pub completed_reason: String,
    /// Whether every conclusive sustain window passed.
    pub sla_passed_during_sustain: bool,
    /// Conclusive sustain window count.
    pub sustain_windows: usize,
    /// Passing sustain window count.
    pub sustain_passed_windows: usize,
    /// Primary SLA metric tag.
    pub sla_metric: String,
    /// Primary SLA statistic.
    pub sla_stat: SlaStat,
    /// Primary SLA operator.
    pub sla_op: SlaOp,
    /// Primary SLA threshold.
    pub sla_bound: f64,
    /// Nested primary SLA block.
    pub sla: AdaptiveSlaSummary,
    /// Terminal-window return totals.
    pub totals: AdaptiveTotals,
    /// Terminal-window throughput.
    pub throughput: f64,
    /// Per-window candidates.
    pub candidates: Vec<AdaptiveCandidate>,
    /// Controller strategy name.
    pub strategy_type: &'static str,
    /// Selected step policy.
    pub step_policy: &'static str,
    /// SLA-margin base step when selected.
    pub base_step: Option<usize>,
    /// SLA-margin maximum multiplier when selected.
    pub max_step_multiplier: Option<usize>,
    /// Fixed-percent value when selected.
    pub step_percent: Option<f64>,
}

impl AdaptiveSummary {
    pub(crate) fn from_terminal(
        snapshot: ControllerSnapshot,
        primary_sla: &SlaFilter,
        sustain_duration_ns: i64,
        terminal: &ControllerEvent,
        candidates: Vec<AdaptiveCandidate>,
    ) -> Result<Self, AdaptiveError> {
        let status = snapshot.status.ok_or_else(|| {
            AdaptiveError::Artifact("terminal summary requested before completion".to_string())
        })?;
        let completed_reason = snapshot.completed_reason.clone().ok_or_else(|| {
            AdaptiveError::Artifact("terminal summary has no completed reason".to_string())
        })?;
        Ok(Self {
            schema_version: ADAPTIVE_SCHEMA_VERSION,
            status,
            control_variable: snapshot.control_variable,
            control_value: snapshot.control_value,
            control: snapshot.control,
            boundary_value: snapshot.boundary_value,
            last_passing_value: snapshot.last_passing_value,
            first_failing_value: snapshot.first_failing_value,
            result: AdaptiveResult {
                last_passing_value: snapshot.last_passing_value,
                first_failing_value: snapshot.first_failing_value,
                boundary_value: snapshot.boundary_value,
            },
            sustain_started_at: snapshot.sustain_started_at_ns,
            sustain_duration_seconds: sustain_duration_ns as f64 / 1_000_000_000.0,
            completed_reason,
            sla_passed_during_sustain: snapshot.sustain_windows > 0
                && snapshot.sustain_passed_windows == snapshot.sustain_windows,
            sustain_windows: snapshot.sustain_windows,
            sustain_passed_windows: snapshot.sustain_passed_windows,
            sla_metric: primary_sla.metric_tag.clone(),
            sla_stat: primary_sla.stat,
            sla_op: primary_sla.op,
            sla_bound: primary_sla.threshold,
            sla: AdaptiveSlaSummary {
                metric: primary_sla.metric_tag.clone(),
                stat: primary_sla.stat,
                op: primary_sla.op,
                bound: primary_sla.threshold,
            },
            totals: AdaptiveTotals {
                sent: terminal.sample_count + terminal.error_count + terminal.cancelled_count,
                completed: terminal.sample_count,
                errored: terminal.error_count,
                cancelled: terminal.cancelled_count,
            },
            throughput: terminal.throughput,
            candidates,
            strategy_type: "ramp_until_fail",
            step_policy: snapshot.step_policy.name,
            base_step: snapshot.step_policy.base_step,
            max_step_multiplier: snapshot.step_policy.max_step_multiplier,
            step_percent: snapshot.step_policy.step_percent,
        })
    }
}

/// Sink for typed adaptive events and the idempotent terminal summary.
pub trait AdaptiveArtifactSink {
    /// Append one adaptive event.
    fn emit_event(&mut self, event: &AdaptiveEvent) -> Result<(), AdaptiveError>;
    /// Write the final adaptive summary once.
    fn write_summary(&mut self, summary: &AdaptiveSummary) -> Result<(), AdaptiveError>;
}

/// JSONL + JSON file implementation of [`AdaptiveArtifactSink`].
pub struct FileArtifactSink {
    event_path: PathBuf,
    summary_path: PathBuf,
}

impl FileArtifactSink {
    /// Create/truncate the schema-v2 files beneath `artifact_dir`.
    pub fn new(artifact_dir: impl AsRef<Path>) -> Result<Self, AdaptiveError> {
        std::fs::create_dir_all(artifact_dir.as_ref())?;
        let event_path = artifact_dir.as_ref().join("adaptive_scale_events.jsonl");
        let summary_path = artifact_dir.as_ref().join("adaptive_scale_summary.json");
        File::create(&event_path)?;
        if summary_path.exists() {
            std::fs::remove_file(&summary_path)?;
        }
        Ok(Self {
            event_path,
            summary_path,
        })
    }

    /// Path of the JSONL event stream.
    pub fn event_path(&self) -> &Path {
        &self.event_path
    }

    /// Path of the terminal JSON summary.
    pub fn summary_path(&self) -> &Path {
        &self.summary_path
    }
}

impl AdaptiveArtifactSink for FileArtifactSink {
    fn emit_event(&mut self, event: &AdaptiveEvent) -> Result<(), AdaptiveError> {
        let value = sorted_json_value(serde_json::to_value(event)?);
        let encoded = serde_json::to_vec(&value)?;
        let mut file = OpenOptions::new().append(true).open(&self.event_path)?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
        Ok(())
    }

    fn write_summary(&mut self, summary: &AdaptiveSummary) -> Result<(), AdaptiveError> {
        let value = sorted_json_value(serde_json::to_value(summary)?);
        let mut encoded = serde_json::to_vec_pretty(&value)?;
        encoded.push(b'\n');
        std::fs::write(&self.summary_path, encoded)?;
        Ok(())
    }
}

fn sorted_json_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sorted_json_value).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::with_capacity(entries.len());
            for (key, value) in entries {
                sorted.insert(key, sorted_json_value(value));
            }
            serde_json::Value::Object(sorted)
        }
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_core::window::{RequestSample, WindowStats};

    #[test]
    fn epoch_ns_renders_iso8601_microsecond_utc_with_trailing_z() {
        // A fixed Unix-epoch nanosecond instant; the render truncates to
        // microsecond precision and appends `Z`.
        let epoch_ns: i64 = 1_784_048_550_123_456_789;
        let rendered = format_epoch_ns_utc(epoch_ns).expect("representable instant");
        assert!(
            regex_like_iso8601_micros(&rendered),
            "unexpected render: {rendered}"
        );
        assert_eq!(rendered, "2026-07-14T17:02:30.123456Z");
    }

    /// Minimal pure check matching the integration contract's
    /// `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z` shape without a regex dep.
    fn regex_like_iso8601_micros(s: &str) -> bool {
        let bytes = s.as_bytes();
        if bytes.len() != 27 {
            return false;
        }
        let digit = |i: usize| bytes[i].is_ascii_digit();
        let sep = |i: usize, c: u8| bytes[i] == c;
        (0..4).all(digit)
            && sep(4, b'-')
            && (5..7).all(digit)
            && sep(7, b'-')
            && (8..10).all(digit)
            && sep(10, b'T')
            && (11..13).all(digit)
            && sep(13, b':')
            && (14..16).all(digit)
            && sep(16, b':')
            && (17..19).all(digit)
            && sep(19, b'.')
            && (20..26).all(digit)
            && sep(26, b'Z')
    }

    #[test]
    fn artifact_value_scrubs_non_finite_numbers_to_null() {
        assert_eq!(serde_json::to_string(&ArtifactValue(1.5)).unwrap(), "1.5");
        assert_eq!(
            serde_json::to_string(&ArtifactValue(f64::INFINITY)).unwrap(),
            "null"
        );
    }

    #[test]
    fn candidate_uses_linear_percentiles_and_return_success_rate() {
        let observation = CandidateObservation {
            adaptive_iteration: 1,
            candidate_value: 4.0,
            stats: WindowStats {
                successful_requests: vec![
                    RequestSample {
                        request_latency_ns: 10_000_000,
                        ttft_ns: None,
                        inter_token_latency_ns: None,
                        output_sequence_length: None,
                    },
                    RequestSample {
                        request_latency_ns: 20_000_000,
                        ttft_ns: None,
                        inter_token_latency_ns: None,
                        output_sequence_length: None,
                    },
                ],
                errors: 1,
                cancelled: 1,
                elapsed_sec: 2.0,
                start_ns: 0,
                end_ns: 2_000_000_000,
            },
            accepted: true,
            rejection_reason: None,
        };
        let candidate = AdaptiveCandidate::from_observation(observation);
        assert_eq!(candidate.success_rate, 0.5);
        assert_eq!(candidate.latency_p50_ms, 15.0);
        assert_eq!(candidate.throughput_rps, 1.0);
    }

    #[test]
    fn artifact_json_keys_are_sorted_recursively() {
        let value = serde_json::json!({
            "z": {"last": 1, "first": 2},
            "a": [{"right": 3, "left": 4}],
        });
        let encoded = serde_json::to_string(&sorted_json_value(value)).unwrap();
        assert_eq!(
            encoded,
            r#"{"a":[{"left":4,"right":3}],"z":{"first":2,"last":1}}"#
        );
    }
}
