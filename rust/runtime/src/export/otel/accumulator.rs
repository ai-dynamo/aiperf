// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-record GenAI-semconv histogram accumulator.
//!
//! This is the native analogue of the Python per-record OTel path
//! (`strategies/metric_results.py::MetricResultsStrategy.process`,
//! `strategies/genai_semconv.py`, `otel_metrics_results_processor.py`). The
//! aggregate report cannot reconstruct a per-bucket distribution from
//! avg/min/max/percentiles. This accumulator receives the exact per-request metric
//! projection the runner
//! computes for each captured record (the same projection the live-streaming
//! sink forwards to Python), buckets each observation into the semconv explicit
//! histograms, and is merged at run end. The sink emits populated
//! `bucket_counts` (+ `count`/`sum`/`min`/`max`) that a collector aggregating
//! Python's per-record stream would compute.
//!
//! # Parity contract
//! - Each duration observation is converted to seconds exactly as Python's
//!   `convert_metric_value` (`_ns_to_s`) does. The runner projects display units
//!   (`ms` for latency), so the net `ms -> s` conversion is byte-equivalent to
//!   Python's `ns -> s` on the same underlying nanoseconds
//!   (`genai_semconv.py:60-76`, `metric_results.py:57-59`).
//! - Token counts are identity-scaled (`genai_semconv._identity`).
//! - Duration histograms are keyed by the spec `error.type` attribute (absent on
//!   success); token histograms carry only `gen_ai.token.type`, mirroring
//!   `_build_duration_attributes` vs `_build_token_usage_attributes`
//!   (`genai_semconv.py:303-330`).
//! - Bucket selection uses OTLP le-semantics: an observation `v` lands in the
//!   first bucket whose upper bound satisfies `v <= bound`, else the overflow
//!   bucket. The buckets vector length is `bounds.len() + 1` (OTLP invariant).
//!
//! Only successful, in-scope records reach a mapped metric: errored/cancelled
//! requests project `error_*` metrics that are not in the semconv map, so they
//! never contribute to these histograms (matching the aggregate report's
//! success-only distributions).

use std::collections::BTreeMap;

use super::{
    DURATION_BOUNDS, INPUT_TOKEN_KEYS, OUTPUT_TOKEN_KEYS, TIME_PER_OUTPUT_CHUNK_BOUNDS,
    TOKEN_USAGE_BOUNDS, TTFT_BOUNDS, seconds_scale,
};

/// The three GenAI duration histograms, discriminated on the record's spec
/// `error.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurationKind {
    /// `request_latency` -> `gen_ai.client.operation.duration`.
    RequestLatency,
    /// `time_to_first_token` -> `gen_ai.client.operation.time_to_first_chunk`.
    TimeToFirstToken,
    /// `inter_token_latency` -> `gen_ai.client.operation.time_per_output_chunk`.
    InterTokenLatency,
}

/// The two token-usage directions merged into `gen_ai.client.token.usage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenKind {
    /// `gen_ai.token.type=input`.
    Input,
    /// `gen_ai.token.type=output`.
    Output,
}

/// One explicit-bucket histogram accumulated from per-record observations.
///
/// `bucket_counts.len()` is always `bounds.len() + 1` (OTLP invariant). `min`/
/// `max` are `None` until the first observation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BucketHistogram {
    bounds: &'static [f64],
    bucket_counts: Vec<u64>,
    count: u64,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl BucketHistogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            bucket_counts: vec![0; bounds.len() + 1],
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
        }
    }

    /// Bucket one finite observation with OTLP le-semantics. Non-finite values
    /// are dropped (the runner only projects finite metric values, so this is a
    /// defensive guard, not an expected path).
    fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let index = self
            .bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(self.bounds.len());
        self.bucket_counts[index] += 1;
        self.count += 1;
        self.sum += value;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }

    /// Merge a per-worker histogram into this one. Bounds are identical by
    /// construction (same spec metric), so buckets add positionally.
    fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.bounds, other.bounds, "merging incompatible histograms");
        for (accumulated, incoming) in self.bucket_counts.iter_mut().zip(&other.bucket_counts) {
            *accumulated += incoming;
        }
        self.count += other.count;
        self.sum += other.sum;
        if let Some(other_min) = other.min {
            self.min = Some(self.min.map_or(other_min, |current| current.min(other_min)));
        }
        if let Some(other_max) = other.max {
            self.max = Some(self.max.map_or(other_max, |current| current.max(other_max)));
        }
    }

    /// Explicit bucket boundaries (borrowed static spec constant).
    pub(crate) fn bounds(&self) -> &'static [f64] {
        self.bounds
    }

    /// Per-bucket counts (`bounds.len() + 1`).
    pub(crate) fn bucket_counts(&self) -> &[u64] {
        &self.bucket_counts
    }

    /// Total observation count.
    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    /// Sum of observations (seconds for durations, tokens for usage).
    pub(crate) fn sum(&self) -> f64 {
        self.sum
    }

    /// Minimum observation, if any.
    pub(crate) fn min(&self) -> Option<f64> {
        self.min
    }

    /// Maximum observation, if any.
    pub(crate) fn max(&self) -> Option<f64> {
        self.max
    }
}

/// Per-record GenAI-semconv histograms, filled during execution and merged at
/// run end. Consumed by [`super::OtelExporter`] to emit populated
/// `bucket_counts`.
///
/// Duration histograms are keyed by the optional spec `error.type`; token
/// histograms carry only their direction. Empty until the first observation so
/// a run that records no semconv metric emits nothing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OtelRecordAccumulator {
    request_latency: BTreeMap<Option<String>, BucketHistogram>,
    time_to_first_token: BTreeMap<Option<String>, BucketHistogram>,
    inter_token_latency: BTreeMap<Option<String>, BucketHistogram>,
    token_input: Option<BucketHistogram>,
    token_output: Option<BucketHistogram>,
}

impl OtelRecordAccumulator {
    /// A fresh empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one record's projected per-request metrics.
    ///
    /// `metrics` maps the aiperf metric name to `(display_value, unit)` — the
    /// exact projection the runner computes for the record (and forwards to the
    /// Python live-streaming sink). `error_type` is the record's spec
    /// `error.type` (`None` on success), applied to the duration histograms
    /// only. Token direction is selected first-present across the direction's
    /// key aliases, matching the aggregate sink's key resolution.
    pub fn observe_record(
        &mut self,
        metrics: &BTreeMap<&str, (f64, &str)>,
        error_type: Option<&str>,
    ) {
        for (kind, report_key, bounds) in [
            (
                DurationKind::RequestLatency,
                "request_latency",
                DURATION_BOUNDS,
            ),
            (
                DurationKind::TimeToFirstToken,
                "time_to_first_token",
                TTFT_BOUNDS,
            ),
            (
                DurationKind::InterTokenLatency,
                "inter_token_latency",
                TIME_PER_OUTPUT_CHUNK_BOUNDS,
            ),
        ] {
            if let Some((value, unit)) = metrics.get(report_key) {
                let seconds = value * seconds_scale(unit);
                self.duration_map(kind)
                    .entry(error_type.map(str::to_string))
                    .or_insert_with(|| BucketHistogram::new(bounds))
                    .observe(seconds);
            }
        }

        if let Some((value, _unit)) = INPUT_TOKEN_KEYS.iter().find_map(|key| metrics.get(key)) {
            self.token_input
                .get_or_insert_with(|| BucketHistogram::new(TOKEN_USAGE_BOUNDS))
                .observe(*value);
        }
        if let Some((value, _unit)) = OUTPUT_TOKEN_KEYS.iter().find_map(|key| metrics.get(key)) {
            self.token_output
                .get_or_insert_with(|| BucketHistogram::new(TOKEN_USAGE_BOUNDS))
                .observe(*value);
        }
    }

    /// Merge a per-worker accumulator into this one (thread-per-core join).
    pub fn merge(&mut self, other: &Self) {
        merge_keyed(&mut self.request_latency, &other.request_latency);
        merge_keyed(&mut self.time_to_first_token, &other.time_to_first_token);
        merge_keyed(&mut self.inter_token_latency, &other.inter_token_latency);
        merge_option(&mut self.token_input, other.token_input.as_ref());
        merge_option(&mut self.token_output, other.token_output.as_ref());
    }

    /// Whether any observation was recorded.
    pub fn is_empty(&self) -> bool {
        self.request_latency.is_empty()
            && self.time_to_first_token.is_empty()
            && self.inter_token_latency.is_empty()
            && self.token_input.is_none()
            && self.token_output.is_none()
    }

    /// Duration histograms for `kind`, keyed by optional spec `error.type`.
    pub(crate) fn duration_series(
        &self,
        kind: DurationKind,
    ) -> impl Iterator<Item = (Option<&str>, &BucketHistogram)> {
        self.duration_ref(kind)
            .iter()
            .map(|(error_type, histogram)| (error_type.as_deref(), histogram))
    }

    /// The token-usage histogram for `kind`, if any observation exists.
    pub(crate) fn token_series(&self, kind: TokenKind) -> Option<&BucketHistogram> {
        match kind {
            TokenKind::Input => self.token_input.as_ref(),
            TokenKind::Output => self.token_output.as_ref(),
        }
    }

    fn duration_map(
        &mut self,
        kind: DurationKind,
    ) -> &mut BTreeMap<Option<String>, BucketHistogram> {
        match kind {
            DurationKind::RequestLatency => &mut self.request_latency,
            DurationKind::TimeToFirstToken => &mut self.time_to_first_token,
            DurationKind::InterTokenLatency => &mut self.inter_token_latency,
        }
    }

    fn duration_ref(&self, kind: DurationKind) -> &BTreeMap<Option<String>, BucketHistogram> {
        match kind {
            DurationKind::RequestLatency => &self.request_latency,
            DurationKind::TimeToFirstToken => &self.time_to_first_token,
            DurationKind::InterTokenLatency => &self.inter_token_latency,
        }
    }
}

/// Classify a record's terminal error into the spec `error.type` value.
///
/// Matches `genai_semconv._classify_error_type` (`genai_semconv.py:254-295`) over
/// the HTTP or pseudo status code, transport error type, and message. Cause-chain
/// details are unavailable, so timeout and cancellation heuristics scan the message.
/// Called only when the record actually errored.
pub fn classify_spec_error_type(code: Option<u16>, type_name: &str, message: &str) -> String {
    if let Some(code) = code {
        if (500..=599).contains(&code) {
            return "http_5xx".to_string();
        }
        if (400..=499).contains(&code) {
            return "http_4xx".to_string();
        }
    }
    match type_name {
        "timeout" | "asyncio.TimeoutError" | "TimeoutError" => return "timeout".to_string(),
        "cancelled" => return "cancelled".to_string(),
        _ => {}
    }
    let lowered = message.to_ascii_lowercase();
    if lowered.contains("timeout") {
        return "timeout".to_string();
    }
    if lowered.contains("cancel") {
        return "cancelled".to_string();
    }
    if lowered.contains("json") && (lowered.contains("parse") || lowered.contains("decode")) {
        return "parse_error".to_string();
    }
    "_OTHER".to_string()
}

fn merge_keyed(
    into: &mut BTreeMap<Option<String>, BucketHistogram>,
    from: &BTreeMap<Option<String>, BucketHistogram>,
) {
    for (key, histogram) in from {
        match into.get_mut(key) {
            Some(existing) => existing.merge(histogram),
            None => {
                into.insert(key.clone(), histogram.clone());
            }
        }
    }
}

fn merge_option(into: &mut Option<BucketHistogram>, from: Option<&BucketHistogram>) {
    if let Some(from) = from {
        match into {
            Some(existing) => existing.merge(from),
            None => *into = Some(from.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(pairs: &'a [(&'a str, f64, &'a str)]) -> BTreeMap<&'a str, (f64, &'a str)> {
        pairs
            .iter()
            .map(|(name, value, unit)| (*name, (*value, *unit)))
            .collect()
    }

    #[test]
    fn buckets_use_le_semantics_and_overflow() {
        let mut histogram = BucketHistogram::new(DURATION_BOUNDS);
        // 0.01 lands in bucket 0 (v <= bounds[0]); 0.02 in bucket 1; a huge
        // value in the overflow bucket.
        histogram.observe(0.01);
        histogram.observe(0.02);
        histogram.observe(1000.0);
        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.bucket_counts()[0], 1);
        assert_eq!(histogram.bucket_counts()[1], 1);
        assert_eq!(*histogram.bucket_counts().last().unwrap(), 1);
        let total: u64 = histogram.bucket_counts().iter().sum();
        assert_eq!(total, histogram.count());
    }

    #[test]
    fn observe_record_converts_units_and_splits_tokens() {
        let mut accumulator = OtelRecordAccumulator::new();
        accumulator.observe_record(
            &lookup(&[
                ("request_latency", 320.0, "ms"),
                ("time_to_first_token", 40.0, "ms"),
                ("inter_token_latency", 20.0, "ms"),
                ("input_sequence_length", 128.0, "tokens"),
                ("output_token_count", 64.0, "tokens"),
            ]),
            None,
        );

        let duration = accumulator
            .duration_series(DurationKind::RequestLatency)
            .find(|(error_type, _)| error_type.is_none())
            .expect("request latency series")
            .1;
        // 320 ms -> 0.32 s exactly, one observation.
        assert_eq!(duration.count(), 1);
        assert_eq!(duration.sum(), 0.32);
        // 0.32 <= bounds[6] (0.64) and > bounds[5] (0.32)? 0.32 <= 0.32 -> bucket 5.
        assert_eq!(duration.bucket_counts()[5], 1);

        let input = accumulator.token_series(TokenKind::Input).unwrap();
        assert_eq!(input.count(), 1);
        assert_eq!(input.sum(), 128.0);
        let output = accumulator.token_series(TokenKind::Output).unwrap();
        assert_eq!(output.sum(), 64.0);
    }

    #[test]
    fn error_type_keys_duration_series_and_merge_sums() {
        let mut worker_a = OtelRecordAccumulator::new();
        worker_a.observe_record(&lookup(&[("request_latency", 100.0, "ms")]), None);
        worker_a.observe_record(
            &lookup(&[("request_latency", 200.0, "ms")]),
            Some("http_5xx"),
        );

        let mut worker_b = OtelRecordAccumulator::new();
        worker_b.observe_record(&lookup(&[("request_latency", 300.0, "ms")]), None);

        worker_a.merge(&worker_b);

        let success = worker_a
            .duration_series(DurationKind::RequestLatency)
            .find(|(error_type, _)| error_type.is_none())
            .unwrap()
            .1;
        assert_eq!(success.count(), 2);
        assert_eq!(success.sum(), 0.1 + 0.3);

        let errored = worker_a
            .duration_series(DurationKind::RequestLatency)
            .find(|(error_type, _)| *error_type == Some("http_5xx"))
            .unwrap()
            .1;
        assert_eq!(errored.count(), 1);
    }

    #[test]
    fn classify_spec_error_type_matches_python_precedence() {
        assert_eq!(
            classify_spec_error_type(Some(503), "HttpError", ""),
            "http_5xx"
        );
        assert_eq!(
            classify_spec_error_type(Some(499), "RequestCancellationError", "cancelled"),
            "http_4xx"
        );
        assert_eq!(
            classify_spec_error_type(None, "TimeoutError", ""),
            "timeout"
        );
        assert_eq!(
            classify_spec_error_type(None, "X", "connection cancel"),
            "cancelled"
        );
        assert_eq!(
            classify_spec_error_type(None, "X", "failed to parse json body"),
            "parse_error"
        );
        assert_eq!(classify_spec_error_type(None, "X", "boom"), "_OTHER");
    }
}
