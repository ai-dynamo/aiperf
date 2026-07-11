// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-side translation from observer events into native metric records.
//!
//! The metrics crate stays transport-neutral: this adapter owns the request-event
//! join because the runtime knows UUIDs, terminal state, usage, and the injected
//! [`aiperf_clock::Clock`] timeline. It is single-loop `Rc`/`RefCell` state and
//! performs one append-only accumulator pass after all request tasks drain.

use std::cell::RefCell;
use std::rc::Rc;

use aiperf_clock::Clock;
use aiperf_metrics::{
    AccumulatorSummary, HttpTrace, MetricsAccumulator, MetricsConfig, Phase, RecordIngest,
    TokenCounts, UsageMetrics,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedTokenKind, ObservedUsage, RequestObserver};
use rustc_hash::FxHashMap;
use uuid::Uuid;

/// Optional workload dimensions registered before request arrival.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestMetricMetadata {
    /// Authoritative phase for summary masking.
    pub phase: Phase,
    /// Session sequence number, when the workload assigns one.
    pub session_num: Option<u64>,
    /// Zero-based turn index within a session.
    pub turn_index: u32,
    /// Worker identity for per-worker series.
    pub worker_id: Option<String>,
    /// Conversation identity for multi-turn series.
    pub conversation_id: Option<String>,
    /// External request correlation id.
    pub correlation_id: Option<String>,
    /// Source audio duration for ASR real-time-factor metrics.
    pub audio_duration_s: Option<f64>,
    /// Whether arrival represents a policy credit whose queue/effective latency
    /// should be reported. Fixed schedules have no credit-issuance phase.
    pub has_credit_timestamp: bool,
}

impl Default for RequestMetricMetadata {
    fn default() -> Self {
        Self {
            phase: Phase::Profiling,
            session_num: None,
            turn_index: 0,
            worker_id: None,
            conversation_id: None,
            correlation_id: None,
            audio_duration_s: None,
            has_credit_timestamp: true,
        }
    }
}

/// Transport facts available after one request reaches terminal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeResponseMetadata {
    /// Clock timestamp when transport dispatch began.
    pub start_ns: Option<i64>,
    /// Clock timestamp when transport dispatch ended.
    pub end_ns: Option<i64>,
    /// Authoritative server-reported prompt tokens.
    pub prompt_tokens: Option<u64>,
    /// Authoritative server-reported completion tokens.
    pub completion_tokens: Option<u64>,
    /// Fine-grained transport timings and byte/chunk counters.
    pub http: HttpTrace,
}

#[derive(Debug)]
struct PendingRequest {
    credit_issued_ns: i64,
    dispatch_start_ns: Option<i64>,
    terminal_ns: Option<i64>,
    response: NativeResponseMetadata,
    input_tokens: u64,
    requested_output_tokens: u64,
    token_arrivals_ns: Vec<i64>,
    output_tokens: u64,
    reasoning_tokens: u64,
    first_output_token_ns: Option<i64>,
    terminal: Option<ReplayTerminalStatus>,
    metadata: RequestMetricMetadata,
}

#[derive(Debug, Default)]
struct ObserverState {
    requests: FxHashMap<Uuid, PendingRequest>,
    order: Vec<Uuid>,
    metadata: FxHashMap<Uuid, RequestMetricMetadata>,
}

/// Observer-backed native metrics collector sharing the runtime's clock origin.
pub struct NativeMetricsObserver {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    state: RefCell<ObserverState>,
    accumulator: RefCell<MetricsAccumulator>,
}

impl NativeMetricsObserver {
    /// Creates an observer with explicit accumulator configuration.
    pub fn new(clock: Rc<dyn Clock>, origin_ns: i64, config: MetricsConfig) -> Self {
        Self {
            clock,
            origin_ns,
            state: RefCell::new(ObserverState::default()),
            accumulator: RefCell::new(MetricsAccumulator::with_config(config)),
        }
    }

    /// Registers workload dimensions before or after the arrival callback.
    pub fn register_metadata(&self, uuid: Uuid, metadata: RequestMetricMetadata) {
        let mut state = self.state.borrow_mut();
        if let Some(request) = state.requests.get_mut(&uuid) {
            request.metadata = metadata;
        } else {
            state.metadata.insert(uuid, metadata);
        }
    }

    /// Adds transport start/end timestamps and authoritative endpoint usage.
    pub fn record_response(&self, uuid: Uuid, mut response: NativeResponseMetadata) {
        response.start_ns = response
            .start_ns
            .map(|timestamp| self.relative_absolute_ns(timestamp));
        response.end_ns = response
            .end_ns
            .map(|timestamp| self.relative_absolute_ns(timestamp));
        if let Some(request) = self.state.borrow_mut().requests.get_mut(&uuid) {
            response.prompt_tokens = response.prompt_tokens.or(request.response.prompt_tokens);
            response.completion_tokens = response
                .completion_tokens
                .or(request.response.completion_tokens);
            request.response = response;
        }
    }

    /// Finalizes every retained request and returns the full native summary.
    ///
    /// Request rows are appended in arrival order, independent of hash-map order.
    pub fn finish(&self) -> AccumulatorSummary {
        let finish_ns = self.relative_now_ns();
        let mut state = std::mem::take(&mut *self.state.borrow_mut());
        let mut accumulator = std::mem::take(&mut *self.accumulator.borrow_mut());
        for (ordinal, uuid) in state.order.into_iter().enumerate() {
            let Some(request) = state.requests.remove(&uuid) else {
                continue;
            };
            accumulator.process_record(&request.into_record(uuid, ordinal as u64, finish_ns));
        }
        accumulator.summarize()
    }

    fn relative_now_ns(&self) -> i64 {
        self.clock.now_ns().saturating_sub(self.origin_ns)
    }

    fn relative_ns_from_ms(&self, milliseconds: f64) -> i64 {
        if !milliseconds.is_finite() {
            return self.relative_now_ns();
        }
        (milliseconds * 1_000_000.0).round_ties_even() as i64
    }

    fn relative_absolute_ns(&self, timestamp_ns: i64) -> i64 {
        timestamp_ns.saturating_sub(self.origin_ns)
    }
}

impl PendingRequest {
    fn into_record(self, uuid: Uuid, ordinal: u64, finish_ns: i64) -> RecordIngest {
        let start_ns = self
            .response
            .start_ns
            .or(self.dispatch_start_ns)
            .unwrap_or(self.credit_issued_ns);
        let end_ns = self
            .response
            .end_ns
            .or(self.terminal_ns)
            .unwrap_or(finish_ns);
        let terminal = self.terminal.unwrap_or(ReplayTerminalStatus::Failed);
        let completion_tokens = self.response.completion_tokens;
        let prompt_tokens = self.response.prompt_tokens;
        RecordIngest {
            correlation_id: self
                .metadata
                .correlation_id
                .unwrap_or_else(|| uuid.to_string()),
            session_num: self.metadata.session_num.unwrap_or(ordinal),
            turn_index: self.metadata.turn_index,
            worker_id: self.metadata.worker_id,
            conversation_id: self.metadata.conversation_id,
            phase: self.metadata.phase,
            start_ns,
            end_ns,
            admit_ns: self
                .metadata
                .has_credit_timestamp
                .then_some(self.credit_issued_ns),
            first_token_ns: self.token_arrivals_ns.first().copied(),
            second_token_ns: self.token_arrivals_ns.get(1).copied(),
            first_output_token_ns: self.first_output_token_ns,
            token_arrival_ns: self.token_arrivals_ns,
            errored: matches!(
                terminal,
                ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed
            ),
            canceled: terminal == ReplayTerminalStatus::Canceled,
            tokens: TokenCounts {
                input: Some(self.input_tokens),
                output: Some(self.output_tokens),
                reasoning: (self.reasoning_tokens > 0).then_some(self.reasoning_tokens),
                requested_output: Some(self.requested_output_tokens),
            },
            usage: UsageMetrics {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens
                    .zip(completion_tokens)
                    .map(|(prompt, completion)| prompt.saturating_add(completion)),
                ..UsageMetrics::default()
            },
            http: self.response.http,
            audio_duration_s: self.metadata.audio_duration_s,
            num_images: None,
            video_inference_seconds: None,
            video_peak_memory_mb: None,
            metric_overrides: Vec::new(),
        }
    }
}

impl RequestObserver for NativeMetricsObserver {
    fn on_arrival(
        &self,
        uuid: Uuid,
        arrival_ms: f64,
        input_length: usize,
        requested_output_length: usize,
    ) {
        let mut state = self.state.borrow_mut();
        let metadata = state.metadata.remove(&uuid).unwrap_or_default();
        if state.requests.contains_key(&uuid) {
            return;
        }
        state.order.push(uuid);
        state.requests.insert(
            uuid,
            PendingRequest {
                credit_issued_ns: self.relative_ns_from_ms(arrival_ms),
                dispatch_start_ns: None,
                terminal_ns: None,
                response: NativeResponseMetadata::default(),
                input_tokens: input_length as u64,
                requested_output_tokens: requested_output_length as u64,
                token_arrivals_ns: Vec::with_capacity(requested_output_length),
                output_tokens: 0,
                reasoning_tokens: 0,
                first_output_token_ns: None,
                terminal: None,
                metadata,
            },
        );
    }

    fn on_admit(&self, uuid: Uuid, admit_ms: f64, _reused_input_tokens: usize) {
        if let Some(request) = self.state.borrow_mut().requests.get_mut(&uuid) {
            request
                .dispatch_start_ns
                .get_or_insert_with(|| self.relative_ns_from_ms(admit_ms));
        }
    }

    fn on_token(&self, uuid: Uuid, at_ms: f64) {
        self.on_classified_token(uuid, at_ms, ObservedTokenKind::Output);
    }

    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        let at_ns = self.relative_ns_from_ms(at_ms);
        if let Some(request) = self.state.borrow_mut().requests.get_mut(&uuid) {
            request.token_arrivals_ns.push(at_ns);
            match kind {
                ObservedTokenKind::Output => {
                    request.output_tokens += 1;
                    request.first_output_token_ns.get_or_insert(at_ns);
                }
                ObservedTokenKind::Reasoning => request.reasoning_tokens += 1,
            }
        }
    }

    fn on_usage(&self, uuid: Uuid, usage: ObservedUsage) {
        if let Some(request) = self.state.borrow_mut().requests.get_mut(&uuid) {
            request.response.prompt_tokens = usage.prompt_tokens.map(|value| value as u64);
            request.response.completion_tokens = usage.completion_tokens.map(|value| value as u64);
        }
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        let terminal_ns = self.relative_now_ns();
        if let Some(request) = self.state.borrow_mut().requests.get_mut(&uuid) {
            request.terminal.get_or_insert(status);
            request.terminal_ns.get_or_insert(terminal_ns);
        }
    }
}

/// Local-loop observer fan-out used to share one event stream across consumers.
pub struct ObserverTee {
    delegates: Vec<Rc<dyn RequestObserver>>,
}

impl ObserverTee {
    /// Creates a fan-out observer in deterministic delegate order.
    pub fn new(delegates: Vec<Rc<dyn RequestObserver>>) -> Self {
        Self { delegates }
    }
}

impl RequestObserver for ObserverTee {
    fn on_arrival(&self, uuid: Uuid, at_ms: f64, input: usize, output: usize) {
        for delegate in &self.delegates {
            delegate.on_arrival(uuid, at_ms, input, output);
        }
    }

    fn on_admit(&self, uuid: Uuid, at_ms: f64, reused_input_tokens: usize) {
        for delegate in &self.delegates {
            delegate.on_admit(uuid, at_ms, reused_input_tokens);
        }
    }

    fn on_token(&self, uuid: Uuid, at_ms: f64) {
        for delegate in &self.delegates {
            delegate.on_token(uuid, at_ms);
        }
    }

    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        for delegate in &self.delegates {
            delegate.on_classified_token(uuid, at_ms, kind);
        }
    }

    fn on_usage(&self, uuid: Uuid, usage: ObservedUsage) {
        for delegate in &self.delegates {
            delegate.on_usage(uuid, usage);
        }
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        for delegate in &self.delegates {
            delegate.on_terminal(uuid, status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf_clock::SimClock;
    use aiperf_metrics::{MetricTag, MetricValue};

    #[test]
    fn classified_events_produce_ttfo_reasoning_and_usage_metrics() {
        let clock = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let uuid = Uuid::from_u128(1);
        observer.register_metadata(
            uuid,
            RequestMetricMetadata {
                session_num: Some(9),
                turn_index: 2,
                conversation_id: Some("conversation".to_string()),
                ..RequestMetricMetadata::default()
            },
        );
        observer.on_arrival(uuid, 0.0, 8, 2);
        observer.on_admit(uuid, 1.0, 0);
        observer.on_classified_token(uuid, 10.0, ObservedTokenKind::Reasoning);
        observer.on_classified_token(uuid, 20.0, ObservedTokenKind::Output);
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(8),
                completion_tokens: Some(2),
            },
        );
        clock.advance_to(25_000_000);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);
        observer.record_response(
            uuid,
            NativeResponseMetadata {
                start_ns: Some(1_000_000),
                end_ns: Some(25_000_000),
                prompt_tokens: None,
                completion_tokens: None,
                http: HttpTrace::default(),
            },
        );

        let summary = observer.finish();
        assert_eq!(
            summary.finite_value(MetricTag::ReasoningTokenCount),
            Some(1.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputTokens),
            Some(1.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputSequenceLength),
            Some(2.0)
        );
        assert_eq!(
            summary
                .result(MetricTag::TimeToFirstOutputToken)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(19.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::TotalUsageTotalTokens),
            Some(10.0)
        );
        assert_eq!(
            summary
                .result(MetricTag::CreditDropLatency)
                .unwrap()
                .distribution()
                .unwrap()
                .avg,
            MetricValue::Finite(1.0)
        );
    }
}
