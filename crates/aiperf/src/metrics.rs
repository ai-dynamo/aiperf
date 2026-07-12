// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-side translation from observer events into native metric records.
//!
//! The metrics crate stays transport-neutral: this adapter owns the request-event
//! join because the runtime knows UUIDs, terminal state, usage, and the injected
//! [`aiperf_clock::Clock`] timeline. It is single-loop `Rc`/`RefCell` state and
//! performs one absolute-request-index-addressed accumulator pass after all
//! request tasks drain.

use std::cell::RefCell;
use std::rc::Rc;

use aiperf_clock::Clock;
use aiperf_metrics::{
    AccumulatorSummary, HttpTrace, InferenceDimensions, MetricsAccumulator, MetricsConfig, Phase,
    RecordIngest, TokenCounts, UsageMetrics,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use rustc_hash::FxHashMap;
use uuid::Uuid;

/// Optional workload dimensions registered before request arrival.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestMetricMetadata {
    /// Absolute zero-based request slot assigned before dispatch.
    pub request_index: Option<usize>,
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
    /// Model and fully resolved endpoint selected for this request.
    pub dimensions: InferenceDimensions,
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
            request_index: None,
            phase: Phase::Profiling,
            session_num: None,
            turn_index: 0,
            worker_id: None,
            conversation_id: None,
            dimensions: InferenceDimensions::default(),
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

#[derive(Clone, Debug)]
struct PendingRequest {
    credit_issued_ns: i64,
    dispatch_start_ns: Option<i64>,
    terminal_ns: Option<i64>,
    response: PendingResponseMetadata,
    input_tokens: u64,
    requested_output_tokens: u64,
    token_arrivals_ns: Vec<i64>,
    output_tokens: u64,
    reasoning_tokens: u64,
    first_output_token_ns: Option<i64>,
    endpoint_metrics: Option<Box<ObservedEndpointMetrics>>,
    observed_usage: CompactObservedUsage,
    terminal: Option<ReplayTerminalStatus>,
    metadata: RequestMetricMetadata,
}

#[derive(Clone, Debug, Default)]
struct PendingResponseMetadata {
    start_ns: Option<i64>,
    end_ns: Option<i64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    http: Option<Box<HttpTrace>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactObservedUsage {
    values: [usize; 7],
    present: u8,
}

impl CompactObservedUsage {
    fn set(&mut self, usage: ObservedUsage) {
        for (index, value) in [
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.reasoning_tokens,
            usage.prompt_cache_read_tokens,
            usage.prompt_cache_write_tokens,
            usage.prompt_cache_miss_tokens,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(value) = value {
                self.values[index] = value;
                self.present |= 1 << index;
            }
        }
    }

    fn get(self, index: usize) -> Option<usize> {
        (self.present & (1 << index) != 0).then_some(self.values[index])
    }
}

#[derive(Debug, Default)]
struct ObserverState {
    requests: Vec<Option<PendingEntry>>,
    request_slots: FxHashMap<Uuid, usize>,
    order: Vec<usize>,
    metadata: FxHashMap<Uuid, RequestMetricMetadata>,
}

#[derive(Debug)]
struct PendingEntry {
    uuid: Uuid,
    request: PendingRequest,
}

impl ObserverState {
    fn request(&self, uuid: Uuid) -> Option<&PendingRequest> {
        let slot = *self.request_slots.get(&uuid)?;
        Some(&self.requests.get(slot)?.as_ref()?.request)
    }

    fn request_mut(&mut self, uuid: Uuid) -> Option<&mut PendingRequest> {
        let slot = *self.request_slots.get(&uuid)?;
        Some(&mut self.requests.get_mut(slot)?.as_mut()?.request)
    }

    fn take_terminal(&mut self, uuid: Uuid) -> Option<PendingEntry> {
        let slot = *self.request_slots.get(&uuid)?;
        self.requests.get(slot)?.as_ref()?.request.terminal?;
        self.request_slots.remove(&uuid);
        self.requests.get_mut(slot)?.take()
    }
}

/// Observer-backed native metrics collector sharing the runtime's clock origin.
pub struct NativeMetricsObserver {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    state: RefCell<ObserverState>,
    accumulator: RefCell<MetricsAccumulator>,
    retain_record_dimensions: bool,
}

/// Final aggregate plus the exact request records that produced it.
///
/// Application-layer record exporters consume the retained records through the
/// same [`MetricsAccumulator`] formulas used for the aggregate. Keeping this
/// boundary in Rust prevents convergence/search consumers from reconstructing
/// latency or token metrics from lossy report statistics.
pub struct NativeMetricsCollection {
    /// Aggregate over every finalized request.
    pub summary: AccumulatorSummary,
    /// Finalized request facts in arrival order.
    pub records: Vec<RecordIngest>,
}

/// Owned post-drain native-metrics reduction.
///
/// The value contains no clock, `Rc`, observer, or runtime handle, so an
/// offline runtime may move it to a worker after all deterministic callbacks
/// have completed.
pub struct NativeMetricsFinalizer {
    finish_ns: i64,
    state: ObserverState,
    accumulator: MetricsAccumulator,
}

impl NativeMetricsObserver {
    /// Creates an observer with explicit accumulator configuration.
    pub fn new(clock: Rc<dyn Clock>, origin_ns: i64, config: MetricsConfig) -> Self {
        Self {
            clock,
            origin_ns,
            state: RefCell::new(ObserverState::default()),
            accumulator: RefCell::new(MetricsAccumulator::with_config(config)),
            retain_record_dimensions: true,
        }
    }

    /// Creates an aggregate-only observer without export/join-only row identities.
    pub(crate) fn new_aggregate_only(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
    ) -> Self {
        Self {
            clock,
            origin_ns,
            state: RefCell::new(ObserverState::default()),
            accumulator: RefCell::new(MetricsAccumulator::with_config(config)),
            retain_record_dimensions: false,
        }
    }

    /// Registers workload dimensions before or after the arrival callback.
    pub fn register_metadata(&self, uuid: Uuid, mut metadata: RequestMetricMetadata) {
        if !self.retain_record_dimensions {
            metadata.worker_id = None;
            metadata.conversation_id = None;
            metadata.correlation_id = Some(String::new());
        }
        let mut state = self.state.borrow_mut();
        if let Some(request) = state.request_mut(uuid) {
            metadata.request_index = metadata.request_index.or(request.metadata.request_index);
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
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
            response.prompt_tokens = response.prompt_tokens.or(request.response.prompt_tokens);
            response.completion_tokens = response
                .completion_tokens
                .or(request.response.completion_tokens);
            request.response = PendingResponseMetadata {
                start_ns: response.start_ns,
                end_ns: response.end_ns,
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                http: (response.http != HttpTrace::default()).then(|| Box::new(response.http)),
            };
        }
    }

    /// Snapshot one terminal request without consuming collector state.
    ///
    /// Live exporters use this after terminal metadata and transport facts are
    /// complete. The normal finalizer still owns aggregate construction, so a
    /// best-effort side channel cannot alter the authoritative report.
    pub fn snapshot_record(&self, uuid: Uuid, ordinal: u64) -> Option<RecordIngest> {
        let finish_ns = self.relative_now_ns();
        self.state
            .borrow()
            .request(uuid)
            .cloned()
            .map(|request| request.into_record(uuid, ordinal, finish_ns))
    }

    /// Move one terminal request out of the observer without cloning token data.
    ///
    /// Worker-local streaming collectors use this when the returned record is
    /// transferred to a coordinator that owns final aggregation. The request
    /// must already have received its terminal callback; nonterminal or unknown
    /// UUIDs return `None` without changing observer state.
    pub fn drain_terminal_record(&self, uuid: Uuid, ordinal: u64) -> Option<RecordIngest> {
        let finish_ns = self.relative_now_ns();
        self.state
            .borrow_mut()
            .take_terminal(uuid)
            .map(|entry| entry.request.into_record(entry.uuid, ordinal, finish_ns))
    }

    /// Total arrivals and terminal records still retained by this observer.
    pub fn record_counts(&self) -> (usize, usize) {
        let state = self.state.borrow();
        (state.order.len(), state.request_slots.len())
    }

    /// Finalizes every retained request and returns the full native summary.
    ///
    /// Requests are visited in arrival order, but every metric column is written
    /// to the absolute slot carried by [`RequestMetricMetadata::request_index`].
    pub fn finish(&self) -> AccumulatorSummary {
        self.finish_at(self.clock.now_ns())
    }

    /// Finalizes every retained request at an already captured absolute clock
    /// timestamp.
    ///
    /// Offline runtimes use this after their deterministic event loop has
    /// drained. Capturing the boundary before leaving the loop lets expensive
    /// aggregate reduction run later without extending incomplete records or
    /// changing any simulated timestamp.
    pub fn finish_at(&self, finish_ns: i64) -> AccumulatorSummary {
        self.take_finalizer_at(finish_ns).finish()
    }

    /// Drain observer state into a runtime-neutral owned finalizer.
    pub fn take_finalizer_at(&self, finish_ns: i64) -> NativeMetricsFinalizer {
        let (finish_ns, state, accumulator) = self.take_finalization_state(finish_ns);
        NativeMetricsFinalizer {
            finish_ns,
            state,
            accumulator,
        }
    }

    /// Whether per-record exporter and analyzer identities are retained.
    pub(crate) fn retains_record_dimensions(&self) -> bool {
        self.retain_record_dimensions
    }

    /// Finalizes every retained request while preserving its ingestion facts.
    ///
    /// The records and summary are created by one pass in arrival order. This
    /// method consumes the observer state just like [`Self::finish`]; calling
    /// either finalizer again returns an empty collection.
    pub fn finish_with_records(&self) -> NativeMetricsCollection {
        self.take_finalizer_at(self.clock.now_ns())
            .finish_with_records()
    }

    fn take_finalization_state(&self, finish_ns: i64) -> (i64, ObserverState, MetricsAccumulator) {
        (
            finish_ns.saturating_sub(self.origin_ns),
            std::mem::take(&mut *self.state.borrow_mut()),
            std::mem::take(&mut *self.accumulator.borrow_mut()),
        )
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

impl NativeMetricsFinalizer {
    /// Reduce retained request facts into the native aggregate.
    pub fn finish(self) -> AccumulatorSummary {
        let finish_ns = self.finish_ns;
        let mut accumulator = self.accumulator;
        let mut state = self.state;
        let order = std::mem::take(&mut state.order);
        for (ordinal, slot) in order.into_iter().enumerate() {
            let Some(entry) = state.requests.get_mut(slot).and_then(Option::take) else {
                continue;
            };
            let record = entry
                .request
                .into_record(entry.uuid, ordinal as u64, finish_ns);
            accumulator.process_record(&record);
        }
        drop(state);
        accumulator.summarize()
    }

    /// Reduce retained request facts while preserving their ingestion records.
    pub fn finish_with_records(self) -> NativeMetricsCollection {
        let finish_ns = self.finish_ns;
        let mut accumulator = self.accumulator;
        let mut state = self.state;
        let mut records = Vec::with_capacity(state.order.len());
        let order = std::mem::take(&mut state.order);
        for (ordinal, slot) in order.into_iter().enumerate() {
            let Some(entry) = state.requests.get_mut(slot).and_then(Option::take) else {
                continue;
            };
            let record = entry
                .request
                .into_record(entry.uuid, ordinal as u64, finish_ns);
            accumulator.process_record(&record);
            records.push(record);
        }
        drop(state);
        NativeMetricsCollection {
            summary: accumulator.summarize(),
            records,
        }
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
        let completion_tokens = self
            .response
            .completion_tokens
            .or_else(|| self.observed_usage.get(1).map(|value| value as u64));
        let prompt_tokens = self
            .response
            .prompt_tokens
            .or_else(|| self.observed_usage.get(0).map(|value| value as u64));
        let endpoint_metrics = self
            .endpoint_metrics
            .as_deref()
            .copied()
            .unwrap_or_default();
        RecordIngest {
            request_index: self.metadata.request_index.or(Some(ordinal as usize)),
            correlation_id: self
                .metadata
                .correlation_id
                .unwrap_or_else(|| uuid.to_string()),
            session_num: self.metadata.session_num.unwrap_or(ordinal),
            turn_index: self.metadata.turn_index,
            worker_id: self.metadata.worker_id,
            conversation_id: self.metadata.conversation_id,
            dimensions: self.metadata.dimensions,
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
                total_tokens: self
                    .observed_usage
                    .get(2)
                    .map(|value| value as u64)
                    .or_else(|| {
                        prompt_tokens
                            .zip(completion_tokens)
                            .map(|(prompt, completion)| prompt.saturating_add(completion))
                    }),
                reasoning_tokens: self.observed_usage.get(3).map(|value| value as u64),
                prompt_cache_read_tokens: self.observed_usage.get(4).map(|value| value as u64),
                prompt_cache_write_tokens: self.observed_usage.get(5).map(|value| value as u64),
                prompt_cache_miss_tokens: self.observed_usage.get(6).map(|value| value as u64),
                ..UsageMetrics::default()
            },
            http: self.response.http.map(|http| *http).unwrap_or_default(),
            audio_duration_s: self.metadata.audio_duration_s,
            num_images: endpoint_metrics.num_images.map(|value| value as u64),
            video_inference_seconds: endpoint_metrics.video_inference_seconds,
            video_peak_memory_mb: endpoint_metrics.video_peak_memory_mb,
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
        let mut metadata = state.metadata.remove(&uuid).unwrap_or_default();
        if state.request_slots.contains_key(&uuid) {
            return;
        }
        let slot = *metadata.request_index.get_or_insert(state.requests.len());
        if state.requests.len() <= slot {
            state.requests.resize_with(slot + 1, || None);
        }
        assert!(
            state.requests[slot].is_none(),
            "native metric request slot {slot} was already populated"
        );
        state.request_slots.insert(uuid, slot);
        state.order.push(slot);
        state.requests[slot] = Some(PendingEntry {
            uuid,
            request: PendingRequest {
                credit_issued_ns: self.relative_ns_from_ms(arrival_ms),
                dispatch_start_ns: None,
                terminal_ns: None,
                response: PendingResponseMetadata::default(),
                input_tokens: input_length as u64,
                requested_output_tokens: requested_output_length as u64,
                token_arrivals_ns: Vec::with_capacity(requested_output_length),
                output_tokens: 0,
                reasoning_tokens: 0,
                first_output_token_ns: None,
                endpoint_metrics: None,
                observed_usage: CompactObservedUsage::default(),
                terminal: None,
                metadata,
            },
        });
    }

    fn on_admit(&self, uuid: Uuid, admit_ms: f64, _reused_input_tokens: usize) {
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
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
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
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
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
            request.response.prompt_tokens = usage.prompt_tokens.map(|value| value as u64);
            request.response.completion_tokens = usage.completion_tokens.map(|value| value as u64);
            request.observed_usage.set(usage);
        }
    }

    fn on_endpoint_metrics(&self, uuid: Uuid, metrics: ObservedEndpointMetrics) {
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
            let endpoint = request
                .endpoint_metrics
                .get_or_insert_with(|| Box::new(ObservedEndpointMetrics::default()));
            endpoint.num_images = metrics.num_images.or(endpoint.num_images);
            endpoint.video_inference_seconds = metrics
                .video_inference_seconds
                .or(endpoint.video_inference_seconds);
            endpoint.video_peak_memory_mb = metrics
                .video_peak_memory_mb
                .or(endpoint.video_peak_memory_mb);
        }
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        let terminal_ns = self.relative_now_ns();
        if let Some(request) = self.state.borrow_mut().request_mut(uuid) {
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

    fn on_endpoint_metrics(&self, uuid: Uuid, metrics: ObservedEndpointMetrics) {
        for delegate in &self.delegates {
            delegate.on_endpoint_metrics(uuid, metrics);
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
                ..ObservedUsage::default()
            },
        );
        observer.on_endpoint_metrics(
            uuid,
            ObservedEndpointMetrics {
                num_images: Some(2),
                video_inference_seconds: Some(0.25),
                video_peak_memory_mb: Some(512.0),
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

        let collection = observer.finish_with_records();
        assert_eq!(collection.records.len(), 1);
        assert_eq!(collection.records[0].correlation_id, uuid.to_string());
        assert_eq!(collection.records[0].session_num, 9);
        assert_eq!(collection.records[0].turn_index, 2);
        assert_eq!(
            collection.records[0].token_arrival_ns,
            vec![10_000_000, 20_000_000]
        );
        let summary = collection.summary;
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
        assert_eq!(summary.finite_value(MetricTag::NumImages), Some(2.0));
        assert_eq!(
            summary.finite_value(MetricTag::VideoInferenceTime),
            Some(250.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::VideoPeakMemory),
            Some(512.0)
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

    #[test]
    fn interleaved_tokens_survive_zero_and_underestimated_requested_output_lengths() {
        let clock = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let zero_osl = Uuid::from_u128(20);
        let underestimated_osl = Uuid::from_u128(21);

        observer.on_arrival(zero_osl, 0.0, 4, 0);
        observer.on_arrival(underestimated_osl, 0.0, 4, 1);
        observer.on_token(zero_osl, 10.0);
        observer.on_token(underestimated_osl, 11.0);
        observer.on_token(zero_osl, 20.0);
        observer.on_token(underestimated_osl, 21.0);
        observer.on_token(underestimated_osl, 31.0);
        clock.advance_to(40_000_000);
        observer.on_terminal(zero_osl, ReplayTerminalStatus::Completed);
        observer.on_terminal(underestimated_osl, ReplayTerminalStatus::Completed);

        assert_eq!(
            observer
                .snapshot_record(zero_osl, 0)
                .unwrap()
                .token_arrival_ns,
            vec![10_000_000, 20_000_000]
        );
        assert_eq!(
            observer
                .snapshot_record(underestimated_osl, 1)
                .unwrap()
                .token_arrival_ns,
            vec![11_000_000, 21_000_000, 31_000_000]
        );

        let summary = observer.finish();
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(2.0));
        assert_eq!(
            summary.finite_value(MetricTag::TotalOutputTokens),
            Some(5.0)
        );
    }

    #[test]
    fn terminal_snapshot_does_not_consume_authoritative_record() {
        let clock = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let uuid = Uuid::from_u128(7);
        observer.register_metadata(
            uuid,
            RequestMetricMetadata {
                session_num: Some(12),
                ..RequestMetricMetadata::default()
            },
        );
        observer.on_arrival(uuid, 1.0, 4, 2);
        observer.on_admit(uuid, 2.0, 0);
        clock.advance_to(9_000_000);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let snapshot = observer.snapshot_record(uuid, 12).unwrap();
        assert_eq!(snapshot.request_index, Some(0));
        assert_eq!(snapshot.session_num, 12);
        assert_eq!(snapshot.start_ns, 2_000_000);
        assert_eq!(snapshot.end_ns, 9_000_000);

        let collection = observer.finish_with_records();
        assert_eq!(collection.records, vec![snapshot]);
    }

    #[test]
    fn terminal_drain_moves_token_storage_to_streaming_owner() {
        let clock = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let uuid = Uuid::from_u128(8);
        observer.on_arrival(uuid, 0.0, 4, 1);
        observer.on_token(uuid, 2.0);
        observer.on_token(uuid, 4.0);
        clock.advance_to(5_000_000);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        assert_eq!(observer.record_counts(), (1, 1));
        let record = observer.drain_terminal_record(uuid, 0).unwrap();
        assert_eq!(record.token_arrival_ns, vec![2_000_000, 4_000_000]);
        assert_eq!(observer.record_counts(), (1, 0));
        assert!(observer.drain_terminal_record(uuid, 0).is_none());

        let collection = observer.finish_with_records();
        assert!(collection.records.is_empty());
        assert_eq!(
            collection.summary.finite_value(MetricTag::RequestCount),
            None
        );
    }

    #[test]
    fn observer_keeps_chunk_facts_but_uses_usage_for_labeled_token_metrics() {
        let clock = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let uuid = Uuid::from_u128(9);
        observer.register_metadata(
            uuid,
            RequestMetricMetadata {
                dimensions: InferenceDimensions {
                    endpoint_url: Some("https://endpoint/v1/chat/completions".to_string()),
                    model: Some("model-a".to_string()),
                },
                ..RequestMetricMetadata::default()
            },
        );
        observer.on_arrival(uuid, 0.0, 8, 5);
        observer.on_admit(uuid, 0.0, 0);
        observer.on_token(uuid, 10.0);
        observer.on_token(uuid, 20.0);
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(8),
                completion_tokens: Some(5),
                ..ObservedUsage::default()
            },
        );
        clock.advance_to(100_000_000);
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let collection = observer.finish_with_records();
        assert_eq!(collection.records[0].token_arrival_ns.len(), 2);
        assert_eq!(collection.records[0].tokens.output, Some(2));
        assert_eq!(collection.records[0].usage.completion_tokens, Some(5));
        assert_eq!(
            collection
                .summary
                .finite_value(MetricTag::TotalOutputSequenceLength),
            Some(5.0)
        );
        assert_eq!(collection.summary.inference_series().len(), 1);
        assert_eq!(
            collection.summary.inference_series()[0]
                .dimensions()
                .model
                .as_deref(),
            Some("model-a")
        );
        let icl = collection
            .summary
            .result(MetricTag::InterChunkLatency)
            .unwrap();
        assert_eq!(icl.distribution().unwrap().count, 1);
    }
}
