// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tumbling returned-request windows.
//!
//! The sampler works as follows: token observations
//! are joined to a request by id, successful requests enter the window only at
//! terminal return, and [`take`](WindowSampler::take) resets completed-window
//! aggregates while retaining still-in-flight request state.

use std::collections::HashMap;

use crate::metrics_core::RecordIngest;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::ObservedUsage;
use uuid::Uuid;

/// Per-request inputs used by quality-filtered goodput.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestSample {
    /// Request-start through last meaningful token latency in nanoseconds.
    pub request_latency_ns: i64,
    /// Time to first token in nanoseconds, when a token was observed.
    pub ttft_ns: Option<i64>,
    /// Mean inter-token latency in nanoseconds, when at least two tokens arrived.
    pub inter_token_latency_ns: Option<f64>,
    /// Authoritative completion-token count, when endpoint usage was returned.
    pub output_sequence_length: Option<usize>,
}

/// One non-overlapping adaptive assessment window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowStats {
    /// Successful requests returned during the window.
    pub successful_requests: Vec<RequestSample>,
    /// Failed or rejected requests returned during the window.
    pub errors: usize,
    /// Cancelled requests returned during the window.
    pub cancelled: usize,
    /// Window duration in seconds on the injected clock.
    pub elapsed_sec: f64,
    /// Inclusive window start on the injected clock's nanosecond timeline.
    pub start_ns: i64,
    /// Window end on the injected clock's nanosecond timeline.
    pub end_ns: i64,
}

impl WindowStats {
    /// Build an empty window spanning `start_ns..end_ns`.
    pub fn empty(start_ns: i64, end_ns: i64) -> Self {
        Self {
            successful_requests: Vec::new(),
            errors: 0,
            cancelled: 0,
            elapsed_sec: elapsed_seconds(start_ns, end_ns),
            start_ns,
            end_ns,
        }
    }

    /// Successful request-latency samples in nanoseconds.
    pub fn latency_samples(&self) -> Vec<f64> {
        self.successful_requests
            .iter()
            .map(|sample| sample.request_latency_ns as f64)
            .collect()
    }

    /// Successful TTFT samples in nanoseconds.
    pub fn ttft_samples(&self) -> Vec<f64> {
        self.successful_requests
            .iter()
            .filter_map(|sample| sample.ttft_ns.map(|value| value as f64))
            .collect()
    }

    /// Successful mean-ITL samples in nanoseconds.
    pub fn itl_samples(&self) -> Vec<f64> {
        self.successful_requests
            .iter()
            .filter_map(|sample| sample.inter_token_latency_ns)
            .collect()
    }

    /// Successful output sequence lengths for requests with an observed count.
    pub fn output_sequence_lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.successful_requests
            .iter()
            .filter_map(|sample| sample.output_sequence_length)
    }

    /// Successful request count.
    pub fn completed(&self) -> usize {
        self.successful_requests.len()
    }

    /// All returned attempts, including failures and cancellations.
    pub fn total(&self) -> usize {
        self.completed() + self.errors + self.cancelled
    }

    /// Completed-request throughput in requests per second.
    pub fn throughput(&self) -> f64 {
        if self.elapsed_sec <= 0.0 {
            0.0
        } else {
            self.completed() as f64 / self.elapsed_sec
        }
    }

    /// Observed output-token throughput in tokens per second.
    pub fn output_token_throughput(&self) -> f64 {
        if self.elapsed_sec <= 0.0 {
            0.0
        } else {
            self.output_sequence_lengths().sum::<usize>() as f64 / self.elapsed_sec
        }
    }
}

/// Accumulates request lifecycle observations and snapshots tumbling windows.
///
/// Implementations are intentionally single-loop objects. Callers typically
/// store one behind `Rc<RefCell<Box<dyn WindowSampler>>>`; no lock is needed.
pub trait WindowSampler {
    /// Ingest one already-terminal native request record.
    ///
    /// Worker-local backends use this path after returning immutable records to
    /// the coordinator; callback-oriented transports use the lifecycle methods
    /// below. Implementations must preserve identical classification and
    /// latency formulas across both inputs.
    fn on_record(&mut self, record: &RecordIngest);
    /// Record the request's arrival on the clock timeline.
    fn on_arrival(&mut self, uuid: Uuid, at_ns: i64);
    /// Replace the provisional arrival start with transport/backend admission.
    /// The default preserves samplers whose source does not expose admission.
    fn on_admit(&mut self, _uuid: Uuid, _at_ns: i64) {}
    /// Record one non-empty output-token delta on the clock timeline.
    fn on_token(&mut self, uuid: Uuid, at_ns: i64);
    /// Record authoritative endpoint usage for a request, when available.
    fn on_usage(&mut self, _uuid: Uuid, _usage: ObservedUsage) {}
    /// Record terminal return and classify it into the current window.
    fn on_terminal(&mut self, uuid: Uuid, status: ReplayTerminalStatus, at_ns: i64);
    /// Snapshot and reset the completed portion of the current window.
    fn take(&mut self, end_ns: i64) -> WindowStats;
}

#[derive(Debug)]
struct InFlightRequest {
    started_ns: i64,
    token_times_ns: Vec<i64>,
    output_sequence_length: Option<usize>,
}

/// Default single-loop [`WindowSampler`] implementation.
#[derive(Debug)]
pub struct TumblingWindowSampler {
    window_started_at_ns: i64,
    in_flight: HashMap<Uuid, InFlightRequest>,
    completed: Vec<RequestSample>,
    errors: usize,
    cancelled: usize,
}

impl TumblingWindowSampler {
    /// Start a sampler whose first window begins at `start_ns`.
    pub fn new(start_ns: i64) -> Self {
        Self {
            window_started_at_ns: start_ns,
            in_flight: HashMap::new(),
            completed: Vec::new(),
            errors: 0,
            cancelled: 0,
        }
    }

    /// Number of request lifecycles currently awaiting terminal return.
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Ingest one already-terminal native request record.
    ///
    /// Thread-per-core graph workers finalize native records before returning
    /// a trace. The coordinator can feed those records directly without
    /// replaying a synthetic callback sequence or sharing a mutable sampler
    /// across worker threads. Formulas match the ordinary observer path:
    /// admission is the latency origin, meaningful token timestamps determine
    /// TTFT/request latency, and endpoint usage owns OSL/ITL reconciliation.
    pub fn process_record(&mut self, record: &RecordIngest) {
        self.on_record(record);
    }
}

impl WindowSampler for TumblingWindowSampler {
    fn on_record(&mut self, record: &RecordIngest) {
        if record.canceled {
            self.cancelled += 1;
            return;
        }
        if record.errored {
            self.errors += 1;
            return;
        }
        let Some(first) = record.token_arrival_ns.first().copied() else {
            self.errors += 1;
            return;
        };
        let last = record
            .token_arrival_ns
            .last()
            .copied()
            .expect("a first token implies a last token");
        let started_ns = record.admit_ns.unwrap_or(record.start_ns);
        let output_sequence_length = record.usage.completion_tokens.and_then(count_to_usize);
        // Match on_terminal: a meaningful ITL needs both authoritative OSL > 1 and
        // more than one observed token, otherwise first == last yields a bogus 0.0.
        let inter_token_latency_ns = output_sequence_length
            .filter(|count| *count > 1 && record.token_arrival_ns.len() > 1)
            .map(|count| last.saturating_sub(first).max(0) as f64 / (count - 1) as f64);
        self.completed.push(RequestSample {
            request_latency_ns: last.saturating_sub(started_ns).max(0),
            ttft_ns: Some(first.saturating_sub(started_ns).max(0)),
            inter_token_latency_ns,
            output_sequence_length,
        });
    }

    fn on_arrival(&mut self, uuid: Uuid, at_ns: i64) {
        self.in_flight.insert(
            uuid,
            InFlightRequest {
                started_ns: at_ns,
                token_times_ns: Vec::new(),
                output_sequence_length: None,
            },
        );
    }

    fn on_admit(&mut self, uuid: Uuid, at_ns: i64) {
        if let Some(request) = self.in_flight.get_mut(&uuid) {
            request.started_ns = at_ns;
        }
    }

    fn on_token(&mut self, uuid: Uuid, at_ns: i64) {
        if let Some(request) = self.in_flight.get_mut(&uuid) {
            request.token_times_ns.push(at_ns);
        }
    }

    fn on_usage(&mut self, uuid: Uuid, usage: ObservedUsage) {
        if let Some(request) = self.in_flight.get_mut(&uuid) {
            request.output_sequence_length = usage.completion_tokens.and_then(count_to_usize);
        }
    }

    fn on_terminal(&mut self, uuid: Uuid, status: ReplayTerminalStatus, _at_ns: i64) {
        let request = self.in_flight.remove(&uuid);
        match status {
            ReplayTerminalStatus::Completed => {
                let Some(request) = request else {
                    self.errors += 1;
                    return;
                };
                let Some(first) = request.token_times_ns.first() else {
                    // Python's credit-return path classifies a completed response
                    // without request_latency_ns as an error: role/usage/finish
                    // frames alone are not a successful inference sample.
                    self.errors += 1;
                    return;
                };
                let last = request
                    .token_times_ns
                    .last()
                    .expect("a first token implies a last token");
                let request_latency_ns = last.saturating_sub(request.started_ns).max(0);
                let ttft_ns = Some(first.saturating_sub(request.started_ns).max(0));
                let inter_token_latency_ns = match (
                    request.token_times_ns.as_slice(),
                    request.output_sequence_length,
                ) {
                    ([first, .., last], Some(output_sequence_length))
                        if request.token_times_ns.len() > 1 && output_sequence_length > 1 =>
                    {
                        Some(
                            last.saturating_sub(*first).max(0) as f64
                                / (output_sequence_length - 1) as f64,
                        )
                    }
                    _ => None,
                };
                self.completed.push(RequestSample {
                    request_latency_ns,
                    ttft_ns,
                    inter_token_latency_ns,
                    output_sequence_length: request.output_sequence_length,
                });
            }
            ReplayTerminalStatus::Canceled => self.cancelled += 1,
            ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => self.errors += 1,
        }
    }

    fn take(&mut self, end_ns: i64) -> WindowStats {
        let start_ns = self.window_started_at_ns;
        self.window_started_at_ns = end_ns;
        WindowStats {
            successful_requests: std::mem::take(&mut self.completed),
            errors: std::mem::take(&mut self.errors),
            cancelled: std::mem::take(&mut self.cancelled),
            elapsed_sec: elapsed_seconds(start_ns, end_ns),
            start_ns,
            end_ns,
        }
    }
}

fn elapsed_seconds(start_ns: i64, end_ns: i64) -> f64 {
    end_ns.saturating_sub(start_ns).max(0) as f64 / 1_000_000_000.0
}

fn count_to_usize<T>(value: T) -> Option<usize>
where
    T: TryInto<usize>,
{
    value.try_into().ok()
}

#[cfg(test)]
mod tests {
    use crate::metrics_core::Phase;

    use super::*;

    #[test]
    fn joins_tokens_to_terminal_and_resets_only_completed_window_state() {
        let mut sampler = TumblingWindowSampler::new(0);
        let completed = Uuid::new_v4();
        let spanning = Uuid::new_v4();
        sampler.on_arrival(completed, 10);
        sampler.on_token(completed, 20);
        sampler.on_token(completed, 30);
        sampler.on_usage(
            completed,
            ObservedUsage {
                prompt_tokens: Some(3),
                completion_tokens: Some(5),
                ..ObservedUsage::default()
            },
        );
        sampler.on_arrival(spanning, 40);
        sampler.on_token(spanning, 50);
        sampler.on_terminal(completed, ReplayTerminalStatus::Completed, 40);

        let first = sampler.take(1_000_000_000);
        assert_eq!(first.completed(), 1);
        assert_eq!(first.successful_requests[0].request_latency_ns, 20);
        assert_eq!(first.successful_requests[0].ttft_ns, Some(10));
        assert_eq!(
            first.successful_requests[0].inter_token_latency_ns,
            Some(2.5)
        );
        assert_eq!(first.successful_requests[0].output_sequence_length, Some(5));
        assert_eq!(
            sampler.in_flight(),
            1,
            "in-flight joins survive a window take"
        );

        sampler.on_terminal(spanning, ReplayTerminalStatus::Completed, 60);
        let second = sampler.take(2_000_000_000);
        assert_eq!(second.completed(), 1);
        assert_eq!(second.successful_requests[0].ttft_ns, Some(10));
        assert_eq!(second.successful_requests[0].inter_token_latency_ns, None);
        assert_eq!(second.successful_requests[0].output_sequence_length, None);
        assert_eq!(second.elapsed_sec, 1.0);
    }

    #[test]
    fn rates_include_failed_and_cancelled_attempts_in_the_total() {
        let stats = WindowStats {
            successful_requests: vec![RequestSample {
                request_latency_ns: 10,
                ttft_ns: None,
                inter_token_latency_ns: None,
                output_sequence_length: Some(8),
            }],
            errors: 1,
            cancelled: 2,
            elapsed_sec: 2.0,
            start_ns: 0,
            end_ns: 2_000_000_000,
        };
        assert_eq!(stats.total(), 4);
        assert_eq!(stats.throughput(), 0.5);
        assert_eq!(stats.output_token_throughput(), 4.0);
    }

    #[test]
    fn errored_and_cancelled_returns_drop_partial_token_joins() {
        let mut sampler = TumblingWindowSampler::new(0);
        let failed = Uuid::new_v4();
        let cancelled = Uuid::new_v4();
        sampler.on_arrival(failed, 0);
        sampler.on_token(failed, 10);
        sampler.on_terminal(failed, ReplayTerminalStatus::Failed, 20);
        sampler.on_arrival(cancelled, 0);
        sampler.on_token(cancelled, 10);
        sampler.on_terminal(cancelled, ReplayTerminalStatus::Canceled, 20);
        let stats = sampler.take(100);
        assert!(stats.successful_requests.is_empty());
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.cancelled, 1);
        assert_eq!(stats.total(), 2);
        assert_eq!(sampler.in_flight(), 0);
    }

    #[test]
    fn completed_response_without_meaningful_tokens_is_an_error() {
        let mut sampler = TumblingWindowSampler::new(0);
        let uuid = Uuid::new_v4();
        sampler.on_arrival(uuid, 10);
        sampler.on_terminal(uuid, ReplayTerminalStatus::Completed, 20);

        let stats = sampler.take(100);
        assert!(stats.successful_requests.is_empty());
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.total(), 1);
    }

    #[test]
    fn admission_replaces_arrival_as_the_latency_origin() {
        let mut sampler = TumblingWindowSampler::new(0);
        let uuid = Uuid::new_v4();
        sampler.on_arrival(uuid, 10);
        sampler.on_admit(uuid, 20);
        sampler.on_token(uuid, 30);
        sampler.on_terminal(uuid, ReplayTerminalStatus::Completed, 40);

        let stats = sampler.take(100);
        assert_eq!(stats.successful_requests[0].ttft_ns, Some(10));
        assert_eq!(stats.successful_requests[0].request_latency_ns, 10);
    }

    #[test]
    fn terminal_native_records_match_observer_window_formulas() {
        let mut sampler = TumblingWindowSampler::new(0);
        let mut completed = RecordIngest::minimal(10, 50, Phase::Profiling);
        completed.admit_ns = Some(20);
        completed.token_arrival_ns = vec![30, 40];
        completed.usage.completion_tokens = Some(5);
        sampler.process_record(&completed);

        let mut failed = RecordIngest::minimal(10, 50, Phase::Profiling);
        failed.errored = true;
        sampler.process_record(&failed);
        let mut cancelled = RecordIngest::minimal(10, 50, Phase::Profiling);
        cancelled.canceled = true;
        sampler.process_record(&cancelled);

        let stats = sampler.take(1_000_000_000);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.cancelled, 1);
        assert_eq!(stats.successful_requests[0].request_latency_ns, 20);
        assert_eq!(stats.successful_requests[0].ttft_ns, Some(10));
        assert_eq!(
            stats.successful_requests[0].inter_token_latency_ns,
            Some(2.5)
        );
        assert_eq!(stats.successful_requests[0].output_sequence_length, Some(5));
    }
}
