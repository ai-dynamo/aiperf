// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime for request-rate, user-centric, and fixed-schedule workloads.
//!
//! The runtime is the small policy-neutral bridge from a [`Workload`] schedule
//! generator to a pluggable [`TurnDispatcher`]. It owns the clock-backed task
//! scheduler, stop/counter state, measurement observer, and detailed schedule
//! trace. Strategy modules decide only when to call
//! [`issue_turn`](ScheduledRuntime::issue_turn) and what continuation to
//! schedule when the dispatch completes. Its synchronous counter mutation and
//! asynchronous return callback preserve credit ordering. Ancillary
//! cancellation and URL issuance remain session-pinned.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use crate::clock::Clock;
use crate::endpoints::ParsedResponse;
use crate::metrics_core::{AccumulatorSummary, InferenceDimensions, MetricsConfig, RequestTrace};
use crate::timing::{CancellationPolicy, Phase, SlotPool, StopChecker, StopConfig, UrlSelector};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::observer::CollectorObserver;
use loadgen_core::sink::RequestObserver;
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::metrics::{
    NativeMetricsObserver, NativeResponseMetadata, ObserverTee, RequestMetricMetadata,
};
use crate::multiturn::{ConversationSource, CreditCounter, IssuedCredit, TurnResponse, TurnToSend};
use crate::scheduler::{ClockTaskScheduler, LocalTaskScheduler};

/// Boxed `!Send` completion future returned by a workload callback.
pub type CompletionTask = Pin<Box<dyn Future<Output = ()> + 'static>>;

/// Completion callback installed for one issued turn.
pub type CompletionHandler =
    Box<dyn FnOnce(IssuedCredit, TurnDispatchOutcome) -> CompletionTask + 'static>;

/// First-token callback installed for one issued turn.
///
/// Admission policies use this edge to release prefill capacity while the
/// request continues decoding. The terminal callback remains responsible for
/// the no-token fallback.
pub type FirstTokenHandler = Box<dyn Fn(i64) + 'static>;

/// Replaceable cancellation latch for one admitted dispatch.
///
/// The issuer selects this future against the ordinary backend dispatch. A
/// cancellation winner drops that dispatch future, emits exactly one cancelled
/// observer terminal, and still invokes the normal completion callback. This
/// keeps cancellation ownership outside the transport-neutral
/// [`TurnDispatcher`] seam.
pub trait DispatchCancellation {
    /// Whether cancellation was requested before dispatch received its first poll.
    fn is_cancelled(&self) -> bool;

    /// Wait until cancellation is requested.
    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + '_>>;
}

/// Backpressured endpoint-normalized response-frame consumer.
///
/// HTTP invokes this callback on the local reactor as each decoded SSE event
/// arrives. The poll/send split reserves bounded downstream capacity without
/// blocking a current-thread reactor or allocating a future per frame. Raw SSE
/// bytes never cross this seam.
pub trait TurnResponseObserver {
    /// Reserve capacity for the next endpoint-parsed frame.
    fn poll_ready(&self, context: &mut Context<'_>) -> Poll<Result<()>>;

    /// Send one frame after [`Self::poll_ready`] returned ready.
    fn start_send(&self, response: ParsedResponse) -> Result<()>;
}

/// Optional external admission gate layered above ordinary stop conditions.
/// Adaptive-scale implements this seam so a terminal controller immediately
/// blocks root and continuation issuance while in-flight dispatches drain.
pub trait IssuanceGate {
    /// Whether another turn may be issued.
    fn can_issue(&self) -> bool;
}

/// Endpoint-normalized assistant and terminal metadata retained by the normal
/// dispatch path.
///
/// Scheduled workloads that only need continuation text can keep using
/// [`TurnDispatchOutcome::response_text`]. Stateful consumers use this richer
/// record to preserve reasoning, truncation, provider correlation, cache usage,
/// and infrastructure failures without reparsing transport payloads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelResponseMetadata {
    /// User-visible assistant content without a separate reasoning channel.
    pub content: Option<String>,
    /// Provider-emitted reasoning content, when the endpoint distinguishes it.
    pub reasoning: Option<String>,
    /// Prompt tokens served from a provider cache.
    pub cached_prompt_tokens: Option<u64>,
    /// Provider response identifier used by stateful APIs and artifacts.
    pub response_id: Option<String>,
    /// Endpoint-normalized finish reason, such as `stop` or `length`.
    pub finish_reason: Option<String>,
    /// Exact generated token IDs from a token-native non-text response.
    pub output_token_ids: Option<Vec<u32>>,
    /// Reassembled OpenAI-compatible assistant message, including tool calls.
    pub assistant_message: Option<Value>,
    /// Stable transport/provider failure category for non-completed requests.
    pub error_kind: Option<String>,
    /// Human-readable transport/provider failure detail.
    pub error_message: Option<String>,
    /// Decoded endpoint response frames retained inside Rust for operation-
    /// specific normalization. These values are never forwarded as raw SSE and
    /// must not enter diagnostics or public artifacts.
    pub wire_responses: Vec<Value>,
}

/// Terminal result returned by a [`TurnDispatcher`].
#[derive(Clone, Debug)]
pub struct TurnDispatchOutcome {
    /// Clock timestamp at which transport/backend dispatch began.
    pub start_ns: i64,
    /// Clock timestamp at which dispatch reached terminal.
    pub end_ns: i64,
    /// Terminal classification emitted to the measurement observer.
    pub terminal: ReplayTerminalStatus,
    /// Assistant text captured for the next turn's dynamic prompt splice.
    pub response_text: String,
    /// Rich model-response metadata captured by the ordinary endpoint parser.
    pub model_response: ModelResponseMetadata,
    /// Authoritative server prompt-token usage, when available.
    pub prompt_tokens: Option<u64>,
    /// Authoritative server completion-token usage, when available.
    pub completion_tokens: Option<u64>,
    /// Fine-grained transport metrics, when the backend supplies them.
    pub http: RequestTrace,
}

impl TurnDispatchOutcome {
    /// Project the fields a continuation request needs into a [`TurnResponse`].
    ///
    /// Request-rate, user-centric, and fixed-schedule completion hooks all build
    /// the same continuation response from this outcome; this keeps that shape in
    /// one place.
    pub(crate) fn to_turn_response(&self) -> TurnResponse {
        TurnResponse {
            text: self.response_text.clone(),
            assistant_message: self.model_response.assistant_message.clone(),
            completion_tokens: self.completion_tokens,
            terminal: self.terminal,
        }
    }
}

/// Transport/backend seam consumed by scheduled multi-turn workloads.
///
/// The current online implementation adapts `TransportSink`, which remains a
/// normal `RequestSink<Request>`. An offline engine or another endpoint
/// dialect implements this trait once; request-rate, user-centric, and
/// fixed-schedule policy stays unchanged.
#[async_trait(?Send)]
pub trait TurnDispatcher {
    /// Whether this backend can emit endpoint-normalized frames before terminal.
    fn supports_response_streaming(&self) -> bool {
        false
    }

    /// Resolve report dimensions using the same backend selection that dispatch
    /// will apply. Alternate backends may omit dimensions explicitly.
    fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions::default()
    }

    /// Dispatch one fully materialized turn. `on_first_token` receives the
    /// backend's TTFT delta in nanoseconds exactly once when a token arrives.
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome>;

    /// Dispatch while emitting live endpoint-normalized response frames.
    ///
    /// Terminal-only backends inherit a fail-closed default instead of
    /// silently buffering a requested stream.
    async fn dispatch_turn_streaming(
        &self,
        _turn: TurnToSend,
        _observer: &dyn RequestObserver,
        _on_first_token: &dyn Fn(i64),
        _responses: &dyn TurnResponseObserver,
    ) -> Result<TurnDispatchOutcome> {
        Err(anyhow!(
            "selected turn dispatcher does not support true response streaming"
        ))
    }

    /// Warm the dispatch path with one discarded round-trip before timed
    /// issuance, so the first authored request is not delayed relative to its
    /// schedule by one-time setup. The default is a no-op; real dispatchers warm
    /// their execution backend without recording the warmup in the metrics.
    async fn prewarm(&self, _turn: TurnToSend) -> Result<()> {
        Ok(())
    }
}

/// Post-dispatch record-processing seam shared by ordinary workloads.
///
/// Consumers such as accuracy grading attach here without owning transport or
/// issuance policy.
#[async_trait(?Send)]
pub trait TurnRecordProcessor {
    /// Process one terminal turn after normal measurement facts are recorded.
    async fn process(&self, credit: &IssuedCredit, outcome: &TurnDispatchOutcome) -> Result<()>;
}

/// Synchronous per-turn lifecycle seam for phase/accounting policy.
///
/// Issuance is observed before the dispatch task is spawned, preserving the
/// freeze-before-return protocol even when a workload finishes scheduling
/// before that task receives its first poll. First-token and terminal calls run
/// on the same local dispatch task. Implementations must not block or await.
pub trait TurnLifecycleObserver {
    /// Observe one accepted turn before asynchronous backend dispatch begins.
    fn on_issue(&self, turn: &TurnToSend);

    /// Observe the first meaningful token for an active request.
    fn on_first_token(&self, uuid: Uuid);

    /// Observe terminal dispatch, including synthesized failure outcomes.
    fn on_terminal(&self, turn: &TurnToSend, outcome: &TurnDispatchOutcome);
}

/// One turn's expected and observed timing, all offsets relative to run start.
#[derive(Clone, Debug, Serialize)]
pub struct TurnTimingRecord {
    /// Request UUID shared with the aggregate collector.
    pub uuid: Uuid,
    /// Template id.
    pub conversation_id: String,
    /// Runtime session id.
    pub x_correlation_id: String,
    /// Simulated user id for user-centric runs; absent for fixed schedule.
    pub user_id: Option<u64>,
    /// Zero-based turn index.
    pub turn_index: usize,
    /// Number of turns planned for this runtime session.
    pub num_turns: usize,
    /// Ideal scheduler target relative to run start; may be negative for trace
    /// timestamps before a manually selected zero.
    pub scheduled_offset_ns: i64,
    /// Actual issuer time relative to run start.
    pub issued_offset_ns: i64,
    /// Backend dispatch start relative to run start.
    pub dispatch_start_offset_ns: Option<i64>,
    /// First output token time relative to run start.
    pub first_token_offset_ns: Option<i64>,
    /// Backend-reported TTFT delta.
    pub ttft_ns: Option<i64>,
    /// Terminal time relative to run start.
    pub terminal_offset_ns: Option<i64>,
    /// Terminal status, once known.
    pub terminal_status: Option<ReplayTerminalStatus>,
}

/// Aggregate schedule fidelity derived from [`TurnTimingRecord`] entries.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ScheduleTimingAnalysis {
    /// Number of issued turns.
    pub issued_turns: usize,
    /// Number of turns observed before their scheduler target. This should be 0.
    pub early_turns: usize,
    /// Mean non-negative issue lateness in milliseconds.
    pub mean_issue_lateness_ms: f64,
    /// Maximum issue lateness in milliseconds.
    pub max_issue_lateness_ms: f64,
    /// Mean backend TTFT in milliseconds over turns with a first token.
    pub mean_ttft_ms: Option<f64>,
    /// Maximum backend TTFT in milliseconds over turns with a first token.
    pub max_ttft_ms: Option<f64>,
}

/// Adaptive user-pool snapshot included in user-centric reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct UserControlSnapshot {
    /// Current requested user target.
    pub target_value: usize,
    /// Users currently present in the pool.
    pub actual_value: usize,
    /// Alias of `actual_value`, matching the adaptive-control vocabulary.
    pub active_users: usize,
    /// Excess users draining after a scale-down.
    pub retiring_users: usize,
    /// Retired users force-cancelled by control policy.
    pub cancelled: usize,
}

/// Complete result of a scheduled workload: aggregate inference metrics plus
/// schedule-level evidence used to verify timing policy.
#[derive(Debug, Serialize)]
pub struct ScheduledRunReport {
    /// Workload strategy name.
    pub strategy: String,
    /// Standard AIPerf request/throughput/latency report.
    pub performance: TraceSimulationReport,
    /// Native typed distributions, sweeps, and derived metrics.
    pub native_metrics: AccumulatorSummary,
    /// Derived schedule fidelity statistics.
    pub schedule_timing: ScheduleTimingAnalysis,
    /// Per-turn expected and observed timing.
    pub turns: Vec<TurnTimingRecord>,
    /// User-pool control state for user-centric runs.
    pub user_control: Option<UserControlSnapshot>,
}

struct TimingRecorder {
    records: Vec<TurnTimingRecord>,
    capture_records: bool,
    issued_turns: usize,
    early_turns: usize,
    lateness_sum_ns: i128,
    max_lateness_ns: i64,
    ttft_sum_ns: i128,
    max_ttft_ns: i64,
    ttft_count: usize,
}

impl Default for TimingRecorder {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            capture_records: true,
            issued_turns: 0,
            early_turns: 0,
            lateness_sum_ns: 0,
            max_lateness_ns: 0,
            ttft_sum_ns: 0,
            max_ttft_ns: 0,
            ttft_count: 0,
        }
    }
}

impl TimingRecorder {
    fn begin(
        &mut self,
        turn: &TurnToSend,
        user_id: Option<u64>,
        start_ns: i64,
        scheduled_ns: i64,
        issued_ns: i64,
    ) -> usize {
        let index = self.issued_turns;
        self.issued_turns += 1;
        let scheduled_offset_ns = scheduled_ns.saturating_sub(start_ns);
        let issued_offset_ns = issued_ns.saturating_sub(start_ns);
        let raw_lateness = issued_offset_ns - scheduled_offset_ns;
        if raw_lateness < 0 {
            self.early_turns += 1;
        }
        let lateness = raw_lateness.max(0);
        self.lateness_sum_ns += i128::from(lateness);
        self.max_lateness_ns = self.max_lateness_ns.max(lateness);
        if self.capture_records {
            self.records.push(TurnTimingRecord {
                uuid: turn.uuid,
                conversation_id: turn.conversation_id.clone(),
                x_correlation_id: turn.x_correlation_id.clone(),
                user_id,
                turn_index: turn.turn_index,
                num_turns: turn.num_turns,
                scheduled_offset_ns,
                issued_offset_ns,
                dispatch_start_offset_ns: None,
                first_token_offset_ns: None,
                ttft_ns: None,
                terminal_offset_ns: None,
                terminal_status: None,
            });
        }
        index
    }

    fn first_token(&mut self, index: usize, at_ns: i64, start_ns: i64, ttft_ns: i64) {
        let ttft_ns = ttft_ns.max(0);
        if self.capture_records {
            if let Some(record) = self.records.get_mut(index)
                && record.first_token_offset_ns.is_none()
            {
                self.ttft_sum_ns += i128::from(ttft_ns);
                self.max_ttft_ns = self.max_ttft_ns.max(ttft_ns);
                self.ttft_count += 1;
                record.first_token_offset_ns = Some(at_ns.saturating_sub(start_ns));
                record.ttft_ns = Some(ttft_ns);
            }
        } else {
            self.ttft_sum_ns += i128::from(ttft_ns);
            self.max_ttft_ns = self.max_ttft_ns.max(ttft_ns);
            self.ttft_count += 1;
        }
    }

    fn terminal(&mut self, index: usize, outcome: &TurnDispatchOutcome, start_ns: i64) {
        if let Some(record) = self.records.get_mut(index) {
            record.dispatch_start_offset_ns = Some(outcome.start_ns.saturating_sub(start_ns));
            record.terminal_offset_ns = Some(outcome.end_ns.saturating_sub(start_ns));
            record.terminal_status = Some(outcome.terminal);
        }
    }

    fn analysis(&self) -> ScheduleTimingAnalysis {
        if self.issued_turns == 0 {
            return ScheduleTimingAnalysis::default();
        }
        ScheduleTimingAnalysis {
            issued_turns: self.issued_turns,
            early_turns: self.early_turns,
            mean_issue_lateness_ms: self.lateness_sum_ns as f64
                / self.issued_turns as f64
                / 1_000_000.0,
            max_issue_lateness_ms: self.max_lateness_ns as f64 / 1_000_000.0,
            mean_ttft_ms: (self.ttft_count > 0)
                .then_some(self.ttft_sum_ns as f64 / self.ttft_count as f64 / 1_000_000.0),
            max_ttft_ms: (self.ttft_count > 0).then_some(self.max_ttft_ns as f64 / 1_000_000.0),
        }
    }
}

/// Shared facilities injected into a [`Workload`].
pub struct ScheduledRuntime {
    clock: Rc<dyn Clock>,
    start_ns: i64,
    scheduler: Rc<ClockTaskScheduler>,
    dispatcher: Rc<dyn TurnDispatcher>,
    collector: Rc<CollectorObserver>,
    native_metrics: Rc<NativeMetricsObserver>,
    observer: Rc<dyn RequestObserver>,
    recorder: Rc<RefCell<TimingRecorder>>,
    stop: StopConfig,
    stop_checker: StopChecker,
    counter: RefCell<CreditCounter>,
    session_numbers: RefCell<FxHashMap<String, u64>>,
    stop_reached: Notify,
    enforce_stop: bool,
    issuance_gate: Option<Rc<dyn IssuanceGate>>,
    credit_latency_enabled: Cell<bool>,
    cancellation_policy: RefCell<Option<Box<dyn CancellationPolicy>>>,
    policy_phase: Cell<Phase>,
    url_selector: RefCell<Option<Box<dyn UrlSelector>>>,
    session_url_indices: RefCell<FxHashMap<String, u32>>,
    record_processors: RefCell<Vec<Rc<dyn TurnRecordProcessor>>>,
    turn_lifecycle_observer: RefCell<Option<Rc<dyn TurnLifecycleObserver>>>,
    record_processor_tasks: RefCell<Vec<tokio::task::JoinHandle<()>>>,
    record_processor_errors: RefCell<Vec<String>>,
    parallel_report_reduction: Cell<bool>,
}

impl ScheduledRuntime {
    /// Build a runtime. `start_ns` is captured after workload setup so `t=0`
    /// excludes dataset parsing and virtual-history seeding.
    pub fn new(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
        stop: StopConfig,
        enforce_stop: bool,
    ) -> Rc<Self> {
        Self::new_with_metrics_config(
            clock,
            start_ns,
            dispatcher,
            stop,
            enforce_stop,
            MetricsConfig::default(),
        )
    }

    /// Build a runtime with an explicit native metrics policy.
    ///
    /// Workload adapters use this constructor when run-level timeslices or
    /// per-request SLOs were prepared by their owning protocol adapter. The
    /// scheduling and observer topology stays identical to [`Self::new`].
    pub fn new_with_metrics_config(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
        stop: StopConfig,
        enforce_stop: bool,
        metrics: MetricsConfig,
    ) -> Rc<Self> {
        let collector = Rc::new(CollectorObserver::new(true));
        let native_metrics = Rc::new(NativeMetricsObserver::new(clock.clone(), start_ns, metrics));
        let delegates: Vec<Rc<dyn RequestObserver>> =
            vec![collector.clone(), native_metrics.clone()];
        let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
        Self::new_with_observer(
            clock,
            start_ns,
            dispatcher,
            stop,
            enforce_stop,
            collector.clone(),
            native_metrics,
            observer,
            None,
        )
    }

    /// Build a runtime with an injected observer tee and optional external
    /// issuance gate. The collector remains separately owned so final report
    /// construction never depends on observer downcasting.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_observer(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
        stop: StopConfig,
        enforce_stop: bool,
        collector: Rc<CollectorObserver>,
        native_metrics: Rc<NativeMetricsObserver>,
        observer: Rc<dyn RequestObserver>,
        issuance_gate: Option<Rc<dyn IssuanceGate>>,
    ) -> Rc<Self> {
        Rc::new(Self {
            scheduler: Rc::new(ClockTaskScheduler::new(clock.clone())),
            clock,
            start_ns,
            dispatcher,
            collector,
            native_metrics,
            observer,
            recorder: Rc::new(RefCell::new(TimingRecorder::default())),
            stop_checker: StopChecker::new(&stop),
            stop,
            counter: RefCell::new(CreditCounter::default()),
            session_numbers: RefCell::new(FxHashMap::default()),
            stop_reached: Notify::new(),
            enforce_stop,
            issuance_gate,
            credit_latency_enabled: Cell::new(true),
            cancellation_policy: RefCell::new(None),
            policy_phase: Cell::new(Phase::Profiling),
            url_selector: RefCell::new(None),
            session_url_indices: RefCell::new(FxHashMap::default()),
            record_processors: RefCell::new(Vec::new()),
            turn_lifecycle_observer: RefCell::new(None),
            record_processor_tasks: RefCell::new(Vec::new()),
            record_processor_errors: RefCell::new(Vec::new()),
            parallel_report_reduction: Cell::new(false),
        })
    }

    /// Configure post-drain parallel reduction of independent report planes.
    pub fn set_parallel_report_reduction(&self, enabled: bool) {
        self.parallel_report_reduction.set(enabled);
    }

    /// Configure retention of export-only per-turn timing rows.
    pub fn set_timing_record_capture(&self, enabled: bool) {
        self.recorder.borrow_mut().capture_records = enabled;
    }

    /// Attach a terminal record processor before workload execution begins.
    pub fn add_record_processor(&self, processor: Rc<dyn TurnRecordProcessor>) {
        self.record_processors.borrow_mut().push(processor);
    }

    /// Attach one synchronous lifecycle observer before workload execution.
    pub fn set_turn_lifecycle_observer(&self, observer: Rc<dyn TurnLifecycleObserver>) {
        *self.turn_lifecycle_observer.borrow_mut() = Some(observer);
    }

    fn spawn_record_processing(
        self: &Rc<Self>,
        credit: IssuedCredit,
        outcome: TurnDispatchOutcome,
        request_id: Uuid,
        correlation_id: String,
    ) {
        let processors = self.record_processors.borrow().clone();
        if processors.is_empty() {
            return;
        }
        let runtime = self.clone();
        let task = tokio::task::spawn_local(async move {
            for processor in processors {
                if let Err(error) = processor.process(&credit, &outcome).await {
                    runtime.record_processor_errors.borrow_mut().push(format!(
                        "request {request_id} correlation {correlation_id:?}: {error:#}"
                    ));
                }
            }
        });
        self.record_processor_tasks.borrow_mut().push(task);
    }

    pub(crate) async fn wait_record_processors(&self) -> Result<()> {
        let tasks = self
            .record_processor_tasks
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>();
        for task in tasks {
            if let Err(error) = task.await {
                self.record_processor_errors
                    .borrow_mut()
                    .push(format!("terminal record processor task failed: {error}"));
            }
        }
        if let Some(error) = self.record_processor_error() {
            return Err(error);
        }
        Ok(())
    }

    fn record_processor_error(&self) -> Option<anyhow::Error> {
        let errors = self.record_processor_errors.borrow();
        (!errors.is_empty()).then(|| {
            anyhow!(
                "{} terminal record processor error(s): {}",
                errors.len(),
                errors.join("; ")
            )
        })
    }

    /// Install ancillary issuance policies before a workload begins.
    ///
    /// URL selection advances only for turn zero. Its result is stored in the
    /// runtime's session map and copied to every continuation's effective
    /// transport index, while the issued credit retains `url_index=None` for
    /// those continuations.
    pub fn configure_ancillary(
        &self,
        cancellation_policy: Option<Box<dyn CancellationPolicy>>,
        url_selector: Option<Box<dyn UrlSelector>>,
        phase: Phase,
    ) {
        *self.cancellation_policy.borrow_mut() = cancellation_policy;
        *self.url_selector.borrow_mut() = url_selector;
        self.policy_phase.set(phase);
        self.session_url_indices.borrow_mut().clear();
    }

    /// Injected clock.
    pub fn clock(&self) -> Rc<dyn Clock> {
        self.clock.clone()
    }

    /// Run start on the injected clock timeline.
    pub fn start_ns(&self) -> i64 {
        self.start_ns
    }

    /// Warm the dispatch path with one discarded round-trip before timed
    /// issuance begins (see [`TurnDispatcher::prewarm`]).
    pub async fn prewarm(&self, turn: TurnToSend) -> Result<()> {
        self.dispatcher.prewarm(turn).await
    }

    /// Current clock time.
    pub fn now_ns(&self) -> i64 {
        self.clock.now_ns()
    }

    /// Shared local-task scheduler.
    pub fn scheduler(&self) -> Rc<ClockTaskScheduler> {
        self.scheduler.clone()
    }

    /// Select whether credit-relative metrics are present for this workload.
    pub(crate) fn set_credit_latency_enabled(&self, enabled: bool) {
        self.credit_latency_enabled.set(enabled);
    }

    /// True if policy permits another continuation or first turn.
    pub fn can_issue(&self, new_session: bool) -> bool {
        if self
            .issuance_gate
            .as_ref()
            .is_some_and(|gate| !gate.can_issue())
        {
            return false;
        }
        if !self.enforce_stop {
            return true;
        }
        let state = self.counter.borrow().run_state(self.start_ns, false);
        if new_session {
            self.stop_checker
                .can_start_new_session(&state, self.clock.now_ns())
        } else {
            self.stop_checker.can_send_any(&state, self.clock.now_ns())
        }
    }

    /// Issue `turn` without awaiting backend completion.
    ///
    /// Counter mutation and arrival stamping happen synchronously before the
    /// dispatch task is spawned, preserving the single-loop atomicity contract.
    /// Returns `false` when a stop condition rejects the turn; otherwise the
    /// callback runs exactly once after terminal dispatch, including failures.
    pub fn issue_turn(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        on_complete: CompletionHandler,
    ) -> bool {
        self.issue_turn_with_hooks_and_cancellation(
            turn,
            scheduled_ns,
            user_id,
            Box::new(|_ttft_ns| {}),
            on_complete,
            None,
        )
    }

    /// Issue one turn with an externally triggered cancellation latch.
    ///
    /// The callback runs exactly once whether cancellation wins before
    /// dispatch, during streaming, or loses the race to a normal terminal.
    pub fn issue_turn_cancellable(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        on_complete: CompletionHandler,
        cancellation: Rc<dyn DispatchCancellation>,
    ) -> bool {
        self.issue_turn_with_hooks_and_cancellation(
            turn,
            scheduled_ns,
            user_id,
            Box::new(|_ttft_ns| {}),
            on_complete,
            Some(cancellation),
        )
    }

    /// Issue one turn with live endpoint-normalized response frames and an
    /// externally triggered cancellation latch.
    ///
    /// Returns `false` when ordinary issuance policy rejects the turn. A
    /// dispatcher that cannot provide true incremental frames fails the
    /// admitted turn through its normal terminal callback.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_turn_streaming_cancellable(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        responses: Rc<dyn TurnResponseObserver>,
        on_complete: CompletionHandler,
        cancellation: Rc<dyn DispatchCancellation>,
    ) -> bool {
        self.issue_turn_internal(
            turn,
            scheduled_ns,
            user_id,
            Box::new(|_ttft_ns| {}),
            on_complete,
            Some(cancellation),
            Some(responses),
        )
    }

    /// Whether the injected backend supports true incremental response frames.
    pub fn supports_response_streaming(&self) -> bool {
        self.dispatcher.supports_response_streaming()
    }

    /// Issue `turn` with both first-token and terminal lifecycle callbacks.
    ///
    /// `on_first_token` runs synchronously on the local dispatch task when the
    /// backend reports its first meaningful token. `on_complete` still runs
    /// exactly once for every admitted turn, including dispatch failures. This
    /// split lets a workload release a per-request prefill guard at TTFT and
    /// retain terminal as the no-token cleanup path without coupling the
    /// policy-neutral runtime to a concrete slot implementation.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_turn_with_hooks(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        on_first_token: FirstTokenHandler,
        on_complete: CompletionHandler,
    ) -> bool {
        self.issue_turn_with_hooks_and_cancellation(
            turn,
            scheduled_ns,
            user_id,
            on_first_token,
            on_complete,
            None,
        )
    }

    /// Issue one turn with first-token, terminal, and external cancellation hooks.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_turn_with_hooks_and_cancellation(
        self: &Rc<Self>,
        turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        on_first_token: FirstTokenHandler,
        on_complete: CompletionHandler,
        cancellation: Option<Rc<dyn DispatchCancellation>>,
    ) -> bool {
        self.issue_turn_internal(
            turn,
            scheduled_ns,
            user_id,
            on_first_token,
            on_complete,
            cancellation,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn issue_turn_internal(
        self: &Rc<Self>,
        mut turn: TurnToSend,
        scheduled_ns: i64,
        user_id: Option<u64>,
        on_first_token: FirstTokenHandler,
        on_complete: CompletionHandler,
        cancellation: Option<Rc<dyn DispatchCancellation>>,
        responses: Option<Rc<dyn TurnResponseObserver>>,
    ) -> bool {
        let new_session = turn.turn_index == 0;
        if !self.can_issue(new_session) {
            self.stop_reached.notify_waiters();
            return false;
        }

        turn.cancel_after_ns = self
            .cancellation_policy
            .borrow_mut()
            .as_mut()
            .and_then(|policy| policy.next_cancel_delay_ns(self.policy_phase.get()));

        let issued_url_index = if new_session {
            self.url_selector.borrow_mut().as_mut().map(|selector| {
                u32::try_from(selector.next_index())
                    .expect("validated endpoint selector index must fit u32")
            })
        } else {
            None
        };
        if let Some(index) = issued_url_index {
            self.session_url_indices
                .borrow_mut()
                .insert(turn.x_correlation_id.clone(), index);
        }
        turn.url_index = issued_url_index.or_else(|| {
            self.session_url_indices
                .borrow()
                .get(&turn.x_correlation_id)
                .copied()
        });

        let turn_lifecycle_observer = self.turn_lifecycle_observer.borrow().clone();
        if let Some(observer) = &turn_lifecycle_observer {
            observer.on_issue(&turn);
        }

        let (credit_id, final_credit) = self.counter.borrow_mut().increment_sent(&turn, &self.stop);
        let issued_ns = self.clock.now_ns();
        let session_num = {
            let mut sessions = self.session_numbers.borrow_mut();
            if let Some(session_num) = sessions.get(&turn.x_correlation_id) {
                *session_num
            } else {
                let session_num = sessions.len() as u64;
                sessions.insert(turn.x_correlation_id.clone(), session_num);
                session_num
            }
        };
        let record_index = self.recorder.borrow_mut().begin(
            &turn,
            user_id,
            self.start_ns,
            scheduled_ns,
            issued_ns,
        );
        self.native_metrics.register_metadata(
            turn.uuid,
            RequestMetricMetadata {
                request_index: Some(record_index),
                session_num: Some(session_num),
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: self
                    .native_metrics
                    .retains_record_dimensions()
                    .then(|| turn.conversation_id.clone()),
                correlation_id: Some(if self.native_metrics.retains_record_dimensions() {
                    turn.request_correlation_id.clone()
                } else {
                    String::new()
                }),
                audio_duration_s: turn.audio_duration_seconds,
                has_credit_timestamp: self.credit_latency_enabled.get(),
                dimensions: self.dispatcher.inference_dimensions(&turn),
                ..RequestMetricMetadata::default()
            },
        );
        self.observer.on_arrival(
            turn.uuid,
            (issued_ns - self.start_ns) as f64 / 1_000_000.0,
            turn.input_length,
            turn.max_output_tokens,
        );
        let credit = IssuedCredit::from_issued_turn(credit_id, issued_ns, &turn, issued_url_index);

        let runtime = self.clone();
        self.scheduler.execute_async(Box::pin(async move {
            // Every post-dispatch consumer here needs only the Copy `uuid`; the
            // one exception (the optional lifecycle observer's `on_terminal`)
            // reads the full turn from `credit.turn`, which `from_issued_turn`
            // already deep-cloned unconditionally. Capturing `uuid` up front and
            // sourcing the full turn from `credit` lets us MOVE the original
            // `turn` into dispatch instead of deep-cloning its message history +
            // maps on every request (the top profiled allocation hotspot).
            let turn_uuid = turn.uuid;
            let recorder = runtime.recorder.clone();
            let clock = runtime.clock.clone();
            let start_ns = runtime.start_ns;
            let first_token = move |ttft_ns: i64| {
                recorder
                    .borrow_mut()
                    .first_token(record_index, clock.now_ns(), start_ns, ttft_ns);
                if let Some(observer) = &turn_lifecycle_observer {
                    observer.on_first_token(turn_uuid);
                }
                on_first_token(ttft_ns);
            };

            let dispatch = async {
                if let Some(responses) = responses.as_deref() {
                    runtime
                        .dispatcher
                        .dispatch_turn_streaming(
                            turn,
                            runtime.observer.as_ref(),
                            &first_token,
                            responses,
                        )
                        .await
                } else {
                    runtime
                        .dispatcher
                        .dispatch_turn(turn, runtime.observer.as_ref(), &first_token)
                        .await
                }
            };
            tokio::pin!(dispatch);
            let dispatch_result = match cancellation {
                Some(cancellation) if cancellation.is_cancelled() => None,
                Some(cancellation) => {
                    let cancelled = cancellation.cancelled();
                    tokio::pin!(cancelled);
                    tokio::select! {
                        biased;
                        result = &mut dispatch => Some(result),
                        _ = &mut cancelled => None,
                    }
                }
                None => Some(dispatch.await),
            };
            let outcome = match dispatch_result {
                None => {
                    runtime
                        .observer
                        .on_terminal(turn_uuid, ReplayTerminalStatus::Canceled);
                    let now = runtime.clock.now_ns();
                    TurnDispatchOutcome {
                        start_ns: issued_ns,
                        end_ns: now,
                        terminal: ReplayTerminalStatus::Canceled,
                        response_text: String::new(),
                        model_response: ModelResponseMetadata {
                            error_kind: Some("dispatch_cancelled".to_string()),
                            error_message: Some(
                                "dispatch cancelled by the owning workload".to_string(),
                            ),
                            ..ModelResponseMetadata::default()
                        },
                        prompt_tokens: None,
                        completion_tokens: None,
                        http: RequestTrace::default(),
                    }
                }
                Some(Ok(outcome)) => outcome,
                Some(Err(error)) => {
                    tracing::warn!(
                        uuid = %turn_uuid,
                        error = %error,
                        "scheduled turn dispatch failed"
                    );
                    runtime
                        .observer
                        .on_terminal(turn_uuid, ReplayTerminalStatus::Failed);
                    let now = runtime.clock.now_ns();
                    TurnDispatchOutcome {
                        start_ns: issued_ns,
                        end_ns: now,
                        terminal: ReplayTerminalStatus::Failed,
                        response_text: String::new(),
                        model_response: ModelResponseMetadata {
                            error_kind: Some("turn_dispatch_error".to_string()),
                            error_message: Some(error.to_string()),
                            ..ModelResponseMetadata::default()
                        },
                        prompt_tokens: None,
                        completion_tokens: None,
                        http: RequestTrace::default(),
                    }
                }
            };
            if let Some(observer) = runtime.turn_lifecycle_observer.borrow().as_ref() {
                // `credit.turn` is the same unconditionally-cloned turn snapshot,
                // taken before dispatch and never mutated, so it is byte-identical
                // to the original turn the observer previously observed at issue.
                observer.on_terminal(&credit.turn, &outcome);
            }
            runtime
                .recorder
                .borrow_mut()
                .terminal(record_index, &outcome, runtime.start_ns);
            runtime.native_metrics.record_response(
                turn_uuid,
                NativeResponseMetadata {
                    start_ns: Some(outcome.start_ns),
                    end_ns: Some(outcome.end_ns),
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                    http: outcome.http,
                },
            );
            let processor_input = (!runtime.record_processors.borrow().is_empty()).then(|| {
                (
                    credit.clone(),
                    outcome.clone(),
                    credit.turn.request_correlation_id.clone(),
                )
            });
            if credit.is_final_turn() {
                runtime
                    .session_url_indices
                    .borrow_mut()
                    .remove(&credit.turn.x_correlation_id);
            }
            on_complete(credit, outcome).await;
            // Return the credit/release admission before downstream processing.
            // The detached local task also keeps grading latency out of the
            // scheduler's dispatch-drain boundary and performance wall time.
            if let Some((processor_credit, processor_outcome, correlation_id)) = processor_input {
                runtime.spawn_record_processing(
                    processor_credit,
                    processor_outcome,
                    turn_uuid,
                    correlation_id,
                );
            }
        }));

        if final_credit {
            self.stop_reached.notify_waiters();
        }
        true
    }

    /// Wait until `target_ns`, returning `false` if a stop condition becomes
    /// active first. The duration bound itself is included even when no other
    /// issuance occurs to send a notification.
    pub async fn wait_until_or_stop(&self, target_ns: i64) -> bool {
        loop {
            let stop_event = self.stop_reached.notified();
            if !self.can_issue(false) {
                return false;
            }
            let mut effective_target = target_ns;
            if self.enforce_stop
                && let Some(duration_ns) = self.stop.expected_duration_ns
            {
                effective_target = effective_target.min(self.start_ns.saturating_add(duration_ns));
            }
            let wait_ns = effective_target.saturating_sub(self.clock.now_ns());
            if wait_ns <= 0 {
                return self.can_issue(false) && self.clock.now_ns() >= target_ns;
            }
            let sleep = self.clock.clone().sleep(wait_ns);
            tokio::pin!(sleep);
            tokio::pin!(stop_event);
            tokio::select! {
                _ = &mut sleep => {
                    if self.clock.now_ns() < target_ns && !self.can_issue(false) {
                        return false;
                    }
                    return self.clock.now_ns() >= target_ns;
                }
                _ = &mut stop_event => {
                    if !self.can_issue(false) {
                        return false;
                    }
                }
            }
        }
    }

    /// Freeze the collector and detailed timing trace into a report.
    pub fn finish(
        &self,
        strategy: impl Into<String>,
        user_control: Option<UserControlSnapshot>,
    ) -> ScheduledRunReport {
        self.finish_at(self.clock.now_ns(), strategy, user_control)
    }

    /// Freeze the collector at a previously captured absolute clock boundary.
    ///
    /// A virtual-time backend can capture this after its last live event, exit
    /// the Tokio `LocalSet`, and perform the aggregate/report reduction later
    /// without changing the run's effective wall time or terminal fallbacks.
    pub fn finish_at(
        &self,
        end_ns: i64,
        strategy: impl Into<String>,
        user_control: Option<UserControlSnapshot>,
    ) -> ScheduledRunReport {
        let wall_ms = end_ns.saturating_sub(self.start_ns) as f64 / 1_000_000.0;
        let (turns, schedule_timing) = {
            let mut recorder = self.recorder.borrow_mut();
            let analysis = recorder.analysis();
            (std::mem::take(&mut recorder.records), analysis)
        };
        let collector = self.collector.take();
        let native_metrics = self.native_metrics.take_finalizer_at(end_ns);
        let (performance, native_metrics) = if self.parallel_report_reduction.get() {
            rayon::join(
                || collector.finish().with_wall_time_ms(wall_ms),
                || native_metrics.finish(),
            )
        } else {
            (
                collector.finish().with_wall_time_ms(wall_ms),
                native_metrics.finish(),
            )
        };
        ScheduledRunReport {
            strategy: strategy.into(),
            performance,
            native_metrics,
            schedule_timing,
            turns,
            user_control,
        }
    }
}

/// Schedule-generating workload seam shared across online and offline backends.
#[async_trait(?Send)]
pub trait Workload {
    /// Stable strategy label used in reports.
    fn name(&self) -> &'static str;

    /// Generate and drain all scheduled work through `runtime`.
    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()>;

    /// Optional final user-control snapshot.
    fn user_control_snapshot(&self) -> Option<UserControlSnapshot> {
        None
    }

    /// Whether issuance has a credit timestamp distinct from a fixed authored
    /// schedule. Disabling this omits credit-to-start/effective latency metrics.
    fn has_credit_timestamps(&self) -> bool {
        true
    }
}

/// One-pass, single-turn dataset workload over the ordinary scheduled runtime.
///
/// This is not accuracy-specific: it is the dataset equivalent of the synthetic
/// closed-loop path. The supplied [`ConversationSource`] owns sampling and
/// endpoint materialization; this workload owns only bounded issuance.
pub struct SingleTurnDatasetWorkload {
    conversations: Rc<RefCell<Box<dyn ConversationSource>>>,
    request_count: usize,
    slots: Rc<SlotPool>,
}

impl SingleTurnDatasetWorkload {
    /// Validate a non-empty single-turn source and concurrency limit.
    pub fn new(conversations: Box<dyn ConversationSource>, concurrency: usize) -> Result<Self> {
        if concurrency == 0 {
            return Err(anyhow!("dataset concurrency must be greater than zero"));
        }
        let request_count = conversations.conversations().len();
        if request_count == 0 {
            return Err(anyhow!("single-turn dataset has no conversations"));
        }
        for conversation in conversations.conversations() {
            if conversation.turns.len() != 1 {
                return Err(anyhow!(
                    "single-turn dataset conversation {:?} has {} turns",
                    conversation.conversation_id,
                    conversation.turns.len()
                ));
            }
        }
        Ok(Self {
            conversations: Rc::new(RefCell::new(conversations)),
            request_count,
            slots: Rc::new(SlotPool::new(concurrency)),
        })
    }

    /// Session admission pool used by optional normal-pipeline actuators.
    pub fn session_slots(&self) -> Rc<SlotPool> {
        self.slots.clone()
    }
}

#[async_trait(?Send)]
impl Workload for SingleTurnDatasetWorkload {
    fn name(&self) -> &'static str {
        "single_turn_dataset"
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        for _ in 0..self.request_count {
            let guard = self.slots.acquire().await;
            let session = self.conversations.borrow_mut().next(None)?;
            let turn = session.build_first_turn(Some(1))?;
            let issued = runtime.issue_turn(
                turn,
                runtime.now_ns(),
                None,
                Box::new(move |_credit, _outcome| {
                    Box::pin(async move {
                        drop(guard);
                    })
                }),
            );
            if !issued {
                break;
            }
        }
        Ok(())
    }
}

/// Ancillary policies injected into the scheduled issuer.
pub struct ScheduledAncillaryPolicies {
    /// Per-turn post-send cancellation decisions.
    pub cancellation_policy: Option<Box<dyn CancellationPolicy>>,
    /// Turn-0 endpoint selector; continuation pinning is owned by the runtime.
    pub url_selector: Option<Box<dyn UrlSelector>>,
    /// Phase passed to cancellation policy (warmup disables cancellation).
    pub phase: Phase,
}

impl Default for ScheduledAncillaryPolicies {
    fn default() -> Self {
        Self {
            cancellation_policy: None,
            url_selector: None,
            phase: Phase::Profiling,
        }
    }
}

/// Drive an already-prepared workload from `start_ns` to quiescence.
pub async fn run_scheduled_workload(
    workload: Rc<dyn Workload>,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
    enforce_stop: bool,
) -> Result<ScheduledRunReport> {
    run_scheduled_workload_with_ancillary(
        workload,
        clock,
        start_ns,
        dispatcher,
        stop,
        enforce_stop,
        ScheduledAncillaryPolicies::default(),
    )
    .await
}

/// Drive a prepared workload with cancellation and endpoint-selection policy.
#[allow(clippy::too_many_arguments)]
pub async fn run_scheduled_workload_with_ancillary(
    workload: Rc<dyn Workload>,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
    enforce_stop: bool,
    policies: ScheduledAncillaryPolicies,
) -> Result<ScheduledRunReport> {
    run_scheduled_workload_with_processors(
        workload,
        clock,
        start_ns,
        dispatcher,
        stop,
        enforce_stop,
        policies,
        Vec::new(),
    )
    .await
}

/// Drive a prepared workload through the normal pipeline with terminal processors.
#[allow(clippy::too_many_arguments)]
pub async fn run_scheduled_workload_with_processors(
    workload: Rc<dyn Workload>,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
    enforce_stop: bool,
    policies: ScheduledAncillaryPolicies,
    record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
) -> Result<ScheduledRunReport> {
    let config =
        crate::timing::PhaseConfig::new("profiling", crate::timing::PhaseKind::Profiling, stop)
            // Single-phase callers drain admitted work; multi-phase callers
            // choose grace policy explicitly.
            .with_grace_period(crate::timing::GracePeriod::Infinite);
    let plan = crate::phase_runtime::ScheduledPhasePlan::new(config, workload, policies)
        .with_enforce_stop(enforce_stop)
        .with_start_ns(start_ns)
        .with_record_processors(record_processors);
    let observer: Rc<dyn crate::timing::PhaseObserver> = Rc::new(crate::timing::NoopPhaseObserver);
    let mut result =
        crate::phase_runtime::run_scheduled_phases(vec![plan], clock, dispatcher, observer).await?;
    let phase = result
        .reports
        .pop()
        .ok_or_else(|| anyhow!("profiling phase completed without a scheduled report"))?;
    debug_assert!(result.reports.is_empty());
    Ok(phase.report)
}
