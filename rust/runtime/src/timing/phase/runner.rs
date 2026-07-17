// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-native single-phase execution driver.
//!
//! The ordering in [`ClockPhaseRunner`] is: configure → setup →
//! start → progress → ramps → issuance → sending timeout/freeze → return grace
//! → cancel → drain → force completion. Transport, workload, scheduler, and
//! ramp ownership stay behind [`PhaseExecution`], so the same driver works with
//! real HTTP, an HTTP mock, or an in-process simulated sink.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use crate::clock::Clock;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use super::config::DISABLED_PROGRESS_INTERVAL_NS;
use crate::timing::{RunState, StopChecker};

use super::{
    PhaseBranchStats, PhaseCompletionReason, PhaseConfig, PhaseConfigError, PhaseLifecycle,
    PhaseLifecycleError, PhaseObserver, PhaseProgress, PhaseProgressError, PhaseReturn,
    PhaseReturnOutcome, PhaseSend, PhaseSendOutcome, PhaseStats,
};

/// Boxed `!Send` future used by phase extension traits.
pub type LocalPhaseFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

/// Error returned by an injected phase execution strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseExecutionError {
    message: String,
}

impl PhaseExecutionError {
    /// Create an execution error with user-facing context.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The execution error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for PhaseExecutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for PhaseExecutionError {}

/// Slots recovered after cancelled requests failed to return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReleasedStuckSlots {
    /// Session slots recovered.
    pub session: u64,
    /// Prefill slots recovered.
    pub prefill: u64,
}

struct CancellationSignal {
    cancelled: Cell<bool>,
    notify: Notify,
}

impl CancellationSignal {
    fn new() -> Self {
        Self {
            cancelled: Cell::new(false),
            notify: Notify::new(),
        }
    }

    fn cancel(&self) {
        if !self.cancelled.replace(true) {
            self.notify.notify_waiters();
        }
    }

    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.cancelled.get() {
                return;
            }
            notified.await;
        }
    }
}

/// Local-loop context shared with one phase execution strategy.
///
/// Issuers mutate progress synchronously and dispatch asynchronously; lifecycle
/// transitions remain runner-owned.
#[derive(Clone)]
pub struct PhaseContext {
    clock: Rc<dyn Clock>,
    lifecycle: Rc<RefCell<PhaseLifecycle>>,
    progress: PhaseProgress,
    stop_checker: Rc<StopChecker>,
    cancellation: Rc<CancellationSignal>,
}

impl PhaseContext {
    fn new(
        clock: Rc<dyn Clock>,
        lifecycle: Rc<RefCell<PhaseLifecycle>>,
        progress: PhaseProgress,
        stop_checker: Rc<StopChecker>,
        cancellation: Rc<CancellationSignal>,
    ) -> Self {
        Self {
            clock,
            lifecycle,
            progress,
            stop_checker,
            cancellation,
        }
    }

    fn run_state(&self) -> RunState {
        let lifecycle = self.lifecycle.borrow().snapshot();
        let counters = self.progress.snapshot();
        RunState {
            requests_sent: counters.requests_sent,
            root_requests_sent: counters.root_requests_sent,
            sent_sessions: counters.sent_sessions,
            total_session_turns: counters.total_session_turns,
            cancelled: lifecycle.was_cancelled,
            sending_complete: matches!(
                lifecycle.state,
                super::PhaseState::SendingComplete | super::PhaseState::Complete
            ),
            started_at_ns: lifecycle
                .started_at_ns
                .unwrap_or_else(|| self.clock.now_ns()),
        }
    }

    /// Injected clock used by every phase timer and execution strategy.
    pub fn clock(&self) -> Rc<dyn Clock> {
        self.clock.clone()
    }

    /// Clone the progress handle for direct observer/callback wiring.
    pub fn progress(&self) -> PhaseProgress {
        self.progress.clone()
    }

    /// Whether any additional root or continuation request may be issued.
    pub fn can_send_any(&self) -> bool {
        self.stop_checker
            .can_send_any(&self.run_state(), self.clock.now_ns())
    }

    /// Whether a new root session may be admitted.
    pub fn can_start_new_session(&self) -> bool {
        self.stop_checker
            .can_start_new_session(&self.run_state(), self.clock.now_ns())
    }

    /// Atomically record one issued request.
    pub fn record_sent(&self, sent: PhaseSend) -> Result<PhaseSendOutcome, PhaseProgressError> {
        self.progress.record_sent(sent)
    }

    /// Atomically record one admitted request batch before stop evaluation.
    pub fn record_sent_batch(
        &self,
        sent: &[PhaseSend],
    ) -> Result<Vec<PhaseSendOutcome>, PhaseProgressError> {
        self.progress.record_sent_batch(sent)
    }

    /// Atomically record one terminal request.
    pub fn record_returned(&self, returned: PhaseReturn) -> PhaseReturnOutcome {
        self.progress.record_returned(returned)
    }

    /// Record a first-token prefill release.
    pub fn record_first_token(&self) {
        self.progress.record_first_token();
    }

    /// Mark natural issuer exhaustion when no configured count bound fired.
    pub fn mark_all_sent(&self) {
        self.progress.mark_sending_complete();
    }

    /// Register branch work that must drain before phase completion.
    pub fn begin_branch_work(&self) {
        self.progress.begin_branch_work();
    }

    /// Finish one unit of branch work and re-evaluate completion.
    pub fn finish_branch_work(&self) -> Result<bool, PhaseProgressError> {
        self.progress.finish_branch_work()
    }

    /// Whether external cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.cancelled.get()
    }

    /// Wait for external cancellation.
    pub fn wait_cancelled(&self) -> LocalPhaseFuture<()> {
        let cancellation = self.cancellation.clone();
        Box::pin(async move { cancellation.wait().await })
    }
}

/// Workload/backend adapter driven by [`PhaseRunner`].
///
/// The default lifecycle hooks are no-ops. Implementations own their scheduler,
/// dispatch tasks, shared slot pools, and ramp handles; the runner owns their
/// ordering and deadlines. Methods are deliberately `!Send` and object-safe.
pub trait PhaseExecution {
    /// Apply this phase's caps to shared long-lived execution state.
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        Ok(())
    }

    /// Perform asynchronous workload setup before the lifecycle starts.
    fn setup(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }

    /// Start phase ramps before the first request can be issued.
    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        Ok(())
    }

    /// Execute the issuance strategy until its sending plan is exhausted.
    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>>;

    /// Synchronously prevent the issuance loop from starting more work.
    fn stop_issuing(&self) {}

    /// Cancel deferred turns that have not begun dispatch.
    fn cancel_pending(&self) {}

    /// Ask every in-flight backend dispatch to cancel.
    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }

    /// Recover admission slots held by requests that will never return.
    fn release_stuck_slots(&self) -> ReleasedStuckSlots {
        ReleasedStuckSlots::default()
    }

    /// Stop and join every phase-owned ramp task.
    fn stop_ramps(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }

    /// Finalize phase-owned reports after every return has drained.
    ///
    /// This is intentionally separate from [`stop_ramps`](Self::stop_ramps): a
    /// seamless phase stops actuators at sending handoff but finalizes metrics
    /// only when its background return wait reaches COMPLETE.
    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }

    /// Optional dataflow counters included with the terminal observation.
    fn branch_stats(&self) -> Option<PhaseBranchStats> {
        None
    }
}

/// Factory seam creating fresh execution state for each phase.
pub trait PhaseExecutionFactory {
    /// Create one phase execution adapter over the runner-owned context.
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution>;

    /// Cancel in-flight work shared across every active phase.
    ///
    /// The orchestrator invokes this before signalling individual runners.
    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }
}

/// Concrete execution adapter that immediately exhausts an empty plan.
pub struct NoopPhaseExecution;

impl PhaseExecution for NoopPhaseExecution {
    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }
}

/// Factory for [`NoopPhaseExecution`], useful for lifecycle-only consumers.
#[derive(Default)]
pub struct NoopPhaseExecutionFactory;

impl PhaseExecutionFactory for NoopPhaseExecutionFactory {
    fn create(&self, _config: &PhaseConfig, _context: PhaseContext) -> Rc<dyn PhaseExecution> {
        Rc::new(NoopPhaseExecution)
    }
}

/// Object-safe single-phase driver seam.
pub trait PhaseRunner {
    /// Copy the phase configuration.
    fn config(&self) -> PhaseConfig;

    /// Run through sending, returning early only for seamless non-final phases.
    fn run(&self, is_final_phase: bool) -> LocalPhaseFuture<Result<PhaseStats, PhaseRunError>>;

    /// Wait for terminal phase completion after a seamless handoff.
    fn wait_complete(&self) -> LocalPhaseFuture<Result<PhaseStats, PhaseRunError>>;

    /// Request external cancellation without blocking the local loop.
    fn cancel(&self);

    /// Whether terminal finalization has completed.
    fn is_complete(&self) -> bool;
}

struct RunnerInner {
    config: PhaseConfig,
    clock: Rc<dyn Clock>,
    lifecycle: Rc<RefCell<PhaseLifecycle>>,
    progress: PhaseProgress,
    execution: Rc<dyn PhaseExecution>,
    observer: Rc<dyn PhaseObserver>,
    cancellation: Rc<CancellationSignal>,
    run_started: Cell<bool>,
    progress_stopped: Cell<bool>,
    progress_stop: Notify,
    progress_task: RefCell<Option<JoinHandle<()>>>,
    return_task: RefCell<Option<JoinHandle<()>>>,
    completion: RefCell<Option<Result<PhaseStats, PhaseRunError>>>,
    completion_notify: Notify,
}

/// Default clock-driven implementation of [`PhaseRunner`].
#[derive(Clone)]
pub struct ClockPhaseRunner {
    inner: Rc<RunnerInner>,
}

impl ClockPhaseRunner {
    /// Validate `config` and construct a runner with fresh per-phase state.
    pub fn new(
        config: PhaseConfig,
        clock: Rc<dyn Clock>,
        observer: Rc<dyn PhaseObserver>,
        execution_factory: Rc<dyn PhaseExecutionFactory>,
    ) -> Result<Self, PhaseRunError> {
        config.validate().map_err(PhaseRunError::InvalidConfig)?;
        let lifecycle = Rc::new(RefCell::new(PhaseLifecycle::new(clock.clone(), &config)));
        let progress = PhaseProgress::new(config.stop);
        let stop_checker = Rc::new(StopChecker::new(&config.stop));
        let cancellation = Rc::new(CancellationSignal::new());
        let context = PhaseContext::new(
            clock.clone(),
            lifecycle.clone(),
            progress.clone(),
            stop_checker,
            cancellation.clone(),
        );
        let execution = execution_factory.create(&config, context);
        Ok(Self {
            inner: Rc::new(RunnerInner {
                config,
                clock,
                lifecycle,
                progress,
                execution,
                observer,
                cancellation,
                run_started: Cell::new(false),
                progress_stopped: Cell::new(false),
                progress_stop: Notify::new(),
                progress_task: RefCell::new(None),
                return_task: RefCell::new(None),
                completion: RefCell::new(None),
                completion_notify: Notify::new(),
            }),
        })
    }

    /// Clone the runner-owned context for direct callback integration.
    pub fn context(&self) -> PhaseContext {
        PhaseContext::new(
            self.inner.clock.clone(),
            self.inner.lifecycle.clone(),
            self.inner.progress.clone(),
            Rc::new(StopChecker::new(&self.inner.config.stop)),
            self.inner.cancellation.clone(),
        )
    }

    fn stats(&self) -> PhaseStats {
        PhaseStats::snapshot(
            &self.inner.config,
            &self.inner.lifecycle.borrow(),
            &self.inner.progress,
        )
    }

    async fn run_entry(self, is_final_phase: bool) -> Result<PhaseStats, PhaseRunError> {
        if self.inner.run_started.replace(true) {
            return Err(PhaseRunError::AlreadyRun(self.inner.config.id.clone()));
        }

        match self.run_inner(is_final_phase).await {
            Ok(RunDisposition::Complete(stats)) => {
                self.store_completion(Ok(stats.clone()));
                Ok(stats)
            }
            Ok(RunDisposition::SeamlessHandoff(stats)) => Ok(stats),
            Err(error) => {
                self.finalize_failure().await;
                self.store_completion(Err(error.clone()));
                Err(error)
            }
        }
    }

    async fn run_inner(&self, is_final_phase: bool) -> Result<RunDisposition, PhaseRunError> {
        self.inner
            .execution
            .configure(&self.inner.config)
            .map_err(PhaseRunError::Execution)?;
        self.inner
            .execution
            .setup()
            .await
            .map_err(PhaseRunError::Execution)?;

        self.inner.lifecycle.borrow_mut().start()?;
        self.inner
            .observer
            .on_phase_start(&self.inner.config, self.stats());
        self.start_progress_loop();

        let sending_outcome = if self.inner.cancellation.cancelled.get() {
            self.finish_sending(false);
            WaitOutcome::Cancelled
        } else {
            self.inner
                .execution
                .start_ramps()
                .map_err(PhaseRunError::Execution)?;
            self.execute_until_sending_complete().await?
        };

        if sending_outcome == WaitOutcome::Cancelled {
            self.inner
                .execution
                .cancel_inflight()
                .await
                .map_err(PhaseRunError::Execution)?;
            self.inner
                .execution
                .stop_ramps()
                .await
                .map_err(PhaseRunError::Execution)?;
            let stats = self
                .complete_phase(PhaseCompletionReason::Cancelled)
                .await?;
            return Ok(RunDisposition::Complete(stats));
        }

        if self.inner.config.seamless && !is_final_phase {
            let runner = self.clone();
            let task = tokio::task::spawn_local(async move {
                let result = runner.finish_returning().await;
                if result.is_err() {
                    runner.finalize_failure().await;
                }
                runner.store_completion(result);
            });
            *self.inner.return_task.borrow_mut() = Some(task);
            self.inner
                .execution
                .stop_ramps()
                .await
                .map_err(PhaseRunError::Execution)?;
            Ok(RunDisposition::SeamlessHandoff(self.stats()))
        } else {
            let stats = self.finish_returning().await?;
            self.inner
                .execution
                .stop_ramps()
                .await
                .map_err(PhaseRunError::Execution)?;
            Ok(RunDisposition::Complete(stats))
        }
    }

    async fn execute_until_sending_complete(&self) -> Result<WaitOutcome, PhaseRunError> {
        let error = Rc::new(RefCell::new(None));
        let execution = self.inner.execution.clone();
        let progress = self.inner.progress.clone();
        let error_for_task = error.clone();
        let mut task = tokio::task::spawn_local(async move {
            if let Err(execution_error) = execution.execute().await {
                *error_for_task.borrow_mut() = Some(execution_error);
            }
            progress.mark_sending_complete();
        });

        let timeout = self.inner.lifecycle.borrow().time_left_ns(false);
        let outcome = self.wait_for_sent(timeout).await;
        if outcome != WaitOutcome::Event {
            self.inner.execution.stop_issuing();
        }
        if !task.is_finished() {
            task.abort();
        }
        let _ = (&mut task).await;

        self.finish_sending(outcome == WaitOutcome::TimedOut);
        if let Some(error) = error.borrow_mut().take() {
            return Err(PhaseRunError::Execution(error));
        }
        Ok(outcome)
    }

    fn finish_sending(&self, timed_out: bool) {
        if !self.inner.lifecycle.borrow().is_sending_complete() {
            self.inner
                .lifecycle
                .borrow_mut()
                .mark_sending_complete(timed_out)
                .expect("runner preserves lifecycle transition order");
        }
        self.inner.progress.freeze_sent_counts();
        self.inner.execution.cancel_pending();
        self.inner.progress.signal_all_sent();
        let stats = self.stats();
        self.inner.observer.on_progress(stats.clone());
        self.inner.observer.on_sending_complete(stats);
    }

    async fn finish_returning(&self) -> Result<PhaseStats, PhaseRunError> {
        if self.inner.lifecycle.borrow().is_complete() {
            return Ok(self.stats());
        }

        let wait = if self.inner.progress.check_all_returned_or_cancelled() {
            WaitOutcome::Event
        } else {
            let timeout = self.inner.lifecycle.borrow().time_left_ns(true);
            self.wait_for_returned(timeout, true).await
        };

        let mut cancellation_error = None;
        let reason = match wait {
            WaitOutcome::Event => PhaseCompletionReason::Completed,
            WaitOutcome::TimedOut | WaitOutcome::Cancelled => {
                if let Err(error) = self.inner.execution.cancel_inflight().await {
                    cancellation_error = Some(error);
                }
                let drained = self
                    .wait_for_returned(Some(self.inner.config.cancel_drain_timeout_ns), false)
                    .await
                    == WaitOutcome::Event;
                if drained {
                    if wait == WaitOutcome::TimedOut {
                        PhaseCompletionReason::GraceTimeout
                    } else {
                        PhaseCompletionReason::Cancelled
                    }
                } else {
                    self.inner
                        .lifecycle
                        .borrow_mut()
                        .mark_cancel_drain_timeout();
                    let released = self.inner.execution.release_stuck_slots();
                    self.inner
                        .progress
                        .record_stuck_slots_released(released.session, released.prefill);
                    self.inner.progress.force_all_returned();
                    PhaseCompletionReason::ForceCompleted
                }
            }
        };

        let stats = self.complete_phase(reason).await?;
        if let Some(error) = cancellation_error {
            return Err(PhaseRunError::Execution(error));
        }
        Ok(stats)
    }

    async fn complete_phase(
        &self,
        reason: PhaseCompletionReason,
    ) -> Result<PhaseStats, PhaseRunError> {
        if !self.inner.lifecycle.borrow().is_complete() {
            self.inner.lifecycle.borrow_mut().mark_complete(reason)?;
            self.inner.progress.freeze_completed_counts();
        }
        if reason == PhaseCompletionReason::Cancelled {
            self.inner.progress.force_all_returned();
        }
        let finalize_result = self.inner.execution.finalize().await;
        let stats = self.stats();
        self.inner.observer.on_progress(stats.clone());
        self.inner
            .observer
            .on_phase_complete(stats.clone(), self.inner.execution.branch_stats());
        self.stop_progress_loop();
        finalize_result.map_err(PhaseRunError::Execution)?;
        Ok(stats)
    }

    async fn finalize_failure(&self) {
        self.inner.execution.stop_issuing();
        self.inner.execution.cancel_pending();
        let _ = self.inner.execution.cancel_inflight().await;
        let _ = self.inner.execution.stop_ramps().await;

        let lifecycle_needs_start = !self.inner.lifecycle.borrow().is_started();
        if lifecycle_needs_start && self.inner.lifecycle.borrow_mut().start().is_ok() {
            self.inner
                .observer
                .on_phase_start(&self.inner.config, self.stats());
        }
        if !self.inner.lifecycle.borrow().is_sending_complete() {
            self.finish_sending(false);
        }
        if !self.inner.lifecycle.borrow().is_complete() {
            self.inner.progress.force_all_returned();
            let _ = self.complete_phase(PhaseCompletionReason::Failed).await;
        }
        self.stop_progress_loop();
    }

    async fn wait_for_sent(&self, timeout_ns: Option<i64>) -> WaitOutcome {
        if self.inner.progress.all_sent() {
            return WaitOutcome::Event;
        }
        if self.inner.cancellation.cancelled.get() {
            return WaitOutcome::Cancelled;
        }
        let sent = self.inner.progress.wait_all_sent();
        let cancelled = self.inner.cancellation.wait();
        tokio::pin!(sent);
        tokio::pin!(cancelled);
        match timeout_ns {
            Some(timeout) if timeout <= 0 => WaitOutcome::TimedOut,
            Some(timeout) => {
                let sleep = self.inner.clock.clone().sleep(timeout);
                tokio::pin!(sleep);
                tokio::select! {
                    biased;
                    _ = &mut cancelled => WaitOutcome::Cancelled,
                    _ = &mut sent => WaitOutcome::Event,
                    _ = &mut sleep => WaitOutcome::TimedOut,
                }
            }
            None => {
                tokio::select! {
                    biased;
                    _ = &mut cancelled => WaitOutcome::Cancelled,
                    _ = &mut sent => WaitOutcome::Event,
                }
            }
        }
    }

    async fn wait_for_returned(
        &self,
        timeout_ns: Option<i64>,
        observe_cancellation: bool,
    ) -> WaitOutcome {
        if self.inner.progress.all_returned() {
            return WaitOutcome::Event;
        }
        if observe_cancellation && self.inner.cancellation.cancelled.get() {
            return WaitOutcome::Cancelled;
        }
        let returned = self.inner.progress.wait_all_returned();
        tokio::pin!(returned);
        match timeout_ns {
            Some(timeout) if timeout <= 0 => WaitOutcome::TimedOut,
            Some(timeout) => {
                let sleep = self.inner.clock.clone().sleep(timeout);
                tokio::pin!(sleep);
                if observe_cancellation {
                    let cancelled = self.inner.cancellation.wait();
                    tokio::pin!(cancelled);
                    tokio::select! {
                        biased;
                        _ = &mut cancelled => WaitOutcome::Cancelled,
                        _ = &mut returned => WaitOutcome::Event,
                        _ = &mut sleep => WaitOutcome::TimedOut,
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = &mut returned => WaitOutcome::Event,
                        _ = &mut sleep => WaitOutcome::TimedOut,
                    }
                }
            }
            None if observe_cancellation => {
                let cancelled = self.inner.cancellation.wait();
                tokio::pin!(cancelled);
                tokio::select! {
                    biased;
                    _ = &mut cancelled => WaitOutcome::Cancelled,
                    _ = &mut returned => WaitOutcome::Event,
                }
            }
            None => {
                returned.await;
                WaitOutcome::Event
            }
        }
    }

    fn start_progress_loop(&self) {
        self.inner.progress_stopped.set(false);
        let runner = self.clone();
        let task = tokio::task::spawn_local(async move {
            loop {
                if runner.inner.progress_stopped.get() {
                    return;
                }
                runner.inner.observer.on_progress(runner.stats());
                let stopped = runner.inner.progress_stop.notified();
                tokio::pin!(stopped);
                if runner.inner.config.progress_interval_ns == DISABLED_PROGRESS_INTERVAL_NS {
                    // Periodic progress is disabled: emit the opening snapshot and
                    // then wait only for the terminal stop, scheduling NO
                    // intermediate clock event. Used by the offline `execute_pass`
                    // single engine, which cannot stop at a finite clock deadline
                    // — a periodic progress sleep would be exactly such a deadline.
                    stopped.await;
                    if runner.inner.progress_stopped.get() {
                        return;
                    }
                    continue;
                }
                let sleep = runner
                    .inner
                    .clock
                    .clone()
                    .sleep(runner.inner.config.progress_interval_ns);
                tokio::pin!(sleep);
                tokio::select! {
                    _ = &mut stopped => {
                        if runner.inner.progress_stopped.get() {
                            return;
                        }
                    }
                    _ = &mut sleep => {}
                }
            }
        });
        *self.inner.progress_task.borrow_mut() = Some(task);
    }

    fn stop_progress_loop(&self) {
        if !self.inner.progress_stopped.replace(true) {
            self.inner.progress_stop.notify_waiters();
        }
    }

    fn store_completion(&self, result: Result<PhaseStats, PhaseRunError>) {
        let mut completion = self.inner.completion.borrow_mut();
        if completion.is_none() {
            *completion = Some(result);
            self.inner.completion_notify.notify_waiters();
        }
    }

    async fn wait_for_completion(&self) -> Result<PhaseStats, PhaseRunError> {
        loop {
            let notified = self.inner.completion_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.inner.completion.borrow().clone() {
                return result;
            }
            notified.await;
        }
    }
}

impl PhaseRunner for ClockPhaseRunner {
    fn config(&self) -> PhaseConfig {
        self.inner.config.clone()
    }

    fn run(&self, is_final_phase: bool) -> LocalPhaseFuture<Result<PhaseStats, PhaseRunError>> {
        let runner = self.clone();
        Box::pin(async move { runner.run_entry(is_final_phase).await })
    }

    fn wait_complete(&self) -> LocalPhaseFuture<Result<PhaseStats, PhaseRunError>> {
        let runner = self.clone();
        Box::pin(async move { runner.wait_for_completion().await })
    }

    fn cancel(&self) {
        self.inner.lifecycle.borrow_mut().cancel();
        self.inner.execution.stop_issuing();
        self.inner.execution.cancel_pending();
        self.inner.cancellation.cancel();
    }

    fn is_complete(&self) -> bool {
        self.inner.lifecycle.borrow().is_complete()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaitOutcome {
    Event,
    TimedOut,
    Cancelled,
}

enum RunDisposition {
    Complete(PhaseStats),
    SeamlessHandoff(PhaseStats),
}

/// Error returned by a phase runner or orchestrator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseRunError {
    /// Local phase configuration was invalid.
    InvalidConfig(PhaseConfigError),
    /// A lifecycle transition invariant was violated.
    Lifecycle(PhaseLifecycleError),
    /// The injected execution strategy failed.
    Execution(PhaseExecutionError),
    /// The same runner was started more than once.
    AlreadyRun(String),
}

impl Display for PhaseRunError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(f, "invalid phase configuration: {error}"),
            Self::Lifecycle(error) => write!(f, "invalid phase lifecycle transition: {error}"),
            Self::Execution(error) => write!(f, "phase execution failed: {error}"),
            Self::AlreadyRun(id) => write!(f, "phase {id:?} was already run"),
        }
    }
}

impl Error for PhaseRunError {}

impl From<PhaseLifecycleError> for PhaseRunError {
    fn from(value: PhaseLifecycleError) -> Self {
        Self::Lifecycle(value)
    }
}
