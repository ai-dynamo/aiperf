// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-paced, single-flight telemetry source drivers.
//!
//! A driver owns one physical source for its entire run. Cadence is anchored
//! once, source calls never overlap, and a stop request closes new issuance
//! while dynamically lowering the active call deadline. Fetch, decode, native
//! delivery, and archive admission remain replaceable seams; this module owns
//! only source scheduling and lifecycle.

use std::cell::RefCell;
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;

#[cfg(test)]
use std::future::Future;

use aiperf_clock::Clock;
use async_trait::async_trait;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::{
    AbsoluteCallDeadline, FetchDisposition, FetchedAttempt, FixedDeadlineCadence,
    MissedCadenceRange, SchedulingError, SourceAttemptGate, SourceAttemptKind,
};

/// Immutable request identity supplied to one source fetch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchRequest {
    /// Stable physical source identity.
    pub source_id: String,
    /// Sequence assigned to every issued source event.
    pub source_record_seq: u64,
    /// Sequence assigned only when network work may begin.
    pub request_attempt_seq: u64,
    /// Cadence or forced-boundary reason.
    pub kind: SourceAttemptKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CancellationSnapshot {
    revision: u64,
    deadline_ns: i64,
    stopped: bool,
}

#[derive(Debug)]
struct CancellationState {
    revision: u64,
    deadline_ns: i64,
    stopped: bool,
}

/// LocalSet-owned cancellation and deadline-lowering signal.
///
/// Clones stay on the source's local thread. An active transport races its own
/// future against [`Self::changed`] and the injected Clock, so shutdown never
/// waits for an originally longer request timeout.
#[derive(Clone, Debug)]
pub struct LocalCancellationSignal {
    state: Rc<RefCell<CancellationState>>,
    notify: Rc<Notify>,
}

impl LocalCancellationSignal {
    fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(CancellationState {
                revision: 0,
                deadline_ns: i64::MAX,
                stopped: false,
            })),
            notify: Rc::new(Notify::new()),
        }
    }

    fn snapshot(&self) -> CancellationSnapshot {
        let state = self.state.borrow();
        CancellationSnapshot {
            revision: state.revision,
            deadline_ns: state.deadline_ns,
            stopped: state.stopped,
        }
    }

    fn stop(&self, shutdown_deadline_ns: i64) {
        let mut state = self.state.borrow_mut();
        let next_deadline = state.deadline_ns.min(shutdown_deadline_ns);
        if !state.stopped || next_deadline != state.deadline_ns {
            state.stopped = true;
            state.deadline_ns = next_deadline;
            state.revision = state.revision.wrapping_add(1);
            drop(state);
            self.notify.notify_waiters();
        }
    }

    /// Returns the current effective lifecycle cap.
    #[must_use]
    pub fn deadline_ns(&self) -> i64 {
        self.state.borrow().deadline_ns
    }

    /// Whether a stop request has closed future issuance.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.state.borrow().stopped
    }

    /// Monotone local change token used to await a later deadline update.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.state.borrow().revision
    }

    /// Waits until the signal changes beyond `observed_revision`.
    pub async fn changed(&self, observed_revision: u64) -> u64 {
        loop {
            let notified = self.notify.notified();
            let revision = self.state.borrow().revision;
            if revision != observed_revision {
                return revision;
            }
            notified.await;
        }
    }
}

/// One prepared physical source fetcher.
#[async_trait(?Send)]
pub trait TelemetryFetcher: Debug {
    /// Fetch one exact all-outcome entity under an absolute Clock deadline.
    async fn fetch(
        &self,
        request: FetchRequest,
        absolute_deadline_ns: i64,
        cancellation: LocalCancellationSignal,
    ) -> FetchedAttempt;

    /// Release source-local transport and credential resources after draining.
    async fn shutdown(&self) -> Result<(), ArchiveSourceError>;
}

/// Ordered consumer of all-outcome source events and compact cadence gaps.
#[async_trait(?Send)]
pub trait TelemetryAttemptConsumer: Debug {
    /// Consume one terminal issued source event.
    async fn observe_attempt(&self, attempt: FetchedAttempt) -> Result<(), DriverConsumerError>;

    /// Consume one exact inclusive range of skipped cadence targets.
    async fn observe_missed(
        &self,
        source_id: &str,
        missed: MissedCadenceRange,
    ) -> Result<(), DriverConsumerError>;
}

/// Frozen source-driver timing and identity policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryDriverConfig {
    /// Stable physical source identity.
    pub source_id: String,
    /// Positive anchor-relative scrape interval.
    pub interval_ns: i64,
    /// Positive per-call lifetime ceiling.
    pub request_timeout_ns: i64,
    /// Optional absolute run-duration deadline.
    pub run_deadline_ns: Option<i64>,
}

impl TelemetryDriverConfig {
    /// Validates all static driver inputs before a task is spawned.
    pub fn validate(&self) -> Result<(), DriverStartError> {
        if self.source_id.is_empty()
            || self.source_id.trim() != self.source_id
            || self.source_id.chars().any(char::is_control)
        {
            return Err(DriverStartError::InvalidSourceId);
        }
        FixedDeadlineCadence::new(0, self.interval_ns).map_err(DriverStartError::Scheduling)?;
        AbsoluteCallDeadline::derive(0, self.request_timeout_ns, None, None, None)
            .map_err(DriverStartError::Scheduling)?;
        Ok(())
    }
}

/// Terminal source-driver counters.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TelemetryDriverSummary {
    /// Number of issued source events delivered to the consumer.
    pub attempts: u64,
    /// Number of cadence targets represented by compact missed ranges.
    pub missed_ticks: u64,
    /// Number of compact missed ranges delivered to the consumer.
    pub missed_ranges: u64,
}

/// Prepared source task before LocalSet activation.
pub trait PreparedTelemetryDriver: Debug {
    /// Stable physical source identity.
    fn source_id(&self) -> &str;

    /// Spawn the source on its current-thread LocalSet.
    fn start(self: Box<Self>) -> Result<Box<dyn RunningTelemetryDriver>, DriverStartError>;
}

/// Running source lifecycle handle.
#[async_trait(?Send)]
pub trait RunningTelemetryDriver: Debug {
    /// Close issuance and lower the active call's effective deadline.
    fn stop(&self, shutdown_deadline_ns: i64);

    /// Drain the terminal observation and source shutdown exactly once.
    async fn join(self: Box<Self>) -> Result<TelemetryDriverSummary, DriverStopError>;
}

/// Generic prepared fixed-deadline source.
pub struct FixedDeadlineTelemetryDriver {
    config: TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: Rc<dyn TelemetryFetcher>,
    consumer: Rc<dyn TelemetryAttemptConsumer>,
}

impl FixedDeadlineTelemetryDriver {
    /// Compose one source driver without starting IO.
    pub fn new(
        config: TelemetryDriverConfig,
        clock: Rc<dyn Clock>,
        fetcher: Rc<dyn TelemetryFetcher>,
        consumer: Rc<dyn TelemetryAttemptConsumer>,
    ) -> Result<Self, DriverStartError> {
        config.validate()?;
        Ok(Self {
            config,
            clock,
            fetcher,
            consumer,
        })
    }
}

impl Debug for FixedDeadlineTelemetryDriver {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixedDeadlineTelemetryDriver")
            .field("config", &self.config)
            .field("virtual_clock", &self.clock.is_virtual())
            .field("fetcher", &self.fetcher)
            .field("consumer", &self.consumer)
            .finish()
    }
}

impl PreparedTelemetryDriver for FixedDeadlineTelemetryDriver {
    fn source_id(&self) -> &str {
        &self.config.source_id
    }

    fn start(self: Box<Self>) -> Result<Box<dyn RunningTelemetryDriver>, DriverStartError> {
        self.config.validate()?;
        let cancellation = LocalCancellationSignal::new();
        let task_cancellation = cancellation.clone();
        let task = tokio::task::spawn_local(async move {
            run_driver(
                self.config,
                self.clock,
                self.fetcher,
                self.consumer,
                task_cancellation,
            )
            .await
        });
        Ok(Box::new(RunningFixedDeadlineTelemetryDriver {
            cancellation,
            task: Some(task),
        }))
    }
}

struct RunningFixedDeadlineTelemetryDriver {
    cancellation: LocalCancellationSignal,
    task: Option<JoinHandle<Result<TelemetryDriverSummary, DriverStopError>>>,
}

impl Debug for RunningFixedDeadlineTelemetryDriver {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunningFixedDeadlineTelemetryDriver")
            .field("stopped", &self.cancellation.is_stopped())
            .field("task_present", &self.task.is_some())
            .finish()
    }
}

impl Drop for RunningFixedDeadlineTelemetryDriver {
    fn drop(&mut self) {
        self.cancellation.stop(i64::MIN);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[async_trait(?Send)]
impl RunningTelemetryDriver for RunningFixedDeadlineTelemetryDriver {
    fn stop(&self, shutdown_deadline_ns: i64) {
        self.cancellation.stop(shutdown_deadline_ns);
    }

    async fn join(mut self: Box<Self>) -> Result<TelemetryDriverSummary, DriverStopError> {
        let task = self.task.take().ok_or(DriverStopError::AlreadyJoined)?;
        task.await.map_err(|error| {
            DriverStopError::Task(if error.is_cancelled() {
                "telemetry driver task was cancelled".to_owned()
            } else {
                format!("telemetry driver task panicked: {error}")
            })
        })?
    }
}

async fn run_driver(
    config: TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: Rc<dyn TelemetryFetcher>,
    consumer: Rc<dyn TelemetryAttemptConsumer>,
    cancellation: LocalCancellationSignal,
) -> Result<TelemetryDriverSummary, DriverStopError> {
    let result = run_driver_inner(
        &config,
        clock,
        fetcher.as_ref(),
        consumer.as_ref(),
        &cancellation,
    )
    .await;
    let shutdown = fetcher.shutdown().await.map_err(DriverStopError::Source);
    match (result, shutdown) {
        (Ok(summary), Ok(())) => Ok(summary),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(primary), Err(shutdown)) => Err(DriverStopError::PrimaryAndShutdown {
            primary: Box::new(primary),
            shutdown: Box::new(shutdown),
        }),
    }
}

async fn run_driver_inner(
    config: &TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: &dyn TelemetryFetcher,
    consumer: &dyn TelemetryAttemptConsumer,
    cancellation: &LocalCancellationSignal,
) -> Result<TelemetryDriverSummary, DriverStopError> {
    let anchor_ns = clock.now_ns();
    let mut cadence = FixedDeadlineCadence::new(anchor_ns, config.interval_ns)
        .map_err(DriverStopError::Scheduling)?;
    let mut gate = SourceAttemptGate::new();
    let mut request_attempt_seq = 0_u64;
    let mut summary = TelemetryDriverSummary::default();

    loop {
        if cancellation.is_stopped() {
            gate.stop(cancellation.deadline_ns());
            break;
        }
        let deadline = cadence
            .next_deadline()
            .map_err(DriverStopError::Scheduling)?;
        if config
            .run_deadline_ns
            .is_some_and(|run_deadline| deadline.scheduled_ns >= run_deadline)
        {
            break;
        }
        if !sleep_until_or_stopped(clock.clone(), deadline.scheduled_ns, cancellation).await {
            gate.stop(cancellation.deadline_ns());
            break;
        }
        let now_ns = clock.now_ns();
        if config
            .run_deadline_ns
            .is_some_and(|run_deadline| now_ns >= run_deadline)
        {
            break;
        }
        let issued_deadline = cadence.issue_next().map_err(DriverStopError::Scheduling)?;
        if issued_deadline != deadline {
            return Err(DriverStopError::Invariant(
                "telemetry cadence changed between wait and issue".to_owned(),
            ));
        }
        let call_deadline = AbsoluteCallDeadline::derive(
            now_ns,
            config.request_timeout_ns,
            None,
            config.run_deadline_ns,
            cancellation
                .is_stopped()
                .then(|| cancellation.deadline_ns()),
        )
        .map_err(DriverStopError::Scheduling)?;
        let issued = gate
            .begin(SourceAttemptKind::Continuous(deadline), call_deadline)
            .map_err(DriverStopError::Scheduling)?;
        let expected_request_attempt_seq = request_attempt_seq;
        let request = FetchRequest {
            source_id: config.source_id.clone(),
            source_record_seq: issued.source_record_seq,
            request_attempt_seq: expected_request_attempt_seq,
            kind: issued.kind.clone(),
        };
        request_attempt_seq =
            request_attempt_seq
                .checked_add(1)
                .ok_or(DriverStopError::Scheduling(
                    SchedulingError::ArithmeticOverflow,
                ))?;
        let attempt = fetch_before_effective_deadline(
            clock.clone(),
            fetcher,
            request,
            call_deadline.get(),
            cancellation.clone(),
        )
        .await;
        validate_fetched_attempt(
            &config.source_id,
            &issued,
            expected_request_attempt_seq,
            &attempt,
        )?;
        gate.complete(issued.source_record_seq)
            .map_err(DriverStopError::Scheduling)?;
        consumer
            .observe_attempt(attempt)
            .await
            .map_err(DriverStopError::Consumer)?;
        summary.attempts = summary
            .attempts
            .checked_add(1)
            .ok_or(DriverStopError::Scheduling(
                SchedulingError::ArithmeticOverflow,
            ))?;

        if cancellation.is_stopped() {
            gate.stop(cancellation.deadline_ns());
            break;
        }
        let completion_ns = clock.now_ns();
        let cadence_horizon_ns = config
            .run_deadline_ns
            .map_or(completion_ns, |run_deadline| {
                completion_ns.min(run_deadline.saturating_sub(1))
            });
        let advance = cadence
            .advance_after(cadence_horizon_ns)
            .map_err(DriverStopError::Scheduling)?;
        if let Some(missed) = advance.missed {
            consumer
                .observe_missed(&config.source_id, missed)
                .await
                .map_err(DriverStopError::Consumer)?;
            summary.missed_ticks = summary.missed_ticks.checked_add(missed.count).ok_or(
                DriverStopError::Scheduling(SchedulingError::ArithmeticOverflow),
            )?;
            summary.missed_ranges =
                summary
                    .missed_ranges
                    .checked_add(1)
                    .ok_or(DriverStopError::Scheduling(
                        SchedulingError::ArithmeticOverflow,
                    ))?;
        }
    }
    Ok(summary)
}

async fn sleep_until_or_stopped(
    clock: Rc<dyn Clock>,
    deadline_ns: i64,
    cancellation: &LocalCancellationSignal,
) -> bool {
    loop {
        if cancellation.is_stopped() {
            return false;
        }
        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return true;
        }
        let revision = cancellation.snapshot().revision;
        let sleep = clock.clone().sleep(remaining_ns);
        tokio::pin!(sleep);
        tokio::select! {
            biased;
            _ = cancellation.changed(revision) => continue,
            () = &mut sleep => return !cancellation.is_stopped(),
        }
    }
}

async fn fetch_before_effective_deadline(
    clock: Rc<dyn Clock>,
    fetcher: &dyn TelemetryFetcher,
    request: FetchRequest,
    authored_deadline_ns: i64,
    cancellation: LocalCancellationSignal,
) -> FetchedAttempt {
    let source_id = request.source_id.clone();
    let source_record_seq = request.source_record_seq;
    let request_attempt_seq = request.request_attempt_seq;
    let scheduled_ns = match request.kind {
        SourceAttemptKind::Continuous(deadline) => Some(deadline.scheduled_ns),
        SourceAttemptKind::Boundary { .. } => None,
    };
    let request_start_ns = clock.now_ns();
    let fetch = fetcher.fetch(request, authored_deadline_ns, cancellation.clone());
    tokio::pin!(fetch);
    let mut revision = cancellation.snapshot().revision;
    loop {
        let snapshot = cancellation.snapshot();
        let effective_deadline_ns = authored_deadline_ns.min(snapshot.deadline_ns);
        let remaining_ns = effective_deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return deadline_attempt(
                source_id,
                source_record_seq,
                request_attempt_seq,
                scheduled_ns,
                request_start_ns,
                clock.now_ns(),
                snapshot.stopped,
            );
        }
        let deadline_sleep = clock.clone().sleep(remaining_ns);
        tokio::pin!(deadline_sleep);
        tokio::select! {
            biased;
            fetched = &mut fetch => return fetched,
            next_revision = cancellation.changed(revision) => {
                revision = next_revision;
            }
            () = &mut deadline_sleep => {
                let stopped = cancellation.snapshot().stopped;
                return deadline_attempt(
                    source_id,
                    source_record_seq,
                    request_attempt_seq,
                    scheduled_ns,
                    request_start_ns,
                    clock.now_ns(),
                    stopped,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn deadline_attempt(
    source_id: String,
    source_record_seq: u64,
    request_attempt_seq: u64,
    scheduled_ns: Option<i64>,
    request_start_ns: i64,
    observed_ns: i64,
    stopped: bool,
) -> FetchedAttempt {
    FetchedAttempt {
        source_id,
        source_record_seq,
        request_attempt_seq: Some(request_attempt_seq),
        scheduled_ns,
        request_start_ns: Some(request_start_ns),
        first_byte_ns: None,
        capture_ns: None,
        latency_ns: Some(observed_ns.saturating_sub(request_start_ns).max(0)),
        disposition: if stopped {
            FetchDisposition::Shutdown
        } else {
            FetchDisposition::Timeout {
                request_started: true,
            }
        },
    }
}

fn validate_fetched_attempt(
    source_id: &str,
    issued: &crate::IssuedSourceAttempt,
    expected_request_attempt_seq: u64,
    fetched: &FetchedAttempt,
) -> Result<(), DriverStopError> {
    if fetched.source_id != source_id || fetched.source_record_seq != issued.source_record_seq {
        return Err(DriverStopError::Invariant(format!(
            "fetcher returned identity ({:?}, {}) for active source event ({source_id:?}, {})",
            fetched.source_id, fetched.source_record_seq, issued.source_record_seq
        )));
    }
    let expected_scheduled = match issued.kind {
        SourceAttemptKind::Continuous(deadline) => Some(deadline.scheduled_ns),
        SourceAttemptKind::Boundary { .. } => None,
    };
    if fetched.scheduled_ns != expected_scheduled {
        return Err(DriverStopError::Invariant(
            "fetcher changed the source event's scheduling identity".to_owned(),
        ));
    }
    if fetched.request_attempt_seq != Some(expected_request_attempt_seq) {
        return Err(DriverStopError::Invariant(
            "fetcher changed the request-attempt identity".to_owned(),
        ));
    }
    Ok(())
}

/// Source-specific cleanup failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveSourceError {
    /// Redaction-safe bounded detail.
    pub message: String,
}

impl Display for ArchiveSourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ArchiveSourceError {}

/// Ordered consumer rejected an attempt or loss fact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DriverConsumerError {
    /// Redaction-safe bounded detail.
    pub message: String,
}

impl Display for DriverConsumerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DriverConsumerError {}

/// Static driver construction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverStartError {
    /// Source ID is empty, padded, or contains a control byte.
    InvalidSourceId,
    /// Cadence/deadline input is invalid.
    Scheduling(SchedulingError),
}

impl Display for DriverStartError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceId => formatter.write_str("invalid telemetry source ID"),
            Self::Scheduling(error) => {
                write!(formatter, "invalid telemetry driver timing: {error}")
            }
        }
    }
}

impl std::error::Error for DriverStartError {}

/// Terminal driver drain failure.
#[derive(Debug)]
pub enum DriverStopError {
    /// Driver cadence or gate invariant failed.
    Scheduling(SchedulingError),
    /// Fetcher returned identity/timing facts for another event.
    Invariant(String),
    /// Ordered attempt/loss consumer failed.
    Consumer(DriverConsumerError),
    /// Source cleanup failed.
    Source(ArchiveSourceError),
    /// Both active execution and source cleanup failed.
    PrimaryAndShutdown {
        /// Active execution failure.
        primary: Box<DriverStopError>,
        /// Cleanup failure.
        shutdown: Box<DriverStopError>,
    },
    /// Local driver task was cancelled or panicked.
    Task(String),
    /// A running handle was consumed twice.
    AlreadyJoined,
}

impl Display for DriverStopError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduling(error) => {
                write!(formatter, "telemetry driver scheduling failed: {error}")
            }
            Self::Invariant(message) => {
                write!(formatter, "telemetry driver invariant failed: {message}")
            }
            Self::Consumer(error) => write!(formatter, "telemetry consumer failed: {error}"),
            Self::Source(error) => write!(formatter, "telemetry source shutdown failed: {error}"),
            Self::PrimaryAndShutdown { primary, shutdown } => write!(
                formatter,
                "telemetry driver failed ({primary}); source cleanup also failed ({shutdown})"
            ),
            Self::Task(message) => formatter.write_str(message),
            Self::AlreadyJoined => formatter.write_str("telemetry driver was already joined"),
        }
    }
}

impl std::error::Error for DriverStopError {}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use aiperf_clock::SimClock;
    use bytes::Bytes;

    use super::*;

    struct SlowFetcher {
        clock: Rc<dyn Clock>,
        latency_ns: i64,
        active: Cell<usize>,
        maximum_active: Cell<usize>,
        shutdowns: Cell<usize>,
    }

    impl Debug for SlowFetcher {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("SlowFetcher")
                .field("latency_ns", &self.latency_ns)
                .field("active", &self.active.get())
                .finish_non_exhaustive()
        }
    }

    #[async_trait(?Send)]
    impl TelemetryFetcher for SlowFetcher {
        async fn fetch(
            &self,
            request: FetchRequest,
            _absolute_deadline_ns: i64,
            _cancellation: LocalCancellationSignal,
        ) -> FetchedAttempt {
            let active = self.active.get() + 1;
            self.active.set(active);
            self.maximum_active
                .set(self.maximum_active.get().max(active));
            let start = self.clock.now_ns();
            self.clock.clone().sleep(self.latency_ns).await;
            let end = self.clock.now_ns();
            self.active.set(self.active.get() - 1);
            FetchedAttempt {
                source_id: request.source_id,
                source_record_seq: request.source_record_seq,
                request_attempt_seq: Some(request.request_attempt_seq),
                scheduled_ns: match request.kind {
                    SourceAttemptKind::Continuous(deadline) => Some(deadline.scheduled_ns),
                    SourceAttemptKind::Boundary { .. } => None,
                },
                request_start_ns: Some(start),
                first_byte_ns: Some(end),
                capture_ns: Some(end),
                latency_ns: Some(end - start),
                disposition: FetchDisposition::Response {
                    status: 200,
                    content_type: Some("text/plain; version=0.0.4".to_owned()),
                    content_encoding: None,
                    encoded_body: Bytes::from_static(b"metric 1\n"),
                    decoded_body: Bytes::from_static(b"metric 1\n"),
                },
            }
        }

        async fn shutdown(&self) -> Result<(), ArchiveSourceError> {
            self.shutdowns.set(self.shutdowns.get() + 1);
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingConsumer {
        attempts: RefCell<Vec<FetchedAttempt>>,
        missed: RefCell<Vec<MissedCadenceRange>>,
    }

    #[async_trait(?Send)]
    impl TelemetryAttemptConsumer for RecordingConsumer {
        async fn observe_attempt(
            &self,
            attempt: FetchedAttempt,
        ) -> Result<(), DriverConsumerError> {
            self.attempts.borrow_mut().push(attempt);
            Ok(())
        }

        async fn observe_missed(
            &self,
            _source_id: &str,
            missed: MissedCadenceRange,
        ) -> Result<(), DriverConsumerError> {
            self.missed.borrow_mut().push(missed);
            Ok(())
        }
    }

    fn drive_sim<T>(clock: Rc<SimClock>, body: impl Future<Output = T> + 'static) -> T
    where
        T: 'static,
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        runtime.block_on(local.run_until(async move {
            let task = tokio::task::spawn_local(body);
            loop {
                tokio::task::yield_now().await;
                if task.is_finished() {
                    return task.await.unwrap();
                }
                let next = clock
                    .next_event_time()
                    .expect("unfinished telemetry driver must expose a Clock event");
                clock.advance_to(next);
            }
        }))
    }

    #[test]
    fn slow_source_never_overlaps_and_compacts_missed_ticks() {
        let sim = Rc::new(SimClock::new());
        let clock: Rc<dyn Clock> = sim.clone();
        let fetcher = Rc::new(SlowFetcher {
            clock: clock.clone(),
            latency_ns: 25,
            active: Cell::new(0),
            maximum_active: Cell::new(0),
            shutdowns: Cell::new(0),
        });
        let consumer = Rc::new(RecordingConsumer::default());
        let prepared = FixedDeadlineTelemetryDriver::new(
            TelemetryDriverConfig {
                source_id: "source-a".to_owned(),
                interval_ns: 10,
                request_timeout_ns: 100,
                run_deadline_ns: Some(50),
            },
            clock,
            fetcher.clone(),
            consumer.clone(),
        )
        .unwrap();

        let summary = drive_sim(sim, async move {
            let running = Box::new(prepared).start().unwrap();
            running.join().await.unwrap()
        });

        assert_eq!(summary.attempts, 2);
        assert_eq!(summary.missed_ticks, 3);
        assert_eq!(summary.missed_ranges, 2);
        assert_eq!(fetcher.maximum_active.get(), 1);
        assert_eq!(fetcher.shutdowns.get(), 1);
        assert_eq!(consumer.missed.borrow()[0].first_tick, 1);
        assert_eq!(consumer.missed.borrow()[0].last_tick, 2);
        assert_eq!(consumer.missed.borrow()[1].first_tick, 4);
        assert_eq!(consumer.missed.borrow()[1].last_tick, 4);
        assert!(matches!(
            consumer.attempts.borrow()[1].disposition,
            FetchDisposition::Timeout {
                request_started: true
            }
        ));
    }

    #[derive(Debug)]
    struct NeverFetcher {
        shutdowns: Cell<usize>,
    }

    #[async_trait(?Send)]
    impl TelemetryFetcher for NeverFetcher {
        async fn fetch(
            &self,
            _request: FetchRequest,
            _absolute_deadline_ns: i64,
            _cancellation: LocalCancellationSignal,
        ) -> FetchedAttempt {
            std::future::pending().await
        }

        async fn shutdown(&self) -> Result<(), ArchiveSourceError> {
            self.shutdowns.set(self.shutdowns.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn stop_lowers_active_deadline_and_emits_one_shutdown_attempt() {
        let sim = Rc::new(SimClock::new());
        let clock: Rc<dyn Clock> = sim.clone();
        let fetcher = Rc::new(NeverFetcher {
            shutdowns: Cell::new(0),
        });
        let consumer = Rc::new(RecordingConsumer::default());
        let prepared = FixedDeadlineTelemetryDriver::new(
            TelemetryDriverConfig {
                source_id: "source-a".to_owned(),
                interval_ns: 100,
                request_timeout_ns: 1_000,
                run_deadline_ns: None,
            },
            clock.clone(),
            fetcher.clone(),
            consumer.clone(),
        )
        .unwrap();

        let summary = drive_sim(sim, async move {
            let running = Box::new(prepared).start().unwrap();
            let stopper = running;
            let clock_for_stop = clock.clone();
            let stop_task = tokio::task::spawn_local(async move {
                clock_for_stop.sleep(5).await;
                stopper.stop(10);
                stopper.join().await.unwrap()
            });
            stop_task.await.unwrap()
        });

        assert_eq!(summary.attempts, 1);
        assert_eq!(fetcher.shutdowns.get(), 1);
        assert_eq!(consumer.attempts.borrow().len(), 1);
        assert!(matches!(
            consumer.attempts.borrow()[0].disposition,
            FetchDisposition::Shutdown
        ));
    }

    #[test]
    fn fetched_attempt_must_preserve_issued_request_sequence() {
        let issued = crate::IssuedSourceAttempt {
            source_record_seq: 0,
            kind: SourceAttemptKind::Boundary {
                transition_id: "profiling-start".to_owned(),
            },
            deadline: AbsoluteCallDeadline::derive(0, 10, None, None, None).unwrap(),
        };
        let mut fetched = deadline_attempt("source-a".to_owned(), 0, 7, None, 0, 1, false);
        validate_fetched_attempt("source-a", &issued, 7, &fetched).unwrap();

        fetched.request_attempt_seq = Some(8);
        let mismatch = validate_fetched_attempt("source-a", &issued, 7, &fetched)
            .unwrap_err()
            .to_string();
        assert!(mismatch.contains("request-attempt identity"), "{mismatch}");

        fetched.request_attempt_seq = None;
        assert!(validate_fetched_attempt("source-a", &issued, 7, &fetched).is_err());
    }
}
