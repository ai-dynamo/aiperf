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
use std::collections::BTreeSet;
use std::fmt::{self, Debug, Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use aiperf_clock::Clock;
use async_trait::async_trait;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::{
    AbsoluteCallDeadline, BoundaryReference, FetchDisposition, FetchedAttempt,
    FixedDeadlineCadence, LossKindV1, LossReasonV1, MissedCadenceRange, SchedulingError,
    ScrapeReasonV1, SourceAttemptGate, SourceAttemptKind, SourceBoundarySnapshotCommand,
};

/// Default boundary command capacity reserved independently from data frames.
pub const DEFAULT_BOUNDARY_COMMAND_CAPACITY: usize = 16;

/// LocalSet-compatible future returned by one nonblocking boundary submission.
pub type LocalDriverFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

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

/// Driver-owned context attached after a source fetch becomes terminal.
///
/// Fetchers see only [`FetchRequest`] and cannot invent phase membership or
/// boundary joins. The run-owned driver snapshots this context at issuance and
/// the ordered consumer receives it together with the terminal fetch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryAttemptEnvelope {
    /// Terminal source fetch facts and exact entity bytes.
    pub attempt: FetchedAttempt,
    /// Continuous cadence or forced-boundary reason.
    pub reason: ScrapeReasonV1,
    /// Explicit boundary joins, empty for continuous attempts.
    pub boundary_refs: Vec<BoundaryReference>,
    /// Active benchmark phases captured at the snapshot instant.
    pub active_phase_ids: BTreeSet<String>,
}

/// Archive-facing terminal interpretation returned by an ordered consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryAttemptDisposition {
    /// The attempt itself terminally represents the source event.
    Attempt,
    /// Native delivery occurred but the archive closed the event as exact loss.
    Loss {
        /// Closed loss class.
        kind: LossKindV1,
        /// Closed reason paired with the loss class.
        reason: LossReasonV1,
    },
}

/// Terminal attempt-or-loss result for one accepted boundary command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundaryAttemptTerminal {
    /// One physical source attempt carries every planned reference.
    Attempt {
        /// Per-source event sequence.
        source_record_seq: u64,
        /// Per-source network sequence when IO began.
        request_attempt_seq: Option<u64>,
        /// Exact command references copied without reconstruction.
        boundary_refs: Vec<BoundaryReference>,
    },
    /// One exact loss frame carries every planned reference.
    Loss {
        /// Closed loss class.
        kind: LossKindV1,
        /// Closed reason paired with the loss class.
        reason: LossReasonV1,
        /// Exact command references copied without reconstruction.
        boundary_refs: Vec<BoundaryReference>,
    },
}

/// Source-scoped boundary completion returned after ordered consumption.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryAttemptCompletion {
    /// Stable physical source identity.
    pub source_id: String,
    /// Stable sealed-plan transition identity.
    pub transition_id: String,
    /// Exactly one attempt-or-loss terminal result.
    pub terminal: BoundaryAttemptTerminal,
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

    /// Consume one driver-owned context envelope.
    ///
    /// Standalone consumers retain source compatibility through the default
    /// continuous-attempt behavior. Attached consumers override this method to
    /// deliver native phase projections and perform nonblocking archive
    /// admission before returning the exact attempt-or-loss disposition.
    async fn observe_attempt_envelope(
        &self,
        envelope: TelemetryAttemptEnvelope,
    ) -> Result<TelemetryAttemptDisposition, DriverConsumerError> {
        self.observe_attempt(envelope.attempt).await?;
        Ok(TelemetryAttemptDisposition::Attempt)
    }

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
    fn start(self: Box<Self>) -> Result<Rc<dyn RunningTelemetryDriver>, DriverStartError>;
}

/// Running source lifecycle handle.
#[async_trait(?Send)]
pub trait RunningTelemetryDriver: Debug {
    /// Stable physical source identity retained from preparation.
    fn source_id(&self) -> &str;

    /// Add or remove one phase from future snapshot membership.
    fn set_phase_active(&self, phase_id: &str, active: bool) -> Result<(), DriverCommandError>;

    /// Nonblockingly submit one already sealed source boundary command.
    fn submit_boundary(
        &self,
        command: SourceBoundarySnapshotCommand,
    ) -> Result<
        LocalDriverFuture<Result<BoundaryAttemptCompletion, DriverStopError>>,
        DriverCommandError,
    >;

    /// Close issuance and lower the active call's effective deadline.
    fn stop(&self, shutdown_deadline_ns: i64);

    /// Drain the terminal observation and source shutdown exactly once.
    async fn join(&self) -> Result<TelemetryDriverSummary, DriverStopError>;
}

/// Generic prepared fixed-deadline source.
pub struct FixedDeadlineTelemetryDriver {
    config: TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: Rc<dyn TelemetryFetcher>,
    consumer: Rc<dyn TelemetryAttemptConsumer>,
    boundary_command_capacity: usize,
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
            boundary_command_capacity: DEFAULT_BOUNDARY_COMMAND_CAPACITY,
        })
    }

    /// Override the positive source-local reserved boundary command capacity.
    pub fn with_boundary_command_capacity(
        mut self,
        capacity: usize,
    ) -> Result<Self, DriverStartError> {
        if capacity == 0 {
            return Err(DriverStartError::ZeroBoundaryCommandCapacity);
        }
        self.boundary_command_capacity = capacity;
        Ok(self)
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
            .field("boundary_command_capacity", &self.boundary_command_capacity)
            .finish()
    }
}

impl PreparedTelemetryDriver for FixedDeadlineTelemetryDriver {
    fn source_id(&self) -> &str {
        &self.config.source_id
    }

    fn start(self: Box<Self>) -> Result<Rc<dyn RunningTelemetryDriver>, DriverStartError> {
        self.config.validate()?;
        if self.boundary_command_capacity == 0 {
            return Err(DriverStartError::ZeroBoundaryCommandCapacity);
        }
        let Self {
            config,
            clock,
            fetcher,
            consumer,
            boundary_command_capacity,
        } = *self;
        let source_id = config.source_id.clone();
        let cancellation = LocalCancellationSignal::new();
        let task_cancellation = cancellation.clone();
        let phase_membership = Rc::new(RefCell::new(PhaseMembershipTimeline::new(clock.now_ns())));
        let task_phase_membership = phase_membership.clone();
        let handle_clock = clock.clone();
        let (boundary_commands, boundary_receiver) = mpsc::channel(boundary_command_capacity);
        let task = tokio::task::spawn_local(async move {
            run_driver(
                config,
                clock,
                fetcher,
                consumer,
                task_cancellation,
                boundary_receiver,
                task_phase_membership,
            )
            .await
        });
        Ok(Rc::new(RunningFixedDeadlineTelemetryDriver {
            cancellation,
            source_id,
            clock: handle_clock,
            boundary_commands,
            phase_membership,
            task: RefCell::new(Some(task)),
        }))
    }
}

#[derive(Debug)]
struct PhaseMembershipTimeline {
    current: BTreeSet<String>,
    history: Vec<(i64, BTreeSet<String>)>,
}

impl PhaseMembershipTimeline {
    fn new(initial_ns: i64) -> Self {
        Self {
            current: BTreeSet::new(),
            history: vec![(initial_ns, BTreeSet::new())],
        }
    }

    fn set(&mut self, observed_ns: i64, phase_id: &str, active: bool) {
        if active {
            self.current.insert(phase_id.to_owned());
        } else {
            self.current.remove(phase_id);
        }
        if let Some((last_ns, last)) = self.history.last_mut()
            && *last_ns == observed_ns
        {
            *last = self.current.clone();
        } else {
            self.history.push((observed_ns, self.current.clone()));
        }
    }

    fn at(&self, observed_ns: i64) -> BTreeSet<String> {
        self.history
            .iter()
            .rev()
            .find(|(timestamp_ns, _)| *timestamp_ns <= observed_ns)
            .map_or_else(BTreeSet::new, |(_, phases)| phases.clone())
    }
}

struct RunningFixedDeadlineTelemetryDriver {
    cancellation: LocalCancellationSignal,
    source_id: String,
    clock: Rc<dyn Clock>,
    boundary_commands: mpsc::Sender<BoundaryCommand>,
    phase_membership: Rc<RefCell<PhaseMembershipTimeline>>,
    task: RefCell<Option<JoinHandle<Result<TelemetryDriverSummary, DriverStopError>>>>,
}

impl Debug for RunningFixedDeadlineTelemetryDriver {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunningFixedDeadlineTelemetryDriver")
            .field("source_id", &self.source_id)
            .field("stopped", &self.cancellation.is_stopped())
            .field("boundary_capacity", &self.boundary_commands.capacity())
            .field("active_phases", &self.phase_membership.borrow().current)
            .field("task_present", &self.task.borrow().is_some())
            .finish()
    }
}

impl Drop for RunningFixedDeadlineTelemetryDriver {
    fn drop(&mut self) {
        self.cancellation.stop(i64::MIN);
        if let Some(task) = self.task.get_mut().take() {
            task.abort();
        }
    }
}

#[async_trait(?Send)]
impl RunningTelemetryDriver for RunningFixedDeadlineTelemetryDriver {
    fn source_id(&self) -> &str {
        &self.source_id
    }

    fn set_phase_active(&self, phase_id: &str, active: bool) -> Result<(), DriverCommandError> {
        validate_driver_identifier("phase_id", phase_id)?;
        if self.cancellation.is_stopped() {
            return Err(DriverCommandError::Stopped);
        }
        self.phase_membership
            .borrow_mut()
            .set(self.clock.now_ns(), phase_id, active);
        Ok(())
    }

    fn submit_boundary(
        &self,
        command: SourceBoundarySnapshotCommand,
    ) -> Result<
        LocalDriverFuture<Result<BoundaryAttemptCompletion, DriverStopError>>,
        DriverCommandError,
    > {
        validate_boundary_command(&self.source_id, &command)?;
        if self.cancellation.is_stopped() {
            return Err(DriverCommandError::Stopped);
        }
        let (response, receiver) = oneshot::channel();
        self.boundary_commands
            .try_send(BoundaryCommand { command, response })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => DriverCommandError::Capacity,
                mpsc::error::TrySendError::Closed(_) => DriverCommandError::Stopped,
            })?;
        Ok(Box::pin(async move {
            receiver.await.map_err(|_| {
                DriverStopError::Task(
                    "telemetry boundary command lost its driver completion".to_owned(),
                )
            })?
        }))
    }

    fn stop(&self, shutdown_deadline_ns: i64) {
        self.cancellation.stop(shutdown_deadline_ns);
    }

    async fn join(&self) -> Result<TelemetryDriverSummary, DriverStopError> {
        let task = self
            .task
            .borrow_mut()
            .take()
            .ok_or(DriverStopError::AlreadyJoined)?;
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
    boundary_receiver: mpsc::Receiver<BoundaryCommand>,
    phase_membership: Rc<RefCell<PhaseMembershipTimeline>>,
) -> Result<TelemetryDriverSummary, DriverStopError> {
    let result = run_driver_inner(
        &config,
        clock.clone(),
        fetcher.as_ref(),
        consumer.as_ref(),
        &cancellation,
        boundary_receiver,
        phase_membership,
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
    mut boundary_receiver: mpsc::Receiver<BoundaryCommand>,
    phase_membership: Rc<RefCell<PhaseMembershipTimeline>>,
) -> Result<TelemetryDriverSummary, DriverStopError> {
    let anchor_ns = clock.now_ns();
    let mut cadence = FixedDeadlineCadence::new(anchor_ns, config.interval_ns)
        .map_err(DriverStopError::Scheduling)?;
    let mut gate = SourceAttemptGate::new();
    let mut request_attempt_seq = 0_u64;
    let mut summary = TelemetryDriverSummary::default();

    loop {
        if cancellation.is_stopped() {
            while let Ok(command) = boundary_receiver.try_recv() {
                process_boundary_command(
                    config,
                    clock.clone(),
                    fetcher,
                    consumer,
                    cancellation,
                    &phase_membership,
                    &mut gate,
                    &mut request_attempt_seq,
                    &mut summary,
                    command,
                )
                .await?;
            }
            gate.stop(cancellation.deadline_ns());
            break;
        }
        if let Ok(command) = boundary_receiver.try_recv() {
            process_boundary_command(
                config,
                clock.clone(),
                fetcher,
                consumer,
                cancellation,
                &phase_membership,
                &mut gate,
                &mut request_attempt_seq,
                &mut summary,
                command,
            )
            .await?;
            if !cancellation.is_stopped() {
                advance_cadence_after_attempt(
                    config,
                    clock.now_ns(),
                    &mut cadence,
                    consumer,
                    &mut summary,
                )
                .await?;
            }
            continue;
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
        match wait_for_driver_action(
            clock.clone(),
            deadline.scheduled_ns,
            cancellation,
            &mut boundary_receiver,
        )
        .await
        {
            DriverAction::Stopped => continue,
            DriverAction::Boundary(command) => {
                process_boundary_command(
                    config,
                    clock.clone(),
                    fetcher,
                    consumer,
                    cancellation,
                    &phase_membership,
                    &mut gate,
                    &mut request_attempt_seq,
                    &mut summary,
                    command,
                )
                .await?;
                if !cancellation.is_stopped() {
                    advance_cadence_after_attempt(
                        config,
                        clock.now_ns(),
                        &mut cadence,
                        consumer,
                        &mut summary,
                    )
                    .await?;
                }
            }
            DriverAction::Cadence => {
                let now_ns = clock.now_ns();
                if config
                    .run_deadline_ns
                    .is_some_and(|run_deadline| now_ns >= run_deadline)
                {
                    continue;
                }
                let issued_deadline = cadence.issue_next().map_err(DriverStopError::Scheduling)?;
                if issued_deadline != deadline {
                    return Err(DriverStopError::Invariant(
                        "telemetry cadence changed between wait and issue".to_owned(),
                    ));
                }
                process_attempt(
                    config,
                    clock.clone(),
                    fetcher,
                    consumer,
                    cancellation,
                    &phase_membership,
                    &mut gate,
                    &mut request_attempt_seq,
                    &mut summary,
                    SourceAttemptKind::Continuous(deadline),
                    Vec::new(),
                    None,
                )
                .await?;
                if !cancellation.is_stopped() {
                    advance_cadence_after_attempt(
                        config,
                        clock.now_ns(),
                        &mut cadence,
                        consumer,
                        &mut summary,
                    )
                    .await?;
                }
            }
        }
    }
    Ok(summary)
}

enum DriverAction {
    Cadence,
    Boundary(BoundaryCommand),
    Stopped,
}

struct BoundaryCommand {
    command: SourceBoundarySnapshotCommand,
    response: oneshot::Sender<Result<BoundaryAttemptCompletion, DriverStopError>>,
}

async fn wait_for_driver_action(
    clock: Rc<dyn Clock>,
    deadline_ns: i64,
    cancellation: &LocalCancellationSignal,
    boundary_receiver: &mut mpsc::Receiver<BoundaryCommand>,
) -> DriverAction {
    loop {
        if cancellation.is_stopped() {
            return DriverAction::Stopped;
        }
        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return DriverAction::Cadence;
        }
        let revision = cancellation.snapshot().revision;
        let sleep = clock.clone().sleep(remaining_ns);
        tokio::pin!(sleep);
        tokio::select! {
            biased;
            command = boundary_receiver.recv() => {
                if let Some(command) = command {
                    return DriverAction::Boundary(command);
                }
            }
            _ = cancellation.changed(revision) => continue,
            () = &mut sleep => return DriverAction::Cadence,
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_boundary_command(
    config: &TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: &dyn TelemetryFetcher,
    consumer: &dyn TelemetryAttemptConsumer,
    cancellation: &LocalCancellationSignal,
    phase_membership: &Rc<RefCell<PhaseMembershipTimeline>>,
    gate: &mut SourceAttemptGate,
    request_attempt_seq: &mut u64,
    summary: &mut TelemetryDriverSummary,
    boundary: BoundaryCommand,
) -> Result<(), DriverStopError> {
    let transition_id = boundary.command.subscribers[0].transition_id.clone();
    let boundary_refs = boundary.command.subscribers;
    let result = process_attempt(
        config,
        clock,
        fetcher,
        consumer,
        cancellation,
        phase_membership,
        gate,
        request_attempt_seq,
        summary,
        SourceAttemptKind::Boundary {
            transition_id: transition_id.clone(),
        },
        boundary_refs.clone(),
        Some(boundary.command.absolute_deadline_ns),
    )
    .await;
    match result {
        Ok(completed) => {
            let terminal = match completed.disposition {
                TelemetryAttemptDisposition::Attempt => BoundaryAttemptTerminal::Attempt {
                    source_record_seq: completed.source_record_seq,
                    request_attempt_seq: completed.request_attempt_seq,
                    boundary_refs,
                },
                TelemetryAttemptDisposition::Loss { kind, reason } => {
                    BoundaryAttemptTerminal::Loss {
                        kind,
                        reason,
                        boundary_refs,
                    }
                }
            };
            let completion = BoundaryAttemptCompletion {
                source_id: config.source_id.clone(),
                transition_id,
                terminal,
            };
            let _ = boundary.response.send(Ok(completion));
            Ok(())
        }
        Err(error) => {
            let _ = boundary
                .response
                .send(Err(DriverStopError::Task(error.to_string())));
            Err(error)
        }
    }
}

struct CompletedAttempt {
    source_record_seq: u64,
    request_attempt_seq: Option<u64>,
    disposition: TelemetryAttemptDisposition,
}

#[allow(clippy::too_many_arguments)]
async fn process_attempt(
    config: &TelemetryDriverConfig,
    clock: Rc<dyn Clock>,
    fetcher: &dyn TelemetryFetcher,
    consumer: &dyn TelemetryAttemptConsumer,
    cancellation: &LocalCancellationSignal,
    phase_membership: &Rc<RefCell<PhaseMembershipTimeline>>,
    gate: &mut SourceAttemptGate,
    next_request_attempt_seq: &mut u64,
    summary: &mut TelemetryDriverSummary,
    kind: SourceAttemptKind,
    boundary_refs: Vec<BoundaryReference>,
    boundary_deadline_ns: Option<i64>,
) -> Result<CompletedAttempt, DriverStopError> {
    let now_ns = clock.now_ns();
    let call_deadline = AbsoluteCallDeadline::derive(
        now_ns,
        config.request_timeout_ns,
        boundary_deadline_ns,
        config.run_deadline_ns,
        cancellation
            .is_stopped()
            .then(|| cancellation.deadline_ns()),
    )
    .map_err(DriverStopError::Scheduling)?;
    let issued = gate
        .begin(kind.clone(), call_deadline)
        .map_err(DriverStopError::Scheduling)?;

    let (expected_request_attempt_seq, attempt) =
        if call_deadline.is_expired_at(now_ns) || cancellation.is_stopped() {
            (
                None,
                pre_io_deadline_attempt(
                    config.source_id.clone(),
                    issued.source_record_seq,
                    match kind {
                        SourceAttemptKind::Continuous(deadline) => Some(deadline.scheduled_ns),
                        SourceAttemptKind::Boundary { .. } => None,
                    },
                    cancellation.is_stopped(),
                ),
            )
        } else {
            let sequence = *next_request_attempt_seq;
            *next_request_attempt_seq =
                (*next_request_attempt_seq)
                    .checked_add(1)
                    .ok_or(DriverStopError::Scheduling(
                        SchedulingError::ArithmeticOverflow,
                    ))?;
            let request = FetchRequest {
                source_id: config.source_id.clone(),
                source_record_seq: issued.source_record_seq,
                request_attempt_seq: sequence,
                kind: issued.kind.clone(),
            };
            (
                Some(sequence),
                fetch_before_effective_deadline(
                    clock.clone(),
                    fetcher,
                    request,
                    call_deadline.get(),
                    cancellation.clone(),
                )
                .await,
            )
        };
    validate_fetched_attempt(
        &config.source_id,
        &issued,
        expected_request_attempt_seq,
        &attempt,
    )?;
    gate.complete(issued.source_record_seq)
        .map_err(DriverStopError::Scheduling)?;
    let membership_ns = attempt.capture_ns.unwrap_or_else(|| clock.now_ns());
    let active_phase_ids = phase_membership.borrow().at(membership_ns);
    let reason = match issued.kind {
        SourceAttemptKind::Continuous(_) => ScrapeReasonV1::Continuous,
        SourceAttemptKind::Boundary { .. } => ScrapeReasonV1::Boundary,
    };
    let request_attempt_seq = attempt.request_attempt_seq;
    let disposition = consumer
        .observe_attempt_envelope(TelemetryAttemptEnvelope {
            attempt,
            reason,
            boundary_refs,
            active_phase_ids,
        })
        .await
        .map_err(DriverStopError::Consumer)?;
    if let TelemetryAttemptDisposition::Loss { kind, reason } = disposition
        && kind.reason() != reason
    {
        return Err(DriverStopError::Invariant(
            "telemetry consumer returned an incompatible loss kind/reason".to_owned(),
        ));
    }
    summary.attempts = summary
        .attempts
        .checked_add(1)
        .ok_or(DriverStopError::Scheduling(
            SchedulingError::ArithmeticOverflow,
        ))?;
    Ok(CompletedAttempt {
        source_record_seq: issued.source_record_seq,
        request_attempt_seq,
        disposition,
    })
}

async fn advance_cadence_after_attempt(
    config: &TelemetryDriverConfig,
    completion_ns: i64,
    cadence: &mut FixedDeadlineCadence,
    consumer: &dyn TelemetryAttemptConsumer,
    summary: &mut TelemetryDriverSummary,
) -> Result<(), DriverStopError> {
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
        summary.missed_ticks =
            summary
                .missed_ticks
                .checked_add(missed.count)
                .ok_or(DriverStopError::Scheduling(
                    SchedulingError::ArithmeticOverflow,
                ))?;
        summary.missed_ranges =
            summary
                .missed_ranges
                .checked_add(1)
                .ok_or(DriverStopError::Scheduling(
                    SchedulingError::ArithmeticOverflow,
                ))?;
    }
    Ok(())
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

fn pre_io_deadline_attempt(
    source_id: String,
    source_record_seq: u64,
    scheduled_ns: Option<i64>,
    stopped: bool,
) -> FetchedAttempt {
    FetchedAttempt {
        source_id,
        source_record_seq,
        request_attempt_seq: None,
        scheduled_ns,
        request_start_ns: None,
        first_byte_ns: None,
        capture_ns: None,
        latency_ns: None,
        disposition: if stopped {
            FetchDisposition::Shutdown
        } else {
            FetchDisposition::Timeout {
                request_started: false,
            }
        },
    }
}

fn validate_fetched_attempt(
    source_id: &str,
    issued: &crate::IssuedSourceAttempt,
    expected_request_attempt_seq: Option<u64>,
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
    if fetched.request_attempt_seq != expected_request_attempt_seq {
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

fn validate_driver_identifier(field: &'static str, value: &str) -> Result<(), DriverCommandError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(DriverCommandError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_boundary_command(
    expected_source_id: &str,
    command: &SourceBoundarySnapshotCommand,
) -> Result<(), DriverCommandError> {
    validate_driver_identifier("boundary source_id", &command.source_id)?;
    if command.source_id != expected_source_id {
        return Err(DriverCommandError::WrongSource {
            expected: expected_source_id.to_owned(),
            actual: command.source_id.clone(),
        });
    }
    if command.absolute_deadline_ns <= 0 {
        return Err(DriverCommandError::InvalidDeadline(
            command.absolute_deadline_ns,
        ));
    }
    if command.subscribers.is_empty() {
        return Err(DriverCommandError::EmptySubscribers);
    }
    let transition_id = &command.subscribers[0].transition_id;
    validate_driver_identifier("boundary transition_id", transition_id)?;
    let mut identities = BTreeSet::new();
    for reference in &command.subscribers {
        if reference.source_id != command.source_id {
            return Err(DriverCommandError::ReferenceSourceMismatch);
        }
        if reference.transition_id != *transition_id {
            return Err(DriverCommandError::ReferenceTransitionMismatch);
        }
        if !identities.insert(reference.boundary_id.clone()) {
            return Err(DriverCommandError::DuplicateReference(
                reference.boundary_id.clone(),
            ));
        }
    }
    Ok(())
}

/// Rejected phase or boundary command before source IO begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverCommandError {
    /// A stable ID was empty, padded, or contained a control byte.
    InvalidIdentifier {
        /// Invalid field.
        field: &'static str,
        /// Rejected redaction-safe value.
        value: String,
    },
    /// A command targeted another physical driver.
    WrongSource {
        /// Driver source.
        expected: String,
        /// Command source.
        actual: String,
    },
    /// A boundary command carried no phase subscribers.
    EmptySubscribers,
    /// A reference carried another source.
    ReferenceSourceMismatch,
    /// References in one source command carried different transitions.
    ReferenceTransitionMismatch,
    /// A source command repeated one boundary identity.
    DuplicateReference(String),
    /// The absolute Clock deadline was not positive.
    InvalidDeadline(i64),
    /// Reserved boundary command capacity was exhausted.
    Capacity,
    /// The driver already closed new phase and boundary commands.
    Stopped,
}

impl Display for DriverCommandError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "{field} has invalid identifier {value:?}")
            }
            Self::WrongSource { expected, actual } => write!(
                formatter,
                "boundary command source {actual:?} does not match driver {expected:?}"
            ),
            Self::EmptySubscribers => {
                formatter.write_str("boundary command requires at least one subscriber")
            }
            Self::ReferenceSourceMismatch => {
                formatter.write_str("boundary reference source does not match its command")
            }
            Self::ReferenceTransitionMismatch => {
                formatter.write_str("boundary references in one command must share one transition")
            }
            Self::DuplicateReference(boundary_id) => {
                write!(
                    formatter,
                    "boundary command repeats reference {boundary_id:?}"
                )
            }
            Self::InvalidDeadline(deadline_ns) => write!(
                formatter,
                "boundary absolute deadline must be positive, got {deadline_ns}"
            ),
            Self::Capacity => {
                formatter.write_str("reserved telemetry boundary command capacity is exhausted")
            }
            Self::Stopped => formatter.write_str("telemetry driver has stopped accepting commands"),
        }
    }
}

impl std::error::Error for DriverCommandError {}

/// Static driver construction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverStartError {
    /// Source ID is empty, padded, or contains a control byte.
    InvalidSourceId,
    /// Reserved boundary command capacity must be positive.
    ZeroBoundaryCommandCapacity,
    /// Cadence/deadline input is invalid.
    Scheduling(SchedulingError),
}

impl Display for DriverStartError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceId => formatter.write_str("invalid telemetry source ID"),
            Self::ZeroBoundaryCommandCapacity => {
                formatter.write_str("telemetry boundary command capacity must be positive")
            }
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
        envelopes: RefCell<Vec<TelemetryAttemptEnvelope>>,
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

        async fn observe_attempt_envelope(
            &self,
            envelope: TelemetryAttemptEnvelope,
        ) -> Result<TelemetryAttemptDisposition, DriverConsumerError> {
            self.attempts.borrow_mut().push(envelope.attempt.clone());
            self.envelopes.borrow_mut().push(envelope);
            Ok(TelemetryAttemptDisposition::Attempt)
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
        validate_fetched_attempt("source-a", &issued, Some(7), &fetched).unwrap();

        fetched.request_attempt_seq = Some(8);
        let mismatch = validate_fetched_attempt("source-a", &issued, Some(7), &fetched)
            .unwrap_err()
            .to_string();
        assert!(mismatch.contains("request-attempt identity"), "{mismatch}");

        fetched.request_attempt_seq = None;
        assert!(validate_fetched_attempt("source-a", &issued, Some(7), &fetched).is_err());
    }

    fn boundary_command(deadline_ns: i64) -> SourceBoundarySnapshotCommand {
        let reference = BoundaryReference {
            transition_id: "warmup-to-profiling".to_owned(),
            boundary_id: "source-a-profiling-start".to_owned(),
            phase_id: "profiling".to_owned(),
            source_id: "source-a".to_owned(),
            role: crate::BoundaryRole::PhaseStart,
            coalescing_group_id: None,
        };
        SourceBoundarySnapshotCommand {
            source_id: "source-a".to_owned(),
            coalescing_group_id: None,
            subscribers: vec![reference],
            absolute_deadline_ns: deadline_ns,
        }
    }

    #[test]
    fn boundary_preempts_cadence_and_preserves_membership_and_join_keys() {
        let sim = Rc::new(SimClock::new());
        let clock: Rc<dyn Clock> = sim.clone();
        let fetcher = Rc::new(SlowFetcher {
            clock: clock.clone(),
            latency_ns: 1,
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
                run_deadline_ns: None,
            },
            clock.clone(),
            fetcher,
            consumer.clone(),
        )
        .unwrap();

        let completion = drive_sim(sim, async move {
            let running = Box::new(prepared).start().unwrap();
            running.set_phase_active("profiling", true).unwrap();
            let completion = running
                .submit_boundary(boundary_command(100))
                .unwrap()
                .await
                .unwrap();
            running.stop(clock.now_ns());
            running.join().await.unwrap();
            completion
        });

        assert_eq!(completion.source_id, "source-a");
        assert_eq!(completion.transition_id, "warmup-to-profiling");
        assert!(matches!(
            completion.terminal,
            BoundaryAttemptTerminal::Attempt {
                source_record_seq: 0,
                request_attempt_seq: Some(0),
                ..
            }
        ));
        let envelopes = consumer.envelopes.borrow();
        assert_eq!(envelopes.len(), 1);
        assert_eq!(envelopes[0].reason, ScrapeReasonV1::Boundary);
        assert_eq!(envelopes[0].boundary_refs.len(), 1);
        assert_eq!(
            envelopes[0].active_phase_ids,
            BTreeSet::from(["profiling".to_owned()])
        );
    }

    #[test]
    fn expired_boundary_emits_one_pre_io_timeout_without_fetching() {
        let sim = Rc::new(SimClock::new());
        sim.advance_to(10);
        let clock: Rc<dyn Clock> = sim.clone();
        let fetcher = Rc::new(SlowFetcher {
            clock: clock.clone(),
            latency_ns: 1,
            active: Cell::new(0),
            maximum_active: Cell::new(0),
            shutdowns: Cell::new(0),
        });
        let consumer = Rc::new(RecordingConsumer::default());
        let prepared = FixedDeadlineTelemetryDriver::new(
            TelemetryDriverConfig {
                source_id: "source-a".to_owned(),
                interval_ns: 100,
                request_timeout_ns: 100,
                run_deadline_ns: None,
            },
            clock.clone(),
            fetcher.clone(),
            consumer.clone(),
        )
        .unwrap();

        drive_sim(sim, async move {
            let running = Box::new(prepared).start().unwrap();
            let completion = running
                .submit_boundary(boundary_command(5))
                .unwrap()
                .await
                .unwrap();
            assert!(matches!(
                completion.terminal,
                BoundaryAttemptTerminal::Attempt {
                    request_attempt_seq: None,
                    ..
                }
            ));
            running.stop(clock.now_ns());
            running.join().await.unwrap();
        });

        assert_eq!(fetcher.maximum_active.get(), 0);
        assert!(matches!(
            consumer.attempts.borrow()[0].disposition,
            FetchDisposition::Timeout {
                request_started: false
            }
        ));
    }
}
