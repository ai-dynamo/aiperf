// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Supervised adapter lifecycle, exact-profile authority, and reuse contracts.

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Display, Formatter},
    rc::Rc,
    time::Duration,
};

use async_trait::async_trait;

use crate::eval::provider::{ModelEndpointAuthority, ProviderError, ProviderProfile};
use crate::eval::{
    AdapterProtocol, AdapterProtocolConfig, AdapterProtocolFactory, AdapterRole, ArtifactDigest,
    ArtifactDownloadHandle, EvalExecutionError, HarborTaskPackage, HostEnvelope, ModelSecretId,
    NativeGraphPackagePlan, NativeGraphProfile, PROTOCOL_VERSION, ProviderCapabilities,
    ResolvedEpisodeTrial,
};

use super::protocol::DriverTerminalProtocol;
use super::{
    AdapterEnvelope, CompatibilityCaptureSession, CompatibilityTerminalReceipt, HostMessage,
    ProtocolError,
    factories::{ExternalDriverError, ExternalDriverSession},
};

const DEFAULT_MAX_STDOUT_FRAME_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_STDERR_BYTES: usize = 16 * 1024;

/// Unforgeable identity shared only by an exact-profile authorization and its requests.
#[derive(Clone, Debug)]
struct ExactSpawnToken(Rc<()>);

impl ExactSpawnToken {
    fn new() -> Self {
        Self(Rc::new(()))
    }

    fn matches(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Independent deadline budgets for each supervised adapter lifecycle boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterLifecycleDeadlines {
    startup: Duration,
    reset: Duration,
    heartbeat: Duration,
    idle: Duration,
    operation: Duration,
    cancel: Duration,
    reap: Duration,
}

impl AdapterLifecycleDeadlines {
    /// Validates nonzero, independently recorded lifecycle budgets.
    pub fn new(
        startup: Duration,
        reset: Duration,
        heartbeat: Duration,
        idle: Duration,
        operation: Duration,
        cancel: Duration,
        reap: Duration,
    ) -> Result<Self, AdapterSupervisionError> {
        let named = [
            ("startup", startup),
            ("reset", reset),
            ("heartbeat", heartbeat),
            ("idle", idle),
            ("operation", operation),
            ("cancel", cancel),
            ("reap", reap),
        ];
        if let Some((name, _)) = named.iter().find(|(_, duration)| duration.is_zero()) {
            return Err(AdapterSupervisionError::InvalidDeadline(name));
        }
        Ok(Self {
            startup,
            reset,
            heartbeat,
            idle,
            operation,
            cancel,
            reap,
        })
    }

    /// Returns the bounded process-startup budget.
    pub const fn startup(self) -> Duration {
        self.startup
    }

    /// Returns the reset acknowledgement budget.
    pub const fn reset(self) -> Duration {
        self.reset
    }

    /// Returns the liveness-heartbeat budget.
    pub const fn heartbeat(self) -> Duration {
        self.heartbeat
    }

    /// Returns the idle-session budget.
    pub const fn idle(self) -> Duration {
        self.idle
    }

    /// Returns the one-operation response budget.
    pub const fn operation(self) -> Duration {
        self.operation
    }

    /// Returns the graceful cancellation budget.
    pub const fn cancel(self) -> Duration {
        self.cancel
    }

    /// Returns the post-cancellation reaping budget.
    pub const fn reap(self) -> Duration {
        self.reap
    }
}

impl Default for AdapterLifecycleDeadlines {
    fn default() -> Self {
        // These are conservative defaults only. An execution plan records and
        // supplies its own seven independent deadlines before a live runner exists.
        Self {
            startup: Duration::from_secs(30),
            reset: Duration::from_secs(10),
            heartbeat: Duration::from_secs(10),
            idle: Duration::from_secs(60),
            operation: Duration::from_secs(60),
            cancel: Duration::from_secs(5),
            reap: Duration::from_secs(10),
        }
    }
}

/// Why the host is terminating a supervised adapter process group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelReason {
    /// The enclosing benchmark is shutting down normally.
    HostShutdown,
    /// A reset acknowledgement failed or violated its contract.
    ResetFailure,
    /// A protocol or endpoint-isolation violation invalidated the episode.
    IntegrityViolation,
    /// The current operation timed out or failed host-side.
    OperationFailure,
}

/// Terminal disposition observed after a bounded child reaping operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterExit {
    /// The child exited before a forced signal was necessary.
    Exited,
    /// The child was terminated and reaped by the host.
    Reaped,
}

/// One bounded streaming adapter launch request with no model secret values.
#[derive(Clone, Debug)]
pub struct AdapterSpawnRequest {
    argv: Vec<String>,
    environment: BTreeMap<String, String>,
    deadlines: AdapterLifecycleDeadlines,
    max_stdout_frame_bytes: usize,
    max_stderr_bytes: usize,
    exact_spawn_token: Option<ExactSpawnToken>,
    external_spawn_token: Option<ExactSpawnToken>,
}

impl AdapterSpawnRequest {
    /// Creates a request for an adapter that cannot participate in a NativeGraph exact profile.
    ///
    /// NativeGraph exact profile requests are constructed only through
    /// [`NativeGraphAdapterAuthorization`], which binds secret stripping and
    /// endpoint-isolation proof to the selected runtime before a spawner is used.
    pub fn for_non_model_adapter(
        argv: impl IntoIterator<Item = String>,
        environment: BTreeMap<String, String>,
        deadlines: AdapterLifecycleDeadlines,
    ) -> Result<Self, AdapterSupervisionError> {
        let argv: Vec<_> = argv.into_iter().collect();
        if argv.is_empty() || argv.iter().any(|argument| argument.trim().is_empty()) {
            return Err(AdapterSupervisionError::InvalidSpawnRequest("argv"));
        }
        Ok(Self {
            argv,
            environment,
            deadlines,
            max_stdout_frame_bytes: DEFAULT_MAX_STDOUT_FRAME_BYTES,
            max_stderr_bytes: DEFAULT_MAX_STDERR_BYTES,
            exact_spawn_token: None,
            external_spawn_token: None,
        })
    }

    fn for_exact_adapter(
        argv: impl IntoIterator<Item = String>,
        environment: BTreeMap<String, String>,
        deadlines: AdapterLifecycleDeadlines,
        exact_spawn_token: ExactSpawnToken,
    ) -> Result<Self, AdapterSupervisionError> {
        let mut request = Self::for_non_model_adapter(argv, environment, deadlines)?;
        request.exact_spawn_token = Some(exact_spawn_token);
        Ok(request)
    }

    fn for_external_driver(
        argv: impl IntoIterator<Item = String>,
        deadlines: AdapterLifecycleDeadlines,
        external_spawn_token: ExactSpawnToken,
    ) -> Result<Self, AdapterSupervisionError> {
        let mut request = Self::for_non_model_adapter(argv, BTreeMap::new(), deadlines)?;
        request.external_spawn_token = Some(external_spawn_token);
        Ok(request)
    }

    /// Lowers the process-output caps for one adapter session.
    pub fn with_output_limits(
        mut self,
        max_stdout_frame_bytes: usize,
        max_stderr_bytes: usize,
    ) -> Result<Self, AdapterSupervisionError> {
        if max_stdout_frame_bytes == 0 || max_stderr_bytes == 0 {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "output limits",
            ));
        }
        if max_stdout_frame_bytes > self.max_stdout_frame_bytes {
            return Err(AdapterSupervisionError::OutputLimitIncrease {
                field: "stdout",
                requested: max_stdout_frame_bytes,
                limit: self.max_stdout_frame_bytes,
            });
        }
        if max_stderr_bytes > self.max_stderr_bytes {
            return Err(AdapterSupervisionError::OutputLimitIncrease {
                field: "stderr",
                requested: max_stderr_bytes,
                limit: self.max_stderr_bytes,
            });
        }
        self.max_stdout_frame_bytes = max_stdout_frame_bytes;
        self.max_stderr_bytes = max_stderr_bytes;
        Ok(self)
    }

    fn intersect_protocol_stdout_limit(mut self, max_frame_bytes: usize) -> Self {
        self.max_stdout_frame_bytes = self.max_stdout_frame_bytes.min(max_frame_bytes);
        self
    }

    /// Borrows the exact non-shell argv to execute.
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// Borrows the already-filtered non-secret adapter environment.
    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.environment
    }

    /// Returns the lifecycle budgets selected for this child.
    pub const fn deadlines(&self) -> AdapterLifecycleDeadlines {
        self.deadlines
    }

    /// Returns the maximum accepted JSONL stdout frame length.
    pub const fn max_stdout_frame_bytes(&self) -> usize {
        self.max_stdout_frame_bytes
    }

    /// Returns the maximum retained stderr diagnostic bytes.
    pub const fn max_stderr_bytes(&self) -> usize {
        self.max_stderr_bytes
    }
}

/// One streaming child process owned by a supervised adapter session.
///
/// Implementations must continuously drain stderr into a buffer bounded by the
/// caller-provided cap while they await stdout. This prevents a diagnostic-only
/// child from deadlocking a protocol operation on a full stderr pipe.
#[async_trait(?Send)]
pub trait AdapterProcess {
    /// Writes one already-bounded JSONL frame to child stdin.
    async fn write_frame(
        &mut self,
        frame: &[u8],
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError>;
    /// Reads exactly one stdout JSONL frame under the supplied byte and time caps.
    async fn read_stdout_frame(
        &mut self,
        max_bytes: usize,
        deadline: Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError>;
    /// Returns and clears the concurrently drained stderr diagnostics.
    async fn drain_stderr(&mut self, max_bytes: usize) -> Result<Vec<u8>, AdapterSupervisionError>;
    /// Sends the graceful process-group termination request.
    async fn cancel(
        &mut self,
        reason: CancelReason,
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError>;
    /// Reaps the owned client process and descendants.
    async fn reap(&mut self, deadline: Duration) -> Result<AdapterExit, AdapterSupervisionError>;
    /// Synchronously fences an unfinished process group during `Drop`.
    fn fence(&mut self);
}

/// Owned launch transaction for one adapter child.
///
/// A spawner returns this guard synchronously once it has created any child-side
/// state. Until [`Self::await_process`] transfers the owned process to the
/// caller, `abort` and `fence` retain full responsibility for that state.
#[async_trait(?Send)]
pub trait AdapterSpawnTransaction {
    /// Waits for the launch to produce its owned supervised process.
    ///
    /// On success, ownership transfers to the returned process and this
    /// transaction retains no process cleanup responsibility.
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError>;
    /// Terminates and reaps every process created by this transaction.
    async fn abort(&mut self, deadline: Duration) -> Result<(), AdapterSupervisionError>;
    /// Synchronously fences every process still owned by this transaction.
    fn fence(&mut self);
}

/// Injectable streaming process creator for one adapter implementation.
pub trait AdapterSpawner {
    /// Begins one isolated adapter launch and returns its cancellation-safe owner.
    ///
    /// Implementations must either return an owned transaction which fences all
    /// launched state, or return an error without launching anything. They may
    /// not launch a child and then await outside the returned transaction.
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError>;
}

/// Drop guard which fences a launch if the factory future is cancelled mid-startup.
struct SpawnTransactionGuard {
    transaction: Option<Box<dyn AdapterSpawnTransaction>>,
}

impl SpawnTransactionGuard {
    fn new(transaction: Box<dyn AdapterSpawnTransaction>) -> Self {
        Self {
            transaction: Some(transaction),
        }
    }

    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.transaction
            .as_deref_mut()
            .ok_or(AdapterSupervisionError::PoolAccounting)?
            .await_process()
            .await
    }

    async fn abort(&mut self, deadline: Duration) -> Result<(), AdapterSupervisionError> {
        self.transaction
            .as_deref_mut()
            .ok_or(AdapterSupervisionError::PoolAccounting)?
            .abort(deadline)
            .await
    }

    fn disarm(&mut self) {
        self.transaction.take();
    }
}

impl Drop for SpawnTransactionGuard {
    fn drop(&mut self) {
        if let Some(transaction) = self.transaction.as_deref_mut() {
            transaction.fence();
        }
    }
}

/// Factory seam for a complete protocol-validated adapter runtime.
#[async_trait(?Send)]
pub trait AdapterRuntimeFactory {
    /// Returns the exact immutable protocol configuration used by [`Self::start`], if exposed.
    ///
    /// Runtimes that cannot bind an adapter session to one immutable configuration return
    /// `None`; callers requiring exact role and capability admission must refuse them.
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        None
    }

    /// Starts one supervised adapter session from a prefiltered spawn request.
    async fn start(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError>;
}

/// The only lifecycle interface used by future live NativeGraph runners.
#[async_trait(?Send)]
pub trait SupervisedAdapter {
    /// Admits and sends exactly one host-authorized protocol transition.
    async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError>;
    /// Receives exactly one protocol-validated adapter transition.
    async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError>;
    /// Receives a protocol transition while enforcing the heartbeat deadline.
    ///
    /// Runners use this path only while waiting for the next adapter heartbeat;
    /// it deliberately does not share the ordinary operation budget.
    async fn receive_heartbeat(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError>;
    /// Receives a protocol transition while enforcing the idle deadline.
    ///
    /// Runners use this path after an adapter has become idle, so a long
    /// operation allowance cannot hide an idle worker.
    async fn receive_idle(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError>;
    /// Applies one Rust-owned reset and validates its matching acknowledgement.
    async fn reset(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError>;
    /// Releases one Rust-revoked download capability from the private protocol ledger.
    fn release_download_handle(
        &mut self,
        download: &ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError>;
    /// Cancels and reaps the child without leaving a reusable process behind.
    async fn cancel_and_reap(
        &mut self,
        reason: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError>;
}

/// Strict adapter runtime factory composed from swappable protocol and process seams.
pub struct ProtocolAdapterRuntimeFactory {
    config: AdapterProtocolConfig,
    protocol_factory: Rc<dyn AdapterProtocolFactory>,
    spawner: Rc<dyn AdapterSpawner>,
}

impl ProtocolAdapterRuntimeFactory {
    /// Creates a factory pinned to one role-admitted Task 4 protocol configuration.
    pub fn new(
        config: AdapterProtocolConfig,
        protocol_factory: Rc<dyn AdapterProtocolFactory>,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Self {
        Self {
            config,
            protocol_factory,
            spawner,
        }
    }
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for ProtocolAdapterRuntimeFactory {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        let protocol = self
            .protocol_factory
            .create(self.config.clone())
            .map_err(AdapterSupervisionError::Protocol)?;
        let request = request.intersect_protocol_stdout_limit(self.config.max_frame_bytes());
        let deadlines = request.deadlines();
        let max_stdout_frame_bytes = request.max_stdout_frame_bytes();
        let max_stderr_bytes = request.max_stderr_bytes();
        let startup_deadline = AdapterDeadline::new(deadlines.startup())?;
        let mut transaction = SpawnTransactionGuard::new(self.spawner.begin_spawn(request)?);
        let process = match tokio::time::timeout(
            startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
            transaction.await_process(),
        )
        .await
        {
            Ok(Ok(process)) => {
                transaction.disarm();
                process
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                let primary = AdapterSupervisionError::StartupDeadlineElapsed;
                let error = match tokio::time::timeout(
                    deadlines.reap(),
                    transaction.abort(deadlines.reap()),
                )
                .await
                {
                    Ok(Ok(())) => primary,
                    Ok(Err(recovery)) => AdapterSupervisionError::Recovery {
                        primary: Box::new(primary),
                        recovery: Box::new(recovery),
                    },
                    Err(_) => AdapterSupervisionError::Recovery {
                        primary: Box::new(primary),
                        recovery: Box::new(AdapterSupervisionError::Process(
                            "adapter startup cleanup deadline elapsed".to_owned(),
                        )),
                    },
                };
                return Err(error);
            }
        };
        let mut adapter = StrictSupervisedAdapter {
            protocol,
            process: Some(process),
            deadlines,
            max_stdout_frame_bytes,
            max_stderr_bytes,
            terminal_exit: None,
        };
        let hello = HostEnvelope::new(
            self.config.episode(),
            "startup",
            0,
            "hello",
            HostMessage::Hello {
                supported_versions: vec![PROTOCOL_VERSION],
                adapter_role: self.config.role(),
                capabilities: self.config.capabilities().iter().copied().collect(),
            },
        );
        let startup = async {
            adapter
                .send_with_deadline(
                    hello,
                    startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
                )
                .await?;
            adapter
                .receive_with_deadline(
                    startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
                )
                .await
                .map(|_| ())
        };
        match tokio::time::timeout(
            startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
            startup,
        )
        .await
        {
            Ok(Ok(())) => Ok(Box::new(adapter)),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(adapter
                .fail_closed(
                    CancelReason::OperationFailure,
                    AdapterSupervisionError::StartupDeadlineElapsed,
                )
                .await),
        }
    }
}

/// One monotonic local deadline shared by a compound adapter lifecycle action.
struct AdapterDeadline {
    end: tokio::time::Instant,
}

impl AdapterDeadline {
    fn new(duration: Duration) -> Result<Self, AdapterSupervisionError> {
        let end = tokio::time::Instant::now()
            .checked_add(duration)
            .ok_or_else(|| {
                AdapterSupervisionError::Process("adapter deadline is invalid".to_owned())
            })?;
        Ok(Self { end })
    }

    fn remaining(
        &self,
        expired: AdapterSupervisionError,
    ) -> Result<Duration, AdapterSupervisionError> {
        self.end
            .checked_duration_since(tokio::time::Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(expired)
    }
}

/// A strict duplex session that makes Task 4 protocol admission authoritative.
pub struct StrictSupervisedAdapter {
    protocol: Box<dyn AdapterProtocol>,
    process: Option<Box<dyn AdapterProcess>>,
    deadlines: AdapterLifecycleDeadlines,
    max_stdout_frame_bytes: usize,
    max_stderr_bytes: usize,
    terminal_exit: Option<AdapterExit>,
}

impl StrictSupervisedAdapter {
    /// Binds one already-created process to the strict protocol before startup negotiation.
    ///
    /// External Driver startup stores this owner before its first protocol-I/O await so
    /// cancellation can run the same confirmed cancel/reap path as an established session.
    pub(crate) fn from_prestarted_process(
        config: AdapterProtocolConfig,
        process: Box<dyn AdapterProcess>,
        deadlines: AdapterLifecycleDeadlines,
        max_stdout_frame_bytes: usize,
        max_stderr_bytes: usize,
    ) -> Self {
        let max_stdout_frame_bytes = max_stdout_frame_bytes.min(config.max_frame_bytes());
        Self {
            protocol: super::protocol::strict_adapter_protocol(config),
            process: Some(process),
            deadlines,
            max_stdout_frame_bytes,
            max_stderr_bytes,
            terminal_exit: None,
        }
    }

    /// Completes the strict Hello/Ready exchange for an already-owned process.
    pub(crate) async fn negotiate_startup(
        &mut self,
        config: &AdapterProtocolConfig,
    ) -> Result<(), AdapterSupervisionError> {
        let startup_deadline = AdapterDeadline::new(self.deadlines.startup())?;
        let hello = HostEnvelope::new(
            config.episode(),
            "startup",
            0,
            "hello",
            HostMessage::Hello {
                supported_versions: vec![PROTOCOL_VERSION],
                adapter_role: config.role(),
                capabilities: config.capabilities().iter().copied().collect(),
            },
        );
        self.send_with_deadline(
            hello,
            startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
        )
        .await?;
        self.receive_with_deadline(
            startup_deadline.remaining(AdapterSupervisionError::StartupDeadlineElapsed)?,
        )
        .await
        .map(|_| ())
    }

    fn process_mut(&mut self) -> Result<&mut Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.process
            .as_mut()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn fail_closed(
        &mut self,
        reason: CancelReason,
        primary: AdapterSupervisionError,
    ) -> AdapterSupervisionError {
        match self.cancel_and_reap(reason).await {
            Ok(_) => primary,
            Err(recovery) => AdapterSupervisionError::Recovery {
                primary: Box::new(primary),
                recovery: Box::new(recovery),
            },
        }
    }

    async fn drain_diagnostics(&mut self) -> Result<(), AdapterSupervisionError> {
        let cap = self.max_stderr_bytes;
        let diagnostics = self.process_mut()?.drain_stderr(cap).await?;
        if diagnostics.len() > cap {
            return Err(AdapterSupervisionError::bounded_diagnostic_output(
                diagnostics.len(),
                cap,
            ));
        }
        Ok(())
    }

    async fn send_with_deadline(
        &mut self,
        message: HostEnvelope,
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let admitted = self
            .protocol
            .accept_host(message)
            .map_err(AdapterSupervisionError::Protocol)?;
        let frame = self
            .protocol
            .encode_host_frame(&admitted)
            .map_err(AdapterSupervisionError::Protocol)?;
        if let Err(error) = self.process_mut()?.write_frame(&frame, deadline).await {
            return Err(self
                .fail_closed(CancelReason::OperationFailure, error)
                .await);
        }
        Ok(())
    }

    async fn receive_with_deadline(
        &mut self,
        deadline: Duration,
    ) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        let max_stdout_frame_bytes = self.max_stdout_frame_bytes;
        let frame = match self
            .process_mut()?
            .read_stdout_frame(max_stdout_frame_bytes, deadline)
            .await
        {
            Ok(frame) if frame.len() <= max_stdout_frame_bytes => frame,
            Ok(frame) => {
                let error = AdapterSupervisionError::bounded_stdout_frame(
                    frame.len(),
                    max_stdout_frame_bytes,
                );
                return Err(self
                    .fail_closed(CancelReason::IntegrityViolation, error)
                    .await);
            }
            Err(error) => {
                return Err(self
                    .fail_closed(CancelReason::OperationFailure, error)
                    .await);
            }
        };
        if let Err(error) = self.drain_diagnostics().await {
            return Err(self
                .fail_closed(CancelReason::IntegrityViolation, error)
                .await);
        }
        match self.protocol.accept_adapter_frame(&frame) {
            Ok(message) => Ok(message.envelope().clone()),
            Err(error) => Err(self
                .fail_closed(
                    CancelReason::IntegrityViolation,
                    AdapterSupervisionError::Protocol(error),
                )
                .await),
        }
    }
}

#[async_trait(?Send)]
impl SupervisedAdapter for StrictSupervisedAdapter {
    async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        self.send_with_deadline(message, self.deadlines.operation())
            .await
    }

    async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive_with_deadline(self.deadlines.operation()).await
    }

    async fn receive_heartbeat(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive_with_deadline(self.deadlines.heartbeat()).await
    }

    async fn receive_idle(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive_with_deadline(self.deadlines.idle()).await
    }

    async fn reset(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        if !matches!(message.message, HostMessage::Reset { .. }) {
            return Err(AdapterSupervisionError::InvalidResetTransition);
        }
        let deadline = AdapterDeadline::new(self.deadlines.reset())?;
        let reset = async {
            self.send_with_deadline(
                message,
                deadline.remaining(AdapterSupervisionError::Process(
                    "adapter reset deadline elapsed".to_owned(),
                ))?,
            )
            .await?;
            self.receive_with_deadline(deadline.remaining(AdapterSupervisionError::Process(
                "adapter reset deadline elapsed".to_owned(),
            ))?)
            .await
            .map(|_| ())
        };
        let remaining = deadline.remaining(AdapterSupervisionError::Process(
            "adapter reset deadline elapsed".to_owned(),
        ))?;
        match tokio::time::timeout(remaining, reset).await {
            Ok(result) => result,
            Err(_) => Err(self
                .fail_closed(
                    CancelReason::ResetFailure,
                    AdapterSupervisionError::Process("adapter reset deadline elapsed".to_owned()),
                )
                .await),
        }
    }

    fn release_download_handle(
        &mut self,
        download: &ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        self.protocol
            .release_download_handle(download)
            .map_err(AdapterSupervisionError::Protocol)
    }

    async fn cancel_and_reap(
        &mut self,
        reason: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        if let Some(exit) = self.terminal_exit {
            return Ok(exit);
        }
        let cancel_deadline = self.deadlines.cancel();
        let reap_deadline = self.deadlines.reap();
        let process = self.process_mut()?;
        let cancel = process.cancel(reason, cancel_deadline).await;
        let reap = process.reap(reap_deadline).await;
        match (cancel, reap) {
            (_, Ok(exit)) => {
                self.terminal_exit = Some(exit);
                self.process = None;
                Ok(exit)
            }
            (Ok(()), Err(error)) => Err(error),
            (Err(cancel), Err(reap)) => Err(AdapterSupervisionError::Recovery {
                primary: Box::new(cancel),
                recovery: Box::new(reap),
            }),
        }
    }
}

impl Drop for StrictSupervisedAdapter {
    fn drop(&mut self) {
        if self.terminal_exit.is_none()
            && let Some(process) = self.process.as_deref_mut()
        {
            // `Drop` cannot drive an async reap. The process implementation owns
            // a synchronous process-group fence so an abandoned operation cannot
            // remain runnable after the Rust owner disappears.
            process.fence();
        }
    }
}

/// Borrowed terminal-only session over one already-started externally driven adapter.
///
/// The owning runner retains cancellation and reaping authority. This session can issue only the
/// one fixed Driver terminal request and converts its response before the raw JSON leaves this
/// private boundary.
pub(crate) struct ProtocolExternalDriverSession<'a> {
    adapter: &'a mut dyn SupervisedAdapter,
    terminal: DriverTerminalProtocol,
    capture_session: Option<CompatibilityCaptureSession>,
}

impl<'a> ProtocolExternalDriverSession<'a> {
    pub(crate) fn new(
        adapter: &'a mut dyn SupervisedAdapter,
        config: AdapterProtocolConfig,
        capture_session: CompatibilityCaptureSession,
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            adapter,
            terminal: DriverTerminalProtocol::new(config)?,
            capture_session: Some(capture_session),
        })
    }
}

#[async_trait(?Send)]
impl ExternalDriverSession for ProtocolExternalDriverSession<'_> {
    async fn request_terminal(
        &mut self,
    ) -> Result<CompatibilityTerminalReceipt, ExternalDriverError> {
        let request = self
            .terminal
            .request_terminal()
            .map_err(|_| ExternalDriverError::TerminalReceiptRejected)?;
        self.adapter
            .send(request)
            .await
            .map_err(|_| ExternalDriverError::TerminalReceiptRejected)?;
        let candidate = self
            .adapter
            .receive()
            .await
            .map_err(|_| ExternalDriverError::TerminalReceiptRejected)?;
        let bytes = self
            .terminal
            .accept_candidate(candidate)
            .map_err(|_| ExternalDriverError::TerminalReceiptRejected)?;
        let capture_session = self
            .capture_session
            .take()
            .ok_or(ExternalDriverError::TerminalReceiptRejected)?;
        CompatibilityTerminalReceipt::from_canonical_terminal_bytes(capture_session, &bytes)
            .map_err(|_| ExternalDriverError::TerminalReceiptRejected)
    }
}

/// Immutable segregation key for opt-in adapter reuse.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterPoolKey {
    task: ArtifactDigest,
    environment: ArtifactDigest,
    implementation: ArtifactDigest,
    role: u8,
    protocol: ArtifactDigest,
}

impl AdapterPoolKey {
    /// Creates a key containing all identities that may affect adapter state.
    pub fn new(
        task: ArtifactDigest,
        environment: ArtifactDigest,
        implementation: ArtifactDigest,
        role: AdapterRole,
        protocol: ArtifactDigest,
    ) -> Self {
        Self {
            task,
            environment,
            implementation,
            role: role_key(role),
            protocol,
        }
    }
}

fn role_key(role: AdapterRole) -> u8 {
    match role {
        AdapterRole::Tool => 0,
        AdapterRole::Policy => 1,
        AdapterRole::Environment => 2,
        AdapterRole::Heuristic => 3,
        AdapterRole::Driver => 4,
    }
}

/// Whether a checked-out adapter was freshly spawned or reset for reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterCheckoutOrigin {
    /// No suitable pooled adapter was retained.
    Fresh,
    /// A same-key adapter reset and acknowledged the supplied deterministic seed.
    Reused,
}

/// One adapter taken from the pool; callers must return it only after clean reuse checks.
pub struct AdapterCheckout {
    adapter: Box<dyn SupervisedAdapter>,
    origin: AdapterCheckoutOrigin,
}

impl AdapterCheckout {
    /// Reports whether this checkout forced a fresh adapter process.
    pub const fn is_fresh(&self) -> bool {
        matches!(self.origin, AdapterCheckoutOrigin::Fresh)
    }

    /// Returns the supervised adapter for one episode.
    pub fn into_adapter(self) -> Box<dyn SupervisedAdapter> {
        self.adapter
    }
}

/// Bounded-key adapter reuse pool; no key component may be omitted.
pub struct AdapterPool {
    max_idle: usize,
    entries: VecDeque<(AdapterPoolKey, Box<dyn SupervisedAdapter>)>,
}

impl Default for AdapterPool {
    fn default() -> Self {
        Self::with_capacity(64)
    }
}

impl AdapterPool {
    /// Creates a reuse pool with an explicit bounded idle-worker capacity.
    pub const fn with_capacity(max_idle: usize) -> Self {
        Self {
            max_idle,
            entries: VecDeque::new(),
        }
    }

    /// Retains a clean adapter under its complete immutable segregation key.
    pub async fn return_adapter(
        &mut self,
        key: AdapterPoolKey,
        adapter: Box<dyn SupervisedAdapter>,
    ) -> Result<(), AdapterSupervisionError> {
        if self.max_idle == 0 {
            let mut adapter = adapter;
            adapter.cancel_and_reap(CancelReason::HostShutdown).await?;
            return Ok(());
        }
        if self.entries.len() == self.max_idle {
            let Some((_, mut evicted)) = self.entries.pop_front() else {
                return Err(AdapterSupervisionError::PoolAccounting);
            };
            evicted.cancel_and_reap(CancelReason::HostShutdown).await?;
        }
        self.entries.push_back((key, adapter));
        Ok(())
    }

    /// Resets a same-key adapter or reaps it before starting a fresh replacement.
    pub async fn checkout_or_start(
        &mut self,
        key: AdapterPoolKey,
        reset: HostEnvelope,
        request: AdapterSpawnRequest,
        factory: &dyn AdapterRuntimeFactory,
    ) -> Result<AdapterCheckout, AdapterSupervisionError> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(entry_key, _)| entry_key == &key)
            && let Some((_, mut adapter)) = self.entries.remove(index)
        {
            match adapter.reset(reset.clone()).await {
                Ok(()) => {
                    return Ok(AdapterCheckout {
                        adapter,
                        origin: AdapterCheckoutOrigin::Reused,
                    });
                }
                Err(reset_error) => {
                    adapter
                        .cancel_and_reap(CancelReason::ResetFailure)
                        .await
                        .map_err(|recovery| AdapterSupervisionError::Recovery {
                            primary: Box::new(reset_error),
                            recovery: Box::new(recovery),
                        })?;
                }
            }
        }
        let mut adapter = factory.start(request).await?;
        if let Err(error) = adapter.reset(reset).await {
            return Err(
                match adapter.cancel_and_reap(CancelReason::ResetFailure).await {
                    Ok(_) => error,
                    Err(recovery) => AdapterSupervisionError::Recovery {
                        primary: Box::new(error),
                        recovery: Box::new(recovery),
                    },
                },
            );
        }
        Ok(AdapterCheckout {
            adapter,
            origin: AdapterCheckoutOrigin::Fresh,
        })
    }
}

/// Host-owned exact-profile authority derived before any adapter is provisioned.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactProfileAuthority {
    model_authorities: BTreeSet<ModelEndpointAuthority>,
    model_secret_environment: BTreeSet<String>,
}

impl ExactProfileAuthority {
    fn new(
        endpoints: impl IntoIterator<Item = impl AsRef<str>>,
        model_secret_environment: BTreeSet<String>,
    ) -> Result<Self, ProviderError> {
        let mut model_authorities = BTreeSet::new();
        for endpoint in endpoints {
            model_authorities.insert(ModelEndpointAuthority::parse(endpoint.as_ref())?);
        }
        Ok(Self {
            model_authorities,
            model_secret_environment,
        })
    }

    fn preflight(&self, provider: &ProviderProfile) -> Result<(), ProviderError> {
        provider.require_model_endpoint_isolation(&self.model_authorities)
    }

    fn adapter_environment(
        &self,
        mut environment: BTreeMap<String, String>,
    ) -> BTreeMap<String, String> {
        for name in &self.model_secret_environment {
            environment.remove(name);
        }
        environment
    }
}

/// Runtime-resolved permission to construct one NativeGraph exact-profile adapter request.
///
/// The value has no public constructor. Docker execution obtains it only from
/// its selected runtime's resolved provider profile and secret-environment map
/// before image build or container creation; callers therefore cannot claim an
/// empty authority or substitute a convenient provider profile.
#[derive(Clone, Debug)]
pub struct NativeGraphAdapterAuthorization {
    authority: ExactProfileAuthority,
    exact_spawn_token: ExactSpawnToken,
}

impl NativeGraphAdapterAuthorization {
    pub(crate) fn resolve(
        native_graph: &NativeGraphPackagePlan,
        capabilities: ProviderCapabilities,
        provider: ProviderProfile,
        model_secret_environment: BTreeMap<ModelSecretId, String>,
    ) -> Result<Self, EvalExecutionError> {
        if native_graph.profile() != NativeGraphProfile::NativeGraph
            || !capabilities.has_model_endpoint_isolation()
        {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "model endpoint isolation",
            ));
        }
        let expected_secrets = native_graph
            .model_bindings()
            .iter()
            .flat_map(|binding| binding.authentication.iter())
            .map(|authentication| authentication.secret.clone())
            .collect::<BTreeSet<_>>();
        if expected_secrets != model_secret_environment.keys().cloned().collect() {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "native graph model secret environment",
            ));
        }
        let authority = ExactProfileAuthority::new(
            native_graph
                .model_bindings()
                .iter()
                .flat_map(|binding| binding.urls.iter()),
            model_secret_environment.into_values().collect(),
        )
        .map_err(|_| EvalExecutionError::UnsupportedEnforcement("model endpoint isolation"))?;
        authority
            .preflight(&provider)
            .map_err(|_| EvalExecutionError::UnsupportedEnforcement("model endpoint isolation"))?;
        Ok(Self {
            authority,
            exact_spawn_token: ExactSpawnToken::new(),
        })
    }

    /// Constructs the only secret-stripped request accepted for a NativeGraph exact adapter.
    pub fn spawn_request(
        &self,
        argv: impl IntoIterator<Item = String>,
        environment: BTreeMap<String, String>,
        deadlines: AdapterLifecycleDeadlines,
    ) -> Result<AdapterSpawnRequest, AdapterSupervisionError> {
        AdapterSpawnRequest::for_exact_adapter(
            argv,
            self.authority.adapter_environment(environment),
            deadlines,
            self.exact_spawn_token.clone(),
        )
    }

    /// Consumes only a request minted by this exact resolved authorization.
    ///
    /// This prevents a caller from exchanging a post-preflight, arbitrary
    /// `AdapterSpawnRequest` for one which contains a mapped model-secret
    /// environment value before Docker forwards its `--env` arguments.
    pub(crate) fn authorize_spawn_request(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<AdapterSpawnRequest, AdapterSupervisionError> {
        let Some(token) = request.exact_spawn_token.as_ref() else {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "native graph authorization",
            ));
        };
        if !self.exact_spawn_token.matches(token) {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "native graph authorization",
            ));
        }
        Ok(request)
    }
}

/// Runtime-minted permission for one declared externally driven Docker request.
///
/// This authority is distinct from [`NativeGraphAdapterAuthorization`]: it
/// carries no model endpoint proof or secret mapping and can mint only the
/// immutable Driver argv selected by the resolved external package.
#[derive(Clone, Debug)]
pub struct ExternallyDrivenAdapterAuthorization {
    driver_argv: Vec<String>,
    container: String,
    deadlines: AdapterLifecycleDeadlines,
    external_spawn_token: ExactSpawnToken,
    has_minted_request: Rc<Cell<bool>>,
    has_authorized_request: Rc<Cell<bool>>,
    minted_deadlines: Rc<Cell<Option<AdapterLifecycleDeadlines>>>,
}

impl ExternallyDrivenAdapterAuthorization {
    pub(crate) fn resolve(
        package: &HarborTaskPackage,
        trial: &ResolvedEpisodeTrial,
        container: &str,
        deadlines: AdapterLifecycleDeadlines,
    ) -> Result<Self, EvalExecutionError> {
        let native_graph = package
            .native_graph()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "externally driven package",
            ))?;
        if native_graph.profile() != NativeGraphProfile::ExternallyDriven {
            return Err(EvalExecutionError::InvalidRecipe(
                "externally driven package profile",
            ));
        }
        if trial.package().source_digest() != package.source_digest()
            || trial.package().native_graph() != Some(native_graph)
        {
            return Err(EvalExecutionError::InvalidRecipe(
                "resolved external Driver trial",
            ));
        }
        let driver = native_graph
            .driver_adapter()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "declared external Driver adapter",
            ))?;
        if driver.role != AdapterRole::Driver {
            return Err(EvalExecutionError::InvalidRecipe(
                "declared external Driver adapter role",
            ));
        }
        if container.trim().is_empty() {
            return Err(EvalExecutionError::InvalidRecipe(
                "external Driver task container",
            ));
        }
        Ok(Self {
            driver_argv: driver.argv.clone(),
            container: container.to_owned(),
            deadlines,
            external_spawn_token: ExactSpawnToken::new(),
            has_minted_request: Rc::new(Cell::new(false)),
            has_authorized_request: Rc::new(Cell::new(false)),
            minted_deadlines: Rc::new(Cell::new(None)),
        })
    }

    #[cfg(test)]
    pub(crate) fn spawn_request(&self) -> Result<AdapterSpawnRequest, AdapterSupervisionError> {
        self.spawn_request_with_deadlines(self.deadlines)
    }

    pub(crate) fn spawn_request_with_deadlines(
        &self,
        deadlines: AdapterLifecycleDeadlines,
    ) -> Result<AdapterSpawnRequest, AdapterSupervisionError> {
        if self.has_minted_request.replace(true) {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "external Driver request already minted",
            ));
        }
        if !deadlines_are_within(deadlines, self.deadlines) {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "external Driver deadline",
            ));
        }
        self.minted_deadlines.set(Some(deadlines));
        AdapterSpawnRequest::for_external_driver(
            self.driver_argv.clone(),
            deadlines,
            self.external_spawn_token.clone(),
        )
    }

    pub(crate) fn authorize_spawn_request(
        &self,
        container: &str,
        request: AdapterSpawnRequest,
    ) -> Result<AdapterSpawnRequest, AdapterSupervisionError> {
        let Some(token) = request.external_spawn_token.as_ref() else {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "external Driver authorization",
            ));
        };
        if self.container != container
            || !self.external_spawn_token.matches(token)
            || request.argv != self.driver_argv
            || !request.environment.is_empty()
            || Some(request.deadlines) != self.minted_deadlines.get()
            || self.has_authorized_request.replace(true)
        {
            return Err(AdapterSupervisionError::InvalidSpawnRequest(
                "external Driver authorization",
            ));
        }
        Ok(request)
    }
}

fn deadlines_are_within(
    requested: AdapterLifecycleDeadlines,
    limit: AdapterLifecycleDeadlines,
) -> bool {
    requested.startup() <= limit.startup()
        && requested.reset() <= limit.reset()
        && requested.heartbeat() <= limit.heartbeat()
        && requested.idle() <= limit.idle()
        && requested.operation() <= limit.operation()
        && requested.cancel() <= limit.cancel()
        && requested.reap() <= limit.reap()
}

/// Supervision failure with no secret values in its diagnostic representation.
#[derive(Debug)]
pub enum AdapterSupervisionError {
    /// One lifecycle deadline was zero.
    InvalidDeadline(&'static str),
    /// One launch field cannot name a safe child process.
    InvalidSpawnRequest(&'static str),
    /// A peer attempted to raise a cap selected at admission.
    OutputLimitIncrease {
        /// Output stream whose cap was raised.
        field: &'static str,
        /// Requested cap.
        requested: usize,
        /// Immutable currently admitted cap.
        limit: usize,
    },
    /// The child ended before supplying its next protocol frame.
    EndOfStream,
    /// A child frame exceeded the configured strict stdout cap.
    StdoutFrameLimit { actual: usize, limit: usize },
    /// Bounded stderr diagnostics overflowed their configured retention cap.
    DiagnosticOutputLimit { actual: usize, limit: usize },
    /// The process was already successfully reaped.
    AlreadyReaped,
    /// A fixture or implementation rejected a protocol reset.
    ResetRejected(String),
    /// Task 4 rejected an unauthorized or malformed protocol transition.
    Protocol(ProtocolError),
    /// The primary failure and terminal cleanup both failed.
    Recovery {
        /// Original failure that required termination.
        primary: Box<AdapterSupervisionError>,
        /// Failure while terminating or reaping the child.
        recovery: Box<AdapterSupervisionError>,
    },
    /// The process provider refused or failed a named lifecycle operation.
    Process(String),
    /// Exact-profile provider admission failed before an adapter was spawned.
    Provider(ProviderError),
    /// Pool bookkeeping became internally inconsistent.
    PoolAccounting,
    /// The adapter spawner exceeded the dedicated startup deadline.
    StartupDeadlineElapsed,
    /// Reuse attempted to reset with a non-reset protocol transition.
    InvalidResetTransition,
}

impl AdapterSupervisionError {
    /// Creates a strict stdout-length violation without retaining child bytes.
    pub const fn bounded_stdout_frame(actual: usize, limit: usize) -> Self {
        Self::StdoutFrameLimit { actual, limit }
    }

    /// Creates a strict stderr diagnostic-length violation without retaining bytes.
    pub const fn bounded_diagnostic_output(actual: usize, limit: usize) -> Self {
        Self::DiagnosticOutputLimit { actual, limit }
    }
}

impl Display for AdapterSupervisionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDeadline(name) => {
                write!(formatter, "adapter {name} deadline must be positive")
            }
            Self::InvalidSpawnRequest(field) => {
                write!(formatter, "invalid adapter spawn request {field}")
            }
            Self::OutputLimitIncrease {
                field,
                requested,
                limit,
            } => write!(
                formatter,
                "adapter {field} cap {requested} exceeds strict limit {limit}"
            ),
            Self::EndOfStream => formatter.write_str("adapter closed its protocol stream"),
            Self::StdoutFrameLimit { actual, limit } => {
                write!(
                    formatter,
                    "adapter stdout frame exceeds {limit} bytes ({actual} bytes)"
                )
            }
            Self::DiagnosticOutputLimit { actual, limit } => {
                write!(
                    formatter,
                    "adapter diagnostic output exceeds {limit} bytes ({actual} bytes)"
                )
            }
            Self::AlreadyReaped => formatter.write_str("adapter process was already reaped"),
            Self::ResetRejected(reason) => write!(formatter, "adapter reset rejected: {reason}"),
            Self::Protocol(error) => write!(formatter, "adapter protocol violation: {error}"),
            Self::Recovery { primary, recovery } => {
                write!(
                    formatter,
                    "adapter failure: {primary}; cleanup failure: {recovery}"
                )
            }
            Self::Process(reason) => write!(formatter, "adapter process failure: {reason}"),
            Self::Provider(error) => {
                write!(formatter, "adapter provider admission failed: {error}")
            }
            Self::PoolAccounting => formatter.write_str("adapter pool bookkeeping failed"),
            Self::StartupDeadlineElapsed => formatter.write_str("adapter startup deadline elapsed"),
            Self::InvalidResetTransition => {
                formatter.write_str("adapter reset requires a reset transition")
            }
        }
    }
}

impl std::error::Error for AdapterSupervisionError {}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell, collections::BTreeMap, fs, num::NonZeroUsize, rc::Rc, time::Duration,
    };

    use crate::eval::native_graph::CompatibilityCaptureSession;
    use crate::eval::{
        AdapterEnvelope, AdapterExit, AdapterMessage, AdapterRole, AgentVariantRef, ArtifactDigest,
        AttemptId, CompatibilityTerminalReceipt, ExternalDriverError, ExternalDriverSession,
        HarborImporter, HarborSource, ModelEndpointIsolationProof, ModelIdentity,
        NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, ProtocolCapability,
        ProtocolLimits, ProviderCapability, ResourceLeaseRequest, RuntimeIdentity, SuiteRunId,
        SuiteTrialSpec, TrialBudget, TrialSpec,
    };

    use super::*;

    struct ReapFailsOnceProcess {
        attempts: Rc<RefCell<usize>>,
    }

    #[async_trait(?Send)]
    impl AdapterProcess for ReapFailsOnceProcess {
        async fn write_frame(
            &mut self,
            _: &[u8],
            _: Duration,
        ) -> Result<(), AdapterSupervisionError> {
            Ok(())
        }

        async fn read_stdout_frame(
            &mut self,
            _: usize,
            _: Duration,
        ) -> Result<Vec<u8>, AdapterSupervisionError> {
            Err(AdapterSupervisionError::EndOfStream)
        }

        async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
            Ok(Vec::new())
        }

        async fn cancel(
            &mut self,
            _: CancelReason,
            _: Duration,
        ) -> Result<(), AdapterSupervisionError> {
            Ok(())
        }

        async fn reap(&mut self, _: Duration) -> Result<AdapterExit, AdapterSupervisionError> {
            let mut attempts = self.attempts.borrow_mut();
            *attempts += 1;
            if *attempts == 1 {
                return Err(AdapterSupervisionError::Process(
                    "transient fixture reap failure".to_owned(),
                ));
            }
            Ok(AdapterExit::Reaped)
        }

        fn fence(&mut self) {}
    }

    #[test]
    fn external_authorization_validates_every_sealed_spawn_field_once() {
        let (imported, trial) = external_authorization_fixture();
        let deadlines = external_authorization_deadlines();

        let authorization = external_authorization(&imported, &trial, deadlines);
        let request = authorization.spawn_request().unwrap();
        assert!(request.external_spawn_token.is_some());
        assert_eq!(request.argv(), ["tools/driver.sh"]);
        assert!(request.environment().is_empty());
        assert_eq!(request.deadlines(), deadlines);

        let missing_token = AdapterSpawnRequest::for_non_model_adapter(
            ["tools/driver.sh".to_owned()],
            BTreeMap::new(),
            deadlines,
        )
        .unwrap();
        assert!(
            authorization
                .authorize_spawn_request("task-container", missing_token)
                .is_err()
        );

        let authorization = external_authorization(&imported, &trial, deadlines);
        let request = authorization.spawn_request().unwrap();
        assert!(
            authorization
                .authorize_spawn_request("substituted-container", request)
                .is_err()
        );

        let authorization = external_authorization(&imported, &trial, deadlines);
        let mut request = authorization.spawn_request().unwrap();
        request.argv.push("--substituted".to_owned());
        assert!(
            authorization
                .authorize_spawn_request("task-container", request)
                .is_err()
        );

        let authorization = external_authorization(&imported, &trial, deadlines);
        let mut request = authorization.spawn_request().unwrap();
        request
            .environment
            .insert("SECRET".to_owned(), "forbidden".to_owned());
        assert!(
            authorization
                .authorize_spawn_request("task-container", request)
                .is_err()
        );

        let authorization = external_authorization(&imported, &trial, deadlines);
        let mut request = authorization.spawn_request().unwrap();
        request.deadlines = AdapterLifecycleDeadlines::default();
        assert!(
            authorization
                .authorize_spawn_request("task-container", request)
                .is_err()
        );

        let authorization = external_authorization(&imported, &trial, deadlines);
        let request = authorization.spawn_request().unwrap();
        let replay = request.clone();
        authorization
            .authorize_spawn_request("task-container", request)
            .unwrap();
        assert!(
            authorization
                .authorize_spawn_request("task-container", replay)
                .is_err()
        );
        assert!(authorization.spawn_request().is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn strict_adapter_retries_cleanup_after_a_transient_reap_failure() {
        let attempts = Rc::new(RefCell::new(0));
        let mut adapter = StrictSupervisedAdapter::from_prestarted_process(
            driver_config(),
            Box::new(ReapFailsOnceProcess {
                attempts: Rc::clone(&attempts),
            }),
            AdapterLifecycleDeadlines::default(),
            1024,
            1024,
        );

        assert!(
            adapter
                .cancel_and_reap(CancelReason::HostShutdown)
                .await
                .is_err()
        );
        assert!(matches!(
            adapter.cancel_and_reap(CancelReason::HostShutdown).await,
            Ok(AdapterExit::Reaped)
        ));
        assert_eq!(*attempts.borrow(), 2);
    }

    fn external_authorization_fixture() -> (crate::eval::ImportedTask, ResolvedEpisodeTrial) {
        let task = tempfile::tempdir().unwrap();
        fs::create_dir_all(task.path().join("environment")).unwrap();
        fs::create_dir_all(task.path().join("tests")).unwrap();
        fs::create_dir_all(task.path().join("tools")).unwrap();
        fs::write(task.path().join("environment/Dockerfile"), "FROM scratch\n").unwrap();
        fs::write(task.path().join("instruction.md"), "Do work.\n").unwrap();
        fs::write(task.path().join("tests/test.sh"), "exit 0\n").unwrap();
        fs::write(
            task.path().join("task.toml"),
            r#"schema_version = "1.1"

[task]
name = "example/external-authorization"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
external_driver_factory_id = "fixture"
"#,
        )
        .unwrap();
        fs::write(
            task.path().join("adapters.toml"),
            r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
        )
        .unwrap();
        fs::write(task.path().join("tools/driver.sh"), "#!/bin/sh\nexit 0\n").unwrap();
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task.path().to_string_lossy()).unwrap())
            .unwrap();
        let trial = TrialSpec::new(
            imported.task.clone(),
            AgentVariantRef::new("external-driver").unwrap(),
            ModelIdentity::new("compatibility", "opaque-driver").unwrap(),
            7,
            PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
            TrialBudget::new(30.0, 30.0).unwrap(),
            ArtifactDigest::from_bytes(b"environment"),
            ArtifactDigest::from_bytes(b"verifier"),
            RuntimeIdentity::new("external").unwrap(),
        )
        .unwrap();
        let manifest = NativeGraphSuiteManifest::new(vec![
            SuiteTrialSpec::from_imported(
                imported.clone(),
                trial,
                NonZeroUsize::new(1).unwrap(),
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
        let trial = manifest
            .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(b"external-run")))
            .unwrap()
            .trials()[0]
            .clone();
        (imported, trial)
    }

    fn external_authorization(
        imported: &crate::eval::ImportedTask,
        trial: &ResolvedEpisodeTrial,
        deadlines: AdapterLifecycleDeadlines,
    ) -> ExternallyDrivenAdapterAuthorization {
        ExternallyDrivenAdapterAuthorization::resolve(
            &imported.package,
            trial,
            "task-container",
            deadlines,
        )
        .unwrap()
    }

    fn external_authorization_deadlines() -> AdapterLifecycleDeadlines {
        AdapterLifecycleDeadlines::new(
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(3),
            Duration::from_secs(4),
            Duration::from_secs(5),
            Duration::from_secs(6),
            Duration::from_secs(7),
        )
        .unwrap()
    }

    #[test]
    fn resolved_native_authorization_rejects_missing_secret_mapping_and_strips_it_at_spawn() {
        let task = tempfile::tempdir().unwrap();
        fs::create_dir_all(task.path().join("environment")).unwrap();
        fs::create_dir_all(task.path().join("tests")).unwrap();
        fs::create_dir_all(task.path().join("tools")).unwrap();
        fs::write(task.path().join("environment/Dockerfile"), "FROM scratch\n").unwrap();
        fs::write(task.path().join("instruction.md"), "Do work.\n").unwrap();
        fs::write(task.path().join("tests/test.sh"), "exit 0\n").unwrap();
        fs::write(
            task.path().join("task.toml"),
            r#"schema_version = "1.1"

[task]
name = "example/native-authorization"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
        )
        .unwrap();
        fs::write(task.path().join("agent_graph.json"), "{}\n").unwrap();
        fs::write(
            task.path().join("models.toml"),
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
authentication = [{ header = "authorization", secret = "model-key" }]
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
        )
        .unwrap();
        fs::write(
            task.path().join("adapters.toml"),
            r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.sh"]
executable = "tools/adapter.sh"
"#,
        )
        .unwrap();
        fs::write(task.path().join("tools/adapter.sh"), "#!/bin/sh\nexit 0\n").unwrap();

        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task.path().to_string_lossy()).unwrap())
            .unwrap();
        let native = imported.package.native_graph().unwrap();
        let profile = ProviderProfile::new(
            "runtime-no-egress",
            vec![ProviderCapability::ModelEndpointIsolation],
        )
        .unwrap()
        .with_model_endpoint_isolation(ModelEndpointIsolationProof::NoAdapterEgress)
        .unwrap();
        let capabilities = ProviderCapabilities::none().with_model_endpoint_isolation();

        let missing = NativeGraphAdapterAuthorization::resolve(
            native,
            capabilities,
            profile.clone(),
            BTreeMap::new(),
        )
        .expect_err("each package model secret needs one resolved host environment name");
        assert!(matches!(
            missing,
            EvalExecutionError::UnsupportedEnforcement("native graph model secret environment")
        ));

        let secret = native.model_bindings()[0].authentication[0].secret.clone();
        let authorization = NativeGraphAdapterAuthorization::resolve(
            native,
            capabilities,
            profile,
            BTreeMap::from([(secret, "MODEL_API_KEY".to_owned())]),
        )
        .expect("runtime-resolved proof and exact secret map authorize the adapter");
        let request = authorization
            .spawn_request(
                ["adapter".to_owned()],
                BTreeMap::from([
                    ("MODEL_API_KEY".to_owned(), "model-secret-value".to_owned()),
                    ("TASK_VALUE".to_owned(), "kept".to_owned()),
                ]),
                AdapterLifecycleDeadlines::default(),
            )
            .expect("authorized adapter spawn request is valid");
        assert_eq!(request.environment().get("MODEL_API_KEY"), None);
        assert_eq!(
            request.environment().get("TASK_VALUE").map(String::as_str),
            Some("kept")
        );

        let injected = AdapterSpawnRequest::for_non_model_adapter(
            ["adapter".to_owned()],
            BTreeMap::from([(
                "MODEL_API_KEY".to_owned(),
                "injected-after-preflight".to_owned(),
            )]),
            AdapterLifecycleDeadlines::default(),
        )
        .expect("a generic non-model request is syntactically valid");
        let error = authorization
            .authorize_spawn_request(injected)
            .expect_err("an exact Docker spawner rejects an unminted secret-bearing request");
        assert!(matches!(
            error,
            AdapterSupervisionError::InvalidSpawnRequest("native graph authorization")
        ));
    }

    struct TerminalCandidateAdapter {
        candidate: Option<AdapterEnvelope>,
        sent: Vec<HostEnvelope>,
    }

    #[async_trait(?Send)]
    impl SupervisedAdapter for TerminalCandidateAdapter {
        async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError> {
            self.sent.push(message);
            Ok(())
        }

        async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
            self.candidate
                .take()
                .ok_or(AdapterSupervisionError::EndOfStream)
        }

        async fn receive_heartbeat(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
            Err(AdapterSupervisionError::EndOfStream)
        }

        async fn receive_idle(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
            Err(AdapterSupervisionError::EndOfStream)
        }

        async fn reset(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
            Err(AdapterSupervisionError::InvalidResetTransition)
        }

        fn release_download_handle(
            &mut self,
            _: &ArtifactDownloadHandle,
        ) -> Result<(), AdapterSupervisionError> {
            Ok(())
        }

        async fn cancel_and_reap(
            &mut self,
            _: CancelReason,
        ) -> Result<AdapterExit, AdapterSupervisionError> {
            Ok(AdapterExit::Reaped)
        }
    }

    fn driver_config() -> AdapterProtocolConfig {
        AdapterProtocolConfig::new(
            AdapterRole::Driver,
            "external-episode",
            [ProtocolCapability::Driver].into_iter().collect(),
            BTreeSet::new(),
            ProtocolLimits::default(),
        )
        .expect("driver-only fixture config is valid")
    }

    fn capture_session() -> CompatibilityCaptureSession {
        CompatibilityCaptureSession::new(
            ArtifactDigest::from_bytes(b"package"),
            ArtifactDigest::from_bytes(b"source"),
            ArtifactDigest::from_bytes(b"task"),
            ArtifactDigest::from_bytes(b"environment"),
            ArtifactDigest::from_bytes(b"trial"),
            AttemptId::new("attempt").expect("fixture attempt is valid"),
        )
    }

    fn candidate(operation: &str, output: serde_json::Value) -> AdapterEnvelope {
        AdapterEnvelope::new(
            "external-episode",
            "external-driver-terminal",
            1,
            operation,
            AdapterMessage::EpisodeTerminalCandidate { output },
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_driver_session_accepts_one_correlated_candidate_as_a_digest_receipt() {
        let mut adapter = TerminalCandidateAdapter {
            candidate: Some(candidate(
                "external-driver-terminal",
                serde_json::json!({"answer": "done"}),
            )),
            sent: Vec::new(),
        };
        let mut session =
            ProtocolExternalDriverSession::new(&mut adapter, driver_config(), capture_session())
                .expect("driver-only config creates the terminal boundary");

        let receipt = session
            .request_terminal()
            .await
            .expect("the one matching candidate becomes an opaque receipt");
        let expected = CompatibilityTerminalReceipt::from_canonical_terminal_bytes(
            capture_session(),
            br#"{"answer":"done"}"#,
        )
        .expect("fixture canonical terminal receipt is bounded");
        assert_eq!(receipt, expected);
        drop(session);
        assert!(matches!(
            adapter.sent.as_slice(),
            [HostEnvelope {
                sequence: 1,
                operation,
                message: HostMessage::RequestEpisodeTerminal { .. },
                ..
            }] if operation == "external-driver-terminal"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_driver_session_refuses_a_non_candidate_terminal_response() {
        let mut adapter = TerminalCandidateAdapter {
            candidate: Some(AdapterEnvelope::new(
                "external-episode",
                "external-driver-terminal",
                1,
                "external-driver-terminal",
                AdapterMessage::OperationFailed {
                    code: "driver-failed".to_owned(),
                    details: serde_json::json!({"untrusted": "details"}),
                },
            )),
            sent: Vec::new(),
        };
        let mut session =
            ProtocolExternalDriverSession::new(&mut adapter, driver_config(), capture_session())
                .expect("driver-only config creates the terminal boundary");
        assert_eq!(
            session.request_terminal().await,
            Err(ExternalDriverError::TerminalReceiptRejected)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_driver_session_refuses_a_foreign_terminal_correlation() {
        let mut foreign = TerminalCandidateAdapter {
            candidate: Some(candidate(
                "foreign-terminal",
                serde_json::json!({"answer": "done"}),
            )),
            sent: Vec::new(),
        };
        let mut session =
            ProtocolExternalDriverSession::new(&mut foreign, driver_config(), capture_session())
                .expect("driver-only config creates the terminal boundary");
        assert_eq!(
            session.request_terminal().await,
            Err(ExternalDriverError::TerminalReceiptRejected)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_driver_session_refuses_an_oversized_candidate_without_exposing_it() {
        let mut oversized = TerminalCandidateAdapter {
            candidate: Some(candidate(
                "external-driver-terminal",
                serde_json::json!({"terminal": "x".repeat(CompatibilityTerminalReceipt::MAX_CANONICAL_BYTES)}),
            )),
            sent: Vec::new(),
        };
        let mut session =
            ProtocolExternalDriverSession::new(&mut oversized, driver_config(), capture_session())
                .expect("driver-only config creates the terminal boundary");
        let error = session
            .request_terminal()
            .await
            .expect_err("oversized terminal bytes are refused before a receipt exists");
        assert_eq!(error, ExternalDriverError::TerminalReceiptRejected);
        assert!(!format!("{error:?}").contains('x'));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_driver_session_refuses_a_second_terminal_request_after_settlement() {
        let mut settled = TerminalCandidateAdapter {
            candidate: Some(candidate(
                "external-driver-terminal",
                serde_json::json!({"answer": "done"}),
            )),
            sent: Vec::new(),
        };
        let mut session =
            ProtocolExternalDriverSession::new(&mut settled, driver_config(), capture_session())
                .expect("driver-only config creates the terminal boundary");
        session
            .request_terminal()
            .await
            .expect("the first candidate settles the boundary");
        assert_eq!(
            session.request_terminal().await,
            Err(ExternalDriverError::TerminalReceiptRejected)
        );
    }
}
