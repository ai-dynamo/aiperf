// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow callback boundary between an acquired task environment and Rust-owned graph execution.

use std::{cell::RefCell, future::Future, pin::Pin, rc::Rc};

#[cfg(feature = "engine")]
use std::time::Duration;

use async_trait::async_trait;

use super::EvalExecutionError;
#[cfg(feature = "engine")]
use super::EvalExecutionPhase;
#[cfg(feature = "engine")]
use crate::clock::Clock;
#[cfg(feature = "engine")]
use crate::engine::graph_execution::NativeGraphLivePolicyCallSummary;
#[cfg(feature = "engine")]
use crate::eval::{
    BoundNativeGraphEnvironmentStepper, CompatibilityTerminalReceipt, FrozenArtifact,
    FrozenRolloutEvidence, NativeGraphAttemptAuthority, NativeGraphRolloutTransitionReceipt,
    NativeGraphWorkspacePatchContract, PreparedNativeGraphLiveRolloutCoordinator,
};

/// Backend-owned external Driver session with no raw process or terminal-payload access.
#[cfg(feature = "engine")]
#[async_trait(?Send)]
pub(crate) trait ExternallyDrivenEpisodeSession {
    /// Runs the exact prepared Driver through its sole terminal protocol exchange.
    async fn request_terminal(
        &mut self,
    ) -> Result<CompatibilityTerminalReceipt, EvalExecutionError>;

    /// Cancels and reaps the started Driver once the episode outcome is fixed.
    async fn cancel_and_reap(&mut self) -> Result<(), EvalExecutionError>;
}

/// Runs one terminal-only external Driver transaction and preserves terminal cleanup ownership.
#[cfg(feature = "engine")]
pub(crate) async fn run_externally_driven_episode_session<T>(
    clock: Rc<dyn Clock>,
    timeout: Duration,
    remaining: Result<Duration, EvalExecutionError>,
    session: &mut dyn ExternallyDrivenEpisodeSession,
    after_terminal: impl FnOnce(CompatibilityTerminalReceipt) -> Result<T, EvalExecutionError>,
) -> Result<T, EvalExecutionError> {
    let outcome = match remaining {
        Ok(remaining) => {
            let terminal = session.request_terminal();
            let timer = clock.sleep(remaining.as_nanos().min(i64::MAX as u128) as i64);
            tokio::pin!(terminal);
            tokio::pin!(timer);
            tokio::select! {
                biased;
                result = &mut terminal => match result {
                    Ok(receipt) => after_terminal(receipt),
                    Err(error) => Err(error),
                },
                () = &mut timer => Err(EvalExecutionError::Timeout {
                    phase: EvalExecutionPhase::Agent,
                    timeout,
                }),
            }
        }
        Err(error) => Err(error),
    };
    let cleanup = session.cancel_and_reap().await;
    match (outcome, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) | (Err(error), Ok(())) => Err(error),
        (Err(primary), Err(cleanup)) => Err(EvalExecutionError::ProcessFailure(format!(
            "external Driver episode failed: {primary}; cleanup failed: {cleanup}"
        ))),
    }
}

/// Opaque, immutable rollout start facts retained by the Docker lease before provisioning.
///
/// The engine callback can create this only from a resolved trial and selected worker-local
/// bindings. Its constructor and accessors remain crate-private so callbacks cannot mint or
/// substitute package, coordinator, or attempt authority after a lease is exposed.
#[cfg(feature = "engine")]
pub struct NativeGraphLeaseRolloutStart {
    stepper: BoundNativeGraphEnvironmentStepper,
    coordinator: PreparedNativeGraphLiveRolloutCoordinator,
    authority: NativeGraphAttemptAuthority,
    live_policy_summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
    workspace_patch: NativeGraphWorkspacePatchContract,
}

#[cfg(feature = "engine")]
impl NativeGraphLeaseRolloutStart {
    pub(crate) fn new(
        stepper: BoundNativeGraphEnvironmentStepper,
        coordinator: PreparedNativeGraphLiveRolloutCoordinator,
        authority: NativeGraphAttemptAuthority,
        live_policy_summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
        workspace_patch: NativeGraphWorkspacePatchContract,
    ) -> Self {
        Self {
            stepper,
            coordinator,
            authority,
            live_policy_summary,
            workspace_patch,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        BoundNativeGraphEnvironmentStepper,
        PreparedNativeGraphLiveRolloutCoordinator,
        NativeGraphAttemptAuthority,
        Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
        NativeGraphWorkspacePatchContract,
    ) {
        (
            self.stepper,
            self.coordinator,
            self.authority,
            self.live_policy_summary,
            self.workspace_patch,
        )
    }
}

/// Opaque, lease-owned operation which starts the package-declared environment adapter.
///
/// The operation intentionally exposes neither its process, spawn request, Docker runtime,
/// nor secret environment. The evaluation backend owns terminal cleanup after the callback.
#[async_trait(?Send)]
pub trait NativeGraphEnvironmentAdapterStart {
    /// Starts the already-authorized package-declared environment adapter exactly once.
    async fn start(&mut self) -> Result<(), EvalExecutionError>;

    /// Starts one Rust-owned supervised rollout session from exact selected package facts.
    ///
    /// The backend retains the adapter process and cleanup authority. Callbacks receive only
    /// descriptor-only reset and transition receipts through [`NativeGraphEnvironmentRolloutSession`].
    #[cfg(feature = "engine")]
    async fn start_rollout(&mut self) -> Result<(), EvalExecutionError> {
        Err(EvalExecutionError::InvalidRecipe(
            "NativeGraph supervised rollout session",
        ))
    }

    /// Borrows the descriptor-only rollout session after [`Self::start_rollout`] succeeds.
    #[cfg(feature = "engine")]
    fn rollout_session(
        &mut self,
    ) -> Result<&mut dyn NativeGraphEnvironmentRolloutSession, EvalExecutionError> {
        Err(EvalExecutionError::InvalidRecipe(
            "NativeGraph supervised rollout session",
        ))
    }
}

/// Descriptor-only live-rollout operations exposed to a Rust graph callback.
///
/// The callback cannot reach child-process, spawn, protocol-handle, or raw model-decision
/// authority through this interface. The backend owns all terminal cleanup.
#[cfg(feature = "engine")]
#[async_trait(?Send)]
pub trait NativeGraphEnvironmentRolloutSession {
    /// Resets the selected environment and returns the frozen initial observation descriptor.
    async fn reset(&mut self) -> Result<FrozenArtifact, EvalExecutionError>;

    /// Obtains and dispatches one selected model decision, returning frozen transition facts only.
    async fn step(
        &mut self,
        observation: &FrozenArtifact,
    ) -> Result<NativeGraphRolloutTransitionReceipt, EvalExecutionError>;

    /// Freezes exactly one terminal trajectory into verifier-isolated evidence.
    fn freeze(&mut self) -> Result<FrozenRolloutEvidence, EvalExecutionError>;
}

/// Borrowed capabilities of an already-authorized NativeGraph task environment.
///
/// Implementations are created only by an evaluation backend after it has
/// acquired the immutable package environment and validated the selected
/// provider authorization. The lease deliberately has no Docker-runtime,
/// command-execution, or host-secret access.
pub trait NativeGraphEpisodeLease {
    /// Whether the backend completed exact-profile authorization before this lease.
    fn is_authorized(&self) -> bool;

    /// Whether the task environment was acquired before this lease was exposed.
    fn is_environment_acquired(&self) -> bool;

    /// Returns the imported task's immutable agent instruction.
    fn instruction(&self) -> &str;

    /// Returns the backend-minted operation for the selected environment adapter.
    ///
    /// Leases for non-rollout graph programs deliberately expose no operation.
    fn environment_adapter_start(
        &mut self,
    ) -> Result<&mut dyn NativeGraphEnvironmentAdapterStart, EvalExecutionError> {
        Err(EvalExecutionError::InvalidRecipe(
            "NativeGraph rollout environment adapter",
        ))
    }
}

/// Backend-only terminal ownership for an acquired NativeGraph episode lease.
///
/// [`NativeGraphEpisodeCallback`] receives only [`NativeGraphEpisodeLease`],
/// never this trait, so adapter cleanup remains Rust-owned after declared
/// artifact collection and verification, or immediately after callback failure.
pub trait NativeGraphEpisodeBackendLease: NativeGraphEpisodeLease {
    /// Whether a started adapter must be reaped before the verifier observes task artifacts.
    ///
    /// Isolated rollout adapters run outside the task container and must be stopped before
    /// collection. Legacy adapter starts execute in the task container and retain the historic
    /// post-verifier cleanup order.
    fn reaps_environment_adapter_before_artifact_collection(&self) -> bool {
        false
    }

    /// Reaps an environment adapter started through this lease.
    fn reap_environment_adapter<'lease>(
        &'lease mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), EvalExecutionError>> + 'lease>>;
}

struct NativeGraphEpisodeCallbackLease<'lease> {
    backend: &'lease mut dyn NativeGraphEpisodeBackendLease,
}

impl NativeGraphEpisodeLease for NativeGraphEpisodeCallbackLease<'_> {
    fn is_authorized(&self) -> bool {
        self.backend.is_authorized()
    }

    fn is_environment_acquired(&self) -> bool {
        self.backend.is_environment_acquired()
    }

    fn instruction(&self) -> &str {
        self.backend.instruction()
    }

    fn environment_adapter_start(
        &mut self,
    ) -> Result<&mut dyn NativeGraphEnvironmentAdapterStart, EvalExecutionError> {
        self.backend.environment_adapter_start()
    }
}

/// Rust-owned execution of one NativeGraph agent episode within an acquired lease.
#[async_trait(?Send)]
pub trait NativeGraphEpisodeCallback {
    /// Executes model and supervised-adapter stages before artifact collection.
    async fn run(
        &mut self,
        lease: &mut dyn NativeGraphEpisodeLease,
    ) -> Result<(), EvalExecutionError>;

    /// Transfers one sealed rollout start to the backend before it provisions an adapter.
    ///
    /// The default retains ordinary graph callbacks. The returned value is opaque: callback code
    /// cannot create or inspect the selected worker-local stepper, policy coordinator, or attempt
    /// authority after the backend has retained it.
    #[cfg(feature = "engine")]
    fn take_lease_rollout_start(&mut self) -> Option<NativeGraphLeaseRolloutStart> {
        None
    }

    /// Transfers one fully frozen rollout document after a successful callback.
    ///
    /// The default keeps non-rollout graph callbacks byte-for-byte behavior-compatible.
    #[cfg(feature = "engine")]
    fn take_rollout_evidence(&mut self) -> Option<FrozenRolloutEvidence> {
        None
    }
}

/// Invokes an authorized NativeGraph callback before collection and verification.
///
/// The backend supplies `after_callback` for its existing declared artifact and
/// verifier lifecycle. A successful callback keeps the adapter alive until that
/// lifecycle completes; a failed callback skips it and reaps immediately. The
/// caller still owns its ordinary reverse cleanup transaction.
pub async fn run_native_graph_episode_callback<T, Lease>(
    callback: &mut dyn NativeGraphEpisodeCallback,
    lease: &mut Lease,
    mut after_callback: impl FnMut() -> Result<T, EvalExecutionError>,
) -> Result<T, EvalExecutionError>
where
    Lease: NativeGraphEpisodeBackendLease,
{
    if !lease.is_authorized() {
        return Err(EvalExecutionError::UnsupportedEnforcement(
            "native graph exact-profile authorization",
        ));
    }
    if !lease.is_environment_acquired() {
        return Err(EvalExecutionError::InvalidRecipe(
            "acquired native graph task environment",
        ));
    }
    let callback_outcome = {
        let mut callback_lease = NativeGraphEpisodeCallbackLease { backend: lease };
        callback.run(&mut callback_lease).await
    };
    match callback_outcome {
        Ok(()) => {
            if lease.reaps_environment_adapter_before_artifact_collection() {
                lease.reap_environment_adapter().await?;
            }
            let lifecycle_outcome = after_callback();
            let cleanup_outcome = if lease.reaps_environment_adapter_before_artifact_collection() {
                Ok(())
            } else {
                lease.reap_environment_adapter().await
            };
            match (lifecycle_outcome, cleanup_outcome) {
                (Ok(value), Ok(())) => Ok(value),
                (Ok(_), Err(error)) => Err(error),
                (Err(error), Ok(())) => Err(error),
                (Err(primary), Err(cleanup)) => Err(EvalExecutionError::ProcessFailure(format!(
                    "native graph collection or verification failed: {primary}; environment adapter cleanup failed: {cleanup}"
                ))),
            }
        }
        Err(primary) => match lease.reap_environment_adapter().await {
            Ok(()) => Err(primary),
            Err(cleanup) => Err(EvalExecutionError::ProcessFailure(format!(
                "native graph callback failed: {primary}; environment adapter cleanup failed: {cleanup}"
            ))),
        },
    }
}
