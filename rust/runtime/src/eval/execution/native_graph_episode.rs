// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow callback boundary between an acquired task environment and Rust-owned graph execution.

use std::{future::Future, pin::Pin};

use async_trait::async_trait;

use super::EvalExecutionError;

/// Opaque, lease-owned operation which starts the package-declared environment adapter.
///
/// The operation intentionally exposes neither its process, spawn request, Docker runtime,
/// nor secret environment. The evaluation backend owns terminal cleanup after the callback.
#[async_trait(?Send)]
pub trait NativeGraphEnvironmentAdapterStart {
    /// Starts the already-authorized package-declared environment adapter exactly once.
    async fn start(&mut self) -> Result<(), EvalExecutionError>;
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
    /// Reaps an environment adapter started through this lease.
    fn reap_environment_adapter<'lease>(
        &'lease mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), EvalExecutionError>> + 'lease>> {
        Box::pin(async { Ok(()) })
    }
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
            let lifecycle_outcome = after_callback();
            let cleanup_outcome = lease.reap_environment_adapter().await;
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
