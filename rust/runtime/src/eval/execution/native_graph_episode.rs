// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow callback boundary between an acquired task environment and Rust-owned graph execution.

use async_trait::async_trait;

use super::EvalExecutionError;

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
/// verifier lifecycle. If graph execution fails, that closure is not entered;
/// the caller still owns its ordinary reverse cleanup transaction.
pub async fn run_native_graph_episode_callback<T>(
    callback: &mut dyn NativeGraphEpisodeCallback,
    lease: &mut dyn NativeGraphEpisodeLease,
    mut after_callback: impl FnMut() -> Result<T, EvalExecutionError>,
) -> Result<T, EvalExecutionError> {
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
    callback.run(lease).await?;
    after_callback()
}
