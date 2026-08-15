// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Verifier sandbox isolation and declared-artifact materialization.

use std::fmt::{self, Display, Formatter};

use crate::eval::ArtifactDigest;

use super::DeclaredArtifactTransfer;

/// Sandbox topology selected for verifier execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierMode {
    /// The verifier uses the task's explicitly shared sandbox.
    Shared,
    /// The verifier uses a fresh sandbox or restored snapshot.
    Separate,
}

/// Boundary that prepares an isolated verifier environment.
pub trait VerifierSandboxFactory {
    /// Prepares the verifier with only declared artifact paths and identities.
    fn prepare(
        &self,
        mode: VerifierMode,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<(), VerifierExecutionError>;
}

/// Materializes the declared artifact handoff in the selected verifier sandbox.
pub fn prepare_verifier(
    factory: &(impl VerifierSandboxFactory + ?Sized),
    mode: VerifierMode,
    transfer: &DeclaredArtifactTransfer,
) -> Result<(), VerifierExecutionError> {
    factory.prepare(mode, transfer.artifacts())
}

/// Failed preparation of an isolated verifier sandbox.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierExecutionError {
    /// The sandbox provider rejected the declared verifier handoff.
    PreparationFailed(String),
}

impl Display for VerifierExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreparationFailed(reason) => {
                write!(formatter, "verifier preparation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for VerifierExecutionError {}
