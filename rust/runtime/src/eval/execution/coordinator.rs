// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native P0 ordering for import, sandbox, and verifier preparation.

use std::fmt::{self, Display, Formatter};

use crate::eval::{
    DeclaredArtifactTransfer, HarborImportError, HarborImporter, HarborSource, ImportedTask,
    SourceAcquirer, VerifierExecutionError, VerifierMode, VerifierSandboxFactory, prepare_verifier,
};

use super::{EvalExecutionError, EvalSandboxFactory, HarborAgentContract, HarborSandboxRecipe};

/// Native composition boundary for the P0 import and preparation lifecycle.
pub struct HarborEvaluationCoordinator<'a> {
    acquirer: &'a dyn SourceAcquirer,
    sandbox: &'a dyn EvalSandboxFactory,
    verifier: &'a dyn VerifierSandboxFactory,
}

impl<'a> HarborEvaluationCoordinator<'a> {
    /// Creates a coordinator over caller-owned native source and sandbox boundaries.
    pub fn new(
        acquirer: &'a dyn SourceAcquirer,
        sandbox: &'a dyn EvalSandboxFactory,
        verifier: &'a dyn VerifierSandboxFactory,
    ) -> Self {
        Self {
            acquirer,
            sandbox,
            verifier,
        }
    }

    /// Imports before provisioning, then preflights, opens, and prepares the verifier.
    pub fn prepare(
        &self,
        source: &HarborSource,
        recipe: &HarborSandboxRecipe,
        agent: &HarborAgentContract,
        verifier_mode: VerifierMode,
        transfer: &DeclaredArtifactTransfer,
    ) -> Result<ImportedTask, HarborEvaluationError> {
        let imported = HarborImporter::new(self.acquirer).import(source)?;
        self.sandbox.preflight(recipe, agent)?;
        self.sandbox.open(recipe)?;
        prepare_verifier(self.verifier, verifier_mode, transfer)?;
        Ok(imported)
    }
}

/// Failure during native P0 lifecycle preparation.
#[derive(Debug)]
pub enum HarborEvaluationError {
    /// Source import failed before environment provisioning.
    Import(HarborImportError),
    /// Agent sandbox validation or opening failed.
    Sandbox(EvalExecutionError),
    /// Verifier isolation preparation failed.
    Verifier(VerifierExecutionError),
}

impl Display for HarborEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Import(error) => error.fmt(formatter),
            Self::Sandbox(error) => error.fmt(formatter),
            Self::Verifier(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HarborEvaluationError {}

impl From<HarborImportError> for HarborEvaluationError {
    fn from(error: HarborImportError) -> Self {
        Self::Import(error)
    }
}

impl From<EvalExecutionError> for HarborEvaluationError {
    fn from(error: EvalExecutionError) -> Self {
        Self::Sandbox(error)
    }
}

impl From<VerifierExecutionError> for HarborEvaluationError {
    fn from(error: VerifierExecutionError) -> Self {
        Self::Verifier(error)
    }
}
