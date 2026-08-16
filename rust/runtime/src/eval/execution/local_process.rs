// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Temporary-root local process sandboxing for native P0 evaluation.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use tempfile::TempDir;

use crate::eval::HarborTaskPackage;

use super::{EvalExecutionError, HarborSandboxRecipe};

/// Selects the isolated root materialized for one evaluation participant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxRole {
    /// Root in which the selected agent executes.
    Agent,
    /// Root for a verifier explicitly authorized to share the task sandbox.
    SharedVerifier,
    /// Fresh root for a separately provisioned verifier.
    SeparateVerifier,
}

/// Concrete local process provider for deterministic P0 package execution.
#[derive(Debug, Default)]
pub struct LocalProcessSandbox;

impl LocalProcessSandbox {
    /// Creates an empty local-process provider.
    pub const fn new() -> Self {
        Self
    }

    /// Writes exactly the acquired package bytes into an isolated temporary root.
    pub fn materialize(
        &self,
        _: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        _: SandboxRole,
    ) -> Result<MaterializedSandbox, EvalExecutionError> {
        let lease = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        fs::write(lease.path().join("task.json"), package.source_bytes())
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        fs::create_dir_all(lease.path().join("results"))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        Ok(MaterializedSandbox { lease })
    }
}

/// A live local sandbox root whose lease removes it when evaluation finishes.
#[derive(Debug)]
pub struct MaterializedSandbox {
    lease: TempDir,
}

impl MaterializedSandbox {
    /// Returns the private process root.
    pub fn root(&self) -> &Path {
        self.lease.path()
    }

    /// Maps a declared absolute artifact path into this private root.
    pub fn artifact_path(&self, declared_path: &str) -> Result<PathBuf, EvalExecutionError> {
        let relative = declared_path
            .strip_prefix('/')
            .filter(|path| !path.is_empty() && !path.split('/').any(|part| part == "." || part == ".."))
            .ok_or_else(|| EvalExecutionError::Materialization("invalid declared artifact path".to_owned()))?;
        Ok(self.root().join(relative))
    }

    /// Runs an argv with no inherited environment in this sandbox root.
    pub fn run(
        &self,
        argv: &[String],
        environment: &[(String, String)],
    ) -> Result<ProcessOutput, EvalExecutionError> {
        let (program, arguments) = argv.split_first().ok_or(EvalExecutionError::InvalidCommand)?;
        if program.trim().is_empty() || arguments.iter().any(|argument| argument.trim().is_empty()) {
            return Err(EvalExecutionError::InvalidCommand);
        }
        let output = Command::new(program)
            .args(arguments)
            .current_dir(self.root())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("AIPERF_EVAL_ROOT", self.root())
            .envs(environment.iter().map(|(key, value)| (key, value)))
            .output()
            .map_err(|_| EvalExecutionError::ProcessSpawn(program.clone()))?;
        if !output.status.success() {
            return Err(EvalExecutionError::ProcessFailure(program.clone()));
        }
        Ok(ProcessOutput {
            status: output.status,
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

/// Captured output from a successful local sandbox process.
#[derive(Debug)]
pub struct ProcessOutput {
    /// Terminal process status.
    pub status: ExitStatus,
    /// Captured standard output bytes.
    pub stdout: Vec<u8>,
    /// Captured standard error bytes.
    pub stderr: Vec<u8>,
}
