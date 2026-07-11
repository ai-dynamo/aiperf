// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy benchmark extension seam.

use crate::{AccuracyError, BenchmarkProblem, DatasetSource};

/// Shared benchmark selection and prompt settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConfig {
    /// Task/category filters. Empty means all tasks.
    pub tasks: Vec<String>,
    /// Number of validation examples included in each prompt.
    pub n_shots: usize,
    /// Whether to request chain-of-thought reasoning.
    pub enable_cot: bool,
    /// Optional cap applied after deterministic dataset ordering.
    pub max_problems: Option<usize>,
    /// Optional generation-token override.
    pub max_tokens: Option<usize>,
}

/// A benchmark that turns a typed dataset source into transport-neutral problems.
///
/// New benchmarks implement this trait; runtime dispatch and accumulation do not
/// branch on benchmark names.
pub trait AccuracyBenchmark {
    /// Stable benchmark name.
    fn name(&self) -> &'static str;
    /// Validate benchmark-specific selectors before dataset acquisition.
    ///
    /// Implementations should reject invalid task names and unsupported knobs
    /// here when that decision does not depend on dataset rows. The default
    /// keeps data-defined benchmark plugins possible without adding no-op
    /// methods to every implementation.
    fn validate_config(&self, _config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        Ok(())
    }
    /// Materialize scored problems from `source` under `config`.
    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError>;
}
