// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static-accuracy evaluator worker seam.
//!
//! The [`AccuracyEvaluator`] protocol isolates benchmark semantics in a Python
//! worker while Rust owns request dispatch, measurement, and reporting.

pub mod protocol;
pub mod worker;

pub use protocol::{
    EVALUATOR_PROTOCOL_VERSION, EvaluatorDatasetIdentity, EvaluatorGenerationConfig,
    EvaluatorGrade, EvaluatorGradeBatch, EvaluatorGradeItem, EvaluatorIdentity,
    EvaluatorLoadConfig, EvaluatorLoadResult, EvaluatorMessage, EvaluatorProblem,
    EvaluatorProblemPage, ProblemId,
};
pub use worker::{
    AccuracyEvaluator, EvaluatorLogSink, EvaluatorWorkerError, PythonEvaluator,
    StderrEvaluatorLogSink, WorkerProcessConfig,
};
