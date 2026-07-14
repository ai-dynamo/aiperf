// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static-accuracy evaluator worker seam.
//!
//! Rust deliberately contains no benchmark prompt builders, answer extractors,
//! hidden-test decoders, code runners, or graders. The [`AccuracyEvaluator`]
//! protocol keeps benchmark semantics in an isolated Python worker
//! (lighteval/deepeval) while Rust owns request dispatch, measurement, and
//! reporting. The external-evaluator provider-host and agentic verticals have
//! been removed; only the static lighteval worker path remains.

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
