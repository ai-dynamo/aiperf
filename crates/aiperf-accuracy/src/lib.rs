// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust control-plane adapter for the canonical Python/Lighteval evaluator.
//!
//! Rust deliberately contains no benchmark prompt builders, answer extractors,
//! hidden-test decoders, code runners, or graders. [`AccuracyEvaluator`] exposes
//! a narrow versioned stdio seam while the application remains responsible for
//! scheduling and all inference-server communication.

pub mod protocol;
pub mod worker;

pub use protocol::{
    AgenticEpisode, AgenticEpisodeOutcome, AgenticEpisodePage, AgenticEpisodeResult,
    AgenticEvaluatorEvent, AgenticEvaluatorIdentity, AgenticEvaluatorLoadConfig, AgenticEventBatch,
    AgenticInferenceStatus, AgenticModelCall, AgenticModelResult, AgenticResultBatch,
    EVALUATOR_PROTOCOL_VERSION, EpisodeId, EvaluatorDatasetIdentity, EvaluatorGenerationConfig,
    EvaluatorGrade, EvaluatorGradeBatch, EvaluatorGradeItem, EvaluatorIdentity,
    EvaluatorLoadConfig, EvaluatorLoadResult, EvaluatorMessage, EvaluatorProblem,
    EvaluatorProblemPage, ModelCallId, ProblemId,
};
pub use worker::{
    AccuracyEvaluator, AgenticEvaluator, EvaluatorLogSink, EvaluatorWorkerError, PythonEvaluator,
    StderrEvaluatorLogSink, WorkerProcessConfig,
};
