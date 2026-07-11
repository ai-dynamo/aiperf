// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native accuracy-benchmark definitions, dataset sources, prompt builders, and graders.
//!
//! This crate is deliberately IO-light: benchmark behavior is written against
//! [`DatasetSource`], while filesystem and remote sources are concrete implementations.
//! Runtime dispatch remains behind AIPerf's transport seam in the application crate, so
//! the same benchmark and grader work with online-real, online-mock, and future offline
//! sinks.

mod benchmark;
mod benchmarks;
mod error;
mod grader;
mod mmlu_pro;
mod model;
mod registry;
mod source;

pub use benchmark::{AccuracyBenchmark, BenchmarkConfig};
pub use benchmarks::{
    AIME_SYSTEM_PROMPT, Aime24Benchmark, Aime25Benchmark, AimeBenchmark, BIGBENCH_TASKS,
    BigBenchBenchmark, GpqaDiamondBenchmark, Gsm8kBenchmark, HELLASWAG_MAX_N_SHOTS,
    HellaSwagBenchmark, LcbCodeGenerationBenchmark, MMLU_SUBJECTS, Math500Benchmark, MmluBenchmark,
};
pub use error::AccuracyError;
pub use grader::{
    BubblewrapPythonExecutor, CodeExecutionGrader, CodeExecutionOutcome, CodeExecutionRequest,
    CodeExecutor, CodeTestCase, ExactMatchGrader, ExpressionGrader, GpqaGrader, Grader,
    Gsm8kGrader, LatexGrader, MathGrader, MmluProGrader, MultipleChoiceGrader,
};
pub use mmlu_pro::{
    MMLU_PRO_CATEGORIES, MMLU_PRO_DATASET, MMLU_PRO_DEFAULT_MAX_TOKENS, MMLU_PRO_DEFAULT_N_SHOTS,
    MMLU_PRO_INITIAL_PROMPT, MmluProBenchmark, MmluProQuestion,
};
pub use model::{BenchmarkProblem, ChatMessage, DatasetSplit, GenerationConfig};
pub use registry::{
    AccuracyRegistry, BenchmarkFactory, BenchmarkMetadata, GraderFactory, RegisteredBenchmark,
};
pub use source::{DatasetSource, InMemoryDatasetSource, JsonDatasetSource};
