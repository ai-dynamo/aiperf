// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned JSONL protocol shared with the canonical Python evaluator worker.
//!
//! Problem IDs are opaque to Rust. Prompts and generation controls cross into
//! the normal AIPerf inference path; ground truth and hidden tests do not.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

/// Current evaluator protocol version.
pub const EVALUATOR_PROTOCOL_VERSION: u32 = 1;

/// Opaque identifier assigned by the evaluator to one benchmark problem.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ProblemId(String);

impl ProblemId {
    /// Build a validated opaque problem ID.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.trim().is_empty() {
            Err("evaluator problem_id must not be empty".to_string())
        } else {
            Ok(Self(value))
        }
    }

    /// Borrow the wire value without interpreting it.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ProblemId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

/// Exact evaluator environment reported by the initialization handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorIdentity {
    /// Negotiated protocol version.
    pub protocol: u32,
    /// Version of the small AIPerf worker adapter.
    pub worker_version: String,
    /// Python runtime version.
    pub python_version: String,
    /// Python executable used by the worker.
    pub python_executable: String,
    /// Evaluator package versions, including absent optional packages as null.
    pub packages: BTreeMap<String, Option<String>>,
    /// SHA-256 of the worker source.
    pub worker_source_sha256: String,
    /// SHA-256 of the fully pinned evaluator dependency lock, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency_lock_sha256: Option<String>,
    /// Optional immutable evaluator container digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_digest: Option<String>,
    /// Supported worker operations.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Canonical benchmark configuration sent to the evaluator.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorLoadConfig {
    /// Optional benchmark tasks/categories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tasks: Option<Vec<String>>,
    /// Optional few-shot count; null selects the evaluator's canonical default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_shots: Option<usize>,
    /// Optional chain-of-thought selection; null selects the canonical default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_cot: Option<bool>,
    /// Optional explicit system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Optional problem cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_problems: Option<usize>,
    /// Optional generation-token override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Reproducibility seed for evaluator-owned sampling.
    #[serde(default)]
    pub seed: u64,
}

/// Dataset/task identity frozen by a successful load operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorDatasetIdentity {
    /// Dataset preparation implementation.
    pub provider: String,
    /// Optional AIPerf benchmark name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<String>,
    /// Optional Hugging Face repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Optional dataset subset/configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subset: Option<String>,
    /// Immutable dataset revision when the evaluator exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Evaluation splits selected by the task.
    #[serde(default)]
    pub evaluation_splits: Vec<String>,
    /// Optional canonical task version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_version: Option<u64>,
}

/// Result of loading one benchmark into the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorLoadResult {
    /// Canonical benchmark name.
    pub benchmark: String,
    /// Number of prepared problems.
    pub problem_count: usize,
    /// Resolved dataset identity.
    pub dataset: EvaluatorDatasetIdentity,
    /// Canonical grader/metric implementation name.
    pub grader: String,
}

/// One OpenAI-compatible message prepared by the evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorMessage {
    /// Message role.
    pub role: String,
    /// Text or structured OpenAI-compatible message content.
    pub content: Value,
}

/// Generation controls authored by the canonical task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorGenerationConfig {
    /// Maximum output tokens.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f64,
    /// Nucleus-sampling probability.
    pub top_p: f64,
    /// Stop strings.
    #[serde(default)]
    pub stop: Vec<String>,
}

/// One evaluator-prepared problem safe to send through the Rust inference path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorProblem {
    /// Opaque worker-owned problem identifier.
    pub problem_id: ProblemId,
    /// Reporting task/category without ground-truth material.
    pub task: String,
    /// Plain-text prompt used for token accounting/completions endpoints.
    pub prompt: String,
    /// Canonical chat messages used for chat endpoints.
    pub messages: Vec<EvaluatorMessage>,
    /// Canonical generation settings.
    pub generation: EvaluatorGenerationConfig,
}

/// One page of evaluator-prepared problems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorProblemPage {
    /// Problems in canonical worker order.
    pub items: Vec<EvaluatorProblem>,
    /// Offset for the next request.
    pub next_offset: usize,
    /// Whether all prepared problems have been returned.
    pub done: bool,
}

/// Terminal Rust inference response submitted for canonical grading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorGradeItem {
    /// Opaque problem identifier.
    pub problem_id: ProblemId,
    /// Parsed model response text captured at terminal, possibly partial or empty.
    pub response: String,
}

/// Canonical evaluator result for one response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorGrade {
    /// Opaque problem identifier.
    pub problem_id: ProblemId,
    /// Reporting task/category.
    pub task: String,
    /// Canonical correctness decision.
    pub correct: bool,
    /// Whether canonical parsing failed.
    pub unparsed: bool,
    /// Canonical score/confidence in the closed interval `[0, 1]`.
    pub confidence: f64,
    /// Canonical evaluator explanation.
    pub reasoning: String,
    /// Optional extracted answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_answer: Option<String>,
}

/// Batch of canonical grade results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorGradeBatch {
    /// Results in submitted-item order.
    pub items: Vec<EvaluatorGrade>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum WorkerRequest<'a> {
    Hello {
        id: u64,
        protocol: u32,
    },
    Load {
        id: u64,
        benchmark: &'a str,
        config: &'a EvaluatorLoadConfig,
        #[serde(skip_serializing_if = "Option::is_none")]
        grader: Option<&'a str>,
    },
    NextProblems {
        id: u64,
        offset: usize,
        limit: usize,
    },
    GradeBatch {
        id: u64,
        items: &'a [EvaluatorGradeItem],
    },
    Shutdown {
        id: u64,
    },
}

impl WorkerRequest<'_> {
    pub(crate) fn id(&self) -> u64 {
        match self {
            Self::Hello { id, .. }
            | Self::Load { id, .. }
            | Self::NextProblems { id, .. }
            | Self::GradeBatch { id, .. }
            | Self::Shutdown { id } => *id,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerResponse {
    pub(crate) id: Option<u64>,
    pub(crate) ok: bool,
    #[serde(default)]
    pub(crate) result: Option<Value>,
    #[serde(default)]
    pub(crate) error: Option<WorkerRemoteError>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerRemoteError {
    pub(crate) kind: String,
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) retryable: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ShutdownResult {
    pub(crate) shutdown: bool,
}
