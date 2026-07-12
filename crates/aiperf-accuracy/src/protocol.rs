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

/// Opaque identifier assigned by the evaluator to one stateful task episode.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct EpisodeId(String);

impl EpisodeId {
    /// Build a validated opaque episode ID.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.trim().is_empty() {
            Err("evaluator episode_id must not be empty".to_string())
        } else {
            Ok(Self(value))
        }
    }

    /// Borrow the wire value without interpreting it.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for EpisodeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

/// Opaque identifier assigned by the evaluator to one requested model call.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ModelCallId(String);

impl ModelCallId {
    /// Build a validated opaque model-call ID.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.trim().is_empty() {
            Err("evaluator call_id must not be empty".to_string())
        } else {
            Ok(Self(value))
        }
    }

    /// Borrow the wire value without interpreting it.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ModelCallId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

/// Configuration for a canonical stateful agent-harness run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEvaluatorLoadConfig {
    /// Optional exact task names/globs selected from the dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_names: Option<Vec<String>>,
    /// Optional deterministic cap applied after task filtering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_episodes: Option<usize>,
    /// Maximum evaluator environments active at once.
    pub task_concurrency: usize,
    /// Harbor environment provider, such as `docker` or `daytona`.
    pub environment: String,
    /// Directory for canonical harness artifacts and trajectories.
    pub output_dir: String,
    /// Optional Terminus-2 model-call limit per episode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    /// Maximum generated tokens for each Rust inference call.
    pub max_tokens: usize,
    /// Explicit model context-window size used by the agent scaffold.
    pub context_window: usize,
    /// Canonical Terminus command protocol (`json` or `xml`).
    pub parser: String,
    /// Whether canonical Terminus context summarization is enabled.
    pub enable_summarize: bool,
    /// Optional verifier reward selected as the primary report metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// Whether Harbor may replace a cached task package.
    #[serde(default)]
    pub overwrite: bool,
}

impl Default for AgenticEvaluatorLoadConfig {
    fn default() -> Self {
        Self {
            task_names: None,
            max_episodes: None,
            task_concurrency: 1,
            environment: "docker".to_string(),
            output_dir: "artifacts/agentic".to_string(),
            max_turns: None,
            max_tokens: 4_096,
            context_window: 131_072,
            parser: "json".to_string(),
            enable_summarize: true,
            primary_reward: None,
            overwrite: false,
        }
    }
}

/// Frozen harness, dataset, agent, environment, and verifier identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEvaluatorIdentity {
    /// Canonical harness name.
    pub harness: String,
    /// Exact harness package version.
    pub harness_version: String,
    /// SHA-256 over installed harness Python sources.
    pub harness_source_sha256: String,
    /// Immutable dataset package identity.
    pub dataset: EvaluatorDatasetIdentity,
    /// Agent scaffold name.
    pub agent: String,
    /// Exact agent adapter and inherited scaffold version.
    pub agent_version: String,
    /// Environment provider selected for task sandboxes.
    pub environment: String,
    /// Canonical verifier implementation description.
    pub verifier: String,
    /// Number of selected task episodes.
    pub episode_count: usize,
    /// Optional primary reward metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
}

/// Model-safe descriptor for one agentic task episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEpisode {
    /// Opaque evaluator-owned episode identifier.
    pub episode_id: EpisodeId,
    /// Reporting task label.
    pub task: String,
    /// Dataset/source label.
    pub source: String,
}

/// One ordered page of agentic task descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEpisodePage {
    /// Episodes in canonical evaluator order.
    pub items: Vec<AgenticEpisode>,
    /// Offset for the next page request.
    pub next_offset: usize,
    /// Whether every selected episode has been returned.
    pub done: bool,
}

/// One evaluator-authored inference call awaiting ordinary Rust dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticModelCall {
    /// Parent task episode.
    pub episode_id: EpisodeId,
    /// Opaque call correlation ID.
    pub call_id: ModelCallId,
    /// Zero-based model-call index within the episode.
    pub turn_index: usize,
    /// Flat prompt used by completion endpoints and token accounting.
    pub prompt: String,
    /// Full evaluator-authored message history.
    pub messages: Vec<EvaluatorMessage>,
    /// Canonical generation controls.
    pub generation: EvaluatorGenerationConfig,
    /// Optional OpenAI-compatible tool schemas.
    #[serde(default)]
    pub tools: Vec<Value>,
    /// Optional OpenAI-compatible tool-choice value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Optional OpenAI-compatible response-format value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

/// Rust inference terminal classification returned to the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticInferenceStatus {
    /// Normal terminal response is available.
    Completed,
    /// Transport/provider dispatch failed.
    Failed,
    /// Rust scheduling policy cancelled the request.
    Cancelled,
}

/// Terminal Rust inference data used to resume one evaluator-owned agent loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticModelResult {
    /// Parent task episode.
    pub episode_id: EpisodeId,
    /// Submitted model call.
    pub call_id: ModelCallId,
    /// Explicit terminal status.
    pub status: AgenticInferenceStatus,
    /// Parsed assistant response, including any partial text on failure.
    pub response: String,
    /// Optional parsed reasoning channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Authoritative prompt-token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Authoritative completion-token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Prompt tokens served from cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Provider response ID used by stateful endpoint dialects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Endpoint-normalized finish reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Infrastructure error category for non-completed calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Infrastructure error detail for non-completed calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Canonical terminal classification for one harness episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticEpisodeOutcome {
    /// Harness verification completed and rewards are available.
    Completed,
    /// Environment, harness, verifier, or inference infrastructure failed.
    InfrastructureError,
    /// Rust explicitly cancelled the episode.
    Cancelled,
}

/// Complete canonical result for one agentic task episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEpisodeResult {
    /// Opaque evaluator-owned episode ID.
    pub episode_id: EpisodeId,
    /// Reporting task label.
    pub task: String,
    /// Terminal episode classification.
    pub outcome: AgenticEpisodeOutcome,
    /// All finite verifier rewards keyed by canonical name.
    pub rewards: BTreeMap<String, f64>,
    /// Reward selected as the report's primary score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// End-to-end harness wall time.
    pub duration_seconds: f64,
    /// Number of Rust model calls made by the episode.
    pub model_calls: usize,
    /// Aggregate prompt tokens reported by Rust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Aggregate completion tokens reported by Rust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Aggregate cached prompt tokens reported by Rust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Infrastructure/cancellation error category.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Infrastructure/cancellation error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Harness artifact directory for this episode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
}

/// One state transition produced by the evaluator-owned harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgenticEvaluatorEvent {
    /// A model call is ready for Rust's ordinary inference pipeline.
    ModelCall {
        /// Complete model-safe call authored by the agent scaffold.
        call: AgenticModelCall,
    },
    /// One task episode reached verifier or infrastructure terminal.
    EpisodeCompleted {
        /// Canonical harness/verifier result.
        result: AgenticEpisodeResult,
    },
}

/// Batch returned by bounded agentic-event polling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEventBatch {
    /// Ready state transitions, possibly empty after a poll timeout.
    pub events: Vec<AgenticEvaluatorEvent>,
}

/// Canonical results returned after every selected episode reaches terminal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticResultBatch {
    /// Results in frozen dataset order.
    pub items: Vec<AgenticEpisodeResult>,
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
    LoadAgentic {
        id: u64,
        dataset: &'a str,
        model: &'a str,
        config: &'a AgenticEvaluatorLoadConfig,
    },
    NextEpisodes {
        id: u64,
        offset: usize,
        limit: usize,
    },
    StartEpisodes {
        id: u64,
        episode_ids: &'a [EpisodeId],
    },
    PollAgentic {
        id: u64,
        limit: usize,
        wait_ms: u64,
    },
    SubmitModelResults {
        id: u64,
        items: &'a [AgenticModelResult],
    },
    CancelEpisodes {
        id: u64,
        episode_ids: &'a [EpisodeId],
    },
    FinishAgentic {
        id: u64,
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
            | Self::LoadAgentic { id, .. }
            | Self::NextEpisodes { id, .. }
            | Self::StartEpisodes { id, .. }
            | Self::PollAgentic { id, .. }
            | Self::SubmitModelResults { id, .. }
            | Self::CancelEpisodes { id, .. }
            | Self::FinishAgentic { id }
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartedEpisodesResult {
    pub(crate) started: Vec<EpisodeId>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AcceptedModelResults {
    pub(crate) accepted: Vec<ModelCallId>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CancelledEpisodesResult {
    pub(crate) cancelled: Vec<EpisodeId>,
}
