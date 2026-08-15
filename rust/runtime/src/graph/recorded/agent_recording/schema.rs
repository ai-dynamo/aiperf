// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serde DTOs for recorded-agent replay input.

use serde::Deserialize;
use serde_json::{Map, Value};

/// Strict replay-manifest defaults shared by every selected task.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayDefaults {
    /// Reference runner configuration profile.
    pub config: String,
    /// Maximum agent steps.
    pub step_limit: u64,
    /// Maximum agent cost.
    pub cost_limit: f64,
    /// Task environment class.
    pub environment_class: String,
    /// Docker network policy.
    pub docker_network: String,
    /// Per-inference timeout in seconds.
    pub per_inference_timeout: f64,
    /// Fallback generation cap for recordings without completion usage.
    pub fallback_max_output_tokens: u64,
    /// Resolved sampling temperature.
    pub temperature: f64,
    /// Resolved nucleus sampling probability.
    pub top_p: f64,
    /// Resolved top-k sampling cap.
    pub top_k: u64,
    /// Resolved min-p sampling probability.
    pub min_p: f64,
    /// Whether reference stream timing is enabled.
    pub stream_for_timing: bool,
    /// Whether raw OpenAI stream timing is enabled.
    pub raw_openai_stream_for_replay_timing: bool,
    /// Whether recorded generation caps are replayed.
    pub replay_max_tokens_from_recording: bool,
    /// Extra generation-cap margin.
    pub replay_max_tokens_margin: u64,
    /// Extra request-body fields.
    pub extra_request_body: Map<String, Value>,
    /// Whether profiling requests isolate cross-run cache state.
    pub cross_run_cache_isolation: bool,
    /// Whether a trace-local warmup is enabled.
    pub warmup: bool,
    /// Reference metric boundary.
    pub measurement_scope: String,
}

/// Source-corpus totals pinned by a replay manifest.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedCorpusShape {
    /// Sum of source model-call input tokens.
    pub total_isl: u64,
    /// Sum of each trace's initial and positive input-token growth.
    pub isl_delta: u64,
    /// Largest source model-call input length.
    pub peak_isl: u64,
    /// Sum of source model-call output tokens.
    pub total_osl: u64,
    /// Successful source model-call count.
    pub model_calls: u64,
    /// Completed source tool-command count.
    pub tool_calls: u64,
    /// Sum of completed source tool-command durations in milliseconds.
    pub tool_duration_ms: f64,
    /// Largest completed source tool-command duration in milliseconds.
    pub max_tool_call_duration_ms: f64,
    /// Source tool-command timeout count.
    pub timed_out_tool_calls: u64,
}

/// Stable task identity retained through replay lowering and reporting.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ReplayTaskIdentity {
    /// Supported source adapter.
    pub adapter: String,
    /// Task-family identifier.
    pub family: String,
    /// Upstream task identifier.
    pub task_id: String,
    /// Optional descriptive workload role.
    #[serde(default)]
    pub primary_role: Option<String>,
}

/// Strictly modeled task entry in a replay manifest.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestTask {
    #[serde(flatten)]
    pub identity: ReplayTaskIdentity,
    pub recording: String,
}

/// Recorded-agent manifest with preserved descriptive provenance.
#[derive(Clone, Debug, Deserialize)]
pub struct RecordedAgentReplayManifest {
    /// Human-readable workload name.
    pub name: String,
    /// Required replay mode.
    pub mode: String,
    /// Defaults used by every recorded task.
    pub defaults: ReplayDefaults,
    /// Expected source-corpus totals.
    pub aggregate: ExpectedCorpusShape,
    /// Ordered replay tasks.
    #[serde(rename = "tasks")]
    pub tasks: Vec<ManifestTask>,
    /// Attribution retained verbatim for run provenance.
    pub attribution: Map<String, Value>,
    /// Unknown descriptive manifest fields retained for provenance.
    #[serde(flatten)]
    pub provenance: Map<String, Value>,
}

/// Recording metadata relevant to replay identity and environment selection.
#[derive(Clone, Debug, Deserialize, Default)]
pub struct RecordedAgentMetadata {
    /// Preferred upstream task identity.
    #[serde(default)]
    pub instance_id: Option<String>,
    /// Fallback upstream task identity.
    #[serde(default)]
    pub task_id: Option<String>,
    /// Source benchmark name.
    #[serde(default)]
    pub benchmark: Option<String>,
    /// Recorded model name.
    #[serde(default)]
    pub model_name: Option<String>,
    /// Preferred SWE image.
    #[serde(default)]
    pub docker_image: Option<String>,
    /// Source manifest identifier.
    #[serde(default)]
    pub manifest: Option<String>,
    /// Upstream instance metadata retained without reshaping.
    #[serde(default)]
    pub instance: Option<Value>,
}

/// Replay-relevant envelope for one Mini-SWE-Agent recording.
#[derive(Clone, Debug, Deserialize)]
pub struct RecordedAgentRecording {
    /// Versioned source format marker.
    pub format: String,
    /// Source task metadata.
    #[serde(default)]
    pub metadata: RecordedAgentMetadata,
    /// Ordered source events.
    pub events: Vec<RecordedAgentEvent>,
}

/// One source event.
#[derive(Clone, Debug, Deserialize)]
pub struct RecordedAgentEvent {
    /// Source event identifier.
    pub id: u64,
    /// Source event type.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Source event end time in seconds.
    pub timestamp: f64,
    /// Optional source duration in integer nanoseconds.
    #[serde(default)]
    pub duration_ns: Option<u64>,
    /// Optional source step.
    #[serde(default)]
    pub step: Option<u64>,
    /// Recorded request used by a model-call event.
    #[serde(default)]
    pub provider_request: Option<RecordedProviderRequest>,
    /// Recorded response envelope used for source token usage.
    #[serde(default)]
    pub response_message: Option<Value>,
    /// Recorded tool action.
    #[serde(default)]
    pub action: Option<Value>,
    /// Recorded terminal error.
    #[serde(default)]
    pub error: Option<Value>,
}

/// Replay-safe recorded model request fields.
#[derive(Clone, Debug, Deserialize)]
pub struct RecordedProviderRequest {
    /// Exact source message array.
    #[serde(default)]
    pub messages: Option<Vec<Value>>,
    /// Exact source tool array.
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    /// Optional recorded model override.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional recorded sampling temperature.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Optional recorded sampling top-p.
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Optional recorded generation cap.
    #[serde(default)]
    pub max_tokens: Option<u64>,
}
