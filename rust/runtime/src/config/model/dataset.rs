// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed dataset configuration.
//!
//! Optional fields are absent when unset.

use serde::{Deserialize, Serialize};

/// The origin format of a recorded-agent graph input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedAgentSourceFormat {
    /// Preserve existing recording detection and sniff a single JSONL input.
    #[default]
    Auto,
    /// Require the Mini-SWE-Agent recording or replay-manifest contract.
    MiniSweAgent,
    /// Require a Codex CLI session export.
    Codex,
    /// Require a Claude Code session export.
    ClaudeCode,
}

/// An invalid recorded-agent source-format spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordedAgentSourceFormatParseError;

impl std::fmt::Display for RecordedAgentSourceFormatParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "recorded-agent source format must be one of: auto, mini_swe_agent, \
             mini-swe-agent, codex, claude_code, claude-code",
        )
    }
}

impl std::error::Error for RecordedAgentSourceFormatParseError {}

impl std::fmt::Display for RecordedAgentSourceFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::MiniSweAgent => "mini_swe_agent",
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
        })
    }
}

impl std::str::FromStr for RecordedAgentSourceFormat {
    type Err = RecordedAgentSourceFormatParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "mini_swe_agent" | "mini-swe-agent" => Ok(Self::MiniSweAgent),
            "codex" => Ok(Self::Codex),
            "claude_code" | "claude-code" => Ok(Self::ClaudeCode),
            _ => Err(RecordedAgentSourceFormatParseError),
        }
    }
}

/// Dataset sampling order, which is extensible by dataset kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sampling(pub String);

/// Where (and how) to inject a per-conversation cache-bust marker.
///
/// Mirrors the Python `CacheBustTarget` enum. `Prefix` variants diverge at
/// token 0 of the prompt (most aggressive — defeats KV-cache prefix matching
/// for the entire prompt); `Suffix` variants append after existing content
/// (preserves leading-prefix caching). `None` disables the feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBustTarget {
    /// Cache-busting disabled (default).
    #[default]
    None,
    /// Prepend the marker at token 0 of the system message.
    SystemPrefix,
    /// Append the marker after the system message content.
    SystemSuffix,
    /// Prepend the marker at token 0 of the first turn.
    FirstTurnPrefix,
    /// Append the marker after the first turn content.
    FirstTurnSuffix,
    /// Constant marker during warmup at the system message.
    WarmupIsolationSystem,
    /// Constant marker during warmup at the first user turn.
    WarmupIsolationFirstTurn,
}

/// Per-conversation cache-bust marker injection policy.
///
/// Mirrors the Python `CacheBustConfig`: a single `target` selects where the
/// deterministic marker is injected. Absent from the wire when unset, so a
/// config without cache-busting serializes byte-identically to before this
/// field existed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBust {
    /// Where (and how) to inject the marker.
    #[serde(default)]
    pub target: CacheBustTarget,
}

/// A scalar or parametric numeric distribution.
///
/// Only present fields serialize.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    /// Scalar value (mutually exclusive with the parametric fields).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Mean of a normal or uniform distribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
    /// Standard deviation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stddev: Option<f64>,
    /// Median (selects a log-normal distribution when paired with `mean`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    /// Lower bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Upper bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Multi-peak mixture components.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peaks: Option<Vec<Peak>>,
}

/// One weighted peak of a multi-modal distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Peak {
    /// The peak's own distribution.
    pub distribution: Distribution,
    /// Mixture weight.
    pub weight: f64,
}

/// One prompt per request, matching `load.rs`/`yaml.rs`/`workload_kind.rs`.
fn default_batch_size() -> u32 {
    1
}

/// Full think-time delay, matching `load.rs`/`yaml.rs`/`workload_kind.rs`.
fn default_turn_delay_ratio() -> f64 {
    1.0
}

/// The default synthetic ISL distribution, matching `load.rs::default_isl`.
fn default_isl() -> Distribution {
    Distribution {
        mean: Some(550.0),
        stddev: Some(0.0),
        ..Default::default()
    }
}

/// Sequential order, matching `load.rs`/`yaml.rs`.
fn default_sampling() -> Sampling {
    Sampling("sequential".into())
}

/// An entirely unauthored prompts block, matching `yaml.rs::extract_prompts`.
fn default_prompts() -> Prompts {
    Prompts {
        batch_size: default_batch_size(),
        isl: default_isl(),
        osl: None,
        num_prefix_prompts: None,
        prefix_prompt_length: None,
        block_size: None,
        corpus: None,
        sequence_distribution: None,
        random_range_ratio: None,
        random_corpus_style: Default::default(),
        prefix_reuse_fraction: None,
        prefix_reuse_ratio: None,
        cache_bust: None,
    }
}

/// Synthetic prompt-generation policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompts {
    /// Prompts per request. Every producer (`load.rs`, `yaml.rs`,
    /// `workload_kind.rs`) already defaults this to one, so omission in an
    /// authored protocol-v2 request must resolve the same way rather than
    /// hard-rejecting the run.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Input sequence length distribution. `yaml.rs` defaults this on its own
    /// when a prompts block authors everything but the ISL.
    #[serde(default = "default_isl")]
    pub isl: Distribution,
    /// Output sequence length distribution (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl: Option<Distribution>,
    /// Shared-prefix prompt count (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_prefix_prompts: Option<u32>,
    /// Shared-prefix length (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_prompt_length: Option<u32>,
    /// Input-token block size (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<u32>,
    /// Prompt corpus selector (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<String>,
    /// Mixed ISL/OSL sequence distribution (`--seq-dist`; present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_distribution: Option<Vec<SeqDistEntry>>,
    /// Uniform random ISL/OSL window ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_range_ratio: Option<crate::dataset::RandomRangeRatioInput>,
    /// Reference random-corpus behavior.
    #[serde(default)]
    pub random_corpus_style: crate::dataset::RandomCorpusStyle,
    /// Fraction of prompts reusing a shared leading prefix (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_reuse_fraction: Option<f64>,
    /// Shared-prefix fraction of each reusing prompt's length (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_reuse_ratio: Option<f64>,
    /// Per-conversation cache-bust marker policy (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_bust: Option<CacheBust>,
}

/// Shared prompt-source selection for non-synthetic dataset kinds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptSelection {
    /// Prompt corpus selector (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<String>,
}

/// One weighted `(isl, osl)` pair of a mixed sequence distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeqDistEntry {
    /// Input sequence length distribution.
    pub isl: Distribution,
    /// Output sequence length distribution.
    pub osl: Distribution,
    /// Mixture probability (percentage, 0–100).
    pub probability: f64,
}

/// Synthetic image-generation policy (`synthetic.images`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageSpec {
    /// Images per request.
    pub batch_size: u32,
    /// Image format (`png`/`jpeg`/…).
    pub format: String,
    /// Height distribution, pixels.
    pub height: Distribution,
    /// Width distribution, pixels.
    pub width: Distribution,
    /// Image source (`noise`/…).
    pub source: String,
    /// Source sampling strategy.
    pub source_sampling: String,
}

/// Synthetic audio-generation policy (`synthetic.audio`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioSpec {
    /// Audio clips per request.
    pub batch_size: u32,
    /// Channel count.
    pub channels: u32,
    /// Bit depths.
    pub depths: Vec<u32>,
    /// Audio format (`wav`/…).
    pub format: String,
    /// Clip length distribution, seconds.
    pub length: Distribution,
    /// Sample rates, kHz (the wire value is Hz/1000).
    pub sample_rates: Vec<f64>,
}

/// The audio track of a synthetic video (`synthetic.video.audio`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoAudio {
    /// Channel count.
    pub channels: u32,
    /// Audio codec (`aac`/`libvorbis`/`libopus`; present only when set + audio enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    /// Bit depth.
    pub depth: u32,
    /// Sample rate, kHz.
    pub sample_rate: f64,
}

/// Synthetic video-generation policy (`synthetic.video`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoSpec {
    /// Audio track.
    pub audio: VideoAudio,
    /// Clips per request.
    pub batch_size: u32,
    /// Video codec.
    pub codec: String,
    /// Duration, seconds.
    pub duration: f64,
    /// Container format (`webm`/`mp4`/…).
    pub format: String,
    /// Frames per second.
    pub fps: u32,
    /// Synthesis pattern.
    pub synth_type: String,
    /// Width, pixels (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Height, pixels (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// Shared-prefix / prefix-pool policy (`synthetic.prefix_prompts`). Two mutually
/// exclusive modes: shared-system+user-context, or a length+pool_size pool.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PrefixPrompts {
    /// Shared system prompt length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_system_length: Option<u32>,
    /// Per-user context prompt length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_context_length: Option<u32>,
    /// Pool prefix length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<u32>,
    /// Prefix pool size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool_size: Option<u32>,
}

/// Rankings query-passage generation policy.
///
/// All fields are always serialized. Default values are 10 passages, 128
/// tokens per passage, and 32 query tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rankings {
    /// Number of passages per ranking request.
    pub passages: Distribution,
    /// Token length per passage.
    pub passage_tokens: Distribution,
    /// Token length of the query.
    pub query_tokens: Distribution,
}

/// The typed synthetic dataset body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synthetic {
    /// Prompt-generation policy. `load.rs` and `yaml.rs` both synthesize a
    /// default block from `default_isl` when none is authored, so an authored
    /// protocol-v2 request omitting it must resolve the same way rather than
    /// hard-rejecting the run.
    #[serde(default = "default_prompts")]
    pub prompts: Prompts,
    /// Shared-prefix / prefix-pool policy (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_prompts: Option<PrefixPrompts>,
    /// Synthetic image generation (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<ImageSpec>,
    /// Synthetic audio generation (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioSpec>,
    /// Synthetic video generation (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoSpec>,
    /// Rankings/rerank query-passage generation (present when a rankings flag is set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rankings: Option<Rankings>,
    /// Sampling order.
    #[serde(default = "default_sampling")]
    pub sampling: Sampling,
    /// Turns-per-session distribution (multi-turn; present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<Distribution>,
    /// Per-turn delay as a ratio of think time. See [`Prompts::batch_size`] for
    /// why omission defaults rather than rejects.
    #[serde(default = "default_turn_delay_ratio")]
    pub turn_delay_ratio: f64,
    /// Number of dataset entries (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<u32>,
    /// Deterministic generation seed (present when set). Without it the synthetic
    /// loader falls back to the run seed, so an authored per-dataset seed that is
    /// not projected here runs unseeded — random session ids and prompts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Number of conversation sessions (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_conversations: Option<u32>,
    /// Per-turn fixed delay distribution in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_delay_ms: Option<Distribution>,
}

/// Recorded-agent replay configuration attached to an `agent_recording` file dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedAgentGraphConfig {
    /// Source format used to decode the recorded session.
    #[serde(default)]
    pub source_format: RecordedAgentSourceFormat,
    /// Whether Claude Code subagent sessions participate in the replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_subagents: Option<bool>,
    /// Root directory resolving manifest-relative recordings and task assets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_root: Option<std::path::PathBuf>,
    /// Execute recorded tool commands instead of retaining their recorded gaps.
    #[serde(default)]
    pub execute_tools: bool,
    /// Fallback image for a low-level tool recipe without an adapter image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_image: Option<String>,
    /// PinchBench environment image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinch_image: Option<String>,
    /// Default per-command wall-clock deadline in seconds.
    #[serde(default = "default_command_timeout_seconds")]
    pub command_timeout_seconds: f64,
    /// Docker stop/removal deadline in seconds.
    #[serde(default = "default_container_stop_timeout_seconds")]
    pub container_stop_timeout_seconds: f64,
    /// Session-shell graceful shutdown deadline in seconds.
    #[serde(default = "default_session_close_grace_seconds")]
    pub session_close_grace_seconds: f64,
    /// Select recorded task-family request profiles.
    #[serde(default = "default_use_family_sampling")]
    pub use_family_sampling: bool,
    /// Emit one excluded warmup plan before each replay trace.
    #[serde(default)]
    pub emit_warmup: bool,
    /// Resume an interrupted manifest run.
    #[serde(default)]
    pub resume: bool,
    /// Stop after the first failed replay task.
    #[serde(default)]
    pub stop_on_failure: bool,
}

fn default_command_timeout_seconds() -> f64 {
    900.0
}

fn default_container_stop_timeout_seconds() -> f64 {
    5.0
}

fn default_session_close_grace_seconds() -> f64 {
    1.0
}

fn default_use_family_sampling() -> bool {
    true
}

/// A file-backed trace or replay dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileDataset {
    /// Native file format id, omitted when the runtime should auto-detect it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Sampling order.
    #[serde(default = "default_sampling")]
    pub sampling: Sampling,
    /// Format-specific loader options (open bag).
    #[serde(default)]
    pub options: serde_json::Map<String, serde_json::Value>,
    /// Absolute path to the dataset file/directory (present when path-backed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Number of dataset entries (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<u32>,
    /// Deterministic sampling seed (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Output sequence length distribution (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl: Option<Distribution>,
    /// Shared prompt-source selection (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptSelection>,
    /// Inline records, passed through verbatim when `path` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records: Option<serde_json::Value>,
    /// Recorded-graph synthesis block set by `--synthesis-*` flags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<serde_json::Value>,
    /// Recorded-agent replay settings, present only for `agent_recording` inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<RecordedAgentGraphConfig>,
    /// Per-conversation cache-bust marker policy (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_bust: Option<CacheBust>,
    /// Fetch remote image URLs at generation time and inline them as data URLs
    /// (`--prefetch-media-urls`). Omitted when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prefetch_media_urls: bool,
}

/// A named public dataset with explicit source coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDataset {
    /// Catalog name (e.g. `sharegpt`).
    pub name: String,
    /// Native loader format id.
    pub format: String,
    /// Source coordinates (HuggingFace or URL; open bag).
    pub source: serde_json::Value,
    /// Loader options (columns/multi_turn/max_conversations).
    #[serde(default)]
    pub options: serde_json::Map<String, serde_json::Value>,
    /// Sampling order.
    #[serde(default = "default_sampling")]
    pub sampling: Sampling,
    /// Number of dataset entries (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<u32>,
    /// Deterministic sampling seed (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Shared prompt-source selection (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptSelection>,
    /// Recorded-graph synthesis block set by `--synthesis-*` flags or a scenario
    /// lock (t* window, idle-gap cap, cache-bust target). Threaded through so the
    /// public recorded-graph path applies the same trajectory-start snapshot the
    /// file path does; absent for non-recorded public datasets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<serde_json::Value>,
    /// Per-conversation cache-bust marker policy (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_bust: Option<CacheBust>,
    /// Fetch remote image URLs at generation time and inline them as data URLs
    /// (`--prefetch-media-urls`). Omitted when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prefetch_media_urls: bool,
}

/// One typed dataset (discriminated by `type`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
// A single run builds exactly one dataset; the size gap between variants is not
// worth a `Box` indirection (which would also complicate the serde tag round-trip).
#[allow(clippy::large_enum_variant)]
pub enum Dataset {
    /// Synthetically generated prompts.
    Synthetic(Synthetic),
    /// File-backed trace/replay dataset.
    File(FileDataset),
    /// Named public dataset.
    Public(PublicDataset),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_tag_and_scalar_dist() {
        let d = Dataset::Synthetic(Synthetic {
            prompts: Prompts {
                batch_size: 1,
                isl: Distribution {
                    mean: Some(550.0),
                    stddev: Some(0.0),
                    ..Default::default()
                },
                osl: None,
                num_prefix_prompts: None,
                prefix_prompt_length: None,
                block_size: None,
                corpus: Some("coding".into()),
                sequence_distribution: None,
                random_range_ratio: None,
                random_corpus_style: Default::default(),
                prefix_reuse_fraction: None,
                prefix_reuse_ratio: None,
                cache_bust: None,
            },
            prefix_prompts: None,
            images: None,
            audio: None,
            video: None,
            rankings: None,
            sampling: Sampling("sequential".into()),
            turns: None,
            turn_delay_ratio: 1.0,
            entries: Some(1),
            random_seed: None,
            num_conversations: None,
            turn_delay_ms: None,
        });
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["type"], serde_json::json!("synthetic"));
        assert_eq!(
            v["prompts"]["isl"],
            serde_json::json!({"mean":550.0,"stddev":0.0})
        );
        assert_eq!(v["prompts"]["isl"].get("value"), None);
        assert_eq!(v["prompts"]["corpus"], serde_json::json!("coding"));
    }
}
