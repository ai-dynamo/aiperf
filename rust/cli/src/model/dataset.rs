// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `datasets` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_authored_dataset_v2`
//! and `_distribution`. `cfg.datasets` is a one-element list on the single-run
//! path. Synthetic is fully modeled here; `file`/`public` variants are added as
//! those paths are exercised. Optional fields use `exclude_none` semantics
//! (`_authored_model_dump`), i.e. absent when unset.

use serde::{Deserialize, Serialize};

/// Dataset sampling order. Open-ish (varies by dataset kind), kept as a
/// transparent newtype rather than guessing a closed variant set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sampling(pub String);

/// A numeric distribution (`_distribution`): either a scalar `{value}` or a
/// parametric spec (`{mean,stddev,...}` / `{peaks:[...]}`). All fields optional
/// so any one shape round-trips; only the present ones serialize.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    /// Scalar value (mutually exclusive with the parametric fields).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Mean of a normal/uniform distribution.
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

/// Synthetic prompt-generation policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompts {
    /// Prompts per request.
    pub batch_size: u32,
    /// Input sequence length distribution.
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
    /// Mixed ISL/OSL sequence distribution (`--seq-dist`; present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_distribution: Option<Vec<SeqDistEntry>>,
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

/// The typed synthetic dataset body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synthetic {
    /// Prompt-generation policy.
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
    /// Sampling order.
    pub sampling: Sampling,
    /// Turns-per-session distribution (multi-turn; present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<Distribution>,
    /// Per-turn delay as a ratio of think time.
    pub turn_delay_ratio: f64,
    /// Number of dataset entries (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<u32>,
    /// Number of conversation sessions (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_conversations: Option<u32>,
    /// Per-turn fixed delay distribution, milliseconds (present when set;
    /// renamed from `turn_delay` by the projection).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_delay_ms: Option<Distribution>,
}

/// A file-backed dataset (trace/replay). Ported from `_authored_dataset_v2`'s
/// `FileDataset` branch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileDataset {
    /// Native file format id (e.g. `single_turn`, `mooncake_trace`).
    pub format: String,
    /// Sampling order.
    pub sampling: Sampling,
    /// Format-specific loader options (open bag).
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
    /// Inline dataset records (present instead of `path` for an in-YAML dataset;
    /// passed through verbatim, matching Python's `_authored_dataset_v2` which
    /// emits `records` when `path` is absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records: Option<serde_json::Value>,
}

/// A named public dataset expanded to explicit source coordinates. Ported from
/// `_authored_dataset_v2`'s `PublicDataset` branch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicDataset {
    /// Catalog name (e.g. `sharegpt`).
    pub name: String,
    /// Native loader format id.
    pub format: String,
    /// Source coordinates (HuggingFace or URL; open bag).
    pub source: serde_json::Value,
    /// Loader options (columns/multi_turn/max_conversations).
    pub options: serde_json::Map<String, serde_json::Value>,
    /// Sampling order.
    pub sampling: Sampling,
    /// Number of dataset entries (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<u32>,
    /// Deterministic sampling seed (present when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
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
                sequence_distribution: None,
            },
            prefix_prompts: None,
            images: None,
            audio: None,
            video: None,
            sampling: Sampling("sequential".into()),
            turns: None,
            turn_delay_ratio: 1.0,
            entries: Some(1),
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
    }
}
