// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral dataset policy and direct input adapters.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::dataset::{
    Dataset, SyntheticMediaGeneratorFactory, TextTokenizer, TracePromptStoragePolicy,
};
use crate::endpoints::EndpointDescriptor;
use crate::extensions::AIPerfRegistry;
use crate::rng::RngRoot;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_json::{Map, Value, value::RawValue};

use crate::engine::execute::{
    FileDatasetBuildContext, SyntheticDatasetBuildContext, build_file_dataset,
    build_public_dataset, build_synthetic_dataset, distribution,
};
use crate::engine::protocol::ModelsSpec;

/// Resolved public dataset source and loader configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicDatasetSpec {
    /// Exact system prompt applied to every composed conversation.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Config-v2 public dataset name, retained for diagnostics.
    pub name: String,
    /// Native loader registration name.
    pub format: String,
    /// Fully resolved remote source.
    pub source: PublicDatasetSourceSpec,
    /// Conversation sampling strategy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional row/conversation cap.
    #[serde(default)]
    pub entries: Option<usize>,
    /// Dataset-local seed overriding the run seed.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Shared prompt-source selection for synthesized prompt content.
    #[serde(default)]
    pub prompts: Option<PromptSelectionSpec>,
    /// Recorded-graph synthesis block (t* trajectory-start window, idle-gap cap,
    /// cache-bust target). Threaded so the public recorded-graph path applies the
    /// same trajectory snapshot the file path does; absent for non-recorded
    /// public datasets (then the path defaults to full replay).
    #[serde(default)]
    pub synthesis: Option<TraceSynthesisSpec>,
    /// Dataset-level cache-bust policy; `synthesis.cache_bust_target` wins.
    #[serde(default)]
    pub cache_bust: Option<DatasetCacheBustSpec>,
    /// Fetch remote `http(s)://` image URLs at dataset generation and inline
    /// them as `data:` URLs, for servers that cannot resolve URLs themselves.
    #[serde(default)]
    pub prefetch_media_urls: bool,
    /// Validated loader/composer options from plugin metadata and Config v2.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// Network source for a resolved public dataset.
#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PublicDatasetSourceSpec {
    /// Generic pinned or authored URL.
    Url {
        /// JSON/JSONL/CSV/Parquet URL.
        url: String,
    },
    /// Hugging Face Dataset Viewer or revision-pinned repository source.
    HuggingFace {
        /// Namespace/repository identifier.
        dataset: String,
        /// Dataset configuration/subset.
        subset: String,
        /// Dataset split.
        split: String,
        /// Optional immutable or symbolic revision.
        #[serde(default)]
        revision: Option<String>,
    },
}

/// Shared prompt-source selection for non-synthetic dataset kinds.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSelectionSpec {
    /// Prompt corpus selector when authored.
    #[serde(default)]
    pub corpus: Option<String>,
}

/// Resolved file/inline dataset configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDatasetSpec {
    /// Exact system prompt applied to every composed conversation.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Absolute resolved path, mutually exclusive with records.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Inline records in the exact Config-v2 shape.
    #[serde(default)]
    pub records: Option<Value>,
    /// Native loader name, or empty for structural auto-detection.
    #[serde(default)]
    pub format: String,
    /// Conversation sampling strategy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional row cap applied before composition.
    #[serde(default)]
    pub entries: Option<usize>,
    /// Dataset-local seed overriding the run seed.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Output-length fallback for rows without an authored limit.
    #[serde(default)]
    pub osl: Option<DistributionSpec>,
    /// Shared prompt-source selection for synthesized prompt content.
    #[serde(default)]
    pub prompts: Option<PromptSelectionSpec>,
    /// Optional native trace transformation and caps.
    #[serde(default)]
    pub synthesis: Option<TraceSynthesisSpec>,
    /// Recorded-agent replay policy for the `agent_recording` graph adapter.
    #[serde(default)]
    pub graph: Option<crate::config::model::dataset::RecordedAgentGraphConfig>,
    /// Dataset-level cache-bust policy; `synthesis.cache_bust_target` wins.
    #[serde(default)]
    pub cache_bust: Option<DatasetCacheBustSpec>,
    /// Fetch remote `http(s)://` image URLs at dataset generation and inline
    /// them as `data:` URLs, for servers that cannot resolve URLs themselves.
    /// Default keeps authored URLs unchanged (dispatch sends them as-is).
    #[serde(default)]
    pub prefetch_media_urls: bool,
    /// Loader/composer-specific options after Config-v2 validation.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// Dataset-level cache-bust policy projected from Config v2 `dataset.cache_bust`.
///
/// The recorded-graph path reads `synthesis.cache_bust_target` first; this block
/// is the equivalent authored spelling when no synthesis block carries it.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCacheBustSpec {
    /// Marker target name; absent or `"none"` disables marker injection.
    #[serde(default)]
    pub target: Option<String>,
}

/// Trace synthesis configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSynthesisSpec {
    /// Optional legacy corpus selector retained for protocol compatibility.
    #[serde(default)]
    pub corpus: Option<String>,
    /// Timestamp divisor.
    pub speedup_ratio: f64,
    /// Shared-prefix length multiplier.
    pub prefix_len_multiplier: f64,
    /// Independent prefix-root count.
    pub prefix_root_multiplier: u64,
    /// Unique-prompt length multiplier.
    pub prompt_len_multiplier: f64,
    /// Output-length multiplier.
    pub output_len_multiplier: f64,
    /// Original-row filter and transformed-length cap.
    #[serde(default)]
    pub max_isl: Option<u64>,
    /// Final output-length cap.
    #[serde(default)]
    pub max_osl: Option<u32>,
    /// Recorded-graph per-trace peak context filter.
    #[serde(default)]
    pub max_context_length: Option<u64>,
    /// Whether outer graph selection may wrap a finite corpus.
    #[serde(default)]
    pub allow_dataset_wrap: Option<bool>,
    /// True-idle gap cap for WEKA/Dynamo replay; null disables compression.
    #[serde(default = "default_recorded_idle_gap_cap")]
    pub idle_gap_cap_seconds: Option<f64>,
    /// Lower trajectory-start bound as a fraction of the replayable span.
    #[serde(default)]
    pub trajectory_start_min_ratio: f64,
    /// Upper trajectory-start bound as a fraction of the replayable span.
    #[serde(default)]
    pub trajectory_start_max_ratio: f64,
    /// Deterministic trajectory-start sampling seed.
    #[serde(default)]
    pub t_star_random_seed: u64,
    /// Recorded-graph recycle strategy: `sequential`, `shuffle`, or `random`.
    #[serde(default)]
    pub dataset_sampling_strategy: Option<String>,
    /// Cache-bust target; absent or `"none"` disables marker injection.
    #[serde(default)]
    pub cache_bust_target: Option<String>,
}

fn default_recorded_idle_gap_cap() -> Option<f64> {
    Some(60.0)
}

fn default_sampling_strategy() -> String {
    "sequential".into()
}

/// Native synthetic dataset configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDatasetSpec {
    /// Exact system prompt applied to every generated conversation.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Number of reusable conversations.
    pub entries: usize,
    /// Dataset-local seed overriding the run seed for generation and sampling.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Conversation sampling policy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional text generation configuration.
    #[serde(default)]
    pub prompts: Option<SyntheticPromptsSpec>,
    /// Optional shared-prefix or per-session context configuration.
    #[serde(default)]
    pub prefix_prompts: Option<SyntheticPrefixPromptsSpec>,
    /// Turns per conversation.
    #[serde(default = "one_distribution")]
    pub turns: DistributionSpec,
    /// Inter-turn delay in milliseconds.
    #[serde(default = "zero_distribution")]
    pub turn_delay_ms: DistributionSpec,
    /// Multiplicative delay scale.
    #[serde(default = "one_f64")]
    pub turn_delay_ratio: f64,
    /// Optional synthetic images.
    #[serde(default)]
    pub images: Option<SyntheticImageSpec>,
    /// Optional synthetic audio.
    #[serde(default)]
    pub audio: Option<SyntheticAudioSpec>,
    /// Optional synthetic video.
    #[serde(default)]
    pub video: Option<SyntheticVideoSpec>,
    /// Optional query/passage shape for ranking endpoints.
    #[serde(default)]
    pub rankings: Option<SyntheticRankingsSpec>,
}

/// Synthetic prompt distributions.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticPromptsSpec {
    /// Input sequence length distribution; absent disables text generation.
    #[serde(default)]
    pub isl: Option<DistributionSpec>,
    /// Output sequence length distribution; absent leaves the server limit unset.
    #[serde(default)]
    pub osl: Option<DistributionSpec>,
    /// Hash block size retained for Config-v2 completeness. Synthetic rows have no hash IDs.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Prompt corpus selector (`sonnet`, `coding`, or `random` when authored).
    #[serde(default)]
    pub corpus: Option<String>,
    /// Independently generated prompt values per turn.
    #[serde(default = "one_usize")]
    pub batch_size: usize,
    /// Paired ISL/OSL mixture, which takes precedence over independent lengths.
    #[serde(default)]
    pub sequence_distribution: Option<Vec<SequenceDistributionEntrySpec>>,
    /// Fraction of prompts, in `[0, 1]`, that reuse a shared leading token prefix
    /// so a server KV cache observes prefix hits. The default `0.0` keeps every
    /// prompt unique.
    #[serde(default)]
    pub prefix_reuse_fraction: f64,
    /// Fraction of each reusing prompt's input length, in `[0, 1]`, occupied by
    /// the shared prefix.
    #[serde(default = "default_prefix_reuse_ratio")]
    pub prefix_reuse_ratio: f64,
    /// Per-conversation cache-bust marker policy.
    #[serde(default)]
    pub cache_bust: Option<DatasetCacheBustSpec>,
}

/// One paired input/output sequence-length bucket.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceDistributionEntrySpec {
    /// Input distribution; Config v2 reduces non-normal variants to their expected value.
    pub isl: DistributionSpec,
    /// Output distribution; Config v2 reduces non-normal variants to their expected value.
    pub osl: DistributionSpec,
    /// Percentage probability.
    pub probability: f64,
}

/// Synthetic shared-prefix and conversation-context shape.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticPrefixPromptsSpec {
    /// Number of reusable first-turn prefixes.
    #[serde(default)]
    pub pool_size: Option<usize>,
    /// Tokens in each reusable prefix.
    #[serde(default)]
    pub length: Option<usize>,
    /// Tokens in the one shared system prompt.
    #[serde(default)]
    pub shared_system_length: Option<usize>,
    /// Tokens in each per-session user context.
    #[serde(default)]
    pub user_context_length: Option<usize>,
}

/// Synthetic image configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticImageSpec {
    /// Images generated per turn.
    pub batch_size: usize,
    /// Width distribution in pixels.
    pub width: DistributionSpec,
    /// Height distribution in pixels.
    pub height: DistributionSpec,
    /// PNG, JPEG, or per-image random selection.
    pub format: SyntheticImageFormatSpec,
    /// `noise`, `assets`, or an absolute local source directory.
    pub source: String,
    /// Selection policy for finite source pools.
    pub source_sampling: SourceImageSamplingSpec,
}

/// Image encoding accepted on the run wire.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticImageFormatSpec {
    /// PNG.
    Png,
    /// JPEG.
    Jpeg,
    /// Randomly select PNG or JPEG per generated image.
    Random,
}

/// Source-image selection policy.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceImageSamplingSpec {
    /// Independent draws with replacement.
    RandomWithReplacement,
    /// Shuffled cycles without replacement.
    ShuffleCycle,
    /// Sorted cycles.
    SequentialCycle,
}

/// Synthetic audio configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticAudioSpec {
    /// Audio clips generated per turn.
    pub batch_size: usize,
    /// Duration distribution in seconds.
    pub length: DistributionSpec,
    /// WAV or MP3 output.
    pub format: SyntheticAudioFormatSpec,
    /// Candidate sample rates in kHz.
    pub sample_rates: Vec<f64>,
    /// Candidate PCM bit depths.
    pub depths: Vec<u16>,
    /// Mono or stereo.
    pub channels: u16,
}

/// Synthetic audio encoding.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticAudioFormatSpec {
    /// PCM WAV.
    Wav,
    /// MP3.
    Mp3,
}

/// Synthetic video configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticVideoSpec {
    /// Videos generated per turn.
    pub batch_size: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// Frames per second.
    pub fps: u32,
    /// Optional frame width; native defaults apply when absent.
    #[serde(default)]
    pub width: Option<u32>,
    /// Optional frame height; native defaults apply when absent.
    #[serde(default)]
    pub height: Option<u32>,
    /// MP4 or WebM container.
    pub format: SyntheticVideoFormatSpec,
    /// FFmpeg video codec.
    pub codec: String,
    /// Deterministic frame-generation algorithm.
    pub synth_type: SyntheticVideoPatternSpec,
    /// Optional embedded generated audio track.
    pub audio: SyntheticVideoAudioSpec,
}

/// Synthetic video container.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticVideoFormatSpec {
    /// MP4.
    Mp4,
    /// WebM.
    Webm,
}

/// Synthetic video frame pattern.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticVideoPatternSpec {
    /// Animated geometric shapes.
    MovingShapes,
    /// Grid and frame clock.
    GridClock,
    /// Random noise frames.
    Noise,
}

/// Embedded video-audio configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticVideoAudioSpec {
    /// Sample rate in kHz.
    pub sample_rate: f64,
    /// Zero disables audio; one and two select mono/stereo.
    pub channels: u16,
    /// Optional FFmpeg audio codec.
    #[serde(default)]
    pub codec: Option<String>,
    /// PCM source bit depth.
    pub depth: u16,
}

/// Synthetic query/passage shape used by ranking endpoints.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticRankingsSpec {
    /// Passage count distribution.
    pub passages: DistributionSpec,
    /// Tokens per passage.
    pub passage_tokens: DistributionSpec,
    /// Query token distribution.
    pub query_tokens: DistributionSpec,
}

/// Config-v2 sampling distribution after Pydantic normalization.
#[derive(Clone)]
pub enum DistributionSpec {
    /// Deterministic value.
    Fixed(FixedDistributionSpec),
    /// Positive normal distribution.
    Normal(NormalDistributionSpec),
    /// Real-space mean/median log-normal distribution.
    LogNormal(LogNormalDistributionSpec),
    /// Weighted mixture.
    Multimodal(MultimodalDistributionSpec),
    /// Discrete weighted values.
    Empirical(EmpiricalDistributionSpec),
}

impl<'de> Deserialize<'de> for DistributionSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // `arbitrary_precision` represents buffered numbers as maps, so dispatch
        // through `Value` before decoding the concrete numeric variant.
        let value = Value::deserialize(deserializer)?;
        let (has_peaks, has_points, has_median, has_stddev, has_value) = {
            let object = value
                .as_object()
                .ok_or_else(|| D::Error::custom("distribution must be a JSON object"))?;
            (
                object.contains_key("peaks"),
                object.contains_key("points"),
                object.contains_key("median"),
                object.contains_key("stddev"),
                object.contains_key("value"),
            )
        };
        let decoded = if has_peaks {
            DistributionSpec::Multimodal(from_value_variant(value)?)
        } else if has_points {
            DistributionSpec::Empirical(from_value_variant(value)?)
        } else if has_median {
            DistributionSpec::LogNormal(from_value_variant(value)?)
        } else if has_stddev {
            DistributionSpec::Normal(from_value_variant(value)?)
        } else if has_value {
            DistributionSpec::Fixed(from_value_variant(value)?)
        } else {
            return Err(D::Error::custom(
                "distribution object must contain one of: value, mean+stddev, mean+median, peaks, points",
            ));
        };
        Ok(decoded)
    }
}

/// Decode one concrete distribution variant from a buffered [`Value`], mapping
/// the `serde_json` error into the caller's deserializer error type.
fn from_value_variant<T, E>(value: Value) -> Result<T, E>
where
    T: serde::de::DeserializeOwned,
    E: serde::de::Error,
{
    serde_json::from_value(value).map_err(E::custom)
}

/// Deterministic distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedDistributionSpec {
    /// Constant value.
    pub value: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Normal distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalDistributionSpec {
    /// Mean.
    pub mean: f64,
    /// Standard deviation.
    pub stddev: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Log-normal distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogNormalDistributionSpec {
    /// Real-space mean.
    pub mean: f64,
    /// Real-space median.
    pub median: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Weighted mixture configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalDistributionSpec {
    /// Weighted component distributions.
    pub peaks: Vec<PeakSpec>,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// One weighted mixture component.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeakSpec {
    /// Nested distribution.
    pub distribution: DistributionSpec,
    /// Relative non-negative weight.
    pub weight: f64,
}

/// Discrete empirical configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalDistributionSpec {
    /// Weighted discrete values.
    pub points: Vec<EmpiricalPointSpec>,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// One discrete value and weight.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalPointSpec {
    /// Sampled value.
    pub value: f64,
    /// Relative positive weight.
    pub weight: f64,
}

fn one_distribution() -> DistributionSpec {
    DistributionSpec::Fixed(FixedDistributionSpec {
        value: 1.0,
        min: None,
        max: None,
    })
}

fn zero_distribution() -> DistributionSpec {
    DistributionSpec::Fixed(FixedDistributionSpec {
        value: 0.0,
        min: None,
        max: None,
    })
}

const fn one_f64() -> f64 {
    1.0
}

const fn one_usize() -> usize {
    1
}

const fn default_prefix_reuse_ratio() -> f64 {
    0.5
}

/// Canonical result of one dataset-input adapter load.
pub struct PreparedDatasetInput {
    /// Canonical composed dataset used by every scheduled phase.
    pub dataset: Dataset,
    /// Dataset-local seed overriding the run root.
    pub random_seed: Option<u64>,
    /// Fallback requested output length for rows without one.
    pub default_output_tokens: usize,
    /// Side-channel subagent join-gate specs for the `agentic_replay` timing
    /// mode (empty for every non-agentic dataset). Carried alongside the
    /// intentionally DAG-free composed dataset and threaded into
    /// `AgenticReplayConfig` at phase-plan build time.
    pub agentic_trees: std::sync::Arc<Vec<crate::agentic_tree::TreeSpec>>,
    /// Type-erased cross-phase accelerated cache-warmup handoff carrier (Python
    /// warmup→profiling `ConversationState` handoff). Empty for every non-agentic
    /// or non-accelerated dataset; `lower_legacy_agentic` installs a live carrier
    /// only when `--agentic-cache-warmup-duration` is set. Threaded into both
    /// agentic `AgenticReplayConfig` instances at phase-plan build time.
    pub warmup_handoff: crate::agentic_tree::WarmupHandoffCarrierAny,
}

/// Inputs shared by all backend-neutral dataset source adapters.
pub struct DatasetInputContext<'a> {
    /// Frozen compile-time loader/sampler/endpoint registry universe.
    pub registry: &'a AIPerfRegistry,
    /// Authored model selection policy.
    pub models: &'a ModelsSpec,
    /// Run-level deterministic RNG root.
    pub run_rng_root: RngRoot,
    /// Fully prepared tokenizer implementation.
    pub tokenizer: &'a dyn TextTokenizer,
    /// Whether the selected endpoint expects ranking-shaped synthesis.
    pub rankings: bool,
    /// Exact descriptor selected for dataset representation and validation.
    pub endpoint_descriptor: &'static EndpointDescriptor,
    /// Trace prompt storage policy selected by the execution backend.
    pub trace_prompt_storage: Arc<dyn TracePromptStoragePolicy>,
    /// Synthetic-media generation/publication policy selected by the backend.
    pub media_generator_factory: Arc<dyn SyntheticMediaGeneratorFactory>,
}

/// One direct authored dataset-source adapter.
#[async_trait(?Send)]
pub trait DatasetInputAdapter: fmt::Debug + Send + Sync {
    /// Stable Config-v2 source discriminator.
    fn source_type(&self) -> &'static str;

    /// Strictly decode and load the authored source exactly once.
    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput>;
}

/// Injected open resolver for dataset-input adapters.
#[async_trait(?Send)]
pub trait DatasetInputAdapterResolver: fmt::Debug + Send + Sync {
    /// Select the adapter from the source discriminator and retain its loaded
    /// canonical dataset as the first shared runtime representation.
    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput>;
}

/// Deterministic built-in dataset-input adapter composition.
pub struct BuiltinRunnerDatasetInputAdapterResolver {
    adapters: BTreeMap<&'static str, Arc<dyn DatasetInputAdapter>>,
}

impl fmt::Debug for BuiltinRunnerDatasetInputAdapterResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BuiltinRunnerDatasetInputAdapterResolver")
            .field("source_types", &self.adapters.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for BuiltinRunnerDatasetInputAdapterResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinRunnerDatasetInputAdapterResolver {
    /// Compose all built-in source adapters in deterministic ID order.
    pub fn new() -> Self {
        let adapters: [Arc<dyn DatasetInputAdapter>; 3] = [
            Arc::new(SyntheticDatasetInputAdapter),
            Arc::new(FileDatasetInputAdapter),
            Arc::new(PublicDatasetInputAdapter),
        ];
        Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.source_type(), adapter))
                .collect(),
        }
    }
}

#[derive(Deserialize)]
// Keeping only the discriminator makes Serde skip unknown fields through
// `IgnoredAny` instead of allocating an adapter-owned `Value` tree.
struct DatasetInputIdentity {
    #[serde(rename = "type")]
    source_type: String,
}

#[async_trait(?Send)]
impl DatasetInputAdapterResolver for BuiltinRunnerDatasetInputAdapterResolver {
    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        // Decode only the discriminator so the selected adapter owns the full load.
        let identity: DatasetInputIdentity = serde_json::from_str(raw.get())
            .context("decoding dataset-input adapter discriminator")?;
        let adapter = self
            .adapters
            .get(identity.source_type.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "no dataset-input adapter is registered for source type {:?}",
                    identity.source_type
                )
            })?;
        adapter.load(raw, context).await
    }
}

#[derive(Debug)]
struct SyntheticDatasetInputAdapter;

#[derive(Debug)]
struct FileDatasetInputAdapter;

#[derive(Debug)]
struct PublicDatasetInputAdapter;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SyntheticDatasetInput {
    Synthetic(SyntheticDatasetSpec),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum FileDatasetInput {
    File(FileDatasetSpec),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PublicDatasetInput {
    Public(PublicDatasetSpec),
}

/// Decode an authored dataset source through `serde_json::Value`.
///
/// Every adapter source is an internally tagged (`#[serde(tag = "type")]`) enum
/// whose payload nests [`DistributionSpec`] fields with a hand-written
/// key-dispatching deserializer. serde's internally tagged machinery buffers
/// input through `serde::__private::de::Content`, and serde_json's streaming
/// `from_str` populates that buffer in a form those nested float fields fail to
/// match, so a valid `{"value": 4.0}` distribution is rejected. The
/// `serde_json::Value` deserializer buffers the same content correctly, so
/// routing every source decode through a `Value` sidesteps the streaming-only
/// defect while preserving each variant's strict field checking.
fn decode_dataset_source<T>(raw: &RawValue) -> serde_json::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(serde_json::from_str::<Value>(raw.get())?)
}

fn validate_system_prompt_endpoint(
    system_prompt: Option<&str>,
    endpoint: &EndpointDescriptor,
) -> Result<()> {
    let Some(system_prompt) = system_prompt else {
        return Ok(());
    };
    ensure!(
        !system_prompt.trim().is_empty(),
        "system prompt cannot be empty or whitespace-only"
    );
    ensure!(
        endpoint.consumes_system_message(),
        "system prompt is not supported by endpoint type {:?} (no system role), so the text \
         would never reach the wire; supported endpoint types: chat, chat_embeddings, messages, \
         responses",
        endpoint.id
    );
    Ok(())
}

fn validate_synthetic_system_prompt(spec: &SyntheticDatasetSpec) -> Result<()> {
    let Some(_system_prompt) = spec.system_prompt.as_deref() else {
        if spec
            .prompts
            .as_ref()
            .and_then(|prompts| prompts.cache_bust.as_ref())
            .and_then(|cache_bust| cache_bust.target.as_deref())
            == Some("warmup_isolation_system")
        {
            let has_generated_system = spec
                .prefix_prompts
                .as_ref()
                .is_some_and(|prefixes| prefixes.shared_system_length.is_some());
            ensure!(
                has_generated_system,
                "cache_bust=warmup_isolation_system requires a shared system prompt, but no \
                 shared_system_length or verbatim system prompt is configured"
            );
        }
        return Ok(());
    };
    if let Some(prefixes) = spec.prefix_prompts.as_ref() {
        ensure!(
            prefixes.shared_system_length.is_none(),
            "system_prompt and prefix_prompts.shared_system_length are mutually exclusive: \
             both fill the system message"
        );
        ensure!(
            !prefixes.pool_size.is_some_and(|value| value > 0)
                && !prefixes.length.is_some_and(|value| value > 0),
            "system_prompt and prefix_prompts.pool_size/length are mutually exclusive: both \
             fill the system-message prefix slot"
        );
    }
    Ok(())
}

#[async_trait(?Send)]
impl DatasetInputAdapter for SyntheticDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "synthetic"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let SyntheticDatasetInput::Synthetic(spec) =
            decode_dataset_source(raw).context("decoding synthetic dataset source")?;
        validate_system_prompt_endpoint(
            spec.system_prompt.as_deref(),
            context.endpoint_descriptor,
        )?;
        validate_synthetic_system_prompt(&spec)?;
        let rng_root = spec
            .random_seed
            .map_or(context.run_rng_root, |seed| RngRoot::new(Some(seed)));
        let default_output_tokens = synthetic_default_output_tokens(&spec)?;
        let dataset = build_synthetic_dataset(
            &spec,
            SyntheticDatasetBuildContext {
                registry: context.registry,
                models: context.models,
                rng_root,
                tokenizer: context.tokenizer,
                rankings: context.rankings,
                media_generator_factory: context.media_generator_factory.clone(),
                requires_raw_token_ids: context.endpoint_descriptor.requires_raw_token_ids,
            },
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens,
            agentic_trees: std::sync::Arc::default(),
            warmup_handoff: crate::agentic_tree::empty_warmup_handoff_carrier(),
        })
    }
}

#[async_trait(?Send)]
impl DatasetInputAdapter for FileDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "file"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let FileDatasetInput::File(spec) =
            decode_dataset_source(raw).context("decoding file dataset source")?;
        validate_system_prompt_endpoint(
            spec.system_prompt.as_deref(),
            context.endpoint_descriptor,
        )?;
        ensure!(
            spec.format != "dag_jsonl",
            "scheduled workloads cannot consume a direct dag_jsonl graph program"
        );
        let rng_root = spec
            .random_seed
            .map_or(context.run_rng_root, |seed| RngRoot::new(Some(seed)));
        let default_output_tokens = file_default_output_tokens(&spec)?;
        let dataset = build_file_dataset(
            &spec,
            FileDatasetBuildContext {
                registry: context.registry,
                models: context.models,
                run_rng_root: rng_root,
                tokenizer: context.tokenizer,
                trace_prompt_storage: context.trace_prompt_storage.clone(),
                requires_raw_token_ids: context.endpoint_descriptor.requires_raw_token_ids,
                consumes_system_message: context.endpoint_descriptor.consumes_system_message(),
            },
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens,
            agentic_trees: std::sync::Arc::default(),
            warmup_handoff: crate::agentic_tree::empty_warmup_handoff_carrier(),
        })
    }
}

#[async_trait(?Send)]
impl DatasetInputAdapter for PublicDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "public"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &DatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let PublicDatasetInput::Public(spec) =
            decode_dataset_source(raw).context("decoding public dataset source")?;
        validate_system_prompt_endpoint(
            spec.system_prompt.as_deref(),
            context.endpoint_descriptor,
        )?;
        ensure!(
            spec.format != "dag_jsonl",
            "scheduled workloads cannot consume a direct dag_jsonl graph program"
        );
        let rng_root = spec
            .random_seed
            .map_or(context.run_rng_root, |seed| RngRoot::new(Some(seed)));
        let dataset = build_public_dataset(
            context.registry,
            &spec,
            context.models,
            rng_root,
            context.tokenizer,
            context.endpoint_descriptor.requires_raw_token_ids,
            context.endpoint_descriptor.consumes_system_message(),
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens: 1,
            agentic_trees: std::sync::Arc::default(),
            warmup_handoff: crate::agentic_tree::empty_warmup_handoff_carrier(),
        })
    }
}

fn synthetic_default_output_tokens(spec: &SyntheticDatasetSpec) -> Result<usize> {
    let expected = spec
        .prompts
        .as_ref()
        .and_then(|prompts| prompts.osl.as_ref())
        .map(distribution)
        .transpose()?
        .map(|distribution| distribution.expected_value().ceil())
        .filter(|value| *value > 0.0)
        .unwrap_or(1.0);
    checked_default_output_tokens(expected)
}

fn file_default_output_tokens(spec: &FileDatasetSpec) -> Result<usize> {
    let expected = spec
        .osl
        .as_ref()
        .map(distribution)
        .transpose()?
        .map(|distribution| distribution.expected_value().ceil())
        .unwrap_or(1.0);
    checked_default_output_tokens(expected)
}

fn checked_default_output_tokens(expected: f64) -> Result<usize> {
    ensure!(
        expected.is_finite() && expected > 0.0 && expected < usize::MAX as f64,
        "default OSL expected value is outside the native usize range"
    );
    Ok(expected as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::{ChatEndpoint, CompletionsEndpoint, Endpoint};

    #[test]
    fn strict_dataset_boundary_validates_system_prompt_capability_and_shape() {
        validate_system_prompt_endpoint(Some("exact"), ChatEndpoint.descriptor()).unwrap();
        let error =
            validate_system_prompt_endpoint(Some("exact"), CompletionsEndpoint.descriptor())
                .unwrap_err()
                .to_string();
        assert!(error.contains("would never reach the wire"), "{error}");

        for prefix_prompts in [
            serde_json::json!({"shared_system_length": 4}),
            serde_json::json!({"pool_size": 2, "length": 4}),
        ] {
            let SyntheticDatasetInput::Synthetic(spec) =
                serde_json::from_value(serde_json::json!({
                    "type": "synthetic",
                    "system_prompt": "exact",
                    "entries": 1,
                    "prefix_prompts": prefix_prompts,
                    "turns": {"value": 1.0},
                    "turn_delay_ms": {"value": 0.0}
                }))
                .unwrap();
            let error = validate_synthetic_system_prompt(&spec)
                .unwrap_err()
                .to_string();
            assert!(error.contains("mutually exclusive"), "{error}");
        }
    }

    #[test]
    fn strict_dataset_boundary_accepts_verbatim_warmup_system_and_user_context() {
        let SyntheticDatasetInput::Synthetic(spec) = serde_json::from_value(serde_json::json!({
            "type": "synthetic",
            "system_prompt": "exact",
            "entries": 1,
            "prompts": {
                "cache_bust": {"target": "warmup_isolation_system"}
            },
            "prefix_prompts": {"user_context_length": 4},
            "turns": {"value": 1.0},
            "turn_delay_ms": {"value": 0.0}
        }))
        .unwrap();
        validate_synthetic_system_prompt(&spec).unwrap();

        let SyntheticDatasetInput::Synthetic(without_system) =
            serde_json::from_value(serde_json::json!({
                "type": "synthetic",
                "entries": 1,
                "prompts": {
                    "cache_bust": {"target": "warmup_isolation_system"}
                },
                "turns": {"value": 1.0},
                "turn_delay_ms": {"value": 0.0}
            }))
            .unwrap();
        let error = validate_synthetic_system_prompt(&without_system)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires a shared system prompt"), "{error}");
    }

    #[test]
    fn distribution_spec_decodes_under_arbitrary_precision() {
        // Numeric variants must decode with workspace-wide `arbitrary_precision`.
        let fixed: DistributionSpec = serde_json::from_str(r#"{"value":256.0}"#).unwrap();
        assert!(matches!(fixed, DistributionSpec::Fixed(spec) if spec.value == 256.0));
        let normal: DistributionSpec =
            serde_json::from_str(r#"{"mean":256.0,"stddev":0.0}"#).unwrap();
        assert!(matches!(normal, DistributionSpec::Normal(_)));
        let lognormal: DistributionSpec =
            serde_json::from_str(r#"{"mean":2.0,"median":1.5}"#).unwrap();
        assert!(matches!(lognormal, DistributionSpec::LogNormal(_)));

        let SyntheticDatasetInput::Synthetic(spec) = serde_json::from_str(
            r#"{"type":"synthetic","entries":1,"sampling":"sequential",
                "prompts":{"isl":{"mean":256.0,"stddev":0.0},"osl":{"value":8.0}},
                "turns":{"value":1.0},"turn_delay_ms":{"value":0.0}}"#,
        )
        .unwrap();
        assert_eq!(spec.entries, 1);
        assert!(matches!(spec.turns, DistributionSpec::Fixed(_)));
    }

    #[test]
    fn synthetic_prompt_corpus_decodes_from_protocol_v2() {
        let SyntheticDatasetInput::Synthetic(spec) = serde_json::from_str(
            r#"{"type":"synthetic","entries":1,"sampling":"sequential",
                "prompts":{"isl":{"mean":256.0,"stddev":0.0},"corpus":"coding"},
                "turns":{"value":1.0},"turn_delay_ms":{"value":0.0}}"#,
        )
        .unwrap();
        assert_eq!(
            spec.prompts
                .as_ref()
                .and_then(|prompts| prompts.corpus.as_deref()),
            Some("coding")
        );
    }

    #[test]
    fn file_prompt_corpus_decodes_from_protocol_v2() {
        let FileDatasetInput::File(spec) = serde_json::from_str(
            r#"{"type":"file","format":"mooncake_trace",
                "records":[{"input_length":16,"output_length":4}],
                "prompts":{"corpus":"random"}}"#,
        )
        .unwrap();
        assert_eq!(
            spec.prompts
                .as_ref()
                .and_then(|prompts| prompts.corpus.as_deref()),
            Some("random")
        );
    }

    #[test]
    fn public_prompt_corpus_decodes_from_protocol_v2() {
        let PublicDatasetInput::Public(spec) = serde_json::from_str(
            r#"{"type":"public","name":"sharegpt","format":"sharegpt",
                "source":{"type":"url","url":"https://example.invalid/sharegpt.jsonl"},
                "prompts":{"corpus":"coding"}}"#,
        )
        .unwrap();
        assert_eq!(
            spec.prompts
                .as_ref()
                .and_then(|prompts| prompts.corpus.as_deref()),
            Some("coding")
        );
    }

    #[test]
    fn discriminator_decode_skips_adapter_fields_without_retaining_them() {
        assert_eq!(
            std::mem::size_of::<DatasetInputIdentity>(),
            std::mem::size_of::<String>(),
            "the selector DTO must retain only its discriminator"
        );
        let identity: DatasetInputIdentity = serde_json::from_value(serde_json::json!({
            "type": "synthetic",
            "entries": 1,
            "prompts": {
                "opaque": "x".repeat(1 << 20),
                "isl": {"value": 8.0}
            }
        }))
        .unwrap();

        assert_eq!(identity.source_type, "synthetic");
    }

    #[test]
    fn trace_synthesis_carries_tstar_and_warmup_knobs() {
        let spec: TraceSynthesisSpec = serde_json::from_str(
            r#"{
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "trajectory_start_min_ratio": 0.25,
                "trajectory_start_max_ratio": 0.75,
                "t_star_random_seed": 4242
            }"#,
        )
        .unwrap();
        assert_eq!(spec.trajectory_start_min_ratio, 0.25);
        assert_eq!(spec.trajectory_start_max_ratio, 0.75);
        assert_eq!(spec.t_star_random_seed, 4242);
        assert_eq!(spec.idle_gap_cap_seconds, Some(60.0));
    }

    #[test]
    fn trace_synthesis_tstar_knobs_default_to_disabled() {
        let spec: TraceSynthesisSpec = serde_json::from_str(
            r#"{
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0
            }"#,
        )
        .unwrap();
        assert_eq!(spec.trajectory_start_min_ratio, 0.0);
        assert_eq!(spec.trajectory_start_max_ratio, 0.0);
        assert_eq!(spec.t_star_random_seed, 0);
    }

    #[test]
    fn trace_synthesis_rejects_unknown_fields_with_tstar_present() {
        let result = serde_json::from_str::<TraceSynthesisSpec>(
            r#"{
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "trajectory_start_min_ratio": 0.25,
                "unknown_tstar_field": true
            }"#,
        );
        let Err(error) = result else {
            panic!("synthesis accepted an unknown field")
        };
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn selected_adapter_decode_remains_strict() {
        let result = serde_json::from_value::<SyntheticDatasetInput>(serde_json::json!({
            "type": "synthetic",
            "entries": 1,
            "unknown_policy": true
        }));
        let Err(error) = result else {
            panic!("selected adapter accepted an unknown policy field")
        };

        assert!(error.to_string().contains("unknown field"));
    }
}
