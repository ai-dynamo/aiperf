// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral authored dataset policy and direct input adapters.
//!
//! These values are shared source policy, not a process-protocol DTO. Protocol
//! v1 wraps them for compatibility; protocol v2 selects a trait adapter and
//! retains the resulting native Dataset as its first shared runtime value.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use aiperf_dataset::{
    Dataset, SyntheticMediaGeneratorFactory, TextTokenizer, TracePromptStoragePolicy,
};
use aiperf_endpoints::EndpointDescriptor;
use aiperf_extensions::AiperfRegistry;
use aiperf_rng::RngRoot;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};

use crate::execute::{
    build_file_dataset, build_public_dataset, build_synthetic_dataset, distribution,
};
use crate::protocol::ModelsSpec;

/// Public dataset configuration resolved from the Python plugin registry.
///
/// Python keeps ownership of the named plugin catalog in
/// `src/aiperf/plugin/plugins.yaml:1733-1957`; Rust receives only the explicit
/// source coordinates and loader options needed for one run.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicDatasetSpec {
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

/// Resolved file/inline dataset configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDatasetSpec {
    /// Absolute resolved path, mutually exclusive with records.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Inline records in the exact Config-v2 shape.
    #[serde(default)]
    pub records: Option<Value>,
    /// Native loader registration name.
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
    /// Optional native trace transformation and caps.
    #[serde(default)]
    pub synthesis: Option<TraceSynthesisSpec>,
    /// Loader/composer-specific options after Config-v2 validation.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// Trace synthesis configuration from
/// `src/aiperf/config/dataset/trace.py:20-117`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSynthesisSpec {
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
    /// Recorded content corpus (`coding` by default, or `sonnet`).
    #[serde(default)]
    pub corpus: Option<String>,
}

fn default_recorded_idle_gap_cap() -> Option<f64> {
    Some(60.0)
}

fn default_sampling_strategy() -> String {
    "sequential".into()
}

/// Native synthetic dataset configuration.
///
/// This is the process-boundary projection of
/// `src/aiperf/config/dataset/config.py:62-245`; content sub-shapes follow
/// `src/aiperf/config/dataset/content.py:50-459` and
/// `src/aiperf/config/dataset/video.py:41-205`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDatasetSpec {
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
    /// Independently generated prompt values per turn.
    #[serde(default = "one_usize")]
    pub batch_size: usize,
    /// Paired ISL/OSL mixture, which takes precedence over independent lengths.
    #[serde(default)]
    pub sequence_distribution: Option<Vec<SequenceDistributionEntrySpec>>,
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
#[derive(Clone, Deserialize)]
#[serde(untagged)]
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

/// Retained result of one direct dataset-input adapter load.
pub struct PreparedDatasetInput {
    /// Canonical composed dataset used by every scheduled phase.
    pub dataset: Dataset,
    /// Dataset-local seed overriding the run root.
    pub random_seed: Option<u64>,
    /// Fallback requested output length for rows without one.
    pub default_output_tokens: usize,
}

/// Inputs shared by all backend-neutral dataset source adapters.
pub struct RunnerDatasetInputContext<'a> {
    /// Frozen compile-time loader/sampler/endpoint registry universe.
    pub registry: &'a AiperfRegistry,
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
pub trait RunnerDatasetInputAdapter: fmt::Debug + Send + Sync {
    /// Stable Config-v2 source discriminator.
    fn source_type(&self) -> &'static str;

    /// Strictly decode and load the authored source exactly once.
    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput>;
}

/// Injected open resolver for dataset-input adapters.
#[async_trait(?Send)]
pub trait RunnerDatasetInputAdapterResolver: fmt::Debug + Send + Sync {
    /// Select the adapter from the source discriminator and retain its loaded
    /// canonical dataset as the first shared runtime representation.
    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput>;
}

/// Deterministic built-in dataset-input adapter composition.
pub struct BuiltinRunnerDatasetInputAdapterResolver {
    adapters: BTreeMap<&'static str, Arc<dyn RunnerDatasetInputAdapter>>,
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
        let adapters: [Arc<dyn RunnerDatasetInputAdapter>; 3] = [
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
impl RunnerDatasetInputAdapterResolver for BuiltinRunnerDatasetInputAdapterResolver {
    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        // This decode reads only the open discriminator. The selected adapter
        // then performs the sole full decode and source load; no intermediate
        // DatasetSpec, Conversation, or alternate runtime representation exists.
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

#[async_trait(?Send)]
impl RunnerDatasetInputAdapter for SyntheticDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "synthetic"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let SyntheticDatasetInput::Synthetic(spec) =
            serde_json::from_str(raw.get()).context("decoding synthetic dataset source")?;
        let rng_root = spec
            .random_seed
            .map_or(context.run_rng_root, |seed| RngRoot::new(Some(seed)));
        let default_output_tokens = synthetic_default_output_tokens(&spec)?;
        let dataset = build_synthetic_dataset(
            context.registry,
            &spec,
            context.models,
            rng_root,
            context.tokenizer,
            context.rankings,
            context.media_generator_factory.clone(),
            context.endpoint_descriptor.requires_raw_token_ids,
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens,
        })
    }
}

#[async_trait(?Send)]
impl RunnerDatasetInputAdapter for FileDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "file"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let FileDatasetInput::File(spec) =
            serde_json::from_str(raw.get()).context("decoding file dataset source")?;
        ensure!(
            spec.format != "dag_jsonl",
            "scheduled workloads cannot consume a direct dag_jsonl graph program"
        );
        let rng_root = spec
            .random_seed
            .map_or(context.run_rng_root, |seed| RngRoot::new(Some(seed)));
        let default_output_tokens = file_default_output_tokens(&spec)?;
        let dataset = build_file_dataset(
            context.registry,
            &spec,
            context.models,
            rng_root,
            context.tokenizer,
            context.trace_prompt_storage.clone(),
            context.endpoint_descriptor.requires_raw_token_ids,
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens,
        })
    }
}

#[async_trait(?Send)]
impl RunnerDatasetInputAdapter for PublicDatasetInputAdapter {
    fn source_type(&self) -> &'static str {
        "public"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &RunnerDatasetInputContext<'_>,
    ) -> Result<PreparedDatasetInput> {
        let PublicDatasetInput::Public(spec) =
            serde_json::from_str(raw.get()).context("decoding public dataset source")?;
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
        )
        .await?;
        dataset.validate_for_endpoint(context.endpoint_descriptor)?;
        Ok(PreparedDatasetInput {
            dataset,
            random_seed: spec.random_seed,
            default_output_tokens: 1,
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
        expected.is_finite() && expected > 0.0 && expected <= usize::MAX as f64,
        "default OSL expected value is outside the native usize range"
    );
    Ok(expected as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

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
