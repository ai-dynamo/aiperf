// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned direct Graph-IR input adapters.
//!
//! Python projects the authored file source without acquisition or parsing.
//! This module performs an
//! identity-only format lookup, then gives the untouched object to exactly one
//! selected adapter. The adapter owns the sole strict full decode and lowers
//! directly to [`GraphInputBundle`]; no protocol-v1 DTO, linear
//! [`crate::dataset::Dataset`],
//! conversation, or second graph-source representation exists in this path.
//!
//! A future linked distribution injects another [`GraphInputAdapter`]
//! through the resolver. Its private authored fields remain invisible to the
//! coordinator.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::dataset::{DatasetSource, LoadConfig, TextTokenizer};
use crate::graph::input::{GraphInputBundle, GraphInputConfig, compile_dag_jsonl_input};
use crate::graph::recorded::{
    PromptCorpus, RecordedTraceInputConfig, compile_aiperf_trace_input, compile_dynamo_trace_input,
    compile_weka_trace_input,
};
use crate::graph::tstar::{
    PermutationDraw, RecycleDrawMode, legacy_random_seed, legacy_shuffle_seed,
};
use crate::rng::RngRoot;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};

use crate::engine::execute::distribution;
use crate::engine::protocol::{
    DistributionSpec, FileDatasetSpec, PublicDatasetSourceSpec, PublicDatasetSpec,
    TraceSynthesisSpec,
};

/// Recorded-graph trajectory-start (`t*`) window bound to a prepared input.
///
/// Carries the C1 protocol-v2 synthesis knobs
/// (`trajectory_start_min_ratio` / `trajectory_start_max_ratio` /
/// `t_star_random_seed`) forward to the graph phase runtime, where each phase
/// samples a per-trace `t*` via [`crate::graph::tstar::WindowTStarSampler`] and
/// applies the warmup/profiling snapshot split. The default `[0.0, 0.0]` window
/// with seed `0` yields `t* = 0` for every trace, i.e. full native replay:
/// profiling runs the whole graph unchanged and warmup is empty.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TStarWindow {
    /// Lower window bound as a fraction of each trace's replayable span.
    pub start_min_ratio: f64,
    /// Upper window bound as a fraction of each trace's replayable span.
    pub start_max_ratio: f64,
    /// Base RNG seed salted per `(trace_id, lane)` by the `t*` sampler
    /// (`_seed_for_trace_lane`). This is `t_star_random_seed`, and is DISTINCT
    /// from the dataset-sampler run root below.
    pub random_seed: u64,
    /// The RESOLVED run root seed (`config.random_seed`) the dataset-sampling
    /// recycle draw derives its child generator from. Legacy `ShuffleSampler` /
    /// `RandomSampler` seed their generators from the run root
    /// (`rng.init(config.random_seed)`), NOT `t_star_random_seed`; the
    /// per-strategy child seed is salted off this root in
    /// [`TStarWindow::recycle_draw`] ([`crate::graph::tstar::legacy_shuffle_seed`]
    /// for shuffle, [`crate::graph::tstar::legacy_random_seed`] for random), so a
    /// future sampler adds a salt rather than a new field. Resolved ONCE at
    /// construction (an absent run seed substitutes a single entropy value there,
    /// documented at the construction site) so every draw site sharing this
    /// `Copy` window agrees. Defaults to `0`.
    pub run_random_seed: u64,
    /// Resolved dataset-sampling strategy governing WHICH corpus template a
    /// freed recycle lane serves. `Sequential` (the default) keeps the historic
    /// `x % total` cursor draw; `Shuffle` routes through the legacy
    /// persistent-epoch shuffle (without replacement); `Random` routes through
    /// the legacy CPython MT19937 `choice` draw (with replacement). See
    /// [`GraphSamplingStrategy`].
    pub sampling_strategy: GraphSamplingStrategy,
}

impl TStarWindow {
    /// Build the resolved recycle-index draw for this window's strategy.
    ///
    /// The child generator seed is salted off [`TStarWindow::run_random_seed`] per
    /// strategy, so `Sequential`/`Shuffle`/`RandomSampler` all derive from the SAME
    /// run root with different salts. Building a fresh draw at each site is safe:
    /// the draw is a pure function of `(mode, child_seed)`, so the pressure stage
    /// and the profiling recycle (built independently) agree draw-for-draw.
    pub fn recycle_draw(&self) -> PermutationDraw {
        match self.sampling_strategy.draw_mode() {
            RecycleDrawMode::Sequential => PermutationDraw::sequential(),
            RecycleDrawMode::Shuffle => {
                PermutationDraw::shuffle(legacy_shuffle_seed(self.run_random_seed))
            }
            RecycleDrawMode::Random => {
                PermutationDraw::random(legacy_random_seed(self.run_random_seed))
            }
        }
    }
}

/// Resolved dataset-sampling strategy for the recorded-graph recycle draw.
///
/// Port of the `sequential`/`shuffle`/`random` values of the Python
/// `DatasetSamplingStrategy` dynamic enum (`aiperf/plugin/enums.py:53`).
/// `Sequential`/`Shuffle`/`Random` reproduce legacy agentx `SequentialSampler`
/// (`x % total`), `ShuffleSampler` (persistent-epoch shuffle, without
/// replacement), and `RandomSampler` (CPython MT19937 `choice`, WITH replacement)
/// BYTE-EXACT. Extension seam: a new sampling policy adds a variant here, a
/// [`RecycleDrawMode`] branch, and a derive salt, never a hardcoded mode string
/// elsewhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GraphSamplingStrategy {
    /// Historic cursor-with-wrap draw (`x % total`); byte-unchanged default
    /// (legacy `SequentialSampler`).
    #[default]
    Sequential,
    /// Legacy `ShuffleSampler` persistent-epoch shuffle (without replacement).
    Shuffle,
    /// Legacy `RandomSampler`: CPython MT19937 `choice` draw (WITH replacement),
    /// seeded from `rng.derive("dataset.sampler.random")`. No longer coerced to
    /// `Shuffle` — see [`crate::graph::tstar::legacy_random_seed`] and
    /// [`crate::graph::tstar::PermutationDraw::random`].
    Random,
}

impl GraphSamplingStrategy {
    /// Parse a wire strategy string, defaulting unknown/absent values to
    /// [`GraphSamplingStrategy::Sequential`] so an older or partial wire request
    /// keeps the byte-unchanged sequential draw. Matches the Python enum values
    /// `sequential`/`shuffle`/`random` (`aiperf/plugin/enums.py:54`).
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some("shuffle") => Self::Shuffle,
            Some("random") => Self::Random,
            _ => Self::Sequential,
        }
    }

    /// Map the resolved wire strategy to the byte-exact legacy sampler family the
    /// recycle draw reproduces. Port of `graph_ir_replay.py:_draw_is_shuffled`
    /// (lines 821-834) generalized to three modes: `sequential` (and any unknown
    /// value) take the byte-identical `x % total` draw, `shuffle` the
    /// persistent-epoch shuffle, `random` the with-replacement CPython draw.
    pub fn draw_mode(self) -> RecycleDrawMode {
        match self {
            Self::Sequential => RecycleDrawMode::Sequential,
            Self::Shuffle => RecycleDrawMode::Shuffle,
            Self::Random => RecycleDrawMode::Random,
        }
    }
}

/// Recorded-graph cache-bust marker target.
///
/// Port of the subset of agentx's `CacheBustTarget`
/// (`ajc/agentx:src/aiperf/common/enums`) that the native recorded-graph path
/// honors. agentx supports SYSTEM_{PREFIX,SUFFIX} and FIRST_TURN_{PREFIX,SUFFIX};
/// the `inferencex-agentx-mvp` scenario locks `first_turn_prefix`, which is the
/// only target the native runner materializes today. `None` (the default) is a
/// byte-unchanged no-op. Extension seam: a new target adds one variant here, one
/// `parse` arm, and one marker-placement arm in
/// [`crate::engine::graph_execution`] — nothing else changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheBustTarget {
    /// No marker; recorded content is sent verbatim.
    #[default]
    None,
    /// Prepend a per-conversation `[rid:<digest>]\n\n` marker to the first user
    /// message of every request (agentx `FIRST_TURN_PREFIX`).
    FirstTurnPrefix,
}

impl CacheBustTarget {
    /// Resolve the projected `cache_bust_target` string. Unknown/absent values
    /// fall back to [`CacheBustTarget::None`] (fail-open to byte-unchanged
    /// replay rather than inventing a target).
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some("first_turn_prefix") => Self::FirstTurnPrefix,
            _ => Self::None,
        }
    }

    /// Whether a marker must be materialized for this target.
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Canonical result retained after one selected graph-input adapter load.
pub struct PreparedRunnerGraphInput {
    /// Complete executable Graph-IR roots plus their frozen segment arena.
    pub bundle: GraphInputBundle,
    /// Dataset-local seed overriding the run root.
    pub random_seed: Option<u64>,
    /// Fallback output-token limit for nodes without an authored value.
    pub default_output_tokens: usize,
    /// Whether phase admission may recycle a finite recorded root corpus.
    pub allow_dataset_wrap: bool,
    /// Trajectory-start (`t*`) window for the warmup/profiling snapshot split.
    pub t_star_window: TStarWindow,
    /// Cache-bust marker target for the recorded first-turn user message.
    pub cache_bust_target: CacheBustTarget,
}

// `GraphInputBundle` holds an `Arc<dyn SegmentStore>` that is not `Debug`, so
// this summarizes the prepared bundle rather than deriving over it.
impl fmt::Debug for PreparedRunnerGraphInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedRunnerGraphInput")
            .field("plans", &self.bundle.plans.len())
            .field("format", &self.bundle.metadata.format)
            .field("random_seed", &self.random_seed)
            .field("default_output_tokens", &self.default_output_tokens)
            .field("allow_dataset_wrap", &self.allow_dataset_wrap)
            .field("t_star_window", &self.t_star_window)
            .finish_non_exhaustive()
    }
}

/// Inputs shared by every direct graph-source adapter.
pub struct GraphInputContext<'a> {
    /// Fully prepared tokenizer used during segment interning and token counts.
    pub tokenizer: &'a dyn TextTokenizer,
    /// Run-root seed used by recorded content reconstruction.
    pub run_random_seed: Option<u64>,
}

/// One direct authored graph-source adapter.
#[async_trait(?Send)]
pub trait GraphInputAdapter: fmt::Debug + Send + Sync {
    /// Stable authored format discriminator.
    fn format(&self) -> &'static str;

    /// Strictly decode and load one authored source exactly once.
    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput>;
}

/// Injected open resolver for direct graph-input adapters.
#[async_trait(?Send)]
pub trait GraphInputAdapterResolver: fmt::Debug + Send + Sync {
    /// Validate only that the open format identity selects a linked adapter.
    ///
    /// Adapter-owned fields remain untouched. Full strict decoding is deferred
    /// to [`Self::load`], which is invoked exactly once during preparation.
    fn validate_identity(&self, raw: &RawValue) -> Result<()>;

    /// Select the format adapter and retain its canonical Graph-IR output.
    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput>;
}

/// Deterministic built-in graph-input adapter composition.
pub struct BuiltinRunnerGraphInputAdapterResolver {
    adapters: BTreeMap<&'static str, Arc<dyn GraphInputAdapter>>,
}

impl fmt::Debug for BuiltinRunnerGraphInputAdapterResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BuiltinRunnerGraphInputAdapterResolver")
            .field("formats", &self.adapters.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for BuiltinRunnerGraphInputAdapterResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinRunnerGraphInputAdapterResolver {
    /// Compose the built-in direct Graph-IR formats.
    pub fn new() -> Self {
        let adapters: [Arc<dyn GraphInputAdapter>; 4] = [
            Arc::new(DagJsonlRunnerGraphInputAdapter),
            Arc::new(WekaTraceRunnerGraphInputAdapter),
            Arc::new(DynamoTraceRunnerGraphInputAdapter),
            Arc::new(AIPerfTraceRunnerGraphInputAdapter),
        ];
        Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.format(), adapter))
                .collect(),
        }
    }

    fn selected(&self, raw: &RawValue) -> Result<&dyn GraphInputAdapter> {
        // This intentionally reads only the open discriminator. The selected
        // adapter below remains the sole owner of the full authored object.
        let identity: GraphInputIdentity = serde_json::from_str(raw.get())
            .context("decoding graph-input adapter discriminator")?;
        self.selected_format(&identity.format)
    }

    fn selected_format(&self, format: &str) -> Result<&dyn GraphInputAdapter> {
        self.adapters.get(format).map(Arc::as_ref).ok_or_else(|| {
            anyhow!(
                "no direct Graph-IR input adapter is registered for format {:?}",
                format
            )
        })
    }
}

/// Strictly decode an authored graph-input object from its raw JSON text.
///
/// `aiperf` no longer links serde_json with the globally side-effecting
/// `arbitrary_precision` feature (recorded-trace hash ids above `u64::MAX` are
/// now captured token-by-token via `RawValue` inside `aiperf-graph`), so
/// streaming `from_str` decodes struct `f64` fields — the recorded-trace
/// `max_osl`/`idle_gap_cap_seconds` knobs — directly again.
fn decode_graph_input<T>(raw: &RawValue) -> serde_json::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(raw.get())
}

#[async_trait(?Send)]
impl GraphInputAdapterResolver for BuiltinRunnerGraphInputAdapterResolver {
    fn validate_identity(&self, raw: &RawValue) -> Result<()> {
        self.selected(raw).map(drop)
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        self.selected(raw)?.load(raw, context).await
    }
}

#[derive(Deserialize)]
// Keeping only the discriminator makes Serde skip unknown fields through
// `IgnoredAny` instead of allocating an adapter-owned `Value` tree.
struct GraphInputIdentity {
    format: String,
}

/// Built-in `dag_jsonl` authored adapter and sole product format boundary.
#[derive(Debug)]
pub struct DagJsonlRunnerGraphInputAdapter;

#[async_trait(?Send)]
impl GraphInputAdapter for DagJsonlRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "dag_jsonl"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: DagJsonlDatasetInput =
            decode_graph_input(raw).context("decoding direct dag_jsonl graph input")?;
        self.load_decoded(input, context).await
    }
}

impl DagJsonlRunnerGraphInputAdapter {
    async fn load_decoded(
        &self,
        input: DagJsonlDatasetInput,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let prepared = match input {
            DagJsonlDatasetInput::File(spec) => spec.prepare(self.format())?,
            DagJsonlDatasetInput::Public(spec) => prepare_public(spec, self.format())?,
        };
        let bundle = compile_dag_jsonl_input(prepared.input, context.tokenizer)
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .context("loading and lowering direct authored dag_jsonl Graph-IR input")?;
        ensure!(
            !bundle.plans.is_empty(),
            "authored Graph-IR input contains no root traces after root limiting"
        );
        ensure!(
            bundle.metadata.format == self.format(),
            "Graph-IR adapter {:?} returned bundle format {:?}",
            self.format(),
            bundle.metadata.format
        );
        Ok(PreparedRunnerGraphInput {
            bundle,
            random_seed: prepared.random_seed,
            default_output_tokens: prepared.default_output_tokens,
            allow_dataset_wrap: true,
            // Authored `dag_jsonl` programs carry no recorded timing, so the
            // trajectory-start split never engages: the default window yields
            // `t* = 0` (profiling full, warmup empty). Authored programs supply
            // their own message content verbatim, so no cache-bust marker is
            // applied.
            t_star_window: TStarWindow::default(),
            cache_bust_target: CacheBustTarget::None,
        })
    }
}

/// Built-in native WEKA recorded-trace adapter.
#[derive(Debug)]
pub struct WekaTraceRunnerGraphInputAdapter;

/// Built-in native Dynamo recorded-trace adapter.
#[derive(Debug)]
pub struct DynamoTraceRunnerGraphInputAdapter;

/// Built-in native `aiperf.trace.v1` recorded-trace adapter.
#[derive(Debug)]
pub struct AIPerfTraceRunnerGraphInputAdapter;

#[async_trait(?Send)]
impl GraphInputAdapter for AIPerfTraceRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "aiperf_trace"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: RecordedDatasetInput =
            decode_graph_input(raw).context("decoding direct aiperf_trace graph input")?;
        self.load_decoded(input, context).await
    }
}

impl AIPerfTraceRunnerGraphInputAdapter {
    async fn load_decoded(
        &self,
        input: RecordedDatasetInput,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let prepared = match input {
            RecordedDatasetInput::File(input) => {
                prepare_recorded_file(input, self.format(), context)?
            }
            RecordedDatasetInput::Public(input) => {
                prepare_recorded_public(input, self.format(), context)?
            }
        };
        let random_seed = prepared.random_seed;
        let default_output_tokens = prepared.default_output_tokens;
        let allow_dataset_wrap = prepared.allow_dataset_wrap;
        let t_star_window = prepared.t_star_window;
        let cache_bust_target = prepared.cache_bust_target;
        let bundle = compile_aiperf_trace_input(prepared.input, context.tokenizer)
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .context("loading and lowering native aiperf_trace Graph-IR input")?;
        finish_recorded_input(
            bundle,
            random_seed,
            default_output_tokens,
            allow_dataset_wrap,
            t_star_window,
            cache_bust_target,
            self.format(),
        )
    }
}

#[async_trait(?Send)]
impl GraphInputAdapter for WekaTraceRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "weka_trace"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: RecordedDatasetInput =
            decode_graph_input(raw).context("decoding direct WEKA graph input")?;
        self.load_decoded(input, context).await
    }
}

#[async_trait(?Send)]
impl GraphInputAdapter for DynamoTraceRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "dynamo_trace"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: RecordedDatasetInput =
            decode_graph_input(raw).context("decoding direct Dynamo graph input")?;
        self.load_decoded(input, context).await
    }
}

impl WekaTraceRunnerGraphInputAdapter {
    async fn load_decoded(
        &self,
        input: RecordedDatasetInput,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let prepared = match input {
            RecordedDatasetInput::File(input) => {
                prepare_recorded_file(input, self.format(), context)?
            }
            RecordedDatasetInput::Public(input) => {
                prepare_recorded_public(input, self.format(), context)?
            }
        };
        let random_seed = prepared.random_seed;
        let default_output_tokens = prepared.default_output_tokens;
        let allow_dataset_wrap = prepared.allow_dataset_wrap;
        let t_star_window = prepared.t_star_window;
        let cache_bust_target = prepared.cache_bust_target;
        let bundle = compile_weka_trace_input(prepared.input, context.tokenizer)
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .context("loading and lowering native WEKA Graph-IR input")?;
        finish_recorded_input(
            bundle,
            random_seed,
            default_output_tokens,
            allow_dataset_wrap,
            t_star_window,
            cache_bust_target,
            self.format(),
        )
    }
}

impl DynamoTraceRunnerGraphInputAdapter {
    async fn load_decoded(
        &self,
        input: RecordedDatasetInput,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let RecordedDatasetInput::File(input) = input else {
            return Err(anyhow!(
                "dynamo_trace does not support public dataset sources"
            ));
        };
        ensure!(
            input.path.is_some(),
            "dynamo_trace product input requires a file, directory, or segmented-prefix path"
        );
        let prepared = prepare_recorded_file(input, self.format(), context)?;
        let random_seed = prepared.random_seed;
        let default_output_tokens = prepared.default_output_tokens;
        let allow_dataset_wrap = prepared.allow_dataset_wrap;
        let t_star_window = prepared.t_star_window;
        let cache_bust_target = prepared.cache_bust_target;
        let bundle = compile_dynamo_trace_input(prepared.input, context.tokenizer)
            .await
            .map_err(|error| anyhow!(error.to_string()))
            .context("loading and lowering native Dynamo Graph-IR input")?;
        finish_recorded_input(
            bundle,
            random_seed,
            default_output_tokens,
            allow_dataset_wrap,
            t_star_window,
            cache_bust_target,
            self.format(),
        )
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RecordedDatasetInput {
    File(RecordedFileInput),
    Public(PublicDatasetSpec),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordedFileInput {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    records: Option<Value>,
    format: String,
    #[serde(default = "default_sequential")]
    sampling: String,
    #[serde(default)]
    synthesis: Option<TraceSynthesisSpec>,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<DistributionSpec>,
    #[serde(default)]
    options: Map<String, Value>,
}

impl From<FileDatasetSpec> for RecordedFileInput {
    fn from(spec: FileDatasetSpec) -> Self {
        Self {
            name: None,
            path: spec.path,
            records: spec.records,
            format: spec.format,
            sampling: spec.sampling,
            synthesis: spec.synthesis,
            entries: spec.entries,
            random_seed: spec.random_seed,
            osl: spec.osl,
            options: spec.options,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_recorded_input(
    bundle: GraphInputBundle,
    random_seed: Option<u64>,
    default_output_tokens: usize,
    allow_dataset_wrap: bool,
    t_star_window: TStarWindow,
    cache_bust_target: CacheBustTarget,
    expected_format: &str,
) -> Result<PreparedRunnerGraphInput> {
    ensure!(
        !bundle.plans.is_empty(),
        "recorded Graph-IR input contains no roots after selection"
    );
    ensure!(
        bundle.metadata.format == expected_format,
        "recorded adapter {expected_format:?} returned bundle format {:?}",
        bundle.metadata.format
    );
    Ok(PreparedRunnerGraphInput {
        bundle,
        random_seed,
        default_output_tokens,
        allow_dataset_wrap,
        t_star_window,
        cache_bust_target,
    })
}

struct PreparedRecordedInput {
    input: RecordedTraceInputConfig,
    random_seed: Option<u64>,
    default_output_tokens: usize,
    allow_dataset_wrap: bool,
    t_star_window: TStarWindow,
    cache_bust_target: CacheBustTarget,
}

fn prepare_recorded_file(
    input: RecordedFileInput,
    expected_format: &str,
    context: &GraphInputContext<'_>,
) -> Result<PreparedRecordedInput> {
    ensure!(
        input
            .name
            .as_ref()
            .is_none_or(|name| !name.trim().is_empty()),
        "recorded graph dataset name must be non-empty when present"
    );
    ensure!(
        input.format == expected_format,
        "recorded graph adapter {expected_format:?} received dataset.format={:?}",
        input.format
    );
    ensure!(
        input.path.is_some() ^ input.records.is_some(),
        "recorded graph input requires exactly one of path or records"
    );
    ensure!(
        input.sampling.eq_ignore_ascii_case("sequential"),
        "recorded Graph-IR input requires sequential root selection"
    );
    ensure!(
        input.entries != Some(0),
        "recorded graph root limit must be positive when configured"
    );
    let default_output_tokens = recorded_default_output_tokens(input.osl.as_ref())?;
    let random_seed = input.random_seed;
    let source = match (input.path, input.records) {
        (Some(path), None) => DatasetSource::Path(path),
        (None, Some(records)) => DatasetSource::Inline(records),
        _ => unreachable!("recorded source exclusivity validated"),
    };
    let mut load = LoadConfig::new(source);
    load.options = input.options;
    let synthesis = input.synthesis;
    let allow_dataset_wrap = synthesis
        .as_ref()
        .and_then(|value| value.allow_dataset_wrap)
        .unwrap_or(false);
    let max_context_length = synthesis
        .as_ref()
        .and_then(|value| value.max_context_length)
        .map(usize::try_from)
        .transpose()
        .context("recorded max_context_length exceeds usize")?;
    let max_osl = synthesis
        .as_ref()
        .and_then(|value| value.max_osl)
        .map(|value| value as usize);
    let idle_gap_cap_seconds = synthesis
        .as_ref()
        .map_or(Some(60.0), |value| value.idle_gap_cap_seconds);
    // The trajectory-start window travels to the phase runtime, which samples a
    // per-trace `t*` and applies the warmup/profiling snapshot split. The C1
    // defaults `[0.0, 0.0]` with seed `0` collapse to `t* = 0` (full replay).
    let t_star_window = synthesis
        .as_ref()
        .map_or_else(TStarWindow::default, |value| TStarWindow {
            start_min_ratio: value.trajectory_start_min_ratio,
            start_max_ratio: value.trajectory_start_max_ratio,
            random_seed: value.t_star_random_seed,
            // Legacy `ShuffleSampler`/`RandomSampler` seed their generators from the
            // RUN root (`rng.init(config.random_seed)`), not `t_star_random_seed`;
            // the per-strategy child seed is salted off this root in
            // `TStarWindow::recycle_draw`. When the run root is absent legacy uses
            // `default_rng(None)`/`Random(None)` (non-deterministic); the scenario
            // always threads a seed, so an absent run seed substitutes ONE entropy
            // value HERE (resolved once, shared by every draw site through the `Copy`
            // window — mirroring `content_root_seed` at the site below), keeping the
            // pressure->profiling recycle order internally consistent.
            run_random_seed: context
                .run_random_seed
                .unwrap_or_else(|| RngRoot::new(None).derive_seed_or_entropy("dataset.sampler")),
            sampling_strategy: GraphSamplingStrategy::parse(
                value.dataset_sampling_strategy.as_deref(),
            ),
        });
    let corpus = PromptCorpus::parse(
        synthesis
            .as_ref()
            .and_then(|value| value.corpus.as_deref())
            .unwrap_or("coding"),
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let content_root_seed = context.run_random_seed.unwrap_or_else(|| {
        RngRoot::new(None).derive_seed_or_entropy("dataset.recorded_graph.content")
    });
    let cache_bust_target = CacheBustTarget::parse(
        synthesis
            .as_ref()
            .and_then(|value| value.cache_bust_target.as_deref()),
    );
    Ok(PreparedRecordedInput {
        input: RecordedTraceInputConfig {
            load,
            root_limit: input.entries,
            max_context_length,
            max_osl,
            idle_gap_cap_seconds,
            prompt_corpus: corpus,
            content_root_seed,
        },
        random_seed,
        default_output_tokens,
        allow_dataset_wrap,
        t_star_window,
        cache_bust_target,
    })
}

fn prepare_recorded_public(
    input: PublicDatasetSpec,
    expected_format: &str,
    context: &GraphInputContext<'_>,
) -> Result<PreparedRecordedInput> {
    ensure!(
        !input.name.trim().is_empty(),
        "public WEKA dataset name cannot be empty"
    );
    ensure!(
        input.format == expected_format,
        "recorded graph adapter {expected_format:?} received dataset.format={:?}",
        input.format
    );
    ensure!(
        input.sampling.eq_ignore_ascii_case("sequential"),
        "public WEKA Graph-IR input requires sequential root selection"
    );
    ensure!(
        input.entries != Some(0),
        "public WEKA root limit must be positive"
    );
    let source = match input.source {
        PublicDatasetSourceSpec::Url { url } => DatasetSource::Url(url),
        PublicDatasetSourceSpec::HuggingFace {
            dataset,
            subset,
            split,
            revision,
        } => DatasetSource::HuggingFace {
            dataset,
            config: subset,
            split,
            max_rows: None,
            revision,
        },
    };
    let mut load = LoadConfig::new(source);
    load.options = input.options;
    let option_limit = load
        .options
        .remove("max_conversations")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| anyhow!("public WEKA max_conversations must be positive"))
        })
        .transpose()?;
    let root_limit = input.entries.or(option_limit);
    if let DatasetSource::HuggingFace { max_rows, .. } = &mut load.source {
        *max_rows = root_limit;
    }
    Ok(PreparedRecordedInput {
        input: RecordedTraceInputConfig {
            load,
            root_limit,
            max_context_length: None,
            max_osl: None,
            idle_gap_cap_seconds: Some(60.0),
            prompt_corpus: PromptCorpus::Coding,
            content_root_seed: context.run_random_seed.unwrap_or_else(|| {
                RngRoot::new(None).derive_seed_or_entropy("dataset.recorded_graph.content")
            }),
        },
        random_seed: input.random_seed,
        default_output_tokens: 1,
        allow_dataset_wrap: false,
        // Public recorded sources carry no synthesis block, so the trajectory
        // window defaults to full replay and no cache-bust marker is applied.
        t_star_window: TStarWindow::default(),
        cache_bust_target: CacheBustTarget::None,
    })
}

fn recorded_default_output_tokens(osl: Option<&DistributionSpec>) -> Result<usize> {
    let expected = osl
        .map(distribution)
        .transpose()?
        .map(|value| value.expected_value().ceil())
        .unwrap_or(1.0);
    ensure!(
        expected.is_finite() && expected > 0.0 && expected < usize::MAX as f64,
        "recorded graph dataset.osl expected value is outside usize range"
    );
    Ok(expected as usize)
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum DagJsonlDatasetInput {
    File(DagJsonlFileInput),
    Public(PublicDatasetSpec),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DagJsonlFileInput {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    records: Option<Value>,
    format: String,
    #[serde(default = "default_sequential")]
    sampling: String,
    #[serde(default)]
    synthesis: Option<TraceSynthesisSpec>,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<DistributionSpec>,
    #[serde(default)]
    options: Map<String, Value>,
}

impl From<FileDatasetSpec> for DagJsonlFileInput {
    fn from(spec: FileDatasetSpec) -> Self {
        Self {
            name: None,
            path: spec.path,
            records: spec.records,
            format: spec.format,
            sampling: spec.sampling,
            synthesis: spec.synthesis,
            entries: spec.entries,
            random_seed: spec.random_seed,
            osl: spec.osl,
            options: spec.options,
        }
    }
}

struct PreparedDagJsonlInput {
    input: GraphInputConfig,
    random_seed: Option<u64>,
    default_output_tokens: usize,
}

impl DagJsonlFileInput {
    fn validate(&self, expected_format: &str) -> Result<()> {
        ensure!(
            self.name
                .as_ref()
                .is_none_or(|name| !name.trim().is_empty()),
            "graph dataset name must be non-empty when present"
        );
        ensure!(
            self.format == expected_format,
            "direct graph adapter {expected_format:?} received dataset.format={:?}",
            self.format
        );
        ensure!(
            self.path.is_some() ^ self.records.is_some(),
            "direct graph input requires exactly one of path or records"
        );
        ensure!(
            self.sampling.eq_ignore_ascii_case("sequential"),
            "direct Graph-IR input currently requires sequential root selection"
        );
        ensure!(
            self.synthesis.is_none(),
            "direct Graph-IR input does not accept linear trace synthesis"
        );
        ensure!(
            self.entries != Some(0),
            "direct graph root limit must be positive when configured"
        );
        for name in self.options.keys() {
            ensure!(
                name == "inter_turn_delay_cap_seconds",
                "dag_jsonl Graph-IR input does not support option {name:?}"
            );
        }
        if let Some(delay) = self.options.get("inter_turn_delay_cap_seconds") {
            ensure!(
                delay
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.0),
                "inter_turn_delay_cap_seconds must be finite and non-negative"
            );
        }
        Ok(())
    }

    fn default_output_tokens(&self) -> Result<usize> {
        let expected = self
            .osl
            .as_ref()
            .map(distribution)
            .transpose()?
            .map(|value| value.expected_value().ceil())
            .unwrap_or(1.0);
        ensure!(
            expected.is_finite() && expected > 0.0 && expected < usize::MAX as f64,
            "graph dataset.osl expected value is outside the native usize range"
        );
        Ok(expected as usize)
    }

    fn prepare(self, expected_format: &str) -> Result<PreparedDagJsonlInput> {
        self.validate(expected_format)?;
        let default_output_tokens = self.default_output_tokens()?;
        let random_seed = self.random_seed;
        let source = match (self.path, self.records) {
            (Some(path), None) => DatasetSource::Path(path),
            (None, Some(records)) => DatasetSource::Inline(records),
            _ => unreachable!("source exclusivity validated"),
        };
        let mut load = LoadConfig::new(source);
        load.options = self.options;
        Ok(PreparedDagJsonlInput {
            input: GraphInputConfig {
                load,
                root_limit: self.entries,
            },
            random_seed,
            default_output_tokens,
        })
    }
}

fn prepare_public(spec: PublicDatasetSpec, expected_format: &str) -> Result<PreparedDagJsonlInput> {
    ensure!(
        !spec.name.trim().is_empty(),
        "public graph dataset name cannot be empty"
    );
    ensure!(
        spec.format == expected_format,
        "direct graph adapter {expected_format:?} received dataset.format={:?}",
        spec.format
    );
    ensure!(
        spec.sampling.eq_ignore_ascii_case("sequential"),
        "direct Graph-IR input currently requires sequential root selection"
    );
    ensure!(
        spec.entries != Some(0),
        "direct graph root limit must be positive when configured"
    );
    let option_limit = match spec.options.get("max_conversations") {
        None => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    anyhow!("public graph option max_conversations must be a positive usize")
                })?,
        ),
    };
    let source = match spec.source {
        PublicDatasetSourceSpec::Url { url } => {
            ensure!(!url.trim().is_empty(), "public dataset URL cannot be empty");
            DatasetSource::Url(url)
        }
        PublicDatasetSourceSpec::HuggingFace {
            dataset,
            subset,
            split,
            revision,
        } => DatasetSource::HuggingFace {
            dataset,
            config: subset,
            split,
            // DAG vertices must be acquired as one complete program.
            max_rows: None,
            revision,
        },
    };
    let root_limit = spec.entries.or(option_limit);
    let mut load = LoadConfig::new(source);
    load.options = spec.options;
    load.options.remove("max_conversations");
    Ok(PreparedDagJsonlInput {
        input: GraphInputConfig { load, root_limit },
        random_seed: spec.random_seed,
        default_output_tokens: 1,
    })
}

fn default_sequential() -> String {
    "sequential".into()
}

#[cfg(test)]
mod tests {
    use crate::dataset::TiktokenTokenizer;
    use serde_json::json;

    use super::*;

    fn raw(value: Value) -> Box<RawValue> {
        serde_json::value::to_raw_value(&value).unwrap()
    }

    #[test]
    fn identity_decode_skips_adapter_owned_fields_without_retaining_them() {
        assert_eq!(
            std::mem::size_of::<GraphInputIdentity>(),
            std::mem::size_of::<String>(),
            "the selector DTO must retain only its discriminator"
        );
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        resolver
            .validate_identity(&raw(json!({
                "type": "file",
                "format": "dag_jsonl",
                "future_adapter_field": {
                    "opaque": "x".repeat(1 << 20),
                    "nested": [{"owned_by": "selected adapter"}]
                }
            })))
            .unwrap();
    }

    #[tokio::test]
    async fn selected_adapter_owns_the_only_strict_decode_and_direct_load() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dag_jsonl",
            "sampling": "sequential",
            "records": [{
                "session_id": "root",
                "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
            }],
            "osl": {"value": 3.0},
            "options": {"inter_turn_delay_cap_seconds": 0.5}
        }));
        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .unwrap();

        assert_eq!(prepared.bundle.metadata.format, "dag_jsonl");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(prepared.default_output_tokens, 3);
    }

    #[tokio::test]
    async fn selected_adapter_rejects_unknown_full_shape_fields() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dag_jsonl",
            "records": [],
            "future_adapter_field": true
        }));
        let error = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .err()
            .expect("unknown adapter fields must fail");
        assert!(format!("{error:#}").contains("future_adapter_field"));
    }

    #[tokio::test]
    async fn weka_adapter_strictly_decodes_and_compiles_directly_to_graph_ir() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "weka_trace",
            "sampling": "sequential",
            "records": {
                "id": "root",
                "models": ["m"],
                "block_size": 16,
                "hash_id_scope": "global",
                "requests": [{
                    "t": 0,
                    "type": "n",
                    "model": "m",
                    "in": 16,
                    "out": 7,
                    "hash_ids": [1]
                }]
            },
            "random_seed": 91,
            "synthesis": {
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "max_osl": 3,
                "allow_dataset_wrap": true,
                "idle_gap_cap_seconds": 60.0,
                "corpus": "sonnet"
            }
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .expect("direct WEKA compiler");
        assert_eq!(prepared.bundle.metadata.format, "weka_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(
            prepared.bundle.plans[0].graph.nodes["root:0"].max_tokens,
            Some(3)
        );
        assert_eq!(prepared.random_seed, Some(91));
        assert!(prepared.allow_dataset_wrap);
    }

    #[tokio::test]
    async fn aiperf_trace_adapter_strictly_decodes_and_compiles_directly_to_graph_ir() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "aiperf_trace",
            "records": {
                "schema": "aiperf.trace.v1",
                "session_id": 7,
                "block_size": 16,
                "segments": [
                    {"role": "user", "kind": ["text"], "hash_ids": [1], "tokens": 16}
                ],
                "inference_calls": [
                    {"ts": 0.0, "segment_refs": [0], "usage": {"output_tokens": 3}}
                ]
            },
            "random_seed": 91
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .expect("direct aiperf_trace compiler");
        assert_eq!(prepared.bundle.metadata.format, "aiperf_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert!(prepared.bundle.plans[0].graph.nodes.contains_key("7:0"));
        assert_eq!(prepared.random_seed, Some(91));
    }

    #[test]
    fn recorded_request_carries_tstar_window_on_synthesis() {
        // A protocol-v2 recorded-graph dataset request carries the trajectory
        // start (t*) window and derived seed on its synthesis block; the strict
        // recorded decode retains them for C2 to bind.
        let RecordedDatasetInput::File(input) = serde_json::from_value(json!({
            "type": "file",
            "format": "weka_trace",
            "sampling": "sequential",
            "records": {"id": "root", "models": ["m"], "block_size": 16,
                "hash_id_scope": "global", "requests": []},
            "synthesis": {
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "trajectory_start_min_ratio": 0.1,
                "trajectory_start_max_ratio": 0.9,
                "t_star_random_seed": 777
            }
        }))
        .expect("recorded request with t* synthesis knobs decodes") else {
            panic!("expected a file-backed recorded input")
        };
        let synthesis = input.synthesis.expect("synthesis block present");
        assert_eq!(synthesis.trajectory_start_min_ratio, 0.1);
        assert_eq!(synthesis.trajectory_start_max_ratio, 0.9);
        assert_eq!(synthesis.t_star_random_seed, 777);
    }

    #[tokio::test]
    async fn dynamo_adapter_rejects_inline_records_before_source_acquisition() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dynamo_trace",
            "sampling": "sequential",
            "records": []
        }));
        let error = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .expect_err("product Dynamo input must remain path-backed");
        assert!(
            format!("{error:#}").contains("requires a file, directory, or segmented-prefix path")
        );
    }

    #[tokio::test]
    async fn dynamo_adapter_streams_a_path_and_compiles_directly_to_graph_ir() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("trace.jsonl");
        std::fs::write(
            &path,
            serde_json::to_vec(&json!({
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 1,
                "agent_context": {"session_id": "root"},
                "request": {
                    "request_id": "r", "model": "m", "input_tokens": 16,
                    "output_tokens": 1, "replay": {
                        "trace_block_size": 16, "input_length": 16,
                        "input_sequence_hashes": [7]
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "dynamo_trace",
            "sampling": "sequential",
            "path": path,
            "synthesis": {
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "corpus": "sonnet"
            }
        }));
        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                },
            )
            .await
            .expect("direct Dynamo compiler");
        assert_eq!(prepared.bundle.metadata.format, "dynamo_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(prepared.bundle.plans[0].trace.id, "root");
    }
}
