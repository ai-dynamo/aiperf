// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned direct Graph-IR input adapters.
//!
//! The resolver reads only the format identity; the selected adapter owns strict
//! decoding and lowers directly to [`GraphInputBundle`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::model::dataset::{RecordedAgentGraphConfig, RecordedAgentSourceFormat};
use crate::dataset::{DatasetSource, LoadConfig, TextTokenizer};
use crate::graph::conditional::compile_conditional_graph_input;
use crate::graph::input::{GraphInputBundle, GraphInputConfig, compile_dag_jsonl_input};
use crate::graph::recorded::agent_recording::{
    BuiltinReplayRequestProfileResolver, RecordedAgentInputSource,
    discover_imported_agent_read_set, discover_recorded_agent_input, lower_imported_agent_sessions,
    lower_recorded_agent_corpus, parse_imported_agent_sessions, resolve_recorded_environment,
};
use crate::graph::recorded::{
    PromptCorpus, RecordedTraceInputConfig, compile_aiperf_trace_input, compile_dynamo_trace_input,
    compile_weka_trace_input,
};
use crate::graph::segment::SegmentPool;
use crate::graph::supplement::PlannedReplayTraceInstance;
use crate::graph::tools::{
    PinchWorkspaceStager, ToolExecutionBackend, WorkspaceEntrySource, WorkspaceTreeStager,
};
use crate::graph::tstar::{
    PermutationDraw, RecycleDrawMode, sampler_random_seed, sampler_shuffle_seed,
};
use crate::rng::RngRoot;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};

use crate::config::model::workload_kind::GRAPH_FORMATS;
use crate::engine::dataset_input::DatasetCacheBustSpec;
use crate::engine::execute::distribution;
use crate::engine::protocol::{
    DistributionSpec, FileDatasetSpec, PhaseSpec, PromptSelectionSpec, PublicDatasetSourceSpec,
    PublicDatasetSpec, TraceSynthesisSpec,
};

mod otlp_genai;
use otlp_genai::OtlpGenaiRunnerGraphInputAdapter;

/// Recorded-graph trajectory-start (`t*`) window bound to a prepared input.
///
/// A `[0.0, 0.0]` window yields full replay with `t* = 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TStarWindow {
    /// Lower window bound as a fraction of each trace's replayable span.
    pub start_min_ratio: f64,
    /// Upper window bound as a fraction of each trace's replayable span.
    pub start_max_ratio: f64,
    /// Base `t*` RNG seed salted per trace and lane, distinct from the
    /// dataset-sampler run root.
    pub random_seed: u64,
    /// Run-root seed used to derive strategy-specific recycle generators.
    pub run_random_seed: u64,
    /// Strategy selecting the next corpus template for a freed recycle lane.
    pub sampling_strategy: GraphSamplingStrategy,
}

impl TStarWindow {
    /// Build the resolved recycle-index draw for this window's strategy.
    ///
    /// The child generator seed is salted off [`TStarWindow::run_random_seed`] per
    /// strategy, so `Sequential`/`Shuffle`/`RandomSampler` all derive from the SAME
    /// run root with different salts. Building a fresh draw at each site is safe:
    /// the draw is a pure function of `(mode, child_seed)`, so pressure warmup
    /// and the profiling recycle (built independently) agree draw-for-draw.
    pub fn recycle_draw(&self) -> PermutationDraw {
        match self.sampling_strategy.draw_mode() {
            RecycleDrawMode::Sequential => PermutationDraw::sequential(),
            RecycleDrawMode::Shuffle => {
                PermutationDraw::shuffle(sampler_shuffle_seed(self.run_random_seed))
            }
            RecycleDrawMode::Random => {
                PermutationDraw::random(sampler_random_seed(self.run_random_seed))
            }
        }
    }
}

/// Resolved dataset-sampling strategy for the recorded-graph recycle draw.
///
/// `Sequential`, `Shuffle`, and `Random` preserve the configured sampler's
/// byte-exact draw sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GraphSamplingStrategy {
    /// Cursor-with-wrap draw (`x % total`).
    #[default]
    Sequential,
    /// Persistent-epoch shuffle without replacement.
    Shuffle,
    /// CPython MT19937 `choice` draw with replacement.
    Random,
}

impl GraphSamplingStrategy {
    /// Parse a wire strategy, selecting sequential behavior for unknown values.
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some("shuffle") => Self::Shuffle,
            Some("random") => Self::Random,
            _ => Self::Sequential,
        }
    }

    /// Map the wire strategy to its byte-exact draw mode.
    pub fn draw_mode(self) -> RecycleDrawMode {
        match self {
            Self::Sequential => RecycleDrawMode::Sequential,
            Self::Shuffle => RecycleDrawMode::Shuffle,
            Self::Random => RecycleDrawMode::Random,
        }
    }
}

/// Recorded-graph cache-bust marker target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheBustTarget {
    /// No marker; recorded content is sent verbatim.
    #[default]
    None,
    /// Prepend a per-conversation `[rid:<digest>]\n\n` marker to the first user
    /// message of every request.
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
            .field("plans", &self.bundle.programs.len())
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
    /// Selected default endpoint factory identifier for format compatibility checks.
    pub endpoint_id: &'a str,
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
    /// Registered graph-input format identifiers in their shared authored order.
    fn supported_formats(&self) -> Vec<&str>;

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
        let adapters: [Arc<dyn GraphInputAdapter>; 7] = [
            Arc::new(DagJsonlRunnerGraphInputAdapter),
            Arc::new(ConditionalGraphRunnerGraphInputAdapter),
            Arc::new(WekaTraceRunnerGraphInputAdapter),
            Arc::new(DynamoTraceRunnerGraphInputAdapter),
            Arc::new(AIPerfTraceRunnerGraphInputAdapter),
            Arc::new(RecordedAgentRunnerGraphInputAdapter),
            Arc::new(OtlpGenaiRunnerGraphInputAdapter),
        ];
        debug_assert_eq!(
            adapters
                .iter()
                .map(|adapter| adapter.format())
                .collect::<BTreeSet<_>>(),
            GRAPH_FORMATS.iter().copied().collect::<BTreeSet<_>>(),
            "built-in graph adapters must match the shared graph-format inventory"
        );
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

/// Decode graph input from raw JSON without `arbitrary_precision`.
fn decode_graph_input<T>(raw: &RawValue) -> serde_json::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(raw.get())
}

#[async_trait(?Send)]
impl GraphInputAdapterResolver for BuiltinRunnerGraphInputAdapterResolver {
    fn supported_formats(&self) -> Vec<&str> {
        GRAPH_FORMATS
            .iter()
            .filter_map(|format| {
                debug_assert!(
                    self.adapters.contains_key(format),
                    "shared graph format {format:?} has no built-in adapter"
                );
                self.adapters.contains_key(format).then_some(*format)
            })
            .collect()
    }

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

/// Load one local graph-inspection input through the selected resolver once.
///
/// This helper only constructs the canonical file dataset wire shape. It leaves
/// strict decoding and all source reads to the resolver's single `load` call.
pub async fn prepare_local_graph_inspection_input(
    resolver: &dyn GraphInputAdapterResolver,
    source: &Path,
    format: &str,
    tokenizer: &dyn TextTokenizer,
    seed: u64,
) -> Result<PreparedRunnerGraphInput> {
    let raw = serde_json::value::to_raw_value(&serde_json::json!({
        "type": "file",
        "format": format,
        "path": source,
        "sampling": "sequential",
    }))?;
    resolver
        .load(
            &raw,
            &GraphInputContext {
                tokenizer,
                run_random_seed: Some(seed),
                endpoint_id: "chat",
            },
        )
        .await
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
            !bundle.programs.is_empty(),
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

/// Built-in authored conditional-graph adapter: model-independent branching plus
/// recorded replay content folded into the flat Graph-IR at lowering.
#[derive(Debug)]
pub struct ConditionalGraphRunnerGraphInputAdapter;

#[async_trait(?Send)]
impl GraphInputAdapter for ConditionalGraphRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "conditional_graph"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        // Authored conditional graphs share the generic authored-graph file
        // envelope (`path`/`records`, sequential selection) with `dag_jsonl`.
        let input: DagJsonlDatasetInput =
            decode_graph_input(raw).context("decoding direct conditional_graph input")?;
        let prepared = match input {
            DagJsonlDatasetInput::File(spec) => spec.prepare(self.format())?,
            DagJsonlDatasetInput::Public(spec) => prepare_public(spec, self.format())?,
        };
        let workload_seed = context.run_random_seed.unwrap_or(0);
        let bundle =
            compile_conditional_graph_input(prepared.input, context.tokenizer, workload_seed)
                .await
                .map_err(|error| anyhow!(error.to_string()))
                .context("loading and lowering direct authored conditional_graph input")?;
        ensure!(
            !bundle.programs.is_empty(),
            "authored conditional_graph input contains no traces after root limiting"
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
            // Authored programs supply verbatim content and carry no recorded
            // timing, so the trajectory window defaults to full profiling and no
            // cache-bust marker is applied.
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

/// Built-in native Mini-SWE-Agent recording adapter.
#[derive(Debug)]
pub struct RecordedAgentRunnerGraphInputAdapter;

#[async_trait(?Send)]
impl GraphInputAdapter for RecordedAgentRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "agent_recording"
    }

    async fn load(
        &self,
        raw: &RawValue,
        context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: RecordedAgentDatasetInput =
            decode_graph_input(raw).context("decoding direct agent_recording graph input")?;
        input.prepare(self.format(), context.endpoint_id)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordedAgentDatasetInput {
    #[serde(rename = "type")]
    input_type: String,
    format: String,
    path: PathBuf,
    #[serde(default = "default_agent_recording_sampling")]
    sampling: String,
    #[serde(default)]
    options: Map<String, Value>,
    /// Recorded-agent replay policy projected from `dataset.graph`.
    #[serde(default)]
    graph: Option<RecordedAgentGraphConfig>,
    #[serde(default)]
    replay_root: Option<PathBuf>,
    #[serde(default)]
    execute_tools: bool,
    #[serde(default)]
    use_recorded_model: bool,
    #[serde(default)]
    use_recorded_sampling: bool,
    #[serde(default = "default_true")]
    streaming: bool,
    #[serde(default)]
    fallback_max_tokens: Option<usize>,
    #[serde(default)]
    standard_scenario: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinchTaskPackManifest {
    tasks: Vec<PinchTaskPackEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinchTaskPackEntry {
    task_id: String,
    path: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinchTaskFile {
    #[serde(default, rename = "id")]
    _id: Option<String>,
    #[serde(default, rename = "name")]
    _name: Option<String>,
    #[serde(default, rename = "category")]
    _category: Option<String>,
    #[serde(default, rename = "grading_type")]
    _grading_type: Option<String>,
    #[serde(default, rename = "timeout_seconds")]
    _timeout_seconds: Option<u64>,
    workspace_files: Vec<PinchWorkspaceFile>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PinchWorkspaceFile {
    Literal(PinchWorkspaceLiteral),
    Asset(PinchWorkspaceAsset),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinchWorkspaceLiteral {
    path: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinchWorkspaceAsset {
    source: String,
    dest: String,
}

fn default_agent_recording_sampling() -> String {
    "sequential".to_string()
}

impl RecordedAgentDatasetInput {
    fn prepare(self, expected_format: &str, endpoint_id: &str) -> Result<PreparedRunnerGraphInput> {
        ensure!(
            self.input_type == "file",
            "agent_recording graph input requires type=file"
        );
        ensure!(
            self.format == expected_format,
            "recorded-agent adapter {expected_format:?} received dataset.format={:?}",
            self.format
        );
        ensure!(
            self.sampling == "sequential",
            "agent_recording requires dataset.sampling=sequential"
        );
        ensure!(
            self.options.is_empty(),
            "agent_recording rejects unsupported dataset.options keys"
        );
        let fallback_max_tokens = self.fallback_max_tokens.unwrap_or(32_768);
        ensure!(
            fallback_max_tokens > 0,
            "agent_recording fallback_max_tokens must be positive"
        );
        let replay_root = self
            .graph
            .as_ref()
            .and_then(|graph| graph.replay_root.as_deref())
            .or(self.replay_root.as_deref());
        let execute_tools = self
            .graph
            .as_ref()
            .map_or(self.execute_tools, |graph| graph.execute_tools);
        let source_format = self
            .graph
            .as_ref()
            .map_or(RecordedAgentSourceFormat::Auto, |graph| graph.source_format);
        let include_subagents = self
            .graph
            .as_ref()
            .and_then(|graph| graph.include_subagents);
        let resolved = resolve_recorded_agent_graph_source(
            &self.path,
            replay_root,
            source_format,
            include_subagents,
        )?;
        if matches!(
            &resolved,
            ResolvedRecordedAgentGraphSource::Imported {
                source: crate::graph::recorded::agent_recording::ImportedAgentSource::ClaudeCode,
                ..
            }
        ) {
            ensure!(
                endpoint_id == "messages",
                "Claude Code imported session replay requires endpoint type messages"
            );
        }
        let resolver = BuiltinReplayRequestProfileResolver::new(
            self.streaming,
            fallback_max_tokens,
            execute_tools,
            self.use_recorded_model,
            self.use_recorded_sampling,
            self.standard_scenario,
        )
        .map_err(|error| anyhow!(error.to_string()))?;
        let mut pool = SegmentPool::new();
        let (mut bundle, strict_corpus) = match resolved {
            ResolvedRecordedAgentGraphSource::Strict(corpus) => {
                let bundle = lower_recorded_agent_corpus(&corpus, &resolver, &mut pool)
                    .map_err(|error| anyhow!(error.to_string()))
                    .context("lowering recorded-agent graph input")?;
                (bundle, Some(corpus))
            }
            ResolvedRecordedAgentGraphSource::Imported { sessions, .. } => {
                let bundle = lower_imported_agent_sessions(&sessions, &resolver, &mut pool)
                    .map_err(|error| anyhow!(error.to_string()))
                    .context("lowering imported recorded-agent session input")?;
                (bundle, None)
            }
        };
        if execute_tools {
            let corpus = strict_corpus
                .as_ref()
                .ok_or_else(|| anyhow!("imported recorded-agent sessions reject execute_tools"))?;
            let pinch_image = self
                .graph
                .as_ref()
                .and_then(|graph| graph.pinch_image.as_deref())
                .unwrap_or_default();
            let tool_image = self
                .graph
                .as_ref()
                .and_then(|graph| graph.tool_image.as_deref());
            for (program, trace) in bundle.programs.iter_mut().zip(&corpus.traces) {
                let identity = program
                    .replay
                    .as_ref()
                    .map(|replay| &replay.identity)
                    .ok_or_else(|| anyhow!("recorded replay program has no task identity"))?;
                let mut environment = resolve_recorded_environment(
                    identity,
                    &trace.recording.metadata,
                    pinch_image,
                    tool_image,
                    self.standard_scenario,
                )
                .map_err(|error| anyhow!(error.to_string()))
                .with_context(|| {
                    format!(
                        "resolving recorded-agent environment for {:?}",
                        identity.task_id
                    )
                })?;
                if identity.adapter == "pinchbench" {
                    let replay_root = replay_root.ok_or_else(|| {
                        anyhow!(
                            "PinchBench task {:?} requires dataset.graph.replay_root",
                            identity.task_id
                        )
                    })?;
                    environment.workspace = stage_pinch_task_workspace(
                        replay_root,
                        &trace.recording.metadata,
                        &identity.task_id,
                        &mut pool,
                    )?;
                } else if identity.adapter == "swebench"
                    && environment.backend == ToolExecutionBackend::Local
                {
                    let replay_root = replay_root.ok_or_else(|| {
                        anyhow!(
                            "local SWE-Bench task {:?} requires dataset.graph.replay_root",
                            identity.task_id
                        )
                    })?;
                    environment.workspace =
                        WorkspaceTreeStager::new(&replay_root.join("testbed"), &mut pool)
                            .stage(
                                "/testbed",
                                vec!["bash".into(), "-c".into()],
                                environment.workspace.command_timeout_ns,
                            )
                            .map_err(|error| anyhow!(error.to_string()))
                            .with_context(|| {
                                format!(
                                    "staging local SWE-Bench workspace for {:?}",
                                    identity.task_id
                                )
                            })?;
                }
                program.environment = Some(
                    crate::graph::driver::TraceEnvironmentSpec::from_resolved(&environment)
                        .map_err(|error| anyhow!(error.to_string()))?,
                );
            }
            bundle.segments = Arc::new(pool.clone().freeze());
        }
        ensure!(
            !bundle.programs.is_empty(),
            "recorded-agent Graph-IR input contains no root traces"
        );
        ensure!(
            bundle.metadata.format == expected_format,
            "recorded-agent adapter {expected_format:?} returned bundle format {:?}",
            bundle.metadata.format
        );
        for warning in &bundle.metadata.warning_facts {
            tracing::warn!(
                warning_code = %warning.code,
                warning_context = ?warning.context,
                "recorded-agent graph input warning"
            );
        }
        Ok(PreparedRunnerGraphInput {
            bundle,
            random_seed: None,
            default_output_tokens: fallback_max_tokens,
            allow_dataset_wrap: false,
            t_star_window: TStarWindow::default(),
            cache_bust_target: CacheBustTarget::None,
        })
    }
}

fn stage_pinch_task_workspace(
    replay_root: &std::path::Path,
    metadata: &crate::graph::recorded::agent_recording::RecordedAgentMetadata,
    task_id: &str,
    pool: &mut SegmentPool,
) -> Result<crate::graph::tools::WorkspaceSpec> {
    let replay_root = replay_root
        .canonicalize()
        .context("canonicalizing recorded-agent replay root for Pinch task pack")?;
    let manifest = metadata.manifest.as_deref().ok_or_else(|| {
        anyhow!("PinchBench task {task_id:?} recording metadata has no task-pack manifest")
    })?;
    let manifest = manifest
        .strip_prefix("<open-lab-root>/")
        .unwrap_or(manifest);
    let manifest_path =
        rooted_existing_path(&replay_root, std::path::Path::new(manifest), "manifest")?;
    let manifest_wire = fs::read(&manifest_path)
        .with_context(|| format!("reading Pinch task-pack manifest {manifest_path:?}"))?;
    let manifest: PinchTaskPackManifest = serde_yaml::from_slice(&manifest_wire)
        .with_context(|| format!("decoding Pinch task-pack manifest {manifest_path:?}"))?;
    let mut matches = manifest.tasks.iter().filter(|task| task.task_id == task_id);
    let task = matches
        .next()
        .ok_or_else(|| anyhow!("Pinch task-pack manifest has no task {task_id:?}"))?;
    ensure!(
        matches.next().is_none(),
        "Pinch task-pack manifest contains duplicate task {task_id:?}"
    );
    let task_pack_root = manifest_path
        .parent()
        .ok_or_else(|| anyhow!("Pinch task-pack manifest has no parent directory"))?;
    let task_path = rooted_existing_path(task_pack_root, &task.path, "task file")?;
    let task_wire =
        fs::read(&task_path).with_context(|| format!("reading Pinch task file {task_path:?}"))?;
    let task = decode_pinch_task_file(&task_wire)
        .with_context(|| format!("decoding Pinch task file {task_path:?}"))?;
    let entries = task.workspace_files.into_iter().map(|entry| match entry {
        PinchWorkspaceFile::Literal(entry) => {
            WorkspaceEntrySource::literal(entry.path, entry.content)
        }
        PinchWorkspaceFile::Asset(entry) => {
            let source = if entry.source == "assets" || entry.source.starts_with("assets/") {
                entry.source
            } else {
                format!("assets/{}", entry.source)
            };
            WorkspaceEntrySource::asset(source, entry.dest)
        }
    });
    PinchWorkspaceStager::new(task_pack_root, pool)
        .stage(entries)
        .map_err(|error| anyhow!(error.to_string()))
}

fn decode_pinch_task_file(wire: &[u8]) -> Result<PinchTaskFile> {
    let text = std::str::from_utf8(wire).context("Pinch task file is not UTF-8")?;
    let yaml = if let Some(frontmatter) = text.strip_prefix("---\n") {
        frontmatter
            .split_once("\n---")
            .map(|(yaml, _)| yaml)
            .ok_or_else(|| anyhow!("Pinch task Markdown has no closing YAML frontmatter fence"))?
    } else {
        text
    };
    serde_yaml::from_str(yaml).context("decoding Pinch task workspace frontmatter")
}

fn rooted_existing_path(
    root: &std::path::Path,
    relative: &std::path::Path,
    label: &str,
) -> Result<PathBuf> {
    ensure!(
        !relative.is_absolute()
            && relative.components().all(|component| matches!(
                component,
                std::path::Component::Normal(_) | std::path::Component::CurDir
            )),
        "Pinch task-pack {label} path {relative:?} is not root-contained"
    );
    let resolved = root
        .join(relative)
        .canonicalize()
        .with_context(|| format!("canonicalizing Pinch task-pack {label} {relative:?}"))?;
    ensure!(
        resolved.starts_with(root),
        "Pinch task-pack {label} path {relative:?} escapes its root"
    );
    Ok(resolved)
}

/// Enumerate the finite recorded-agent profiling assignments a controller gives one cell.
///
/// The plan is built before cells receive START. Runtime `run_id` values are intentionally
/// unavailable here; the resulting identity is instead the resolved trace instance plus its
/// controller-selected cell. Duration-bounded replay is refused because its terminal set is
/// not knowable before execution.
pub fn plan_recorded_agent_cell_assignments(
    dataset: &Value,
    phases: &[PhaseSpec],
    cell_id: u32,
    cell_count: u32,
) -> Result<BTreeSet<PlannedReplayTraceInstance>> {
    let input: RecordedAgentDatasetInput = serde_json::from_value(dataset.clone())
        .context("decoding recorded-agent input for cellular assignment planning")?;
    if input.format != "agent_recording" {
        return Ok(BTreeSet::new());
    }
    ensure!(
        cell_count > 0 && cell_id < cell_count,
        "invalid cellular graph assignment"
    );
    let prepared = input.prepare("agent_recording", "chat")?;
    let profiling = phases
        .iter()
        .filter(|phase| !phase.common().exclude_from_results)
        .collect::<Vec<_>>();
    ensure!(
        profiling.len() <= 1,
        "cellular recorded-agent replay requires exactly one profiling phase"
    );
    let Some(phase) = profiling.first() else {
        return Ok(BTreeSet::new());
    };
    let common = phase.common();
    ensure!(
        common.requests.is_none(),
        "cellular recorded-agent replay cannot plan a static-node requests budget"
    );
    ensure!(
        common.duration.is_none(),
        "cellular recorded-agent replay requires a finite sessions budget so the controller can author expected trace identities"
    );
    let session_limit = common
        .sessions
        .unwrap_or(prepared.bundle.programs.len() as u64);
    let templates = prepared
        .bundle
        .programs
        .iter()
        .map(|program| program.profiling.trace.id.as_str())
        .collect::<Vec<_>>();
    ensure!(
        !templates.is_empty(),
        "recorded-agent graph input contains no root traces"
    );
    let mut planned = BTreeSet::new();
    let mut ordinal = u64::from(cell_id);
    while ordinal < session_limit {
        let template = templates[ordinal as usize % templates.len()];
        let trace_id = format!("{template}::instance-{ordinal}");
        planned.insert(
            PlannedReplayTraceInstance::new(cell_id, format!("{trace_id}::trajectory"), trace_id)
                .with_template_trace_id(template),
        );
        ordinal = ordinal
            .checked_add(u64::from(cell_count))
            .ok_or_else(|| anyhow!("cellular recorded-agent assignment ordinal overflow"))?;
    }
    Ok(planned)
}

enum ResolvedRecordedAgentGraphSource {
    Strict(crate::graph::recorded::agent_recording::ValidatedRecordedAgentCorpus),
    Imported {
        source: crate::graph::recorded::agent_recording::ImportedAgentSource,
        sessions: Vec<crate::graph::recorded::agent_recording::ImportedAgentSession>,
    },
}

fn resolve_recorded_agent_graph_source(
    path: &PathBuf,
    replay_root: Option<&std::path::Path>,
    source_format: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
) -> Result<ResolvedRecordedAgentGraphSource> {
    if source_format == RecordedAgentSourceFormat::MiniSweAgent {
        return strict_recorded_agent_graph_source(path, replay_root);
    }
    if matches!(
        source_format,
        RecordedAgentSourceFormat::Codex | RecordedAgentSourceFormat::ClaudeCode
    ) {
        return imported_recorded_agent_graph_source(
            path,
            replay_root,
            source_format,
            include_subagents,
        );
    }
    let candidate = replay_root.map_or_else(|| path.clone(), |root| root.join(path));
    match strict_recorded_agent_graph_source(path, replay_root) {
        Ok(strict) => Ok(strict),
        Err(_)
            if candidate
                .extension()
                .is_some_and(|extension| extension == "jsonl") =>
        {
            imported_recorded_agent_graph_source(
                path,
                replay_root,
                RecordedAgentSourceFormat::Auto,
                include_subagents,
            )
        }
        Err(error) => Err(error),
    }
}

fn strict_recorded_agent_graph_source(
    path: &PathBuf,
    replay_root: Option<&std::path::Path>,
) -> Result<ResolvedRecordedAgentGraphSource> {
    let source = recorded_agent_source(path, replay_root)?;
    let corpus = discover_recorded_agent_input(replay_root, source)
        .map_err(|error| anyhow!(error.to_string()))
        .context("discovering strict recorded-agent graph input")?;
    Ok(ResolvedRecordedAgentGraphSource::Strict(corpus))
}

fn imported_recorded_agent_graph_source(
    path: &PathBuf,
    replay_root: Option<&std::path::Path>,
    source_format: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
) -> Result<ResolvedRecordedAgentGraphSource> {
    let read_set =
        discover_imported_agent_read_set(path, replay_root, source_format, include_subagents)
            .map_err(|error| anyhow!(error.to_string()))
            .context("discovering imported recorded-agent session input")?;
    let source = read_set.source;
    let sessions = parse_imported_agent_sessions(&read_set)
        .map_err(|error| anyhow!(error.to_string()))
        .context("parsing imported recorded-agent session input")?;
    Ok(ResolvedRecordedAgentGraphSource::Imported { source, sessions })
}

fn recorded_agent_source(
    path: &PathBuf,
    replay_root: Option<&std::path::Path>,
) -> Result<RecordedAgentInputSource> {
    let candidate = replay_root.map_or_else(|| path.clone(), |root| root.join(path));
    let metadata = fs::metadata(&candidate)
        .with_context(|| format!("reading recorded-agent input {}", candidate.display()))?;
    if metadata.is_dir() {
        return Ok(RecordedAgentInputSource::Directory(path.clone()));
    }
    if candidate
        .extension()
        .is_some_and(|extension| extension == "gz")
    {
        return Ok(RecordedAgentInputSource::Recording(path.clone()));
    }
    let bytes = fs::read(&candidate)
        .with_context(|| format!("reading recorded-agent input {}", candidate.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("decoding recorded-agent input {}", candidate.display()))?;
    let is_recording = value
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|format| format.starts_with("mini-swe-agent-recording-"));
    Ok(if is_recording {
        RecordedAgentInputSource::Recording(path.clone())
    } else {
        RecordedAgentInputSource::Manifest(path.clone())
    })
}

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
    cache_bust: Option<DatasetCacheBustSpec>,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<DistributionSpec>,
    #[serde(default)]
    prompts: Option<PromptSelectionSpec>,
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
            cache_bust: spec.cache_bust,
            entries: spec.entries,
            random_seed: spec.random_seed,
            osl: spec.osl,
            prompts: spec.prompts,
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
        !bundle.programs.is_empty(),
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
    let cache_bust = input.cache_bust;
    let prompts = input.prompts;
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
    // Absent synthesis → no idle-gap warp (raw recorded gaps). A present
    // TraceSynthesisSpec still defaults its field to 60s when the key is omitted.
    let idle_gap_cap_seconds = synthesis
        .as_ref()
        .and_then(|value| value.idle_gap_cap_seconds);
    // The default window collapses to `t* = 0` and full replay.
    let t_star_window = synthesis
        .as_ref()
        .map_or_else(TStarWindow::default, |value| TStarWindow {
            start_min_ratio: value.trajectory_start_min_ratio,
            start_max_ratio: value.trajectory_start_max_ratio,
            random_seed: value.t_star_random_seed,
            // Resolve absent run seeds once so every recycle site shares one order.
            run_random_seed: context
                .run_random_seed
                .unwrap_or_else(|| RngRoot::new(None).derive_seed_or_entropy("dataset.sampler")),
            sampling_strategy: GraphSamplingStrategy::parse(
                value.dataset_sampling_strategy.as_deref(),
            ),
        });
    let corpus = PromptCorpus::parse(
        prompts
            .as_ref()
            .and_then(|value| value.corpus.as_deref())
            .unwrap_or("coding"),
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    let content_root_seed = context.run_random_seed.unwrap_or_else(|| {
        RngRoot::new(None).derive_seed_or_entropy("dataset.recorded_graph.content")
    });
    // `synthesis.cache_bust_target` is the recorded-graph spelling; the
    // dataset-level `cache_bust` block is the equivalent authored form.
    let cache_bust_target = CacheBustTarget::parse(
        synthesis
            .as_ref()
            .and_then(|value| value.cache_bust_target.as_deref())
            .or_else(|| {
                cache_bust
                    .as_ref()
                    .and_then(|value| value.target.as_deref())
            }),
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
    // A synthesis block (authored via `--synthesis-*` or materialized by an
    // agentic scenario lock) carries the trajectory-start (`t*`) window, idle-gap
    // cap, and cache-bust target. Public recorded sources historically ignored it
    // and always full-replayed; thread it here so `--public-dataset` applies the
    // same snapshot the file path does (else the scenario's t* warmup/profiling
    // clip and cache-bust silently vanish for the live HF corpora).
    let synthesis = input.synthesis;
    let cache_bust = input.cache_bust;
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
    // Absent synthesis → no idle-gap warp (raw recorded gaps). A present
    // TraceSynthesisSpec still defaults its field to 60s when the key is omitted.
    let idle_gap_cap_seconds = synthesis
        .as_ref()
        .and_then(|value| value.idle_gap_cap_seconds);
    let t_star_window = synthesis
        .as_ref()
        .map_or_else(TStarWindow::default, |value| TStarWindow {
            start_min_ratio: value.trajectory_start_min_ratio,
            start_max_ratio: value.trajectory_start_max_ratio,
            random_seed: value.t_star_random_seed,
            run_random_seed: context
                .run_random_seed
                .unwrap_or_else(|| RngRoot::new(None).derive_seed_or_entropy("dataset.sampler")),
            sampling_strategy: GraphSamplingStrategy::parse(
                value.dataset_sampling_strategy.as_deref(),
            ),
        });
    // `synthesis.cache_bust_target` is the recorded-graph spelling; the
    // dataset-level `cache_bust` block is the equivalent authored form.
    let cache_bust_target = CacheBustTarget::parse(
        synthesis
            .as_ref()
            .and_then(|value| value.cache_bust_target.as_deref())
            .or_else(|| {
                cache_bust
                    .as_ref()
                    .and_then(|value| value.target.as_deref())
            }),
    );
    Ok(PreparedRecordedInput {
        input: RecordedTraceInputConfig {
            load,
            root_limit,
            max_context_length,
            max_osl,
            idle_gap_cap_seconds,
            prompt_corpus: PromptCorpus::parse(
                input
                    .prompts
                    .as_ref()
                    .and_then(|value| value.corpus.as_deref())
                    .unwrap_or("coding"),
            )
            .map_err(|error| anyhow!(error.to_string()))?,
            content_root_seed: context.run_random_seed.unwrap_or_else(|| {
                RngRoot::new(None).derive_seed_or_entropy("dataset.recorded_graph.content")
            }),
        },
        random_seed: input.random_seed,
        default_output_tokens: 1,
        allow_dataset_wrap,
        t_star_window,
        cache_bust_target,
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
            "direct Graph-IR input requires sequential root selection"
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
        "direct Graph-IR input requires sequential root selection"
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

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use std::io::Write;

    use crate::dataset::{Payload, TiktokenTokenizer};
    use async_trait::async_trait;
    use flate2::{Compression, write::GzEncoder};
    use serde_json::json;

    use super::*;

    fn raw(value: Value) -> Box<RawValue> {
        serde_json::value::to_raw_value(&value).unwrap()
    }

    #[test]
    fn built_in_resolver_inventory_matches_workload_inventory() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        assert_eq!(
            resolver.supported_formats(),
            crate::config::model::workload_kind::GRAPH_FORMATS.to_vec()
        );
    }

    #[derive(Debug)]
    struct CountingResolver {
        load_calls: AtomicUsize,
    }

    #[async_trait(?Send)]
    impl GraphInputAdapterResolver for CountingResolver {
        fn supported_formats(&self) -> Vec<&str> {
            vec!["dag_jsonl"]
        }

        fn validate_identity(&self, _raw: &RawValue) -> Result<()> {
            panic!("inspection preparation must load without a separate validation pass")
        }

        async fn load(
            &self,
            raw: &RawValue,
            context: &GraphInputContext<'_>,
        ) -> Result<PreparedRunnerGraphInput> {
            self.load_calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(context.run_random_seed, Some(73));
            assert_eq!(
                raw.get(),
                r#"{"type":"file","format":"dag_jsonl","path":"/tmp/input.dag.jsonl","sampling":"sequential"}"#
            );
            let bundle = compile_dag_jsonl_input(
                GraphInputConfig {
                    load: LoadConfig::new(DatasetSource::Inline(json!([
                        {"session_id":"root","turns":[{"messages":[{"role":"user","content":"hello"}]}]}
                    ]))),
                    root_limit: None,
                },
                context.tokenizer,
            )
            .await
            .map_err(|error| anyhow!(error.to_string()))?;
            Ok(PreparedRunnerGraphInput {
                bundle,
                random_seed: None,
                default_output_tokens: 16,
                allow_dataset_wrap: true,
                t_star_window: TStarWindow::default(),
                cache_bust_target: CacheBustTarget::None,
            })
        }
    }

    #[tokio::test]
    async fn prepare_local_graph_inspection_input_loads_once_with_sequential_file_wire_shape() {
        let resolver = CountingResolver {
            load_calls: AtomicUsize::new(0),
        };
        let tokenizer = TiktokenTokenizer::builtin();

        let prepared = prepare_local_graph_inspection_input(
            &resolver,
            std::path::Path::new("/tmp/input.dag.jsonl"),
            "dag_jsonl",
            &tokenizer,
            73,
        )
        .await
        .unwrap();

        assert_eq!(resolver.load_calls.load(Ordering::Relaxed), 1);
        assert_eq!(prepared.bundle.programs.len(), 1);
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

    #[test]
    fn stock_resolver_selects_the_strict_recorded_agent_adapter() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        resolver
            .validate_identity(&raw(json!({
                "type": "file",
                "format": "agent_recording",
                "path": "recording.json"
            })))
            .expect("stock composition registers agent_recording");
    }

    #[tokio::test]
    async fn otlp_genai_adapter_lowers_openinference_and_genai_spans_to_graph_ir() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "otlp_genai",
            "sampling": "sequential",
            "records": {
                "resourceSpans": [{
                    "scopeSpans": [{
                        "spans": [
                            {
                                "traceId": "0123456789abcdef0123456789abcdef",
                                "spanId": "0000000000000001",
                                "name": "agent run",
                                "kind": 1,
                                "attributes": [{
                                    "key": "openinference.span.kind",
                                    "value": {"stringValue": "AGENT"}
                                }]
                            },
                            {
                                "traceId": "0123456789abcdef0123456789abcdef",
                                "spanId": "0000000000000002",
                                "parentSpanId": "0000000000000001",
                                "name": "chat completion",
                                "kind": 3,
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "chat"}
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": "[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"hello\\\"}]"}
                                    },
                                    {
                                        "key": "gen_ai.system",
                                        "value": {"stringValue": "openai"}
                                    },
                                    {
                                        "key": "server.address",
                                        "value": {"stringValue": "example.test"}
                                    }
                                ],
                                "events": [{"name": "gen_ai.choice"}]
                            }
                        ]
                    }]
                }]
            }
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("OTLP GenAI input lowers through the stock adapter");

        assert_eq!(prepared.bundle.metadata.format, "otlp_genai");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 2);
        let graph = &prepared.bundle.programs[0].profiling.graph;
        assert_eq!(graph.llm_node_count(), 2);
        let llm = graph
            .nodes
            .values()
            .filter_map(|node| node.as_llm())
            .find(|node| {
                node.metadata.get("otlp.span_id") == Some(&Value::String("0000000000000002".into()))
            })
            .expect("GenAI chat span lowers to an LLM node");
        assert!(llm.streaming);
        assert_eq!(llm.items.len(), 1);
    }

    #[tokio::test]
    async fn otlp_genai_adapter_reads_gzipped_jsonl_and_typed_message_attributes() {
        let fixture = tempfile::Builder::new()
            .suffix(".jsonl.gz")
            .tempfile()
            .expect("temporary OTLP fixture");
        let mut encoder = GzEncoder::new(
            fixture.reopen().expect("reopen fixture"),
            Compression::default(),
        );
        writeln!(encoder, "{}", json!({
            "resourceSpans": [{"scopeSpans": [{"spans": [{
                "traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "spanId": "0000000000000001",
                "name": "typed chat",
                "kind": 3,
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                    {"key": "gen_ai.input.messages", "value": {"arrayValue": {"values": [{"kvlistValue": {"values": [
                        {"key": "role", "value": {"stringValue": "user"}},
                        {"key": "content", "value": {"stringValue": "hello"}}
                    ]}}]}}}
                ]
            }]}]}]
        }))
        .expect("write gzipped OTLP JSONL");
        encoder.finish().expect("finish gzip fixture");

        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file", "format": "otlp_genai", "path": fixture.path()
        }));
        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: None,
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("gzipped OTLP JSONL lowers");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
    }

    #[tokio::test]
    async fn otlp_genai_adapter_folds_recorded_non_llm_spans_into_state_and_delay() {
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file", "format": "otlp_genai", "records": {
                "resourceSpans": [{"scopeSpans": [{"spans": [
                    {
                        "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "spanId": "0000000000000001", "name": "retrieval",
                        "kind": 1, "startTimeUnixNano": "1000000", "endTimeUnixNano": "51000000",
                        "attributes": [
                            {"key": "gen_ai.operation.name", "value": {"stringValue": "retrieve"}},
                            {"key": "output.value", "value": {"stringValue": "{\"documents\":[\"a\"]}"}}
                        ]
                    },
                    {
                        "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "spanId": "0000000000000005", "parentSpanId": "0000000000000001",
                        "name": "nested retrieval", "kind": 1,
                        "startTimeUnixNano": "20000000", "endTimeUnixNano": "31000000",
                        "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "retrieve"}}]
                    },
                    {
                        "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "spanId": "0000000000000002", "parentSpanId": "0000000000000005",
                        "name": "chat", "kind": 3,
                        "attributes": [
                            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                            {"key": "gen_ai.request.max_tokens", "value": {"intValue": "9"}}
                        ]
                    },
                    {
                        "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "spanId": "0000000000000003", "parentSpanId": "0000000000000002",
                        "name": "guardrail", "kind": 1, "startTimeUnixNano": "60000000", "endTimeUnixNano": "80000000",
                        "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "guardrail"}}]
                    },
                    {
                        "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "spanId": "0000000000000004", "parentSpanId": "0000000000000003",
                        "name": "second chat", "kind": 3,
                        "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}}]
                    }
                ]}]}]
            }
        }));
        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: None,
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("OTLP replay ancestors lower through state and edges");
        let plan = &prepared.bundle.programs[0].profiling;
        assert_eq!(plan.graph.llm_node_count(), 2);
        assert_eq!(
            plan.trace
                .initial_state
                .get("otlp_replay_0000000000000001_output"),
            Some(&json!({"documents":["a"]}))
        );
        let first = plan
            .graph
            .nodes
            .iter()
            .find_map(|(id, node)| {
                (node.as_llm()?.metadata.get("otlp.span_id") == Some(&json!("0000000000000002")))
                    .then_some(id)
            })
            .expect("first chat node");
        let second = plan
            .graph
            .nodes
            .iter()
            .find_map(|(id, node)| {
                (node.as_llm()?.metadata.get("otlp.span_id") == Some(&json!("0000000000000004")))
                    .then_some(id)
            })
            .expect("second chat node");
        let leading = plan
            .graph
            .edges
            .iter()
            .find(|edge| edge.target == *first)
            .expect("leading replay chain reaches first LLM");
        assert_eq!(leading.source, crate::graph::model::START_NODE_ID);
        assert_eq!(leading.min_start_delay_us, Some(50_000.0));
        assert_eq!(
            plan.graph.nodes[first].as_llm().unwrap().max_tokens,
            Some(9)
        );
        let rerouted = plan
            .graph
            .edges
            .iter()
            .find(|edge| edge.source == *first && edge.target == *second)
            .expect("intermediate replay span is folded into direct edge");
        assert_eq!(rerouted.delay_after_predecessor_us, Some(20_000.0));
    }

    #[tokio::test]
    async fn recorded_agent_adapter_discovers_and_lowers_the_manifest_corpus() {
        let fixture_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recorded_agent_replay");
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "agent_recording",
            "replay_root": fixture_root,
            "path": "canonical_manifest.json"
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("recorded-agent manifest lowers through the stock adapter");

        assert_eq!(prepared.bundle.metadata.format, "agent_recording");
        assert_eq!(prepared.bundle.metadata.root_count, 8);
        assert_eq!(prepared.bundle.metadata.node_count, 168);
        assert!(
            prepared
                .bundle
                .programs
                .iter()
                .all(|program| program.driver.kind == "recorded_replay")
        );
    }

    #[tokio::test]
    async fn recorded_agent_tool_execution_stages_pinch_task_pack_workspace_files() {
        let fixture_root = tempfile::tempdir().expect("temporary replay root");
        fs::create_dir_all(fixture_root.path().join("benchmark/pinchbench/tasks"))
            .expect("task-pack directories exist");
        fs::create_dir_all(fixture_root.path().join("benchmark/pinchbench/assets"))
            .expect("task-pack assets directory exists");
        fs::write(
            fixture_root
                .path()
                .join("benchmark/pinchbench/manifest.json"),
            serde_json::to_vec(&json!({
                "tasks": [{
                    "task_id": "task_k8s_debugging",
                    "path": "tasks/task_k8s_debugging.md"
                }]
            }))
            .expect("manifest serializes"),
        )
        .expect("task-pack manifest is written");
        fs::write(
            fixture_root
                .path()
                .join("benchmark/pinchbench/tasks/task_k8s_debugging.md"),
            b"---\nworkspace_files:\n  - source: input.txt\n    dest: config/input.txt\n---\n# Task\n",
        )
        .expect("task file is written");
        fs::write(
            fixture_root
                .path()
                .join("benchmark/pinchbench/assets/input.txt"),
            b"staged product fixture\n",
        )
        .expect("task asset is written");
        let source_fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "tests/fixtures/recorded_agent_replay/recordings/\
             pinchbench-openclaw-task_k8s_debugging-recording.json",
        );
        let recording: Value = serde_json::from_slice(
            &fs::read(source_fixture).expect("source recording fixture is readable"),
        )
        .expect("source recording is JSON");
        fs::write(
            fixture_root.path().join("recording.json"),
            serde_json::to_vec(&recording).expect("recording serializes"),
        )
        .expect("recording is written below replay root");
        let resolver = BuiltinRunnerGraphInputAdapterResolver::new();
        let input = raw(json!({
            "type": "file",
            "format": "agent_recording",
            "replay_root": fixture_root.path(),
            "path": "recording.json",
            "graph": {
                "execute_tools": true,
                "pinch_image": "aiperf-recorded-agent-pinchbench:v1"
            }
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("tool-enabled recorded-agent input resolves every recipe");

        let environment = prepared.bundle.programs[0]
            .environment
            .as_ref()
            .expect("tool-enabled replay retains its environment")
            .resolve()
            .expect("transport recipe resolves");
        assert_eq!(environment.workspace.files.len(), 1);
        assert_eq!(
            environment.workspace.files[0].destination,
            "config/input.txt"
        );
        let Payload::Raw { wire } = prepared
            .bundle
            .segments
            .get(environment.workspace.files[0].content)
            .expect("staged task file travels in the graph segment store")
        else {
            panic!("Pinch workspace content must be stored as raw bytes")
        };
        assert_eq!(wire.as_ref(), b"staged product fixture\n");
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
                    endpoint_id: "chat",
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
                    endpoint_id: "chat",
                },
            )
            .await
            .expect_err("unknown adapter fields must fail");
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
                "idle_gap_cap_seconds": 60.0
            },
            "prompts": {
                "corpus": "sonnet"
            }
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("direct WEKA compiler");
        assert_eq!(prepared.bundle.metadata.format, "weka_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(
            prepared.bundle.programs[0].profiling.graph.nodes["root:0"]
                .as_llm()
                .unwrap()
                .max_tokens,
            Some(3)
        );
        assert_eq!(prepared.random_seed, Some(91));
        assert!(prepared.allow_dataset_wrap);
    }

    #[tokio::test]
    async fn weka_adapter_accepts_random_prompt_corpus() {
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
            "synthesis": {
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "max_osl": 3,
                "allow_dataset_wrap": true,
                "idle_gap_cap_seconds": 60.0
            },
            "prompts": {
                "corpus": "random"
            }
        }));

        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("direct WEKA compiler with random prompt corpus");
        assert_eq!(prepared.bundle.metadata.format, "weka_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(
            prepared.bundle.programs[0].profiling.graph.nodes["root:0"]
                .as_llm()
                .unwrap()
                .max_tokens,
            Some(3)
        );
        assert_eq!(
            prepared.bundle.programs[0].profiling.graph.nodes["root:0"]
                .as_llm()
                .unwrap()
                .items
                .len(),
            1
        );
    }

    #[test]
    fn recorded_public_prompt_corpus_is_resolved_from_prompts() {
        let prepared = prepare_recorded_public(
            PublicDatasetSpec {
                name: "weka-public".into(),
                format: "weka_trace".into(),
                source: PublicDatasetSourceSpec::Url {
                    url: "https://example.invalid/weka.json".into(),
                },
                sampling: "sequential".into(),
                entries: Some(1),
                random_seed: None,
                options: Map::new(),
                prompts: Some(PromptSelectionSpec {
                    corpus: Some("random".into()),
                }),
                synthesis: None,
                cache_bust: None,
                prefetch_media_urls: false,
            },
            "weka_trace",
            &GraphInputContext {
                tokenizer: &TiktokenTokenizer::builtin(),
                run_random_seed: Some(42),
                endpoint_id: "chat",
            },
        )
        .expect("public recorded input with prompt corpus");
        assert_eq!(prepared.input.prompt_corpus, PromptCorpus::Random);
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
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("direct aiperf_trace compiler");
        assert_eq!(prepared.bundle.metadata.format, "aiperf_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert!(
            prepared.bundle.programs[0]
                .profiling
                .graph
                .nodes
                .contains_key("7:0")
        );
        assert_eq!(prepared.random_seed, Some(91));
    }

    #[test]
    fn recorded_request_carries_tstar_window_on_synthesis() {
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
                    endpoint_id: "chat",
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
                "output_len_multiplier": 1.0
            },
            "prompts": {
                "corpus": "sonnet"
            }
        }));
        let prepared = resolver
            .load(
                &input,
                &GraphInputContext {
                    tokenizer: &TiktokenTokenizer::builtin(),
                    run_random_seed: Some(42),
                    endpoint_id: "chat",
                },
            )
            .await
            .expect("direct Dynamo compiler");
        assert_eq!(prepared.bundle.metadata.format, "dynamo_trace");
        assert_eq!(prepared.bundle.metadata.root_count, 1);
        assert_eq!(prepared.bundle.metadata.node_count, 1);
        assert_eq!(prepared.bundle.programs[0].profiling.trace.id, "root");
    }
}
