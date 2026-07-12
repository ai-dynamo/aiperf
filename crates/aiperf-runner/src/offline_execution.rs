// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-gated runner composition for Dynamo's in-process offline engine.
//!
//! This module owns only the authored runner boundary. The virtual clock,
//! steppable engine host, scheduling loops, cancellation, observers, and exact
//! AIPerf/Dynamo parity proof remain in [`aiperf::dynamo_offline`]. A registered
//! backend/workload pair supplies one of the typed execution plans below; no
//! string branch or alternate executable is introduced here.
//!
//! The strict authored projection mirrors the canonical Python wire producer
//! in `src/aiperf/orchestrator/rust_wire.py:81` and consumes its dataset and
//! phase shapes as defined by `src/aiperf/config/dataset/config.py:280` and
//! `src/aiperf/config/phases.py:63`. The graph adapter resolves `dag_jsonl`
//! directly into Graph-IR once; it never constructs an intermediate linear
//! dataset or re-projects the authored object through protocol v1.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::dynamo_offline::{
    CanonicalSharedMetrics, DeferredOfflineScheduledFuture, DeferredOfflineScheduledRunFactory,
    OfflineAicConfig, OfflineDirectGraphReport, OfflineEngineConfig, OfflineGraphReport,
    OfflineGraphRunFactory, OfflineKvEventVisibility, OfflineMetricParity, OfflineRouterMode,
    OfflineRunReport, OfflineScheduledExecution, OfflineScheduledExecutionFinalizer,
    OfflineScheduledReport, OfflineScheduledRunFactory, OfflineTopology, OfflineTraceConfig,
    run_graph_offline, run_graph_workload_offline, run_scheduled_backend_offline,
    run_scheduled_backend_offline_deferred, run_trace_offline, write_dynamo_per_request_jsonl,
    write_dynamo_report_json, write_dynamo_worker_artifacts_json,
};
use aiperf::metrics::NativeMetricsObserver;
use aiperf::multiturn::{
    ConversationSource, EndpointInputTokenCounter, InputTokenCounter,
    NativeDatasetConversationSource, PreparedEndpointReference, PreparedEndpointTableResolver,
    PreparedTurnEndpointResolver,
};
use aiperf::phase_runtime::run_scheduled_phases_with_aggregate_deferred;
use aiperf_clock::Clock;
use aiperf_dataset::{
    SamplerRegistry, SegmentStore, TextTokenizer, TraceHashAwareRequestMaterializer,
};
use aiperf_endpoints::{Modality, PreparedEndpointTable};
use aiperf_graph::bench::BenchConfig;
use aiperf_graph::input::GraphInputBundle;
use aiperf_graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, CompositeNodeDispatchPolicy,
    FailFastRunFailurePolicy, NodeDispatchPolicy, NodeFailurePolicy, PrefillSlotNodePolicy,
};
use aiperf_graph::workload::{
    CyclingGraphTraceSource, DurationGraphStop, GraphArrivalPolicy, GraphTraceInstanceSequence,
    GraphTraceSource, GraphWorkload, ImmediateGraphArrival, IntervalGraphArrival,
    SlotPoolTraceAdmission,
};
use aiperf_metrics::{
    CATALOG, MetricFlags, MetricTag, MetricType, MetricsConfig, NativeReport, ReportClockKind,
    ReportDynamoCapacityInfo, ReportDynamoParityInfo, ReportDynamoRouter, ReportDynamoRunInfo,
    ReportDynamoTopology, ReportGraphOutcomeInfo, ReportGraphRunInfo, ReportPairRunFacts,
    ReportRunInfo, ReportSummary, RunOutcome, SloThreshold,
};
use aiperf_rng::RngRoot;
use aiperf_timing::{
    BernoulliFixedDelay, NoopPhaseObserver, Phase, SlotPool, make_interval_generator,
};
use anyhow::{Context, Result, anyhow, ensure};
use loadgen_core::sink::RequestObserver;
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

use crate::dataset_input::{PreparedDatasetInput, RunnerDatasetInputContext};
use crate::execute::{
    NativeConversationSourceFactory, build_native_scheduled_phase_plan_with_source_factory,
    load_tokenizer, metrics_config, native_scheduled_resources,
};
use crate::graph_input::RunnerGraphInputContext;
use crate::online_execution::{
    NativeOnlineTokenizerSourceResolver, OnlineTokenizerSourceResolver, lower_authored_tokenizer,
    validate_authored_tokenizer,
};
use crate::protocol::{MetricsSpec, ModelSelectionStrategy, PhaseSpec};
use crate::protocol_v2::AuthoredRunSpecV2;
use crate::registry::{
    GraphWorkloadConfigV2, PreparedRunOutcome, PreparedRunnerOperation, RunnerBackendDescriptor,
    RunnerBackendFactory, RunnerClockKind, RunnerPairFactory, RunnerRegistryBuilder,
    RunnerRunContext, ScheduledWorkloadConfigV2, ValidatedBackendConfig, ValidatedWorkloadConfig,
    WorkloadRequirements,
};

/// Stable runner-registry ID for the in-process Dynamo backend.
pub const DYNAMO_OFFLINE_BACKEND_ID: &str = "dynamo_offline";

static DYNAMO_OFFLINE_BACKEND_DESCRIPTOR: RunnerBackendDescriptor = RunnerBackendDescriptor {
    id: DYNAMO_OFFLINE_BACKEND_ID,
    description: "Dynamo passive-engine co-simulation on one deterministic SimClock",
    clock: RunnerClockKind::Sim,
    semantic_responses: false,
    features: &[
        "steppable_replay",
        "aggregate",
        "disaggregate",
        "kv_routing",
        "cancellation",
        "canonical_trace_formats",
        "worker_artifacts",
        "exact_metric_parity",
        #[cfg(feature = "dynamo-router-runtime")]
        "dynamo-router-runtime",
        #[cfg(feature = "dynamo-zmq-events")]
        "dynamo-zmq-events",
        #[cfg(feature = "dynamo-kvbm-offload")]
        "dynamo-kvbm-offload",
        #[cfg(feature = "dynamo-aic-forward-pass")]
        "dynamo-aic-forward-pass",
        #[cfg(feature = "dynamo-profile")]
        "dynamo-profile",
        #[cfg(feature = "dynamo-full")]
        "dynamo-full",
        #[cfg(feature = "dynamo-parity")]
        "dynamo-parity",
    ],
};

/// Optional Dynamo build capability that an authored run may require.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum DynamoBuildFeature {
    /// Dynamo's application router runtime.
    DynamoRouterRuntime,
    /// Dynamo's ZMQ event publisher.
    DynamoZmqEvents,
    /// G1/G2/G3/G4 KV offload simulation.
    DynamoKvbmOffload,
    /// Embedded AIConfigurator forward-pass modeling.
    DynamoAicForwardPass,
    /// Profiling-friendly no-inline annotations.
    DynamoProfile,
    /// Complete optional Dynamo feature family.
    DynamoFull,
    /// Exact official parity build (which also enables `dynamo-full`).
    DynamoParity,
}

impl DynamoBuildFeature {
    /// Cargo feature spelling used in errors and capabilities.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DynamoRouterRuntime => "dynamo-router-runtime",
            Self::DynamoZmqEvents => "dynamo-zmq-events",
            Self::DynamoKvbmOffload => "dynamo-kvbm-offload",
            Self::DynamoAicForwardPass => "dynamo-aic-forward-pass",
            Self::DynamoProfile => "dynamo-profile",
            Self::DynamoFull => "dynamo-full",
            Self::DynamoParity => "dynamo-parity",
        }
    }

    const fn is_compiled(self) -> bool {
        match self {
            Self::DynamoRouterRuntime => cfg!(feature = "dynamo-router-runtime"),
            Self::DynamoZmqEvents => cfg!(feature = "dynamo-zmq-events"),
            Self::DynamoKvbmOffload => cfg!(feature = "dynamo-kvbm-offload"),
            Self::DynamoAicForwardPass => cfg!(feature = "dynamo-aic-forward-pass"),
            Self::DynamoProfile => cfg!(feature = "dynamo-profile"),
            Self::DynamoFull => cfg!(feature = "dynamo-full"),
            Self::DynamoParity => cfg!(feature = "dynamo-parity"),
        }
    }
}

/// Authored offline deployment topology.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DynamoOfflineTopologySpec {
    /// One eventized aggregate worker without a routing choice.
    #[default]
    Single,
    /// Multiple aggregate workers behind the selected router.
    Aggregated,
    /// Separate prefill and decode worker pools.
    Disaggregated,
}

impl DynamoOfflineTopologySpec {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Aggregated => "aggregated",
            Self::Disaggregated => "disaggregated",
        }
    }

    const fn report(self) -> ReportDynamoTopology {
        match self {
            Self::Single => ReportDynamoTopology::Single,
            Self::Aggregated => ReportDynamoTopology::Aggregated,
            Self::Disaggregated => ReportDynamoTopology::Disaggregated,
        }
    }
}

impl From<DynamoOfflineTopologySpec> for OfflineTopology {
    fn from(value: DynamoOfflineTopologySpec) -> Self {
        match value {
            DynamoOfflineTopologySpec::Single => Self::Single,
            DynamoOfflineTopologySpec::Aggregated => Self::Aggregated,
            DynamoOfflineTopologySpec::Disaggregated => Self::Disaggregated,
        }
    }
}

/// Authored router policy for routed offline topologies.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DynamoOfflineRouterSpec {
    /// Stable deterministic worker rotation.
    #[default]
    RoundRobin,
    /// Prefix-affinity/load-aware KV routing.
    Kv,
}

impl DynamoOfflineRouterSpec {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::Kv => "kv",
        }
    }

    const fn report(self) -> ReportDynamoRouter {
        match self {
            Self::RoundRobin => ReportDynamoRouter::RoundRobin,
            Self::Kv => ReportDynamoRouter::Kv,
        }
    }
}

impl From<DynamoOfflineRouterSpec> for OfflineRouterMode {
    fn from(value: DynamoOfflineRouterSpec) -> Self {
        match value {
            DynamoOfflineRouterSpec::RoundRobin => Self::RoundRobin,
            DynamoOfflineRouterSpec::Kv => Self::Kv,
        }
    }
}

/// Strict structured AIConfigurator overrides for offline timing.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoOfflineAicSpec {
    /// Serving backend identity.
    #[serde(default)]
    pub backend: Option<String>,
    /// GPU system identity.
    #[serde(default)]
    pub system: Option<String>,
    /// Performance-database backend version.
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Tensor-parallel degree.
    #[serde(default)]
    pub tp_size: Option<usize>,
    /// Hugging Face model path.
    #[serde(default)]
    pub model_path: Option<String>,
    /// MoE tensor-parallel degree.
    #[serde(default)]
    pub moe_tp_size: Option<usize>,
    /// MoE expert-parallel degree.
    #[serde(default)]
    pub moe_ep_size: Option<usize>,
    /// Attention data-parallel degree.
    #[serde(default)]
    pub attention_dp_size: Option<usize>,
    /// GEMM quantization override.
    #[serde(default)]
    pub gemm_dtype: Option<String>,
    /// MoE quantization override.
    #[serde(default)]
    pub moe_dtype: Option<String>,
    /// FMHA quantization override.
    #[serde(default)]
    pub fmha_dtype: Option<String>,
    /// KV-cache quantization override.
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    /// Communication quantization override.
    #[serde(default)]
    pub comm_dtype: Option<String>,
    /// Speculative draft-token count.
    #[serde(default)]
    pub nextn: Option<usize>,
    /// Comma-separated conditional draft acceptance rates.
    #[serde(default)]
    pub nextn_accept_rates: Option<String>,
}

impl DynamoOfflineAicSpec {
    fn requested(&self) -> bool {
        self.backend.is_some()
            || self.system.is_some()
            || self.backend_version.is_some()
            || self.tp_size.is_some()
            || self.model_path.is_some()
            || self.moe_tp_size.is_some()
            || self.moe_ep_size.is_some()
            || self.attention_dp_size.is_some()
            || self.gemm_dtype.is_some()
            || self.moe_dtype.is_some()
            || self.fmha_dtype.is_some()
            || self.kv_cache_dtype.is_some()
            || self.comm_dtype.is_some()
            || self.nextn.is_some()
            || self.nextn_accept_rates.is_some()
    }

    fn validate(&self) -> Result<()> {
        if !self.requested() {
            return Ok(());
        }
        ensure!(self.backend.is_some(), "AIC modeling requires aic.backend");
        ensure!(self.system.is_some(), "AIC modeling requires aic.system");
        ensure!(
            self.model_path.is_some(),
            "AIC modeling requires aic.model_path"
        );
        for (name, value) in [
            ("aic.tp_size", self.tp_size),
            ("aic.moe_tp_size", self.moe_tp_size),
            ("aic.moe_ep_size", self.moe_ep_size),
            ("aic.attention_dp_size", self.attention_dp_size),
        ] {
            if let Some(value) = value {
                ensure!(value > 0, "{name} must be positive");
            }
        }
        Ok(())
    }

    fn into_runtime(self) -> OfflineAicConfig {
        OfflineAicConfig {
            backend: self.backend,
            system: self.system,
            backend_version: self.backend_version,
            tp_size: self.tp_size,
            model_path: self.model_path,
            moe_tp_size: self.moe_tp_size,
            moe_ep_size: self.moe_ep_size,
            attention_dp_size: self.attention_dp_size,
            gemm_dtype: self.gemm_dtype,
            moe_dtype: self.moe_dtype,
            fmha_dtype: self.fmha_dtype,
            kv_cache_dtype: self.kv_cache_dtype,
            comm_dtype: self.comm_dtype,
            nextn: self.nextn,
            nextn_accept_rates: self.nextn_accept_rates,
        }
    }
}

/// Canonical goodput thresholds owned by Dynamo's collector.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoOfflineSlaSpec {
    /// Maximum time to first token in milliseconds.
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Maximum mean inter-token latency in milliseconds.
    #[serde(default)]
    pub itl_ms: Option<f64>,
    /// Maximum end-to-end latency in milliseconds.
    #[serde(default)]
    pub e2e_ms: Option<f64>,
}

impl DynamoOfflineSlaSpec {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("sla.ttft_ms", self.ttft_ms),
            ("sla.itl_ms", self.itl_ms),
            ("sla.e2e_ms", self.e2e_ms),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "{name} must be finite and non-negative"
                );
            }
        }
        Ok(())
    }

    fn native_metrics_config(&self) -> Result<Option<MetricsConfig>> {
        let mut slos = Vec::new();
        for (tag, value) in [
            (MetricTag::TimeToFirstToken, self.ttft_ms),
            (MetricTag::InterTokenLatency, self.itl_ms),
            (MetricTag::RequestLatency, self.e2e_ms),
        ] {
            if let Some(value) = value {
                slos.push(SloThreshold::from_display(tag, value)?);
            }
        }
        Ok((!slos.is_empty()).then_some(MetricsConfig {
            slos,
            ..MetricsConfig::default()
        }))
    }
}

/// Visibility point for timed KV-event worker artifacts.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DynamoKvEventVisibilitySpec {
    /// Publish mutations at scheduler-pass start.
    PassStart,
    /// Publish mutations at scheduler-pass completion.
    PassEnd,
}

impl From<DynamoKvEventVisibilitySpec> for OfflineKvEventVisibility {
    fn from(value: DynamoKvEventVisibilitySpec) -> Self {
        match value {
            DynamoKvEventVisibilitySpec::PassStart => Self::PassStart,
            DynamoKvEventVisibilitySpec::PassEnd => Self::PassEnd,
        }
    }
}

/// Backend-specific artifacts written after a successful offline run.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoOfflineArtifactSpec {
    /// Canonical aggregate Dynamo JSON relative to the run artifact target.
    #[serde(default)]
    pub report_json: Option<PathBuf>,
    /// Canonical per-request Dynamo JSONL relative to the artifact target.
    #[serde(default)]
    pub per_request_jsonl: Option<PathBuf>,
    /// Timed worker/request/KV artifact JSON for canonical trace workloads.
    #[serde(default)]
    pub worker_artifacts_json: Option<PathBuf>,
    /// Optional pass-start/pass-end KV visibility override.
    #[serde(default)]
    pub kv_event_visibility: Option<DynamoKvEventVisibilitySpec>,
}

impl DynamoOfflineArtifactSpec {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.kv_event_visibility.is_none() || self.worker_artifacts_json.is_some(),
            "artifacts.kv_event_visibility requires artifacts.worker_artifacts_json"
        );
        let mut seen = BTreeSet::new();
        for (name, path) in [
            ("artifacts.report_json", self.report_json.as_deref()),
            (
                "artifacts.per_request_jsonl",
                self.per_request_jsonl.as_deref(),
            ),
            (
                "artifacts.worker_artifacts_json",
                self.worker_artifacts_json.as_deref(),
            ),
        ] {
            if let Some(path) = path {
                validate_relative_artifact_path(name, path)?;
                ensure!(
                    seen.insert(path.to_path_buf()),
                    "offline artifact paths must be distinct; duplicate {}",
                    path.display()
                );
            }
        }
        Ok(())
    }
}

/// Strict authored configuration owned by the `dynamo_offline` backend.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamoOfflineBackendSpec {
    /// Optional JSON profile consumed by Dynamo's canonical engine parser.
    #[serde(default)]
    pub engine_profile: Option<PathBuf>,
    /// Inline aggregate/single `MockEngineArgs` object.
    #[serde(default)]
    pub engine: Option<Box<RawValue>>,
    /// Inline disaggregated prefill `MockEngineArgs` object.
    #[serde(default)]
    pub prefill_engine: Option<Box<RawValue>>,
    /// Inline disaggregated decode `MockEngineArgs` object.
    #[serde(default)]
    pub decode_engine: Option<Box<RawValue>>,
    /// Complete inline `KvRouterConfig` object.
    #[serde(default)]
    pub router: Option<Box<RawValue>>,
    /// Startup router policy-family YAML path overriding the inline field.
    #[serde(default)]
    pub router_policy_config: Option<PathBuf>,
    /// Model selector for a multi-model router policy document.
    #[serde(default)]
    pub router_model_name: Option<String>,
    /// Optional structured AIConfigurator overrides.
    #[serde(default)]
    pub aic: Option<DynamoOfflineAicSpec>,
    /// Capture backend per-request records even without a JSONL artifact.
    #[serde(default)]
    pub capture_per_request: bool,
    /// Canonical backend-owned goodput thresholds.
    #[serde(default)]
    pub sla: DynamoOfflineSlaSpec,
    /// Deployment topology.
    #[serde(default)]
    pub topology: DynamoOfflineTopologySpec,
    /// Aggregate worker count.
    #[serde(default = "one")]
    pub workers: usize,
    /// Disaggregated prefill worker count.
    #[serde(default = "one")]
    pub prefill_workers: usize,
    /// Disaggregated decode worker count.
    #[serde(default = "one")]
    pub decode_workers: usize,
    /// Router policy for routed topologies.
    #[serde(default)]
    pub router_mode: DynamoOfflineRouterSpec,
    /// Optional build capabilities that must exist in the exact runner image.
    #[serde(default)]
    pub required_features: BTreeSet<DynamoBuildFeature>,
    /// Backend-owned output artifacts.
    #[serde(default)]
    pub artifacts: DynamoOfflineArtifactSpec,
}

const fn one() -> usize {
    1
}

impl DynamoOfflineBackendSpec {
    fn validate(self) -> Result<ValidatedDynamoOfflineBackend> {
        ensure!(
            self.engine_profile.is_none() || self.engine.is_none(),
            "engine_profile conflicts with inline engine"
        );
        for (name, value) in [
            ("engine", self.engine.as_deref()),
            ("prefill_engine", self.prefill_engine.as_deref()),
            ("decode_engine", self.decode_engine.as_deref()),
            ("router", self.router.as_deref()),
        ] {
            validate_raw_object(name, value)?;
        }
        ensure!(self.workers > 0, "workers must be positive");
        ensure!(self.prefill_workers > 0, "prefill_workers must be positive");
        ensure!(self.decode_workers > 0, "decode_workers must be positive");
        ensure!(
            self.prefill_engine.is_some() == self.decode_engine.is_some(),
            "prefill_engine and decode_engine must be authored together"
        );
        if self.topology != DynamoOfflineTopologySpec::Disaggregated {
            ensure!(
                self.prefill_engine.is_none(),
                "prefill_engine/decode_engine require topology=disaggregated"
            );
        }
        self.sla.validate()?;
        self.artifacts.validate()?;
        if let Some(aic) = &self.aic {
            aic.validate()?;
        }

        let mut required = self.required_features.clone();
        if self
            .aic
            .as_ref()
            .is_some_and(DynamoOfflineAicSpec::requested)
            || [&self.engine, &self.prefill_engine, &self.decode_engine]
                .into_iter()
                .flatten()
                .any(|value| raw_requests_aic(value))
        {
            required.insert(DynamoBuildFeature::DynamoAicForwardPass);
        }
        if [&self.engine, &self.prefill_engine, &self.decode_engine]
            .into_iter()
            .flatten()
            .any(|value| raw_requests_offload(value))
        {
            required.insert(DynamoBuildFeature::DynamoKvbmOffload);
        }
        for feature in &required {
            ensure!(
                feature.is_compiled(),
                "Dynamo offline configuration requires runner feature {:?}, but it is absent from this distribution",
                feature.as_str()
            );
        }

        let topology = self.topology;
        let router_mode = self.router_mode;
        let artifacts = self.artifacts;
        let aic = self
            .aic
            .filter(DynamoOfflineAicSpec::requested)
            .map(DynamoOfflineAicSpec::into_runtime);
        let capture_per_request = self.capture_per_request || artifacts.per_request_jsonl.is_some();
        let mut engine = OfflineEngineConfig {
            profile: self.engine_profile,
            extra_engine_args: self.engine.map(|value| value.get().to_owned()),
            prefill_engine_args: self.prefill_engine.map(|value| value.get().to_owned()),
            decode_engine_args: self.decode_engine.map(|value| value.get().to_owned()),
            router_config: self.router.map(|value| value.get().to_owned()),
            router_policy_config: self.router_policy_config,
            router_model_name: self.router_model_name,
            aic,
            capture_per_request,
            topology: topology.into(),
            workers: self.workers,
            prefill_workers: self.prefill_workers,
            decode_workers: self.decode_workers,
            router_mode: router_mode.into(),
            ..OfflineEngineConfig::default()
        };
        engine = engine.with_sla_thresholds(self.sla.ttft_ms, self.sla.itl_ms, self.sla.e2e_ms)?;
        Ok(ValidatedDynamoOfflineBackend {
            engine,
            artifacts,
            sla: self.sla,
            topology,
            router_mode,
            required_features: required,
        })
    }
}

fn validate_raw_object(name: &str, value: Option<&RawValue>) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    let parsed: Value =
        serde_json::from_str(value.get()).with_context(|| format!("{name} is not valid JSON"))?;
    ensure!(parsed.is_object(), "{name} must be a JSON object");
    Ok(())
}

fn raw_requests_offload(value: &RawValue) -> bool {
    raw_object_requests(value, |name, value| {
        matches!(
            name,
            "num_g2_blocks" | "num_g3_blocks" | "num_g4_blocks" | "enable_g4_storage"
        ) && value_requests_capability(value)
    })
}

fn raw_requests_aic(value: &RawValue) -> bool {
    raw_object_requests(value, |name, value| {
        name.starts_with("aic_") && value_requests_capability(value)
    })
}

fn raw_object_requests(value: &RawValue, predicate: impl Fn(&str, &Value) -> bool) -> bool {
    serde_json::from_str::<Value>(value.get())
        .ok()
        .and_then(|value| value.as_object().cloned())
        .is_some_and(|object| object.iter().any(|(name, value)| predicate(name, value)))
}

fn value_requests_capability(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn validate_relative_artifact_path(name: &str, path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{name} cannot be empty");
    ensure!(!path.is_absolute(), "{name} must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{name} must contain only normal relative path components"
    );
    Ok(())
}

/// Strictly validated runner backend state retained until pair preparation.
#[derive(Clone, Debug)]
pub struct ValidatedDynamoOfflineBackend {
    engine: OfflineEngineConfig,
    artifacts: DynamoOfflineArtifactSpec,
    sla: DynamoOfflineSlaSpec,
    topology: DynamoOfflineTopologySpec,
    router_mode: DynamoOfflineRouterSpec,
    required_features: BTreeSet<DynamoBuildFeature>,
}

impl ValidatedDynamoOfflineBackend {
    /// Build a no-side-effect execution adapter rooted at the selected target.
    pub fn executor(
        &self,
        model: impl Into<String>,
        artifact_target: impl Into<PathBuf>,
    ) -> Result<DynamoOfflineExecutor> {
        let model = model.into();
        ensure!(!model.trim().is_empty(), "offline model cannot be empty");
        let artifact_target = artifact_target.into();
        ensure!(
            !artifact_target.as_os_str().is_empty(),
            "offline artifact target cannot be empty"
        );
        let mut engine = self.engine.clone();
        if engine.router_model_name.is_none() {
            engine.router_model_name = Some(model.clone());
        }
        Ok(DynamoOfflineExecutor {
            engine,
            artifacts: self.artifacts.clone(),
            topology: self.topology,
            router_mode: self.router_mode,
            required_features: self.required_features.clone(),
            model,
            artifact_target,
        })
    }
}

fn dynamo_report_facts(
    backend: &ValidatedDynamoOfflineBackend,
    parity: OfflineMetricParity,
    performance: &loadgen_core::collector::TraceSimulationReport,
) -> Result<ReportDynamoRunInfo> {
    let parity = ReportDynamoParityInfo::new(
        parity.shared_fields,
        parity.independently_accumulated_fields,
        parity.backend_owned_fields,
        parity.serialized_bytes,
    )?;
    let throughput = &performance.throughput;
    let capacity = ReportDynamoCapacityInfo::new(
        throughput.prefill_worker_seconds,
        throughput.decode_worker_seconds,
        throughput.prefill_gpus_per_worker,
        throughput.decode_gpus_per_worker,
        throughput.gpu_hours,
    )?;
    Ok(ReportDynamoRunInfo::new(
        ReportClockKind::Sim,
        backend.topology.report(),
        backend.router_mode.report(),
        backend
            .required_features
            .iter()
            .map(|feature| feature.as_str().to_owned())
            .collect(),
        backend.engine.workers,
        backend.engine.prefill_workers,
        backend.engine.decode_workers,
        parity,
    )?
    .with_capacity(capacity))
}

/// Registered strict decoder for the feature-bearing offline backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamoOfflineBackendFactory;

impl RunnerBackendFactory for DynamoOfflineBackendFactory {
    fn descriptor(&self) -> &'static RunnerBackendDescriptor {
        &DYNAMO_OFFLINE_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedBackendConfig>> {
        let spec = serde_json::from_str::<DynamoOfflineBackendSpec>(authored.get())
            .map_err(|error| anyhow!("invalid dynamo_offline backend config: {error}"))?;
        Ok(Box::new(spec.validate()?))
    }
}

/// Add the offline backend and its executable workload pairs to a mutable
/// runner registry.
///
/// Direct graph preparation resolves its authored-input adapter from the
/// coordinator-owned [`RunnerRunContext`], so the pair never constructs or
/// retains a private adapter universe.
pub fn register_dynamo_offline_backend(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_backend(Arc::new(DynamoOfflineBackendFactory))?;
    builder.register_pair(Arc::new(DynamoOfflineGraphPairFactory::default()))?;
    builder.register_pair(Arc::new(DynamoOfflineScheduledPairFactory::default()))
}

/// Downcast one pair-factory backend value with an actionable invariant error.
pub fn validated_dynamo_offline_backend(
    config: &dyn ValidatedBackendConfig,
) -> Result<&ValidatedDynamoOfflineBackend> {
    config
        .as_any()
        .downcast_ref::<ValidatedDynamoOfflineBackend>()
        .ok_or_else(|| anyhow!("dynamo_offline pair received a different backend config type"))
}

#[derive(Clone)]
struct DynamoOfflineScheduledPairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl Default for DynamoOfflineScheduledPairFactory {
    fn default() -> Self {
        Self {
            tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
        }
    }
}

impl fmt::Debug for DynamoOfflineScheduledPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DynamoOfflineScheduledPairFactory")
    }
}

impl RunnerPairFactory for DynamoOfflineScheduledPairFactory {
    fn backend_id(&self) -> &'static str {
        DYNAMO_OFFLINE_BACKEND_ID
    }

    fn workload_id(&self) -> &'static str {
        "scheduled"
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        let _ = validated_dynamo_offline_backend(backend)?;
        let workload = validated_scheduled_workload(workload)?;
        ensure!(
            workload.worker_count == 1,
            "dynamo_offline scheduled execution owns one LocalSet around one globally contended engine; worker_count must be 1"
        );
        validate_offline_scheduled_phases(&workload.phases)
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        ensure!(
            context.sidecar_inputs().is_empty(),
            "dynamo_offline scheduled execution does not support online sidecars"
        );
        ensure!(
            run.artifacts.records_path.is_none()
                && run.artifacts.raw_path.is_none()
                && run.artifacts.outputs_path.is_none()
                && !run.artifacts.trace
                && run.artifacts.user_files.is_empty(),
            "dynamo_offline scheduled execution does not project common request/raw/output/user-file artifacts; use backend Dynamo artifacts or disable them"
        );
        ensure!(
            !run.artifact_target.exists(),
            "artifact_target already exists; protocol-v2 execution requires an exclusive uncreated target"
        );
        Ok(())
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        Err(anyhow!(
            "dynamo_offline + scheduled preparation requires the coordinator-owned RunnerRunContext"
        ))
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        let backend = validated_dynamo_offline_backend(backend.as_ref())?.clone();
        let workload = validated_scheduled_workload(workload.as_ref())?;
        validate_offline_scheduled_phases(&workload.phases)?;

        let tokenizer_spec =
            lower_authored_tokenizer(&workload.tokenizer, self.tokenizers.as_ref())?;
        let tokenizer = load_tokenizer(Some(&tokenizer_spec.name))?;
        let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(
            EndpointInputTokenCounter::new(tokenizer.clone(), tokenizer_spec.apply_chat_template),
        );
        let profile = context.default_endpoint_profile()?;
        let prepared_endpoint = context
            .product_registry()
            .endpoints()
            .prepare(&profile.endpoint_id, profile.config.clone())
            .context("preparing offline scheduled endpoint materialization policy")?;
        let rankings = prepared_endpoint
            .descriptor()
            .output_modalities
            .contains(&Modality::Rankings);
        let mut endpoint_table = PreparedEndpointTable::new();
        let endpoint_key = endpoint_table.push(prepared_endpoint)?;
        let endpoint_reference = PreparedEndpointReference {
            key: endpoint_key,
            endpoint_id: profile.endpoint_id.clone(),
        };
        let endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver> = Rc::new(
            PreparedEndpointTableResolver::single(Rc::new(endpoint_table), endpoint_reference)?,
        );

        let rng_root = RngRoot::new(run.identity.random_seed);
        let dataset_context = RunnerDatasetInputContext {
            registry: context.product_registry(),
            models: &run.models,
            run_rng_root: rng_root,
            tokenizer: tokenizer.as_ref(),
            rankings,
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating offline scheduled preparation runtime")?;
        let local = tokio::task::LocalSet::new();
        let prepared_dataset = local
            .block_on(
                &runtime,
                context
                    .dataset_inputs()
                    .load(&workload.dataset, &dataset_context),
            )
            .context("loading and validating authored offline scheduled dataset")?;
        let metrics = metrics_config(&run.metrics)?;
        let model = run
            .models
            .items
            .first()
            .map(|item| item.name.clone())
            .ok_or_else(|| anyhow!("dynamo_offline scheduled execution requires a model"))?;

        let adaptive_paths = [
            Path::new("adaptive_scale_events.jsonl"),
            Path::new("adaptive_scale_summary.json"),
        ];
        for backend_path in [
            backend.artifacts.report_json.as_deref(),
            backend.artifacts.per_request_jsonl.as_deref(),
            backend.artifacts.worker_artifacts_json.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            ensure!(
                !workload
                    .phases
                    .iter()
                    .any(|phase| phase.common().adaptive_scale.is_some())
                    || !adaptive_paths.contains(&backend_path),
                "backend artifact path conflicts with an adaptive-scale artifact: {}",
                backend_path.display()
            );
        }

        Ok(Box::new(PreparedDynamoOfflineScheduledOperation {
            backend,
            dataset: prepared_dataset,
            source_factory: DynamoOfflinePreparedConversationSourceFactory {
                endpoint_resolver,
                samplers: context.product_registry().samplers().clone(),
            },
            tokenizer,
            input_token_counter,
            phases: workload.phases.clone(),
            metrics,
            model,
            rng_root,
            artifact_target: run.artifact_target.clone(),
            benchmark_id: run.identity.benchmark_id.clone(),
        }))
    }
}

fn validated_scheduled_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&ScheduledWorkloadConfigV2> {
    config
        .as_any()
        .downcast_ref::<ScheduledWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("dynamo_offline scheduled pair received a different workload type"))
}

fn validate_offline_scheduled_phases(phases: &[PhaseSpec]) -> Result<()> {
    ensure!(
        !phases.is_empty(),
        "offline scheduled phases cannot be empty"
    );
    ensure!(
        phases
            .iter()
            .filter(|phase| phase.common().name == "profiling")
            .count()
            == 1,
        "offline scheduled execution requires exactly one profiling phase"
    );
    let mut saw_profiling = false;
    for (index, phase) in phases.iter().enumerate() {
        let common = phase.common();
        ensure!(
            matches!(common.name.as_str(), "warmup" | "profiling"),
            "offline scheduled phase {index}.name must be warmup or profiling"
        );
        ensure!(
            common.exclude_from_results == (common.name == "warmup"),
            "offline scheduled phase {:?} exclude_from_results disagrees with its semantic kind",
            common.name
        );
        ensure!(
            !saw_profiling || common.name != "warmup",
            "warmup phases must precede the profiling phase"
        );
        saw_profiling |= common.name == "profiling";
        ensure!(
            common.requests != Some(0),
            "phase requests must be positive"
        );
        ensure!(
            common.sessions != Some(0),
            "phase sessions must be positive"
        );
        if let Some(duration) = common.duration {
            ensure!(
                duration.is_finite() && duration > 0.0,
                "phase duration must be finite and positive"
            );
        }
        if let Some(grace) = common.grace_period {
            ensure!(
                grace.is_finite() && grace >= 0.0,
                "phase grace_period must be finite and non-negative"
            );
        }
        ensure!(
            phase.concurrency() != Some(0),
            "phase concurrency must be positive"
        );
        ensure!(
            common.prefill_concurrency != Some(0),
            "phase prefill_concurrency must be positive"
        );
        match phase {
            PhaseSpec::Concurrency { .. } => {}
            PhaseSpec::Poisson { rate, .. } | PhaseSpec::Constant { rate, .. } => ensure!(
                rate.is_finite() && *rate > 0.0,
                "phase rate must be finite and positive"
            ),
            PhaseSpec::Gamma {
                rate, smoothness, ..
            } => {
                ensure!(
                    rate.is_finite() && *rate > 0.0,
                    "phase rate must be finite and positive"
                );
                ensure!(
                    smoothness.is_none_or(|value| value.is_finite() && value > 0.0),
                    "gamma smoothness must be finite and positive"
                );
            }
            PhaseSpec::UserCentric { rate, users, .. } => {
                ensure!(
                    rate.is_finite() && *rate > 0.0,
                    "user_centric rate must be finite and positive"
                );
                ensure!(*users > 0, "user_centric users must be positive");
            }
            PhaseSpec::FixedSchedule { .. } => {}
        }
    }
    Ok(())
}

struct DynamoOfflinePreparedConversationSourceFactory {
    endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    samplers: SamplerRegistry,
}

impl NativeConversationSourceFactory for DynamoOfflinePreparedConversationSourceFactory {
    fn build(
        &self,
        dataset: aiperf_dataset::Dataset,
        model: String,
        default_output_tokens: usize,
        rng_root: RngRoot,
        tokenizer: Arc<dyn TextTokenizer>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        sequential: bool,
    ) -> Result<Box<dyn ConversationSource>> {
        let source = if sequential {
            NativeDatasetConversationSource::sequential_with_prepared_resolver(
                dataset,
                model,
                default_output_tokens,
                self.endpoint_resolver.clone(),
            )?
        } else {
            NativeDatasetConversationSource::preferred_with_prepared_resolver(
                dataset,
                model,
                default_output_tokens,
                rng_root,
                &self.samplers,
                self.endpoint_resolver.clone(),
            )?
        };
        Ok(Box::new(
            source
                .with_request_materializer(Arc::new(TraceHashAwareRequestMaterializer))
                .with_response_tokenizer(tokenizer)
                .with_input_token_counter(input_token_counter),
        ))
    }
}

struct PreparedDynamoOfflineScheduledOperation {
    backend: ValidatedDynamoOfflineBackend,
    dataset: PreparedDatasetInput,
    source_factory: DynamoOfflinePreparedConversationSourceFactory,
    tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    phases: Vec<PhaseSpec>,
    metrics: MetricsConfig,
    model: String,
    rng_root: RngRoot,
    artifact_target: PathBuf,
    benchmark_id: String,
}

impl fmt::Debug for PreparedDynamoOfflineScheduledOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedDynamoOfflineScheduledOperation")
            .field("phase_count", &self.phases.len())
            .field("model", &self.model)
            .field("artifact_target", &self.artifact_target)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedDynamoOfflineScheduledOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let Self {
            backend,
            dataset,
            source_factory,
            tokenizer,
            input_token_counter,
            phases,
            metrics,
            model,
            rng_root,
            artifact_target,
            benchmark_id,
        } = *self;
        let dataset_rng_root = dataset
            .random_seed
            .map_or(rng_root, |seed| RngRoot::new(Some(seed)));
        let default_output_tokens = dataset.default_output_tokens;
        let dataset = dataset.dataset;
        create_artifact_target(&artifact_target)?;

        let phase_count = phases.len();
        let backend_sla_metrics = backend.sla.native_metrics_config()?;
        let artifact_for_factory = artifact_target.clone();
        let benchmark_for_factory = benchmark_id.clone();
        let model_for_factory = model.clone();
        let factory: Box<dyn DeferredOfflineScheduledRunFactory> =
            Box::new(move |clock: Rc<dyn Clock>, start_ns, dispatcher| {
                Box::pin(async move {
                    let shared = native_scheduled_resources(&phases);
                    let backend_goodput = backend_sla_metrics.map(|config| {
                        Rc::new(NativeMetricsObserver::new(clock.clone(), start_ns, config))
                    });
                    let mut plans = Vec::with_capacity(phases.len());
                    for (phase_index, phase) in phases.iter().enumerate() {
                        let mut plan = build_native_scheduled_phase_plan_with_source_factory(
                            phase_index,
                            phase,
                            &dataset,
                            &model_for_factory,
                            default_output_tokens,
                            dataset_rng_root,
                            rng_root,
                            &source_factory,
                            tokenizer.clone(),
                            input_token_counter.clone(),
                            clock.clone(),
                            start_ns,
                            &benchmark_for_factory,
                            &artifact_for_factory,
                            &["dynamo-offline".to_owned()],
                            &shared,
                        )?
                        .with_metrics_config(metrics.clone())
                        .with_performance_record_capture(false);
                        if phase.common().name == "profiling"
                            && let Some(observer) = &backend_goodput
                        {
                            let observer: Rc<dyn RequestObserver> = observer.clone();
                            plan = plan.with_additional_observers(vec![observer]);
                        }
                        plans.push(plan);
                    }
                    let aggregate = run_scheduled_phases_with_aggregate_deferred(
                        plans,
                        clock.clone(),
                        start_ns,
                        dispatcher,
                        Rc::new(NoopPhaseObserver),
                    )
                    .await?;
                    let finalizer: Box<dyn OfflineScheduledExecutionFinalizer> =
                        Box::new(move || {
                            let aggregate = aggregate.finish();
                            let mut execution = OfflineScheduledExecution::phased(
                                aggregate.phased,
                                aggregate.performance,
                            )?;
                            if let Some(observer) = backend_goodput {
                                let goodput = observer.finish();
                                for tag in [
                                    MetricTag::GoodRequestCount,
                                    MetricTag::Goodput,
                                    MetricTag::GoodRequestFraction,
                                ] {
                                    if let Some(value) = goodput.finite_value(tag) {
                                        execution
                                            .profiling
                                            .native_metrics
                                            .insert_finite(tag, value);
                                    }
                                }
                            }
                            Ok(execution)
                        });
                    Ok(finalizer)
                }) as DeferredOfflineScheduledFuture
            });
        let outcome = backend
            .executor(model.clone(), &artifact_target)?
            .execute_scheduled_deferred(factory)?;
        let warmup = outcome
            .report
            .auxiliary_phase_reports
            .iter()
            .find(|report| report.kind == aiperf_timing::PhaseKind::Warmup)
            .map(|report| report.report.native_metrics.clone());
        let native_report = NativeReport::from_outcome(
            &outcome.report.aiperf.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some("offline:scheduled".into()),
                    model: Some(model),
                },
                summary: ReportSummary {
                    endpoints_configured: vec!["dynamo://offline".into()],
                    ..ReportSummary::default()
                },
                warmup,
                ..RunOutcome::default()
            },
        );
        let report_facts = ReportPairRunFacts::new().with_dynamo(dynamo_report_facts(
            &backend,
            outcome.report.parity,
            &outcome.report.performance,
        )?);
        let mut provenance = outcome.provenance;
        provenance.insert("workload".into(), "scheduled".into());
        provenance.insert("phase_count".into(), phase_count.to_string());
        provenance.insert("benchmark_id".into(), benchmark_id);
        Ok(PreparedRunOutcome {
            native_report,
            report_facts,
            provenance,
            report_commit: None,
        })
    }
}

fn validate_direct_graph_phase(phase: &PhaseSpec) -> Result<()> {
    let common = phase.common();
    ensure!(
        common.name == "profiling" && !common.exclude_from_results,
        "the current offline direct-graph pair requires one profiling phase"
    );
    ensure!(
        phase.request_arrival().is_some(),
        "offline direct graph supports concurrency, poisson, gamma, or constant scheduling"
    );
    ensure!(
        common.concurrency_ramp.is_none()
            && common.prefill_ramp.is_none()
            && common.rate_ramp.is_none(),
        "offline direct graph does not yet support actuator ramps"
    );
    ensure!(
        common.adaptive_scale.is_none(),
        "offline direct graph does not yet support adaptive scale"
    );
    ensure!(
        !common.seamless,
        "offline direct graph does not support seamless handoff"
    );
    ensure!(
        common.grace_period.is_none(),
        "offline direct graph drains admitted roots and does not accept grace_period"
    );
    ensure!(
        common.requests != Some(0),
        "phase requests must be positive"
    );
    ensure!(
        common.sessions != Some(0),
        "phase sessions must be positive"
    );
    if let Some(duration) = common.duration {
        let _ = seconds_to_ns(duration, "phase duration")?;
        ensure!(duration > 0.0, "phase duration must be positive");
    }
    if let Some(concurrency) = phase.concurrency() {
        ensure!(concurrency > 0, "phase concurrency must be positive");
    }
    if let Some(prefill) = common.prefill_concurrency {
        ensure!(prefill > 0, "phase prefill_concurrency must be positive");
        if let Some(concurrency) = phase.concurrency() {
            ensure!(
                prefill <= concurrency,
                "phase prefill_concurrency must be <= concurrency"
            );
        }
    }
    match phase {
        PhaseSpec::Concurrency { .. } => {}
        PhaseSpec::Poisson { rate, .. } | PhaseSpec::Constant { rate, .. } => ensure!(
            rate.is_finite() && *rate > 0.0,
            "rate graph phase requires a finite positive rate"
        ),
        PhaseSpec::Gamma {
            rate, smoothness, ..
        } => {
            ensure!(
                rate.is_finite() && *rate > 0.0,
                "rate graph phase requires a finite positive rate"
            );
            ensure!(
                smoothness.is_none_or(|value| value.is_finite() && value > 0.0),
                "gamma smoothness must be finite and positive"
            );
        }
        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => {
            unreachable!("unsupported phase kind rejected above")
        }
    }
    if let Some(cancellation) = common.cancellation {
        ensure!(
            cancellation.rate.is_finite() && (0.0..=100.0).contains(&cancellation.rate),
            "phase cancellation.rate must be finite and in 0..=100"
        );
        ensure!(
            cancellation.delay.is_finite() && cancellation.delay >= 0.0,
            "phase cancellation.delay must be finite and non-negative"
        );
    }
    Ok(())
}

fn seconds_to_ns(value: f64, name: &str) -> Result<i64> {
    ensure!(
        value.is_finite() && value >= 0.0 && value * 1_000_000_000.0 <= i64::MAX as f64,
        "{name} must be finite, non-negative, and representable in nanoseconds"
    );
    Ok((value * 1_000_000_000.0).round_ties_even() as i64)
}

#[derive(Clone)]
struct DynamoOfflineGraphPairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl Default for DynamoOfflineGraphPairFactory {
    fn default() -> Self {
        Self {
            tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
        }
    }
}

impl fmt::Debug for DynamoOfflineGraphPairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DynamoOfflineGraphPairFactory")
            .finish_non_exhaustive()
    }
}

impl RunnerPairFactory for DynamoOfflineGraphPairFactory {
    fn backend_id(&self) -> &'static str {
        DYNAMO_OFFLINE_BACKEND_ID
    }

    fn workload_id(&self) -> &'static str {
        "graph"
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        let _ = validated_dynamo_offline_backend(backend)?;
        let workload = validated_graph_workload(workload)?;
        ensure!(
            workload.worker_count == 1,
            "dynamo_offline direct graph uses one LocalSet around one globally contended engine; worker_count must be 1"
        );
        validate_authored_tokenizer(&workload.tokenizer)?;
        ensure!(
            workload.phases.len() == 1,
            "dynamo_offline direct graph currently requires exactly one profiling phase"
        );
        validate_direct_graph_phase(&workload.phases[0])?;
        Ok(())
    }

    fn validate_run(
        &self,
        _run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        let workload = validated_graph_workload(workload)?;
        context
            .graph_inputs()
            .validate_identity(&workload.dataset)?;
        ensure!(
            context.sidecar_inputs().is_empty(),
            "dynamo_offline graph execution does not support online sidecars"
        );
        Ok(())
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        Err(anyhow!(
            "dynamo_offline + graph preparation requires the coordinator-owned RunnerRunContext"
        ))
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        let backend = validated_dynamo_offline_backend(backend.as_ref())?.clone();
        let workload = validated_graph_workload(workload.as_ref())?;
        let phase = workload.phases[0].clone();
        ensure!(
            run.models.items.len() == 1
                && matches!(run.models.strategy, ModelSelectionStrategy::RoundRobin),
            "dynamo_offline direct graph requires exactly one round_robin model"
        );
        ensure!(
            run.artifacts.records_path.is_none()
                && run.artifacts.raw_path.is_none()
                && run.artifacts.outputs_path.is_none()
                && !run.artifacts.trace
                && run.artifacts.user_files.is_empty(),
            "dynamo_offline direct graph does not yet project common request/raw/output/user-file artifacts; use backend Dynamo artifacts or disable them"
        );
        ensure!(
            !run.artifact_target.exists(),
            "artifact_target already exists; protocol-v2 execution requires an exclusive uncreated target"
        );
        let model = run.models.items[0].name.clone();
        let tokenizer_spec =
            lower_authored_tokenizer(&workload.tokenizer, self.tokenizers.as_ref())?;
        let tokenizer = load_tokenizer(Some(&tokenizer_spec.name))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating direct graph preparation runtime")?;
        let prepared = runtime
            .block_on(context.graph_inputs().load(
                &workload.dataset,
                &RunnerGraphInputContext {
                    tokenizer: tokenizer.as_ref(),
                },
            ))
            .context("loading direct authored Graph-IR input")?;
        let metrics = offline_metrics_config(&run.metrics)?;
        let default_max_tokens = prepared.default_output_tokens;
        let random_seed = prepared.random_seed.or(run.identity.random_seed);
        Ok(Box::new(PreparedDynamoOfflineGraphOperation {
            backend,
            input: prepared.bundle,
            phase,
            metrics,
            model,
            random_seed,
            artifact_target: run.artifact_target.clone(),
            default_max_tokens,
            worker_count: workload.worker_count,
            phase_count: workload.phases.len(),
        }))
    }
}

fn validated_graph_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&GraphWorkloadConfigV2> {
    config
        .as_any()
        .downcast_ref::<GraphWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("dynamo_offline graph pair received a different workload type"))
}

struct PreparedDynamoOfflineGraphOperation {
    backend: ValidatedDynamoOfflineBackend,
    input: GraphInputBundle,
    phase: PhaseSpec,
    metrics: MetricsConfig,
    model: String,
    random_seed: Option<u64>,
    artifact_target: PathBuf,
    default_max_tokens: usize,
    worker_count: usize,
    phase_count: usize,
}

impl fmt::Debug for PreparedDynamoOfflineGraphOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedDynamoOfflineGraphOperation")
            .field("roots", &self.input.plans.len())
            .field("model", &self.model)
            .field("artifact_target", &self.artifact_target)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedDynamoOfflineGraphOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let Self {
            backend,
            input,
            phase,
            metrics,
            model,
            random_seed,
            artifact_target,
            default_max_tokens,
            worker_count,
            phase_count,
        } = *self;
        create_artifact_target(&artifact_target)?;
        let graph_report = ReportGraphRunInfo::new(
            input.metadata.format.clone(),
            input.metadata.root_count,
            input.metadata.node_count,
            worker_count,
            phase_count,
        )?;
        let rng_root = RngRoot::new(random_seed);
        let node_policy = graph_node_policy(&phase, rng_root)?;
        let workload = graph_workload_factory(input.plans, &phase, rng_root)?;
        let outcome = backend
            .executor(model.clone(), &artifact_target)?
            .execute_graph(
                input.segments,
                default_max_tokens,
                metrics,
                node_policy,
                Rc::new(AbortTraceNodeFailurePolicy),
                workload,
            )?;
        if outcome.report.workload.failed > 0 {
            let detail = outcome
                .report
                .workload
                .traces
                .iter()
                .find_map(|trace| {
                    trace
                        .result
                        .as_ref()
                        .err()
                        .map(|error| format!("trace {:?}: {error}", trace.trace_id))
                })
                .unwrap_or_else(|| "unknown trace failure".into());
            return Err(anyhow!(
                "offline graph aborted after {} failed root(s): {detail}",
                outcome.report.workload.failed
            ));
        }

        let native_report = NativeReport::from_outcome(
            &outcome.report.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some("offline:graph".into()),
                    model: Some(model),
                },
                summary: ReportSummary {
                    duration_s: Some(outcome.report.performance.throughput.wall_time_ms / 1_000.0),
                    ..ReportSummary::default()
                },
                ..RunOutcome::default()
            },
        );
        let graph_report = graph_report.with_outcome(ReportGraphOutcomeInfo::new(
            outcome.report.workload.admitted,
            outcome.report.workload.completed,
            outcome.report.workload.failed,
        ))?;
        let report_facts = ReportPairRunFacts::new()
            .with_graph(graph_report)
            .with_dynamo(dynamo_report_facts(
                &backend,
                outcome.report.parity,
                &outcome.report.performance,
            )?);
        let mut provenance = outcome.provenance;
        provenance.insert("workload".into(), "graph".into());
        provenance.insert("graph_input".into(), input.metadata.format);
        provenance.insert("graph_roots".into(), input.metadata.root_count.to_string());
        provenance.insert("graph_nodes".into(), input.metadata.node_count.to_string());
        Ok(PreparedRunOutcome {
            native_report,
            report_facts,
            provenance,
            report_commit: None,
        })
    }
}

fn graph_node_policy(
    phase: &PhaseSpec,
    rng_root: RngRoot,
) -> Result<Option<Rc<dyn NodeDispatchPolicy>>> {
    let common = phase.common();
    let mut policies = Vec::<Rc<dyn NodeDispatchPolicy>>::new();
    if let Some(limit) = common.prefill_concurrency {
        policies.push(Rc::new(PrefillSlotNodePolicy::new(Rc::new(SlotPool::new(
            limit,
        )))));
    }
    if let Some(cancellation) = common.cancellation {
        policies.push(Rc::new(CancellationNodePolicy::new(
            Box::new(BernoulliFixedDelay::new(
                Some(cancellation.rate),
                cancellation.delay,
                RngRoot::new(rng_root.derive_seed("runner.offline.graph.cancellation")),
            )?),
            Phase::Profiling,
        )));
    }
    Ok(match policies.len() {
        0 => None,
        1 => policies.pop(),
        _ => Some(Rc::new(CompositeNodeDispatchPolicy::new(policies))),
    })
}

fn graph_workload_factory(
    plans: Vec<aiperf_graph::model::GraphTracePlan>,
    phase: &PhaseSpec,
    rng_root: RngRoot,
) -> Result<Box<dyn OfflineGraphRunFactory>> {
    let phase = phase.clone();
    let common = phase.common();
    let one_pass =
        common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
    let session_limit = if one_pass {
        Some(u64::try_from(plans.len()).context("graph root count exceeds u64")?)
    } else {
        common.sessions
    };
    let arrival_seed = rng_root
        .derive_seed("runner.offline.graph.arrival")
        .unwrap_or(0);
    Ok(Box::new(move |clock, backend| {
        let common = phase.common();
        let source: Rc<dyn GraphTraceSource> =
            Rc::new(CyclingGraphTraceSource::with_budgets_and_sequence(
                plans,
                session_limit,
                common.requests,
                GraphTraceInstanceSequence::default(),
            )?);
        let (arrival_pattern, rate, smoothness) = phase
            .request_arrival()
            .ok_or_else(|| anyhow!("unsupported direct graph arrival policy"))?;
        let arrival: Rc<dyn GraphArrivalPolicy> = if matches!(&phase, PhaseSpec::Concurrency { .. })
        {
            Rc::new(ImmediateGraphArrival)
        } else {
            Rc::new(IntervalGraphArrival::new(Rc::new(RefCell::new(
                make_interval_generator(arrival_pattern, rate, smoothness, arrival_seed),
            ))))
        };
        let mut workload = GraphWorkload::new(clock, source, backend)
            .with_arrival(arrival)
            .with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
        if let Some(concurrency) = phase.concurrency() {
            workload = workload.with_admission(Rc::new(SlotPoolTraceAdmission::new(Rc::new(
                SlotPool::new(concurrency),
            ))));
        }
        if let Some(duration) = common.duration {
            workload = workload.with_stop_policy(Rc::new(DurationGraphStop::new(seconds_to_ns(
                duration,
                "phase duration",
            )?)?));
        }
        Ok(workload)
    }))
}

fn offline_metrics_config(spec: &MetricsSpec) -> Result<MetricsConfig> {
    let slice_duration_ns = spec
        .slice_duration_seconds
        .map(|seconds| {
            ensure!(seconds > 0.0, "metrics slice duration must be positive");
            seconds_to_ns(seconds, "metrics slice duration")
        })
        .transpose()?;
    let mut slos = Vec::with_capacity(spec.slos.len());
    for (name, value) in &spec.slos {
        ensure!(value.is_finite(), "SLO {name:?} threshold must be finite");
        let metric = CATALOG
            .iter()
            .find(|metric| metric.tag.as_str() == name)
            .ok_or_else(|| anyhow!("SLO metric {name:?} is not in the native metric catalog"))?;
        ensure!(
            metric.kind == MetricType::Record
                && !metric.flags.contains(MetricFlags::NO_INDIVIDUAL_RECORDS),
            "SLO metric {name:?} does not produce one value per request"
        );
        slos.push(SloThreshold::from_display(metric.tag, *value)?);
    }
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        ..MetricsConfig::default()
    })
}

/// Resolved backend artifact paths committed after successful execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamoOfflineArtifactOutputs {
    /// Canonical aggregate Dynamo report, when requested.
    pub report_json: Option<PathBuf>,
    /// Canonical per-request Dynamo records, when requested.
    pub per_request_jsonl: Option<PathBuf>,
    /// Timed worker artifacts, when requested for a trace workload.
    pub worker_artifacts_json: Option<PathBuf>,
}

/// Prepared offline backend adapter shared by scheduled and graph pair factories.
///
/// Construction and static validation perform no IO. Each execution method
/// initializes exactly one library-owned engine/clock composition, waits for
/// its exact parity proof, and only then writes optional backend artifacts.
#[derive(Clone, Debug)]
pub struct DynamoOfflineExecutor {
    engine: OfflineEngineConfig,
    artifacts: DynamoOfflineArtifactSpec,
    topology: DynamoOfflineTopologySpec,
    router_mode: DynamoOfflineRouterSpec,
    required_features: BTreeSet<DynamoBuildFeature>,
    model: String,
    artifact_target: PathBuf,
}

impl DynamoOfflineExecutor {
    /// Run any registered scheduled workload over the shared virtual clock and
    /// dispatcher supplied by `aiperf::dynamo_offline`.
    pub fn execute_scheduled(
        self,
        workload: Box<dyn OfflineScheduledRunFactory>,
    ) -> Result<DynamoOfflineScheduledOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let report =
            run_scheduled_backend_offline(self.engine.clone(), self.model.clone(), workload)?;
        verify_parity(&report.performance, &report.dynamo, report.parity)?;
        let artifacts = self.emit_backend_artifacts(
            |path| write_dynamo_report_json(&report.dynamo, path),
            |path| write_dynamo_per_request_jsonl(&report.dynamo, path),
        )?;
        let provenance = self.provenance(report.parity);
        Ok(DynamoOfflineScheduledOutcome {
            report,
            artifacts,
            provenance,
        })
    }

    /// Run a scheduled workload whose deterministic observer reduction occurs
    /// after the virtual-time driver and its Tokio `LocalSet` have exited.
    pub fn execute_scheduled_deferred(
        self,
        workload: Box<dyn DeferredOfflineScheduledRunFactory>,
    ) -> Result<DynamoOfflineScheduledOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let report = run_scheduled_backend_offline_deferred(
            self.engine.clone(),
            self.model.clone(),
            workload,
        )?;
        verify_parity(&report.performance, &report.dynamo, report.parity)?;
        let artifacts = self.emit_backend_artifacts(
            |path| write_dynamo_report_json(&report.dynamo, path),
            |path| write_dynamo_per_request_jsonl(&report.dynamo, path),
        )?;
        let provenance = self.provenance(report.parity);
        Ok(DynamoOfflineScheduledOutcome {
            report,
            artifacts,
            provenance,
        })
    }

    /// Run the existing generated Graph-IR benchmark entrypoint without HTTP.
    ///
    /// Direct authored Graph-IR pair factories should use the direct-plan API
    /// when it is supplied by the graph workload module; this method preserves
    /// the existing library benchmark consumer and its DES/parity gates.
    pub fn execute_graph_bench(self, mut config: BenchConfig) -> Result<DynamoOfflineGraphOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        config.model.clone_from(&self.model);
        let report = run_graph_offline(self.engine.clone(), config)?;
        verify_parity(&report.performance, &report.dynamo, report.parity)?;
        let artifacts = self.emit_backend_artifacts(
            |path| write_dynamo_report_json(&report.dynamo, path),
            |path| write_dynamo_per_request_jsonl(&report.dynamo, path),
        )?;
        let provenance = self.provenance(report.parity);
        Ok(DynamoOfflineGraphOutcome {
            report,
            artifacts,
            provenance,
        })
    }

    /// Execute a direct authored Graph-IR workload over the shared SimClock and
    /// steppable engine, without the generated benchmark graph or a linear
    /// dataset conversion.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_graph(
        self,
        segments: Arc<dyn SegmentStore>,
        default_max_tokens: usize,
        metrics: MetricsConfig,
        node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
        node_failure: Rc<dyn NodeFailurePolicy>,
        workload: Box<dyn OfflineGraphRunFactory>,
    ) -> Result<DynamoOfflineDirectGraphOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let report = run_graph_workload_offline(
            self.engine.clone(),
            self.model.clone(),
            segments,
            default_max_tokens,
            metrics,
            node_policy,
            node_failure,
            workload,
        )?;
        verify_parity(&report.performance, &report.dynamo, report.parity)?;
        if report.workload.failed > 0 {
            let detail = report
                .workload
                .traces
                .iter()
                .find_map(|trace| {
                    trace
                        .result
                        .as_ref()
                        .err()
                        .map(|error| format!("trace {:?}: {error}", trace.trace_id))
                })
                .unwrap_or_else(|| "unknown trace failure".into());
            return Err(anyhow!(
                "offline graph aborted after {} failed root(s): {detail}",
                report.workload.failed
            ));
        }
        let artifacts = self.emit_backend_artifacts(
            |path| write_dynamo_report_json(&report.dynamo, path),
            |path| write_dynamo_per_request_jsonl(&report.dynamo, path),
        )?;
        let provenance = self.provenance(report.parity);
        Ok(DynamoOfflineDirectGraphOutcome {
            report,
            artifacts,
            provenance,
        })
    }

    /// Run one canonical Dynamo trace workload through AIPerf's observer and
    /// native metrics stack, retaining the complete byte-exact parity proof.
    pub fn execute_trace(self, trace: OfflineTraceConfig) -> Result<DynamoOfflineTraceOutcome> {
        let report = run_trace_offline(self.engine.clone(), trace.clone())?;
        verify_parity(&report.aiperf.performance, &report.dynamo, report.parity)?;
        let mut artifacts = self.emit_backend_artifacts(
            |path| write_dynamo_report_json(&report.dynamo, path),
            |path| write_dynamo_per_request_jsonl(&report.dynamo, path),
        )?;
        if let Some(relative) = &self.artifacts.worker_artifacts_json {
            let path = self.artifact_path(relative);
            prepare_output_parent(&path)?;
            ensure!(
                !path.exists(),
                "offline worker artifact target already exists: {}",
                path.display()
            );
            write_dynamo_worker_artifacts_json(
                &self.engine,
                &trace,
                self.artifacts.kv_event_visibility.map(Into::into),
                &path,
            )?;
            artifacts.worker_artifacts_json = Some(path);
        }
        let provenance = self.provenance(report.parity);
        Ok(DynamoOfflineTraceOutcome {
            report,
            artifacts,
            provenance,
        })
    }

    fn emit_backend_artifacts(
        &self,
        write_report: impl FnOnce(&Path) -> Result<()>,
        write_records: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<DynamoOfflineArtifactOutputs> {
        let report_json = self
            .artifacts
            .report_json
            .as_deref()
            .map(|relative| self.artifact_path(relative));
        let per_request_jsonl = self
            .artifacts
            .per_request_jsonl
            .as_deref()
            .map(|relative| self.artifact_path(relative));
        for path in [report_json.as_deref(), per_request_jsonl.as_deref()]
            .into_iter()
            .flatten()
        {
            ensure!(
                !path.exists(),
                "offline backend artifact target already exists: {}",
                path.display()
            );
            prepare_output_parent(path)?;
        }
        if let Some(path) = &report_json {
            write_report(path)?;
        }
        if let Some(path) = &per_request_jsonl {
            write_records(path)?;
        }
        Ok(DynamoOfflineArtifactOutputs {
            report_json,
            per_request_jsonl,
            worker_artifacts_json: None,
        })
    }

    fn artifact_path(&self, relative: &Path) -> PathBuf {
        self.artifact_target.join(relative)
    }

    fn provenance(&self, parity: OfflineMetricParity) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("backend".into(), DYNAMO_OFFLINE_BACKEND_ID.into()),
            ("clock".into(), "sim".into()),
            ("topology".into(), self.topology.as_str().into()),
            ("router".into(), self.router_mode.as_str().into()),
            (
                "required_features".into(),
                self.required_features
                    .iter()
                    .map(|feature| feature.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            (
                "parity_shared_fields".into(),
                parity.shared_fields.to_string(),
            ),
            (
                "parity_independent_fields".into(),
                parity.independently_accumulated_fields.to_string(),
            ),
            (
                "parity_backend_owned_fields".into(),
                parity.backend_owned_fields.to_string(),
            ),
            (
                "parity_serialized_bytes".into(),
                parity.serialized_bytes.to_string(),
            ),
        ])
    }
}

fn prepare_output_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating offline artifact directory {}", parent.display()))?;
    }
    Ok(())
}

fn create_artifact_target(path: &Path) -> Result<()> {
    ensure!(
        !path.exists(),
        "artifact_target already exists: {}",
        path.display()
    );
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating offline artifact parent {}", parent.display()))?;
    }
    std::fs::create_dir(path).with_context(|| {
        format!(
            "creating exclusive offline artifact target {}",
            path.display()
        )
    })
}

fn verify_parity(
    aiperf: &impl CanonicalSharedMetrics,
    dynamo: &impl CanonicalSharedMetrics,
    parity: OfflineMetricParity,
) -> Result<()> {
    let aiperf_bytes = aiperf
        .canonical_shared_metric_bytes()
        .context("serializing runner-side AIPerf parity summary")?;
    let dynamo_bytes = dynamo
        .canonical_shared_metric_bytes()
        .context("serializing runner-side Dynamo parity summary")?;
    ensure!(
        aiperf_bytes == dynamo_bytes,
        "offline library returned mismatched AIPerf/Dynamo summaries to the runner adapter"
    );
    ensure!(
        parity.independently_accumulated_fields + parity.backend_owned_fields
            == parity.shared_fields,
        "offline parity evidence has inconsistent field accounting"
    );
    ensure!(
        parity.serialized_bytes == aiperf_bytes.len(),
        "offline parity evidence has an inconsistent serialized byte count"
    );
    ensure!(parity.shared_fields > 0, "offline parity schema is empty");
    Ok(())
}

/// Successful scheduled offline execution and backend-owned outputs.
pub struct DynamoOfflineScheduledOutcome {
    /// AIPerf scheduled metrics plus Dynamo's co-observed report.
    pub report: OfflineScheduledReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynamoOfflineArtifactOutputs,
    /// Additive terminal/native-report provenance.
    pub provenance: BTreeMap<String, String>,
}

/// Successful generated Graph-IR offline execution and outputs.
pub struct DynamoOfflineGraphOutcome {
    /// Graph metrics plus Dynamo's co-observed report.
    pub report: OfflineGraphReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynamoOfflineArtifactOutputs,
    /// Additive terminal/native-report provenance.
    pub provenance: BTreeMap<String, String>,
}

/// Successful direct authored Graph-IR offline execution and outputs.
pub struct DynamoOfflineDirectGraphOutcome {
    /// Root/trace outcomes, request metrics, and Dynamo parity report.
    pub report: OfflineDirectGraphReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynamoOfflineArtifactOutputs,
    /// Additive terminal/native-report provenance.
    pub provenance: BTreeMap<String, String>,
}

/// Successful canonical trace execution and outputs.
pub struct DynamoOfflineTraceOutcome {
    /// AIPerf metrics plus Dynamo's co-observed report.
    pub report: OfflineRunReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynamoOfflineArtifactOutputs,
    /// Additive terminal/native-report provenance.
    pub provenance: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{
        BuiltinRunnerRegistryFactory, RunnerBackendFactory, RunnerRegistryFactory,
    };

    fn raw(value: Value) -> Box<RawValue> {
        RawValue::from_string(value.to_string()).unwrap()
    }

    fn validate(value: Value) -> Result<Box<dyn ValidatedBackendConfig>> {
        DynamoOfflineBackendFactory.validate(&raw(value), &WorkloadRequirements::default())
    }

    #[test]
    fn strict_config_rejects_unknown_fields_and_invalid_worker_counts() {
        let error = validate(serde_json::json!({"unknown": true}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field `unknown`"), "{error}");

        let error = validate(serde_json::json!({"workers": 0}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("workers must be positive"), "{error}");
    }

    #[test]
    fn disaggregated_profiles_are_transactional() {
        let error = validate(serde_json::json!({
            "topology": "disaggregated",
            "prefill_engine": {"worker_type": "prefill"}
        }))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("prefill_engine and decode_engine must be authored together"),
            "{error}"
        );
    }

    #[test]
    fn artifact_paths_cannot_escape_the_run_target() {
        let error = validate(serde_json::json!({
            "artifacts": {"report_json": "../outside.json"}
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("normal relative path components"), "{error}");
    }

    #[test]
    fn requested_uncompiled_optional_features_fail_static_validation() {
        if !cfg!(feature = "dynamo-kvbm-offload") {
            let error = validate(serde_json::json!({
                "engine": {"num_g2_blocks": 8}
            }))
            .unwrap_err()
            .to_string();
            assert!(error.contains("dynamo-kvbm-offload"), "{error}");
        }
        if !cfg!(feature = "dynamo-aic-forward-pass") {
            let error = validate(serde_json::json!({
                "aic": {
                    "backend": "vllm",
                    "system": "h200_sxm",
                    "model_path": "Qwen/Qwen3-0.6B"
                }
            }))
            .unwrap_err()
            .to_string();
            assert!(error.contains("dynamo-aic-forward-pass"), "{error}");
        }
    }

    #[test]
    fn factory_registration_is_derived_from_the_feature_bearing_registry() {
        let registry = BuiltinRunnerRegistryFactory.build().unwrap();
        let descriptor = registry
            .backend_descriptors()
            .into_iter()
            .find(|descriptor| descriptor.id == DYNAMO_OFFLINE_BACKEND_ID)
            .unwrap();
        assert!(descriptor.features.contains(&"exact_metric_parity"));
        assert!(
            registry
                .supported_pairs()
                .contains(&(DYNAMO_OFFLINE_BACKEND_ID, "graph"))
        );
        assert!(
            registry
                .supported_pairs()
                .contains(&(DYNAMO_OFFLINE_BACKEND_ID, "scheduled"))
        );
    }

    #[test]
    fn graph_adapter_runs_without_http_and_commits_exact_backend_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "aiperf-runner-offline-graph-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        let validated = validate(serde_json::json!({
            "artifacts": {
                "report_json": "dynamo/report.json",
                "per_request_jsonl": "dynamo/requests.jsonl"
            }
        }))
        .unwrap();
        let backend = validated_dynamo_offline_backend(validated.as_ref()).unwrap();
        let outcome = backend
            .executor("model", &root)
            .unwrap()
            .execute_graph_bench(BenchConfig {
                base_urls: Vec::new(),
                model: "ignored".into(),
                turns: 2,
                instances: 2,
                workers: 1,
                concurrency: 2,
                max_tokens: 2,
                request_concurrency: None,
                prefill_concurrency: None,
                max_duration_ns: None,
            })
            .unwrap();
        assert_eq!(outcome.report.aiperf.completed, 4);
        assert_eq!(outcome.provenance["backend"], DYNAMO_OFFLINE_BACKEND_ID);
        let report_path = outcome.artifacts.report_json.unwrap();
        let records_path = outcome.artifacts.per_request_jsonl.unwrap();
        assert!(report_path.is_file());
        assert_eq!(
            std::fs::read_to_string(records_path)
                .unwrap()
                .lines()
                .count(),
            4
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_graph_plans_use_the_shared_segment_store_and_engine() {
        let (segments, graph, _) = aiperf_graph::bench::build_workload(2);
        let plan = aiperf_graph::model::GraphTracePlan {
            graph,
            trace: aiperf_graph::model::TraceRecord {
                id: "direct-root".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let factory: Box<dyn OfflineGraphRunFactory> = Box::new(move |clock, backend| {
            Ok(aiperf_graph::workload::GraphWorkload::new(
                clock,
                Rc::new(aiperf_graph::workload::VecGraphTraceSource::new([plan])),
                backend,
            ))
        });
        let validated = validate(serde_json::json!({})).unwrap();
        let backend = validated_dynamo_offline_backend(validated.as_ref()).unwrap();
        let outcome = backend
            .executor("model", "/tmp/aiperf-direct-graph-unused")
            .unwrap()
            .execute_graph(
                Arc::new(segments),
                2,
                MetricsConfig::default(),
                None,
                Rc::new(aiperf_graph::policy::AbortTraceNodeFailurePolicy),
                factory,
            )
            .unwrap();
        assert_eq!(outcome.report.workload.admitted, 1);
        assert_eq!(outcome.report.workload.completed, 1);
        assert_eq!(outcome.report.performance.request_counts.num_requests, 2);
        assert_eq!(
            outcome.report.performance.request_counts.completed_requests,
            2
        );
        assert_eq!(outcome.provenance["parity_shared_fields"], "74");
    }
}
