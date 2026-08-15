// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-gated composition for Dynamo's in-process execution engine.
//!
//! [`crate::dynosim`] owns clock driving, scheduling, cancellation, observation,
//! and AIPerf/Dynamo metric verification.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::clock::Clock;
use crate::dataset::{
    HashIdentityTracePromptStorage, NativeSyntheticMediaGeneratorFactory, SamplerRegistry,
    SegmentStore, TextTokenizer, TraceHashAwareRequestMaterializer,
};
use crate::dispatch::sink::RequestObserver;
use crate::dynosim::{
    CanonicalSharedMetrics, DeferredOfflineGraphFuture, DeferredOfflineGraphRunFactory,
    DeferredOfflineScheduledFuture, DeferredOfflineScheduledRunFactory,
    IncrementalOfflineEventDelivery, OfflineAicConfig, OfflineDirectGraphReport,
    OfflineEngineConfig, OfflineEventDeliveryPolicy, OfflineGraphBackendConfig,
    OfflineGraphBackendFactory, OfflineGraphEventSink, OfflineGraphExecution, OfflineGraphReport,
    OfflineGraphRequestRecord, OfflineGraphRunFactory, OfflineKvEventVisibility,
    OfflineMetricParity, OfflineRouterMode, OfflineRunReport, OfflineScheduledExecution,
    OfflineScheduledExecutionFinalizer, OfflineScheduledReport, OfflineTopology,
    OfflineTraceConfig, TerminalOfflineEventDelivery, run_graph_offline,
    run_graph_workload_offline, run_graph_workload_offline_deferred, run_graph_workload_online,
    run_graph_workload_online_deferred, run_scheduled_backend_offline_deferred_with_delivery,
    run_scheduled_backend_online_deferred_with_delivery, run_trace_offline,
    write_dynamo_per_request_jsonl, write_dynamo_report_json, write_dynamo_worker_artifacts_json,
};
use crate::endpoints::{Modality, PreparedEndpointTable};
use crate::failure::OnFailure;
use crate::graph::bench::BenchConfig;
use crate::graph::input::GraphInputBundle;
use crate::graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, NodeDispatchPolicy, NodeFailurePolicy,
};
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::{
    MetricTag, MetricsConfig, NativeReport, ReportClockKind, ReportDynamoCapacityInfo,
    ReportDynamoParityInfo, ReportDynamoRouter, ReportDynamoRunInfo, ReportDynamoTopology,
    ReportGraphOutcomeInfo, ReportGraphRunInfo, ReportPairRunFacts, ReportRunInfo, ReportSummary,
    RunOutcome, SloThreshold,
};
use crate::multiturn::{
    ConversationSource, EndpointInputTokenCounter, InputTokenCounter,
    NativeDatasetConversationSource, PreparedEndpointReference, PreparedEndpointTableResolver,
    PreparedTurnEndpointResolver,
};
use crate::phase_runtime::run_scheduled_phases_with_aggregate_deferred;
use crate::rng::{RngRoot, namespace};
use crate::timing::{BernoulliFixedDelay, DISABLED_PROGRESS_INTERVAL_NS, NoopPhaseObserver};
use anyhow::{Context, Result, anyhow, ensure};
use dynamo_mocker::replay::TraceSimulationReport as DynamoSimulationReport;
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

use crate::engine::dataset_input::{DatasetInputContext, PreparedDatasetInput};
use crate::engine::execute::{
    NativeConversationSourceFactory, build_native_scheduled_phase_plan_with_source_factory,
    load_tokenizer, metrics_config, native_scheduled_resources, phase_seamless_to_next,
    resolve_slice_duration_ns, resolve_slos,
};
use crate::engine::graph_execution::{GraphExecutionEvent, GraphExecutionEventSink};
use crate::engine::graph_input::GraphInputContext;
use crate::engine::graph_phase_runtime::{
    GraphPhaseBackendConfig, GraphPhaseBackendFactory, PreparedGraphPhaseBackend, run_graph_phases,
    validate_graph_phases,
};
use crate::engine::online_execution::{
    OnlineTokenizerSourceResolver, lower_authored_tokenizer, validate_authored_tokenizer,
};
use crate::engine::protocol::{MetricsSpec, ModelSelectionStrategy, PhaseSpec};
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::records::{CapturedModelOutput, CapturedRecord};
use crate::engine::registry::{
    ClockKind, GraphWorkloadConfigV2, PreparedRunOutcome, PreparedRunnerOperation, RunContext,
    ScheduledWorkloadConfigV2, TransportDescriptor, TransportFactory, ValidatedTransportConfig,
    ValidatedWorkloadConfig, WorkloadRequirements,
};

/// Stable runner-registry transport IDs for the in-process Dynamo engine.
///
/// The clock rides on the transport ID: `dynosim_offline` fast-forwards a
/// deterministic `SimClock` through the discrete-event pump for byte-exact
/// replay; `dynosim_online` drives the *same* passive engine under the real wall
/// clock (`drive_real_with_source`) for live-throughput measurement. Both open no
/// sockets and share one materialization/report path.
pub const DYNOSIM_OFFLINE_ID: &str = "dynosim_offline";
pub const DYNOSIM_ONLINE_ID: &str = "dynosim_online";

const DYNOSIM_TRANSPORT_FEATURES: &[&str] = &[
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
];

static DYNOSIM_OFFLINE_DESCRIPTOR: TransportDescriptor = TransportDescriptor {
    id: DYNOSIM_OFFLINE_ID,
    description: "Dynamo passive-engine co-simulation on one deterministic SimClock",
    clock: ClockKind::Sim,
    features: DYNOSIM_TRANSPORT_FEATURES,
    url_schemes: &["dynosim"],
};

static DYNOSIM_ONLINE_DESCRIPTOR: TransportDescriptor = TransportDescriptor {
    id: DYNOSIM_ONLINE_ID,
    description: "Dynamo passive-engine in-process replay under the real wall clock",
    clock: ClockKind::Real,
    features: DYNOSIM_TRANSPORT_FEATURES,
    url_schemes: &["dynosim"],
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
pub enum DynosimTopologySpec {
    /// One eventized aggregate worker without a routing choice.
    #[default]
    Single,
    /// Multiple aggregate workers behind the selected router.
    Aggregated,
    /// Separate prefill and decode worker pools.
    Disaggregated,
}

impl DynosimTopologySpec {
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

impl From<DynosimTopologySpec> for OfflineTopology {
    fn from(value: DynosimTopologySpec) -> Self {
        match value {
            DynosimTopologySpec::Single => Self::Single,
            DynosimTopologySpec::Aggregated => Self::Aggregated,
            DynosimTopologySpec::Disaggregated => Self::Disaggregated,
        }
    }
}

/// Authored router policy for routed offline topologies.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DynosimRouterSpec {
    /// Stable deterministic worker rotation.
    #[default]
    RoundRobin,
    /// Prefix-affinity/load-aware KV routing.
    Kv,
}

impl DynosimRouterSpec {
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

impl From<DynosimRouterSpec> for OfflineRouterMode {
    fn from(value: DynosimRouterSpec) -> Self {
        match value {
            DynosimRouterSpec::RoundRobin => Self::RoundRobin,
            DynosimRouterSpec::Kv => Self::Kv,
        }
    }
}

/// Strict structured AIConfigurator overrides for offline timing.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynosimAicSpec {
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

impl DynosimAicSpec {
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
pub struct DynosimSlaSpec {
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

impl DynosimSlaSpec {
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
pub struct DynosimArtifactSpec {
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

impl DynosimArtifactSpec {
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

/// Strict authored configuration owned by the `dynosim` transport.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynosimTransportSpec {
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
    pub aic: Option<DynosimAicSpec>,
    /// Capture backend per-request records even without a JSONL artifact.
    #[serde(default)]
    pub capture_per_request: bool,
    /// Canonical backend-owned goodput thresholds.
    #[serde(default)]
    pub sla: DynosimSlaSpec,
    /// Deployment topology.
    #[serde(default)]
    pub topology: DynosimTopologySpec,
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
    pub router_mode: DynosimRouterSpec,
    /// Optional build capabilities that must exist in the exact runner image.
    #[serde(default)]
    pub required_features: BTreeSet<DynamoBuildFeature>,
    /// Backend-owned output artifacts.
    #[serde(default)]
    pub artifacts: DynosimArtifactSpec,
}

const fn one() -> usize {
    1
}

impl DynosimTransportSpec {
    fn validate(self, online: bool) -> Result<ValidatedDynosimTransport> {
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
        if self.topology != DynosimTopologySpec::Disaggregated {
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
        if self.aic.as_ref().is_some_and(DynosimAicSpec::requested)
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
            .filter(DynosimAicSpec::requested)
            .map(DynosimAicSpec::into_runtime);
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
        Ok(ValidatedDynosimTransport {
            engine,
            artifacts,
            sla: self.sla,
            topology,
            router_mode,
            online,
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

/// Strictly validated runner transport state retained until pair preparation.
///
/// `online` is derived from the selected transport ID (`dynosim_online` ⇒
/// `true`, `dynosim_offline` ⇒ `false`), not from an authored field: the clock
/// axis rides on the transport ID.
#[derive(Clone, Debug)]
pub struct ValidatedDynosimTransport {
    engine: OfflineEngineConfig,
    artifacts: DynosimArtifactSpec,
    sla: DynosimSlaSpec,
    topology: DynosimTopologySpec,
    router_mode: DynosimRouterSpec,
    online: bool,
    required_features: BTreeSet<DynamoBuildFeature>,
}

impl ValidatedDynosimTransport {
    /// Report mode prefix for the selected clock axis.
    ///
    /// `"offline"` for deterministic virtual-clock replay, `"online"` for
    /// wall-clock in-process replay (`drive_real_with_source`).
    pub const fn mode_prefix(&self) -> &'static str {
        if self.online { "online" } else { "offline" }
    }

    /// Build a no-side-effect execution adapter rooted at the selected target.
    pub fn executor(
        &self,
        model: impl Into<String>,
        artifact_target: impl Into<PathBuf>,
    ) -> Result<DynosimExecutor> {
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
        Ok(DynosimExecutor {
            engine,
            artifacts: self.artifacts.clone(),
            topology: self.topology,
            router_mode: self.router_mode,
            online: self.online,
            required_features: self.required_features.clone(),
            model,
            artifact_target,
        })
    }
}

fn dynamo_report_facts(
    backend: &ValidatedDynosimTransport,
    parity: OfflineMetricParity,
    performance: &crate::dispatch::collector::TraceSimulationReport,
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
        if backend.online {
            ReportClockKind::Real
        } else {
            ReportClockKind::Sim
        },
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

/// Registered strict decoder for the feature-bearing offline/online transports.
///
/// One factory type serves both `dynosim_offline` and `dynosim_online`; the
/// `online` flag selects the descriptor (Sim vs Real clock) and is stamped onto
/// the validated config so the shared executor can pick the driver without a
/// wire `replay_mode` field.
#[derive(Debug, Clone, Copy)]
pub struct DynosimTransportFactory {
    online: bool,
}

impl DynosimTransportFactory {
    /// Deterministic virtual-clock (`dynosim_offline`) transport factory.
    pub const fn offline() -> Self {
        Self { online: false }
    }

    /// Wall-clock in-process (`dynosim_online`) transport factory.
    pub const fn online() -> Self {
        Self { online: true }
    }
}

impl TransportFactory for DynosimTransportFactory {
    fn descriptor(&self) -> &'static TransportDescriptor {
        if self.online {
            &DYNOSIM_ONLINE_DESCRIPTOR
        } else {
            &DYNOSIM_OFFLINE_DESCRIPTOR
        }
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn ValidatedTransportConfig>> {
        let spec = serde_json::from_str::<DynosimTransportSpec>(authored.get())
            .map_err(|error| anyhow!("invalid dynosim transport config: {error}"))?;
        Ok(Box::new(spec.validate(self.online)?))
    }
}

/// Add both offline and online Dynamo transports to a mutable runner registry.
///
/// Registers **2 transports** (`dynosim_offline`, `dynosim_online`). The
/// scheduled and graph workload factories resolve these transports by type and
/// dispatch to [`prepare_dynosim_scheduled`]/[`prepare_dynosim_graph`]; there is
/// no per-transport pair object. Direct graph preparation resolves its
/// authored-input adapter from the coordinator-owned [`RunContext`].
pub fn register_dynosim_transport(registry: &mut crate::extensions::AIPerfRegistry) -> Result<()> {
    registry.register_transport(Arc::new(DynosimTransportFactory::offline()))?;
    registry.register_transport(Arc::new(DynosimTransportFactory::online()))?;
    Ok(())
}

/// Downcast one pair-factory transport value with an actionable invariant error.
pub fn validated_dynosim_transport(
    config: &dyn ValidatedTransportConfig,
) -> Result<&ValidatedDynosimTransport> {
    config
        .as_any()
        .downcast_ref::<ValidatedDynosimTransport>()
        .ok_or_else(|| anyhow!("dynosim pair received a different transport config type"))
}

/// Reject a dynosim run that authored common (non-backend) artifacts.
///
/// dynosim paths project no request/raw/output/user-file artifacts of their own;
/// callers supply the path-specific `rejection` message so the scheduled and
/// direct-graph sites keep their distinct wording while sharing one predicate.
fn ensure_no_common_artifacts(run: &AuthoredRunSpecV2, rejection: &str) -> Result<()> {
    ensure!(
        run.artifacts.records_path.is_none()
            && run.artifacts.raw_path.is_none()
            && run.artifacts.outputs_path.is_none()
            && !run.artifacts.trace
            && run.artifacts.user_files.is_empty(),
        "{rejection}"
    );
    Ok(())
}

/// Run-level validation for a dynosim scheduled run.
///
/// Called by the scheduled workload factory when the resolved transport is a
/// dynosim transport (there is no per-transport pair object).
pub(crate) fn dynosim_scheduled_validate_run(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    transport: &dyn ValidatedTransportConfig,
    workload: &ScheduledWorkloadConfigV2,
) -> Result<()> {
    let _ = validated_dynosim_transport(transport)?;
    ensure!(
        workload.worker_count == 1,
        "dynosim scheduled execution owns one LocalSet around one globally contended engine; worker_count must be 1"
    );
    validate_offline_scheduled_phases(&workload.phases)?;
    ensure!(
        context.sidecar_inputs().is_empty(),
        "dynosim scheduled execution does not support online sidecars"
    );
    ensure_no_common_artifacts(
        run,
        "dynosim scheduled execution does not project common request/raw/output/user-file artifacts; use backend Dynamo artifacts or disable them",
    )?;
    // The artifact directory may already exist because the parent creates it for
    // logging. Backend artifacts still reject pre-existing output files.
    Ok(())
}

/// Prepare a dynosim scheduled operation from the coordinator-owned context.
pub(crate) fn prepare_dynosim_scheduled(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    transport: Box<dyn ValidatedTransportConfig>,
    workload: Box<dyn ValidatedWorkloadConfig>,
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
) -> Result<Box<dyn PreparedRunnerOperation>> {
    let backend = validated_dynosim_transport(transport.as_ref())?.clone();
    let workload = validated_scheduled_workload(workload.as_ref())?;
    validate_offline_scheduled_phases(&workload.phases)?;

    let tokenizer_spec = lower_authored_tokenizer(&workload.tokenizer, tokenizers.as_ref())?;
    let tokenizer = load_tokenizer(Some(&tokenizer_spec.name))?;
    let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(EndpointInputTokenCounter::new(
        tokenizer.clone(),
        tokenizer_spec.apply_chat_template,
    ));
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
    let endpoint_descriptor = prepared_endpoint.descriptor();
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
    let dataset_context = DatasetInputContext {
        registry: context.product_registry(),
        models: &run.models,
        run_rng_root: rng_root,
        tokenizer: tokenizer.as_ref(),
        rankings,
        endpoint_descriptor,
        trace_prompt_storage: Arc::new(HashIdentityTracePromptStorage),
        media_generator_factory: Arc::new(NativeSyntheticMediaGeneratorFactory::default()),
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
    // Offline co-simulation drives no real server, so server usage counts
    // never arrive; input accounting stays on the client tokenizer.
    let metrics = metrics_config(&run.metrics, false)?;
    let model = run
        .models
        .items
        .first()
        .map(|item| item.name.clone())
        .ok_or_else(|| anyhow!("dynosim scheduled execution requires a model"))?;

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

    Ok(Box::new(PreparedDynosimScheduledOperation {
        backend,
        dataset: prepared_dataset,
        source_factory: DynosimPreparedConversationSourceFactory {
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

fn validated_scheduled_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&ScheduledWorkloadConfigV2> {
    config
        .as_any()
        .downcast_ref::<ScheduledWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("dynosim scheduled pair received a different workload type"))
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
            PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. } => {}
        }
    }
    Ok(())
}

struct DynosimPreparedConversationSourceFactory {
    endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    samplers: SamplerRegistry,
}

impl NativeConversationSourceFactory for DynosimPreparedConversationSourceFactory {
    fn build(
        &self,
        dataset: crate::dataset::Dataset,
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

struct PreparedDynosimScheduledOperation {
    backend: ValidatedDynosimTransport,
    dataset: PreparedDatasetInput,
    source_factory: DynosimPreparedConversationSourceFactory,
    tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    phases: Vec<PhaseSpec>,
    metrics: MetricsConfig,
    model: String,
    rng_root: RngRoot,
    artifact_target: PathBuf,
    benchmark_id: String,
}

impl fmt::Debug for PreparedDynosimScheduledOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedDynosimScheduledOperation")
            .field("phase_count", &self.phases.len())
            .field("model", &self.model)
            .field("artifact_target", &self.artifact_target)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedDynosimScheduledOperation {
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

        // Match Dynamo's own offline replay: drive a single round-robin worker
        // with the `execute_pass` engine when the run is a closed-loop, single-
        // turn concurrency replay with no clock-scheduled events. `build_native`
        // additionally gates on backend single-worker eligibility, so setting the
        // flag on any other topology is a no-op. This makes offline concurrency
        // byte-exact with `dynamo.replay --replay-mode offline` under saturation.
        let mut backend = backend;
        backend.engine.single_pass_engine =
            dataset.is_single_turn() && phases.iter().all(phase_allows_single_pass_engine);

        let phase_count = phases.len();
        let single_pass_engine = backend.engine.single_pass_engine;
        tracing::info!(
            topology = backend.topology.as_str(),
            workers = backend.engine.workers,
            online = backend.online,
            single_pass_engine,
            phase_count,
            "offline scheduled execution starting"
        );
        let terminal_event_delivery = terminal_event_delivery_is_safe(&phases);
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
                            phase_seamless_to_next(&phases, phase_index),
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
                            &["dynosim".to_owned()],
                            &shared,
                            // Offline co-simulation feeds the callback observer
                            // directly, so the sampler needs no worker-record
                            // source (and must not be double-fed).
                            None,
                            // Co-simulation uses its configured resilient failure policy.
                            OnFailure::for_scheduled_default(),
                            // Dynamo co-simulation is never agentic: no join trees.
                            std::sync::Arc::default(),
                            // ...and no accelerated cache-warmup carrier either.
                            crate::agentic_tree::empty_warmup_handoff_carrier(),
                            // Offline co-simulation dispatches on the caller's
                            // reactor with no worker to defer materialization to.
                            false,
                        )?
                        .with_metrics_config(metrics.clone())
                        .with_performance_record_capture(false)
                        .with_performance_summary_collection(phase_count != 1)
                        .with_native_metric_record_dimensions(false)
                        .with_timing_record_capture(false);
                        if single_pass_engine {
                            // The `execute_pass` single engine cannot stop at a
                            // finite clock deadline, so no periodic progress event
                            // may be scheduled during the run.
                            plan.config.progress_interval_ns = DISABLED_PROGRESS_INTERVAL_NS;
                        }
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
        let event_delivery: Rc<dyn OfflineEventDeliveryPolicy> = if terminal_event_delivery {
            Rc::new(TerminalOfflineEventDelivery)
        } else {
            Rc::new(IncrementalOfflineEventDelivery)
        };
        let outcome = backend
            .executor(model.clone(), &artifact_target)?
            .execute_scheduled_deferred_with_delivery(factory, event_delivery)?;
        tracing::info!(
            online = backend.online,
            shared_fields = outcome.report.parity.shared_fields,
            serialized_bytes = outcome.report.parity.serialized_bytes,
            "offline scheduled parity verified"
        );
        let warmup = outcome
            .report
            .auxiliary_phase_reports
            .iter()
            .find(|report| report.kind == crate::timing::PhaseKind::Warmup)
            .map(|report| report.report.native_metrics.clone());
        let native_report = NativeReport::from_outcome(
            &outcome.report.aiperf.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some(format!("{}:scheduled", backend.mode_prefix())),
                    model: Some(model),
                },
                summary: ReportSummary {
                    endpoints_configured: vec![format!("dynosim://{}", backend.mode_prefix())],
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
        let mut report_metadata = outcome.run_metadata;
        report_metadata.insert("workload".into(), "scheduled".into());
        report_metadata.insert("phase_count".into(), phase_count.to_string());
        report_metadata.insert("benchmark_id".into(), benchmark_id);
        report_metadata.insert(
            "event_delivery".into(),
            if terminal_event_delivery {
                "terminal_coalesced"
            } else {
                "incremental"
            }
            .into(),
        );
        Ok(PreparedRunOutcome {
            native_report,
            report_facts,
            run_metadata: report_metadata,
            report_commit: None,
        })
    }
}

fn terminal_event_delivery_is_safe(phases: &[PhaseSpec]) -> bool {
    let [PhaseSpec::Concurrency { common, .. }] = phases else {
        return false;
    };
    common.name == "profiling"
        && !common.exclude_from_results
        && common.requests.is_some()
        && common.sessions.is_none()
        && common.duration.is_none()
        && common.prefill_concurrency.is_none()
        && common.grace_period.is_none()
        && !common.seamless
        && common.concurrency_ramp.is_none()
        && common.prefill_ramp.is_none()
        && common.rate_ramp.is_none()
        && common.cancellation.is_none()
        && common.adaptive_scale.is_none()
}

/// Whether a single scheduled phase schedules no clock events during engine
/// processing, so a closed-loop concurrency run can be driven by Dynamo's
/// `execute_pass` single engine (which cannot stop at a finite clock deadline).
///
/// Only a `Concurrency` phase is closed-loop (request-rate/user-centric/fixed
/// phases pace arrivals on the clock). A `duration` stop arms a clock timer;
/// ramps drive clock-paced actuators; cancellation schedules clock-timed aborts;
/// adaptive scale runs Clock-paced assessment. Any of these forces bounded
/// stepping, so they disqualify the single-pass engine.
fn phase_allows_single_pass_engine(phase: &PhaseSpec) -> bool {
    let PhaseSpec::Concurrency { common, .. } = phase else {
        return false;
    };
    common.duration.is_none()
        && common.concurrency_ramp.is_none()
        && common.prefill_ramp.is_none()
        && common.rate_ramp.is_none()
        && common.cancellation.is_none()
        && common.adaptive_scale.is_none()
}

/// Run-level validation for a dynosim direct-graph run.
///
/// Called by the graph workload factory when the resolved transport is a dynosim
/// transport (there is no per-transport pair object).
pub(crate) fn dynosim_graph_validate_run(
    _run: &AuthoredRunSpecV2,
    context: &RunContext,
    transport: &dyn ValidatedTransportConfig,
    workload: &GraphWorkloadConfigV2,
) -> Result<()> {
    let _ = validated_dynosim_transport(transport)?;
    ensure!(
        workload.worker_count == 1,
        "dynosim direct graph uses one LocalSet around one globally contended engine; worker_count must be 1"
    );
    validate_authored_tokenizer(&workload.tokenizer)?;
    validate_graph_phases(&workload.phases)?;
    context
        .graph_inputs()
        .validate_identity(&workload.dataset)?;
    ensure!(
        context.sidecar_inputs().is_empty(),
        "dynosim graph execution does not support online sidecars"
    );
    for (profile_id, profile) in context.endpoint_profiles() {
        let descriptor = context
            .product_registry()
            .endpoints()
            .resolve_factory(&profile.endpoint_id)
            .with_context(|| format!("resolving offline graph endpoint profile {profile_id:?}"))?
            .descriptor();
        ensure!(
            !descriptor.requires_raw_token_ids,
            "offline graph endpoint profile {profile_id:?} selects {:?}, which requires raw token IDs; direct Graph-IR nodes do not carry the dataset raw-token handle",
            descriptor.id
        );
    }
    Ok(())
}

/// Prepare a dynosim direct-graph operation from the coordinator-owned context.
pub(crate) fn prepare_dynosim_graph(
    run: &AuthoredRunSpecV2,
    context: &RunContext,
    transport: Box<dyn ValidatedTransportConfig>,
    workload: Box<dyn ValidatedWorkloadConfig>,
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
) -> Result<Box<dyn PreparedRunnerOperation>> {
    let backend = validated_dynosim_transport(transport.as_ref())?.clone();
    let workload = validated_graph_workload(workload.as_ref())?;
    let phases = workload.phases.clone();
    ensure!(
        run.models.items.len() == 1
            && matches!(run.models.strategy, ModelSelectionStrategy::RoundRobin),
        "dynosim direct graph requires exactly one round_robin model"
    );
    ensure_no_common_artifacts(
        run,
        "dynosim direct graph rejects common request/raw/output/user-file artifacts; use backend Dynamo artifacts or disable them",
    )?;
    // The artifact directory may already exist because the parent creates it for
    // logging. Backend artifacts still reject pre-existing output files.
    let model = run.models.items[0].name.clone();
    let tokenizer_spec = lower_authored_tokenizer(&workload.tokenizer, tokenizers.as_ref())?;
    let tokenizer = load_tokenizer(Some(&tokenizer_spec.name))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating direct graph preparation runtime")?;
    let prepared = runtime
        .block_on(context.graph_inputs().load(
            &workload.dataset,
            &GraphInputContext {
                tokenizer: tokenizer.as_ref(),
                run_random_seed: run.identity.random_seed,
            },
        ))
        .context("loading direct authored Graph-IR input")?;
    let metrics = offline_metrics_config(&run.metrics)?;
    let default_max_tokens = prepared.default_output_tokens;
    let allow_dataset_wrap = prepared.allow_dataset_wrap;
    let cache_bust_enabled = prepared.cache_bust_target.is_enabled();
    let random_seed = prepared.random_seed.or(run.identity.random_seed);
    Ok(Box::new(PreparedDynosimGraphOperation {
        backend,
        input: prepared.bundle,
        phases,
        metrics,
        model,
        benchmark_id: run.identity.benchmark_id.clone(),
        random_seed,
        artifact_target: run.artifact_target.clone(),
        default_max_tokens,
        allow_dataset_wrap,
        cache_bust_enabled,
        t_star_window: prepared.t_star_window,
        worker_count: workload.worker_count,
        phase_count: workload.phases.len(),
        ignore_trace_delays: workload.ignore_trace_delays,
    }))
}

fn validated_graph_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&GraphWorkloadConfigV2> {
    config
        .as_any()
        .downcast_ref::<GraphWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("dynosim graph pair received a different workload type"))
}

struct OfflineGraphRunnerEventSink {
    events: Arc<dyn GraphExecutionEventSink>,
}

impl OfflineGraphEventSink for OfflineGraphRunnerEventSink {
    fn first_token(&self, uuid: uuid::Uuid, trace_id: &str) -> Result<()> {
        self.events
            .emit(GraphExecutionEvent::FirstToken {
                trace_id: trace_id.to_owned(),
                uuid,
            })
            .map_err(Into::into)
    }

    fn record(&self, record: OfflineGraphRequestRecord) -> Result<()> {
        self.events
            .emit(GraphExecutionEvent::Record {
                record: Box::new(CapturedRecord {
                    uuid: record.uuid,
                    x_correlation_id: record.trace_id,
                    output: CapturedModelOutput::from_parts(&record.response_text, None, None),
                    raw: None,
                    ingest: record.ingest,
                }),
                // The offline dynosim adapter carries no static node id and never
                // feeds the (online-only) cache-pressure warmup handoff.
                node_id: None,
            })
            .map_err(Into::into)
    }
}

struct DynosimGraphPhaseBackendFactory {
    backends: Rc<dyn OfflineGraphBackendFactory>,
}

impl GraphPhaseBackendFactory for DynosimGraphPhaseBackendFactory {
    fn prepare_backend(
        &self,
        config: GraphPhaseBackendConfig,
    ) -> Result<PreparedGraphPhaseBackend> {
        let node_policy = config
            .cancellation
            .map(|cancellation| -> Result<Rc<dyn NodeDispatchPolicy>> {
                let worker_rng = cancellation
                    .rng_root
                    .derive_indexed_root(namespace::GRAPH_NODE_CANCELLATION_WORKER, 0);
                Ok(Rc::new(CancellationNodePolicy::new(
                    Box::new(BernoulliFixedDelay::new(
                        Some(cancellation.rate),
                        cancellation.delay_seconds,
                        worker_rng,
                    )?),
                    cancellation.phase,
                )) as Rc<dyn NodeDispatchPolicy>)
            })
            .transpose()?;
        let events: Rc<dyn OfflineGraphEventSink> = Rc::new(OfflineGraphRunnerEventSink {
            events: config.events,
        });
        let placement = self.backends.create_backend(OfflineGraphBackendConfig {
            phase: config.metrics_phase,
            prefill_concurrency: config.prefill_concurrency,
            node_policy,
            node_failure: Rc::new(AbortTraceNodeFailurePolicy),
            events,
        })?;
        Ok(PreparedGraphPhaseBackend {
            placement,
            requires_node_records: true,
        })
    }
}

struct PreparedDynosimGraphOperation {
    backend: ValidatedDynosimTransport,
    input: GraphInputBundle,
    phases: Vec<PhaseSpec>,
    metrics: MetricsConfig,
    model: String,
    benchmark_id: String,
    random_seed: Option<u64>,
    artifact_target: PathBuf,
    default_max_tokens: usize,
    allow_dataset_wrap: bool,
    cache_bust_enabled: bool,
    t_star_window: crate::engine::graph_input::TStarWindow,
    worker_count: usize,
    phase_count: usize,
    ignore_trace_delays: bool,
}

impl fmt::Debug for PreparedDynosimGraphOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedDynosimGraphOperation")
            .field("roots", &self.input.programs.len())
            .field("model", &self.model)
            .field("artifact_target", &self.artifact_target)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedDynosimGraphOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let Self {
            backend,
            input,
            phases,
            metrics,
            model,
            benchmark_id,
            random_seed,
            artifact_target,
            default_max_tokens,
            allow_dataset_wrap,
            cache_bust_enabled,
            t_star_window,
            worker_count,
            phase_count,
            ignore_trace_delays,
        } = *self;
        create_artifact_target(&artifact_target)?;
        let metadata = input.metadata.clone();
        let graph_report = ReportGraphRunInfo::new(
            metadata.format.clone(),
            metadata.root_count,
            metadata.node_count,
            worker_count,
            phase_count,
        )?;
        let rng_root = RngRoot::new(random_seed);
        let segments = input.segments.clone();
        let artifact_dir = artifact_target.clone();
        let factory: Box<dyn DeferredOfflineGraphRunFactory> = Box::new(
            move |clock: Rc<dyn Clock>, backends: Rc<dyn OfflineGraphBackendFactory>| {
                let backends = DynosimGraphPhaseBackendFactory { backends };
                Ok(Box::pin(async move {
                    // The offline in-process replay path has no external
                    // telemetry sources to probe; every phase runs without
                    // side-channel sidecars.
                    let phase_sidecars = phases.iter().map(|_| Vec::new()).collect::<Vec<_>>();
                    let phased = run_graph_phases(
                        &phases,
                        &benchmark_id,
                        &artifact_dir,
                        &input,
                        clock,
                        rng_root,
                        allow_dataset_wrap,
                        cache_bust_enabled,
                        t_star_window,
                        phase_sidecars,
                        &backends,
                        // Co-simulation uses fail-fast graph execution.
                        OnFailure::for_graph_default(),
                        None,
                    )
                    .await?;
                    Ok(OfflineGraphExecution {
                        workload: phased.workload,
                        phases: phased.phases,
                    })
                }) as DeferredOfflineGraphFuture)
            },
        );
        let outcome = backend
            .executor(model.clone(), &artifact_target)?
            .execute_graph_deferred(
                segments,
                default_max_tokens,
                metrics,
                ignore_trace_delays,
                factory,
            )?;
        ensure_no_failed_traces(&outcome.report.workload)?;

        let native_report = NativeReport::from_outcome(
            &outcome.report.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some(format!("{}:graph", backend.mode_prefix())),
                    model: Some(model),
                },
                summary: ReportSummary {
                    was_cancelled: outcome
                        .report
                        .phases
                        .iter()
                        .any(|phase| phase.was_cancelled),
                    ..ReportSummary::default()
                },
                warmup: outcome.report.warmup_metrics.clone(),
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
        let mut report_metadata = outcome.run_metadata;
        report_metadata.insert("workload".into(), "graph".into());
        report_metadata.insert("graph_input".into(), metadata.format);
        report_metadata.insert("graph_roots".into(), metadata.root_count.to_string());
        report_metadata.insert("graph_nodes".into(), metadata.node_count.to_string());
        report_metadata.insert("phase_count".into(), phase_count.to_string());
        Ok(PreparedRunOutcome {
            native_report,
            report_facts,
            run_metadata: report_metadata,
            report_commit: None,
        })
    }
}

fn offline_metrics_config(spec: &MetricsSpec) -> Result<MetricsConfig> {
    let slice_duration_ns =
        resolve_slice_duration_ns(spec.slice_duration_seconds, "metrics slice duration")?;
    let slos = resolve_slos(spec)?;
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        ..MetricsConfig::default()
    })
}

/// Resolved backend artifact paths committed after successful execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynosimArtifactOutputs {
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
/// exact parity validation, and only then writes optional backend artifacts.
#[derive(Clone, Debug)]
pub struct DynosimExecutor {
    engine: OfflineEngineConfig,
    artifacts: DynosimArtifactSpec,
    topology: DynosimTopologySpec,
    router_mode: DynosimRouterSpec,
    online: bool,
    required_features: BTreeSet<DynamoBuildFeature>,
    model: String,
    artifact_target: PathBuf,
}

impl DynosimExecutor {
    /// Run a scheduled workload whose deterministic observer reduction occurs
    /// after the virtual-time driver and its Tokio `LocalSet` have exited.
    pub fn execute_scheduled_deferred(
        self,
        workload: Box<dyn DeferredOfflineScheduledRunFactory>,
    ) -> Result<DynosimScheduledOutcome> {
        self.execute_scheduled_deferred_with_delivery(
            workload,
            Rc::new(IncrementalOfflineEventDelivery),
        )
    }

    /// Run a deferred scheduled workload with an injected Dynamo event-delivery policy.
    pub fn execute_scheduled_deferred_with_delivery(
        self,
        workload: Box<dyn DeferredOfflineScheduledRunFactory>,
        event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    ) -> Result<DynosimScheduledOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let online = self.online;
        let report = if online {
            run_scheduled_backend_online_deferred_with_delivery(
                self.engine.clone(),
                self.model.clone(),
                workload,
                event_delivery,
            )?
        } else {
            run_scheduled_backend_offline_deferred_with_delivery(
                self.engine.clone(),
                self.model.clone(),
                workload,
                event_delivery,
            )?
        };
        verify_parity_for(online, &report.performance, &report.dynamo, report.parity)?;
        let artifacts = self.emit_backend_artifacts(&report.dynamo)?;
        let report_metadata = self.report_metadata(report.parity);
        Ok(DynosimScheduledOutcome {
            report,
            artifacts,
            run_metadata: report_metadata,
        })
    }

    /// Run the existing generated Graph-IR benchmark entrypoint without HTTP.
    ///
    /// Direct authored Graph-IR pair factories should use the direct-plan API
    /// when it is supplied by the graph workload module; this method preserves
    /// the existing library benchmark consumer and its DES/parity gates.
    pub fn execute_graph_bench(self, mut config: BenchConfig) -> Result<DynosimGraphOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        config.model.clone_from(&self.model);
        let report = run_graph_offline(self.engine.clone(), config)?;
        verify_parity(&report.performance, &report.dynamo, report.parity)?;
        let artifacts = self.emit_backend_artifacts(&report.dynamo)?;
        let report_metadata = self.report_metadata(report.parity);
        Ok(DynosimGraphOutcome {
            report,
            artifacts,
            run_metadata: report_metadata,
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
        ignore_trace_delays: bool,
        workload: Box<dyn OfflineGraphRunFactory>,
    ) -> Result<DynosimDirectGraphOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let online = self.online;
        let report = if online {
            run_graph_workload_online(
                self.engine.clone(),
                self.model.clone(),
                segments,
                default_max_tokens,
                metrics,
                node_policy,
                node_failure,
                ignore_trace_delays,
                workload,
            )?
        } else {
            run_graph_workload_offline(
                self.engine.clone(),
                self.model.clone(),
                segments,
                default_max_tokens,
                metrics,
                node_policy,
                node_failure,
                ignore_trace_delays,
                workload,
            )?
        };
        verify_parity_for(online, &report.performance, &report.dynamo, report.parity)?;
        ensure_no_failed_traces(&report.workload)?;
        let artifacts = self.emit_backend_artifacts(&report.dynamo)?;
        let report_metadata = self.report_metadata(report.parity);
        Ok(DynosimDirectGraphOutcome {
            report,
            artifacts,
            run_metadata: report_metadata,
        })
    }

    /// Execute an injected multi-phase Graph-IR driver over one SimClock and
    /// one passive engine.
    pub fn execute_graph_deferred(
        self,
        segments: Arc<dyn SegmentStore>,
        default_max_tokens: usize,
        metrics: MetricsConfig,
        ignore_trace_delays: bool,
        workload: Box<dyn DeferredOfflineGraphRunFactory>,
    ) -> Result<DynosimDirectGraphOutcome> {
        ensure!(
            self.artifacts.worker_artifacts_json.is_none(),
            "worker_artifacts_json is supported only by canonical trace workloads"
        );
        let online = self.online;
        let report = if online {
            run_graph_workload_online_deferred(
                self.engine.clone(),
                self.model.clone(),
                segments,
                default_max_tokens,
                metrics,
                ignore_trace_delays,
                workload,
            )?
        } else {
            run_graph_workload_offline_deferred(
                self.engine.clone(),
                self.model.clone(),
                segments,
                default_max_tokens,
                metrics,
                ignore_trace_delays,
                workload,
            )?
        };
        verify_parity_for(online, &report.performance, &report.dynamo, report.parity)?;
        ensure_no_failed_traces(&report.workload)?;
        let artifacts = self.emit_backend_artifacts(&report.dynamo)?;
        let report_metadata = self.report_metadata(report.parity);
        Ok(DynosimDirectGraphOutcome {
            report,
            artifacts,
            run_metadata: report_metadata,
        })
    }

    /// Run one canonical Dynamo trace workload through AIPerf's observer and
    /// native metrics stack, retaining complete byte-exact parity evidence.
    pub fn execute_trace(self, trace: OfflineTraceConfig) -> Result<DynosimTraceOutcome> {
        ensure!(
            !self.online,
            "dynosim_online drives the trace through AIPerf's own graph flow; \
             the canonical mocker trace driver runs only under dynosim_offline replay"
        );
        let report = run_trace_offline(self.engine.clone(), trace.clone())?;
        verify_parity(&report.aiperf.performance, &report.dynamo, report.parity)?;
        let mut artifacts = self.emit_backend_artifacts(&report.dynamo)?;
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
        let report_metadata = self.report_metadata(report.parity);
        Ok(DynosimTraceOutcome {
            report,
            artifacts,
            run_metadata: report_metadata,
        })
    }

    fn emit_backend_artifacts(
        &self,
        dynamo: &DynamoSimulationReport,
    ) -> Result<DynosimArtifactOutputs> {
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
            write_dynamo_report_json(dynamo, path)?;
        }
        if let Some(path) = &per_request_jsonl {
            write_dynamo_per_request_jsonl(dynamo, path)?;
        }
        Ok(DynosimArtifactOutputs {
            report_json,
            per_request_jsonl,
            worker_artifacts_json: None,
        })
    }

    fn artifact_path(&self, relative: &Path) -> PathBuf {
        self.artifact_target.join(relative)
    }

    fn report_metadata(&self, parity: OfflineMetricParity) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "transport".into(),
                if self.online {
                    DYNOSIM_ONLINE_ID.into()
                } else {
                    DYNOSIM_OFFLINE_ID.into()
                },
            ),
            (
                "clock".into(),
                if self.online {
                    "real".into()
                } else {
                    "sim".into()
                },
            ),
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
    // The artifact directory may already exist because the parent creates it for
    // logging. Backend artifacts still reject pre-existing output files.
    std::fs::create_dir_all(path)
        .with_context(|| format!("creating offline artifact target {}", path.display()))
}

/// Abort an offline graph run when any root trace failed.
fn ensure_no_failed_traces(workload: &crate::graph::workload::GraphWorkloadReport) -> Result<()> {
    if workload.failed == 0 {
        return Ok(());
    }
    let detail = workload
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
    Err(anyhow!(
        "offline graph aborted after {} failed root(s): {detail}",
        workload.failed
    ))
}

/// Field-accounting, serialized-byte, and non-empty-schema invariants shared by
/// the byte-exact offline and relaxed online parity checks.
fn verify_parity_invariants(parity: OfflineMetricParity, aiperf_bytes_len: usize) -> Result<()> {
    ensure!(
        parity.independently_accumulated_fields + parity.backend_owned_fields
            == parity.shared_fields,
        "offline parity evidence has inconsistent field accounting"
    );
    ensure!(
        parity.serialized_bytes == aiperf_bytes_len,
        "offline parity evidence has an inconsistent serialized byte count"
    );
    ensure!(parity.shared_fields > 0, "offline parity schema is empty");
    Ok(())
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
    verify_parity_invariants(parity, aiperf_bytes.len())
}

/// Relaxed parity verification for wall-clock in-process (`dynosim_online`)
/// runs. The AIPerf and Dynamo shared summaries are *not* byte-identical — real
/// timers cannot reproduce the engine's internal completion times — so only the
/// field-accounting and non-empty-schema invariants are enforced. Request and
/// token counts remain exact; latency/throughput are the live measured values.
fn verify_parity_online(
    aiperf: &impl CanonicalSharedMetrics,
    parity: OfflineMetricParity,
) -> Result<()> {
    let aiperf_bytes = aiperf
        .canonical_shared_metric_bytes()
        .context("serializing runner-side AIPerf online parity summary")?;
    verify_parity_invariants(parity, aiperf_bytes.len())
}

/// Select byte-exact (`dynosim_offline`) or relaxed wall-clock (`dynosim_online`)
/// parity verification on the shared clock axis.
fn verify_parity_for(
    online: bool,
    aiperf: &impl CanonicalSharedMetrics,
    dynamo: &impl CanonicalSharedMetrics,
    parity: OfflineMetricParity,
) -> Result<()> {
    if online {
        verify_parity_online(aiperf, parity)
    } else {
        verify_parity(aiperf, dynamo, parity)
    }
}

/// Successful scheduled offline execution and backend-owned outputs.
pub struct DynosimScheduledOutcome {
    /// AIPerf scheduled metrics plus Dynamo's co-observed report.
    pub report: OfflineScheduledReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynosimArtifactOutputs,
    /// Additive terminal and native-report metadata.
    pub run_metadata: BTreeMap<String, String>,
}

/// Successful generated Graph-IR offline execution and outputs.
pub struct DynosimGraphOutcome {
    /// Graph metrics plus Dynamo's co-observed report.
    pub report: OfflineGraphReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynosimArtifactOutputs,
    /// Additive terminal and native-report metadata.
    pub run_metadata: BTreeMap<String, String>,
}

/// Successful direct authored Graph-IR offline execution and outputs.
pub struct DynosimDirectGraphOutcome {
    /// Root/trace outcomes, request metrics, and Dynamo parity report.
    pub report: OfflineDirectGraphReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynosimArtifactOutputs,
    /// Additive terminal and native-report metadata.
    pub run_metadata: BTreeMap<String, String>,
}

/// Successful canonical trace execution and outputs.
pub struct DynosimTraceOutcome {
    /// AIPerf metrics plus Dynamo's co-observed report.
    pub report: OfflineRunReport,
    /// Optional backend-specific artifact paths.
    pub artifacts: DynosimArtifactOutputs,
    /// Additive terminal and native-report metadata.
    pub run_metadata: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::registry::TransportFactory;
    use crate::extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};

    fn raw(value: Value) -> Box<RawValue> {
        RawValue::from_string(value.to_string()).unwrap()
    }

    fn validate(value: Value) -> Result<Box<dyn ValidatedTransportConfig>> {
        DynosimTransportFactory::offline().validate(&raw(value), &WorkloadRequirements::default())
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
        let registry = BuiltinAIPerfRegistryFactory.build().unwrap();
        for (transport_id, clock) in [
            (DYNOSIM_OFFLINE_ID, ClockKind::Sim),
            (DYNOSIM_ONLINE_ID, ClockKind::Real),
        ] {
            let descriptor = registry
                .transport_descriptors()
                .into_iter()
                .find(|descriptor| descriptor.id == transport_id)
                .unwrap();
            assert!(descriptor.features.contains(&"exact_metric_parity"));
            assert_eq!(descriptor.clock, clock);
        }
        // Transport and workload registries are independent; the dynosim
        // transports and the scheduled/graph workloads are all present and any
        // workload can drive any transport with no pair/cross-product table.
        let workloads = registry
            .workload_descriptors()
            .into_iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        assert!(workloads.contains(&"graph"));
        assert!(workloads.contains(&"scheduled"));
    }

    #[test]
    fn graph_adapter_runs_without_http_and_commits_exact_backend_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "aiperf runner-offline-graph-{}-{}",
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
        let backend = validated_dynosim_transport(validated.as_ref()).unwrap();
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
        assert_eq!(outcome.run_metadata["transport"], DYNOSIM_OFFLINE_ID);
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
        let (segments, graph, _) = crate::graph::bench::build_workload(2);
        let plan = crate::graph::model::GraphTracePlan {
            graph,
            trace: crate::graph::model::TraceRecord {
                id: "direct-root".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let factory: Box<dyn OfflineGraphRunFactory> = Box::new(move |clock, backend| {
            Ok(crate::graph::workload::GraphWorkload::new(
                clock,
                Rc::new(crate::graph::workload::VecGraphTraceSource::new([
                    crate::graph::model::GraphTraceProgram::static_graph(plan),
                ])),
                backend,
            ))
        });
        let validated = validate(serde_json::json!({})).unwrap();
        let backend = validated_dynosim_transport(validated.as_ref()).unwrap();
        let outcome = backend
            .executor("model", "/tmp/aiperf-direct-graph-unused")
            .unwrap()
            .execute_graph(
                Arc::new(segments),
                2,
                MetricsConfig::default(),
                None,
                Rc::new(crate::graph::policy::AbortTraceNodeFailurePolicy),
                false,
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
        assert_eq!(outcome.run_metadata["parity_shared_fields"], "74");
    }
}
