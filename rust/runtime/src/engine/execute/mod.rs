// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::accuracy::{
    AccuracyDataset, AccuracyRecordProcessor, CapturedResponse, ProblemAssociation,
    accuracy_report_errors, grade_accuracy_captures, load_evaluator_problems_with_grader,
};
use crate::accuracy_core::{
    AccuracyEvaluator, EvaluatorLoadConfig, EvaluatorLoadResult, PythonEvaluator,
    WorkerProcessConfig,
};
use crate::adaptive::{
    AdaptiveControlVariable, AdaptiveRunConfig, AdaptiveStepConfig, build_adaptive_with_origins,
    positive_seconds_to_ns,
};
use crate::adaptive_core::{
    AdaptiveScale, CorrelationContext, SharedWindowSampler, SlaFilter, UserTarget,
};
use crate::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use crate::cellular::{CellPartition, IssuanceAuthority, ModuloCellPartition};
use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::content_server::{
    ContentRequestRecord, ContentServerConfig, ContentServerFactory, ContentServerRuntime,
    MediaFetchAggregator, MediaMetricsSummary, MediaRecordWriter, NativeContentServerFactory,
};
use crate::dataset::{
    ComposeConfig, CorpusPromptGeneratorFactory, Dataset, DatasetSource, HuggingFaceTokenizer,
    LoadConfig, ModelId, ModelSelector, ModelSelectorFactory, NativeTiktokenTokenizer,
    PrefetchMediaResolver, PromptGeneratorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, ServerTokenizer, SourceImageSampling, SyntheticAudioConfig,
    SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig, SyntheticImageFormat,
    SyntheticImageSource, SyntheticMediaGeneratorFactory, SyntheticPrefixConfig,
    SyntheticPromptConfig, SyntheticRankingsConfig, SyntheticVideoAudioConfig,
    SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern, TextTokenizer,
    TiktokenEncoding, TiktokenTokenizer, TracePromptStoragePolicy, TraceSynthesisConfig,
    find_tiktoken_model_file,
};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::RequestObserver;
use crate::endpoints::{EndpointKey, EndpointRegistry, PreparedEndpointTable};
use crate::export::otel::OtelRecordAccumulator;
use crate::extensions::AIPerfRegistry;
use crate::failure::OnFailure;
use crate::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use crate::graph::input::GraphInputBundle;
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use crate::metrics_core::{
    AccumulatorSummary, CATALOG, ExportContext, InferenceDimensions, MetricTag, MetricsAccumulator,
    MetricsConfig, NativeReport, Phase as MetricsPhase, RecordIngest, ReportRunInfo,
    ReportServerMetricsMetadata, ReportSummary, RunOutcome, SloThreshold,
};
use crate::multiturn::{
    AuthoredInputTokenCounter, ConversationSource, EndpointInputTokenCounter, InputTokenCounter,
    IssuedCredit, NativeDatasetConversationSource, PreparedEndpointReference,
    PreparedEndpointTableResolver, PreparedTurnEndpointResolver, TurnToSend,
};
use crate::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhasePlan,
    ScheduledPhaseResources, ScheduledPhaseSidecar, ScheduledRuntimeExtension,
    ScheduledRuntimeExtensionParts, SlotPoolPhaseResources, run_scheduled_phases,
};
use crate::request_rate::RequestRateWorkload;
use crate::rng::{
    EmpiricalPoint, PeakEntry, RngRoot, SamplingDistribution, SequenceLengthDistribution,
    SequenceLengthPair, namespace,
};
use crate::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, TurnCreditReport, TurnCreditReportKind,
    TurnDispatchOutcome, TurnDispatcher, TurnRecordProcessor, Workload,
};
use crate::timing::{
    BernoulliFixedDelay, CancellationPolicy, ExponentialRamp, GracePeriod, LinearRamp,
    NoopPhaseObserver, PhaseConfig, PhaseKind, PhaseObserver, PoissonRamp, RampDriver,
    RampStrategy, RamperConfig, RoundRobinUrlSelector, SlotPool, StopConfig, UrlSelector,
    make_interval_generator,
};
use crate::transport::core::{CreditReportKind, MeasuredContext, MeasuredOutcome};
use crate::transport::core::{PreparedTurn, RequestExecutor};
use crate::transport::http::TransportSinkConfig;
use crate::user_centric::{UserCentricConfig, UserCentricWorkload};
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use uuid::Uuid;

use crate::engine::dataset_input::PreparedDatasetInput;
use crate::engine::execution_factories::ExecutionFactories;
use crate::engine::gpu_telemetry::GpuTelemetryRun;
use crate::engine::graph_execution::{
    GraphBackendFactory, GraphBackendFactoryConfig, GraphEndpointRuntimeFactory,
    GraphPlacementFactory, PreparedRunnerGraphEndpointRuntimeFactory,
};
use crate::engine::graph_phase_runtime::{
    GraphPhaseBackendConfig, GraphPhaseBackendFactory, PreparedGraphPhaseBackend, run_graph_phases,
    validate_graph_phases,
};
use crate::engine::heartbeat_lane::{
    CompositePhaseObserver, HeartbeatLane, HeartbeatPhaseObserver,
};
use crate::engine::live_streaming::{LiveResultsSink, PythonLiveStreamingRun, live_phase_observer};
use crate::engine::network_latency::NetworkLatencyRun;
use crate::engine::phase_identity::{PhaseIdentity, phase_identity_from_spec};
use crate::engine::protocol::{
    AdaptiveControlVariableSpec, AdaptiveScaleSpec, AdaptiveStepPolicySpec, DispatchMode,
    DistributionSpec, FileDatasetSpec, MetricsSpec, ModelSelectionStrategy, ModelsSpec, PhaseSpec,
    PromptSelectionSpec, PublicDatasetSourceSpec, PublicDatasetSpec, RampSpec, RampStrategySpec,
    SequenceDistributionEntrySpec, SourceImageSamplingSpec, SyntheticAudioFormatSpec,
    SyntheticAudioSpec, SyntheticDatasetSpec, SyntheticImageFormatSpec, SyntheticImageSpec,
    SyntheticPrefixPromptsSpec, SyntheticPromptsSpec, SyntheticVideoFormatSpec,
    SyntheticVideoPatternSpec, SyntheticVideoSpec,
};
use crate::engine::readiness::{PreparedOnlineReadiness, ReadinessTransportFactory};
use crate::engine::record_lane::RecordArtifactLane;
use crate::engine::records::{
    CapturedHttpExchange, CapturedModelOutput, CapturedRecord, InputSession, group_record_errors,
    observe_otel_record, write_inputs_json, write_outputs_json, write_raw_records_jsonl,
    write_records_csv, write_records_jsonl,
};
use crate::engine::registry::ValidatedEndpointProfileV2;
use crate::engine::server_metrics::ServerMetricsRun;
use crate::engine::sidecar_input::{
    CONTENT_SERVER_SIDECAR_ID, ContentServerSpec, GPU_TELEMETRY_SIDECAR_ID, GpuTelemetrySpec,
    LIVE_STREAMING_SIDECAR_ID, LiveStreamingSpec, NETWORK_LATENCY_SIDECAR_ID, NetworkLatencySpec,
    PreparedSidecarInputs, SERVER_METRICS_SIDECAR_ID, ServerMetricsSpec,
};
use crate::engine::turn_execution::{
    ExecutionBackendConfig, PreparedEndpointTableFactory, RequestExecutorFactory,
};
use crate::server_metrics::ServerMetricsSummary;

mod capture;
mod compose_sidecars;
mod dataset_build;
mod entrypoints;
mod graph_backend;
mod plan;
mod ramp_adaptive;
mod sharding;
mod sidecars;

pub(crate) use capture::*;
pub(crate) use compose_sidecars::*;
pub(crate) use dataset_build::*;
pub(crate) use entrypoints::*;
pub(crate) use graph_backend::*;
pub(crate) use plan::*;
pub(crate) use ramp_adaptive::*;
pub(crate) use sharding::*;
pub(crate) use sidecars::*;

// Preserve the crate-external public surface (these are `pub`, not `pub(crate)`).
pub use plan::{
    NativeStaticAccuracyEvaluatorFactory, StaticAccuracyEvaluatorFactory,
    StaticAccuracyEvaluatorProcessSpec,
};

pub(crate) type PhaseRuntimeParts = (
    Rc<dyn Workload>,
    Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    Option<Rc<SlotPool>>,
    Option<Rc<SlotPool>>,
    bool,
    Rc<dyn ScheduledPhaseResources>,
    Option<Rc<dyn UserTarget>>,
);

/// Execute a protocol-v2 plan through a transport-selected request executor.
///
/// The workload resolves the transport's turn-placement factory by
/// `transport_id` (HTTP vs gRPC `RequestExecutorFactory`) and passes it here
/// with an optional readiness plan (present only for transports that expose a
/// readiness control plane). The graph placement and its `transport_kind` arm
/// are resolved from the shared factories, so both scheduled and graph plans run
/// over either transport through one entry point. Graph input, if present, is
/// already fully prepared: once the workload has returned a canonical
/// `GraphInputBundle`, the harness cannot reinterpret the authored source again.
pub(crate) fn execute_prepared_native_plan_uncommitted_selected(
    plan: NativeRunSpec,
    request_executor: Arc<dyn RequestExecutorFactory>,
    factories: &ExecutionFactories,
    registry: &AIPerfRegistry,
    readiness: Option<Box<dyn PreparedOnlineReadiness>>,
) -> Result<NativeReport> {
    execute_prepared_native_plan_uncommitted_with_runtime_factories(
        plan,
        request_executor,
        factories.graph(),
        factories.trace_driver_handle(),
        factories.control_plane_http_handle(),
        registry,
        &BuiltinNativeSidecarResourceFactory,
        readiness.map(|readiness| (readiness, factories.readiness_transport())),
    )
}

pub(crate) type NativeEndpointExecutionParts<'a> = (
    Vec<String>,
    TransportSinkConfig,
    Option<Arc<dyn PreparedEndpointTableFactory>>,
    Box<dyn NativeConversationSourceFactory + 'a>,
);
