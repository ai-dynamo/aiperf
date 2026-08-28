// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 dataset-stream resources and shadow-replay configuration.
//!
//! Every assertion here is a startup-time decision: authored decoding, wire
//! validation, Config-v2 cross-field refusal, resource presence, and the
//! descriptor-only stream resolution. The effect counters prove that none of it
//! reaches a factory `validate`/`prepare`, opens a source, or allocates a lease.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::config::model::BenchmarkConfig;
use aiperf_runtime::config::model::workload_kind::{WorkloadKind, workload_kind};
use aiperf_runtime::endpoints::EndpointDescriptor;
use aiperf_runtime::engine::protocol_v2::{
    AuthoredRunSpecV2, BenchmarkRunWireV2, DatasetStreamsSpecV2, ReliabilityPolicyDigestV2,
    RunResourceV2,
};
use aiperf_runtime::engine::registry::{
    ResourceRequirementsV2, StreamingResourceContext, StreamingResourceError, TransportDescriptor,
    ValidatedWorkloadConfig, WorkloadDescriptor, WorkloadFactory, WorkloadRequirements,
};
use aiperf_runtime::extensions::AIPerfRegistry;
use aiperf_runtime::streaming::{
    action::{
        ActionExecutionError, ActionFailureCode, ActionPlacement, ActionResultRetention,
        DatasetActionSchema, PreparedStreamingActionBinding, StreamingActionSinkDescriptor,
        StreamingActionSinkFactory, StreamingActionSinkPrepareContext,
        ValidatedStreamingActionSinkConfig,
    },
    checkpoint::CheckpointError,
    checkpoint_backend::{
        CheckpointBackendPlacement, CheckpointBackendPrepareContext, CheckpointBackendRequirements,
        CheckpointRetention, StreamingCheckpointBackend, StreamingCheckpointBackendDescriptor,
        StreamingCheckpointBackendFactory, ValidatedCheckpointBackendConfig,
    },
    format::{
        DecodeFailureCode, FormatProjection, FormatStateRetention, StreamFormatError,
        StreamingDatasetFormat, StreamingDatasetFormatFactory, StreamingFormatDescriptor,
        StreamingFormatPrepareContext, ValidatedStreamingFormatConfig,
    },
    identity::ContentDigest,
    reliability::PreparedStreamingIssuePolicy,
    session::{
        SessionClosureCapability, SessionCoordinatorError, SessionFailureCode, SessionPlacement,
        SessionStateRetention, StreamingSessionCoordinator, StreamingSessionPrepareContext,
        StreamingSessionProgramDescriptor, StreamingSessionProgramFactory,
        ValidatedStreamingSessionProgramConfig,
    },
    source::{
        PartitionAccessKind, PreparedStreamingDatasetSource, SourceFailureCode, StreamSourceError,
        StreamingDatasetSourceFactory, StreamingResumeGranularity, StreamingSourceDescriptor,
        StreamingSourceMode, StreamingSourceOrdering, StreamingSourcePlacement,
        StreamingSourcePrepareContext, StreamingSourceRetention, ValidatedStreamingSourceConfig,
    },
};
use serde_json::json;
use serde_json::value::RawValue;

// ---------------------------------------------------------------------------
// One shared effect counter, incremented by every fake `validate`/`prepare`.
// A non-zero count after a descriptor-only path is a test failure.
// ---------------------------------------------------------------------------

static EFFECTS: AtomicUsize = AtomicUsize::new(0);

fn effect() {
    EFFECTS.fetch_add(1, Ordering::SeqCst);
}

fn effects() -> usize {
    EFFECTS.load(Ordering::SeqCst)
}

#[derive(Debug)]
struct FakeSourceFactory {
    descriptor: &'static StreamingSourceDescriptor,
}

impl FakeSourceFactory {
    fn new(id: &'static str) -> Self {
        Self::with_modes(
            id,
            &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
            &[
                PartitionAccessKind::Sequential,
                PartitionAccessKind::SeekableLocal,
                PartitionAccessKind::RangeReadable,
            ],
        )
    }

    /// A follow-only, sequential-access source: the axis test #18 narrows.
    fn follow_only(id: &'static str) -> Self {
        Self::with_modes(
            id,
            &[StreamingSourceMode::Follow],
            &[PartitionAccessKind::Sequential],
        )
    }

    fn with_modes(
        id: &'static str,
        modes: &'static [StreamingSourceMode],
        access: &'static [PartitionAccessKind],
    ) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingSourceDescriptor {
                id,
                description: "test-only streaming source",
                modes,
                access,
                ordering: StreamingSourceOrdering::EventTime,
                resume: &[
                    StreamingResumeGranularity::Partition,
                    StreamingResumeGranularity::Record,
                ],
                has_event_time: true,
                has_stable_record_ids: true,
                retention: StreamingSourceRetention::BoundedMemory,
                placement: StreamingSourcePlacement::ControllerOnly,
                supports_virtual_clock: true,
            })),
        }
    }
}

impl StreamingDatasetSourceFactory for FakeSourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        self.descriptor
    }

    fn validate(
        &self,
        _authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        effect();
        Err(StreamSourceError::source(
            SourceFailureCode::SourceUnavailable,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingSourceConfig>,
        _context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        effect();
        Err(StreamSourceError::source(
            SourceFailureCode::SourceUnavailable,
        ))
    }
}

#[derive(Debug)]
struct FakeFormatFactory {
    descriptor: &'static StreamingFormatDescriptor,
}

impl FakeFormatFactory {
    fn new(id: &'static str) -> Self {
        Self::with_retention(id, FormatStateRetention::BoundedMemory)
    }

    /// A format that requires the complete input resident: the axis test #18
    /// uses to force an agreement refusal.
    fn resident(id: &'static str) -> Self {
        Self::with_retention(id, FormatStateRetention::ResidentInput)
    }

    fn with_retention(id: &'static str, retention: FormatStateRetention) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingFormatDescriptor {
                id,
                description: "test-only streaming format",
                semantic_digest: ContentDigest::from_bytes([0u8; 32]),
                media_types: &["application/jsonl"],
                input_schemas: &["test.source.v1"],
                required_access: PartitionAccessKind::Sequential,
                projection: FormatProjection::FullRecord,
                output_schema: "test.fragment.v1",
                has_event_time: true,
                has_stable_record_ids: true,
                retention,
                supports_virtual_clock: true,
            })),
        }
    }
}

impl StreamingDatasetFormatFactory for FakeFormatFactory {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor {
        self.descriptor
    }

    fn validate(
        &self,
        _authored: &RawValue,
        _source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError> {
        effect();
        Err(StreamFormatError::decode(DecodeFailureCode::Schema))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingFormatConfig>,
        _context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError> {
        effect();
        Err(StreamFormatError::decode(DecodeFailureCode::Schema))
    }
}

#[derive(Debug)]
struct FakeSessionProgramFactory {
    descriptor: &'static StreamingSessionProgramDescriptor,
}

impl FakeSessionProgramFactory {
    fn new(id: &'static str) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingSessionProgramDescriptor {
                id,
                description: "test-only session program",
                fragment_input_schemas: &["test.fragment.v1"],
                action_schemas: &["test.action.v1"],
                closure: &[
                    SessionClosureCapability::ExplicitClose,
                    SessionClosureCapability::HardWatermark,
                ],
                retention: SessionStateRetention::BoundedMemory,
                placement: SessionPlacement::ControllerCanonical,
                supports_virtual_clock: true,
            })),
        }
    }
}

impl StreamingSessionProgramFactory for FakeSessionProgramFactory {
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor {
        self.descriptor
    }

    fn validate(
        &self,
        _authored: &RawValue,
        _format: &StreamingFormatDescriptor,
        _workload: &WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError> {
        effect();
        Err(SessionCoordinatorError::session(
            SessionFailureCode::MissingPredecessor,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        _context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError> {
        effect();
        Err(SessionCoordinatorError::session(
            SessionFailureCode::MissingPredecessor,
        ))
    }
}

#[derive(Debug)]
struct FakeActionSinkFactory {
    descriptor: &'static StreamingActionSinkDescriptor,
}

impl FakeActionSinkFactory {
    fn new(id: &'static str) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingActionSinkDescriptor {
                id,
                description: "test-only action sink",
                accepted_schemas: &["test.action.v1"],
                transport_ids: &["dry_run", "http"],
                endpoint_kinds: &["chat"],
                retention: ActionResultRetention::StreamingTerminal,
                placement: ActionPlacement::WorkerLocal,
                supports_virtual_clock: true,
            })),
        }
    }
}

impl StreamingActionSinkFactory for FakeActionSinkFactory {
    fn descriptor(&self) -> &'static StreamingActionSinkDescriptor {
        self.descriptor
    }

    fn validate_binding(
        &self,
        _authored: &RawValue,
        _action: &DatasetActionSchema,
        _transport: &TransportDescriptor,
        _endpoint: &EndpointDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingActionSinkConfig>, ActionExecutionError> {
        effect();
        Err(ActionExecutionError::action(
            ActionFailureCode::MissingBinding,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingActionSinkConfig>,
        _context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError> {
        effect();
        Err(ActionExecutionError::action(
            ActionFailureCode::MissingBinding,
        ))
    }
}

#[derive(Debug)]
struct FakeCheckpointBackendFactory {
    descriptor: &'static StreamingCheckpointBackendDescriptor,
}

impl FakeCheckpointBackendFactory {
    fn new(id: &'static str) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingCheckpointBackendDescriptor {
                id,
                description: "test-only checkpoint backend",
                is_durable: true,
                has_leased_readers: true,
                has_atomic_generations: true,
                has_result_segments: true,
                protects_sensitive_state: false,
                retention: CheckpointRetention::GenerationReachability,
                placement: CheckpointBackendPlacement::SharedAcrossCells,
                supports_virtual_clock: true,
            })),
        }
    }
}

impl StreamingCheckpointBackendFactory for FakeCheckpointBackendFactory {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
        self.descriptor
    }

    fn validate(
        &self,
        _authored: &RawValue,
        _requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError> {
        effect();
        Err(CheckpointError::ParticipantSetMismatch)
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedCheckpointBackendConfig>,
        _context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        effect();
        Err(CheckpointError::ParticipantSetMismatch)
    }
}

/// A workload factory whose only job is to report one resource matrix.
#[derive(Debug)]
struct FakeWorkloadFactory {
    descriptor: &'static WorkloadDescriptor,
    resources: ResourceRequirementsV2,
}

impl FakeWorkloadFactory {
    fn new(id: &'static str, resources: ResourceRequirementsV2) -> Self {
        Self {
            descriptor: Box::leak(Box::new(WorkloadDescriptor {
                id,
                description: "test-only workload",
            })),
            resources,
        }
    }
}

impl WorkloadFactory for FakeWorkloadFactory {
    fn descriptor(&self) -> &'static WorkloadDescriptor {
        self.descriptor
    }

    fn validate(&self, _authored: &RawValue) -> anyhow::Result<Box<dyn ValidatedWorkloadConfig>> {
        Ok(Box::new(()))
    }

    fn requirements(
        &self,
        _config: &dyn ValidatedWorkloadConfig,
    ) -> anyhow::Result<WorkloadRequirements> {
        Ok(WorkloadRequirements {
            transport_features: Default::default(),
            resources: self.resources,
        })
    }
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn streaming_registry() -> AIPerfRegistry {
    let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
    registry
        .register_extension(&aiperf_runtime::engine::registry::HttpExtension)
        .expect("http and dry_run transports");
    registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("local")))
        .expect("source");
    registry
        .register_stream_source(Arc::new(FakeSourceFactory::follow_only("tail_only")))
        .expect("follow-only source");
    registry
        .register_stream_format(Arc::new(FakeFormatFactory::new("jsonl")))
        .expect("format");
    registry
        .register_stream_format(Arc::new(FakeFormatFactory::resident("resident")))
        .expect("resident format");
    registry
        .register_stream_session_program(Arc::new(FakeSessionProgramFactory::new("conversation")))
        .expect("session program");
    registry
        .register_stream_action_sink(Arc::new(FakeActionSinkFactory::new("scheduled_request")))
        .expect("action sink");
    registry
        .register_stream_action_sink(Arc::new(FakeActionSinkFactory::new("session_state")))
        .expect("second action sink");
    registry
        .register_stream_checkpoint_backend(Arc::new(FakeCheckpointBackendFactory::new("memory")))
        .expect("checkpoint backend");
    registry
}

fn http_transport(registry: &AIPerfRegistry) -> &'static TransportDescriptor {
    registry
        .transport_descriptors()
        .into_iter()
        .find(|descriptor| descriptor.id == "http")
        .expect("http transport is linked")
}

fn empty_policy() -> PreparedStreamingIssuePolicy {
    PreparedStreamingIssuePolicy::new([]).expect("empty policy is canonical")
}

fn limits() -> serde_json::Value {
    json!({
        "acquired_partitions": 2,
        "decoded_fragments": 32,
        "decoded_bytes": 4096,
        "state_memory": 4096,
        "state_disk": 8192,
    })
}

/// The authored Config-v2 `dataset_streams:`/`shadow_replay:` pair.
fn authored_streams_yaml() -> (serde_json::Value, serde_json::Value) {
    (
        json!({
            "items": [{
                "id": "shadow_input",
                "source": {"id": "local", "config": {"mode": "follow", "path": "/traces"}},
                "format": {"id": "jsonl", "config": {"schema": "test.source.v1"}},
                "session_program": {"id": "conversation", "config": {}},
                "limits": limits(),
            }],
        }),
        json!({
            "stream": "shadow_input",
            "actions": {
                "request": {"id": "scheduled_request", "config": {}},
                "session_terminal": {"id": "session_state", "config": {}},
            },
            "time": {"mode": "relative"},
            "ordering": {"watermark": "source_order", "late": "fail"},
            "overload": {"mode": "backpressure"},
            "checkpoint": {"mode": "none"},
        }),
    )
}

fn stream_config(streams: serde_json::Value, replay: serde_json::Value) -> BenchmarkConfig {
    serde_json::from_value(json!({
        "dataset_streams": streams,
        "shadow_replay": replay,
    }))
    .expect("stream config decodes")
}

fn spec_from(streams: serde_json::Value, replay: serde_json::Value) -> DatasetStreamsSpecV2 {
    let mut value = streams;
    value["shadow_replay"] = replay;
    serde_json::from_value(value).expect("dataset stream spec decodes")
}

fn wire(cfg: serde_json::Value) -> serde_json::Value {
    json!({
        "benchmark_id": "stream-test",
        "artifact_dir": "/tmp/stream-test",
        "cfg": cfg,
    })
}

fn finite_dataset() -> serde_json::Value {
    json!({"type": "synthetic", "prompts": {"batch_size": 1, "isl": {"mean": 8.0}}})
}

fn authored_run(resources: serde_json::Value, workload: &str) -> AuthoredRunSpecV2 {
    serde_json::from_value(json!({
        "identity": {"benchmark_id": "stream-test"},
        "artifact_target": "/tmp/stream-test",
        "transport": {"type": "http", "config": {}},
        "workload": {"type": workload, "config": {}},
        "resources": resources,
    }))
    .expect("authored run decodes")
}

// ---------------------------------------------------------------------------
// 1–2: authored decoding and strictness.
// ---------------------------------------------------------------------------

#[test]
fn dataset_stream_resource_projects_without_opening_factories() {
    let before = effects();
    let (streams, replay) = authored_streams_yaml();
    let cfg = stream_config(streams, replay);
    let stream_section = cfg
        .dataset_streams
        .as_ref()
        .expect("dataset_streams present");
    assert_eq!(stream_section.items.len(), 1);
    assert_eq!(workload_kind(&cfg), WorkloadKind::ShadowReplay);
    assert_eq!(workload_kind(&cfg).workload_id(), "shadow_replay");
    assert_eq!(effects(), before, "authored decoding must open no factory");
}

#[test]
fn unknown_field_in_dataset_stream_is_rejected() {
    // Strictness lives on the nested types: `BenchmarkConfig` itself is
    // deliberately lenient, so the typo must be inside `dataset_streams`.
    let (mut streams, replay) = authored_streams_yaml();
    streams["items"][0]["limitz"] = limits();
    let error = serde_json::from_value::<BenchmarkConfig>(json!({
        "dataset_streams": streams,
        "shadow_replay": replay,
    }))
    .expect_err("unknown nested key must fail");
    assert!(error.to_string().contains("limitz"), "{error}");
}

// ---------------------------------------------------------------------------
// 3–4, 7–9: wire-only structural validation.
// ---------------------------------------------------------------------------

fn validate_streams(streams: serde_json::Value, replay: serde_json::Value) -> anyhow::Result<()> {
    let run = serde_json::from_value::<BenchmarkRunWireV2>(wire(json!({
        "dataset_streams": streams,
        "shadow_replay": replay,
    })))?;
    run.into_authored()?.validate_outer()
}

#[test]
fn duplicate_stream_ids_are_rejected() {
    let (mut streams, replay) = authored_streams_yaml();
    let duplicate = streams["items"][0].clone();
    streams["items"].as_array_mut().expect("items").push(duplicate);
    let error = validate_streams(streams, replay).expect_err("duplicate id must fail");
    assert!(
        error.to_string().contains("duplicate dataset_streams.items id"),
        "{error}"
    );
}

#[test]
fn duplicate_shadow_replay_action_binding_is_rejected() {
    // A JSON object cannot express a repeated key, so the duplicate is authored
    // as the sequence-of-pairs spelling the strict deserializer also accepts.
    let (streams, mut replay) = authored_streams_yaml();
    replay["actions"] = json!([
        ["request", {"id": "scheduled_request", "config": {}}],
        ["request", {"id": "session_state", "config": {}}],
    ]);
    let error = validate_streams(streams, replay).expect_err("duplicate binding must fail");
    assert!(
        error.to_string().contains("duplicate shadow_replay action binding"),
        "{error}"
    );
}

#[test]
fn shadow_replay_stream_must_name_a_configured_stream() {
    let (streams, mut replay) = authored_streams_yaml();
    replay["stream"] = json!("missing_stream");
    let error = validate_streams(streams, replay).expect_err("dangling reference must fail");
    assert!(error.to_string().contains("missing_stream"), "{error}");
}

#[test]
fn zero_stream_limit_is_rejected() {
    for field in [
        "acquired_partitions",
        "decoded_fragments",
        "decoded_bytes",
        "state_memory",
        "state_disk",
    ] {
        let (mut streams, replay) = authored_streams_yaml();
        streams["items"][0]["limits"][field] = json!(0);
        let error = validate_streams(streams, replay).expect_err("zero limit must fail");
        let message = error.to_string();
        assert!(message.contains(field), "{field}: {message}");
        assert!(message.contains("must be positive"), "{field}: {message}");
    }
}

#[test]
fn non_finite_checkpoint_interval_is_rejected() {
    // `mode: none` forbids both companion fields.
    let (streams, mut replay) = authored_streams_yaml();
    replay["checkpoint"] = json!({"mode": "none", "interval_seconds": 1.0});
    let error = validate_streams(streams, replay).expect_err("none + interval must fail");
    assert!(error.to_string().contains("forbids"), "{error}");

    // `mode: periodic` requires both, and the interval must be finite/positive.
    for interval in [json!(0.0), json!(-1.0)] {
        let (streams, mut replay) = authored_streams_yaml();
        replay["checkpoint"] = json!({
            "mode": "periodic",
            "interval_seconds": interval,
            "backend": {"id": "memory", "config": {}},
        });
        let error = validate_streams(streams, replay).expect_err("bad interval must fail");
        assert!(
            error.to_string().contains("finite and positive"),
            "{interval}: {error}"
        );
    }

    let (streams, mut replay) = authored_streams_yaml();
    replay["checkpoint"] = json!({"mode": "periodic", "interval_seconds": 1.0});
    let error = validate_streams(streams, replay).expect_err("periodic needs a backend");
    assert!(error.to_string().contains("requires"), "{error}");
}

// ---------------------------------------------------------------------------
// 5–6, 10–11: Config-v2 cross-field refusal, from both owners.
// ---------------------------------------------------------------------------

#[test]
fn mixed_datasets_and_dataset_streams_is_a_validation_error() {
    let (streams, replay) = authored_streams_yaml();
    let cfg: BenchmarkConfig = serde_json::from_value(json!({
        "datasets": [finite_dataset()],
        "dataset_streams": streams.clone(),
        "shadow_replay": replay.clone(),
    }))
    .expect("mixed config decodes");
    let error = aiperf_runtime::config::validate::validate(&cfg)
        .expect_err("config::validate rejects the mix");
    assert!(error.to_string().contains("mutually exclusive"), "{error}");

    let run = serde_json::from_value::<BenchmarkRunWireV2>(wire(json!({
        "datasets": [finite_dataset()],
        "dataset_streams": streams,
        "shadow_replay": replay,
    })))
    .expect("mixed wire decodes");
    let error = run.validate_outer().expect_err("the wire rejects the mix too");
    assert!(
        error.to_string().contains("cannot author both"),
        "{error}"
    );
}

#[test]
fn shadow_replay_without_dataset_streams_is_rejected() {
    let (streams, replay) = authored_streams_yaml();

    let cfg: BenchmarkConfig = serde_json::from_value(json!({
        "datasets": [finite_dataset()],
        "shadow_replay": replay,
    }))
    .expect("replay-only config decodes");
    let error = aiperf_runtime::config::validate::validate(&cfg).expect_err("replay needs streams");
    assert!(
        error.to_string().contains("shadow_replay requires dataset_streams"),
        "{error}"
    );

    let cfg: BenchmarkConfig =
        serde_json::from_value(json!({"dataset_streams": streams})).expect("streams-only decodes");
    let error = aiperf_runtime::config::validate::validate(&cfg).expect_err("streams need replay");
    assert!(
        error.to_string().contains("dataset_streams requires shadow_replay"),
        "{error}"
    );
}

#[test]
fn accuracy_with_dataset_streams_is_refused() {
    let (streams, replay) = authored_streams_yaml();
    let cfg: BenchmarkConfig = serde_json::from_value(json!({
        "dataset_streams": streams,
        "shadow_replay": replay,
        "accuracy": {"benchmark": "mmlu", "enable_cot": null, "grader": null, "n_shots": null,
                     "system_prompt": null, "tasks": null, "verbose": false},
    }))
    .expect("accuracy stream config decodes");
    let error = aiperf_runtime::config::validate::validate(&cfg).expect_err("accuracy is refused");
    assert!(error.to_string().contains("finite dataset"), "{error}");
}

#[test]
fn resident_exporter_with_dataset_streams_is_refused() {
    let (streams, replay) = authored_streams_yaml();
    let with_artifacts = |artifacts: serde_json::Value| -> BenchmarkConfig {
        serde_json::from_value(json!({
            "dataset_streams": streams.clone(),
            "shadow_replay": replay.clone(),
            "artifacts": artifacts,
        }))
        .expect("artifact stream config decodes")
    };

    for field in [
        "dataset_analysis_path",
        "graph_trace_summary_path",
        "graph_replay_provenance_path",
    ] {
        let cfg = with_artifacts(json!({field: "out.json"}));
        let error = aiperf_runtime::config::validate::validate(&cfg)
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{field}: {error}");
        assert!(error.contains("complete dataset resident"), "{field}: {error}");
    }

    // Per-record exporters append per record and stay accepted.
    let cfg = with_artifacts(json!({"records_path": "records.jsonl"}));
    aiperf_runtime::config::validate::validate(&cfg).expect("per-record exporters are accepted");
}

// ---------------------------------------------------------------------------
// 12–14: resource presence and the stock catalog.
// ---------------------------------------------------------------------------

#[test]
fn stock_catalog_advertises_no_shadow_replay_workload() {
    let registry = AIPerfRegistry::builtin().expect("builtin registry");
    assert!(
        !registry
            .workload_descriptors()
            .iter()
            .any(|descriptor| descriptor.id == "shadow_replay"),
        "the stock catalog must not advertise an unexecutable streaming workload"
    );
    assert!(registry.stream_source_descriptors().is_empty());
    assert!(registry.stream_format_descriptors().is_empty());
    assert!(registry.stream_session_program_descriptors().is_empty());
    assert!(registry.stream_action_sink_descriptors().is_empty());
    assert!(registry.stream_checkpoint_backend_descriptors().is_empty());
}

fn registry_with_workload(id: &'static str, resources: ResourceRequirementsV2) -> AIPerfRegistry {
    let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
    registry
        .register_extension(&aiperf_runtime::engine::registry::HttpExtension)
        .expect("http transport");
    registry
        .register_workload(Arc::new(FakeWorkloadFactory::new(id, resources)))
        .expect("workload registration");
    registry
}

fn baseline_resources() -> serde_json::Value {
    json!({
        "models": {"items": [{"name": "model"}]},
        "endpoints": {"profiles": {}},
    })
}

#[test]
fn shadow_replay_requirements_require_dataset_streams() {
    let registry = registry_with_workload("fake_replay", ResourceRequirementsV2::shadow_replay());
    let run = authored_run(baseline_resources(), "fake_replay");
    assert!(!run.resource_is_present(RunResourceV2::DatasetStreams));
    let error = registry
        .validate_selection_for_run(&run)
        .expect_err("a stream workload without the resource must be refused")
        .to_string();
    assert!(error.contains("requires run.resources.dataset_streams"), "{error}");
}

#[test]
fn inference_workload_forbids_dataset_streams() {
    let registry = registry_with_workload("fake_inference", ResourceRequirementsV2::inference());
    let (streams, replay) = authored_streams_yaml();
    let mut resources = baseline_resources();
    let mut stream_resource = streams;
    stream_resource["shadow_replay"] = replay;
    resources["dataset_streams"] = stream_resource;
    let run = authored_run(resources, "fake_inference");
    assert!(run.resource_is_present(RunResourceV2::DatasetStreams));
    let error = registry
        .validate_selection_for_run(&run)
        .expect_err("a finite workload with a stream resource must be refused")
        .to_string();
    assert!(error.contains("forbids run.resources.dataset_streams"), "{error}");
}

// ---------------------------------------------------------------------------
// 15–18: descriptor-only stream resolution.
// ---------------------------------------------------------------------------

#[test]
fn unknown_stream_component_fails_closed_with_available_ids() {
    let registry = streaming_registry();
    let transport = http_transport(&registry);
    let policy = empty_policy();
    let context = StreamingResourceContext {
        transport,
        endpoint: None,
        reliability_policy: &policy,
    };

    let cases: [(&str, Box<dyn Fn(&mut serde_json::Value, &mut serde_json::Value)>); 5] = [
        (
            "source",
            Box::new(|s: &mut serde_json::Value, _r: &mut serde_json::Value| {
                s["items"][0]["source"]["id"] = json!("nope");
            }),
        ),
        (
            "format",
            Box::new(|s: &mut serde_json::Value, _r: &mut serde_json::Value| {
                s["items"][0]["format"]["id"] = json!("nope");
            }),
        ),
        (
            "session_program",
            Box::new(|s: &mut serde_json::Value, _r: &mut serde_json::Value| {
                s["items"][0]["session_program"]["id"] = json!("nope");
            }),
        ),
        (
            "action_sink",
            Box::new(|_s: &mut serde_json::Value, r: &mut serde_json::Value| {
                r["actions"]["request"]["id"] = json!("nope");
            }),
        ),
        (
            "checkpoint_backend",
            Box::new(|_s: &mut serde_json::Value, r: &mut serde_json::Value| {
                r["checkpoint"] = json!({
                    "mode": "periodic",
                    "interval_seconds": 1.0,
                    "backend": {"id": "nope", "config": {}},
                });
            }),
        ),
    ];

    for (kind, mutate) in cases {
        let (mut streams, mut replay) = authored_streams_yaml();
        mutate(&mut streams, &mut replay);
        let spec = spec_from(streams, replay);
        let error = registry
            .validate_dataset_streams(&spec, context)
            .expect_err("unknown component must fail closed");
        let message = error.to_string();
        assert!(message.contains(kind), "{kind}: {message}");
        assert!(message.contains("nope"), "{kind}: {message}");
        assert!(
            message.contains("available:"),
            "{kind}: message must list what is compiled in: {message}"
        );
    }
}

#[test]
fn protocol_rejects_reliability_digest_mismatch_before_effects() {
    let registry = streaming_registry();
    let transport = http_transport(&registry);
    let policy = empty_policy();
    let (streams, replay) = authored_streams_yaml();
    let mut spec = spec_from(streams, replay);
    spec.reliability_policy_digest = Some(ReliabilityPolicyDigestV2::from_bytes([7u8; 32]));

    let before = effects();
    let error = registry
        .validate_dataset_streams(
            &spec,
            StreamingResourceContext {
                transport,
                endpoint: None,
                reliability_policy: &policy,
            },
        )
        .expect_err("a foreign policy digest must be refused");
    assert!(matches!(
        error,
        StreamingResourceError::ReliabilityPolicyDigestMismatch
    ));
    assert_eq!(
        effects(),
        before,
        "the digest gate must precede every lookup, construction, and poll"
    );
}

#[test]
fn descriptor_validation_performs_no_factory_effects() {
    let registry = streaming_registry();
    let transport = http_transport(&registry);
    let policy = empty_policy();
    let (streams, replay) = authored_streams_yaml();
    let spec = spec_from(streams, replay);

    let before = effects();
    let plan = registry
        .validate_dataset_streams(
            &spec,
            StreamingResourceContext {
                transport,
                endpoint: None,
                reliability_policy: &policy,
            },
        )
        .expect("the fake matrix must be admitted");
    assert_eq!(plan.stream_id(), "shadow_input");
    assert_eq!(plan.plans().len(), 2, "one plan per bound action kind");
    assert_eq!(
        effects(),
        before,
        "descriptor-only resolution must call no factory"
    );

    // The digest is a pure function of the admitted selection.
    let repeat = registry
        .validate_dataset_streams(
            &spec,
            StreamingResourceContext {
                transport,
                endpoint: None,
                reliability_policy: &policy,
            },
        )
        .expect("second resolution");
    assert_eq!(plan.selection_digest(), repeat.selection_digest());
}

#[test]
fn incompatible_selection_reports_the_first_capability() {
    let registry = streaming_registry();
    let transport = http_transport(&registry);
    let policy = empty_policy();
    let (mut streams, replay) = authored_streams_yaml();
    streams["items"][0]["source"]["id"] = json!("tail_only");
    streams["items"][0]["format"]["id"] = json!("resident");
    let spec = spec_from(streams, replay);

    let error = registry
        .validate_dataset_streams(
            &spec,
            StreamingResourceContext {
                transport,
                endpoint: None,
                reliability_policy: &policy,
            },
        )
        .expect_err("a resident format over an unbounded source must be refused");
    assert!(
        matches!(error, StreamingResourceError::Incompatible(_)),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// 19: the finite projection is unchanged.
// ---------------------------------------------------------------------------

#[test]
fn absent_dataset_streams_serializes_byte_identically() {
    let cfg = BenchmarkConfig::default();
    let value = serde_json::to_value(&cfg).expect("serialize");
    let object = value.as_object().expect("cfg is an object");
    assert!(
        !object.contains_key("dataset_streams"),
        "an absent stream resource must not appear on the wire"
    );
    assert!(!object.contains_key("shadow_replay"));
}

#[test]
fn finite_run_projection_keeps_its_dataset_key() {
    let run = serde_json::from_value::<BenchmarkRunWireV2>(wire(json!({
        "datasets": [finite_dataset()],
        "transport": {"type": "http"},
    })))
    .expect("finite wire decodes");
    let authored = run.into_authored().expect("finite projection");
    assert_eq!(authored.workload.id.as_str(), "scheduled");
    assert!(authored.dataset_streams.is_none());
    let config = serde_json::from_str::<serde_json::Value>(authored.workload.config.get())
        .expect("workload config is JSON");
    let keys = config
        .as_object()
        .expect("workload config is an object")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        ["worker_count", "dataset", "tokenizer", "phases", "failure_policy"],
        "the finite projection's key order must not move"
    );
}
