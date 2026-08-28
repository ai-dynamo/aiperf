// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen streaming registry and descriptor-only capability agreement.

use std::sync::Arc;

use aiperf_runtime::endpoints::EndpointDescriptor;
use aiperf_runtime::engine::protocol::Catalog;
use aiperf_runtime::engine::registry::{
    StreamingCapabilityAgreement, StreamingCapabilitySelection, StreamingIncompatibleCapability,
    TransportDescriptor,
};
use aiperf_runtime::extensions::{
    AIPerfExtension, AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory,
    ExtensionError,
};
use aiperf_runtime::streaming::{
    action::{
        ActionExecutionError, ActionFailureCode, ActionPlacement, ActionResultRetention,
        DatasetActionSchema, EndpointRetrySafety, PreparedStreamingActionBinding,
        StreamingActionSinkDescriptor, StreamingActionSinkFactory,
        StreamingActionSinkPrepareContext, ValidatedStreamingActionSinkConfig,
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
use serde_json::value::RawValue;

// ---------------------------------------------------------------------------
// Test-local fakes. Each leaks one `'static` descriptor so a single fake type
// can stand in for a whole axis of the capability cross product.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct FakeSourceFactory {
    descriptor: &'static StreamingSourceDescriptor,
}

impl FakeSourceFactory {
    fn new(id: &'static str) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingSourceDescriptor {
                id,
                description: "test-only streaming source",
                modes: &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
                access: &[
                    PartitionAccessKind::Sequential,
                    PartitionAccessKind::SeekableLocal,
                    PartitionAccessKind::RangeReadable,
                ],
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
        Err(StreamSourceError::source(
            SourceFailureCode::SourceUnavailable,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingSourceConfig>,
        _context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
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
                retention: FormatStateRetention::BoundedMemory,
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
        Err(StreamFormatError::decode(DecodeFailureCode::Schema))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingFormatConfig>,
        _context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError> {
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
        _workload: &aiperf_runtime::engine::registry::WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError> {
        Err(SessionCoordinatorError::session(
            SessionFailureCode::MissingPredecessor,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        _context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError> {
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
        Self::with_schemas(id, &["test.action.v1"])
    }

    fn with_schemas(id: &'static str, accepted_schemas: &'static [&'static str]) -> Self {
        Self {
            descriptor: Box::leak(Box::new(StreamingActionSinkDescriptor {
                id,
                description: "test-only action sink",
                accepted_schemas,
                transport_ids: &["dry_run", "http"],
                endpoint_kinds: &["chat"],
                retention: ActionResultRetention::StreamingTerminal,
                placement: ActionPlacement::WorkerLocal,
                supports_virtual_clock: true,
                endpoint_retry_safety: EndpointRetrySafety::Unproven,
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
        Err(ActionExecutionError::action(
            ActionFailureCode::MissingBinding,
        ))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingActionSinkConfig>,
        _context: &StreamingActionSinkPrepareContext,
    ) -> Result<PreparedStreamingActionBinding, ActionExecutionError> {
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
                // Generation one never consults this field; it is declared
                // false so the agreement provably ignores it.
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
        Err(CheckpointError::ParticipantSetMismatch)
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedCheckpointBackendConfig>,
        _context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        Err(CheckpointError::ParticipantSetMismatch)
    }
}

fn fake_cross_product_registry(
    sources: [&'static str; 2],
    formats: [&'static str; 2],
    programs: [&'static str; 2],
    sinks: [&'static str; 2],
    transports: [&'static str; 2],
) -> AIPerfRegistry {
    // Transports come from the stock protocol-v2 registry, so the matrix runs
    // over real transport descriptors rather than a streaming-local fake.
    // `HttpExtension` registers both `http` and the always-built `dry_run`
    // execution leaf, so one extension supplies the whole transport axis.
    let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
    registry
        .register_extension(&aiperf_runtime::engine::registry::HttpExtension)
        .expect("http and dry_run transports");
    for id in sources {
        registry
            .register_stream_source(Arc::new(FakeSourceFactory::new(id)))
            .expect("source registration");
    }
    for id in formats {
        registry
            .register_stream_format(Arc::new(FakeFormatFactory::new(id)))
            .expect("format registration");
    }
    for id in programs {
        registry
            .register_stream_session_program(Arc::new(FakeSessionProgramFactory::new(id)))
            .expect("session program registration");
    }
    for id in sinks {
        registry
            .register_stream_action_sink(Arc::new(FakeActionSinkFactory::new(id)))
            .expect("action sink registration");
    }
    assert_eq!(
        registry
            .transport_descriptors()
            .into_iter()
            .filter(|descriptor| transports.contains(&descriptor.id))
            .count(),
        transports.len(),
        "matrix transports must be linked"
    );
    registry
}

// ---------------------------------------------------------------------------
// Tests named by the plan.
// ---------------------------------------------------------------------------

#[test]
fn duplicate_stream_source_registration_is_atomic() {
    let mut registry = AIPerfRegistry::empty_or_base();
    registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("fake")))
        .expect("first registration");
    let error = registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("FAKE")))
        .expect_err("normalized duplicate must fail");
    assert!(error.to_string().contains("duplicate streaming source ID"));
    assert_eq!(registry.stream_source_descriptors().len(), 1);
}

#[test]
fn supported_capability_cross_product_composes_without_concrete_type_switches() {
    let registry = fake_cross_product_registry(
        ["finite", "follow"],
        ["jsonl", "columnar"],
        ["conversation", "agent_graph"],
        ["scheduled_request", "session_state"],
        ["dry_run", "http"],
    );
    let selections = registry.declared_supported_cross_product();
    assert_eq!(selections.len(), 2 * 2 * 2 * 2 * 2);
    for selection in selections {
        let plan = StreamingCapabilityAgreement::validate(selection.descriptors())
            .expect("declared-supported combination must validate");
        assert_eq!(plan.selected_ids(), selection.ids());
        assert_eq!(plan.preparation_count_per_factory(), 1);
    }
}

#[test]
fn stream_descriptor_inventories_are_ordered_by_normalized_id() {
    let mut registry = AIPerfRegistry::empty_or_base();
    for id in ["Zeta", "alpha", "mid-name"] {
        registry
            .register_stream_source(Arc::new(FakeSourceFactory::new(id)))
            .expect("registration");
    }
    let ordered = registry
        .stream_source_descriptors()
        .into_iter()
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    // Iteration order is the normalized key order, but each entry still reports
    // its authored descriptor ID verbatim.
    assert_eq!(ordered, ["alpha", "mid-name", "Zeta"]);
}

#[test]
fn unknown_streaming_identifier_is_refused_by_every_lookup() {
    let mut registry = AIPerfRegistry::empty_or_base();
    registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("known")))
        .expect("registration");

    assert!(registry.stream_source_factory("shadow_replay").is_none());
    assert!(registry.stream_format_factory("shadow_replay").is_none());
    assert!(
        registry
            .stream_session_program_factory("shadow_replay")
            .is_none()
    );
    assert!(
        registry
            .stream_action_sink_factory("shadow_replay")
            .is_none()
    );
    assert!(registry.stream_checkpoint_backend_factory("s3").is_none());
    assert!(registry.stream_format_descriptors().is_empty());
    assert!(registry.stream_action_sink_descriptors().is_empty());
    assert!(registry.stream_checkpoint_backend_descriptors().is_empty());
    // No capability combination can be minted for an identifier that has no
    // descriptor, so an unknown selection is unrepresentable rather than
    // rejected late.
    assert!(registry.declared_supported_cross_product().is_empty());
}

struct PartiallyFailingStreamingExtension;

impl AIPerfExtension for PartiallyFailingStreamingExtension {
    fn name(&self) -> &str {
        "partial-streaming"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .register_stream_source(Arc::new(FakeSourceFactory::new("staged_source")))
            .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        registry
            .register_stream_format(Arc::new(FakeFormatFactory::new("staged_format")))
            .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        registry
            .register_stream_format(Arc::new(FakeFormatFactory::new("staged_format")))
            .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        Ok(())
    }
}

#[test]
fn failed_streaming_extension_leaves_no_partial_registration() {
    let mut registry = AIPerfRegistry::empty_or_base();
    let error = registry
        .register_extension(&PartiallyFailingStreamingExtension)
        .expect_err("duplicate format aborts the extension");
    assert!(error.to_string().contains("duplicate streaming format ID"));
    assert!(registry.stream_source_descriptors().is_empty());
    assert!(registry.stream_format_descriptors().is_empty());
    assert_eq!(registry.extension_names().len(), 0);
}

#[test]
fn cross_product_mismatch_names_every_selected_id_and_first_capability() {
    let source = FakeSourceFactory::new("source_a");
    let format = FakeFormatFactory::new("format_a");
    let session = FakeSessionProgramFactory::new("session_a");
    let sink = FakeActionSinkFactory::with_schemas("sink_a", &["other.action.v1"]);
    let transport = TransportDescriptor {
        id: "dry_run",
        description: "test-only transport descriptor",
        clock: aiperf_runtime::engine::registry::ClockKind::Real,
        features: &[],
        url_schemes: &[],
    };
    let transport: &'static TransportDescriptor = Box::leak(Box::new(transport));

    let selection = StreamingCapabilitySelection {
        source: source.descriptor(),
        format: format.descriptor(),
        session: session.descriptor(),
        action_sink: sink.descriptor(),
        transport,
        endpoint: None,
        checkpoint_backend: None,
    };
    let error = StreamingCapabilityAgreement::validate(selection.descriptors())
        .expect_err("no shared action schema");
    assert_eq!(
        error.capability,
        StreamingIncompatibleCapability::ActionSchema
    );
    let rendered = error.to_string();
    for id in ["source_a", "format_a", "session_a", "sink_a", "dry_run"] {
        assert!(rendered.contains(id), "missing {id} in {rendered}");
    }
    assert!(rendered.contains("action_schema"));
}

#[test]
fn catalog_omits_streaming_maps_when_no_streaming_factory_is_registered() {
    // `AIPerfRegistry::builtin()` is the engine-free subset: it applies no
    // streaming extension, so every streaming map stays empty and omitted. The
    // stock composition root is exercised separately below.
    let registry = AIPerfRegistry::builtin().expect("builtin registry");
    let document =
        serde_json::to_value(Catalog::from_registry(&registry)).expect("catalog serializes");
    let object = document.as_object().expect("catalog is an object");
    for key in [
        "stream_source",
        "stream_format",
        "stream_session_program",
        "stream_action_sink",
        "stream_checkpoint_backend",
    ] {
        assert!(
            !object.contains_key(key),
            "{key} must be omitted when empty"
        );
    }
}

#[test]
fn builtin_registry_exposes_only_the_conversation_session_program() {
    let registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("stock registry universe");
    let ids = registry
        .stream_session_program_descriptors()
        .into_iter()
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, ["conversation"]);
    assert!(
        registry
            .stream_session_program_factory("conversation")
            .is_some()
    );
    // The stock catalog registers no action-sink binding, so no streaming
    // capability combination can be minted from it yet. Source and format
    // built-ins are owned by their own lanes and are deliberately not asserted
    // here.
    assert!(registry.stream_action_sink_descriptors().is_empty());
    assert!(
        registry
            .stream_action_sink_factory("scheduled_request")
            .is_none()
    );
    assert!(registry.declared_supported_cross_product().is_empty());
    assert!(
        registry
            .stream_session_program_factory("shadow_replay")
            .is_none()
    );
}

#[test]
fn duplicate_conversation_session_program_is_rejected() {
    let mut registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("stock registry universe");
    let before = registry
        .stream_session_program_descriptors()
        .into_iter()
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    let error = registry
        .register_stream_session_program(Arc::new(
            aiperf_runtime::streaming::session::conversation::StreamingConversationProgramFactory,
        ))
        .expect_err("the built-in program already owns this identifier");
    assert!(
        error
            .to_string()
            .contains("duplicate streaming session program ID")
    );
    let after = registry
        .stream_session_program_descriptors()
        .into_iter()
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    assert_eq!(before, after);
}

#[test]
fn catalog_lists_registered_streaming_descriptors() {
    let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
    registry
        .register_stream_source(Arc::new(FakeSourceFactory::new("fake_source")))
        .expect("source registration");
    registry
        .register_stream_checkpoint_backend(Arc::new(FakeCheckpointBackendFactory::new(
            "fake_backend",
        )))
        .expect("backend registration");

    let document =
        serde_json::to_value(Catalog::from_registry(&registry)).expect("catalog serializes");
    assert_eq!(
        document["stream_source"]["fake_source"]["metadata"]["id"],
        serde_json::json!("fake_source")
    );
    assert_eq!(
        document["stream_checkpoint_backend"]["fake_backend"]["metadata"]["is_durable"],
        serde_json::json!(true)
    );
    assert!(document.get("stream_format").is_none());
}
