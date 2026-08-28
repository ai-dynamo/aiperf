// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate extension registration contracts.

use std::collections::BTreeMap;

use aiperf_runtime::dataset::{ConversationMetadata, Sampler, SamplerFactory, SessionId};
use aiperf_runtime::endpoints::{
    CreditPhase, EffectiveEndpointConfig, EndpointDescriptor, EndpointFactory, EndpointId,
    EndpointResult, ExtractedPayload, Media, Modality, ParsedResponse, PreparedEndpoint,
    PreparedRequest, RawEndpointConfig, ReadinessPolicy, RequestRecord, ResponseData,
    ServerResponse, Turn,
};
use aiperf_runtime::extensions::{
    AIPerfExtension, AIPerfRegistry, AIPerfRegistryFactory, ExtensionError,
};
use aiperf_runtime::rng::RngRoot;

struct ExternalSampler {
    id: SessionId,
}

impl Sampler for ExternalSampler {
    fn next(&mut self) -> SessionId {
        self.id.clone()
    }
}

#[derive(Clone, Copy)]
struct ExternalSamplerFactory {
    name: &'static str,
}

impl SamplerFactory for ExternalSamplerFactory {
    fn name(&self) -> &str {
        self.name
    }

    fn create(
        &self,
        metadata: &[ConversationMetadata],
        _root: RngRoot,
    ) -> aiperf_runtime::dataset::Result<Box<dyn Sampler>> {
        let id = metadata
            .first()
            .map(|metadata| metadata.conversation_id.clone())
            .unwrap_or_else(|| SessionId::from("external"));
        Ok(Box::new(ExternalSampler { id }))
    }
}

struct ExternalExtension;

static EXTERNAL_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "external_echo",
    aliases: &["external_echo_v1"],
    description: "Test-only compiled echo dialect",
    endpoint_path: Some("/v1/external/echo"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "External Echo Metrics",
    service_kind: "external_echo",
};

#[derive(Debug, Clone, Copy)]
struct ExternalEndpointFactory;

impl EndpointFactory for ExternalEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &EXTERNAL_DESCRIPTOR
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        Ok(Box::new(ExternalPreparedEndpoint {
            config,
            headers: BTreeMap::from([("x-external".into(), "echo".into())]),
        }))
    }
}

#[derive(Debug)]
struct ExternalPreparedEndpoint {
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
}

impl PreparedEndpoint for ExternalPreparedEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &EXTERNAL_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(
        &self,
        request: &PreparedRequest<'_>,
    ) -> EndpointResult<aiperf_runtime::body_plan::BodyPlan> {
        let text = request
            .turns()
            .last()
            .and_then(|turn| turn.texts.first())
            .and_then(|media| media.contents.first())
            .cloned()
            .unwrap_or_default();
        let payload = serde_json::json!({
            "model": request.primary_model_name(),
            "echo": text,
            "stream": self.config.streaming(),
        });
        Ok(aiperf_runtime::body_plan::BodyPlan::from_object(
            payload.as_object().expect("external payload is an object"),
        )?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, _model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Unsupported {
            reason: "test dialect has no readiness request",
        })
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        Ok(response
            .json
            .as_ref()
            .and_then(|value| value.get("echo"))
            .and_then(serde_json::Value::as_str)
            .map(|text| ParsedResponse {
                perf_ns: response.perf_ns,
                data: Some(ResponseData::Text { text: text.into() }),
                usage: None,
                sources: None,
            }))
    }

    fn extract_payload_inputs(&self, body: &serde_json::Value) -> ExtractedPayload {
        ExtractedPayload {
            texts: body
                .get("echo")
                .and_then(serde_json::Value::as_str)
                .map(|text| vec![text.into()])
                .unwrap_or_default(),
            ..ExtractedPayload::default()
        }
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        record
            .responses
            .iter()
            .filter_map(|response| self.parse_response(response).transpose())
            .collect()
    }

    fn build_assistant_turn(&self, _record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        Ok(None)
    }

    fn captures_assistant_turn(&self) -> bool {
        false
    }
}

impl AIPerfExtension for ExternalExtension {
    fn name(&self) -> &str {
        "external-test"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "external" })?;
        registry.register_endpoint_factory(ExternalEndpointFactory)?;
        Ok(())
    }
}

struct ExternalRegistryFactory;

impl AIPerfRegistryFactory for ExternalRegistryFactory {
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError> {
        AIPerfRegistry::builtin()?.with_extensions([&ExternalExtension as &dyn AIPerfExtension])
    }
}

struct PartiallyFailingExtension;

impl AIPerfExtension for PartiallyFailingExtension {
    fn name(&self) -> &str {
        "partial"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.register_endpoint_factory(ExternalEndpointFactory)?;
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "staged" })?;
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "random" })?;
        Ok(())
    }
}

fn metadata() -> Vec<ConversationMetadata> {
    vec![ConversationMetadata {
        conversation_id: SessionId::from("conversation-1"),
        turns: Vec::new(),
        context_mode: None,
        accuracy: None,
        dag: None,
    }]
}

#[test]
fn linked_extension_registers_and_resolves_a_trait_implementation() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let mut sampler = registry
        .samplers()
        .create("external", &metadata(), RngRoot::new(Some(7)))
        .unwrap();
    assert_eq!(sampler.next().as_str(), "conversation-1");
    let alias = EndpointId::new("external_echo_v1").unwrap();
    assert_eq!(
        registry.endpoints().canonical_id(&alias).unwrap().as_str(),
        "external_echo"
    );
    let prepared = registry
        .endpoints()
        .prepare(&alias, RawEndpointConfig::default())
        .unwrap();
    let turn = Turn {
        texts: vec![Media::new(vec!["hello".into()])],
        ..Turn::default()
    };
    let request = PreparedRequest::new(
        "external-model",
        std::slice::from_ref(&turn),
        None,
        None,
        CreditPhase::Profiling,
        None,
        None,
        None,
    );
    let body: serde_json::Value = serde_json::from_slice(
        &prepared
            .format_payload(&request)
            .unwrap()
            .materialize_standalone()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        body,
        serde_json::json!({"model":"external-model","echo":"hello","stream":false})
    );
    assert_eq!(prepared.headers()["x-external"], "echo");
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        ["external-test"]
    );
}

#[test]
fn custom_distribution_builds_its_registry_through_the_factory_seam() {
    let registry = ExternalRegistryFactory.build().unwrap();
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        ["external-test"]
    );
    assert!(
        registry
            .samplers()
            .create("external", &metadata(), RngRoot::new(Some(7)))
            .is_ok()
    );
}

#[test]
fn duplicate_extension_name_is_rejected() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let error = registry.register_extension(&ExternalExtension).unwrap_err();
    assert!(error.to_string().contains("duplicate AIPerf extension"));
}

#[test]
fn failed_extension_does_not_leak_earlier_registrations() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    let error = registry
        .register_extension(&PartiallyFailingExtension)
        .unwrap_err();
    assert!(error.to_string().contains("duplicate sampler strategy"));
    assert!(
        registry
            .samplers()
            .create("staged", &metadata(), RngRoot::new(Some(7)))
            .is_err()
    );
    assert!(
        registry
            .endpoints()
            .canonical_id(&EndpointId::new("external_echo").unwrap())
            .is_err()
    );
    assert_eq!(registry.extension_names().len(), 0);
}

#[cfg(feature = "streaming")]
mod streaming_categories {
    use std::sync::Arc;

    use aiperf_runtime::endpoints::EndpointDescriptor;
    use aiperf_runtime::engine::registry::{TransportDescriptor, WorkloadDescriptor};
    use aiperf_runtime::extensions::{AIPerfExtension, AIPerfRegistry, ExtensionError};
    use aiperf_runtime::streaming::{
        action::{
            ActionExecutionError, ActionFailureCode, ActionPlacement, ActionResultRetention,
            DatasetActionSchema, EndpointRetrySafety, PreparedStreamingActionBinding,
            StreamingActionSinkDescriptor, StreamingActionSinkFactory,
            StreamingActionSinkPrepareContext, ValidatedStreamingActionSinkConfig,
        },
        checkpoint::CheckpointError,
        checkpoint_backend::{
            CheckpointBackendPlacement, CheckpointBackendPrepareContext,
            CheckpointBackendRequirements, CheckpointRetention, StreamingCheckpointBackend,
            StreamingCheckpointBackendDescriptor, StreamingCheckpointBackendFactory,
            ValidatedCheckpointBackendConfig,
        },
        failure::{DecodeFailureCode, SessionFailureCode, SourceFailureCode},
        format::{
            FormatProjection, FormatStateRetention, StreamFormatError, StreamingDatasetFormat,
            StreamingDatasetFormatFactory, StreamingFormatDescriptor,
            StreamingFormatPrepareContext, ValidatedStreamingFormatConfig,
        },
        identity::ContentDigest,
        session::{
            SessionClosureCapability, SessionCoordinatorError, SessionPlacement,
            SessionStateRetention, StreamingSessionCoordinator, StreamingSessionPrepareContext,
            StreamingSessionProgramDescriptor, StreamingSessionProgramFactory,
            ValidatedStreamingSessionProgramConfig,
        },
        source::{
            PartitionAccessKind, PreparedStreamingDatasetSource, StreamSourceError,
            StreamingDatasetSourceFactory, StreamingResumeGranularity, StreamingSourceDescriptor,
            StreamingSourceMode, StreamingSourceOrdering, StreamingSourcePlacement,
            StreamingSourcePrepareContext, StreamingSourceRetention,
            ValidatedStreamingSourceConfig,
        },
    };
    use serde_json::value::RawValue;

    static EXTERNAL_SOURCE: StreamingSourceDescriptor = StreamingSourceDescriptor {
        id: "external_stream_source",
        description: "Test-only compiled streaming source",
        modes: &[StreamingSourceMode::Finite],
        access: &[PartitionAccessKind::Sequential],
        ordering: StreamingSourceOrdering::Partition,
        resume: &[StreamingResumeGranularity::Partition],
        has_event_time: false,
        has_stable_record_ids: false,
        retention: StreamingSourceRetention::BoundedMemory,
        placement: StreamingSourcePlacement::ControllerOnly,
        supports_virtual_clock: true,
    };

    static EXTERNAL_FORMAT: StreamingFormatDescriptor = StreamingFormatDescriptor {
        id: "external_stream_format",
        description: "Test-only compiled streaming format",
        semantic_digest: ContentDigest::from_bytes([7u8; 32]),
        media_types: &["application/jsonl"],
        input_schemas: &["external.source.v1"],
        required_access: PartitionAccessKind::Sequential,
        projection: FormatProjection::FullRecord,
        output_schema: "external.fragment.v1",
        has_event_time: false,
        has_stable_record_ids: false,
        retention: FormatStateRetention::BoundedMemory,
        supports_virtual_clock: true,
    };

    static EXTERNAL_SESSION: StreamingSessionProgramDescriptor =
        StreamingSessionProgramDescriptor {
            id: "external_stream_session",
            description: "Test-only compiled session program",
            fragment_input_schemas: &["external.fragment.v1"],
            action_schemas: &["external.action.v1"],
            closure: &[SessionClosureCapability::ExplicitClose],
            retention: SessionStateRetention::BoundedMemory,
            placement: SessionPlacement::ControllerCanonical,
            supports_virtual_clock: true,
        };

    static EXTERNAL_SINK: StreamingActionSinkDescriptor = StreamingActionSinkDescriptor {
        id: "external_stream_sink",
        description: "Test-only compiled action sink",
        accepted_schemas: &["external.action.v1"],
        transport_ids: &["http"],
        endpoint_kinds: &["chat"],
        retention: ActionResultRetention::StreamingTerminal,
        placement: ActionPlacement::WorkerLocal,
        supports_virtual_clock: true,
        endpoint_retry_safety: EndpointRetrySafety::Unproven,
    };

    static EXTERNAL_BACKEND: StreamingCheckpointBackendDescriptor =
        StreamingCheckpointBackendDescriptor {
            id: "external_stream_backend",
            description: "Test-only compiled checkpoint backend",
            is_durable: false,
            has_leased_readers: false,
            has_atomic_generations: true,
            has_result_segments: false,
            protects_sensitive_state: false,
            retention: CheckpointRetention::Ephemeral,
            placement: CheckpointBackendPlacement::ControllerLocal,
            supports_virtual_clock: true,
        };

    #[derive(Debug, Clone, Copy)]
    struct ExternalStreamSource;

    impl StreamingDatasetSourceFactory for ExternalStreamSource {
        fn descriptor(&self) -> &'static StreamingSourceDescriptor {
            &EXTERNAL_SOURCE
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

    #[derive(Debug, Clone, Copy)]
    struct ExternalStreamFormat;

    impl StreamingDatasetFormatFactory for ExternalStreamFormat {
        fn descriptor(&self) -> &'static StreamingFormatDescriptor {
            &EXTERNAL_FORMAT
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

    #[derive(Debug, Clone, Copy)]
    struct ExternalStreamSession;

    impl StreamingSessionProgramFactory for ExternalStreamSession {
        fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor {
            &EXTERNAL_SESSION
        }

        fn validate(
            &self,
            _authored: &RawValue,
            _format: &StreamingFormatDescriptor,
            _workload: &WorkloadDescriptor,
        ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError>
        {
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

    #[derive(Debug, Clone, Copy)]
    struct ExternalStreamSink;

    impl StreamingActionSinkFactory for ExternalStreamSink {
        fn descriptor(&self) -> &'static StreamingActionSinkDescriptor {
            &EXTERNAL_SINK
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

    #[derive(Debug, Clone, Copy)]
    struct ExternalStreamBackend;

    impl StreamingCheckpointBackendFactory for ExternalStreamBackend {
        fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
            &EXTERNAL_BACKEND
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

    struct ExternalStreamingExtension;

    impl AIPerfExtension for ExternalStreamingExtension {
        fn name(&self) -> &str {
            "external-streaming"
        }

        fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
            registry
                .register_stream_source(Arc::new(ExternalStreamSource))
                .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
            registry
                .register_stream_format(Arc::new(ExternalStreamFormat))
                .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
            registry
                .register_stream_session_program(Arc::new(ExternalStreamSession))
                .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
            registry
                .register_stream_action_sink(Arc::new(ExternalStreamSink))
                .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
            registry
                .register_stream_checkpoint_backend(Arc::new(ExternalStreamBackend))
                .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
            Ok(())
        }
    }

    #[test]
    fn linked_extension_registers_one_factory_in_every_streaming_category() {
        let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
        registry
            .register_extension(&ExternalStreamingExtension)
            .expect("streaming extension registers");

        assert!(
            registry
                .stream_source_factory("external_stream_source")
                .is_some()
        );
        assert!(
            registry
                .stream_format_factory("external_stream_format")
                .is_some()
        );
        assert!(
            registry
                .stream_session_program_factory("external_stream_session")
                .is_some()
        );
        assert!(
            registry
                .stream_action_sink_factory("external_stream_sink")
                .is_some()
        );
        assert!(
            registry
                .stream_checkpoint_backend_factory("external_stream_backend")
                .is_some()
        );
        assert_eq!(
            registry.extension_names().collect::<Vec<_>>(),
            ["external-streaming"]
        );
    }

    #[test]
    fn duplicate_streaming_extension_leaves_no_streaming_registration() {
        let mut registry = AIPerfRegistry::builtin().expect("builtin registry");
        registry
            .register_extension(&ExternalStreamingExtension)
            .expect("first application");
        let error = registry
            .register_extension(&ExternalStreamingExtension)
            .expect_err("duplicate extension name");
        assert!(error.to_string().contains("duplicate AIPerf extension"));
        assert_eq!(registry.stream_source_descriptors().len(), 1);
        assert_eq!(registry.stream_format_descriptors().len(), 1);
        assert_eq!(registry.stream_session_program_descriptors().len(), 1);
        assert_eq!(registry.stream_action_sink_descriptors().len(), 1);
        assert_eq!(registry.stream_checkpoint_backend_descriptors().len(), 1);
    }
}
