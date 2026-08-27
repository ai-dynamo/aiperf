// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior tests for the generation-1 category contract boundary.
//!
//! These compile from outside the crate, exactly as a plugin `cdylib` does, so
//! anything they reach is reachable from `aiperf_plugin_api::` alone. They pin
//! the three factory trait shapes, the receipt's canonical field key set, the
//! sealed category and host-resource vocabularies, and the typed refusals a
//! factory returns instead of unwinding.

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use aiperf_core::artifact::ArtifactAccess;
use aiperf_core::endpoint::{Handle, Overrides, SegmentReader};
use aiperf_plugin_api::{
    AuthoredConfigV1, BoundaryRequest, BoundaryTerminal, CAPTURE_REQUIREMENTS_V1,
    CaptureRequirementV1, CategoryError, CategoryOutcome, ContentDigest, EMPTY_AUTHORED_CONFIG,
    Endpoint, EndpointFactory, ExportInputV1, ExporterCaptureRequirementsV1,
    ExporterCaptureRequirementsV1 as CaptureSet, ExporterError, ExporterFactory,
    FactoryValidationReceiptV1, FoldedProjectionV1, HOST_RESOURCES_V1, HostResourceSetV1,
    HostResourceV1, PLUGIN_CATEGORIES, PluginCategory, PreparedEndpoint, PreparedExporter,
    PreparedExporterV1, PreparedTransport, REGISTRY_ID_NORMALIZATION_VERSION,
    ReadinessCapabilityV1, RegistryId, RequestExecutionBuildContextV1, RequestExecutor,
    RequestTransportExecution, TransportExecutionShapeV1, TransportFactory, ValidationError,
    WebSocketCapabilityV1,
};

/// Normalize under the only supported version.
fn id(input: &str) -> RegistryId {
    RegistryId::new(input, REGISTRY_ID_NORMALIZATION_VERSION)
        .unwrap_or_else(|error| panic!("test id must normalize: {error}"))
}

/// A distinguishable digest without hashing anything.
fn digest(fill: u8) -> ContentDigest {
    ContentDigest::from_bytes([fill; 32])
}

fn receipt(category: PluginCategory, factory_id: RegistryId) -> FactoryValidationReceiptV1 {
    FactoryValidationReceiptV1::new(
        category,
        factory_id,
        digest(1),
        digest(2),
        digest(3),
        HostResourceSetV1::new([HostResourceV1::Clock]),
        CaptureSet::default(),
    )
}

// ---------------------------------------------------------------------------
// Endpoint category
// ---------------------------------------------------------------------------

struct TestEndpoint {
    id: RegistryId,
}

impl Endpoint for TestEndpoint {
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn format_payload(
        &self,
        segments: &dyn SegmentReader,
        handles: &[Handle],
        overrides: &Overrides,
    ) -> Result<Vec<u8>, CategoryError> {
        let mut body = Vec::new();
        for handle in handles {
            let wire = segments.wire(*handle).ok_or(CategoryError::Runtime {
                category: PluginCategory::Endpoint,
                reason: format!("unknown handle {handle}"),
            })?;
            body.extend_from_slice(&wire);
        }
        if !overrides.is_empty() {
            body.extend_from_slice(b"+overrides");
        }
        Ok(body)
    }
}

struct TestEndpointFactory {
    id: RegistryId,
}

impl EndpointFactory for TestEndpointFactory {
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn validate(&self, config: AuthoredConfigV1<'_>) -> Result<PreparedEndpoint, CategoryError> {
        if !config.is_empty_object() {
            return Err(CategoryError::InvalidConfiguration {
                category: PluginCategory::Endpoint,
                reason: "this factory takes no fields".to_owned(),
            });
        }
        Ok(PreparedEndpoint::new(
            receipt(PluginCategory::Endpoint, self.id.clone()),
            Rc::new(TestEndpoint {
                id: self.id.clone(),
            }),
        ))
    }
}

// ---------------------------------------------------------------------------
// Transport category
// ---------------------------------------------------------------------------

struct TestTerminal;

impl BoundaryTerminal for TestTerminal {
    fn is_success(&self) -> bool {
        true
    }

    fn error_type(&self) -> Option<&str> {
        None
    }
}

struct TestRequest {
    body: Vec<u8>,
}

impl BoundaryRequest for TestRequest {
    fn correlation_id(&self) -> u64 {
        7
    }

    fn body(&self) -> &[u8] {
        &self.body
    }
}

struct TestExecutor;

impl RequestExecutor for TestExecutor {
    fn execute<'a>(
        &'a self,
        request: &'a dyn BoundaryRequest,
    ) -> Pin<Box<dyn Future<Output = Box<dyn BoundaryTerminal>> + 'a>> {
        Box::pin(async move {
            let _ = request.correlation_id();
            Box::new(TestTerminal) as Box<dyn BoundaryTerminal>
        })
    }
}

struct TestExecution;

impl RequestTransportExecution for TestExecution {
    fn readiness(&self) -> ReadinessCapabilityV1 {
        ReadinessCapabilityV1::Supported
    }

    fn websocket(&self) -> WebSocketCapabilityV1 {
        WebSocketCapabilityV1::Supported {
            needs_session_affinity: true,
        }
    }

    fn build_executor(
        &self,
        _context: RequestExecutionBuildContextV1,
    ) -> Result<Rc<dyn RequestExecutor>, ValidationError> {
        Ok(Rc::new(TestExecutor))
    }
}

struct TestTransportFactory {
    id: RegistryId,
    shape: TransportExecutionShapeV1,
}

impl TransportFactory for TestTransportFactory {
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn execution_shape(&self) -> TransportExecutionShapeV1 {
        self.shape
    }

    fn validate(&self, _config: AuthoredConfigV1<'_>) -> Result<PreparedTransport, CategoryError> {
        let receipt = receipt(PluginCategory::Transport, self.id.clone());
        Ok(match self.shape {
            TransportExecutionShapeV1::Request => {
                PreparedTransport::request(receipt, Rc::new(TestExecution))
            }
            TransportExecutionShapeV1::Direct => PreparedTransport::direct(receipt),
        })
    }
}

// ---------------------------------------------------------------------------
// Exporter category
// ---------------------------------------------------------------------------

struct TestExporter {
    id: RegistryId,
    requirements: ExporterCaptureRequirementsV1,
}

impl PreparedExporterV1 for TestExporter {
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn requirements(&self) -> &ExporterCaptureRequirementsV1 {
        &self.requirements
    }

    fn export(
        &self,
        input: ExportInputV1<'_>,
        _artifacts: &dyn ArtifactAccess,
    ) -> Result<(), ExporterError> {
        if input.exact_records().is_none() {
            return Err(ExporterError::MissingCapture(
                CaptureRequirementV1::ExactRecordsV1,
            ));
        }
        Ok(())
    }
}

struct TestExporterFactory {
    id: RegistryId,
}

impl TestExporterFactory {
    fn requirements() -> ExporterCaptureRequirementsV1 {
        ExporterCaptureRequirementsV1::new([
            CaptureRequirementV1::FoldedProjectionV1(FoldedProjectionV1::GenAiClientHistogramsV1),
            CaptureRequirementV1::ExactRecordsV1,
        ])
    }
}

impl ExporterFactory for TestExporterFactory {
    fn id(&self) -> &RegistryId {
        &self.id
    }

    fn capture_requirements(&self) -> ExporterCaptureRequirementsV1 {
        Self::requirements()
    }

    fn validate(&self, _config: AuthoredConfigV1<'_>) -> Result<PreparedExporter, CategoryError> {
        Ok(PreparedExporter::new(
            receipt(PluginCategory::Exporter, self.id.clone()),
            Rc::new(TestExporter {
                id: self.id.clone(),
                requirements: Self::requirements(),
            }),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn receipt_field_keys_are_the_complete_canonical_set() {
    assert_eq!(
        FactoryValidationReceiptV1::FIELD_KEYS,
        &[
            "category",
            "factory_id",
            "descriptor_digest",
            "authored_config_digest",
            "semantic_config_digest",
            "host_resources",
            "capture_requirements",
        ]
    );
}

#[test]
fn receipt_carries_every_field_it_keys() {
    let value = receipt(PluginCategory::Exporter, id("otel"));

    assert_eq!(value.category(), PluginCategory::Exporter);
    assert_eq!(value.factory_id().as_str(), "otel");
    assert_eq!(value.descriptor_digest(), digest(1));
    assert_eq!(value.authored_config_digest(), digest(2));
    assert_eq!(value.semantic_config_digest(), digest(3));
    assert_eq!(
        value.host_resources().as_slice(),
        &[HostResourceV1::Clock][..]
    );
    assert_eq!(
        value.capture_requirements().as_slice(),
        &[CaptureRequirementV1::FinalReport][..]
    );

    assert_eq!(value.expect_category(PluginCategory::Exporter), Ok(()));
    assert_eq!(
        value.expect_category(PluginCategory::Transport),
        Err(ValidationError::CategoryMismatch {
            expected: PluginCategory::Transport,
            found: PluginCategory::Exporter,
        })
    );
}

#[test]
fn category_error_names_its_category_and_carries_a_message() {
    let invalid = CategoryError::InvalidConfiguration {
        category: PluginCategory::Endpoint,
        reason: "missing model".to_owned(),
    };
    assert_eq!(invalid.category(), Some(PluginCategory::Endpoint));
    assert!(invalid.to_string().contains("missing model"));

    let unsupported = CategoryError::UnsupportedCapability {
        category: PluginCategory::Transport,
        capability: "websocket".to_owned(),
    };
    assert_eq!(unsupported.category(), Some(PluginCategory::Transport));
    assert!(unsupported.to_string().contains("websocket"));

    let runtime = CategoryError::Runtime {
        category: PluginCategory::Exporter,
        reason: "disk full".to_owned(),
    };
    assert_eq!(runtime.category(), Some(PluginCategory::Exporter));
    assert!(runtime.to_string().contains("disk full"));

    // A pre-factory refusal names no category and keeps the validation error
    // reachable as an error source.
    let validation = CategoryError::from(ValidationError::UnknownCategory);
    assert_eq!(validation.category(), None);
    assert_eq!(validation.to_string(), "unknown plugin category");
}

#[test]
fn category_outcome_variants_are_exhausted_by_a_match() {
    let outcomes: [CategoryOutcome<u32>; 2] = [
        CategoryOutcome::Accepted(9),
        CategoryOutcome::Refused(CategoryError::Runtime {
            category: PluginCategory::Transport,
            reason: "refused".to_owned(),
        }),
    ];

    let mut accepted = 0_u32;
    let mut refused = 0_u32;
    for outcome in &outcomes {
        match outcome {
            CategoryOutcome::Accepted(value) => accepted += *value,
            CategoryOutcome::Refused(_) => refused += 1,
        }
    }
    assert_eq!((accepted, refused), (9, 1));

    assert!(outcomes[0].is_accepted());
    assert!(!outcomes[1].is_accepted());

    let ok: CategoryOutcome<u32> = Ok(4).into();
    assert_eq!(ok.into_result(), Ok(4));
    let err: CategoryOutcome<u32> = Err(CategoryError::Validation(ValidationError::Rejected(
        "no".to_owned(),
    )))
    .into();
    assert!(err.into_result().is_err());
}

#[test]
fn validation_errors_are_constructible_and_describe_themselves() {
    let cases = [
        (ValidationError::UnknownCategory, "unknown plugin category"),
        (
            ValidationError::UnknownCaptureProjection("nope".to_owned()),
            "nope",
        ),
        (ValidationError::AmbiguousExecutionShape, "exactly one"),
        (ValidationError::Rejected("bad".to_owned()), "bad"),
    ];
    for (error, fragment) in cases {
        assert!(
            error.to_string().contains(fragment),
            "{error} should mention {fragment}"
        );
    }

    assert_eq!(
        TransportExecutionShapeV1::exactly_one(&[]),
        Err(ValidationError::AmbiguousExecutionShape)
    );
    assert_eq!(
        TransportExecutionShapeV1::exactly_one(&[
            TransportExecutionShapeV1::Request,
            TransportExecutionShapeV1::Direct,
        ]),
        Err(ValidationError::AmbiguousExecutionShape)
    );
    assert_eq!(
        TransportExecutionShapeV1::exactly_one(&[TransportExecutionShapeV1::Direct]),
        Ok(TransportExecutionShapeV1::Direct)
    );
}

#[test]
fn plugin_category_vocabulary_is_sealed() {
    assert_eq!(
        PLUGIN_CATEGORIES,
        &[
            PluginCategory::Endpoint,
            PluginCategory::Transport,
            PluginCategory::Exporter,
        ]
    );
    for category in PLUGIN_CATEGORIES {
        assert_eq!(PluginCategory::parse(category.label()), Ok(*category));
        assert_eq!(category.to_string(), category.label());
    }
    assert_eq!(
        PluginCategory::parse("Endpoint"),
        Err(ValidationError::UnknownCategory)
    );
}

#[test]
fn host_resource_set_is_sorted_and_deduplicated() {
    assert_eq!(HOST_RESOURCES_V1.len(), 5);
    assert_eq!(HostResourceV1::Cancellation.to_string(), "cancellation");

    let set = HostResourceSetV1::new([
        HostResourceV1::Metrics,
        HostResourceV1::Clock,
        HostResourceV1::Metrics,
    ]);
    assert_eq!(
        set.as_slice(),
        &[HostResourceV1::Clock, HostResourceV1::Metrics][..]
    );
    assert!(set.contains(HostResourceV1::Clock));
    assert!(!set.contains(HostResourceV1::Graph));
    assert!(!set.is_empty());
    assert!(HostResourceSetV1::default().is_empty());

    // Declaration order must not change the receipt.
    assert_eq!(
        HostResourceSetV1::new([HostResourceV1::Clock, HostResourceV1::Metrics]),
        set
    );
}

#[test]
fn endpoint_factory_validates_authored_configuration() {
    let factory = TestEndpointFactory { id: id("chat") };
    assert_eq!(EndpointFactory::id(&factory).as_str(), "chat");

    let empty = AuthoredConfigV1::empty(EndpointFactory::id(&factory));
    assert!(empty.is_empty_object());
    assert_eq!(empty.json(), EMPTY_AUTHORED_CONFIG);
    assert_eq!(empty.id().as_str(), "chat");

    let prepared = factory
        .validate(empty)
        .unwrap_or_else(|error| panic!("empty config must validate: {error}"));
    assert_eq!(
        prepared.receipt().category(),
        PluginCategory::Endpoint,
        "the receipt binds the category position the factory occupies"
    );
    assert_eq!(prepared.endpoint().id().as_str(), "chat");

    let authored = AuthoredConfigV1::new(EndpointFactory::id(&factory), br#"{"model":"x"}"#);
    assert!(!authored.is_empty_object());
    // `PreparedEndpoint` is deliberately not `Debug` — it holds a plugin-owned
    // trait object the host may not inspect — so the refusal is matched rather
    // than unwrapped.
    let Err(refusal) = factory.validate(authored) else {
        panic!("unknown fields must be refused");
    };
    assert_eq!(refusal.category(), Some(PluginCategory::Endpoint));
}

#[test]
fn transport_factory_declares_exactly_one_execution_shape() {
    let request = TestTransportFactory {
        id: id("http"),
        shape: TransportExecutionShapeV1::Request,
    };
    let prepared = request
        .validate(AuthoredConfigV1::empty(TransportFactory::id(&request)))
        .unwrap_or_else(|error| panic!("request transport must validate: {error}"));
    assert_eq!(
        request.execution_shape(),
        TransportExecutionShapeV1::Request
    );
    assert_eq!(prepared.shape(), TransportExecutionShapeV1::Request);
    let execution = prepared
        .execution()
        .expect("a request transport contributes an execution");
    assert_eq!(execution.readiness(), ReadinessCapabilityV1::Supported);
    assert_eq!(
        execution.websocket(),
        WebSocketCapabilityV1::Supported {
            needs_session_affinity: true
        }
    );

    let direct = TestTransportFactory {
        id: id("dry_run"),
        shape: TransportExecutionShapeV1::Direct,
    };
    let prepared = direct
        .validate(AuthoredConfigV1::empty(TransportFactory::id(&direct)))
        .unwrap_or_else(|error| panic!("direct transport must validate: {error}"));
    assert_eq!(prepared.shape(), TransportExecutionShapeV1::Direct);
    assert!(
        prepared.execution().is_none(),
        "a direct transport drives its own execution"
    );
    assert_eq!(prepared.receipt().category(), PluginCategory::Transport);
}

#[test]
fn exporter_factory_declares_captures_before_validation() {
    let factory = TestExporterFactory { id: id("parquet") };

    // Requirements are answerable without validated configuration, and always
    // contain the finalized report.
    let requirements = factory.capture_requirements();
    assert_eq!(
        requirements.as_slice(),
        &[
            CaptureRequirementV1::FinalReport,
            CaptureRequirementV1::ExactRecordsV1,
            CaptureRequirementV1::FoldedProjectionV1(FoldedProjectionV1::GenAiClientHistogramsV1),
        ][..]
    );
    assert!(requirements.needs_exact_records());
    assert_eq!(requirements.as_slice(), CAPTURE_REQUIREMENTS_V1);
    assert!(!CaptureSet::default().needs_exact_records());

    let prepared = factory
        .validate(AuthoredConfigV1::empty(ExporterFactory::id(&factory)))
        .unwrap_or_else(|error| panic!("exporter must validate: {error}"));
    assert_eq!(prepared.receipt().category(), PluginCategory::Exporter);
    let exporter = prepared.exporter();
    assert_eq!(exporter.id().as_str(), "parquet");
    assert_eq!(exporter.requirements().as_slice(), requirements.as_slice());

    assert_eq!(
        ExporterError::MissingCapture(CaptureRequirementV1::ExactRecordsV1).to_string(),
        "required capture ExactRecordsV1 was not supplied"
    );
    assert_eq!(
        CaptureRequirementV1::parse("ExactRecordsV1"),
        Ok(CaptureRequirementV1::ExactRecordsV1)
    );
    assert_eq!(
        CaptureRequirementV1::parse("nope"),
        Err(ValidationError::UnknownCaptureProjection("nope".to_owned()))
    );
}

/// The three factories must be usable as trait objects: a host holds a
/// registered plugin behind a pointer, never as a concrete type.
#[test]
fn category_factories_are_object_safe() {
    let endpoint: Rc<dyn EndpointFactory> = Rc::new(TestEndpointFactory { id: id("chat") });
    let transport: Rc<dyn TransportFactory> = Rc::new(TestTransportFactory {
        id: id("http"),
        shape: TransportExecutionShapeV1::Request,
    });
    let exporter: Rc<dyn ExporterFactory> = Rc::new(TestExporterFactory { id: id("parquet") });

    assert_eq!(endpoint.id().as_str(), "chat");
    assert_eq!(
        transport.execution_shape(),
        TransportExecutionShapeV1::Request
    );
    assert!(exporter.capture_requirements().needs_exact_records());
}

/// The boundary request/terminal pair and the executor future are reachable and
/// driveable without any host runtime.
#[test]
fn request_executor_drives_a_boundary_request_to_terminal() {
    let executor: Rc<dyn RequestExecutor> = Rc::new(TestExecutor);
    let request = TestRequest {
        body: b"{}".to_vec(),
    };
    assert_eq!(request.correlation_id(), 7);
    assert_eq!(request.body(), b"{}");

    let terminal = block_on(executor.execute(&request));
    assert!(terminal.is_success());
    assert_eq!(terminal.error_type(), None);
}

/// Poll a future to completion without pulling in an async runtime: the
/// executor future here never yields.
fn block_on<F: Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|data| RawWaker::new(data, &VTABLE), |_| {}, |_| {}, |_| {});
    // SAFETY: every vtable entry is a no-op over a null data pointer, so the
    // waker is never dereferenced and cloning it produces the same inert value.
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the boundary executor future must not yield"),
    }
}
