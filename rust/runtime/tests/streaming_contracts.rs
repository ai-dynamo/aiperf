// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{cell::Cell, fmt::Debug, num::NonZeroUsize, rc::Rc};

use aiperf_runtime::streaming::{
    action::{
        ActionCancelReceipt, ActionFailureCode, ActionPlacement, ActionResultRetention,
        BudgetedActionUpdate, PreparedStreamingActionBinding, StreamingActionDriver,
        StreamingActionDriverControl, StreamingActionDriverControlOps,
        StreamingActionSinkDescriptor, StreamingActionSinkFactory, StreamingActionSubmitter,
        ValidatedStreamingActionSinkConfig, action_execution_control,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind, CheckpointError,
        StreamingCheckpointParticipant,
    },
    checkpoint_backend::{
        CheckpointBackendPlacement, CheckpointRetention, LeasedGenerationReader,
        StreamingCheckpointBackend, StreamingCheckpointBackendDescriptor,
        StreamingCheckpointBackendFactory, StreamingGenerationTransaction,
        ValidatedCheckpointBackendConfig,
    },
    failure::{
        AcquisitionFailureCode, DecodeFailureCode, OrderingFailureCode, OrdinaryStreamingFailure,
        OrdinaryStreamingIssue, PlacementFailureCode, StableStreamingFailure, StreamFormatError,
        StreamSourceError, StreamingFailureStage, StreamingIssueClass, StreamingIssueOrderKey,
        StreamingIssueReportError, StreamingIssueReportStatus, StreamingIssueReporter,
        StreamingIssueReporterEndpoint, StreamingIssueReporterHandle, StreamingIssueScope,
    },
    format::{
        DecoderResumeState, FormatProjection, FormatStateRetention, StreamingDatasetFormat,
        StreamingDatasetFormatFactory, StreamingFormatDescriptor, StreamingPartitionDecoder,
        ValidatedStreamingFormatConfig,
    },
    identity::{ContentDigest, ImmutableObjectIdentity, LogicalReplayRunId},
    session::{
        DatasetActionSink, SessionClosureCapability, SessionPlacement, SessionStateRetention,
        StreamingSessionCoordinator, StreamingSessionProgramDescriptor,
        StreamingSessionProgramFactory, ValidatedStreamingSessionProgramConfig,
    },
    source::{
        AcquiredPartition, AcquisitionBudget, PartitionAccessKind, PreparedStreamingDatasetSource,
        SourcePartitionContent, StreamingDatasetSource, StreamingDatasetSourceFactory,
        StreamingResumeGranularity, StreamingSourceDescriptor, StreamingSourceMode,
        StreamingSourceOrdering, StreamingSourcePlacement, StreamingSourcePrepareContext,
        StreamingSourceRetention, StreamingStopReceiver, ValidatedStreamingSourceConfig,
        streaming_stop_channel,
    },
    unit::{SourcePosition, StateBudgetFailureCode},
};
use async_trait::async_trait;
use bytes::Bytes;

fn assert_factory<T: Debug + Send + Sync + ?Sized>() {}
fn assert_validated<T: Debug + Send + Sync + ?Sized>() {}
fn assert_clone<T: Clone>() {}

#[test]
fn factories_and_validated_configs_have_host_safe_bounds() {
    assert_factory::<dyn StreamingDatasetSourceFactory>();
    assert_factory::<dyn StreamingDatasetFormatFactory>();
    assert_factory::<dyn StreamingSessionProgramFactory>();
    assert_factory::<dyn StreamingActionSinkFactory>();
    assert_factory::<dyn StreamingCheckpointBackendFactory>();

    assert_validated::<dyn ValidatedStreamingSourceConfig>();
    assert_validated::<dyn ValidatedStreamingFormatConfig>();
    assert_validated::<dyn ValidatedStreamingSessionProgramConfig>();
    assert_validated::<dyn ValidatedStreamingActionSinkConfig>();
    assert_validated::<dyn ValidatedCheckpointBackendConfig>();
}

#[derive(Debug)]
struct CaptureReporter {
    calls: Rc<Cell<usize>>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for CaptureReporter {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        assert_eq!(issue.class(), StreamingIssueClass::Retryable);
        assert_eq!(issue.failure().stage(), StreamingFailureStage::Acquisition);
        assert_eq!(issue.failure().code(), "read");
        self.calls.set(self.calls.get() + 1);
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

#[tokio::test]
async fn cloned_issue_reporter_forwards_closed_typed_facts_to_the_host() {
    let calls = Rc::new(Cell::new(0));
    let reporter = StreamingIssueReporterHandle::new(CaptureReporter {
        calls: Rc::clone(&calls),
    });
    let issue = OrdinaryStreamingIssue::new(
        LogicalReplayRunId::from_bytes([1; 32]),
        StreamingIssueScope::Partition {
            partition: ImmutableObjectIdentity::from_bytes([2; 32]),
        },
        StreamingIssueClass::Retryable,
        ContentDigest::from_bytes([3; 32]),
        StreamingIssueOrderKey::Source(SourcePosition::new(4)),
        OrdinaryStreamingFailure::Source(StreamSourceError::acquisition(
            AcquisitionFailureCode::Read,
        )),
    )
    .unwrap_or_else(|error| panic!("valid typed issue was rejected: {error}"));

    let status = reporter
        .clone()
        .report(issue)
        .await
        .unwrap_or_else(|error| panic!("host reporter failed: {error}"));

    assert_eq!(status, StreamingIssueReportStatus::Accepted);
    assert_eq!(calls.get(), 1);
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn runtime_contracts_are_object_safe(
    participant: Box<dyn StreamingCheckpointParticipant>,
    issue_reporter: Box<dyn StreamingIssueReporter>,
    backend: Box<dyn StreamingCheckpointBackend>,
    reader: Box<dyn LeasedGenerationReader>,
    transaction: Box<dyn StreamingGenerationTransaction>,
    prepared_source: Box<dyn PreparedStreamingDatasetSource>,
    source: Box<dyn StreamingDatasetSource>,
    content: Box<dyn SourcePartitionContent>,
    format: Box<dyn StreamingDatasetFormat>,
    decoder: Box<dyn StreamingPartitionDecoder>,
    session: Box<dyn StreamingSessionCoordinator>,
    actions: Box<dyn DatasetActionSink>,
    submitter: Box<dyn StreamingActionSubmitter>,
    driver: Box<dyn StreamingActionDriver>,
) {
    let _ = (
        participant,
        issue_reporter,
        backend,
        reader,
        transaction,
        prepared_source,
        source,
        content,
        format,
        decoder,
        session,
        actions,
        submitter,
        driver,
    );
}

#[allow(dead_code)]
fn action_binding_is_split(binding: PreparedStreamingActionBinding) {
    let _: Box<dyn StreamingActionSubmitter> = binding.submitter;
    let _: Box<dyn StreamingActionDriver> = binding.driver;
    let _: StreamingActionDriverControl = binding.control;
}

#[test]
fn driver_control_is_a_cheaply_cloneable_concrete_handle() {
    assert_clone::<StreamingActionDriverControl>();
    assert_clone::<StreamingIssueReporterHandle>();
    assert!(
        std::mem::size_of::<StreamingActionDriverControl>()
            <= 2 * std::mem::size_of::<std::rc::Rc<()>>()
    );
}

#[test]
fn failure_stages_and_codes_do_not_collapse() {
    let acquisition = StreamSourceError::acquisition(AcquisitionFailureCode::Read);
    let decode = StreamFormatError::decode(DecodeFailureCode::Syntax);
    let late = StreamFormatError::ordering(OrderingFailureCode::LateData);
    let budget = StreamFormatError::state_budget(StateBudgetFailureCode::ByteCapacity);
    let placement = aiperf_runtime::streaming::action::ActionExecutionError::placement(
        PlacementFailureCode::RouteUnavailable,
    );

    assert_eq!(
        (acquisition.stage(), acquisition.code()),
        (StreamingFailureStage::Acquisition, "read")
    );
    assert_eq!(
        (decode.stage(), decode.code()),
        (StreamingFailureStage::Decode, "syntax")
    );
    assert_eq!(
        (late.stage(), late.code()),
        (StreamingFailureStage::Ordering, "late_data")
    );
    assert_eq!(
        (budget.stage(), budget.code()),
        (StreamingFailureStage::StateBudget, "byte_capacity")
    );
    assert_eq!(
        (placement.stage(), placement.code()),
        (StreamingFailureStage::Placement, "route_unavailable")
    );
}

#[allow(dead_code)]
fn prepare_contexts_receive_the_host_issue_reporter(
    source: aiperf_runtime::streaming::source::StreamingSourcePrepareContext,
    format: aiperf_runtime::streaming::format::StreamingFormatPrepareContext,
) {
    let _: StreamingIssueReporterHandle = source.issue_reporter;
    let _: StreamingIssueReporterHandle = format.issue_reporter;
}

#[test]
fn descriptors_serialize_complete_agreement_facts() {
    let source = StreamingSourceDescriptor {
        id: "source",
        description: "source",
        mode: StreamingSourceMode::Follow,
        access: &[PartitionAccessKind::Sequential],
        ordering: StreamingSourceOrdering::EventTime,
        resume: &[StreamingResumeGranularity::Byte],
        has_event_time: true,
        has_stable_record_ids: true,
        retention: StreamingSourceRetention::ResumeRootReachability,
        placement: StreamingSourcePlacement::ImmutablePartitionAssignment,
        supports_virtual_clock: true,
    };
    let format = StreamingFormatDescriptor {
        id: "format",
        description: "format",
        semantic_digest: ContentDigest::from_bytes([1; 32]),
        media_types: &["application/jsonl"],
        input_schemas: &["record/v1"],
        required_access: PartitionAccessKind::Sequential,
        projection: FormatProjection::BoundedFields,
        output_schema: "fragment/v1",
        has_event_time: true,
        has_stable_record_ids: true,
        retention: FormatStateRetention::BoundedMemory,
        supports_virtual_clock: true,
    };
    let session = StreamingSessionProgramDescriptor {
        id: "session",
        description: "session",
        fragment_input_schemas: &["fragment/v1"],
        action_schemas: &["action/v1"],
        closure: &[SessionClosureCapability::HardWatermark],
        retention: SessionStateRetention::BoundedMemory,
        placement: SessionPlacement::RoutedByStableSession,
        supports_virtual_clock: true,
    };
    let action = StreamingActionSinkDescriptor {
        id: "action",
        description: "action",
        accepted_schemas: &["action/v1"],
        transport_ids: &["http"],
        endpoint_kinds: &["chat"],
        retention: ActionResultRetention::StreamingTerminal,
        placement: ActionPlacement::WorkerLocal,
        supports_virtual_clock: true,
    };
    let checkpoint = StreamingCheckpointBackendDescriptor {
        id: "checkpoint",
        description: "checkpoint",
        is_durable: true,
        has_leased_readers: true,
        has_atomic_generations: true,
        has_result_segments: true,
        protects_sensitive_state: true,
        retention: CheckpointRetention::GenerationReachability,
        placement: CheckpointBackendPlacement::SharedAcrossCells,
        supports_virtual_clock: true,
    };

    for (value, required) in [
        (
            serde_json::to_value(source).unwrap_or_else(|error| panic!("source: {error}")),
            &[
                "mode",
                "access",
                "ordering",
                "resume",
                "retention",
                "placement",
                "supports_virtual_clock",
            ][..],
        ),
        (
            serde_json::to_value(format).unwrap_or_else(|error| panic!("format: {error}")),
            &[
                "media_types",
                "required_access",
                "projection",
                "output_schema",
                "retention",
                "supports_virtual_clock",
            ][..],
        ),
        (
            serde_json::to_value(session).unwrap_or_else(|error| panic!("session: {error}")),
            &[
                "fragment_input_schemas",
                "action_schemas",
                "closure",
                "retention",
                "placement",
                "supports_virtual_clock",
            ][..],
        ),
        (
            serde_json::to_value(action).unwrap_or_else(|error| panic!("action: {error}")),
            &[
                "accepted_schemas",
                "transport_ids",
                "endpoint_kinds",
                "retention",
                "placement",
                "supports_virtual_clock",
            ][..],
        ),
        (
            serde_json::to_value(checkpoint).unwrap_or_else(|error| panic!("checkpoint: {error}")),
            &[
                "is_durable",
                "has_leased_readers",
                "has_atomic_generations",
                "has_result_segments",
                "protects_sensitive_state",
                "retention",
                "placement",
                "supports_virtual_clock",
            ][..],
        ),
    ] {
        let object = value
            .as_object()
            .unwrap_or_else(|| panic!("descriptor was not an object"));
        for field in required {
            assert!(
                object.contains_key(*field),
                "missing descriptor fact {field}"
            );
        }
    }
}

#[tokio::test]
async fn source_stop_wakes_a_pending_receiver_with_an_unforgeable_outcome() {
    let (control, mut receiver) = streaming_stop_channel();
    tokio::select! {
        result = receiver.stopped() => panic!("stop completed before control request: {result:?}"),
        () = tokio::task::yield_now() => {}
    }

    control.stop();
    let error = receiver
        .stopped()
        .await
        .expect_err("controlled stop must be distinguishable from source completion");
    assert!(error.is_stopped());
    assert_eq!(error.stage(), StreamingFailureStage::Source);
    assert_eq!(error.code(), "stopped");
}

#[tokio::test]
async fn action_cancellation_wakes_a_pending_receiver() {
    let (control, mut receiver) = action_execution_control();
    tokio::select! {
        () = receiver.cancelled() => panic!("cancel completed before control request"),
        () = tokio::task::yield_now() => {}
    }
    control.cancel();
    receiver.cancelled().await;
    assert!(receiver.is_cancelled());
}

#[derive(Debug)]
struct CountingDriverControl {
    stop_calls: Rc<Cell<usize>>,
    pending_calls: Rc<Cell<usize>>,
    inflight_calls: Rc<Cell<usize>>,
}

#[async_trait(?Send)]
impl StreamingActionDriverControlOps for CountingDriverControl {
    fn stop_issuing(&self) {
        self.stop_calls.set(self.stop_calls.get() + 1);
    }

    fn cancel_pending(&self) {
        self.pending_calls.set(self.pending_calls.get() + 1);
    }

    async fn cancel_inflight(
        &self,
    ) -> Result<ActionCancelReceipt, aiperf_runtime::streaming::action::ActionExecutionError> {
        self.inflight_calls.set(self.inflight_calls.get() + 1);
        Ok(ActionCancelReceipt {
            cancelled: 7,
            digest: ContentDigest::from_bytes([7; 32]),
        })
    }
}

#[tokio::test]
async fn cloned_driver_control_delegates_without_borrowing_the_driver() {
    let stop_calls = Rc::new(Cell::new(0));
    let pending_calls = Rc::new(Cell::new(0));
    let inflight_calls = Rc::new(Cell::new(0));
    let control = StreamingActionDriverControl::new(CountingDriverControl {
        stop_calls: Rc::clone(&stop_calls),
        pending_calls: Rc::clone(&pending_calls),
        inflight_calls: Rc::clone(&inflight_calls),
    });

    control.clone().stop_issuing();
    control.clone().cancel_pending();
    let receipt = control
        .cancel_inflight()
        .await
        .unwrap_or_else(|error| panic!("cancel failed: {error}"));

    assert_eq!(
        (stop_calls.get(), pending_calls.get(), inflight_calls.get()),
        (1, 1, 1)
    );
    assert_eq!(receipt.cancelled, 7);
}

#[tokio::test]
async fn acquired_partition_exposes_bounded_sequential_access_and_holds_its_budget() {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 6,
    })
    .unwrap_or_else(|error| panic!("budget: {error}"));
    let lease = budget
        .acquire(1, 6)
        .await
        .unwrap_or_else(|error| panic!("lease: {error}"));
    let mut partition = AcquiredPartition::new(
        SourcePosition::new(9),
        ImmutableObjectIdentity::from_bytes([9; 32]),
        1,
        Bytes::from_static(b"abcdef"),
        lease,
    )
    .unwrap_or_else(|error| panic!("partition: {error}"));

    assert_eq!(partition.position(), SourcePosition::new(9));
    assert_eq!(
        partition.next_chunk(NonZeroUsize::new(2).unwrap()),
        Some(&b"bc"[..])
    );
    assert_eq!(
        partition.next_chunk(NonZeroUsize::new(8).unwrap()),
        Some(&b"def"[..])
    );
    assert_eq!(partition.next_chunk(NonZeroUsize::new(1).unwrap()), None);
    assert_eq!(
        (budget.snapshot().used_items, budget.snapshot().used_bytes),
        (1, 6)
    );
    drop(partition);
    assert_eq!(
        (budget.snapshot().used_items, budget.snapshot().used_bytes),
        (0, 0)
    );
}

#[tokio::test]
async fn decoder_resume_state_is_move_only_and_holds_exact_budget() {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 3,
    })
    .unwrap_or_else(|error| panic!("budget: {error}"));
    let lease = budget
        .acquire(1, 3)
        .await
        .unwrap_or_else(|error| panic!("lease: {error}"));
    let state = DecoderResumeState::new(Bytes::from_static(b"xyz"), lease)
        .unwrap_or_else(|error| panic!("resume: {error}"));
    assert_eq!(state.as_bytes(), b"xyz");
    assert_eq!(
        (budget.snapshot().used_items, budget.snapshot().used_bytes),
        (1, 3)
    );
    drop(state);
    assert_eq!(
        (budget.snapshot().used_items, budget.snapshot().used_bytes),
        (0, 0)
    );
}

#[tokio::test]
async fn decoder_resume_lease_mismatch_is_a_decode_invariant() {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 2,
    })
    .unwrap_or_else(|error| panic!("budget: {error}"));
    let lease = budget
        .acquire(1, 1)
        .await
        .unwrap_or_else(|error| panic!("lease: {error}"));
    let error = DecoderResumeState::new(Bytes::from_static(b"xy"), lease)
        .expect_err("undercharged resume state must fail");
    assert_eq!(error.stage(), StreamingFailureStage::Decode);
    assert_eq!(error.code(), "budget_invariant");
}

#[tokio::test]
async fn acquired_partition_lease_mismatch_is_an_acquisition_invariant() {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 2,
    })
    .unwrap_or_else(|error| panic!("budget: {error}"));
    let lease = budget
        .acquire(1, 1)
        .await
        .unwrap_or_else(|error| panic!("lease: {error}"));
    let error = AcquiredPartition::new(
        SourcePosition::new(1),
        ImmutableObjectIdentity::from_bytes([1; 32]),
        0,
        Bytes::from_static(b"xy"),
        lease,
    )
    .expect_err("undercharged acquired content must fail");
    assert_eq!(error.stage(), StreamingFailureStage::Acquisition);
    assert_eq!(error.code(), "budget_invariant");
}

#[test]
fn checkpoint_backend_budget_codes_preserve_nested_classification() {
    for (nested, expected) in [
        (
            CheckpointBackendBudgetFailureCode::ItemCapacity,
            "backend_item_capacity",
        ),
        (
            CheckpointBackendBudgetFailureCode::ByteCapacity,
            "backend_byte_capacity",
        ),
        (CheckpointBackendBudgetFailureCode::Closed, "backend_closed"),
        (
            CheckpointBackendBudgetFailureCode::Unrepresentable,
            "backend_unrepresentable",
        ),
    ] {
        let error = CheckpointError::BackendBudget {
            budget: CheckpointBackendBudgetKind::Storage,
            code: nested,
        };
        assert_eq!(error.stage(), StreamingFailureStage::Checkpoint);
        assert_eq!(error.code(), expected);
    }
}

#[tokio::test]
async fn action_payload_lease_mismatch_is_an_invariant_not_capacity() {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 2,
    })
    .unwrap_or_else(|error| panic!("budget: {error}"));
    let lease = budget
        .acquire(1, 1)
        .await
        .unwrap_or_else(|error| panic!("lease: {error}"));
    let error = BudgetedActionUpdate::new(Bytes::from_static(b"ab"), lease)
        .expect_err("undercharged payload must fail");
    assert_eq!(error.stage(), StreamingFailureStage::Dispatch);
    assert_eq!(error.code(), ActionFailureCode::BudgetInvariant.code());
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TestSourceConfig {
    marker: u8,
}

#[derive(Debug)]
struct TestSourceFactory;

static TEST_SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: "test",
    description: "test source factory",
    mode: StreamingSourceMode::Finite,
    access: &[PartitionAccessKind::Sequential],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[StreamingResumeGranularity::Byte],
    has_event_time: false,
    has_stable_record_ids: false,
    retention: StreamingSourceRetention::BoundedMemory,
    placement: StreamingSourcePlacement::ControllerOnly,
    supports_virtual_clock: true,
};

impl StreamingDatasetSourceFactory for TestSourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        &TEST_SOURCE_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        let config: TestSourceConfig = serde_json::from_str(authored.get()).map_err(|_| {
            StreamSourceError::source(
                aiperf_runtime::streaming::failure::SourceFailureCode::Snapshot,
            )
        })?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        _context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        let config = config
            .into_any()
            .downcast::<TestSourceConfig>()
            .map_err(|_| {
                StreamSourceError::source(
                    aiperf_runtime::streaming::failure::SourceFailureCode::Snapshot,
                )
            })?;
        Ok(Box::new(TestPreparedSource {
            marker: config.marker,
        }))
    }
}

struct TestPreparedSource {
    marker: u8,
}

#[async_trait(?Send)]
impl PreparedStreamingDatasetSource for TestPreparedSource {
    async fn open(
        self: Box<Self>,
        _stop: StreamingStopReceiver,
    ) -> Result<aiperf_runtime::streaming::source::OpenedStreamingDatasetSource, StreamSourceError>
    {
        let code = if self.marker == 7 {
            aiperf_runtime::streaming::failure::SourceFailureCode::SourceUnavailable
        } else {
            aiperf_runtime::streaming::failure::SourceFailureCode::Snapshot
        };
        Err(StreamSourceError::source(code))
    }
}

#[tokio::test]
async fn source_factory_strictly_validates_downcasts_and_prepares_real_behavior() {
    let factory = TestSourceFactory;
    let authored = serde_json::value::RawValue::from_string(r#"{"marker":7}"#.to_owned())
        .unwrap_or_else(|error| panic!("raw config: {error}"));
    let config = factory
        .validate(&authored)
        .unwrap_or_else(|error| panic!("validate: {error}"));
    assert_eq!(
        ValidatedStreamingSourceConfig::as_any(config.as_ref())
            .downcast_ref::<TestSourceConfig>()
            .map(|config| config.marker),
        Some(7)
    );
    let calls = Rc::new(Cell::new(0));
    let reporter = StreamingIssueReporterHandle::new(CaptureReporter {
        calls: Rc::clone(&calls),
    });
    let acquisition_budget = AcquisitionBudget::new(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: 1,
        })
        .unwrap_or_else(|error| panic!("budget: {error}")),
    );
    let prepared = factory
        .prepare(
            config,
            &StreamingSourcePrepareContext {
                acquisition_budget,
                issue_reporter: reporter,
            },
        )
        .unwrap_or_else(|error| panic!("prepare: {error}"));
    let (_, stop) = streaming_stop_channel();
    let error = match prepared.open(stop).await {
        Ok(_) => panic!("prepared marker should select failure"),
        Err(error) => error,
    };
    assert_eq!(error.code(), "source_unavailable");

    let unknown =
        serde_json::value::RawValue::from_string(r#"{"marker":7,"unexpected":true}"#.to_owned())
            .unwrap_or_else(|error| panic!("raw config: {error}"));
    assert_eq!(
        factory
            .validate(&unknown)
            .expect_err("unknown field must fail")
            .code(),
        "snapshot"
    );
}
