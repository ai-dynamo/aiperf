// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{cell::RefCell, fmt::Debug, rc::Rc};

use aiperf_runtime::streaming::{
    action::{
        PreparedStreamingActionBinding, StreamingActionDriver, StreamingActionDriverControl,
        StreamingActionSinkFactory, StreamingActionSubmitter, ValidatedStreamingActionSinkConfig,
    },
    checkpoint::StreamingCheckpointParticipant,
    checkpoint_backend::{
        LeasedGenerationReader, StreamingCheckpointBackend, StreamingCheckpointBackendFactory,
        StreamingGenerationTransaction, ValidatedCheckpointBackendConfig,
    },
    failure::{
        AcquisitionFailureCode, DecodeFailureCode, OrderingFailureCode, PlacementFailureCode,
        StableStreamingFailure, StreamFormatError, StreamSourceError, StreamingFailureStage,
        StreamingIssue, StreamingIssueReporter, StreamingIssueReporterOps,
    },
    format::{
        StreamingDatasetFormat, StreamingDatasetFormatFactory, StreamingPartitionDecoder,
        ValidatedStreamingFormatConfig,
    },
    session::{
        DatasetActionSink, StreamingSessionCoordinator, StreamingSessionProgramFactory,
        ValidatedStreamingSessionProgramConfig,
    },
    source::{
        PreparedStreamingDatasetSource, SourcePartitionContent, StreamingDatasetSource,
        StreamingDatasetSourceFactory, ValidatedStreamingSourceConfig,
    },
    unit::StateBudgetFailureCode,
};

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

#[test]
fn cloned_issue_reporter_forwards_typed_issues_to_the_host() {
    #[derive(Clone)]
    struct Capture(Rc<RefCell<Vec<StreamingIssue>>>);

    impl StreamingIssueReporterOps for Capture {
        fn report(&self, issue: StreamingIssue) {
            self.0.borrow_mut().push(issue);
        }
    }

    let captured = Rc::new(RefCell::new(Vec::new()));
    let reporter = StreamingIssueReporter::new(Capture(Rc::clone(&captured)));
    let stopped = StreamSourceError::stopped("source stop requested");
    reporter
        .clone()
        .report(StreamingIssue::from_failure(&stopped));

    assert_eq!(
        captured.borrow().as_slice(),
        [StreamingIssue {
            stage: StreamingFailureStage::Source,
            code: "stopped",
            message: "stopped: source stop requested".to_owned(),
        }]
    );
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn runtime_contracts_are_object_safe(
    participant: Box<dyn StreamingCheckpointParticipant>,
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
    assert_clone::<StreamingIssueReporter>();
    assert!(
        std::mem::size_of::<StreamingActionDriverControl>()
            <= 2 * std::mem::size_of::<std::rc::Rc<()>>()
    );
}

#[test]
fn failure_stages_and_codes_do_not_collapse() {
    let acquisition = StreamSourceError::acquisition(AcquisitionFailureCode::Read, "read failed");
    let decode = StreamFormatError::decode(DecodeFailureCode::Syntax, "invalid record");
    let late = StreamFormatError::ordering(OrderingFailureCode::LateData, "late record");
    let budget = StreamFormatError::state_budget(
        StateBudgetFailureCode::ByteCapacity,
        "batch budget exhausted",
    );
    let placement = aiperf_runtime::streaming::action::ActionExecutionError::placement(
        PlacementFailureCode::RouteUnavailable,
        "route missing",
    );
    let stopped = StreamSourceError::stopped("source stop requested");

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
    assert_eq!(
        (stopped.stage(), stopped.code()),
        (StreamingFailureStage::Source, "stopped")
    );
}

#[allow(dead_code)]
fn prepare_contexts_receive_the_host_issue_reporter(
    source: aiperf_runtime::streaming::source::StreamingSourcePrepareContext,
    format: aiperf_runtime::streaming::format::StreamingFormatPrepareContext,
) {
    let _: StreamingIssueReporter = source.issue_reporter;
    let _: StreamingIssueReporter = format.issue_reporter;
}
