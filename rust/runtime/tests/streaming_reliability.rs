// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU64;

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
        CheckpointGeneration, DecodeHorizon, DiscoveryHorizon, EventTimeWatermark,
        OrderedActionHorizon, StreamRunIdentity, TerminalActionHorizon,
    },
    failure::{
        AcquisitionFailureCode, ActionExecutionError, ActionFailureCode, CheckpointAttemptError,
        CheckpointAttemptFailureCode, DecodeFailureCode, OrdinaryStreamingFailure,
        ResultExportError, ResultExportFailureCode, SessionCoordinatorError, SessionFailureCode,
        StreamFormatError, StreamSourceError, StreamingFailureStage,
    },
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId,
        SessionCausalFrontier, StableActionId, StableRecordId, StableSessionKey,
    },
    reliability::{
        ActionFailureDisposition, BudgetOwnedStreamingIssueReporter, HandledIssueCut,
        IssueSequenceUpdate, OrdinaryStreamingIssue, PreparedActionFailureIdentity,
        PreparedStreamingIssuePolicy, StreamingInputDomainIdentity, StreamingIssueClass,
        StreamingIssueComponentId, StreamingIssueDisposition, StreamingIssueOrderKey,
        StreamingIssueReporter, StreamingIssueScope, StreamingIssueScopeKind,
        StreamingIssueThresholdRule, StreamingIssueValidationError,
    },
    results::{
        BudgetedResultDescriptor, CellId, ResultProjectionId, ResultSchemaVersion,
        ResultSegmentDescriptor, WorkerId,
    },
    unit::{EventTimeUtc, SourcePosition, StateBudgetFailureCode},
};

fn component(value: &str) -> StreamingIssueComponentId {
    StreamingIssueComponentId::new(value)
        .unwrap_or_else(|error| panic!("valid component ID {value:?}: {error}"))
}

fn run(value: u8) -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([value; 32]))
}

fn domain(stream: u8, source: u8) -> StreamingInputDomainIdentity {
    StreamingInputDomainIdentity::new(
        ContentDigest::from_bytes([stream; 32]),
        ImmutableObjectIdentity::from_bytes([source; 32]),
    )
}

fn record_issue(code: DecodeFailureCode) -> OrdinaryStreamingIssue {
    OrdinaryStreamingIssue::record(
        run(0x11),
        domain(0x21, 0x20),
        StableRecordId::from_bytes([0x22; 32]),
        StreamingIssueClass::Permanent,
        ContentDigest::from_bytes([0x33; 32]),
        SourcePosition::new(7),
        0,
        ContentDigest::from_bytes([0x44; 32]),
        OrdinaryStreamingFailure::Format(StreamFormatError::decode(code)),
    )
    .unwrap_or_else(|error| panic!("valid record issue: {error}"))
}

fn wildcard_record_rule() -> StreamingIssueThresholdRule {
    StreamingIssueThresholdRule::new(
        component("record_default"),
        StreamingIssueScopeKind::Record,
        StreamingIssueClass::Permanent,
        None,
        0,
        StreamingIssueDisposition::Quarantine,
        NonZeroU64::new(3),
    )
    .unwrap_or_else(|error| panic!("valid wildcard rule: {error}"))
}

fn exact_record_rule() -> StreamingIssueThresholdRule {
    StreamingIssueThresholdRule::new(
        component("syntax_exact"),
        StreamingIssueScopeKind::Record,
        StreamingIssueClass::Permanent,
        Some(component("syntax")),
        1,
        StreamingIssueDisposition::Quarantine,
        None,
    )
    .unwrap_or_else(|error| panic!("valid exact rule: {error}"))
}

fn budget(items: usize, bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: items,
        max_bytes: bytes,
    })
    .unwrap_or_else(|error| panic!("valid reliability budget: {error}"))
}

fn record_policy() -> PreparedStreamingIssuePolicy {
    PreparedStreamingIssuePolicy::new(vec![exact_record_rule(), wildcard_record_rule()])
        .unwrap_or_else(|error| panic!("valid record policy: {error}"))
}

fn barrier(run: StreamRunIdentity, epoch: u64) -> CheckpointBarrier {
    CheckpointBarrier {
        run,
        epoch: CheckpointEpoch::new(epoch),
        cut: CheckpointCut {
            discovered: DiscoveryHorizon::new(SourcePosition::new(20)),
            acquired: AcquisitionHorizon::new(SourcePosition::new(20)),
            decoded: DecodeHorizon::new(SourcePosition::new(20)),
            ordered: OrderedActionHorizon::new(GlobalSequence::new(20)),
            admitted: AdmissionHorizon::new(GlobalSequence::new(20)),
            terminal: TerminalActionHorizon::new(GlobalSequence::new(20)),
            event_watermark: EventTimeWatermark::Hard {
                through: EventTimeUtc::new(20)
                    .unwrap_or_else(|error| panic!("valid event time: {error}")),
            },
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(20),
                event_time: Some(
                    EventTimeUtc::new(20)
                        .unwrap_or_else(|error| panic!("valid event time: {error}")),
                ),
                digest: ContentDigest::from_bytes([0x71; 32]),
            },
        },
        plan_digest: ContentDigest::from_bytes([0x72; 32]),
    }
}

#[test]
fn component_ids_are_closed_ascii_and_serde_checked() {
    assert_eq!(component("jsonl_2").as_str(), "jsonl_2");
    assert!(StreamingIssueComponentId::new("").is_err());
    assert!(StreamingIssueComponentId::new("Upper").is_err());
    assert!(StreamingIssueComponentId::new("contains-dash").is_err());
    assert!(StreamingIssueComponentId::new("a".repeat(129)).is_err());
    assert!(serde_json::from_str::<StreamingIssueComponentId>(r#""bad-code""#).is_err());
}

#[test]
fn policy_matching_is_order_invariant_exact_before_wildcard_and_unambiguous() {
    let forward =
        PreparedStreamingIssuePolicy::new(vec![exact_record_rule(), wildcard_record_rule()])
            .unwrap_or_else(|error| panic!("valid forward policy: {error}"));
    let reversed =
        PreparedStreamingIssuePolicy::new(vec![wildcard_record_rule(), exact_record_rule()])
            .unwrap_or_else(|error| panic!("valid reversed policy: {error}"));

    assert_eq!(forward.digest(), reversed.digest());
    assert_eq!(
        forward
            .rule_for(&record_issue(DecodeFailureCode::Syntax))
            .unwrap_or_else(|error| panic!("exact match: {error}"))
            .rule_id()
            .as_str(),
        "syntax_exact"
    );
    assert_eq!(
        forward
            .rule_for(&record_issue(DecodeFailureCode::Schema))
            .unwrap_or_else(|error| panic!("wildcard match: {error}"))
            .rule_id()
            .as_str(),
        "record_default"
    );

    assert!(
        PreparedStreamingIssuePolicy::new(vec![
            wildcard_record_rule(),
            exact_record_rule(),
            exact_record_rule(),
        ])
        .is_err()
    );
    assert!(
        PreparedStreamingIssuePolicy::new(vec![wildcard_record_rule(), wildcard_record_rule(),])
            .is_err()
    );
    assert!(PreparedStreamingIssuePolicy::new(vec![exact_record_rule()]).is_err());
}

#[test]
fn ordinary_receipt_identity_matches_the_v2_golden() {
    assert_eq!(
        record_issue(DecodeFailureCode::Syntax).issue_id(),
        ContentDigest::from_bytes([
            0x92, 0xe6, 0x8d, 0xa0, 0xea, 0xe7, 0xdc, 0x5a, 0xcf, 0x38, 0xdb, 0x5f, 0x66, 0xee,
            0xb0, 0xf2, 0x21, 0x4c, 0xbe, 0x35, 0x8f, 0xdb, 0xfc, 0x43, 0xc4, 0xc0, 0xdc, 0xdd,
            0x59, 0x89, 0x2d, 0xb6,
        ])
    );
}

#[test]
fn record_and_action_constructors_derive_their_exact_order_domains() {
    let input_domain = domain(1, 2);
    let record = OrdinaryStreamingIssue::record(
        run(3),
        input_domain.clone(),
        StableRecordId::from_bytes([4; 32]),
        StreamingIssueClass::Retryable,
        ContentDigest::from_bytes([5; 32]),
        SourcePosition::new(6),
        7,
        ContentDigest::from_bytes([8; 32]),
        OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
    )
    .unwrap_or_else(|error| panic!("valid record issue: {error}"));
    assert_eq!(record.scope().kind(), StreamingIssueScopeKind::Record);
    assert_eq!(record.order().input_domain.as_ref(), Some(&input_domain));
    assert_eq!(record.order().source_position, Some(SourcePosition::new(6)));
    assert_eq!(record.order().global_sequence, None);
    assert_eq!(record.order().retry_ordinal, 7);

    let action = OrdinaryStreamingIssue::action(
        run(3),
        StableActionId::from_bytes([9; 32]),
        StreamingIssueClass::Retryable,
        ContentDigest::from_bytes([10; 32]),
        GlobalSequence::new(11),
        12,
        ContentDigest::from_bytes([13; 32]),
        OrdinaryStreamingFailure::Action(ActionExecutionError::action(ActionFailureCode::Endpoint)),
    )
    .unwrap_or_else(|error| panic!("valid action issue: {error}"));
    assert_eq!(action.scope().kind(), StreamingIssueScopeKind::Action);
    assert_eq!(action.order().input_domain, None);
    assert_eq!(action.order().source_position, None);
    assert_eq!(
        action.order().global_sequence,
        Some(GlobalSequence::new(11))
    );
}

#[test]
fn action_scope_rejects_decode_failure_before_identity() {
    let rejected = OrdinaryStreamingIssue::action(
        run(3),
        StableActionId::from_bytes([9; 32]),
        StreamingIssueClass::Permanent,
        ContentDigest::from_bytes([10; 32]),
        GlobalSequence::new(11),
        12,
        ContentDigest::from_bytes([13; 32]),
        OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
    );

    assert!(rejected.is_err());
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestScopeFamily {
    Partition,
    Record,
    Session,
    Action,
    CheckpointAttempt,
    Export,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestFailureFamily {
    Source,
    Format,
    Session,
    Action,
    CheckpointAttempt,
    Export,
}

fn representative_failure(family: TestFailureFamily) -> OrdinaryStreamingFailure {
    match family {
        TestFailureFamily::Source => OrdinaryStreamingFailure::Source(
            StreamSourceError::acquisition(AcquisitionFailureCode::Read),
        ),
        TestFailureFamily::Format => {
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax))
        }
        TestFailureFamily::Session => OrdinaryStreamingFailure::Session(
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor),
        ),
        TestFailureFamily::Action => OrdinaryStreamingFailure::Action(
            ActionExecutionError::action(ActionFailureCode::Endpoint),
        ),
        TestFailureFamily::CheckpointAttempt => OrdinaryStreamingFailure::CheckpointAttempt(
            CheckpointAttemptError::failure(CheckpointAttemptFailureCode::Io),
        ),
        TestFailureFamily::Export => OrdinaryStreamingFailure::Export(ResultExportError::failure(
            ResultExportFailureCode::Io,
        )),
    }
}

fn representative_scope_and_order(
    family: TestScopeFamily,
) -> (StreamingIssueScope, StreamingIssueOrderKey) {
    let input_domain = domain(0x71, 0x72);
    let tiebreaker = ContentDigest::from_bytes([0x73; 32]);
    match family {
        TestScopeFamily::Partition => (
            StreamingIssueScope::Partition {
                input_domain: input_domain.clone(),
                object: ImmutableObjectIdentity::from_bytes([0x74; 32]),
            },
            StreamingIssueOrderKey::input(input_domain, SourcePosition::new(4), 2, tiebreaker),
        ),
        TestScopeFamily::Record => (
            StreamingIssueScope::Record {
                input_domain: input_domain.clone(),
                record_id: StableRecordId::from_bytes([0x75; 32]),
            },
            StreamingIssueOrderKey::input(input_domain, SourcePosition::new(4), 2, tiebreaker),
        ),
        TestScopeFamily::Session => (
            StreamingIssueScope::Session {
                input_domain: input_domain.clone(),
                session_key: StableSessionKey::from_bytes([0x76; 32]),
            },
            StreamingIssueOrderKey::input(input_domain, SourcePosition::new(4), 2, tiebreaker),
        ),
        TestScopeFamily::Action => (
            StreamingIssueScope::Action {
                action_id: StableActionId::from_bytes([0x77; 32]),
            },
            StreamingIssueOrderKey::action(GlobalSequence::new(4), 2, tiebreaker),
        ),
        TestScopeFamily::CheckpointAttempt => (
            StreamingIssueScope::CheckpointAttempt {
                generation: CheckpointEpoch::new(4),
                attempt_ordinal: 2,
            },
            StreamingIssueOrderKey::run(2, tiebreaker),
        ),
        TestScopeFamily::Export => (
            StreamingIssueScope::Export {
                exporter_id: component("jsonl"),
                generation: CheckpointGeneration::new(
                    CheckpointEpoch::new(4),
                    ContentDigest::from_bytes([0x78; 32]),
                ),
            },
            StreamingIssueOrderKey::run(2, tiebreaker),
        ),
    }
}

#[test]
fn every_ordinary_scope_accepts_only_its_typed_failure_family() {
    let scopes = [
        TestScopeFamily::Partition,
        TestScopeFamily::Record,
        TestScopeFamily::Session,
        TestScopeFamily::Action,
        TestScopeFamily::CheckpointAttempt,
        TestScopeFamily::Export,
    ];
    let failures = [
        TestFailureFamily::Source,
        TestFailureFamily::Format,
        TestFailureFamily::Session,
        TestFailureFamily::Action,
        TestFailureFamily::CheckpointAttempt,
        TestFailureFamily::Export,
    ];

    for scope_family in scopes {
        for failure_family in failures {
            let (scope, order) = representative_scope_and_order(scope_family);
            let result = OrdinaryStreamingIssue::new(
                run(0x70),
                scope,
                StreamingIssueClass::Retryable,
                ContentDigest::from_bytes([0x79; 32]),
                order,
                representative_failure(failure_family),
            );
            let is_matching_family = matches!(
                (scope_family, failure_family),
                (TestScopeFamily::Partition, TestFailureFamily::Source)
                    | (TestScopeFamily::Record, TestFailureFamily::Format)
                    | (TestScopeFamily::Session, TestFailureFamily::Session)
                    | (TestScopeFamily::Action, TestFailureFamily::Action)
                    | (
                        TestScopeFamily::CheckpointAttempt,
                        TestFailureFamily::CheckpointAttempt
                    )
                    | (TestScopeFamily::Export, TestFailureFamily::Export)
            );
            if is_matching_family {
                assert!(
                    result.is_ok(),
                    "{scope_family:?} rejected {failure_family:?}"
                );
            } else {
                assert_eq!(
                    result,
                    Err(StreamingIssueValidationError::FailureScopeMismatch),
                    "{scope_family:?} accepted {failure_family:?}"
                );
            }
        }
    }
}

#[test]
fn checkpoint_attempt_mismatch_cannot_mint_a_second_dedup_identity() {
    let issue = |run_byte, epoch, attempt_ordinal, retry_ordinal| {
        OrdinaryStreamingIssue::new(
            run(run_byte),
            StreamingIssueScope::CheckpointAttempt {
                generation: CheckpointEpoch::new(epoch),
                attempt_ordinal,
            },
            StreamingIssueClass::Retryable,
            ContentDigest::from_bytes([0x81; 32]),
            StreamingIssueOrderKey::run(retry_ordinal, ContentDigest::from_bytes([0x82; 32])),
            OrdinaryStreamingFailure::CheckpointAttempt(CheckpointAttemptError::failure(
                CheckpointAttemptFailureCode::Attempt,
            )),
        )
    };

    assert_eq!(
        issue(0x80, 5, 7, 8),
        Err(StreamingIssueValidationError::OrderScopeMismatch)
    );
    let first = issue(0x80, 5, 7, 7).unwrap_or_else(|error| panic!("valid issue: {error}"));
    let replay = issue(0x80, 5, 7, 7).unwrap_or_else(|error| panic!("valid replay: {error}"));
    let other_run = issue(0x81, 5, 7, 7).unwrap_or_else(|error| panic!("valid run: {error}"));
    let other_epoch = issue(0x80, 6, 7, 7).unwrap_or_else(|error| panic!("valid epoch: {error}"));
    assert_eq!(first.issue_id(), replay.issue_id());
    assert_ne!(first.issue_id(), other_run.issue_id());
    assert_ne!(first.issue_id(), other_epoch.issue_id());
}

#[test]
fn checkpoint_and_export_failure_codes_retain_exact_stages() {
    let checkpoint = OrdinaryStreamingFailure::CheckpointAttempt(CheckpointAttemptError::failure(
        CheckpointAttemptFailureCode::Unavailable,
    ));
    assert_eq!(checkpoint.stage(), StreamingFailureStage::Checkpoint);
    assert_eq!(checkpoint.code(), "checkpoint_unavailable");

    let checkpoint_capacity = OrdinaryStreamingFailure::CheckpointAttempt(
        CheckpointAttemptError::state_budget(StateBudgetFailureCode::ItemCapacity),
    );
    assert_eq!(
        checkpoint_capacity.stage(),
        StreamingFailureStage::StateBudget
    );
    assert_eq!(checkpoint_capacity.code(), "item_capacity");

    let export = OrdinaryStreamingFailure::Export(ResultExportError::failure(
        ResultExportFailureCode::Attempt,
    ));
    assert_eq!(export.stage(), StreamingFailureStage::Result);
    assert_eq!(export.code(), "result_export_attempt");
}

#[test]
fn persisted_scope_rejects_unknown_fields() {
    let value = serde_json::json!({
        "scope": "action",
        "action_id": vec![0; 32],
        "unexpected": true,
    });
    assert!(serde_json::from_value::<StreamingIssueScope>(value).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn ordered_reporter_is_arrival_invariant_and_replay_counts_once() {
    let input_domain = domain(0x31, 0x32);
    let make_issue = |position: u64, tie: u8| {
        OrdinaryStreamingIssue::record(
            run(0x11),
            input_domain.clone(),
            StableRecordId::from_bytes([tie; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0x41; 32]),
            SourcePosition::new(position),
            0,
            ContentDigest::from_bytes([tie; 32]),
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
        )
        .unwrap_or_else(|error| panic!("valid ordered issue: {error}"))
    };

    let mut reverse =
        BudgetOwnedStreamingIssueReporter::new(run(0x11), record_policy(), budget(64, 64 * 1024))
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
    assert_eq!(
        reverse
            .report(IssueSequenceUpdate::Issue(make_issue(9, 9)))
            .await
            .unwrap_or_else(|error| panic!("retain later issue: {error}")),
        None
    );
    assert_eq!(
        reverse
            .report(IssueSequenceUpdate::Issue(make_issue(7, 7)))
            .await
            .unwrap_or_else(|error| panic!("retain earlier issue: {error}")),
        None
    );
    reverse
        .report(IssueSequenceUpdate::NoMoreBefore {
            input_domain: input_domain.clone(),
            through: SourcePosition::new(9),
        })
        .await
        .unwrap_or_else(|error| panic!("advance reverse frontier: {error}"));

    let replay_id = make_issue(7, 7).issue_id();
    let replay = reverse
        .report(IssueSequenceUpdate::Issue(make_issue(7, 7)))
        .await
        .unwrap_or_else(|error| panic!("replay retained issue: {error}"))
        .unwrap_or_else(|| panic!("replay returns prior outcome"));
    assert_eq!(replay.issue_id(), replay_id);
    assert_eq!(reverse.summary().unwrap().total, 2);

    let mut forward =
        BudgetOwnedStreamingIssueReporter::new(run(0x11), record_policy(), budget(64, 64 * 1024))
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
    for (position, tie) in [(7, 7), (9, 9)] {
        forward
            .report(IssueSequenceUpdate::Issue(make_issue(position, tie)))
            .await
            .unwrap_or_else(|error| panic!("retain forward issue: {error}"));
    }
    forward
        .report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(9),
        })
        .await
        .unwrap_or_else(|error| panic!("advance forward frontier: {error}"));

    let reverse_view = reverse
        .receipt_partition_view(&barrier(run(0x11), 1))
        .await
        .unwrap_or_else(|error| panic!("reverse receipt view: {error}"));
    let forward_view = forward
        .receipt_partition_view(&barrier(run(0x11), 1))
        .await
        .unwrap_or_else(|error| panic!("forward receipt view: {error}"));
    assert_eq!(reverse_view.receipt_root(), forward_view.receipt_root());
    assert_eq!(reverse_view.payload_bytes(), forward_view.payload_bytes());
}

#[tokio::test(flavor = "current_thread")]
async fn tiny_reporter_budget_refuses_without_frontier_or_counter_mutation() {
    let shared_budget = budget(1, 8);
    let mut reporter =
        BudgetOwnedStreamingIssueReporter::new(run(0x11), record_policy(), shared_budget.clone())
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
    let error = reporter
        .report(IssueSequenceUpdate::Issue(record_issue(
            DecodeFailureCode::Syntax,
        )))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        aiperf_runtime::streaming::reliability::StreamingReliabilityError::StateBudget(_)
    ));
    assert_eq!(reporter.summary().unwrap().total, 0);
    assert_eq!(reporter.counters().iter().count(), 0);
    assert_eq!(shared_budget.snapshot().used_items, 0);
    assert_eq!(shared_budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn receipt_partition_handoff_moves_payload_and_view_leases_without_copy() {
    let reporter_budget = budget(64, 64 * 1024);
    let descriptor_budget = budget(4, 4096);
    let mut reporter =
        BudgetOwnedStreamingIssueReporter::new(run(0x11), record_policy(), reporter_budget.clone())
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
    let input_domain = domain(0x21, 0x20);
    reporter
        .report(IssueSequenceUpdate::Issue(record_issue(
            DecodeFailureCode::Syntax,
        )))
        .await
        .unwrap_or_else(|error| panic!("retain issue: {error}"));
    reporter
        .report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(7),
        })
        .await
        .unwrap_or_else(|error| panic!("advance issue: {error}"));
    let view = reporter
        .receipt_partition_view(&barrier(run(0x11), 4))
        .await
        .unwrap_or_else(|error| panic!("prepare issue partition: {error}"));
    let payload_ptr = view.payload_bytes().as_ptr();
    let payload_len = view.payload_bytes().len();
    let receipt_root = *view.receipt_root();
    let projection = ResultProjectionId::new("streaming_issue_receipts")
        .unwrap_or_else(|error| panic!("valid issue projection: {error}"));
    let descriptor = ResultSegmentDescriptor {
        run: run(0x11),
        epoch: CheckpointEpoch::new(4),
        cell_id: CellId::new(0),
        worker_id: WorkerId::new(0),
        projection,
        schema: ResultSchemaVersion::new(2),
        first_sequence: GlobalSequence::new(0),
        last_sequence: GlobalSequence::new(0),
        item_count: 1,
        byte_length: payload_len as u64,
        membership_root: receipt_root,
        payload_digest: ContentDigest::from_bytes(*blake3::hash(view.payload_bytes()).as_bytes()),
    };
    let descriptor_bytes = std::mem::size_of::<ResultSegmentDescriptor>()
        + descriptor.projection.retained_allocation_bytes();
    let descriptor_lease = descriptor_budget
        .acquire(1, descriptor_bytes)
        .await
        .unwrap_or_else(|error| panic!("charge issue descriptor: {error}"));
    let descriptor = BudgetedResultDescriptor::new(descriptor, descriptor_lease)
        .unwrap_or_else(|error| panic!("budgeted issue descriptor: {error}"));
    let handoff = view
        .into_result_partition(descriptor)
        .unwrap_or_else(|error| panic!("move issue partition: {error}"));
    assert_eq!(handoff.partition().payload_bytes().as_ptr(), payload_ptr);
    assert_eq!(handoff.receipt_root(), &receipt_root);
    assert_eq!(reporter.retained_receipt_count(), 1);
    drop(handoff);
    assert_eq!(descriptor_budget.snapshot().used_items, 0);
    assert_eq!(reporter.retained_receipt_count(), 1);
}

#[test]
fn handled_issue_cut_is_clone_safe_and_strictly_decoded() {
    let empty = HandledIssueCut::empty();
    assert_eq!(empty.clone(), empty);
    assert_ne!(empty.receipt_root(), &ContentDigest::from_bytes([0; 32]));

    let encoded = serde_json::to_value(&empty)
        .unwrap_or_else(|error| panic!("serialize handled cut: {error}"));
    assert_eq!(
        serde_json::from_value::<HandledIssueCut>(encoded.clone())
            .unwrap_or_else(|error| panic!("strict handled cut: {error}")),
        empty
    );
    let mut unknown = encoded;
    unknown
        .as_object_mut()
        .unwrap_or_else(|| panic!("handled cut object"))
        .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<HandledIssueCut>(unknown).is_err());
}

#[allow(dead_code)]
fn only_terminal_action_disposition_carries_failure_identity(
    disposition: ActionFailureDisposition,
) -> Option<PreparedActionFailureIdentity> {
    match disposition {
        ActionFailureDisposition::Pending(_)
        | ActionFailureDisposition::Retry(_)
        | ActionFailureDisposition::Backpressure(_) => None,
        ActionFailureDisposition::TerminalActionReceipt(identity) => Some(identity),
    }
}
