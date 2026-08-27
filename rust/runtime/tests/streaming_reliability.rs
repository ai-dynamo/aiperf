// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU64;

use aiperf_runtime::streaming::{
    checkpoint::StreamRunIdentity,
    failure::{DecodeFailureCode, OrdinaryStreamingFailure, StreamFormatError},
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId, StableActionId,
        StableRecordId,
    },
    reliability::{
        ActionFailureDisposition, OrdinaryStreamingIssue, PreparedActionFailureIdentity,
        PreparedStreamingIssuePolicy, StreamingInputDomainIdentity, StreamingIssueClass,
        StreamingIssueComponentId, StreamingIssueDisposition, StreamingIssueScope,
        StreamingIssueScopeKind, StreamingIssueThresholdRule,
    },
    unit::SourcePosition,
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
        OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
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
fn persisted_scope_rejects_unknown_fields() {
    let value = serde_json::json!({
        "scope": "action",
        "action_id": vec![0; 32],
        "unexpected": true,
    });
    assert!(serde_json::from_value::<StreamingIssueScope>(value).is_err());
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
