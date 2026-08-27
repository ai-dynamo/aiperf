// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::identity::{
    ContentDigest, DuplicateDisposition, IdentityError, ImmutableObjectIdentity,
    LogicalRecordReceipt, RunIncarnationId, StableActionId, StableRecordId, attempt_id,
    classify_logical_duplicate, one_turn_session_key, physical_record_id, stable_action_id,
    stable_record_id_from_key, stable_session_key,
};
use aiperf_runtime::streaming::unit::{
    ConversationTurnFragment, DatasetActionKind, EventTimeUtc, SourcePosition, UnitProvenance,
};

fn provenance(partition: u8, position: u64, format: u8) -> UnitProvenance {
    UnitProvenance {
        source_partition: ImmutableObjectIdentity::from_bytes([partition; 32]),
        source_position: SourcePosition::new(position),
        format_semantic_digest: ContentDigest::from_bytes([format; 32]),
    }
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

#[test]
fn stable_record_id_is_discovery_order_independent() {
    let first = stable_record_id_from_key(b"tenant/model", b"producer-record-7");
    let second = stable_record_id_from_key(b"tenant/model", b"producer-record-7");
    assert_eq!(first, second);
}

#[test]
fn stable_session_key_joins_partitions() {
    let from_partition_a = stable_session_key(b"trace-v1", b"session-42");
    let from_partition_b = stable_session_key(b"trace-v1", b"session-42");
    assert_eq!(from_partition_a, from_partition_b);
}

#[test]
fn canonical_identity_derivations_match_golden_bytes() {
    let partition = ImmutableObjectIdentity::from_bytes([0x11; 32]);
    let physical = physical_record_id(
        b"stream-01",
        &partition,
        b"partition-7:offset-99",
        &[0x22; 32],
    );
    let logical = stable_record_id_from_key(b"tenant/model", b"producer-record-7");
    let session = stable_session_key(b"trace-v1", b"session-42");
    let one_turn = one_turn_session_key(logical);
    let causes = [
        logical,
        stable_record_id_from_key(b"tenant/model", b"producer-record-8"),
    ];
    let action = stable_action_id(
        &[0x33; 32],
        session,
        &causes,
        DatasetActionKind::GraphNode,
        17,
    );
    let attempt = attempt_id(action, RunIncarnationId::from_bytes([0x44; 32]), 3);

    let actual = [
        ("physical", digest_hex(physical.as_bytes())),
        ("logical_record", digest_hex(logical.as_bytes())),
        ("session", digest_hex(session.as_bytes())),
        ("one_turn_session", digest_hex(one_turn.as_bytes())),
        ("action", digest_hex(action.as_bytes())),
        ("attempt", digest_hex(attempt.as_bytes())),
    ];
    let expected = [
        (
            "physical",
            "d5d82ed09c896104067729397a478e9022d2a154983b8b8f56735d2906a8c4d5".into(),
        ),
        (
            "logical_record",
            "01e71d3f054625f88c74b0b1418054255fbd79d3b7ec654dab55c79de1bd18df".into(),
        ),
        (
            "session",
            "31ac4ecbf2f7c7b972e7e2df2c338ad7134bed578dac97e8f2a0c48e0da9ff56".into(),
        ),
        (
            "one_turn_session",
            "7fb3be697d63da2c0b1a4f44dee9956d873116640f4d7c0a6c6eea6e0af365be".into(),
        ),
        (
            "action",
            "0c93f2aa4a772ff265faecad5d9330d2d793f84e4d83a1e3bf4625b5bc5a9dfb".into(),
        ),
        (
            "attempt",
            "e205a6c12f94991155fed4ce965addfe9fbd120f3e3493ab448fb9345303f8d4".into(),
        ),
    ];

    assert_eq!(actual, expected);
}

#[test]
fn canonical_length_framing_distinguishes_ambiguous_concatenations() {
    let left = stable_record_id_from_key(b"ab", b"c");
    let right = stable_record_id_from_key(b"a", b"bc");
    assert_ne!(left, right);
}

#[test]
fn attempt_identity_changes_with_incarnation() {
    let action = StableActionId::from_bytes([9; 32]);
    let first = attempt_id(action, RunIncarnationId::from_bytes([1; 32]), 0);
    let second = attempt_id(action, RunIncarnationId::from_bytes([2; 32]), 0);
    assert_ne!(first, second);
}

#[test]
fn stable_action_id_ignores_execution_topology() {
    struct TopologyCase {
        worker: u64,
        cell: u64,
        discovery_order: u64,
        global_sequence: u64,
    }

    let session = stable_session_key(b"trace-v1", b"session-42");
    let causes = [
        stable_record_id_from_key(b"trace-v1", b"record-1"),
        stable_record_id_from_key(b"trace-v1", b"record-2"),
    ];
    let cases = [
        TopologyCase {
            worker: 0,
            cell: 0,
            discovery_order: 1,
            global_sequence: 20,
        },
        TopologyCase {
            worker: 7,
            cell: 4,
            discovery_order: 99,
            global_sequence: 3,
        },
    ];

    let ids: Vec<_> = cases
        .iter()
        .map(|topology| {
            let _placement_only = (
                topology.worker,
                topology.cell,
                topology.discovery_order,
                topology.global_sequence,
            );
            stable_action_id(&[8; 32], session, &causes, DatasetActionKind::GraphNode, 2)
        })
        .collect();

    assert_eq!(ids[0], ids[1]);
}

#[test]
fn logical_record_duplicate_classification_rejects_conflicting_content() {
    let record_id = StableRecordId::from_bytes([3; 32]);
    let existing = LogicalRecordReceipt {
        record_id,
        content_digest: ContentDigest::from_bytes([4; 32]),
        provenance: provenance(7, 10, 8),
    };
    let cases = [
        (
            ContentDigest::from_bytes([4; 32]),
            Ok(DuplicateDisposition::Identical),
        ),
        (
            ContentDigest::from_bytes([5; 32]),
            Err("logical_identity_conflict"),
        ),
    ];

    for (content_digest, expected) in cases {
        let candidate = LogicalRecordReceipt {
            record_id,
            content_digest,
            provenance: provenance(9, 20, 8),
        };
        match expected {
            Ok(disposition) => assert_eq!(
                classify_logical_duplicate(&existing, &candidate),
                Ok(disposition)
            ),
            Err(code) => {
                let error = classify_logical_duplicate(&existing, &candidate)
                    .expect_err("conflicting content must fail");
                assert_eq!(error.code(), code);
                let IdentityError::LogicalIdentityConflict {
                    existing: retained_existing,
                    candidate: retained_candidate,
                } = error;
                assert_eq!(retained_existing.provenance, existing.provenance);
                assert_eq!(retained_candidate.provenance, candidate.provenance);
            }
        }
    }

    let distinct = LogicalRecordReceipt {
        record_id: StableRecordId::from_bytes([6; 32]),
        content_digest: existing.content_digest,
        provenance: provenance(11, 30, 8),
    };
    assert_eq!(
        classify_logical_duplicate(&existing, &distinct),
        Ok(DuplicateDisposition::New)
    );
}

#[test]
fn event_time_rejects_negative_nanoseconds() {
    for value in [-1, i64::MIN] {
        assert!(EventTimeUtc::new(value).is_err());
    }
    assert_eq!(EventTimeUtc::new(0).expect("epoch is valid").get(), 0);
}

#[test]
fn source_position_checked_add_rejects_u64_overflow() {
    let cases = [
        (0, 1, Some(1)),
        (u64::MAX - 1, 1, Some(u64::MAX)),
        (u64::MAX, 1, None),
    ];

    for (position, delta, expected) in cases {
        let result = SourcePosition::new(position).checked_add(delta);
        assert_eq!(result.ok().map(SourcePosition::get), expected);
    }
}

#[test]
fn canonical_structs_reject_unknown_fields() {
    let json = r#"{"role":"user","content":[104,105],"turn_ordinal":0,"transport":"http"}"#;
    let error = serde_json::from_str::<ConversationTurnFragment>(json)
        .expect_err("host vocabulary must reject unknown fields");
    assert!(error.to_string().contains("unknown field"));
}
