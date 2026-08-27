// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins exporter-member evidence to the controlled observable bytes.

use std::collections::BTreeSet;

use aiperf_bench_tools::exporter_policy::{
    ExporterObservablePolicyV1, parse_exporter_observable_policy,
};
use aiperf_bench_tools::plugin_stats::{
    ControlledAttemptDecision, ControlledMeasurementEvaluator, ExporterEvidenceMode,
    ExporterMember, ExporterMemberBinding, ExporterMemberEvidence, ExporterObservableKind,
    ExporterSampleContract, PairAttemptDecision, RetainedExporterEvidence,
    validate_exporter_member_evidence, validate_exporter_member_record,
    validate_exporter_pair_evidence,
};

const DIGEST: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const RAW_OBSERVABLE: &[u8] = b"[{\"blake3\":\"blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262\",\"kind\":\"regular_file\",\"length\":0,\"path\":\"records.json\"}]\n";
const RAW_OBSERVABLE_DIGEST: &str =
    "blake3:12c662e7e69f13a334a6a1fceeb8d2cf315eea47d82ff4ff644225e7bbe84b4a";
const EMPTY_PROVENANCE: &[u8] = b"[]\n";
const EMPTY_PROVENANCE_DIGEST: &str =
    "blake3:9fa8dc9570625be2be53d308f958332981ec8fb8137d3dd7ba0ae5da317eaa7d";

fn empty_paired_policy() -> ExporterObservablePolicyV1 {
    parse_exporter_observable_policy(
        b"{\"mode\":\"paired\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"artifact_tree\",\"provenance_slots\":[],\"scenario_id\":\"exporter_100k\"}],\"schema_version\":1}\n",
        &BTreeSet::new(),
    )
    .expect("literal policy validates")
}

fn paired_member_receipts(member: ExporterMember, observable_digest: &str) -> Vec<u8> {
    let receipts = (0_u64..16)
        .map(|repetition_ordinal| {
            serde_json::json!({
                "active_duration_ns": 1,
                "attempt_ordinal": 0,
                "build_artifact_blake3": DIGEST,
                "build_receipt_blake3": DIGEST,
                "comparison_observable_blake3": observable_digest,
                "corpus_blake3": DIGEST,
                "experiment_identity_blake3": DIGEST,
                "member": member,
                "observable_kind": "artifact_tree",
                "pair_id": "pair-00",
                "processed_records": 100_000,
                "provenance_receipt_blake3": EMPTY_PROVENANCE_DIGEST,
                "raw_observable_blake3": observable_digest,
                "repetition_ordinal": repetition_ordinal,
                "scenario_id": "exporter_100k",
                "schema_version": 1
            })
        })
        .collect::<Vec<_>>();
    let mut bytes = serde_json::to_vec(&receipts).expect("literal receipts serialize");
    bytes.push(b'\n');
    bytes
}

fn binding(member: ExporterMember) -> ExporterMemberBinding {
    let policy_blake3 = empty_paired_policy()
        .canonical_blake3()
        .expect("literal policy canonicalizes");
    ExporterMemberBinding {
        mode: ExporterEvidenceMode::Paired,
        experiment_identity_blake3: DIGEST.to_owned(),
        attempt_ordinal: 0,
        scenario_id: "exporter_100k".to_owned(),
        pair_id: "pair-00".to_owned(),
        member,
        corpus_blake3: DIGEST.to_owned(),
        observable_kind: ExporterObservableKind::ArtifactTree,
        observable_policy_blake3: policy_blake3,
        build_artifact_blake3: DIGEST.to_owned(),
        build_receipt_blake3: DIGEST.to_owned(),
    }
}

fn member_evidence(member: ExporterMember) -> ExporterMemberEvidence {
    ExporterMemberEvidence {
        repetition_receipt_bytes: paired_member_receipts(member, RAW_OBSERVABLE_DIGEST),
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
            comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
            provenance_receipt_bytes: EMPTY_PROVENANCE.to_vec(),
        },
    }
}

fn member_record_bytes(
    member: ExporterMember,
    evidence: &ExporterMemberEvidence,
    duration_adjustment: u64,
) -> Vec<u8> {
    let summary = validate_exporter_member_evidence(
        &ExporterSampleContract::normative(),
        &binding(member),
        evidence,
    )
    .expect("member evidence validates");
    let retained = &summary.repetitions[evidence.retained.repetition_ordinal];
    let record = serde_json::json!({
        "active_duration_ns": summary.active_duration_nanoseconds + duration_adjustment,
        "attempt_ordinal": 0,
        "build_artifact_blake3": DIGEST,
        "build_receipt_blake3": DIGEST,
        "comparison_observable_blake3": RAW_OBSERVABLE_DIGEST,
        "experiment_identity_blake3": DIGEST,
        "member": member,
        "observable_policy_blake3": binding(member).observable_policy_blake3,
        "pair_id": "pair-00",
        "processed_records": 1_600_000,
        "repetition_receipts_blake3": summary.repetition_receipts_blake3,
        "retained_artifact_records": 100_000,
        "retained_comparison_observable_blake3": retained.comparison_observable_blake3,
        "retained_provenance_receipt_blake3": retained.provenance_receipt_blake3,
        "retained_raw_observable_blake3": retained.raw_observable_blake3,
        "retained_repetition_ordinal": 0,
        "scenario_id": "exporter_100k",
        "schema_version": 1
    });
    let mut bytes = serde_json::to_vec(&record).expect("literal record serializes");
    bytes.push(b'\n');
    bytes
}

fn mutate_receipts(
    member: ExporterMember,
    mutate: impl FnOnce(&mut Vec<serde_json::Value>),
) -> Vec<u8> {
    let mut receipts: Vec<serde_json::Value> =
        serde_json::from_slice(&paired_member_receipts(member, RAW_OBSERVABLE_DIGEST))
            .expect("literal receipts parse");
    mutate(&mut receipts);
    let mut bytes = serde_json::to_vec(&receipts).expect("mutated receipts serialize");
    bytes.push(b'\n');
    bytes
}

fn assert_receipts_rejected(label: &str, repetition_receipt_bytes: Vec<u8>) {
    let mut evidence = member_evidence(ExporterMember::Dynamic);
    evidence.repetition_receipt_bytes = repetition_receipt_bytes;
    assert!(
        validate_exporter_member_evidence(
            &ExporterSampleContract::normative(),
            &binding(ExporterMember::Dynamic),
            &evidence,
        )
        .is_err(),
        "forged receipt vector was accepted: {label}"
    );
}

#[test]
fn repetition_receipts_reject_every_binding_and_shape_substitution() {
    let field_mutations = [
        (
            "identity",
            "experiment_identity_blake3",
            serde_json::json!(EMPTY_PROVENANCE_DIGEST),
        ),
        ("attempt", "attempt_ordinal", serde_json::json!(1)),
        ("scenario", "scenario_id", serde_json::json!("other")),
        ("pair", "pair_id", serde_json::json!("pair-01")),
        ("member", "member", serde_json::json!("static")),
        (
            "class",
            "observable_kind",
            serde_json::json!("captured_stream"),
        ),
        (
            "record count",
            "processed_records",
            serde_json::json!(99_999),
        ),
        (
            "corpus",
            "corpus_blake3",
            serde_json::json!(EMPTY_PROVENANCE_DIGEST),
        ),
        (
            "artifact build",
            "build_artifact_blake3",
            serde_json::json!(EMPTY_PROVENANCE_DIGEST),
        ),
        (
            "build receipt",
            "build_receipt_blake3",
            serde_json::json!(EMPTY_PROVENANCE_DIGEST),
        ),
        ("zero duration", "active_duration_ns", serde_json::json!(0)),
    ];
    for (label, field, replacement) in field_mutations {
        assert_receipts_rejected(
            label,
            mutate_receipts(ExporterMember::Dynamic, |receipts| {
                receipts[0][field] = replacement;
            }),
        );
    }

    assert_receipts_rejected(
        "missing field",
        mutate_receipts(ExporterMember::Dynamic, |receipts| {
            receipts[0]
                .as_object_mut()
                .expect("receipt is an object")
                .remove("corpus_blake3");
        }),
    );
    assert_receipts_rejected(
        "extra field",
        mutate_receipts(ExporterMember::Dynamic, |receipts| {
            receipts[0]["forged"] = serde_json::json!(true);
        }),
    );
    assert_receipts_rejected(
        "reordered ordinal",
        mutate_receipts(ExporterMember::Dynamic, |receipts| receipts.swap(0, 1)),
    );
    assert_receipts_rejected(
        "duplicate ordinal",
        mutate_receipts(ExporterMember::Dynamic, |receipts| {
            receipts[1]["repetition_ordinal"] = serde_json::json!(0);
        }),
    );

    let canonical = paired_member_receipts(ExporterMember::Dynamic, RAW_OBSERVABLE_DIGEST);
    let duplicate_key = String::from_utf8(canonical)
        .expect("literal receipt bytes are UTF-8")
        .replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        )
        .into_bytes();
    assert_receipts_rejected("duplicate field", duplicate_key);
}

#[test]
fn retained_observable_mutation_invalidates_the_exporter_member() {
    let receipt_bytes = paired_member_receipts(ExporterMember::Dynamic, RAW_OBSERVABLE_DIGEST);
    let retained = RetainedExporterEvidence {
        repetition_ordinal: 0,
        raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
        comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
        provenance_receipt_bytes: EMPTY_PROVENANCE.to_vec(),
    };
    let evidence = ExporterMemberEvidence {
        repetition_receipt_bytes: receipt_bytes,
        retained,
    };

    let summary = validate_exporter_member_evidence(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Dynamic),
        &evidence,
    )
    .expect("complete paired exporter evidence is valid");
    assert_eq!(summary.active_duration_nanoseconds, 16);
    assert_eq!(summary.processed_records, 1_600_000);

    let mut forged = evidence;
    forged.retained.raw_observable_bytes.push(b' ');
    let error = validate_exporter_member_evidence(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Dynamic),
        &forged,
    )
    .expect_err("retained raw bytes cannot be rewritten under the receipt");
    assert_eq!(
        error.to_string(),
        "retained raw observable digest does not match its repetition receipt"
    );
}

#[test]
fn comparison_observable_must_match_across_the_pair() {
    let static_evidence = ExporterMemberEvidence {
        repetition_receipt_bytes: paired_member_receipts(
            ExporterMember::Static,
            RAW_OBSERVABLE_DIGEST,
        ),
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
            comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
            provenance_receipt_bytes: EMPTY_PROVENANCE.to_vec(),
        },
    };
    let dynamic_evidence = ExporterMemberEvidence {
        repetition_receipt_bytes: paired_member_receipts(
            ExporterMember::Dynamic,
            EMPTY_PROVENANCE_DIGEST,
        ),
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: EMPTY_PROVENANCE.to_vec(),
            comparison_observable_bytes: EMPTY_PROVENANCE.to_vec(),
            provenance_receipt_bytes: EMPTY_PROVENANCE.to_vec(),
        },
    };

    let error = validate_exporter_pair_evidence(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Static),
        &static_evidence,
        &binding(ExporterMember::Dynamic),
        &dynamic_evidence,
    )
    .expect_err("a pair cannot compare unequal observable bytes");
    assert_eq!(
        error.to_string(),
        "static and dynamic exporter comparison observables differ"
    );
}

#[test]
fn member_record_cannot_claim_a_duration_other_than_the_receipt_sum() {
    let evidence = ExporterMemberEvidence {
        repetition_receipt_bytes: paired_member_receipts(
            ExporterMember::Dynamic,
            RAW_OBSERVABLE_DIGEST,
        ),
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
            comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
            provenance_receipt_bytes: EMPTY_PROVENANCE.to_vec(),
        },
    };
    let summary = validate_exporter_member_evidence(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Dynamic),
        &evidence,
    )
    .expect("member evidence validates");
    let retained = &summary.repetitions[evidence.retained.repetition_ordinal];
    let mut record = serde_json::json!({
        "active_duration_ns": summary.active_duration_nanoseconds + 1,
        "attempt_ordinal": 0,
        "build_artifact_blake3": DIGEST,
        "build_receipt_blake3": DIGEST,
        "comparison_observable_blake3": RAW_OBSERVABLE_DIGEST,
        "experiment_identity_blake3": DIGEST,
        "member": "dynamic",
        "observable_policy_blake3": binding(ExporterMember::Dynamic).observable_policy_blake3,
        "pair_id": "pair-00",
        "processed_records": 1_600_000,
        "repetition_receipts_blake3": summary.repetition_receipts_blake3,
        "retained_artifact_records": 100_000,
        "retained_comparison_observable_blake3": retained.comparison_observable_blake3,
        "retained_provenance_receipt_blake3": retained.provenance_receipt_blake3,
        "retained_raw_observable_blake3": retained.raw_observable_blake3,
        "retained_repetition_ordinal": 0,
        "scenario_id": "exporter_100k",
        "schema_version": 1
    });
    let mut record_bytes = serde_json::to_vec(&record).expect("literal record serializes");
    record_bytes.push(b'\n');

    let error = validate_exporter_member_record(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Dynamic),
        &evidence,
        &record_bytes,
    )
    .expect_err("the member duration must equal the repetition sum");
    assert_eq!(
        error.to_string(),
        "exporter member record does not match its validated evidence"
    );

    record["active_duration_ns"] = serde_json::json!(summary.active_duration_nanoseconds);
    let mut valid_bytes = serde_json::to_vec(&record).expect("literal record serializes");
    valid_bytes.push(b'\n');
    validate_exporter_member_record(
        &ExporterSampleContract::normative(),
        &binding(ExporterMember::Dynamic),
        &evidence,
        &valid_bytes,
    )
    .expect("the exact receipt sum is valid");
}

#[test]
fn controlled_evaluator_classifies_invalid_exporter_evidence_as_a_product_failure() {
    let static_evidence = member_evidence(ExporterMember::Static);
    let dynamic_evidence = member_evidence(ExporterMember::Dynamic);
    let static_record = member_record_bytes(ExporterMember::Static, &static_evidence, 0);
    let forged_dynamic_record = member_record_bytes(ExporterMember::Dynamic, &dynamic_evidence, 1);
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("first attempt starts");

    let decision = evaluator
        .record_exporter_pair_evidence(
            &empty_paired_policy(),
            &binding(ExporterMember::Static),
            &static_evidence,
            &[],
            &static_record,
            &binding(ExporterMember::Dynamic),
            &dynamic_evidence,
            &[],
            &forged_dynamic_record,
        )
        .expect("controlled validation returns a terminal product decision");
    assert_eq!(decision, PairAttemptDecision::ExperimentFailed);
    assert_eq!(
        evaluator.history()[0].decision,
        ControlledAttemptDecision::ValidFailure
    );
}

#[test]
fn controlled_evaluator_retains_validated_exporter_member_records() {
    let static_evidence = member_evidence(ExporterMember::Static);
    let dynamic_evidence = member_evidence(ExporterMember::Dynamic);
    let static_record = member_record_bytes(ExporterMember::Static, &static_evidence, 0);
    let dynamic_record = member_record_bytes(ExporterMember::Dynamic, &dynamic_evidence, 0);
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("first attempt starts");

    let decision = evaluator
        .record_exporter_pair_evidence(
            &empty_paired_policy(),
            &binding(ExporterMember::Static),
            &static_evidence,
            &[],
            &static_record,
            &binding(ExporterMember::Dynamic),
            &dynamic_evidence,
            &[],
            &dynamic_record,
        )
        .expect("controlled exporter pair validates");
    assert_eq!(decision, PairAttemptDecision::RetainPair);
    let retained = evaluator.exporter_pair_history();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].experiment_attempt, 1);
    assert_eq!(retained[0].scenario, "exporter_100k");
    assert_eq!(retained[0].pair_id, "pair-00");
    assert_eq!(
        retained[0].static_record.repetition_receipts_blake3,
        retained[0].static_member.repetition_receipts_blake3
    );
    assert_eq!(
        retained[0].dynamic_record.repetition_receipts_blake3,
        retained[0].dynamic_member.repetition_receipts_blake3
    );
}

#[test]
fn controlled_evaluator_refuses_evidence_not_derived_from_its_bound_policy() {
    let static_evidence = member_evidence(ExporterMember::Static);
    let dynamic_evidence = member_evidence(ExporterMember::Dynamic);
    let static_record = member_record_bytes(ExporterMember::Static, &static_evidence, 0);
    let dynamic_record = member_record_bytes(ExporterMember::Dynamic, &dynamic_evidence, 0);
    let mut forged_static_binding = binding(ExporterMember::Static);
    forged_static_binding.observable_policy_blake3 = DIGEST.to_owned();
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("first attempt starts");

    assert_eq!(
        evaluator
            .record_exporter_pair_evidence(
                &empty_paired_policy(),
                &forged_static_binding,
                &static_evidence,
                &[],
                &static_record,
                &binding(ExporterMember::Dynamic),
                &dynamic_evidence,
                &[],
                &dynamic_record,
            )
            .expect("policy mismatch is a terminal product decision"),
        PairAttemptDecision::ExperimentFailed
    );
    assert_eq!(
        evaluator.history()[0].decision,
        ControlledAttemptDecision::ValidFailure
    );
}
