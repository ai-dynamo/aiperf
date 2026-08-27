// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Artifact-bound exporter child protocol admitted by the controlled runner.

use aiperf_bench_tools::plugin_stats::{
    ArtifactBoundExporterMemberV1, ExporterEvidenceMode, ExporterMember, ExporterMemberBinding,
    ExporterMemberEvidence, ExporterMemberRecord, ExporterObservableKind,
    ExporterRepetitionReceipt, ExporterSampleContract, RetainedExporterEvidence, Variant,
};
use aiperf_bench_tools::runtime_runner::{
    ExporterChildExpectationV1, ExporterMemberChildOutputV1,
    validate_exporter_member_child_output_v1,
};

const IDENTITY: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CORPUS: &str = "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const POLICY: &str = "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const ARTIFACT: &str = "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const RECEIPT: &str = "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const SCENARIO: &str = "exporter_100k";
const PAIR_ID: &str = "pair-01";
const RAW_OBSERVABLE: &[u8] = b"[{\"kind\":\"regular_file\",\"path\":\"records.json\"}]\n";
const PROVENANCE: &[u8] = b"[]\n";
const REPETITION_NS: u64 = 2_000_000_000;

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

fn canonical_line<T: serde::Serialize>(value: &T) -> Vec<u8> {
    let mut bytes = serde_json_canonicalizer::to_vec(value).expect("value canonicalizes");
    bytes.push(b'\n');
    bytes
}

fn binding(member: ExporterMember) -> ExporterMemberBinding {
    ExporterMemberBinding {
        mode: ExporterEvidenceMode::Paired,
        experiment_identity_blake3: IDENTITY.to_owned(),
        attempt_ordinal: 0,
        scenario_id: SCENARIO.to_owned(),
        pair_id: PAIR_ID.to_owned(),
        member,
        corpus_blake3: CORPUS.to_owned(),
        observable_kind: ExporterObservableKind::ArtifactTree,
        observable_policy_blake3: POLICY.to_owned(),
        build_artifact_blake3: ARTIFACT.to_owned(),
        build_receipt_blake3: RECEIPT.to_owned(),
    }
}

fn receipts(member: ExporterMember, repetitions: usize) -> Vec<ExporterRepetitionReceipt> {
    (0..repetitions)
        .map(|ordinal| ExporterRepetitionReceipt {
            schema_version: 1,
            experiment_identity_blake3: IDENTITY.to_owned(),
            attempt_ordinal: 0,
            scenario_id: SCENARIO.to_owned(),
            pair_id: PAIR_ID.to_owned(),
            member,
            repetition_ordinal: ordinal as u64,
            corpus_blake3: CORPUS.to_owned(),
            processed_records: 100_000,
            observable_kind: ExporterObservableKind::ArtifactTree,
            raw_observable_blake3: digest(RAW_OBSERVABLE),
            comparison_observable_blake3: digest(RAW_OBSERVABLE),
            provenance_receipt_blake3: digest(PROVENANCE),
            active_duration_ns: REPETITION_NS,
            build_artifact_blake3: ARTIFACT.to_owned(),
            build_receipt_blake3: RECEIPT.to_owned(),
        })
        .collect()
}

fn evidence(member: ExporterMember, repetitions: usize) -> ExporterMemberEvidence {
    ExporterMemberEvidence {
        repetition_receipt_bytes: canonical_line(&receipts(member, repetitions)),
        retained: RetainedExporterEvidence {
            repetition_ordinal: 0,
            raw_observable_bytes: RAW_OBSERVABLE.to_vec(),
            comparison_observable_bytes: RAW_OBSERVABLE.to_vec(),
            provenance_receipt_bytes: PROVENANCE.to_vec(),
        },
    }
}

fn record(member: ExporterMember, evidence: &ExporterMemberEvidence) -> ExporterMemberRecord {
    let contract = ExporterSampleContract::normative();
    ExporterMemberRecord {
        schema_version: 1,
        experiment_identity_blake3: IDENTITY.to_owned(),
        attempt_ordinal: 0,
        scenario_id: SCENARIO.to_owned(),
        pair_id: PAIR_ID.to_owned(),
        member,
        active_duration_ns: REPETITION_NS * contract.sample_repetitions as u64,
        processed_records: contract.processed_records,
        retained_artifact_records: contract.retained_artifact_records,
        comparison_observable_blake3: digest(RAW_OBSERVABLE),
        repetition_receipts_blake3: digest(&evidence.repetition_receipt_bytes),
        retained_repetition_ordinal: 0,
        retained_raw_observable_blake3: digest(RAW_OBSERVABLE),
        retained_comparison_observable_blake3: digest(RAW_OBSERVABLE),
        retained_provenance_receipt_blake3: digest(PROVENANCE),
        observable_policy_blake3: POLICY.to_owned(),
        build_artifact_blake3: ARTIFACT.to_owned(),
        build_receipt_blake3: RECEIPT.to_owned(),
    }
}

fn child_output(member: ExporterMember, repetitions: usize) -> ExporterMemberChildOutputV1 {
    let member_evidence = evidence(member, repetitions);
    let record_bytes = canonical_line(&record(member, &member_evidence));
    ExporterMemberChildOutputV1 {
        artifact_bound: ArtifactBoundExporterMemberV1 {
            binding: binding(member),
            evidence: member_evidence,
            backing_payloads: Vec::new(),
            record_bytes,
            receiver_protocol: None,
            receiver_protocol_authority_blake3: None,
        },
        experiment_identity_blake3: IDENTITY.to_owned(),
        pair_id: PAIR_ID.to_owned(),
        scenario: SCENARIO.to_owned(),
        schema_version: 1,
        variant: match member {
            ExporterMember::Static => Variant::Static,
            ExporterMember::Dynamic => Variant::Dynamic,
        },
    }
}

fn expectation(member: ExporterMember) -> ExporterChildExpectationV1 {
    ExporterChildExpectationV1 {
        experiment_identity_blake3: IDENTITY.to_owned(),
        attempt_ordinal: 0,
        scenario_id: SCENARIO.to_owned(),
        pair_id: PAIR_ID.to_owned(),
        member,
        corpus_blake3: CORPUS.to_owned(),
        observable_kind: ExporterObservableKind::ArtifactTree,
        observable_policy_blake3: POLICY.to_owned(),
        build_artifact_blake3: ARTIFACT.to_owned(),
        build_receipt_blake3: RECEIPT.to_owned(),
        minimum_active_duration_ns: 30_000_000_000,
    }
}

#[test]
fn conforming_artifact_bound_exporter_child_is_admitted() {
    let member = ExporterMember::Dynamic;
    let bytes = canonical_line(&child_output(member, 16));
    let admitted = validate_exporter_member_child_output_v1(&bytes, &expectation(member))
        .expect("conforming artifact-bound exporter child is admitted");

    assert_eq!(admitted.summary.repetitions.len(), 16);
    assert_eq!(admitted.summary.processed_records, 1_600_000);
    assert_eq!(admitted.summary.retained_artifact_records, 100_000);
    assert_eq!(
        admitted.summary.active_duration_nanoseconds,
        REPETITION_NS * 16
    );
    assert!(
        (admitted.summary.exporter_nanoseconds_per_record
            - (REPETITION_NS * 16) as f64 / 1_600_000.0)
            .abs()
            < f64::EPSILON
    );
    assert_eq!(admitted.record.pair_id, PAIR_ID);
    assert_eq!(admitted.record.member, member);
    assert_eq!(
        admitted
            .artifact_bound
            .evidence
            .retained
            .raw_observable_bytes,
        RAW_OBSERVABLE
    );
    assert_eq!(admitted.artifact_bound.binding.member, member);
}

#[test]
fn bare_exporter_metric_child_line_is_refused_by_the_artifact_bound_protocol() {
    // The fixture must itself be canonical JCS (note `1`, not `1.0`), or the
    // canonical-line check refuses it first and this test proves nothing about
    // the artifact-bound schema. Non-canonical input is covered separately by
    // `child_output_that_is_not_one_canonical_line_is_refused`.
    let bare = concat!(
        "{\"active_duration_nanoseconds\":30000000000,\"completed_budget\":1,",
        "\"experiment_identity_blake3\":\"",
        "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "\",\"metrics\":{\"exporter_nanoseconds_per_record\":1},",
        "\"pair_id\":\"pair-01\",\"scenario\":\"exporter_100k\",",
        "\"schema_version\":1,\"variant\":\"dynamic\"}\n"
    );
    let error = validate_exporter_member_child_output_v1(
        bare.as_bytes(),
        &expectation(ExporterMember::Dynamic),
    )
    .expect_err("a bare exporter metric line carries no artifact-bound evidence");
    assert!(
        error.to_string().contains("artifact-bound"),
        "unexpected error: {error}"
    );
}

#[test]
fn child_binding_that_contradicts_the_controller_expectation_is_refused() {
    let member = ExporterMember::Dynamic;
    let mut output = child_output(member, 16);
    output.artifact_bound.binding.build_artifact_blake3 = RECEIPT.to_owned();
    let bytes = canonical_line(&output);
    let error = validate_exporter_member_child_output_v1(&bytes, &expectation(member))
        .expect_err("a child cannot rebind itself to a different artifact");
    assert!(
        error.to_string().contains("controller expectation"),
        "unexpected error: {error}"
    );
}

#[test]
fn child_with_an_incomplete_repetition_vector_is_refused() {
    let member = ExporterMember::Dynamic;
    let bytes = canonical_line(&child_output(member, 15));
    let error = validate_exporter_member_child_output_v1(&bytes, &expectation(member))
        .expect_err("fifteen repetitions cannot satisfy the frozen sample contract");
    assert!(
        error.to_string().contains("16 repetitions"),
        "unexpected error: {error}"
    );
}

#[test]
fn child_output_that_is_not_one_canonical_line_is_refused() {
    let member = ExporterMember::Dynamic;
    let mut bytes = canonical_line(&child_output(member, 16));
    bytes.extend_from_slice(b"\n");
    let error = validate_exporter_member_child_output_v1(&bytes, &expectation(member))
        .expect_err("trailing bytes are not one exact canonical line");
    assert!(
        error.to_string().contains("canonical"),
        "unexpected error: {error}"
    );
}
