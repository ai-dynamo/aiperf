// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins exporter-member evidence to the controlled observable bytes.

use aiperf_bench_tools::plugin_stats::{
    ExporterEvidenceMode, ExporterMember, ExporterMemberBinding, ExporterMemberEvidence,
    ExporterObservableKind, ExporterSampleContract, RetainedExporterEvidence,
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
    ExporterMemberBinding {
        mode: ExporterEvidenceMode::Paired,
        experiment_identity_blake3: DIGEST.to_owned(),
        attempt_ordinal: 0,
        scenario_id: "exporter_100k".to_owned(),
        pair_id: "pair-00".to_owned(),
        member,
        corpus_blake3: DIGEST.to_owned(),
        observable_kind: ExporterObservableKind::ArtifactTree,
        observable_policy_blake3: DIGEST.to_owned(),
        build_artifact_blake3: DIGEST.to_owned(),
        build_receipt_blake3: DIGEST.to_owned(),
    }
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
        "observable_policy_blake3": DIGEST,
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
