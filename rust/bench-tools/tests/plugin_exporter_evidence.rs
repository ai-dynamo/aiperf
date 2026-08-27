// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins exporter-member evidence to the controlled observable bytes.

use aiperf_bench_tools::plugin_stats::{
    ExporterEvidenceMode, ExporterMember, ExporterMemberBinding, ExporterMemberEvidence,
    ExporterObservableKind, ExporterSampleContract, RetainedExporterEvidence,
    validate_exporter_member_evidence,
};

const DIGEST: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const RAW_OBSERVABLE: &[u8] = b"[{\"blake3\":\"blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262\",\"kind\":\"regular_file\",\"length\":0,\"path\":\"records.json\"}]\n";
const RAW_OBSERVABLE_DIGEST: &str =
    "blake3:12c662e7e69f13a334a6a1fceeb8d2cf315eea47d82ff4ff644225e7bbe84b4a";
const EMPTY_PROVENANCE: &[u8] = b"[]\n";
const EMPTY_PROVENANCE_DIGEST: &str =
    "blake3:9fa8dc9570625be2be53d308f958332981ec8fb8137d3dd7ba0ae5da317eaa7d";

fn paired_member_receipts() -> Vec<u8> {
    let receipts = (0_u64..16)
        .map(|repetition_ordinal| {
            serde_json::json!({
                "active_duration_ns": 1,
                "attempt_ordinal": 0,
                "build_artifact_blake3": DIGEST,
                "build_receipt_blake3": DIGEST,
                "comparison_observable_blake3": RAW_OBSERVABLE_DIGEST,
                "corpus_blake3": DIGEST,
                "experiment_identity_blake3": DIGEST,
                "member": "dynamic",
                "observable_kind": "artifact_tree",
                "pair_id": "pair-00",
                "processed_records": 100_000,
                "provenance_receipt_blake3": EMPTY_PROVENANCE_DIGEST,
                "raw_observable_blake3": RAW_OBSERVABLE_DIGEST,
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

fn binding() -> ExporterMemberBinding {
    ExporterMemberBinding {
        mode: ExporterEvidenceMode::Paired,
        experiment_identity_blake3: DIGEST.to_owned(),
        attempt_ordinal: 0,
        scenario_id: "exporter_100k".to_owned(),
        pair_id: "pair-00".to_owned(),
        member: ExporterMember::Dynamic,
        corpus_blake3: DIGEST.to_owned(),
        observable_kind: ExporterObservableKind::ArtifactTree,
        observable_policy_blake3: DIGEST.to_owned(),
        build_artifact_blake3: DIGEST.to_owned(),
        build_receipt_blake3: DIGEST.to_owned(),
    }
}

#[test]
fn retained_observable_mutation_invalidates_the_exporter_member() {
    let receipt_bytes = paired_member_receipts();
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
        &binding(),
        &evidence,
    )
    .expect("complete paired exporter evidence is valid");
    assert_eq!(summary.active_duration_nanoseconds, 16);
    assert_eq!(summary.processed_records, 1_600_000);

    let mut forged = evidence;
    forged.retained.raw_observable_bytes.push(b' ');
    let error = validate_exporter_member_evidence(
        &ExporterSampleContract::normative(),
        &binding(),
        &forged,
    )
    .expect_err("retained raw bytes cannot be rewritten under the receipt");
    assert_eq!(
        error.to_string(),
        "retained raw observable digest does not match its repetition receipt"
    );
}
