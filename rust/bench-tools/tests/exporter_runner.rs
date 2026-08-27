// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral tests for harness-owned exporter workload and capture authority.

use std::collections::BTreeSet;

use aiperf_bench_tools::exporter_observable::parse_receiver_transcript_observable;
use aiperf_bench_tools::exporter_policy::{
    AuthenticatedReceiverProtocolV1, SelectedBackingPayloadV1, parse_exporter_observable_policy,
};
use aiperf_bench_tools::exporter_runner::{
    ExporterHarnessError, ExporterHarnessRunner, ExporterMemberSource, ExporterRecordStream,
    ExporterWorkload, HostExporterCapture,
};
use aiperf_bench_tools::plugin_stats::{
    ExporterMember, ExporterRepetitionReceipt, validate_exporter_member_record,
    validate_exporter_pair_evidence,
};

fn canonical_policy(value: serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json_canonicalizer::to_vec(&value).expect("policy canonicalizes");
    bytes.push(b'\n');
    bytes
}

fn stream_policy(mode: &str) -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    let expected = b"100000:c91b21b6231cb062a5b30678615b82be824bb561de0088a0406bf31563be1221"
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut slot = serde_json::json!({
        "locator": {"kind": "whole_output"},
        "output_selector": {"kind": "captured_stream"},
        "replacement": {"encoding": "hex_bytes", "value": "00"},
        "slot_id": "stream",
        "static_expected": {"encoding": "hex_bytes", "value": expected},
    });
    if mode == "paired" {
        slot["dynamic_expected"] = slot["static_expected"].clone();
    }
    parse_exporter_observable_policy(
        &canonical_policy(serde_json::json!({
            "mode": mode,
            "receiver_transport_fields_removed": [],
            "scenarios": [{
                "allows_empty": false,
                "observable_kind": "captured_stream",
                "provenance_slots": [slot],
                "scenario_id": "exporter_100k",
            }],
            "schema_version": 1,
        })),
        &BTreeSet::new(),
    )
    .expect("stream policy validates")
}

fn artifact_policy() -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    parse_exporter_observable_policy(
        &canonical_policy(serde_json::json!({
            "mode": "paired",
            "receiver_transport_fields_removed": [],
            "scenarios": [{
                "allows_empty": false,
                "observable_kind": "artifact_tree",
                "provenance_slots": [{
                    "locator": {"kind": "json_pointer", "pointer": "/plugin_lock_digest"},
                    "output_selector": {"kind": "artifact_content", "path": "records.json"},
                    "replacement": {"encoding": "canonical_json", "value": "@plugin-lock@"},
                    "slot_id": "plugin_lock",
                    "static_expected": {"encoding": "canonical_json", "value": "static-lock"},
                    "dynamic_expected": {"encoding": "canonical_json", "value": "dynamic-lock"},
                }],
                "scenario_id": "exporter_100k",
            }],
            "schema_version": 1,
        })),
        &BTreeSet::new(),
    )
    .expect("artifact policy validates")
}

fn receiver_policy(
    protocol: &str,
) -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    parse_exporter_observable_policy(
        &canonical_policy(serde_json::json!({
            "mode": "paired",
            "receiver_transport_fields_removed": [{
                "keys": ["date"],
                "protocol": protocol,
            }],
            "scenarios": [{
                "allows_empty": false,
                "observable_kind": "receiver_transcript",
                "provenance_slots": [{
                    "locator": {"kind": "byte_range", "length": 3, "offset": 3},
                    "output_selector": {"kind": "transcript_body", "sequence": 0},
                    "replacement": {"encoding": "hex_bytes", "value": "00ff"},
                    "slot_id": "tail",
                    "static_expected": {"encoding": "hex_bytes", "value": "58595a"},
                    "dynamic_expected": {"encoding": "hex_bytes", "value": "58595a"},
                }],
                "scenario_id": "exporter_100k",
            }],
            "schema_version": 1,
        })),
        &BTreeSet::from([protocol.to_owned()]),
    )
    .expect("receiver policy validates")
}

fn source<'a>(
    pair_id: &'a str,
    member: ExporterMember,
    artifact: &'a tempfile::NamedTempFile,
) -> ExporterMemberSource<'a> {
    ExporterMemberSource {
        experiment_identity_bytes: b"canonical experiment identity",
        attempt_ordinal: 0,
        scenario_id: "exporter_100k",
        pair_id,
        member,
        build_artifact: artifact.as_file(),
        build_receipt_bytes: b"authenticated build receipt",
        receiver_protocol: None,
    }
}

fn receiver_source<'a>(
    pair_id: &'a str,
    member: ExporterMember,
    artifact: &'a tempfile::NamedTempFile,
    protocol: Option<&'a AuthenticatedReceiverProtocolV1>,
) -> ExporterMemberSource<'a> {
    ExporterMemberSource {
        receiver_protocol: protocol,
        ..source(pair_id, member, artifact)
    }
}

#[derive(Default)]
struct DigestingStreamExporter {
    repetition_ordinals: Vec<u64>,
    processed_per_repetition: Vec<u64>,
    stop_after: Option<u64>,
}

impl ExporterWorkload for DigestingStreamExporter {
    fn export(
        &mut self,
        repetition_ordinal: u64,
        records: &mut ExporterRecordStream<'_>,
        capture: &mut HostExporterCapture,
    ) -> Result<(), ExporterHarnessError> {
        let mut digest = blake3::Hasher::new();
        let mut processed = 0_u64;
        for record in records {
            assert_eq!(record.ordinal(), processed);
            digest.update(record.jsonl_bytes());
            processed += 1;
            if self.stop_after == Some(processed) {
                break;
            }
        }
        self.repetition_ordinals.push(repetition_ordinal);
        self.processed_per_repetition.push(processed);
        capture.write_stream(format!("{processed}:{}", digest.finalize()).as_bytes())
    }
}

struct ArtifactExporter {
    lock_digest: &'static str,
}

impl ExporterWorkload for ArtifactExporter {
    fn export(
        &mut self,
        _repetition_ordinal: u64,
        records: &mut ExporterRecordStream<'_>,
        capture: &mut HostExporterCapture,
    ) -> Result<(), ExporterHarnessError> {
        let processed = records.count();
        assert_eq!(processed, 100_000);
        capture.create_artifact_directory("empty")?;
        capture.write_artifact(
            "records.json",
            format!(
                "{{\"plugin_lock_digest\" : \"{}\", \"other\" : 1.0}}",
                self.lock_digest
            )
            .as_bytes(),
        )
    }
}

struct ReceiverExporter {
    acknowledgements: Vec<(u64, usize)>,
    metadata: Vec<[String; 2]>,
}

impl Default for ReceiverExporter {
    fn default() -> Self {
        Self {
            acknowledgements: Vec::new(),
            metadata: vec![
                ["content-type".to_owned(), "application/json".to_owned()],
                ["date".to_owned(), "generated".to_owned()],
                ["x-request-id".to_owned(), "retained".to_owned()],
            ],
        }
    }
}

impl ExporterWorkload for ReceiverExporter {
    fn export(
        &mut self,
        _repetition_ordinal: u64,
        records: &mut ExporterRecordStream<'_>,
        capture: &mut HostExporterCapture,
    ) -> Result<(), ExporterHarnessError> {
        assert_eq!(records.count(), 100_000);
        let acknowledgement =
            capture.accept_receiver("POST", "/v1/traces", self.metadata.clone(), b"abcXYZ")?;
        self.acknowledgements.push((
            acknowledgement.sequence(),
            acknowledgement.recorded_acceptances(),
        ));
        Ok(())
    }
}

fn build_artifact() -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().expect("temporary artifact opens");
    std::fs::write(file.path(), b"fake exporter artifact").expect("artifact bytes write");
    file
}

#[test]
fn runner_owns_the_fixed_corpus_receipts_timing_and_member_record() {
    let artifact = build_artifact();
    let runner = ExporterHarnessRunner::new(stream_policy("paired"))
        .expect("runner builds its fixed corpus");
    assert_eq!(
        runner.corpus_blake3(),
        "blake3:c91b21b6231cb062a5b30678615b82be824bb561de0088a0406bf31563be1221"
    );
    let mut exporter = DigestingStreamExporter::default();

    let completed = runner
        .run_member(
            source("pair-00", ExporterMember::Dynamic, &artifact),
            &mut exporter,
        )
        .expect("fast paired member completes without a thirty-second minimum");

    assert_eq!(
        exporter.repetition_ordinals,
        (0_u64..16).collect::<Vec<_>>()
    );
    assert_eq!(exporter.processed_per_repetition, vec![100_000; 16]);
    assert_eq!(completed.summary().processed_records, 1_600_000);
    assert_eq!(completed.summary().retained_artifact_records, 100_000);
    assert!(completed.summary().active_duration_nanoseconds > 0);
    assert_eq!(completed.summary().repetitions.len(), 16);
    assert_eq!(
        completed.evidence().retained.raw_observable_bytes,
        b"100000:c91b21b6231cb062a5b30678615b82be824bb561de0088a0406bf31563be1221"
    );
    assert_eq!(
        completed.backing_payloads(),
        vec![SelectedBackingPayloadV1::CapturedStream {
            bytes: b"100000:c91b21b6231cb062a5b30678615b82be824bb561de0088a0406bf31563be1221"
                .to_vec(),
        }]
    );
    assert!(
        completed
            .summary()
            .repetitions
            .iter()
            .all(|receipt| receipt.active_duration_ns > 0)
    );

    let receipts: Vec<serde_json::Value> =
        serde_json::from_slice(&completed.evidence().repetition_receipt_bytes)
            .expect("receipt array parses");
    assert_eq!(receipts.len(), 16);
    assert!(
        receipts
            .iter()
            .all(|receipt| receipt.as_object().unwrap().len() == 16)
    );
    let typed: Vec<ExporterRepetitionReceipt> =
        serde_json::from_slice(&completed.evidence().repetition_receipt_bytes)
            .expect("receipt array has the canonical typed schema");
    assert_eq!(typed[0].corpus_blake3, runner.corpus_blake3());
    assert_eq!(
        completed.binding().build_artifact_blake3,
        format!("blake3:{}", blake3::hash(b"fake exporter artifact"))
    );
    assert_eq!(
        completed.binding().build_receipt_blake3,
        format!("blake3:{}", blake3::hash(b"authenticated build receipt"))
    );

    let validated = validate_exporter_member_record(
        &aiperf_bench_tools::plugin_stats::ExporterSampleContract::normative(),
        completed.binding(),
        completed.evidence(),
        completed.record_bytes(),
    )
    .expect("runner-authored member record revalidates");
    assert_eq!(&validated, completed.record());
}

#[test]
fn fast_static_calibration_is_rejected_instead_of_sleeping_or_padding() {
    let artifact = build_artifact();
    let runner = ExporterHarnessRunner::new(stream_policy("static_calibration"))
        .expect("runner builds its fixed corpus");
    let mut exporter = DigestingStreamExporter::default();

    let error = runner
        .run_member(
            source(
                "task1-static-calibration",
                ExporterMember::Static,
                &artifact,
            ),
            &mut exporter,
        )
        .expect_err("active exporter work shorter than thirty seconds is not padded");

    assert!(error.to_string().contains("shorter than 30 seconds"));
    assert_eq!(exporter.repetition_ordinals.len(), 16);
}

#[test]
fn invalid_calibration_coordinates_are_rejected_before_exporter_effects() {
    let artifact = build_artifact();
    let runner = ExporterHarnessRunner::new(stream_policy("static_calibration"))
        .expect("runner builds its fixed corpus");
    let mut exporter = DigestingStreamExporter::default();

    let error = runner
        .run_member(
            source(
                "task1-static-calibration",
                ExporterMember::Dynamic,
                &artifact,
            ),
            &mut exporter,
        )
        .expect_err("calibration is reserved for the original static member");

    assert!(error.to_string().contains("calibration binding is invalid"));
    assert!(exporter.repetition_ordinals.is_empty());
}

#[test]
fn runner_rejects_an_exporter_that_does_not_consume_the_exact_corpus() {
    let artifact = build_artifact();
    let runner = ExporterHarnessRunner::new(stream_policy("paired"))
        .expect("runner builds its fixed corpus");
    let mut exporter = DigestingStreamExporter {
        stop_after: Some(99_999),
        ..DigestingStreamExporter::default()
    };

    let error = runner
        .run_member(
            source("pair-00", ExporterMember::Dynamic, &artifact),
            &mut exporter,
        )
        .expect_err("partial corpus consumption is a product error");

    assert!(error.to_string().contains("exactly 100000"));
    assert_eq!(exporter.repetition_ordinals, vec![0]);
}

#[test]
fn runner_acquires_artifact_tree_and_exact_selected_backing_bytes() {
    let artifact = build_artifact();
    let runner =
        ExporterHarnessRunner::new(artifact_policy()).expect("runner builds its fixed corpus");
    let mut static_exporter = ArtifactExporter {
        lock_digest: "static-lock",
    };

    let static_completed = runner
        .run_member(
            source("pair-00", ExporterMember::Static, &artifact),
            &mut static_exporter,
        )
        .expect("artifact member completes");
    let mut dynamic_exporter = ArtifactExporter {
        lock_digest: "dynamic-lock",
    };
    let dynamic_completed = runner
        .run_member(
            source("pair-00", ExporterMember::Dynamic, &artifact),
            &mut dynamic_exporter,
        )
        .expect("dynamic artifact member completes");

    assert!(
        static_completed
            .evidence()
            .retained
            .raw_observable_bytes
            .windows(b"empty_directory".len())
            .any(|window| window == b"empty_directory")
    );
    assert_eq!(static_completed.backing_payloads().len(), 1);
    assert_eq!(
        static_completed.backing_payloads()[0],
        SelectedBackingPayloadV1::ArtifactContent {
            path: "records.json".to_owned(),
            bytes: b"{\"plugin_lock_digest\" : \"static-lock\", \"other\" : 1.0}".to_vec(),
        }
    );
    assert_ne!(
        static_completed.evidence().retained.raw_observable_bytes,
        static_completed
            .evidence()
            .retained
            .comparison_observable_bytes
    );
    assert_ne!(
        static_completed
            .evidence()
            .retained
            .provenance_receipt_bytes,
        b"[]\n"
    );
    assert_eq!(
        dynamic_completed.backing_payloads(),
        vec![SelectedBackingPayloadV1::ArtifactContent {
            path: "records.json".to_owned(),
            bytes: b"{\"plugin_lock_digest\" : \"dynamic-lock\", \"other\" : 1.0}".to_vec(),
        }]
    );
    let pair = validate_exporter_pair_evidence(
        &aiperf_bench_tools::plugin_stats::ExporterSampleContract::normative(),
        static_completed.binding(),
        static_completed.evidence(),
        dynamic_completed.binding(),
        dynamic_completed.evidence(),
    )
    .expect("policy-authorized provenance is the only cross-member difference");
    assert_eq!(
        pair.static_member.comparison_observable_blake3,
        pair.dynamic_member.comparison_observable_blake3
    );
}

#[test]
fn receiver_records_exact_body_before_returning_acknowledgement() {
    let artifact = build_artifact();
    let policy = receiver_policy("otel_http_v1");
    let protocol = policy
        .authenticate_receiver_protocol("otel_http_v1")
        .expect("controller authenticates the receiver protocol");
    let runner = ExporterHarnessRunner::new(policy).expect("runner builds its fixed corpus");
    let mut exporter = ReceiverExporter::default();

    let completed = runner
        .run_member(
            receiver_source(
                "pair-00",
                ExporterMember::Static,
                &artifact,
                Some(&protocol),
            ),
            &mut exporter,
        )
        .expect("receiver member completes");

    assert_eq!(completed.receiver_protocol(), Some("otel_http_v1"));
    assert_eq!(exporter.acknowledgements, vec![(0, 1); 16]);
    assert_eq!(completed.backing_payloads().len(), 1);
    assert_eq!(
        completed.backing_payloads()[0],
        SelectedBackingPayloadV1::TranscriptBody {
            sequence: 0,
            bytes: b"abcXYZ".to_vec(),
        }
    );
    assert!(
        completed
            .evidence()
            .retained
            .raw_observable_bytes
            .windows(b"application/json".len())
            .any(|window| window == b"application/json")
    );
}

#[test]
fn receiver_protocol_identity_is_controller_bound_and_mismatches_fail_before_export() {
    let artifact = build_artifact();
    let policy = receiver_policy("otel_http_v1");
    let protocol = policy
        .authenticate_receiver_protocol("otel_http_v1")
        .expect("controller authenticates the receiver protocol");
    let other_policy = receiver_policy("zipkin_http_v1");
    let mismatched = other_policy
        .authenticate_receiver_protocol("zipkin_http_v1")
        .expect("other controller authenticates its protocol");
    let runner = ExporterHarnessRunner::new(policy).expect("runner binds its policy");

    for supplied in [None, Some(&mismatched)] {
        let mut exporter = ReceiverExporter::default();
        let error = runner
            .run_member(
                receiver_source("pair-00", ExporterMember::Static, &artifact, supplied),
                &mut exporter,
            )
            .expect_err("missing or cross-policy receiver identity fails");
        assert!(error.to_string().contains("receiver protocol"));
        assert!(exporter.acknowledgements.is_empty());
    }

    let mut exporter = ReceiverExporter::default();
    let completed = runner
        .run_member(
            receiver_source(
                "pair-00",
                ExporterMember::Static,
                &artifact,
                Some(&protocol),
            ),
            &mut exporter,
        )
        .expect("matching receiver identity completes");
    assert_eq!(completed.receiver_protocol(), Some("otel_http_v1"));
    assert_eq!(
        completed.receiver_protocol_authority_blake3(),
        Some(protocol.authority_blake3())
    );
}

#[test]
fn receiver_removes_only_bound_transport_fields_and_refuses_noncanonical_metadata() {
    let artifact = build_artifact();
    let policy = receiver_policy("otel_http_v1");
    let protocol = policy
        .authenticate_receiver_protocol("otel_http_v1")
        .expect("controller authenticates the receiver protocol");
    let runner = ExporterHarnessRunner::new(policy).expect("runner binds its policy");
    let mut exporter = ReceiverExporter::default();
    let completed = runner
        .run_member(
            receiver_source(
                "pair-00",
                ExporterMember::Static,
                &artifact,
                Some(&protocol),
            ),
            &mut exporter,
        )
        .expect("canonical receiver metadata completes");
    let retained = &completed.evidence().retained.raw_observable_bytes;
    let entries = parse_receiver_transcript_observable(retained, false)
        .expect("harness forms a canonical transcript");
    assert!(entries.iter().all(|entry| {
        entry.metadata
            == [
                ["content-type".to_owned(), "application/json".to_owned()],
                ["x-request-id".to_owned(), "retained".to_owned()],
            ]
    }));
    assert_eq!(
        completed.summary().repetitions[0].raw_observable_blake3,
        format!("blake3:{}", blake3::hash(retained))
    );

    for metadata in [
        vec![
            ["x-request-id".to_owned(), "retained".to_owned()],
            ["content-type".to_owned(), "application/json".to_owned()],
        ],
        vec![
            ["date".to_owned(), "first".to_owned()],
            ["date".to_owned(), "second".to_owned()],
        ],
    ] {
        let mut exporter = ReceiverExporter {
            acknowledgements: Vec::new(),
            metadata,
        };
        let error = runner
            .run_member(
                receiver_source(
                    "pair-01",
                    ExporterMember::Static,
                    &artifact,
                    Some(&protocol),
                ),
                &mut exporter,
            )
            .expect_err("unsorted or duplicate metadata fails closed");
        assert!(
            error.to_string().contains("sorted, and unique"),
            "unexpected receiver failure: {error}"
        );
        assert!(exporter.acknowledgements.is_empty());
    }
}
