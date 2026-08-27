// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end fixed vectors for applying exporter-observable policy slots.

use std::collections::BTreeSet;

use aiperf_bench_tools::exporter_policy::{
    ProvenanceBindingV1, SelectedBackingPayloadV1, apply_exporter_observable_policy_v1,
    parse_exporter_observable_policy,
};
use aiperf_bench_tools::plugin_stats::ExporterMember;

const DIGEST: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EMPTY_JSON_DIGEST: &str =
    "blake3:9fa8dc9570625be2be53d308f958332981ec8fb8137d3dd7ba0ae5da317eaa7d";
const EMPTY_BYTES_DIGEST: &str =
    "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
const ARTIFACT_PAYLOAD: &[u8] = b"{\"plugin_lock_digest\" : \"static-lock\", \"other\" : 1.0}";
const ARTIFACT_RAW: &[u8] = b"[{\"blake3\":\"blake3:b2e0b1550c11802fb38eee05e94eda5c97d6e35b191391d6301d1527f1482c19\",\"kind\":\"regular_file\",\"length\":53,\"path\":\"records.json\"}]\n";
const ARTIFACT_COMPARISON: &[u8] = b"[{\"blake3\":\"blake3:8aeb3c90c029057217e47fdd004c5b3e4e456f94c72383bc1e8382ec1e8bd1f4\",\"kind\":\"regular_file\",\"length\":132,\"path\":\"records.json\"}]\n";
const RECEIVER_BODY: &[u8] = b"abcXYZ";
const RECEIVER_RAW: &[u8] = b"[{\"body\":{\"blake3\":\"blake3:e3a6f28f4b37f16f261a87dff0951f3ee92a507047c26f01da59027d343f154d\",\"encoding\":\"bytes\",\"length\":6},\"metadata\":[],\"operation\":\"POST\",\"sequence\":0,\"target\":\"/v1/traces\"}]\n";
const RECEIVER_COMPARISON: &[u8] = b"[{\"body\":{\"blake3\":\"blake3:eb87064da0369c3633da80fbef7483d153eb94634ba611483cdcc210a26c7eed\",\"encoding\":\"bytes\",\"length\":66},\"metadata\":[],\"operation\":\"POST\",\"sequence\":0,\"target\":\"/v1/traces\"}]\n";

fn canonical_policy(value: serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json_canonicalizer::to_vec(&value).expect("fixture canonicalizes");
    bytes.push(b'\n');
    bytes
}

fn policy(
    value: serde_json::Value,
) -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    parse_exporter_observable_policy(&canonical_policy(value), &BTreeSet::new())
        .expect("application policy fixture validates")
}

fn binding(scenario_id: &str) -> ProvenanceBindingV1 {
    ProvenanceBindingV1 {
        experiment_identity_blake3: DIGEST.to_owned(),
        attempt_ordinal: 0,
        scenario_id: scenario_id.to_owned(),
        pair_id: "pair-1".to_owned(),
        member: ExporterMember::Static,
        repetition_ordinal: 0,
    }
}

fn artifact_policy() -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    policy(serde_json::json!({
        "mode": "static_calibration",
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
            }],
            "scenario_id": "artifact",
        }],
        "schema_version": 1,
    }))
}

fn stream_policy() -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "captured_stream",
            "provenance_slots": [{
                "locator": {"kind": "whole_output"},
                "output_selector": {"kind": "captured_stream"},
                "replacement": {"encoding": "hex_bytes", "value": "00"},
                "slot_id": "stream",
                "static_expected": {"encoding": "hex_bytes", "value": "616263"},
            }],
            "scenario_id": "stream",
        }],
        "schema_version": 1,
    }))
}

fn receiver_policy(
    sequence: u64,
) -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "receiver_transcript",
            "provenance_slots": [{
                "locator": {"kind": "byte_range", "length": 3, "offset": 3},
                "output_selector": {"kind": "transcript_body", "sequence": sequence},
                "replacement": {"encoding": "hex_bytes", "value": "00ff"},
                "slot_id": "tail",
                "static_expected": {"encoding": "hex_bytes", "value": "58595a"},
            }],
            "scenario_id": "receiver",
        }],
        "schema_version": 1,
    }))
}

fn decode_hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("hex fixture is UTF-8");
            u8::from_str_radix(text, 16).expect("hex fixture is valid")
        })
        .collect()
}

#[test]
fn applies_json_pointer_to_artifact_content_without_canonicalizing_unlisted_bytes() {
    let applied = apply_exporter_observable_policy_v1(
        &artifact_policy(),
        &binding("artifact"),
        ARTIFACT_RAW,
        &[SelectedBackingPayloadV1::ArtifactContent {
            path: "records.json".to_owned(),
            bytes: ARTIFACT_PAYLOAD.to_vec(),
        }],
    )
    .expect("artifact policy applies");

    assert_eq!(
        applied.raw_observable_blake3,
        "blake3:5d62d8d51ccbbc1e6fdf9963c124814ffff64360372367ff150abf19b61664c6"
    );
    assert_eq!(applied.comparison_bytes, ARTIFACT_COMPARISON);
    assert_eq!(
        applied.comparison_observable_blake3,
        "blake3:657c08ab03bbbb294c1b9b69f0745ceb8c625cca7a1e25afc16eee4f665df7ae"
    );
    assert_eq!(
        applied.provenance_receipt_blake3,
        "blake3:e1153b17e6ef758e60a5cb8e1d3be024d96f7cd4b94f8a7ef0bd00c7a1ecbb8c"
    );
    assert!(
        applied
            .provenance_receipt_bytes
            .windows(26)
            .any(|window| window == b"227374617469632d6c6f636b22")
    );

    let expected_payload = decode_hex(
        "4149504552465f4558504f525445525f434f4d50415249534f4e5f5631000000000000000000187b22706c7567696e5f6c6f636b5f64696765737422203a2001000000000000000b706c7567696e5f6c6f636b000000000000000f2240706c7567696e2d6c6f636b40220000000000000000102c20226f7468657222203a20312e307dff",
    );
    assert_eq!(expected_payload.len(), 132);
    assert_eq!(
        format!("blake3:{}", blake3::hash(&expected_payload)),
        "blake3:8aeb3c90c029057217e47fdd004c5b3e4e456f94c72383bc1e8382ec1e8bd1f4"
    );
}

#[test]
fn applies_whole_output_to_captured_stream_with_literal_frames() {
    let applied = apply_exporter_observable_policy_v1(
        &stream_policy(),
        &binding("stream"),
        b"abc",
        &[SelectedBackingPayloadV1::CapturedStream {
            bytes: b"abc".to_vec(),
        }],
    )
    .expect("captured-stream policy applies");
    let expected = decode_hex(
        "4149504552465f4558504f525445525f434f4d50415249534f4e5f56310001000000000000000673747265616d000000000000000100ff",
    );

    assert_eq!(applied.comparison_bytes, expected);
    assert_eq!(
        applied.comparison_observable_blake3,
        "blake3:29dd7f4deddbf8c7fa843c349b7264c22b63f528b1ff00a4ddcab5ec847938bd"
    );
    assert_eq!(
        applied.provenance_receipt_blake3,
        "blake3:faa01f08c9827bfd87f2b9675037dbd4ab0d7c79f984f04d6e69c0dc85bd04ec"
    );
}

#[test]
fn applies_byte_range_to_receiver_body_and_rebuilds_only_body_identity() {
    let applied = apply_exporter_observable_policy_v1(
        &receiver_policy(0),
        &binding("receiver"),
        RECEIVER_RAW,
        &[SelectedBackingPayloadV1::TranscriptBody {
            sequence: 0,
            bytes: RECEIVER_BODY.to_vec(),
        }],
    )
    .expect("receiver policy applies");

    assert_eq!(applied.comparison_bytes, RECEIVER_COMPARISON);
    assert_eq!(
        applied.comparison_observable_blake3,
        "blake3:fa73c770fd19c9b9fa3c19c6b084cf1a463c700e73a9add86362b25b7c198379"
    );
    assert_eq!(
        applied.provenance_receipt_blake3,
        "blake3:63e18d3e334646afb9d016f3a9d61717c01824020fc2ac99c9f8acfe5e0b5318"
    );
}

#[test]
fn empty_slot_policies_preserve_raw_observables_for_all_classes() {
    for (scenario_id, kind, allows_empty, raw, expected_digest) in [
        (
            "artifact",
            "artifact_tree",
            false,
            b"[]\n".as_slice(),
            EMPTY_JSON_DIGEST,
        ),
        (
            "stream",
            "captured_stream",
            true,
            b"".as_slice(),
            EMPTY_BYTES_DIGEST,
        ),
        (
            "receiver",
            "receiver_transcript",
            true,
            b"[]\n".as_slice(),
            EMPTY_JSON_DIGEST,
        ),
    ] {
        let empty_policy = policy(serde_json::json!({
            "mode": "static_calibration",
            "receiver_transport_fields_removed": [],
            "scenarios": [{
                "allows_empty": allows_empty,
                "observable_kind": kind,
                "provenance_slots": [],
                "scenario_id": scenario_id,
            }],
            "schema_version": 1,
        }));
        let applied =
            apply_exporter_observable_policy_v1(&empty_policy, &binding(scenario_id), raw, &[])
                .expect("empty policy preserves its raw observable");

        assert_eq!(applied.comparison_bytes, raw);
        assert_eq!(applied.raw_observable_blake3, expected_digest);
        assert_eq!(applied.comparison_observable_blake3, expected_digest);
        assert_eq!(applied.provenance_receipt_bytes, b"[]\n");
        assert_eq!(applied.provenance_receipt_blake3, EMPTY_JSON_DIGEST);
    }
}

#[test]
fn whole_output_can_select_an_allowed_empty_captured_stream() {
    let empty_stream_policy = policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": true,
            "observable_kind": "captured_stream",
            "provenance_slots": [{
                "locator": {"kind": "whole_output"},
                "output_selector": {"kind": "captured_stream"},
                "replacement": {"encoding": "hex_bytes", "value": "00"},
                "slot_id": "empty",
                "static_expected": {"encoding": "hex_bytes", "value": ""},
            }],
            "scenario_id": "empty_stream",
        }],
        "schema_version": 1,
    }));
    let applied = apply_exporter_observable_policy_v1(
        &empty_stream_policy,
        &binding("empty_stream"),
        b"",
        &[SelectedBackingPayloadV1::CapturedStream { bytes: Vec::new() }],
    )
    .expect("whole_output may select an allowed empty stream");

    assert_eq!(
        applied.comparison_bytes,
        decode_hex(
            "4149504552465f4558504f525445525f434f4d50415249534f4e5f563100010000000000000005656d707479000000000000000100ff"
        )
    );
}

#[test]
fn rejects_missing_ambiguous_wrong_class_and_identity_mismatched_backing_payloads() {
    assert!(
        apply_exporter_observable_policy_v1(
            &artifact_policy(),
            &binding("artifact"),
            ARTIFACT_RAW,
            &[],
        )
        .is_err()
    );
    assert!(
        apply_exporter_observable_policy_v1(
            &artifact_policy(),
            &binding("artifact"),
            ARTIFACT_RAW,
            &[
                SelectedBackingPayloadV1::ArtifactContent {
                    path: "records.json".to_owned(),
                    bytes: ARTIFACT_PAYLOAD.to_vec(),
                },
                SelectedBackingPayloadV1::ArtifactContent {
                    path: "records.json".to_owned(),
                    bytes: ARTIFACT_PAYLOAD.to_vec(),
                },
            ],
        )
        .is_err()
    );
    assert!(
        apply_exporter_observable_policy_v1(
            &artifact_policy(),
            &binding("artifact"),
            ARTIFACT_RAW,
            &[SelectedBackingPayloadV1::CapturedStream {
                bytes: ARTIFACT_PAYLOAD.to_vec(),
            }],
        )
        .is_err()
    );
    let mut mutated = ARTIFACT_PAYLOAD.to_vec();
    let last = mutated.len() - 2;
    mutated[last] = b'2';
    assert!(
        apply_exporter_observable_policy_v1(
            &artifact_policy(),
            &binding("artifact"),
            ARTIFACT_RAW,
            &[SelectedBackingPayloadV1::ArtifactContent {
                path: "records.json".to_owned(),
                bytes: mutated,
            }],
        )
        .is_err()
    );
}

#[test]
fn rejects_out_of_range_receiver_and_application_time_locator_overlap() {
    assert!(
        apply_exporter_observable_policy_v1(
            &receiver_policy(1),
            &binding("receiver"),
            RECEIVER_RAW,
            &[SelectedBackingPayloadV1::TranscriptBody {
                sequence: 1,
                bytes: RECEIVER_BODY.to_vec(),
            }],
        )
        .is_err()
    );

    let overlap_policy = policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "captured_stream",
            "provenance_slots": [
                {
                    "locator": {"kind": "byte_range", "length": 5, "offset": 5},
                    "output_selector": {"kind": "captured_stream"},
                    "replacement": {"encoding": "hex_bytes", "value": "00"},
                    "slot_id": "a_bytes",
                    "static_expected": {"encoding": "hex_bytes", "value": "2261626322"},
                },
                {
                    "locator": {"kind": "json_pointer", "pointer": "/x"},
                    "output_selector": {"kind": "captured_stream"},
                    "replacement": {"encoding": "canonical_json", "value": "replacement"},
                    "slot_id": "b_json",
                    "static_expected": {"encoding": "canonical_json", "value": "abc"},
                },
            ],
            "scenario_id": "overlap",
        }],
        "schema_version": 1,
    }));
    let raw = br#"{"x":"abc"}"#;
    assert!(
        apply_exporter_observable_policy_v1(
            &overlap_policy,
            &binding("overlap"),
            raw,
            &[SelectedBackingPayloadV1::CapturedStream {
                bytes: raw.to_vec()
            }],
        )
        .is_err()
    );
}

#[test]
fn rejects_duplicate_key_pointer_payload_and_preserves_one_byte_unlisted_mutations() {
    let pointer_policy = policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "captured_stream",
            "provenance_slots": [{
                "locator": {"kind": "json_pointer", "pointer": "/x"},
                "output_selector": {"kind": "captured_stream"},
                "replacement": {"encoding": "canonical_json", "value": "replacement"},
                "slot_id": "x",
                "static_expected": {"encoding": "canonical_json", "value": "static"},
            }],
            "scenario_id": "pointer",
        }],
        "schema_version": 1,
    }));
    let duplicate = br#"{"x":"static","x":"static"}"#;
    assert!(
        apply_exporter_observable_policy_v1(
            &pointer_policy,
            &binding("pointer"),
            duplicate,
            &[SelectedBackingPayloadV1::CapturedStream {
                bytes: duplicate.to_vec()
            }],
        )
        .is_err()
    );

    let first = apply_exporter_observable_policy_v1(
        &artifact_policy(),
        &binding("artifact"),
        ARTIFACT_RAW,
        &[SelectedBackingPayloadV1::ArtifactContent {
            path: "records.json".to_owned(),
            bytes: ARTIFACT_PAYLOAD.to_vec(),
        }],
    )
    .expect("first unlisted-byte vector applies");
    let mutated_payload = b"{\"plugin_lock_digest\" : \"static-lock\", \"other\" : 2.0}";
    let mutated_manifest = canonical_policy(serde_json::json!([{
        "blake3": format!("blake3:{}", blake3::hash(mutated_payload)),
        "kind": "regular_file",
        "length": mutated_payload.len(),
        "path": "records.json",
    }]));
    let second = apply_exporter_observable_policy_v1(
        &artifact_policy(),
        &binding("artifact"),
        &mutated_manifest,
        &[SelectedBackingPayloadV1::ArtifactContent {
            path: "records.json".to_owned(),
            bytes: mutated_payload.to_vec(),
        }],
    )
    .expect("one-byte unlisted mutation remains observable");

    assert_ne!(first.comparison_bytes, second.comparison_bytes);
    assert_ne!(
        first.comparison_observable_blake3,
        second.comparison_observable_blake3
    );
}
