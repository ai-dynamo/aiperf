// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed policy vectors for exporter-observable provenance authority.

use std::collections::BTreeSet;

use aiperf_bench_tools::exporter_policy::{
    ComparisonReplacementV1, ComparisonSelectionV1, ProvenanceBindingV1, ProvenanceObservationV1,
    build_comparison_payload_v1, generate_provenance_receipt_v1, parse_exporter_observable_policy,
    validate_provenance_receipt_v1,
};
use aiperf_bench_tools::plugin_stats::ExporterMember;

const TASK1_POLICY: &[u8] = include_bytes!("../../benchmarks/exporter-observable-policy.json");
const TASK1_POLICY_DIGEST: &str =
    "blake3:98d991e68d29ba0368856cab0773d77f17edcd1b2e9b1e39f8a16165bc84d0d7";
const DIGEST: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EMPTY_PROVENANCE_DIGEST: &str =
    "blake3:9fa8dc9570625be2be53d308f958332981ec8fb8137d3dd7ba0ae5da317eaa7d";
const STATIC_PROVENANCE_RECEIPT: &[u8] = b"[{\"attempt_ordinal\":2,\"expected\":{\"encoding\":\"canonical_json\",\"value\":\"static-lock\"},\"experiment_identity_blake3\":\"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"locator\":{\"kind\":\"json_pointer\",\"pointer\":\"/plugin_lock_digest\"},\"member\":\"static\",\"observed_raw_hex\":\"227374617469632d6c6f636b22\",\"observed_value\":{\"encoding\":\"canonical_json\",\"value\":\"static-lock\"},\"output_selector\":{\"kind\":\"artifact_content\",\"path\":\"records.json\"},\"pair_id\":\"pair-7\",\"policy_mode\":\"paired\",\"repetition_ordinal\":3,\"replacement\":{\"encoding\":\"canonical_json\",\"value\":\"@plugin-lock@\"},\"scenario_id\":\"exporter\",\"schema_version\":1,\"slot_id\":\"plugin_lock\"}]\n";
const STATIC_PROVENANCE_DIGEST: &str =
    "blake3:7a4c8500a57a21a58c423a265c1ac55026a4b2beb780a18bc291fd9a626ada56";

fn canonical_policy(value: serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json_canonicalizer::to_vec(&value).expect("fixture canonicalizes");
    bytes.push(b'\n');
    bytes
}

fn base_slot() -> serde_json::Value {
    serde_json::json!({
        "locator": {"kind": "json_pointer", "pointer": "/plugin_lock_digest"},
        "output_selector": {"kind": "artifact_content", "path": "records.json"},
        "replacement": {"encoding": "canonical_json", "value": "@plugin-lock@"},
        "slot_id": "plugin_lock",
        "static_expected": {"encoding": "canonical_json", "value": "static-lock"},
    })
}

fn base_policy() -> serde_json::Value {
    serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "artifact_tree",
            "provenance_slots": [base_slot()],
            "scenario_id": "exporter",
        }],
        "schema_version": 1,
    })
}

fn paired_policy() -> aiperf_bench_tools::exporter_policy::ExporterObservablePolicyV1 {
    let mut value = base_policy();
    value["mode"] = serde_json::json!("paired");
    value["scenarios"][0]["provenance_slots"][0]["dynamic_expected"] =
        serde_json::json!({"encoding": "canonical_json", "value": "dynamic-lock"});
    parse_exporter_observable_policy(&canonical_policy(value), &BTreeSet::new())
        .expect("paired policy fixture validates")
}

fn provenance_binding(member: ExporterMember) -> ProvenanceBindingV1 {
    ProvenanceBindingV1 {
        experiment_identity_blake3: DIGEST.to_owned(),
        attempt_ordinal: 2,
        scenario_id: "exporter".to_owned(),
        pair_id: "pair-7".to_owned(),
        member,
        repetition_ordinal: 3,
    }
}

fn assert_rejected(value: serde_json::Value) {
    let error = parse_exporter_observable_policy(&canonical_policy(value), &BTreeSet::new())
        .expect_err("invalid policy must be rejected");
    assert!(!error.to_string().is_empty());
}

#[test]
fn parses_only_exact_jcs_and_pins_policy_digest() {
    let policy = parse_exporter_observable_policy(TASK1_POLICY, &BTreeSet::new())
        .expect("checked-in Task-1 policy validates");

    assert_eq!(
        policy.canonical_bytes().expect("policy canonicalizes"),
        TASK1_POLICY
    );
    assert_eq!(
        policy.canonical_blake3().expect("policy hashes"),
        TASK1_POLICY_DIGEST
    );

    for malformed in [
        b"{\"mode\":\"static_calibration\",\"mode\":\"paired\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n".as_slice(),
        b"{\"extra\":0,\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n".as_slice(),
        b"{\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n".as_slice(),
        b"{\"mode\": \"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n".as_slice(),
        b"{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}".as_slice(),
    ] {
        assert!(
            parse_exporter_observable_policy(malformed, &BTreeSet::new()).is_err(),
            "malformed policy was accepted: {}",
            String::from_utf8_lossy(malformed)
        );
    }
}

#[test]
fn enforces_mode_sorted_identity_and_transport_authority_rules() {
    let mut paired = base_policy();
    paired["mode"] = serde_json::json!("paired");
    assert_rejected(paired.clone());
    paired["scenarios"][0]["provenance_slots"][0]["dynamic_expected"] =
        serde_json::json!({"encoding": "canonical_json", "value": "dynamic-lock"});
    parse_exporter_observable_policy(&canonical_policy(paired), &BTreeSet::new())
        .expect("paired slot with both expected members validates");

    let mut static_with_dynamic = base_policy();
    static_with_dynamic["scenarios"][0]["provenance_slots"][0]["dynamic_expected"] =
        serde_json::json!({"encoding": "canonical_json", "value": "dynamic-lock"});
    assert_rejected(static_with_dynamic);

    let mut invalid_identifier = base_policy();
    invalid_identifier["scenarios"][0]["scenario_id"] = serde_json::json!("Upper");
    assert_rejected(invalid_identifier);

    let mut unordered_scenarios = base_policy();
    let mut first = unordered_scenarios["scenarios"][0].clone();
    first["scenario_id"] = serde_json::json!("z");
    let mut second = first.clone();
    second["scenario_id"] = serde_json::json!("a");
    unordered_scenarios["scenarios"] = serde_json::json!([first, second]);
    assert_rejected(unordered_scenarios);

    let transport_policy = serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [{
            "keys": ["date", "x-request-id"],
            "protocol": "otel_http_v1",
        }],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "receiver_transcript",
            "provenance_slots": [],
            "scenario_id": "otel",
        }],
        "schema_version": 1,
    });
    assert!(
        parse_exporter_observable_policy(
            &canonical_policy(transport_policy.clone()),
            &BTreeSet::new()
        )
        .is_err()
    );
    parse_exporter_observable_policy(
        &canonical_policy(transport_policy),
        &BTreeSet::from(["otel_http_v1".to_owned()]),
    )
    .expect("used authenticated receiver transport rule validates");

    let mut unused_transport = base_policy();
    unused_transport["receiver_transport_fields_removed"] = serde_json::json!([{
        "keys": ["date"], "protocol": "otel_http_v1"
    }]);
    let error = parse_exporter_observable_policy(
        &canonical_policy(unused_transport),
        &BTreeSet::from(["otel_http_v1".to_owned()]),
    )
    .expect_err("transport rule unused by every receiver scenario must fail");
    assert!(error.to_string().contains("unused"));

    let mut unordered_keys = serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [{
            "keys": ["z", "a"], "protocol": "otel_http_v1"
        }],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "receiver_transcript",
            "provenance_slots": [],
            "scenario_id": "otel",
        }],
        "schema_version": 1,
    });
    assert!(
        parse_exporter_observable_policy(
            &canonical_policy(unordered_keys.clone()),
            &BTreeSet::from(["otel_http_v1".to_owned()]),
        )
        .is_err()
    );
    unordered_keys["receiver_transport_fields_removed"][0]["keys"] =
        serde_json::json!(["date", "date"]);
    assert!(
        parse_exporter_observable_policy(
            &canonical_policy(unordered_keys),
            &BTreeSet::from(["otel_http_v1".to_owned()]),
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_selectors_locators_encodings_and_shape_decidable_overlap() {
    let mut wrong_class = base_policy();
    wrong_class["scenarios"][0]["provenance_slots"][0]["output_selector"] =
        serde_json::json!({"kind": "captured_stream"});
    assert_rejected(wrong_class);

    let mut bad_path = base_policy();
    bad_path["scenarios"][0]["provenance_slots"][0]["output_selector"]["path"] =
        serde_json::json!("../records.json");
    assert_rejected(bad_path);

    for pointer in ["not-rooted", "/bad~2escape", "/trailing~"] {
        let mut invalid_pointer = base_policy();
        invalid_pointer["scenarios"][0]["provenance_slots"][0]["locator"]["pointer"] =
            serde_json::json!(pointer);
        assert_rejected(invalid_pointer);
    }

    let mut empty_range = base_policy();
    empty_range["scenarios"][0]["provenance_slots"][0]["locator"] =
        serde_json::json!({"kind": "byte_range", "length": 0, "offset": 0});
    empty_range["scenarios"][0]["provenance_slots"][0]["static_expected"] =
        serde_json::json!({"encoding": "hex_bytes", "value": ""});
    empty_range["scenarios"][0]["provenance_slots"][0]["replacement"] =
        serde_json::json!({"encoding": "hex_bytes", "value": ""});
    assert_rejected(empty_range);

    let mut bad_hex = base_policy();
    bad_hex["scenarios"][0]["provenance_slots"][0]["locator"] =
        serde_json::json!({"kind": "whole_output"});
    bad_hex["scenarios"][0]["provenance_slots"][0]["static_expected"] =
        serde_json::json!({"encoding": "hex_bytes", "value": "0A"});
    bad_hex["scenarios"][0]["provenance_slots"][0]["replacement"] =
        serde_json::json!({"encoding": "hex_bytes", "value": "00"});
    assert_rejected(bad_hex);

    let mut encoding_mismatch = base_policy();
    encoding_mismatch["scenarios"][0]["provenance_slots"][0]["replacement"] =
        serde_json::json!({"encoding": "hex_bytes", "value": "00"});
    assert_rejected(encoding_mismatch);

    let mut overlapping = base_policy();
    let mut ancestor = base_slot();
    ancestor["locator"]["pointer"] = serde_json::json!("/a");
    ancestor["slot_id"] = serde_json::json!("a");
    let mut descendant = base_slot();
    descendant["locator"]["pointer"] = serde_json::json!("/a/b");
    descendant["slot_id"] = serde_json::json!("b");
    overlapping["scenarios"][0]["provenance_slots"] = serde_json::json!([ancestor, descendant]);
    assert_rejected(overlapping);

    let mut duplicate_locator = base_policy();
    let mut first = base_slot();
    first["slot_id"] = serde_json::json!("a");
    let mut second = base_slot();
    second["slot_id"] = serde_json::json!("b");
    duplicate_locator["scenarios"][0]["provenance_slots"] = serde_json::json!([first, second]);
    assert_rejected(duplicate_locator);
}

#[test]
fn comparison_payload_uses_the_literal_v1_frame_grammar() {
    let selections = vec![
        ComparisonSelectionV1 {
            slot_id: "second".to_owned(),
            offset: 4,
            length: 1,
            replacement: ComparisonReplacementV1::CanonicalJson(serde_json::json!("x")),
        },
        ComparisonSelectionV1 {
            slot_id: "first".to_owned(),
            offset: 1,
            length: 2,
            replacement: ComparisonReplacementV1::HexBytes("ff".to_owned()),
        },
    ];
    let expected = b"AIPERF_EXPORTER_COMPARISON_V1\0\
\x00\x00\x00\x00\x00\x00\x00\x00\x01a\
\x01\x00\x00\x00\x00\x00\x00\x00\x05first\x00\x00\x00\x00\x00\x00\x00\x01\xff\
\x00\x00\x00\x00\x00\x00\x00\x00\x01d\
\x01\x00\x00\x00\x00\x00\x00\x00\x06second\x00\x00\x00\x00\x00\x00\x00\x03\"x\"\
\x00\x00\x00\x00\x00\x00\x00\x00\x01f\xff";

    assert_eq!(
        build_comparison_payload_v1(b"abcdef", &selections).expect("disjoint selections transform"),
        expected
    );

    let mut overlapping = selections.clone();
    overlapping[0].offset = 2;
    assert!(build_comparison_payload_v1(b"abcdef", &overlapping).is_err());
    let mut out_of_range = selections.clone();
    out_of_range[0].offset = 6;
    assert!(build_comparison_payload_v1(b"abcdef", &out_of_range).is_err());
    let mut invalid_hex = selections;
    invalid_hex[1].replacement = ComparisonReplacementV1::HexBytes("AA".to_owned());
    assert!(build_comparison_payload_v1(b"abcdef", &invalid_hex).is_err());
}

#[test]
fn provenance_receipt_pins_literal_jcs_digest_and_rejects_duplicate_keys() {
    let policy = paired_policy();
    let binding = provenance_binding(ExporterMember::Static);
    let observations = [ProvenanceObservationV1 {
        slot_id: "plugin_lock".to_owned(),
        observed_raw: br#""static-lock""#.to_vec(),
    }];

    let receipt = generate_provenance_receipt_v1(&policy, &binding, &observations)
        .expect("static member observation matches policy");
    assert_eq!(receipt.bytes, STATIC_PROVENANCE_RECEIPT);
    assert_eq!(receipt.blake3, STATIC_PROVENANCE_DIGEST);
    validate_provenance_receipt_v1(STATIC_PROVENANCE_RECEIPT, &policy, &binding, &observations)
        .expect("literal receipt validates against supplied evidence");

    let duplicate = String::from_utf8(STATIC_PROVENANCE_RECEIPT.to_vec())
        .expect("receipt is UTF-8")
        .replace(
            "\"slot_id\":\"plugin_lock\"",
            "\"slot_id\":\"plugin_lock\",\"slot_id\":\"plugin_lock\"",
        );
    assert!(
        validate_provenance_receipt_v1(duplicate.as_bytes(), &policy, &binding, &observations,)
            .is_err()
    );
}

#[test]
fn provenance_expected_mapping_covers_dynamic_calibration_and_empty_slots() {
    let paired = paired_policy();
    let dynamic = generate_provenance_receipt_v1(
        &paired,
        &provenance_binding(ExporterMember::Dynamic),
        &[ProvenanceObservationV1 {
            slot_id: "plugin_lock".to_owned(),
            observed_raw: br#""dynamic-lock""#.to_vec(),
        }],
    )
    .expect("dynamic member uses dynamic_expected");
    let decoded: serde_json::Value =
        serde_json::from_slice(&dynamic.bytes).expect("generated receipt is JSON");
    assert_eq!(
        decoded.pointer("/0/expected/value"),
        Some(&serde_json::json!("dynamic-lock"))
    );

    let static_policy =
        parse_exporter_observable_policy(&canonical_policy(base_policy()), &BTreeSet::new())
            .expect("calibration policy validates");
    generate_provenance_receipt_v1(
        &static_policy,
        &provenance_binding(ExporterMember::Static),
        &[ProvenanceObservationV1 {
            slot_id: "plugin_lock".to_owned(),
            observed_raw: br#""static-lock""#.to_vec(),
        }],
    )
    .expect("calibration member uses static_expected");
    assert!(
        generate_provenance_receipt_v1(
            &static_policy,
            &provenance_binding(ExporterMember::Dynamic),
            &[ProvenanceObservationV1 {
                slot_id: "plugin_lock".to_owned(),
                observed_raw: br#""static-lock""#.to_vec(),
            }],
        )
        .is_err()
    );

    let empty_policy = parse_exporter_observable_policy(TASK1_POLICY, &BTreeSet::new())
        .expect("empty-slot policy validates");
    let mut empty_binding = provenance_binding(ExporterMember::Static);
    empty_binding.scenario_id = "exporter_100k".to_owned();
    let empty = generate_provenance_receipt_v1(&empty_policy, &empty_binding, &[])
        .expect("empty policy emits empty receipt");
    assert_eq!(empty.bytes, b"[]\n");
    assert_eq!(empty.blake3, EMPTY_PROVENANCE_DIGEST);
}

#[test]
fn provenance_byte_range_requires_the_exact_declared_span_length() {
    let policy_bytes = canonical_policy(serde_json::json!({
        "mode": "static_calibration",
        "receiver_transport_fields_removed": [],
        "scenarios": [{
            "allows_empty": false,
            "observable_kind": "captured_stream",
            "provenance_slots": [{
                "locator": {"kind": "byte_range", "length": 2, "offset": 0},
                "output_selector": {"kind": "captured_stream"},
                "replacement": {"encoding": "hex_bytes", "value": "00"},
                "slot_id": "bytes",
                "static_expected": {"encoding": "hex_bytes", "value": "ab"},
            }],
            "scenario_id": "exporter",
        }],
        "schema_version": 1,
    }));
    let policy = parse_exporter_observable_policy(&policy_bytes, &BTreeSet::new())
        .expect("byte-range policy validates structurally");

    assert!(
        generate_provenance_receipt_v1(
            &policy,
            &provenance_binding(ExporterMember::Static),
            &[ProvenanceObservationV1 {
                slot_id: "bytes".to_owned(),
                observed_raw: vec![0xab],
            }],
        )
        .is_err()
    );
}
