// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for `artifact_section.rs`: encode/decode round-trips.
//!
//! We test the pure encode/decode path without invoking `objcopy` so that
//! the tests run on the rig without requiring external tools in this suite.

use aiperf_plugin_sdk::artifact_section::{decode_section, encode_section};
use aiperf_plugin_sdk::identity::{NativeDep, PluginArtifactBuildRecordV1};

fn make_build_record() -> PluginArtifactBuildRecordV1 {
    let universe_digest = "a".repeat(64);
    let package_name = "test-plugin".to_string();
    let package_version = "0.2.0".to_string();
    let build_script_digest: Option<String> = None;
    let common_sources_digest = "b".repeat(64);
    let private_sources_digest = "c".repeat(64);
    let native_deps: Vec<NativeDep> = vec![];
    let pre_embed_payload_digest = "d".repeat(64);
    let artifact_digest = "e".repeat(64);
    let canonical_digest = PluginArtifactBuildRecordV1::compute_digest(
        &universe_digest,
        &package_name,
        &package_version,
        build_script_digest.as_deref(),
        &common_sources_digest,
        &private_sources_digest,
        &native_deps,
        &pre_embed_payload_digest,
        &artifact_digest,
    );
    PluginArtifactBuildRecordV1 {
        universe_digest,
        package_name,
        package_version,
        build_script_digest,
        common_sources_digest,
        private_sources_digest,
        native_deps,
        pre_embed_payload_digest,
        artifact_digest,
        canonical_digest,
    }
}

#[test]
fn encode_decode_round_trip() {
    let record = make_build_record();
    let encoded = encode_section(&record);
    let decoded = decode_section(&encoded).unwrap().expect("should decode a record");
    assert_eq!(record, decoded);
}

#[test]
fn encode_starts_with_magic() {
    let record = make_build_record();
    let encoded = encode_section(&record);
    assert_eq!(&encoded[..4], b"APF1", "section must start with APF1 magic");
}

#[test]
fn length_field_matches_json_payload() {
    let record = make_build_record();
    let encoded = encode_section(&record);
    let declared_len = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
    assert_eq!(
        declared_len,
        encoded.len() - 8,
        "declared length must match actual JSON payload length"
    );
}

#[test]
fn decode_returns_none_on_wrong_magic() {
    let mut encoded = encode_section(&make_build_record());
    // Corrupt the magic.
    encoded[0] = b'X';
    let result = decode_section(&encoded).unwrap();
    assert!(result.is_none(), "wrong magic must return None");
}

#[test]
fn decode_returns_none_on_empty_slice() {
    let result = decode_section(&[]).unwrap();
    assert!(result.is_none(), "empty slice must return None");
}

#[test]
fn decode_errors_on_truncated_payload() {
    let encoded = encode_section(&make_build_record());
    // Trim last 10 bytes to truncate the JSON payload.
    let truncated = &encoded[..encoded.len() - 10];
    let result = decode_section(truncated);
    assert!(result.is_err(), "truncated payload must return an error");
}
