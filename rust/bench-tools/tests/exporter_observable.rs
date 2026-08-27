// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API coverage for exporter raw-observable validation.

use aiperf_bench_tools::exporter_observable::{
    ArtifactTreeKind, parse_artifact_tree_observable, parse_receiver_transcript_observable,
    validate_captured_stream_observable, validate_receiver_transcript_bodies,
};

const EMPTY_DIGEST: &str =
    "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
const FILE_DIGEST: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn parses_the_canonical_artifact_tree_manifest() {
    let bytes = format!(
        "[{{\"blake3\":\"{EMPTY_DIGEST}\",\"kind\":\"empty_directory\",\"length\":0,\"path\":\"logs\"}},{{\"blake3\":\"{FILE_DIGEST}\",\"kind\":\"regular_file\",\"length\":3,\"path\":\"logs/out.txt\"}}]\n"
    );

    let entries = parse_artifact_tree_observable(bytes.as_bytes())
        .expect("canonical artifact-tree manifest must parse");

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].kind, ArtifactTreeKind::EmptyDirectory);
    assert_eq!(entries[1].kind, ArtifactTreeKind::RegularFile);
    assert_eq!(entries[1].path, "logs/out.txt");
}

#[test]
fn rejects_noncanonical_artifact_tree_manifest_bytes() {
    let bytes = format!(
        "[{{\"path\":\"out.txt\",\"kind\":\"regular_file\",\"length\":3,\"blake3\":\"{FILE_DIGEST}\"}}]\n"
    );

    let error = parse_artifact_tree_observable(bytes.as_bytes())
        .expect_err("reordered manifest fields must not be accepted as a raw observable");

    assert_eq!(
        error.to_string(),
        "artifact-tree observable is not exact RFC 8785 JCS plus newline"
    );
}

#[test]
fn receiver_transcript_requires_dense_sequences_and_canonical_bytes() {
    let bytes = format!(
        "[{{\"body\":{{\"blake3\":\"{FILE_DIGEST}\",\"encoding\":\"bytes\",\"length\":3}},\"metadata\":[[\"content-type\",\"application/json\"]],\"operation\":\"POST\",\"sequence\":0,\"target\":\"/v1/traces\"}}]\n"
    );
    let entries = parse_receiver_transcript_observable(bytes.as_bytes(), false)
        .expect("canonical receiver transcript must parse");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].sequence, 0);

    let reordered = bytes.replace("\"sequence\":0", "\"sequence\":1");
    let error = parse_receiver_transcript_observable(reordered.as_bytes(), false)
        .expect_err("receiver sequences must be dense from zero");
    assert_eq!(
        error.to_string(),
        "receiver transcript sequences must be dense from zero"
    );
}

#[test]
fn captured_stream_empty_bytes_follow_the_frozen_scenario_policy() {
    assert!(validate_captured_stream_observable(b"", true).is_ok());
    let error = validate_captured_stream_observable(b"", false)
        .expect_err("a nonempty scenario must reject an empty capture");
    assert_eq!(
        error.to_string(),
        "captured-stream observable is empty but the scenario forbids it"
    );
    assert!(validate_captured_stream_observable(b"exact bytes\n", false).is_ok());
}

#[test]
fn receiver_transcript_body_identity_covers_the_exact_retained_bytes() {
    const BODY_DIGEST: &str =
        "blake3:6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85";
    let manifest = format!(
        "[{{\"body\":{{\"blake3\":\"{BODY_DIGEST}\",\"encoding\":\"bytes\",\"length\":3}},\"metadata\":[],\"operation\":\"POST\",\"sequence\":0,\"target\":\"/v1/traces\"}}]\n"
    );
    let entries = parse_receiver_transcript_observable(manifest.as_bytes(), false)
        .expect("canonical receiver transcript must parse");
    validate_receiver_transcript_bodies(&entries, &[b"abc".as_slice()])
        .expect("the exact retained body must match its transcript identity");

    let error = validate_receiver_transcript_bodies(&entries, &[b"ab!".as_slice()])
        .expect_err("a one-byte retained-body mutation must fail");
    assert_eq!(
        error.to_string(),
        "retained receiver body does not match its transcript identity"
    );
}
