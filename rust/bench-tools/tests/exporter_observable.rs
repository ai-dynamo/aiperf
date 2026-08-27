// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API coverage for exporter raw-observable validation.

use aiperf_bench_tools::exporter_observable::{ArtifactTreeKind, parse_artifact_tree_observable};

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
