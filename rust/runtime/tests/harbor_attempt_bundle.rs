// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, DeclaredArtifactManifest, MaterializedArtifactManifest,
};

#[test]
fn manifests_sort_paths_and_change_only_for_identity_bearing_content() {
    let declared =
        DeclaredArtifactManifest::new(["/results/z.txt".to_owned(), "/results/a.txt".to_owned()])
            .unwrap();
    let reordered =
        DeclaredArtifactManifest::new(["/results/a.txt".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    assert_eq!(declared, reordered);

    let materialized = MaterializedArtifactManifest::new([
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"a"),
        ),
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
    ])
    .unwrap();
    assert_ne!(declared.digest, materialized.digest);
}

#[test]
fn manifests_canonicalize_import_aliases_and_reject_invalid_or_tampered_values() {
    let aliases =
        DeclaredArtifactManifest::new(["//results//a/".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    let canonical =
        DeclaredArtifactManifest::new(["/results/a".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    assert_eq!(aliases, canonical);
    assert!(
        DeclaredArtifactManifest::new(["/results/a".to_owned(), "//results/a/".to_owned(),])
            .is_err()
    );
    assert!(DeclaredArtifactManifest::new(["relative".to_owned()]).is_err());
    assert!(serde_json::from_str::<DeclaredArtifactManifest>(
        r#"{"paths":["/results/a"],"digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#,
    )
    .is_err());
}

#[test]
fn declared_manifest_identity_uses_versioned_length_delimited_bytes() {
    let manifest = DeclaredArtifactManifest::new(["/results/a".to_owned()]).unwrap();
    let expected = ArtifactDigest::from_bytes(
        b"harbor-declared-artifacts-v1\x1f\0\0\0\0\0\0\0\x01\x1e\0\0\0\0\0\0\0\x0a/results/a",
    );

    assert_eq!(manifest.digest, expected);
}
