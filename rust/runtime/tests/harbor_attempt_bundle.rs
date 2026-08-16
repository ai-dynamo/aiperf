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

#[test]
fn materialized_manifest_sorts_paths_and_binds_content_digests() {
    let first = MaterializedArtifactManifest::new([
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"a"),
        ),
    ])
    .unwrap();
    let reordered = MaterializedArtifactManifest::new([
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
    let changed_content = MaterializedArtifactManifest::new([
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"changed"),
        ),
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
    ])
    .unwrap();

    assert_eq!(first, reordered);
    assert_ne!(first.digest, changed_content.digest);
}

#[test]
fn manifests_reject_serde_tampering_and_unknown_fields() {
    let declared = DeclaredArtifactManifest::new(["/results/a".to_owned()]).unwrap();
    let materialized = MaterializedArtifactManifest::new([(
        "/results/a".to_owned(),
        ArtifactDigest::from_bytes(b"a"),
    )])
    .unwrap();

    let declared_json = serde_json::to_value(&declared).unwrap();
    let materialized_json = serde_json::to_value(&materialized).unwrap();
    assert_eq!(
        serde_json::from_value::<DeclaredArtifactManifest>(declared_json.clone()).unwrap(),
        declared
    );
    assert_eq!(
        serde_json::from_value::<MaterializedArtifactManifest>(materialized_json.clone()).unwrap(),
        materialized
    );

    let mut declared_unknown = declared_json.clone();
    declared_unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<DeclaredArtifactManifest>(declared_unknown).is_err());
    let mut materialized_unknown = materialized_json.clone();
    materialized_unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<MaterializedArtifactManifest>(materialized_unknown).is_err());

    let mut declared_tampered = declared_json;
    declared_tampered.as_object_mut().unwrap().insert(
        "digest".to_owned(),
        serde_json::json!(ArtifactDigest::from_bytes(b"tampered")),
    );
    assert!(serde_json::from_value::<DeclaredArtifactManifest>(declared_tampered).is_err());
    let mut materialized_tampered = materialized_json;
    materialized_tampered.as_object_mut().unwrap().insert(
        "digest".to_owned(),
        serde_json::json!(ArtifactDigest::from_bytes(b"tampered")),
    );
    assert!(serde_json::from_value::<MaterializedArtifactManifest>(materialized_tampered).is_err());
}

#[test]
fn empty_manifest_kinds_have_distinct_identities() {
    let declared = DeclaredArtifactManifest::new([]).unwrap();
    let materialized = MaterializedArtifactManifest::new([]).unwrap();

    assert_ne!(declared.digest, materialized.digest);
}
