// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Acquisition tests: manifest, artifact, closure, and staging.

use std::fs;
use std::path::PathBuf;

use aiperf_plugin_host::acquire::{AcquiredArtifact, AcquiredManifest};
use aiperf_plugin_host::closure::AcquiredClosure;
use aiperf_plugin_host::error::AcquireError;
use aiperf_plugin_host::stage::CanonicalObjectMap;

/// Build a minimal valid plugins.yaml in `dir` with one artifact at `artifact_path`
/// whose BLAKE3 digest is `digest`.
fn write_manifest(dir: &std::path::Path, artifact_path: &str, digest: &str) -> PathBuf {
    let yaml = format!(
        r#"schema_version: "2.0"
packages:
  - id: test-plugin
    version: 1.0.0
    categories:
      - category: exporter
        id: test-exporter
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "{artifact_path}"
        digest: "{digest}"
"#
    );
    let path = dir.join("plugins.yaml");
    fs::write(&path, yaml.as_bytes()).unwrap();
    path
}

/// Compute the BLAKE3 hex digest of `bytes`.
fn b3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

// ── test 1: valid manifest acquisition ────────────────────────────────────────

#[test]
fn valid_manifest_acquires_and_digest_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let artifact_bytes: &[u8] = b"fake-so-bytes";
    let digest = b3(artifact_bytes);
    let manifest_path = write_manifest(tmp.path(), "fake.so", &digest);

    let acquired = AcquiredManifest::acquire(&manifest_path).unwrap();
    assert_eq!(acquired.source_path, manifest_path);
    assert!(!acquired.digest.is_empty());
    assert!(!acquired.canonical.packages.is_empty());
}

// ── test 2: symlink manifest is rejected ──────────────────────────────────────

#[test]
fn symlink_manifest_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real.yaml");
    fs::write(&real, b"schema_version: \"2.0\"\npackages: []\n").unwrap();
    let link = tmp.path().join("link.yaml");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real, &link).unwrap();
    #[cfg(not(unix))]
    {
        // Skip on non-Unix where symlink creation may require elevated privileges.
        return;
    }
    let err = AcquiredManifest::acquire(&link).unwrap_err();
    assert!(matches!(err, AcquireError::Symlink(_)), "got {err:?}");
}

// ── test 3: artifact digest mismatch is rejected ──────────────────────────────

#[test]
fn artifact_digest_mismatch_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let artifact_path = tmp.path().join("plugin.so");
    fs::write(&artifact_path, b"real-bytes").unwrap();

    let wrong_digest = b3(b"different-bytes");
    let err = AcquiredArtifact::acquire(&artifact_path, &wrong_digest, "x86_64-unknown-linux-gnu")
        .unwrap_err();
    assert!(
        matches!(err, AcquireError::DigestMismatch { .. }),
        "got {err:?}"
    );
}

// ── test 4: valid artifact acquisition ───────────────────────────────────────

#[test]
fn valid_artifact_acquires_successfully() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes: &[u8] = b"my-plugin-shared-object";
    let artifact_path = tmp.path().join("plugin.so");
    fs::write(&artifact_path, bytes).unwrap();
    let digest = b3(bytes);

    let acquired =
        AcquiredArtifact::acquire(&artifact_path, &digest, "x86_64-unknown-linux-gnu").unwrap();
    assert_eq!(acquired.digest, digest);
    assert_eq!(acquired.target, "x86_64-unknown-linux-gnu");
    assert_eq!(acquired.raw_bytes, bytes);
}

// ── test 5: AcquiredClosure from manifest ────────────────────────────────────

#[test]
fn closure_acquires_from_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let artifact_bytes: &[u8] = b"so-content";
    let digest = b3(artifact_bytes);
    let artifact_path = tmp.path().join("plugin.so");
    fs::write(&artifact_path, artifact_bytes).unwrap();
    let manifest_path = write_manifest(tmp.path(), "plugin.so", &digest);

    let closure = AcquiredClosure::acquire_from_manifest(
        &manifest_path,
        &["x86_64-unknown-linux-gnu"],
    )
    .unwrap();
    assert_eq!(closure.artifacts.len(), 1);
    assert_eq!(closure.artifacts[0].target, "x86_64-unknown-linux-gnu");
    assert_eq!(closure.artifacts[0].digest, digest);
}

// ── test 6: identical-digest artifacts coalesce in CanonicalObjectMap ─────────

#[test]
fn identical_digest_coalesces_in_object_map() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_dir = tmp.path().join("stage");
    fs::create_dir_all(&stage_dir).unwrap();
    let mut map = CanonicalObjectMap::new(stage_dir);

    let bytes: &[u8] = b"same-bytes";
    let digest = b3(bytes);

    let a1 = AcquiredArtifact {
        raw_bytes: bytes.to_vec(),
        source_path: PathBuf::from("/fake/a.so"),
        digest: digest.clone(),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };
    let a2 = AcquiredArtifact {
        raw_bytes: bytes.to_vec(),
        source_path: PathBuf::from("/fake/b.so"),
        digest: digest.clone(),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };

    let s1 = map.stage(&a1, "loader-a").unwrap();
    let path1 = s1.staged_path.clone();

    let s2 = map.stage(&a2, "loader-a").unwrap();
    let path2 = s2.staged_path.clone();

    // Same key → same staged path returned.
    assert_eq!(path1, path2);
}

// ── test 7: different-digest artifacts stage separately ───────────────────────

#[test]
fn different_digests_stage_separately() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_dir = tmp.path().join("stage");
    fs::create_dir_all(&stage_dir).unwrap();
    let mut map = CanonicalObjectMap::new(stage_dir);

    let bytes1: &[u8] = b"artifact-one";
    let bytes2: &[u8] = b"artifact-two";

    let a1 = AcquiredArtifact {
        raw_bytes: bytes1.to_vec(),
        source_path: PathBuf::from("/fake/a.so"),
        digest: b3(bytes1),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };
    let a2 = AcquiredArtifact {
        raw_bytes: bytes2.to_vec(),
        source_path: PathBuf::from("/fake/b.so"),
        digest: b3(bytes2),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };

    let path1 = map.stage(&a1, "loader-x").unwrap().staged_path.clone();
    let path2 = map.stage(&a2, "loader-x").unwrap().staged_path.clone();

    assert_ne!(path1, path2);
    assert!(path1.exists());
    assert!(path2.exists());
}

// ── test 8: staged file content matches original after staging ────────────────

#[test]
fn staged_bytes_match_original() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_dir = tmp.path().join("stage");
    fs::create_dir_all(&stage_dir).unwrap();
    let mut map = CanonicalObjectMap::new(stage_dir);

    let bytes: &[u8] = b"exact-plugin-bytes";
    let artifact = AcquiredArtifact {
        raw_bytes: bytes.to_vec(),
        source_path: PathBuf::from("/fake/p.so"),
        digest: b3(bytes),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };

    let staged = map.stage(&artifact, "loader-z").unwrap();
    let read_back = fs::read(&staged.staged_path).unwrap();
    assert_eq!(read_back, bytes);
}
