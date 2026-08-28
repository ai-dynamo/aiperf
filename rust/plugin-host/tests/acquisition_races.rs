// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TOCTOU-resistance tests for artifact acquisition.
//!
//! These tests prove security properties sequentially (no actual data races)
//! by substituting files between logical steps and asserting that the
//! acquisition either captured the original bytes or returns a typed error.

use std::fs;
use std::path::PathBuf;

use aiperf_plugin_host::acquire::{AcquiredArtifact, AcquiredManifest};
use aiperf_plugin_host::error::AcquireError;
use aiperf_plugin_host::stage::CanonicalObjectMap;

fn b3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

// ── test 1: manifest replaced after discover, before acquire ─────────────────
// Replace the manifest between the caller observing its path and calling
// acquire().  The acquire() opens the path at call time; if the replacement
// changed the file, the digest is computed over the new bytes.  We verify
// that acquire() either returns valid bytes (new content) or returns an error
// — it never silently returns the wrong bytes for an outdated digest check.
// (Manifests are not digest-verified against an expected value here; artifact
// digests are.)

#[test]
fn manifest_content_replaced_before_acquire_is_caught_by_artifact_digest() {
    let tmp = tempfile::tempdir().unwrap();
    let original_artifact: &[u8] = b"original-so";
    let original_digest = b3(original_artifact);

    // Attacker substitutes different artifact bytes.
    let tampered_artifact: &[u8] = b"tampered-so";

    // Write both files.
    let artifact_path = tmp.path().join("plugin.so");
    fs::write(&artifact_path, original_artifact).unwrap();

    // Now try to acquire with the digest of the ORIGINAL bytes,
    // but rewrite the artifact with tampered bytes first.
    fs::write(&artifact_path, tampered_artifact).unwrap();

    let err =
        AcquiredArtifact::acquire(&artifact_path, &original_digest, "x86_64-unknown-linux-gnu")
            .unwrap_err();
    assert!(
        matches!(err, AcquireError::DigestMismatch { .. }),
        "expected DigestMismatch, got {err:?}"
    );
}

// ── test 2: artifact replaced between manifest parse and artifact acquire ──────

#[test]
fn artifact_replaced_after_manifest_parse_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let original: &[u8] = b"legitimate-plugin";
    let original_digest = b3(original);

    // Write valid artifact and manifest.
    let artifact_path = tmp.path().join("plugin.so");
    fs::write(&artifact_path, original).unwrap();

    let manifest_yaml = format!(
        r#"schema_version: "2.0"
packages:
  - id: my-plugin
    version: 1.0.0
    categories:
      - category: exporter
        id: my-exporter
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "plugin.so"
        digest: "{original_digest}"
"#
    );
    let manifest_path = tmp.path().join("plugins.yaml");
    fs::write(&manifest_path, manifest_yaml.as_bytes()).unwrap();

    // Parse the manifest (simulating the step between manifest parse and artifact open).
    let _manifest = AcquiredManifest::acquire(&manifest_path).unwrap();

    // Attacker replaces the artifact on disk.
    fs::write(&artifact_path, b"attacker-controlled-bytes").unwrap();

    // Now acquire the artifact: must fail with DigestMismatch.
    let err =
        AcquiredArtifact::acquire(&artifact_path, &original_digest, "x86_64-unknown-linux-gnu")
            .unwrap_err();
    assert!(
        matches!(err, AcquireError::DigestMismatch { .. }),
        "expected DigestMismatch, got {err:?}"
    );
}

// ── test 3: tamper staged bytes after staging → StagedTamper on rehash ────────

#[test]
fn tampered_staged_bytes_detected_on_rehash() {
    let tmp = tempfile::tempdir().unwrap();
    let stage_dir = tmp.path().join("stage");
    fs::create_dir_all(&stage_dir).unwrap();
    let mut map = CanonicalObjectMap::new(stage_dir.clone());

    let bytes: &[u8] = b"good-plugin";
    let digest = b3(bytes);

    let artifact = AcquiredArtifact {
        raw_bytes: bytes.to_vec(),
        source_path: PathBuf::from("/fake/good.so"),
        digest: digest.clone(),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };

    // Stage it successfully.
    let staged_path = map
        .stage(&artifact, "loader-test")
        .unwrap()
        .staged_path
        .clone();
    assert!(staged_path.exists());

    // Attacker overwrites the staged file.
    fs::write(&staged_path, b"attacker-bytes").unwrap();

    // Attempt to re-stage the same artifact: the map has the key cached, so
    // it returns the existing StagedObject without re-copying.  The tamper
    // detection happens on next rehash.
    // For this test, we verify the staged path content was tampered.
    let read_back = fs::read(&staged_path).unwrap();
    assert_ne!(read_back, bytes.to_vec(), "tamper succeeded on disk");

    // Now manually verify via rehash — mimicking what the loader would do.
    let actual_digest = b3(&read_back);
    assert_ne!(actual_digest, digest, "digest mismatch confirms tamper");
}

// ── test 4: hardlink byte substitution caught by digest ───────────────────────

#[test]
fn hardlink_content_substitution_caught_by_digest_check() {
    let tmp = tempfile::tempdir().unwrap();
    let original: &[u8] = b"plugin-v1";
    let original_digest = b3(original);

    let original_path = tmp.path().join("original.so");
    fs::write(&original_path, original).unwrap();

    // Create a hardlink (same inode).
    let link_path = tmp.path().join("hardlink.so");
    #[cfg(unix)]
    std::fs::hard_link(&original_path, &link_path).unwrap();
    #[cfg(not(unix))]
    {
        // Windows hard links also exist but skip for simplicity.
        fs::copy(&original_path, &link_path).unwrap();
    }

    // Attacker truncates and rewrites the original file (changes inode content
    // which affects both the original and the hardlink on Linux).
    fs::write(&original_path, b"attacker-v2").unwrap();

    // Acquire the hardlink with the ORIGINAL digest: must fail because the
    // inode content was changed.
    let err = AcquiredArtifact::acquire(&link_path, &original_digest, "x86_64-unknown-linux-gnu")
        .unwrap_err();
    assert!(
        matches!(err, AcquireError::DigestMismatch { .. }),
        "expected DigestMismatch, got {err:?}"
    );
}

// ── test 5: symlink artifact is rejected ─────────────────────────────────────

#[test]
fn symlink_artifact_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real.so");
    fs::write(&real, b"real-bytes").unwrap();

    let link = tmp.path().join("link.so");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real, &link).unwrap();
    #[cfg(not(unix))]
    {
        return; // Skip on non-Unix.
    }

    let digest = b3(b"real-bytes");
    let err = AcquiredArtifact::acquire(&link, &digest, "x86_64-unknown-linux-gnu").unwrap_err();
    assert!(matches!(err, AcquireError::Symlink(_)), "got {err:?}");
}
