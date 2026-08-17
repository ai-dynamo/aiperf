// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial contract coverage for NativeGraph artifact capabilities.

use std::{
    fs,
    io::{self, Cursor, Read},
};

use aiperf_runtime::eval::{ArtifactError, ArtifactQuota, EpisodeArtifactStore};

fn quota() -> ArtifactQuota {
    ArtifactQuota {
        max_artifacts: 2,
        max_total_bytes: 8,
        max_artifact_bytes: 4,
        max_download_handles: 1,
    }
}

fn store(root: &std::path::Path) -> EpisodeArtifactStore {
    EpisodeArtifactStore::new(root, quota()).expect("fixture artifact store is valid")
}

fn commit(store: &mut EpisodeArtifactStore, bytes: &[u8]) -> aiperf_runtime::eval::FrozenArtifact {
    let upload = store
        .begin_upload(bytes.len() as u64)
        .expect("fixture reserves bounded upload");
    store
        .write_upload(&upload, &mut Cursor::new(bytes.to_vec()))
        .expect("fixture upload succeeds");
    store
        .commit_upload(&upload)
        .expect("fixture upload freezes after a full rehash")
}

#[test]
fn upload_capability_is_revoked_after_commit_and_frozen_bytes_stay_immutable() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    artifacts
        .write_upload(&upload, &mut Cursor::new(b"one".to_vec()))
        .expect("write staged bytes");
    let frozen = artifacts
        .commit_upload(&upload)
        .expect("commit exact bytes atomically");

    let error = artifacts
        .write_upload(&upload, &mut Cursor::new(b"two".to_vec()))
        .expect_err("a committed upload capability is revoked");
    assert_eq!(error, ArtifactError::UnknownUploadHandle);
    assert_eq!(
        artifacts
            .read_frozen(&frozen)
            .expect("frozen bytes remain host-owned and readable"),
        b"one"
    );
}

#[test]
fn download_capabilities_are_bounded_one_shot_and_revocable() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let first = commit(&mut artifacts, b"one");
    let second = commit(&mut artifacts, b"two");
    let download = artifacts
        .issue_download(&first)
        .expect("issue one opaque download capability");
    let error = artifacts
        .issue_download(&second)
        .expect_err("active download grants obey their own bound");
    assert!(matches!(error, ArtifactError::DownloadHandleLimit { .. }));

    let mut copied = Vec::new();
    artifacts
        .copy_download(&download, &mut copied)
        .expect("one-shot download streams only validated frozen bytes");
    assert_eq!(copied, b"one");
    let error = artifacts
        .copy_download(&download, &mut Vec::new())
        .expect_err("the consumed download capability is revoked");
    assert_eq!(error, ArtifactError::UnknownDownloadHandle);
}

#[test]
fn partial_upload_failure_poisoned_the_capability_and_never_publishes_bytes() {
    struct FailsAfterPrefix {
        has_sent_prefix: bool,
    }

    impl Read for FailsAfterPrefix {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.has_sent_prefix {
                return Err(io::Error::other("simulated child stream failure"));
            }
            self.has_sent_prefix = true;
            buffer[..2].copy_from_slice(b"no");
            Ok(2)
        }
    }

    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    let error = artifacts
        .write_upload(
            &upload,
            &mut FailsAfterPrefix {
                has_sent_prefix: false,
            },
        )
        .expect_err("a partial source stream poisons rather than leaving resumable bytes");
    assert!(matches!(error, ArtifactError::Io(_)));
    let error = artifacts
        .commit_upload(&upload)
        .expect_err("poisoned partial data is never eligible for publication");
    assert_eq!(error, ArtifactError::UploadPoisoned);
}

#[test]
fn staged_uploads_append_across_streaming_writes_before_rehashing_on_commit() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    artifacts
        .write_upload(&upload, &mut Cursor::new(b"o".to_vec()))
        .expect("first stream chunk is accepted");
    artifacts
        .write_upload(&upload, &mut Cursor::new(b"ne".to_vec()))
        .expect("second stream chunk appends rather than overwriting the first");
    let frozen = artifacts
        .commit_upload(&upload)
        .expect("full streamed bytes are rehashed and frozen");
    assert_eq!(
        artifacts.read_frozen(&frozen).expect("frozen bytes read"),
        b"one"
    );
}

#[test]
fn nofollow_validation_poisoned_a_staging_capability_replaced_with_a_symlink() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    let root = std::fs::read_dir(directory.path())
        .expect("store root is discoverable to this hostile same-user test")
        .next()
        .expect("one store root exists")
        .expect("store root entry is readable")
        .path();
    let staging = std::fs::read_dir(root.join("staging"))
        .expect("staging entry is present")
        .next()
        .expect("one staging entry exists")
        .expect("staging entry is readable")
        .path();
    let replacement = directory.path().join("replacement");
    std::fs::write(&replacement, b"not-upload").expect("replacement fixture writes");
    std::fs::remove_file(&staging).expect("remove test staging file");
    std::os::unix::fs::symlink(&replacement, &staging).expect("replace staging path with symlink");

    let error = artifacts
        .write_upload(&upload, &mut Cursor::new(b"one".to_vec()))
        .expect_err("the retained descriptor rejects a staging name removed underneath it");
    assert!(matches!(error, ArtifactError::StagingValidation(_)));
    assert_eq!(
        artifacts
            .commit_upload(&upload)
            .expect_err("descriptor failure poisons the upload"),
        ArtifactError::UploadPoisoned
    );
}

#[test]
fn commit_rejects_a_staging_name_replaced_after_the_upload_descriptor_was_issued() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    artifacts
        .write_upload(&upload, &mut Cursor::new(b"one".to_vec()))
        .expect("fixture writes the original bytes");

    let root = fs::read_dir(directory.path())
        .expect("store root is discoverable to this hostile same-user test")
        .next()
        .expect("one store root exists")
        .expect("store root entry is readable")
        .path();
    let staging = fs::read_dir(root.join("staging"))
        .expect("staging entry is present")
        .next()
        .expect("one staging entry exists")
        .expect("staging entry is readable")
        .path();
    let replacement = directory.path().join("replacement");
    fs::write(&replacement, b"bad").expect("replacement fixture writes");
    fs::remove_file(&staging).expect("remove the store name without closing its descriptor");
    fs::hard_link(&replacement, &staging).expect("replace the upload name with another file");

    let error = artifacts
        .commit_upload(&upload)
        .expect_err("publication must use the retained upload descriptor, not its mutable name");
    assert!(matches!(error, ArtifactError::StagingValidation(_)));
}

#[test]
fn replaced_frozen_directory_never_delivers_any_bytes_to_the_download_writer() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let frozen = commit(&mut artifacts, b"one");
    let download = artifacts
        .issue_download(&frozen)
        .expect("issue an opaque one-shot read capability");

    let root = fs::read_dir(directory.path())
        .expect("store root is discoverable to this hostile same-user test")
        .next()
        .expect("one store root exists")
        .expect("store root entry is readable")
        .path();
    let frozen_directory = root.join("frozen");
    let frozen_path = fs::read_dir(&frozen_directory)
        .expect("frozen entry is present")
        .next()
        .expect("one frozen entry exists")
        .expect("frozen entry is readable")
        .path();
    let frozen_name = frozen_path
        .file_name()
        .expect("frozen entry has one filename")
        .to_owned();
    let replacement = directory.path().join("replacement");
    fs::write(&replacement, b"bad").expect("replacement fixture writes");
    fs::remove_file(&frozen_path).expect("remove the frozen name before delivery");
    fs::remove_dir(&frozen_directory).expect("remove the frozen directory name before delivery");
    fs::create_dir(&frozen_directory).expect("replace the frozen directory name");
    fs::hard_link(&replacement, frozen_directory.join(&frozen_name))
        .expect("replace the frozen entry with equal-length bytes");

    let mut delivered = Vec::new();
    let error = artifacts
        .copy_download(&download, &mut delivered)
        .expect_err("the held directory descriptor must not follow a replacement path");
    assert!(matches!(error, ArtifactError::Io(_)));
    assert!(
        delivered.is_empty(),
        "unvalidated bytes never cross the store boundary"
    );
    fs::remove_file(frozen_directory.join(frozen_name))
        .expect("test-owned replacement is removed before store cleanup");
}

#[test]
fn failed_staging_removal_rolls_back_the_new_frozen_link_and_tracks_the_upload() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let upload = artifacts.begin_upload(3).expect("reserve upload");
    artifacts
        .write_upload(&upload, &mut Cursor::new(b"one".to_vec()))
        .expect("fixture bytes write");
    let root = std::fs::read_dir(directory.path())
        .expect("store root is discoverable to this hostile same-user test")
        .next()
        .expect("one store root exists")
        .expect("store root entry is readable")
        .path();
    let staging_root = root.join("staging");
    std::fs::set_permissions(&staging_root, std::fs::Permissions::from_mode(0o500))
        .expect("make staging removal fail after publication");

    let error = artifacts
        .commit_upload(&upload)
        .expect_err("failed post-link staging cleanup must not publish a frozen artifact");
    assert!(matches!(error, ArtifactError::Io(_)));
    assert_eq!(
        std::fs::read_dir(root.join("frozen"))
            .expect("frozen root readable")
            .count(),
        0,
        "the new frozen link is rolled back"
    );
    std::fs::set_permissions(&staging_root, std::fs::Permissions::from_mode(0o700))
        .expect("restore cleanup permission");
    artifacts
        .abort_upload(&upload)
        .expect("tracked staging upload remains abortable after rollback");
}

#[test]
fn artifact_quota_is_reserved_before_writes_and_released_by_abort() {
    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = EpisodeArtifactStore::new(
        directory.path(),
        ArtifactQuota {
            max_artifacts: 1,
            ..quota()
        },
    )
    .expect("single-entry fixture artifact store is valid");
    let first = artifacts.begin_upload(4).expect("reserve first upload");
    let second = artifacts
        .begin_upload(4)
        .expect_err("second active reservation exceeds artifact count quota");
    assert!(matches!(
        second,
        ArtifactError::ArtifactCountQuotaExceeded { .. }
    ));
    artifacts
        .abort_upload(&first)
        .expect("abort removes the staging file before releasing quota");
    artifacts
        .begin_upload(4)
        .expect("released reservation permits a replacement upload");
}

#[test]
fn manifest_rejects_a_duplicate_before_exhausting_its_input_iterator() {
    struct DuplicateThenPanic {
        artifact: aiperf_runtime::eval::FrozenArtifact,
        emitted: usize,
    }

    impl Iterator for DuplicateThenPanic {
        type Item = aiperf_runtime::eval::FrozenArtifact;

        fn next(&mut self) -> Option<Self::Item> {
            self.emitted += 1;
            match self.emitted {
                1 | 2 => Some(self.artifact.clone()),
                _ => panic!("manifest must reject the duplicate before collecting another item"),
            }
        }
    }

    let directory = tempfile::tempdir().expect("temporary root");
    let mut artifacts = store(directory.path());
    let artifact = commit(&mut artifacts, b"one");
    let error = artifacts
        .freeze_manifest(DuplicateThenPanic {
            artifact,
            emitted: 0,
        })
        .expect_err("manifest validation must reject duplicate inputs incrementally");
    assert_eq!(error, ArtifactError::DuplicateFrozenArtifact);
}
