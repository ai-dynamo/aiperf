// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery trust tests: a discovery path is only trusted when its owner and
//! mode prove that no unauthorized user can replace the bytes the host is about
//! to load, and when the platform's ACL semantics are understood.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use aiperf_plugin_host::error::AuthorityError;
use aiperf_plugin_host::platform::acl_unix::{
    AuthorityPolicy, check_path_authority, check_path_authority_with, check_tree_authority,
    current_euid, has_extended_acl,
};

fn write_mode(path: &Path, bytes: &[u8], mode: u32) {
    fs::write(path, bytes).expect("write test file");
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).expect("chmod test file");
}

#[test]
fn owner_only_file_in_owner_only_directory_is_trusted() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("chmod dir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"schema_version: \"2.0\"\n", 0o644);

    check_path_authority(&file).expect("0644 file owned by the caller is trusted");
}

#[test]
fn world_writable_file_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o666);

    let err = check_path_authority(&file).expect_err("world-writable file must be rejected");
    assert!(
        matches!(err, AuthorityError::WorldWritable { .. }),
        "expected WorldWritable, got {err:?}"
    );
}

#[test]
fn group_writable_file_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o664);

    let err = check_path_authority(&file).expect_err("group-writable file must be rejected");
    assert!(
        matches!(err, AuthorityError::GroupWritable { .. }),
        "expected GroupWritable, got {err:?}"
    );
}

#[test]
fn symlinked_path_is_rejected_without_following_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("real.yaml");
    write_mode(&target, b"x", 0o644);
    let link = dir.path().join("link.yaml");
    std::os::unix::fs::symlink(&target, &link).expect("symlink");

    let err = check_path_authority(&link).expect_err("symlink must be rejected");
    assert!(
        matches!(err, AuthorityError::Symlink(_)),
        "expected Symlink, got {err:?}"
    );
}

#[test]
fn file_owned_by_an_untrusted_uid_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o644);

    // The caller's own uid is deliberately excluded from the trusted set, which
    // is the same decision the host makes for a user-owned file discovered by a
    // privileged process.
    let policy = AuthorityPolicy {
        trusted_uids: vec![current_euid().wrapping_add(1)],
        ..AuthorityPolicy::default()
    };
    let err = check_path_authority_with(&file, &policy)
        .expect_err("file owned by an untrusted uid must be rejected");
    assert!(
        matches!(err, AuthorityError::UntrustedOwner { .. }),
        "expected UntrustedOwner, got {err:?}"
    );
}

#[test]
fn world_writable_parent_directory_is_rejected_by_the_tree_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sub = dir.path().join("plugins");
    fs::create_dir(&sub).expect("mkdir");
    let file = sub.join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o644);
    fs::set_permissions(&sub, fs::Permissions::from_mode(0o777)).expect("chmod dir");

    // The file itself is fine; the directory it lives in lets anyone replace it.
    check_path_authority(&file).expect("file mode alone is trusted");
    let err = check_tree_authority(&file, dir.path())
        .expect_err("world-writable ancestor directory must be rejected");
    assert!(
        matches!(err, AuthorityError::WorldWritable { .. }),
        "expected WorldWritable on the ancestor, got {err:?}"
    );
}

#[test]
fn a_plain_file_reports_no_extended_acl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o644);

    assert!(
        !has_extended_acl(&file).expect("acl probe must be conclusive for a plain file"),
        "a freshly created 0644 file must not carry an extended ACL"
    );
}

#[test]
fn unknown_acl_semantics_fail_closed_when_the_probe_is_required() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("plugin.manifest.yaml");
    write_mode(&file, b"x", 0o644);

    let policy = AuthorityPolicy {
        require_acl_probe: true,
        ..AuthorityPolicy::default()
    };
    // On a platform whose ACL semantics this host cannot read, the strict policy
    // must refuse rather than trust the mode bits alone.
    match check_path_authority_with(&file, &policy) {
        Ok(()) => assert!(
            cfg!(target_os = "linux"),
            "only a platform with a conclusive ACL probe may accept under a strict policy"
        ),
        Err(e) => assert!(
            matches!(e, AuthorityError::UnknownAclSemantics { .. }),
            "expected UnknownAclSemantics, got {e:?}"
        ),
    }
}

#[test]
fn a_missing_path_is_an_authority_failure_not_a_silent_pass() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = check_path_authority(&dir.path().join("absent.yaml"))
        .expect_err("a missing path must not be reported as trusted");
    assert!(
        matches!(err, AuthorityError::Io { .. }),
        "expected Io, got {err:?}"
    );
}

// Directory-level trust: the platform-neutral entry point discovery uses.

#[test]
fn trusted_root_owned_by_root_passes() {
    // The process is not usually root, so exercise the root-owned branch against
    // a directory that genuinely is root-owned on every Linux host.
    let path = Path::new("/usr");
    if !path.is_dir() {
        return;
    }
    aiperf_plugin_host::platform::check_directory_trust(path)
        .expect("/usr must be a trusted root-owned directory");
}

#[test]
fn trusted_root_owned_by_self_passes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    fs::set_permissions(&plugins, fs::Permissions::from_mode(0o755)).expect("chmod");
    aiperf_plugin_host::platform::check_directory_trust(&plugins)
        .expect("a 0755 directory owned by the caller is trusted");
}

#[test]
fn group_writable_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    fs::set_permissions(&plugins, fs::Permissions::from_mode(0o775)).expect("chmod");
    assert!(aiperf_plugin_host::platform::check_directory_trust(&plugins).is_err());
}

#[test]
fn world_writable_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    fs::set_permissions(&plugins, fs::Permissions::from_mode(0o777)).expect("chmod");
    assert!(aiperf_plugin_host::platform::check_directory_trust(&plugins).is_err());
}

#[test]
fn symlinked_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let real = dir.path().join("real");
    fs::create_dir(&real).expect("create_dir");
    fs::set_permissions(&real, fs::Permissions::from_mode(0o755)).expect("chmod");
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    assert!(aiperf_plugin_host::platform::check_directory_trust(&link).is_err());
}
