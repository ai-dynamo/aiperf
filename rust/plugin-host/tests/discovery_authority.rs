// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery trust: a plugin directory is only trusted when it is a real
//! directory owned by root or the current user with no group/world write bits.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use aiperf_plugin_host::platform::check_directory_trust;

fn set_mode(path: &Path, mode: u32) {
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).expect("set_permissions");
}

#[test]
fn trusted_root_owned_by_root_passes() {
    // The process is not usually root, so exercise the root-owned branch against
    // a directory that genuinely is root-owned on every Linux host.
    let path = Path::new("/usr");
    if !path.is_dir() {
        return;
    }
    assert!(
        check_directory_trust(path).is_ok(),
        "/usr must be a trusted root-owned directory"
    );
}

#[test]
fn trusted_root_owned_by_self_passes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    set_mode(&plugins, 0o755);
    assert!(check_directory_trust(&plugins).is_ok());
}

#[test]
fn group_writable_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    set_mode(&plugins, 0o775);
    assert!(check_directory_trust(&plugins).is_err());
}

#[test]
fn world_writable_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plugins = dir.path().join("plugins");
    fs::create_dir(&plugins).expect("create_dir");
    set_mode(&plugins, 0o777);
    assert!(check_directory_trust(&plugins).is_err());
}

#[test]
fn symlinked_directory_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let real = dir.path().join("real");
    fs::create_dir(&real).expect("create_dir");
    set_mode(&real, 0o755);
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    assert!(check_directory_trust(&link).is_err());
}
