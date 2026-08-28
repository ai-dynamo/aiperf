// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin discovery tests (Task 13).

use std::path::PathBuf;

use aiperf_plugin_host::discovery::{
    discover_plugins, DiscoverySource, MANIFEST_FILENAME,
};

/// A non-existent explicit directory returns an empty list (not an error).
#[test]
fn nonexistent_explicit_dir_is_empty() {
    let sources = vec![DiscoverySource::ExplicitDirectory(
        PathBuf::from("/tmp/aiperf_test_nonexistent_99999"),
    )];
    let results = discover_plugins(&sources, false).expect("no error for non-existent dir");
    assert!(results.is_empty());
}

/// An explicit manifest path that does not exist is silently skipped.
#[test]
fn nonexistent_explicit_manifest_is_skipped() {
    let sources = vec![DiscoverySource::ExplicitManifest(
        PathBuf::from("/tmp/aiperf_test_nonexistent_manifest.yaml"),
    )];
    let results = discover_plugins(&sources, false).expect("no error");
    assert!(results.is_empty());
}

/// A directory containing a subdirectory with `plugin.manifest.yaml` is discovered.
#[test]
fn subdir_manifest_discovered() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let pkg_dir = dir.path().join("my-plugin");
    std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
    let manifest_path = pkg_dir.join(MANIFEST_FILENAME);
    std::fs::write(&manifest_path, b"schema_version: '2.0'\npackages: []\n")
        .expect("write manifest");

    let sources = vec![DiscoverySource::ExplicitDirectory(dir.path().to_path_buf())];
    let results = discover_plugins(&sources, false).expect("discover");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].manifest_path.file_name().unwrap(), MANIFEST_FILENAME);
}

/// `no_auto_plugins = true` skips Distribution, PlatformSystem, PlatformUser.
#[test]
fn no_auto_plugins_skips_auto_sources() {
    let sources = vec![
        DiscoverySource::Distribution,
        DiscoverySource::PlatformSystem,
        DiscoverySource::PlatformUser,
    ];
    // Even if those dirs existed we'd get nothing with no_auto_plugins=true.
    let results = discover_plugins(&sources, true).expect("no error");
    assert!(results.is_empty());
}

/// An explicit manifest file at the root of a directory is discovered.
#[test]
fn root_manifest_file_discovered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest_path = dir.path().join(MANIFEST_FILENAME);
    std::fs::write(&manifest_path, b"schema_version: '2.0'\npackages: []\n")
        .expect("write");

    let sources = vec![DiscoverySource::ExplicitDirectory(dir.path().to_path_buf())];
    let results = discover_plugins(&sources, false).expect("discover");
    assert_eq!(results.len(), 1);
}

/// Source kind ordinals are strictly ordered.
#[test]
fn source_kind_ordinal_ordering() {
    use aiperf_plugin_host::priority::source_kind_ordinal;

    let dist = source_kind_ordinal(&DiscoverySource::Distribution);
    let sys = source_kind_ordinal(&DiscoverySource::PlatformSystem);
    let user = source_kind_ordinal(&DiscoverySource::PlatformUser);
    let env = source_kind_ordinal(&DiscoverySource::Environment("TEST".into()));
    let expdir = source_kind_ordinal(&DiscoverySource::ExplicitDirectory("/".into()));
    let expman = source_kind_ordinal(&DiscoverySource::ExplicitManifest("/x.yaml".into()));
    let bundle = source_kind_ordinal(&DiscoverySource::HermeticBundle("/b".into()));

    assert!(dist < sys);
    assert!(sys < user);
    assert!(user < env);
    assert!(env < expdir);
    assert!(expdir < expman);
    assert!(expman < bundle);
}
