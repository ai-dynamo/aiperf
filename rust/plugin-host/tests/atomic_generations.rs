// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic generation tests: an install either publishes one complete immutable
//! generation or leaves the previous one untouched, and a concurrent reader
//! never observes a partially materialized generation.

use std::fs;

use aiperf_plugin_host::error::InstallError;
use aiperf_plugin_host::install::{InstallFile, InstallRoot, READY_MARKER};
use aiperf_plugin_host::inventory::AuthenticatedInventory;

fn inventory(build_id: &str) -> AuthenticatedInventory {
    AuthenticatedInventory::synthetic(build_id)
}

fn files(body: &[u8]) -> Vec<InstallFile> {
    vec![
        InstallFile::new("plugin.manifest.yaml", b"schema_version: \"2.0\"\n".to_vec()),
        InstallFile::new("lib/libplugin.so", body.to_vec()),
    ]
}

#[test]
fn a_fresh_install_publishes_one_complete_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    assert!(root.current().expect("current").is_none());

    let generation = root
        .atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install");

    assert_eq!(generation.id, 1);
    let current = root.current().expect("current").expect("a current generation");
    assert_eq!(current.id, generation.id);
    assert_eq!(
        fs::read(current.dir.join("lib/libplugin.so")).expect("read installed artifact"),
        b"one"
    );
    assert!(current.dir.join(READY_MARKER).exists());
}

#[test]
fn a_second_install_advances_current_and_retains_the_previous_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let first = root
        .atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install one");
    let second = root
        .atomic_install(&inventory("build-2"), &files(b"two"))
        .expect("install two");

    assert_eq!(second.id, first.id + 1);
    assert_eq!(
        root.current().expect("current").expect("current").id,
        second.id
    );
    assert_eq!(
        root.previous().expect("previous").expect("previous").id,
        first.id
    );
    // Both complete generations remain readable: a reader that resolved the old
    // pointer before the swap still sees intact bytes.
    assert_eq!(
        fs::read(first.dir.join("lib/libplugin.so")).expect("read old artifact"),
        b"one"
    );
    assert_eq!(
        fs::read(second.dir.join("lib/libplugin.so")).expect("read new artifact"),
        b"two"
    );
}

#[test]
fn a_generation_without_a_ready_marker_is_never_observed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let first = root
        .atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install one");

    // Simulate a crash after the generation directory landed but before the
    // ready marker was written.
    let orphan = root.generations_dir().join("7");
    fs::create_dir_all(orphan.join("lib")).expect("mkdir orphan");
    fs::write(orphan.join("lib/libplugin.so"), b"partial").expect("write partial");

    let ids = root.complete_generations().expect("complete generations");
    assert_eq!(
        ids,
        vec![first.id],
        "the marker-less generation must be invisible"
    );
    assert_eq!(
        root.current().expect("current").expect("current").id,
        first.id
    );
}

#[test]
fn a_crash_before_the_pointer_swap_leaves_the_previous_generation_current() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let first = root
        .atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install one");

    // Staging debris from an interrupted install must not affect resolution.
    let staged = root.staging_dir().join("in-flight");
    fs::create_dir_all(&staged).expect("mkdir staging");
    fs::write(staged.join("lib.so"), b"half").expect("write staged");

    assert_eq!(
        root.current().expect("current").expect("current").id,
        first.id
    );
    assert_eq!(
        fs::read(first.dir.join("lib/libplugin.so")).expect("read"),
        b"one"
    );
}

#[test]
fn rollback_restores_the_previous_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let first = root
        .atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install one");
    root.atomic_install(&inventory("build-2"), &files(b"two"))
        .expect("install two");

    let restored = root.rollback().expect("rollback");
    assert_eq!(restored.id, first.id);
    assert_eq!(
        root.current().expect("current").expect("current").id,
        first.id
    );
    assert_eq!(
        fs::read(restored.dir.join("lib/libplugin.so")).expect("read"),
        b"one"
    );
}

#[test]
fn rollback_without_a_previous_generation_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    root.atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install one");

    let err = root.rollback().expect_err("nothing to roll back to");
    assert!(
        matches!(err, InstallError::NoPreviousGeneration),
        "expected NoPreviousGeneration, got {err:?}"
    );
}

#[test]
fn garbage_collection_never_removes_the_current_or_previous_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    for n in 0..4 {
        root.atomic_install(&inventory(&format!("build-{n}")), &files(b"x"))
            .expect("install");
    }

    let removed = root.gc_old_generations(2).expect("gc");
    assert_eq!(
        removed,
        vec![1, 2],
        "only the oldest generations are collected"
    );
    assert_eq!(
        root.complete_generations().expect("generations"),
        vec![3, 4]
    );
    assert_eq!(root.current().expect("current").expect("current").id, 4);
    assert_eq!(root.previous().expect("previous").expect("previous").id, 3);
}

#[test]
fn an_escaping_relative_path_is_refused_before_anything_is_written() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");

    for bad in ["/etc/passwd", "../escape.so", "lib/../../escape.so"] {
        let err = root
            .atomic_install(
                &inventory("build-1"),
                &[InstallFile::new(bad, b"x".to_vec())],
            )
            .expect_err("escaping path must be refused");
        assert!(
            matches!(err, InstallError::InvalidRelativePath(_)),
            "expected InvalidRelativePath for {bad}, got {err:?}"
        );
    }
    assert!(root.current().expect("current").is_none());
    assert!(root.complete_generations().expect("generations").is_empty());
}

#[test]
fn a_generation_verifies_against_the_inventory_it_was_installed_from() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let inv = inventory("build-1");
    let generation = root.atomic_install(&inv, &files(b"one")).expect("install");

    root.verify_generation(generation.id, &inv)
        .expect("generation matches its own inventory");

    let err = root
        .verify_generation(generation.id, &inventory("build-other"))
        .expect_err("a different inventory must not verify");
    assert!(
        matches!(err, InstallError::InventoryDigestMismatch { .. }),
        "expected InventoryDigestMismatch, got {err:?}"
    );
}

#[test]
fn uninstall_removes_every_generation_and_the_current_pointer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root_path = dir.path().join("install");
    let root = InstallRoot::open(&root_path).expect("open install root");
    root.atomic_install(&inventory("build-1"), &files(b"one"))
        .expect("install");

    root.uninstall().expect("uninstall");
    assert!(!root_path.exists(), "the install root must be gone");
}
