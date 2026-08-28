// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic generation installation, rollback, and garbage collection.

use std::fs;

use aiperf_plugin_host::install::{gc_generations, install_generation, rollback_to_generation};

#[test]
fn install_generation_creates_expected_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let artifacts: Vec<(String, &[u8])> = vec![
        ("libplugin.so".to_string(), b"ELF-bytes".as_slice()),
        ("plugin.yaml".to_string(), b"schema_version: 2.0".as_slice()),
    ];
    let generation = install_generation(root, 1, &artifacts).expect("install_generation");

    assert_eq!(generation.generation, 1);
    assert_eq!(generation.root, root.join("generations").join("1"));
    assert_eq!(
        fs::read(generation.root.join("libplugin.so")).expect("read artifact"),
        b"ELF-bytes"
    );
    assert_eq!(
        fs::read(generation.root.join("plugin.yaml")).expect("read manifest"),
        b"schema_version: 2.0"
    );
    let marker = fs::read_to_string(generation.root.join("generation.marker")).expect("marker");
    assert_eq!(marker.trim(), "1");
    // Staging must not survive a successful install.
    assert!(!root.join("staging").join("1").exists());
}

#[test]
fn rollback_to_old_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let g1: Vec<(String, &[u8])> = vec![("libplugin.so".to_string(), b"v1".as_slice())];
    let g2: Vec<(String, &[u8])> = vec![("libplugin.so".to_string(), b"v2".as_slice())];
    install_generation(root, 1, &g1).expect("install gen 1");
    install_generation(root, 2, &g2).expect("install gen 2");

    rollback_to_generation(root, 2).expect("point at gen 2");
    let current = fs::read_to_string(root.join("current")).expect("current");
    assert_eq!(current.trim(), root.join("generations").join("2").display().to_string());

    rollback_to_generation(root, 1).expect("rollback to gen 1");
    let current = fs::read_to_string(root.join("current")).expect("current");
    assert_eq!(current.trim(), root.join("generations").join("1").display().to_string());

    // Rolling back to a generation that was never installed must fail closed.
    assert!(rollback_to_generation(root, 7).is_err());
}

#[test]
fn gc_removes_old_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let g1: Vec<(String, &[u8])> = vec![("libplugin.so".to_string(), b"v1".as_slice())];
    let g2: Vec<(String, &[u8])> = vec![("libplugin.so".to_string(), b"v2".as_slice())];
    install_generation(root, 1, &g1).expect("install gen 1");
    install_generation(root, 2, &g2).expect("install gen 2");
    rollback_to_generation(root, 2).expect("point at gen 2");

    gc_generations(root, &[2]).expect("gc");

    assert!(!root.join("generations").join("1").exists());
    assert!(root.join("generations").join("2").exists());
}

#[test]
fn gc_never_removes_the_current_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let g1: Vec<(String, &[u8])> = vec![("libplugin.so".to_string(), b"v1".as_slice())];
    install_generation(root, 1, &g1).expect("install gen 1");
    rollback_to_generation(root, 1).expect("point at gen 1");

    // An empty keep set still must not collect the live generation.
    gc_generations(root, &[]).expect("gc");
    assert!(root.join("generations").join("1").exists());
}
