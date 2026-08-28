// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loader tests (Task 14).
//!
//! dlopen calls against real shared libraries would need a compiled artifact;
//! these tests cover the ActivatingLibrarySet state machine and error paths
//! using a non-existent path (expected to fail dlopen).

use std::path::PathBuf;

use aiperf_plugin_host::loader::ActivatingLibrarySet;

/// A successful finalize with no loads produces an empty LoadedLibrarySet.
#[test]
fn empty_set_finalizes_ok() {
    let set = ActivatingLibrarySet::new();
    let loaded = set
        .finalize("empty-lock-digest".to_owned())
        .expect("empty set should finalize ok");
    assert!(loaded.handles.is_empty());
    assert_eq!(loaded.lock_digest, "empty-lock-digest");
}

/// Loading a non-existent path poisons the set.
#[test]
fn nonexistent_path_poisons() {
    let mut set = ActivatingLibrarySet::new();
    set.load_one(
        PathBuf::from("/tmp/aiperf_definitely_does_not_exist_99999.so"),
        "fake-digest".to_owned(),
    );
    let result = set.finalize("lock".to_owned());
    assert!(
        result.is_err(),
        "dlopen of missing file must poison the set"
    );
}

/// Once poisoned, finalize returns the error regardless of no further loads.
#[test]
fn poison_is_sticky() {
    let mut set = ActivatingLibrarySet::new();
    set.load_one(PathBuf::from("/tmp/nonexistent.so"), "d1".to_owned());
    // No more loads, but poison should still be there.
    let result = set.finalize("lock".to_owned());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.partial_handles.is_empty(),
        "no partial handles for zero successful loads"
    );
}
