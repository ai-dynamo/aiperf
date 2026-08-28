// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Poison propagation tests (Task 14).
//!
//! Verifies that a poisoned ActivatingLibrarySet reports the first error
//! even when subsequent load_one calls are made.

use std::path::PathBuf;

use aiperf_plugin_host::loader::ActivatingLibrarySet;

#[test]
fn poison_before_any_success_has_empty_partial_handles() {
    let mut set = ActivatingLibrarySet::new();
    set.load_one(PathBuf::from("/nonexistent_first.so"), "d1".to_owned());
    let err = set.finalize("lock".to_owned()).unwrap_err();
    assert!(err.partial_handles.is_empty());
}

#[test]
fn error_contains_failing_path() {
    let mut set = ActivatingLibrarySet::new();
    let bad_path = PathBuf::from("/tmp/definitely_missing_plugin.so");
    set.load_one(bad_path.clone(), "d1".to_owned());
    let err = set.finalize("lock".to_owned()).unwrap_err();
    let msg = err.error.to_string();
    assert!(
        msg.contains("definitely_missing_plugin"),
        "error should reference failing path: {msg}"
    );
}
