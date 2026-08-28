// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for cellular plugin lock attestation (Task 21).
//!
//! The plugin lock digest must flow: controller env → `CellLaunchContext` →
//! cell subprocess env → cell verification before any effects. These tests
//! cover the runtime struct surface and the propagation helpers.

use std::collections::BTreeMap;

use aiperf_runtime::engine::cell_launcher::{CellLaunchContext, LocalLauncher, CELL_PLUGIN_LOCK_ENV};

/// `CellLaunchContext` exposes a `plugin_lock_digest` field; when `None` the
/// cell command does not inject the env variable.
#[test]
#[cfg(feature = "cellular")]
fn cell_command_omits_env_when_no_digest() {
    let ctx = CellLaunchContext {
        cell_count: 2,
        controller_coordinate: "file:/tmp/coord".to_owned(),
        phase_ordinal_bases: BTreeMap::new(),
        artifact_authority: None,
        local_roles: None,
        plugin_lock_digest: None,
    };
    let launcher = LocalLauncher;
    let cmd = launcher.cell_command(&ctx, 0);
    // The debug repr is the only stable way to inspect a tokio::process::Command's
    // environment in tests without spawning it.
    let dbg = format!("{cmd:?}");
    assert!(
        !dbg.contains(CELL_PLUGIN_LOCK_ENV),
        "plugin lock env must be absent when digest is None; got: {dbg}"
    );
}

/// When `plugin_lock_digest` is `Some`, `cell_command` injects
/// `AIPERF_PLUGIN_LOCK_DIGEST` with the hex string.
#[test]
#[cfg(feature = "cellular")]
fn cell_command_injects_digest_when_present() {
    let digest = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
    let ctx = CellLaunchContext {
        cell_count: 2,
        controller_coordinate: "file:/tmp/coord".to_owned(),
        phase_ordinal_bases: BTreeMap::new(),
        artifact_authority: None,
        local_roles: None,
        plugin_lock_digest: Some(digest.to_owned()),
    };
    let launcher = LocalLauncher;
    let cmd = launcher.cell_command(&ctx, 0);
    let dbg = format!("{cmd:?}");
    assert!(
        dbg.contains(CELL_PLUGIN_LOCK_ENV),
        "plugin lock env key missing; got: {dbg}"
    );
    assert!(
        dbg.contains(digest),
        "plugin lock digest value missing; got: {dbg}"
    );
}

/// `CellRegister` carries a `plugin_lock_digest` field that round-trips through
/// `rmp_serde` serialization; a `None` value is skipped (default).
#[test]
fn cell_register_digest_roundtrips_msgpack() {
    use aiperf_runtime::cellular::transport::CellRegister;

    let with_digest = CellRegister {
        cell_id: 0,
        cell_peer: vec![],
        artifact_capability_digest: None,
        registration_proof: None,
        plugin_lock_digest: Some("deadbeef".repeat(8)),
    };
    let bytes = rmp_serde::to_vec(&with_digest).expect("encode");
    let decoded: CellRegister = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded.plugin_lock_digest, with_digest.plugin_lock_digest);

    // `None` must decode from a payload that omits the field (forward compat).
    let without = CellRegister {
        cell_id: 0,
        cell_peer: vec![],
        artifact_capability_digest: None,
        registration_proof: None,
        plugin_lock_digest: None,
    };
    let bytes_none = rmp_serde::to_vec(&without).expect("encode none");
    let decoded_none: CellRegister = rmp_serde::from_slice(&bytes_none).expect("decode none");
    assert!(decoded_none.plugin_lock_digest.is_none());
}

/// `verify_propagated_digest` accepts matching digests and rejects mismatches.
#[test]
fn verify_propagated_digest_matches_and_rejects() {
    use aiperf_cli::plugins::propagate::verify_propagated_digest;

    let hex = "aabbcc112233";
    assert!(
        verify_propagated_digest(hex, hex).is_ok(),
        "matching digests must succeed"
    );
    let result = verify_propagated_digest(hex, "different");
    assert!(result.is_err(), "mismatching digests must fail");
}
