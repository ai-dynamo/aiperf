// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for `abi_closure.rs`: `AbiClosureRecordV1` validation.

use aiperf_plugin_sdk::abi_closure::{
    AbiClosureError, AbiClosureRecordV1, AbiClosureRevocationRecordV1,
};
use aiperf_plugin_sdk::identity::{
    CrateIdentity, HostAbiUniverseRecordV1, NativeDep, PluginArtifactBuildRecordV1,
};

fn make_universe() -> HostAbiUniverseRecordV1 {
    let rustc_exe_digest = "a".repeat(64);
    let rustc_commit = "abc01234".to_string();
    let rustc_full_version = "rustc 1.98.0 (abc01234 2026-07-01)".to_string();
    let sysroot_digest = "b".repeat(64);
    let target_triple = "x86_64-unknown-linux-gnu".to_string();
    let pointer_width: u8 = 64;
    let endian = "little".to_string();
    let codegen_backend = "llvm".to_string();
    let cfg_flags = vec!["unix".to_string()];
    let codegen_flags: Vec<String> = vec![];
    let abi_crates = vec![CrateIdentity {
        name: "aiperf-core".to_string(),
        version: "0.1.0".to_string(),
        source_digest: "c".repeat(64),
    }];
    let allocator = "aiperf-allocator-provider".to_string();
    let panic_strategy = "abort".to_string();
    let target_policy_version: u32 = 1;
    let linker_exe_digest: Option<String> = None;
    let canonical_digest = HostAbiUniverseRecordV1::compute_digest(
        &rustc_exe_digest,
        &rustc_commit,
        &rustc_full_version,
        &sysroot_digest,
        &target_triple,
        pointer_width,
        &endian,
        &codegen_backend,
        &cfg_flags,
        &codegen_flags,
        &abi_crates,
        &allocator,
        &panic_strategy,
        target_policy_version,
        linker_exe_digest.as_deref(),
    );
    HostAbiUniverseRecordV1 {
        rustc_exe_digest,
        rustc_commit,
        rustc_full_version,
        sysroot_digest,
        target_triple,
        pointer_width,
        endian,
        codegen_backend,
        cfg_flags,
        codegen_flags,
        abi_crates,
        allocator,
        panic_strategy,
        target_policy_version,
        linker_exe_digest,
        canonical_digest,
    }
}

fn make_build(universe: &HostAbiUniverseRecordV1) -> PluginArtifactBuildRecordV1 {
    let universe_digest = universe.canonical_digest.clone();
    let package_name = "my-plugin".to_string();
    let package_version = "0.1.0".to_string();
    let build_script_digest: Option<String> = None;
    let common_sources_digest = "e".repeat(64);
    let private_sources_digest = "f".repeat(64);
    let native_deps: Vec<NativeDep> = vec![];
    let pre_embed_payload_digest = "1".repeat(64);
    let artifact_digest = "2".repeat(64);
    let canonical_digest = PluginArtifactBuildRecordV1::compute_digest(
        &universe_digest,
        &package_name,
        &package_version,
        build_script_digest.as_deref(),
        &common_sources_digest,
        &private_sources_digest,
        &native_deps,
        &pre_embed_payload_digest,
        &artifact_digest,
    );
    PluginArtifactBuildRecordV1 {
        universe_digest,
        package_name,
        package_version,
        build_script_digest,
        common_sources_digest,
        private_sources_digest,
        native_deps,
        pre_embed_payload_digest,
        artifact_digest,
        canonical_digest,
    }
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn valid_closure_validates() {
    let u = make_universe();
    let b = make_build(&u);
    let closure = AbiClosureRecordV1 {
        universe: u,
        builds: vec![b],
    };
    assert!(
        closure.validate().is_ok(),
        "a correctly constructed closure must validate"
    );
}

#[test]
fn empty_builds_list_validates() {
    let u = make_universe();
    let closure = AbiClosureRecordV1 {
        universe: u,
        builds: vec![],
    };
    assert!(closure.validate().is_ok(), "empty builds list is valid");
}

// ---------------------------------------------------------------------------
// Universe digest mismatch
// ---------------------------------------------------------------------------

#[test]
fn corrupted_universe_digest_is_rejected() {
    let mut u = make_universe();
    u.canonical_digest = "0".repeat(64); // corrupt
    let closure = AbiClosureRecordV1 {
        universe: u,
        builds: vec![],
    };
    let err = closure.validate().unwrap_err();
    assert!(
        matches!(err, AbiClosureError::UniverseDigestMismatch { .. }),
        "corrupted universe digest must produce UniverseDigestMismatch"
    );
}

// ---------------------------------------------------------------------------
// Build universe_digest mismatch
// ---------------------------------------------------------------------------

#[test]
fn build_referencing_wrong_universe_is_rejected() {
    let u = make_universe();
    let mut b = make_build(&u);
    b.universe_digest = "9".repeat(64); // wrong
    let closure = AbiClosureRecordV1 {
        universe: u,
        builds: vec![b],
    };
    let err = closure.validate().unwrap_err();
    assert!(
        matches!(
            err,
            AbiClosureError::BuildUniverseMismatch { build_index: 0, .. }
        ),
        "build with wrong universe_digest must produce BuildUniverseMismatch"
    );
}

// ---------------------------------------------------------------------------
// Build canonical_digest mismatch
// ---------------------------------------------------------------------------

#[test]
fn corrupted_build_digest_is_rejected() {
    let u = make_universe();
    let mut b = make_build(&u);
    b.canonical_digest = "0".repeat(64); // corrupt
    let closure = AbiClosureRecordV1 {
        universe: u,
        builds: vec![b],
    };
    let err = closure.validate().unwrap_err();
    assert!(
        matches!(
            err,
            AbiClosureError::BuildDigestMismatch { build_index: 0, .. }
        ),
        "corrupted build canonical_digest must produce BuildDigestMismatch"
    );
}

// ---------------------------------------------------------------------------
// Revocation record serde
// ---------------------------------------------------------------------------

#[test]
fn revocation_record_round_trips() {
    let rev = AbiClosureRevocationRecordV1 {
        revoked_universe_digest: "a".repeat(64),
        reason: "cfg_flags reclassified as common".to_string(),
        replacement_universe_digest: Some("b".repeat(64)),
    };
    let json = serde_json::to_string(&rev).unwrap();
    let rev2: AbiClosureRevocationRecordV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(rev, rev2);
}

#[test]
fn revocation_record_rejects_unknown_fields() {
    let rev = AbiClosureRevocationRecordV1 {
        revoked_universe_digest: "a".repeat(64),
        reason: "test".to_string(),
        replacement_universe_digest: None,
    };
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&serde_json::to_string(&rev).unwrap()).unwrap();
    map.insert("extra_field".to_string(), serde_json::Value::Bool(false));
    let json = serde_json::to_string(&map).unwrap();
    let result: Result<AbiClosureRevocationRecordV1, _> = serde_json::from_str(&json);
    assert!(
        result.is_err(),
        "unknown fields must be rejected in revocation record"
    );
}

// ---------------------------------------------------------------------------
// Reclassification semantics (documented behavior, not code-level test)
// ---------------------------------------------------------------------------

/// When a private field is reclassified as common, the universe digest changes,
/// revoking all prior build IDs.  We demonstrate this by showing that adding
/// a field to `abi_crates` (common scope) produces a different universe digest,
/// making all existing build records invalid for the new universe.
#[test]
fn reclassification_changes_universe_digest() {
    let u = make_universe();
    let original_universe_digest = u.canonical_digest.clone();

    // Simulate reclassification: add a new crate to abi_crates.
    let mut new_abi_crates = u.abi_crates.clone();
    new_abi_crates.push(CrateIdentity {
        name: "newly-common-crate".to_string(),
        version: "0.1.0".to_string(),
        source_digest: "d".repeat(64),
    });
    let new_universe_digest = HostAbiUniverseRecordV1::compute_digest(
        &u.rustc_exe_digest,
        &u.rustc_commit,
        &u.rustc_full_version,
        &u.sysroot_digest,
        &u.target_triple,
        u.pointer_width,
        &u.endian,
        &u.codegen_backend,
        &u.cfg_flags,
        &u.codegen_flags,
        &new_abi_crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );

    assert_ne!(
        original_universe_digest, new_universe_digest,
        "reclassification (adding a crate to abi_crates) must change the universe digest \
         and thereby revoke all prior build IDs"
    );
}
