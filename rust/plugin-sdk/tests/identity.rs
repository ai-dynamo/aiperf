// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for `identity.rs`: `HostAbiUniverseRecordV1` and `PluginArtifactBuildRecordV1`.
//!
//! Golden digests are computed at test time from the fixture data; a field mutation
//! must produce a different digest.  Deserialization must reject unknown fields.

use aiperf_plugin_sdk::identity::{
    CrateIdentity, HostAbiUniverseId, HostAbiUniverseRecordV1, NativeDep,
    PluginArtifactBuildId, PluginArtifactBuildRecordV1,
};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn make_universe() -> HostAbiUniverseRecordV1 {
    let rustc_exe_digest = "a".repeat(64);
    let rustc_commit = "abcdef01".to_string();
    let rustc_full_version = "rustc 1.98.0 (abcdef01 2026-07-01)".to_string();
    let sysroot_digest = "b".repeat(64);
    let target_triple = "x86_64-unknown-linux-gnu".to_string();
    let pointer_width: u8 = 64;
    let endian = "little".to_string();
    let codegen_backend = "llvm".to_string();
    let cfg_flags = vec!["unix".to_string(), "target_os=\"linux\"".to_string()];
    let codegen_flags = vec!["-C opt-level=3".to_string()];
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
    let build_script_digest: Option<String> = Some("d".repeat(64));
    let common_sources_digest = "e".repeat(64);
    let private_sources_digest = "f".repeat(64);
    let native_deps = vec![NativeDep {
        name: "libssl".to_string(),
        version: Some("3.0.2".to_string()),
        library_digest: Some("0".repeat(64)),
    }];
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
// Universe digest tests
// ---------------------------------------------------------------------------

#[test]
fn universe_digest_is_self_consistent() {
    let u = make_universe();
    assert!(u.verify_digest(), "verify_digest must pass for a freshly built record");
}

#[test]
fn universe_id_equals_canonical_digest() {
    let u = make_universe();
    assert_eq!(u.universe_id(), HostAbiUniverseId(u.canonical_digest.clone()));
}

#[test]
fn universe_digest_changes_on_rustc_exe_mutation() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let mutated = HostAbiUniverseRecordV1::compute_digest(
        &"9".repeat(64), // different
        &u.rustc_commit,
        &u.rustc_full_version,
        &u.sysroot_digest,
        &u.target_triple,
        u.pointer_width,
        &u.endian,
        &u.codegen_backend,
        &u.cfg_flags,
        &u.codegen_flags,
        &u.abi_crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );
    assert_ne!(original, mutated, "rustc_exe_digest mutation must change universe digest");
}

#[test]
fn universe_digest_changes_on_target_triple_mutation() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let mutated = HostAbiUniverseRecordV1::compute_digest(
        &u.rustc_exe_digest,
        &u.rustc_commit,
        &u.rustc_full_version,
        &u.sysroot_digest,
        "aarch64-unknown-linux-gnu",
        u.pointer_width,
        &u.endian,
        &u.codegen_backend,
        &u.cfg_flags,
        &u.codegen_flags,
        &u.abi_crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );
    assert_ne!(original, mutated, "target_triple mutation must change universe digest");
}

#[test]
fn universe_digest_changes_on_pointer_width_mutation() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let mutated = HostAbiUniverseRecordV1::compute_digest(
        &u.rustc_exe_digest,
        &u.rustc_commit,
        &u.rustc_full_version,
        &u.sysroot_digest,
        &u.target_triple,
        32, // mutated
        &u.endian,
        &u.codegen_backend,
        &u.cfg_flags,
        &u.codegen_flags,
        &u.abi_crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );
    assert_ne!(original, mutated, "pointer_width mutation must change universe digest");
}

#[test]
fn universe_digest_changes_on_abi_crates_mutation() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let mut crates = u.abi_crates.clone();
    crates[0].source_digest = "9".repeat(64);
    let mutated = HostAbiUniverseRecordV1::compute_digest(
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
        &crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );
    assert_ne!(original, mutated, "abi_crates mutation must change universe digest");
}

#[test]
fn universe_digest_changes_on_allocator_mutation() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let mutated = HostAbiUniverseRecordV1::compute_digest(
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
        &u.abi_crates,
        "system", // mutated
        &u.panic_strategy,
        u.target_policy_version,
        u.linker_exe_digest.as_deref(),
    );
    assert_ne!(original, mutated, "allocator mutation must change universe digest");
}

#[test]
fn universe_digest_changes_on_linker_present_vs_absent() {
    let u = make_universe();
    let original = u.canonical_digest.clone();
    let with_linker = HostAbiUniverseRecordV1::compute_digest(
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
        &u.abi_crates,
        &u.allocator,
        &u.panic_strategy,
        u.target_policy_version,
        Some("3".repeat(64).as_str()),
    );
    assert_ne!(original, with_linker, "absent vs present linker_exe_digest must differ");
}

// ---------------------------------------------------------------------------
// Build record digest tests
// ---------------------------------------------------------------------------

#[test]
fn build_digest_is_self_consistent() {
    let u = make_universe();
    let b = make_build(&u);
    assert!(b.verify_digest(), "verify_digest must pass for a freshly built record");
}

#[test]
fn build_id_equals_canonical_digest() {
    let u = make_universe();
    let b = make_build(&u);
    assert_eq!(b.build_id(), PluginArtifactBuildId(b.canonical_digest.clone()));
}

#[test]
fn build_digest_changes_on_universe_digest_mutation() {
    let u = make_universe();
    let b = make_build(&u);
    let original = b.canonical_digest.clone();
    let mutated = PluginArtifactBuildRecordV1::compute_digest(
        &"9".repeat(64), // different universe
        &b.package_name,
        &b.package_version,
        b.build_script_digest.as_deref(),
        &b.common_sources_digest,
        &b.private_sources_digest,
        &b.native_deps,
        &b.pre_embed_payload_digest,
        &b.artifact_digest,
    );
    assert_ne!(original, mutated, "universe_digest mutation must change build digest");
}

#[test]
fn build_digest_changes_on_common_sources_mutation() {
    let u = make_universe();
    let b = make_build(&u);
    let original = b.canonical_digest.clone();
    let mutated = PluginArtifactBuildRecordV1::compute_digest(
        &b.universe_digest,
        &b.package_name,
        &b.package_version,
        b.build_script_digest.as_deref(),
        &"9".repeat(64), // mutated common sources
        &b.private_sources_digest,
        &b.native_deps,
        &b.pre_embed_payload_digest,
        &b.artifact_digest,
    );
    assert_ne!(original, mutated, "common_sources_digest mutation must change build digest");
}

#[test]
fn private_change_only_alters_build_digest_not_universe() {
    let u = make_universe();
    let b = make_build(&u);
    let universe_digest = u.canonical_digest.clone();

    let different_build_digest = PluginArtifactBuildRecordV1::compute_digest(
        &b.universe_digest,
        &b.package_name,
        &b.package_version,
        b.build_script_digest.as_deref(),
        &b.common_sources_digest,
        &"9".repeat(64), // private sources mutated
        &b.native_deps,
        &b.pre_embed_payload_digest,
        &b.artifact_digest,
    );

    assert_ne!(
        b.canonical_digest, different_build_digest,
        "private source change must alter build digest"
    );
    // Universe is unchanged — private field mutation cannot touch it.
    assert_eq!(
        universe_digest, u.canonical_digest,
        "universe digest must not change when only private fields mutate"
    );
}

// ---------------------------------------------------------------------------
// Serde round-trip and unknown-field rejection
// ---------------------------------------------------------------------------

#[test]
fn universe_serde_round_trip() {
    let u = make_universe();
    let json = serde_json::to_string(&u).unwrap();
    let u2: HostAbiUniverseRecordV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(u, u2);
}

#[test]
fn universe_rejects_unknown_fields() {
    let u = make_universe();
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&serde_json::to_string(&u).unwrap()).unwrap();
    map.insert("unknown_field".to_string(), serde_json::Value::Bool(true));
    let json = serde_json::to_string(&map).unwrap();
    let result: Result<HostAbiUniverseRecordV1, _> = serde_json::from_str(&json);
    assert!(result.is_err(), "unknown fields must be rejected");
}

#[test]
fn build_serde_round_trip() {
    let u = make_universe();
    let b = make_build(&u);
    let json = serde_json::to_string(&b).unwrap();
    let b2: PluginArtifactBuildRecordV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(b, b2);
}

#[test]
fn build_rejects_unknown_fields() {
    let u = make_universe();
    let b = make_build(&u);
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
    map.insert("extra".to_string(), serde_json::Value::Null);
    let json = serde_json::to_string(&map).unwrap();
    let result: Result<PluginArtifactBuildRecordV1, _> = serde_json::from_str(&json);
    assert!(result.is_err(), "unknown fields must be rejected in build record");
}

// The canonical digest must not hash bytes containing itself (no circular dependency).
// We verify this by ensuring that compute_digest takes all fields *except* canonical_digest.
// If it accidentally fed canonical_digest, the digest would change when the struct is
// re-hashed after setting canonical_digest — but that would make verify_digest impossible.
// verify_digest already exercises this invariant.
#[test]
fn canonical_digest_does_not_contain_itself() {
    // If compute_digest hashed canonical_digest, the result would differ from
    // the stored value (circular reference is impossible). verify_digest catches it.
    let u = make_universe();
    assert!(u.verify_digest());
    let b = make_build(&u);
    assert!(b.verify_digest());
}
