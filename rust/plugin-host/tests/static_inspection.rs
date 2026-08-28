// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closure-level static inspection tests.
//!
//! `tests/inspection.rs` covers single-artifact inspection. These tests cover
//! `StaticallyValidatedCatalog::validate`, which inspects every artifact in an
//! acquired closure and fails on the first artifact that does not pass.

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_plugin_host::closure::AcquiredClosure;
use aiperf_plugin_host::inspect::{StaticInspectionError, StaticallyValidatedCatalog};

const HOST_TARGET: &str = "x86_64-unknown-linux-gnu";

/// Compute the BLAKE3 hex digest of `bytes`.
fn b3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Write a one-artifact `plugins.yaml` plus the artifact file it names, and
/// return the manifest path.
fn write_closure(dir: &Path, artifact_bytes: &[u8]) -> PathBuf {
    let artifact_path = dir.join("plugin.so");
    fs::write(&artifact_path, artifact_bytes).unwrap();
    let digest = b3(artifact_bytes);
    let yaml = format!(
        r#"schema_version: "2.0"
packages:
  - id: test-plugin
    version: 1.0.0
    categories:
      - category: exporter
        id: test-exporter
    artifacts:
      - target: {HOST_TARGET}
        path: "plugin.so"
        digest: "{digest}"
"#
    );
    let manifest_path = dir.join("plugins.yaml");
    fs::write(&manifest_path, yaml.as_bytes()).unwrap();
    manifest_path
}

/// Minimal ELF64 little-endian x86_64 `ET_DYN` header with no program or
/// section headers, so it parses but exports no symbols.
fn elf64_x86_64_header() -> Vec<u8> {
    let h = vec![
        0x7f, 0x45, 0x4c, 0x46, // EI_MAG0..3
        0x02, // EI_CLASS: ELFCLASS64
        0x01, // EI_DATA: ELFDATA2LSB
        0x01, // EI_VERSION: EV_CURRENT
        0x00, // EI_OSABI: ELFOSABI_NONE
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // EI_ABIVERSION + padding
        0x03, 0x00, // e_type: ET_DYN
        0x3e, 0x00, // e_machine: EM_X86_64
        0x01, 0x00, 0x00, 0x00, // e_version
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // e_entry
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // e_phoff
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // e_shoff
        0x00, 0x00, 0x00, 0x00, // e_flags
        0x40, 0x00, // e_ehsize
        0x38, 0x00, // e_phentsize
        0x00, 0x00, // e_phnum
        0x40, 0x00, // e_shentsize
        0x00, 0x00, // e_shnum
        0x00, 0x00, // e_shstrndx
    ];
    assert_eq!(h.len(), 64);
    h
}

/// An artifact whose bytes are not a recognized binary format fails closure
/// validation rather than being silently accepted.
#[test]
fn catalog_rejects_unrecognized_artifact_format() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = write_closure(tmp.path(), b"not a binary at all XYZZY");
    let closure = AcquiredClosure::acquire_from_manifest(&manifest_path, &[HOST_TARGET]).unwrap();
    assert_eq!(closure.artifacts.len(), 1);

    let err = StaticallyValidatedCatalog::validate(closure).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::UnsupportedFormat(_)),
        "expected UnsupportedFormat, got {err:?}"
    );
}

/// A parseable ELF artifact that does not export the plugin entry symbol fails
/// closure validation.
#[test]
fn catalog_rejects_elf_without_entry_symbol() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = write_closure(tmp.path(), &elf64_x86_64_header());
    let closure = AcquiredClosure::acquire_from_manifest(&manifest_path, &[HOST_TARGET]).unwrap();

    let err = StaticallyValidatedCatalog::validate(closure).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::MissingEntrySymbol),
        "expected MissingEntrySymbol, got {err:?}"
    );
}

/// A closure that selected no artifacts validates successfully and yields an
/// empty catalog; inspection never invents artifacts.
#[test]
fn catalog_accepts_closure_with_no_selected_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = write_closure(tmp.path(), b"never inspected");
    let closure =
        AcquiredClosure::acquire_from_manifest(&manifest_path, &["s390x-unknown-linux-gnu"])
            .unwrap();
    assert!(closure.artifacts.is_empty());

    let catalog = StaticallyValidatedCatalog::validate(closure).unwrap();
    assert!(catalog.artifacts.is_empty());
}
