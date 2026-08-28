// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static binary inspection tests.
//!
//! Tests use hand-crafted minimal binary byte sequences that goblin can parse.

use std::path::PathBuf;

use aiperf_plugin_host::acquire::AcquiredArtifact;
use aiperf_plugin_host::inspect::{statically_inspect, StaticInspectionError};

/// Build an `AcquiredArtifact` from raw bytes without going through the
/// acquire path (used for testing the inspection logic only).
fn make_artifact(raw_bytes: Vec<u8>, target: &str) -> AcquiredArtifact {
    let digest = blake3::hash(&raw_bytes).to_hex().to_string();
    AcquiredArtifact {
        raw_bytes,
        source_path: PathBuf::from("/fake/test.so"),
        digest,
        target: target.to_string(),
    }
}

// ── Minimal ELF64 x86_64 header (64 bytes) ────────────────────────────────
//
// goblin parses this header successfully as an ELF64 x86_64 shared object.
// There are no section or program headers, so there are no symbols, no dynamic
// section, and therefore no entry symbol and no DF_BIND_NOW.
fn elf64_x86_64_header() -> Vec<u8> {
    let mut h = vec![
        0x7f, 0x45, 0x4c, 0x46, // EI_MAG0..3
        0x02, // EI_CLASS: ELFCLASS64
        0x01, // EI_DATA: ELFDATA2LSB (little-endian)
        0x01, // EI_VERSION: EV_CURRENT
        0x00, // EI_OSABI: ELFOSABI_NONE
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // EI_ABIVERSION + padding
        0x03, 0x00, // e_type: ET_DYN
        0x3e, 0x00, // e_machine: EM_X86_64 = 62
        0x01, 0x00, 0x00, 0x00, // e_version: 1
        // e_entry (8 bytes):
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // e_phoff (8 bytes): 0 → no program headers
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // e_shoff (8 bytes): 0 → no section headers
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, // e_flags
        0x40, 0x00, // e_ehsize: 64
        0x38, 0x00, // e_phentsize: 56
        0x00, 0x00, // e_phnum: 0
        0x40, 0x00, // e_shentsize: 64
        0x00, 0x00, // e_shnum: 0
        0x00, 0x00, // e_shstrndx: 0
    ];
    assert_eq!(h.len(), 64);
    h
}

/// Same as above but with `e_machine` set to EM_AARCH64 (0xB7 = 183).
fn elf64_aarch64_header() -> Vec<u8> {
    let mut h = elf64_x86_64_header();
    // e_machine is at offset 18-19 (little-endian)
    h[18] = 0xB7;
    h[19] = 0x00;
    h
}

// ── test 1: garbage bytes → UnsupportedFormat ─────────────────────────────

#[test]
fn unsupported_format_rejected() {
    let artifact = make_artifact(b"not a binary at all XYZZY".to_vec(), "x86_64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::UnsupportedFormat(_)),
        "expected UnsupportedFormat, got {err:?}"
    );
}

// ── test 2: ELF arch mismatch → ArchMismatch ─────────────────────────────

#[test]
fn elf_arch_mismatch_is_rejected() {
    // x86_64 ELF but declared as aarch64.
    let artifact = make_artifact(elf64_x86_64_header(), "aarch64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::ArchMismatch { .. }),
        "expected ArchMismatch, got {err:?}"
    );
    if let StaticInspectionError::ArchMismatch { declared, detected } = err {
        assert_eq!(declared, "aarch64-unknown-linux-gnu");
        assert_eq!(detected, "x86_64");
    }
}

// ── test 3: ELF missing entry symbol → MissingEntrySymbol ─────────────────

#[test]
fn elf_missing_entry_symbol_is_rejected() {
    // Correct arch but no symbols → MissingEntrySymbol (checked before LazyBinding).
    let artifact = make_artifact(elf64_x86_64_header(), "x86_64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::MissingEntrySymbol),
        "expected MissingEntrySymbol, got {err:?}"
    );
}

// ── test 4: ELF aarch64 arch mismatch with aarch64 header ─────────────────

#[test]
fn elf_aarch64_declared_matches_aarch64_binary_arch() {
    // aarch64 ELF header + declared aarch64 — should fail on MissingEntrySymbol
    // (not ArchMismatch), proving arch detection is correct.
    let artifact = make_artifact(elf64_aarch64_header(), "aarch64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::MissingEntrySymbol),
        "expected MissingEntrySymbol (arch matched), got {err:?}"
    );
}

// ── test 5: ELF x86_64 declared as x86_64, wrong arch header → ArchMismatch

#[test]
fn elf_aarch64_binary_declared_as_x86_64_is_arch_mismatch() {
    // aarch64 ELF header declared as x86_64.
    let artifact = make_artifact(elf64_aarch64_header(), "x86_64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(err, StaticInspectionError::ArchMismatch { .. }),
        "expected ArchMismatch, got {err:?}"
    );
    if let StaticInspectionError::ArchMismatch { declared, detected } = err {
        assert_eq!(declared, "x86_64-unknown-linux-gnu");
        assert_eq!(detected, "aarch64");
    }
}

// ── test 6: empty bytes → UnsupportedFormat or ParseError ─────────────────

#[test]
fn empty_bytes_rejected() {
    let artifact = make_artifact(vec![], "x86_64-unknown-linux-gnu");
    let err = statically_inspect(&artifact).unwrap_err();
    assert!(
        matches!(
            err,
            StaticInspectionError::UnsupportedFormat(_) | StaticInspectionError::ParseError(_)
        ),
        "expected UnsupportedFormat or ParseError, got {err:?}"
    );
}
