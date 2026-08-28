// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static binary inspection of acquired plugin artifacts.
//!
//! Verifies that each artifact's binary format (ELF/Mach-O/PE) matches the
//! declared target triple, that the required `aiperf_plugin_entry_v1` symbol
//! is exported, and that ELF artifacts carry eager-binding flags.

use thiserror::Error;

use crate::acquire::{AcquiredArtifact, AcquiredClosure};

const ENTRY_SYMBOL: &str = "aiperf_plugin_entry_v1";
/// Mach-O exports prepend an underscore.
const ENTRY_SYMBOL_MACHO: &str = "_aiperf_plugin_entry_v1";

/// Static inspection result for one artifact.
#[derive(Debug)]
pub struct InspectedArtifact {
    /// Declared target triple.
    pub target: String,
    /// Whether `aiperf_plugin_entry_v1` is present in the symbol table.
    pub entry_symbol_present: bool,
    /// ELF: whether `DF_BIND_NOW` or `DF_1_NOW` is set. Always `true` for
    /// non-ELF formats where eager binding is the platform default.
    pub bind_now: bool,
    /// Whether the binary's architecture matches the declared target triple.
    pub arch_matches: bool,
}

/// Errors produced by static artifact inspection.
#[derive(Debug, Error)]
pub enum StaticInspectionError {
    #[error("unsupported binary format for artifact: {0}")]
    UnsupportedFormat(String),

    #[error("binary parse error: {0}")]
    ParseError(String),

    #[error("architecture mismatch: declared {declared}, binary reports {detected}")]
    ArchMismatch { declared: String, detected: String },

    #[error("required export `aiperf_plugin_entry_v1` not found in artifact")]
    MissingEntrySymbol,

    #[error("ELF artifact does not set DF_BIND_NOW / DF_1_NOW; lazy binding is forbidden")]
    LazyBinding,
}

impl From<goblin::error::Error> for StaticInspectionError {
    fn from(e: goblin::error::Error) -> Self {
        StaticInspectionError::ParseError(e.to_string())
    }
}

/// Map a target triple's arch component to the ELF `e_machine` value.
fn elf_machine_for_target(target: &str) -> Option<u16> {
    if target.starts_with("x86_64") {
        Some(goblin::elf::header::EM_X86_64)
    } else if target.starts_with("aarch64") {
        Some(goblin::elf::header::EM_AARCH64)
    } else if target.starts_with("i686") || target.starts_with("i386") {
        Some(goblin::elf::header::EM_386)
    } else if target.starts_with("arm") {
        Some(goblin::elf::header::EM_ARM)
    } else if target.starts_with("riscv64") {
        Some(goblin::elf::header::EM_RISCV)
    } else if target.starts_with("s390x") {
        Some(goblin::elf::header::EM_S390)
    } else {
        None
    }
}

fn arch_name_from_elf_machine(machine: u16) -> &'static str {
    match machine {
        goblin::elf::header::EM_X86_64 => "x86_64",
        goblin::elf::header::EM_AARCH64 => "aarch64",
        goblin::elf::header::EM_386 => "i686",
        goblin::elf::header::EM_ARM => "arm",
        goblin::elf::header::EM_RISCV => "riscv64",
        goblin::elf::header::EM_S390 => "s390x",
        _ => "unknown",
    }
}

/// Mach-O CPU type constants.
const CPU_TYPE_X86_64: u32 = 0x0100_0007;
const CPU_TYPE_ARM64: u32 = 0x0100_000C;
const CPU_TYPE_X86: u32 = 7;
const CPU_TYPE_ARM: u32 = 12;

fn macho_cputype_for_target(target: &str) -> Option<u32> {
    if target.starts_with("x86_64") {
        Some(CPU_TYPE_X86_64)
    } else if target.starts_with("aarch64") {
        Some(CPU_TYPE_ARM64)
    } else if target.starts_with("i686") || target.starts_with("i386") {
        Some(CPU_TYPE_X86)
    } else if target.starts_with("arm") {
        Some(CPU_TYPE_ARM)
    } else {
        None
    }
}

fn arch_name_from_macho_cputype(cputype: u32) -> &'static str {
    match cputype {
        CPU_TYPE_X86_64 => "x86_64",
        CPU_TYPE_ARM64 => "aarch64",
        CPU_TYPE_X86 => "i686",
        CPU_TYPE_ARM => "arm",
        _ => "unknown",
    }
}

/// PE machine constants from the COFF spec.
const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;
const IMAGE_FILE_MACHINE_I386: u16 = 0x014C;
const IMAGE_FILE_MACHINE_ARMNT: u16 = 0x01C4;

fn pe_machine_for_target(target: &str) -> Option<u16> {
    if target.starts_with("x86_64") {
        Some(IMAGE_FILE_MACHINE_AMD64)
    } else if target.starts_with("aarch64") {
        Some(IMAGE_FILE_MACHINE_ARM64)
    } else if target.starts_with("i686") || target.starts_with("i386") {
        Some(IMAGE_FILE_MACHINE_I386)
    } else if target.starts_with("arm") {
        Some(IMAGE_FILE_MACHINE_ARMNT)
    } else {
        None
    }
}

fn arch_name_from_pe_machine(machine: u16) -> &'static str {
    match machine {
        IMAGE_FILE_MACHINE_AMD64 => "x86_64",
        IMAGE_FILE_MACHINE_ARM64 => "aarch64",
        IMAGE_FILE_MACHINE_I386 => "i686",
        IMAGE_FILE_MACHINE_ARMNT => "arm",
        _ => "unknown",
    }
}

/// Inspect one acquired artifact statically.
///
/// Parses the binary format, verifies architecture against the declared target
/// triple, checks for the required entry symbol, and (for ELF) verifies eager
/// binding.
pub fn statically_inspect(
    artifact: &AcquiredArtifact,
) -> Result<InspectedArtifact, StaticInspectionError> {
    use goblin::Object;

    let obj = Object::parse(&artifact.raw_bytes)?;
    match obj {
        Object::Elf(elf) => inspect_elf(&elf, &artifact.target),
        Object::Mach(mach) => inspect_mach(&mach, &artifact.target),
        Object::PE(pe) => inspect_pe(&pe, &artifact.target),
        _ => Err(StaticInspectionError::UnsupportedFormat(
            artifact.target.clone(),
        )),
    }
}

fn inspect_elf(
    elf: &goblin::elf::Elf<'_>,
    target: &str,
) -> Result<InspectedArtifact, StaticInspectionError> {
    use goblin::elf::dynamic::{DF_1_NOW, DF_BIND_NOW, DT_FLAGS, DT_FLAGS_1};

    // Verify architecture.
    let expected_machine = elf_machine_for_target(target).unwrap_or(u16::MAX);
    let arch_matches = elf.header.e_machine == expected_machine;
    if !arch_matches {
        let detected = arch_name_from_elf_machine(elf.header.e_machine).to_string();
        return Err(StaticInspectionError::ArchMismatch {
            declared: target.to_string(),
            detected,
        });
    }

    // Check for eager binding.
    let bind_now = if let Some(dynamic) = &elf.dynamic {
        dynamic.dyns.iter().any(|d| {
            (d.d_tag == DT_FLAGS && d.d_val & DF_BIND_NOW as u64 != 0)
                || (d.d_tag == DT_FLAGS_1 && d.d_val & DF_1_NOW as u64 != 0)
        })
    } else {
        // No dynamic section: if it has no dynamic section at all, it cannot
        // be dlopen-ed meaningfully, but we treat absence as lazy (not eager).
        false
    };

    // Check for entry symbol in dynsyms and syms.
    let entry_symbol_present = elf.dynsyms.iter().any(|sym| {
        elf.dynstrtab
            .get_at(sym.st_name)
            .map_or(false, |n| n == ENTRY_SYMBOL)
    }) || elf.syms.iter().any(|sym| {
        elf.strtab
            .get_at(sym.st_name)
            .map_or(false, |n| n == ENTRY_SYMBOL)
    });

    if !entry_symbol_present {
        return Err(StaticInspectionError::MissingEntrySymbol);
    }
    if !bind_now {
        return Err(StaticInspectionError::LazyBinding);
    }

    Ok(InspectedArtifact {
        target: target.to_string(),
        entry_symbol_present: true,
        bind_now: true,
        arch_matches: true,
    })
}

fn inspect_mach(
    mach: &goblin::mach::Mach<'_>,
    target: &str,
) -> Result<InspectedArtifact, StaticInspectionError> {
    use goblin::mach::Mach;

    let macho = match mach {
        Mach::Binary(m) => m,
        Mach::Fat(fat) => {
            // Pick the slice matching the target arch.
            let expected_cputype = macho_cputype_for_target(target).unwrap_or(u32::MAX);
            let mut found = None;
            for arch in fat.iter_arches().flatten() {
                if arch.cputype() as u32 == expected_cputype {
                    found = fat.get(0).ok();
                    break;
                }
            }
            match found {
                Some(m) => {
                    // Recurse on the selected slice.
                    let mach_single = Mach::Binary(m);
                    return inspect_mach(&mach_single, target);
                }
                None => {
                    let detected = "unknown (fat without matching arch)".to_string();
                    return Err(StaticInspectionError::ArchMismatch {
                        declared: target.to_string(),
                        detected,
                    });
                }
            }
        }
    };

    // Verify architecture.
    let expected_cputype = macho_cputype_for_target(target).unwrap_or(u32::MAX);
    let arch_matches = macho.header.cputype as u32 == expected_cputype;
    if !arch_matches {
        let detected = arch_name_from_macho_cputype(macho.header.cputype as u32).to_string();
        return Err(StaticInspectionError::ArchMismatch {
            declared: target.to_string(),
            detected,
        });
    }

    // Check for entry symbol in exports.
    let entry_symbol_present = macho
        .exports()
        .unwrap_or_default()
        .iter()
        .any(|e| e.name == ENTRY_SYMBOL || e.name == ENTRY_SYMBOL_MACHO);

    if !entry_symbol_present {
        return Err(StaticInspectionError::MissingEntrySymbol);
    }

    Ok(InspectedArtifact {
        target: target.to_string(),
        entry_symbol_present: true,
        bind_now: true, // Mach-O bind-at-load is enforced by build flags, not inspected here.
        arch_matches: true,
    })
}

fn inspect_pe(
    pe: &goblin::pe::PE<'_>,
    target: &str,
) -> Result<InspectedArtifact, StaticInspectionError> {
    // Verify architecture.
    let expected_machine = pe_machine_for_target(target).unwrap_or(u16::MAX);
    let arch_matches = pe.header.coff_header.machine == expected_machine;
    if !arch_matches {
        let detected = arch_name_from_pe_machine(pe.header.coff_header.machine).to_string();
        return Err(StaticInspectionError::ArchMismatch {
            declared: target.to_string(),
            detected,
        });
    }

    // Check export directory.
    let entry_symbol_present = pe
        .exports
        .iter()
        .any(|e| e.name.map_or(false, |n| n == ENTRY_SYMBOL));

    if !entry_symbol_present {
        return Err(StaticInspectionError::MissingEntrySymbol);
    }

    Ok(InspectedArtifact {
        target: target.to_string(),
        entry_symbol_present: true,
        bind_now: true, // PE IAT is always eager; no lazy-binding check required.
        arch_matches: true,
    })
}

/// An artifact that has passed static binary inspection.
#[derive(Debug)]
pub struct StaticallyValidatedArtifact {
    pub acquired: AcquiredArtifact,
    pub inspected: InspectedArtifact,
}

/// A complete closure where every artifact has been statically validated.
#[derive(Debug)]
pub struct StaticallyValidatedCatalog {
    pub artifacts: Vec<StaticallyValidatedArtifact>,
}

impl StaticallyValidatedCatalog {
    /// Statically inspect every artifact in the closure.
    ///
    /// Returns an error on the first artifact that fails inspection.
    pub fn validate(closure: AcquiredClosure) -> Result<Self, StaticInspectionError> {
        let mut validated = Vec::with_capacity(closure.artifacts.len());
        for artifact in closure.artifacts {
            let inspected = statically_inspect(&artifact)?;
            validated.push(StaticallyValidatedArtifact {
                acquired: artifact,
                inspected,
            });
        }
        Ok(StaticallyValidatedCatalog {
            artifacts: validated,
        })
    }
}
