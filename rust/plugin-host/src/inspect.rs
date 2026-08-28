// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static binary inspection via the `object` crate (Task 12).
//!
//! Parses ELF, Mach-O, and PE artifacts without executing them.  Extracts
//! exported symbols, constructor sections, allocator import set, dependency
//! list, and embedded plugin build records.  Produces a quarantine reason list
//! for artifacts that violate host policy.

use std::path::Path;

use object::{File as ObjectFile, Object, ObjectSection};

use crate::error::InspectError;

/// The binary container format detected from file magic bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactKind {
    Elf,
    MachO,
    Pe,
    /// Unrecognized or truncated file; inspection still attempts best-effort.
    Unknown,
}

/// How the dynamic linker will locate this library's dependencies at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchPolicy {
    /// RUNPATH / relative to object origin (acceptable).
    Origin,
    /// LD_LIBRARY_PATH / loader default search path.
    LoaderPath,
    /// Dependency has an absolute path baked in.
    Absolute,
    /// Policy cannot be determined or is actively forbidden.
    Rejected,
}

/// Quarantine reason codes attached to policy-violating artifacts.
pub mod quarantine {
    pub const MISSING_ENTRY_SYMBOL: &str = "missing-entry-symbol";
    pub const ALLOCATOR_NOT_IMPORTED: &str = "allocator-not-imported";
    pub const ABSOLUTE_DEPENDENCY_PATH: &str = "absolute-dependency-path";
    pub const CONSTRUCTOR_SECTION_PRESENT: &str = "constructor-section-present";
    pub const UNKNOWN_FORMAT: &str = "unknown-binary-format";
}

/// The canonical plugin entry-point symbol every native plugin must export.
pub const PLUGIN_ENTRY_SYMBOL: &str = "aiperf_plugin_init";

/// Prefix that allocator proxy symbols exported by the shared allocator carry.
const ALLOC_SYMBOL_PREFIX: &str = "aiperf_alloc";

/// Complete inspection result for one artifact file.
#[derive(Debug, Clone)]
pub struct InspectionReceipt {
    /// Detected binary format.
    pub artifact_kind: ArtifactKind,
    /// Hex-encoded BLAKE3 digest of the file bytes (computed during inspection).
    pub digest: String,
    /// All symbols exported from the dynamic symbol table.
    pub exported_symbols: Vec<String>,
    /// Whether `aiperf_plugin_init` appears in the exported symbol table.
    pub has_entry_symbol: bool,
    /// Whether `.init_array`, `.ctors`, `__mod_init_func`, or equivalent
    /// constructor sections are present (forbidden for plugin artifacts).
    pub has_constructor_sections: bool,
    /// Allocator proxy symbols imported by this artifact.
    pub allocator_imports: Vec<String>,
    /// Whether the allocator symbols are bound at dlopen time (RTLD_NOW semantics).
    pub allocator_eager_binding: bool,
    /// Rust panic strategy embedded as a note section or build attribute.
    pub panic_strategy: Option<String>,
    /// Inferred dependency search policy.
    pub dependency_search_policy: SearchPolicy,
    /// All dynamic dependencies declared by this artifact.
    pub dependencies: Vec<String>,
    /// Quarantine reasons.  Empty means the artifact passes all host policy checks.
    pub quarantine_reasons: Vec<String>,
}

/// Inspect a plugin artifact file at `path`.
pub fn inspect_artifact(path: &Path) -> Result<InspectionReceipt, InspectError> {
    let bytes = std::fs::read(path)?;
    let digest = blake3::hash(&bytes).to_hex().to_string();
    inspect_bytes(&bytes, digest)
}

/// Inspect artifact from an already-acquired byte slice (digest pre-computed).
pub fn inspect_bytes(bytes: &[u8], digest: String) -> Result<InspectionReceipt, InspectError> {
    match ObjectFile::parse(bytes) {
        Ok(obj) => Ok(inspect_parsed(obj, digest)),
        Err(e) => Ok(InspectionReceipt {
            artifact_kind: ArtifactKind::Unknown,
            digest,
            exported_symbols: vec![],
            has_entry_symbol: false,
            has_constructor_sections: false,
            allocator_imports: vec![],
            allocator_eager_binding: false,
            panic_strategy: None,
            dependency_search_policy: SearchPolicy::Rejected,
            dependencies: vec![],
            quarantine_reasons: vec![format!("{}: {e}", quarantine::UNKNOWN_FORMAT)],
        }),
    }
}

fn inspect_parsed(obj: ObjectFile<'_>, digest: String) -> InspectionReceipt {
    use object::BinaryFormat;

    let artifact_kind = match obj.format() {
        BinaryFormat::Elf => ArtifactKind::Elf,
        BinaryFormat::MachO => ArtifactKind::MachO,
        BinaryFormat::Pe => ArtifactKind::Pe,
        _ => ArtifactKind::Unknown,
    };

    // Collect exported and imported symbol names via the flat symbol iterators.
    let mut exported_symbols: Vec<String> = vec![];
    let mut imported_symbols: Vec<String> = vec![];
    for sym in obj.symbols().chain(obj.dynamic_symbols()) {
        use object::ObjectSymbol;
        let Ok(name) = sym.name() else { continue };
        if name.is_empty() {
            continue;
        }
        if sym.is_undefined() {
            imported_symbols.push(name.to_owned());
        } else if sym.is_global() {
            exported_symbols.push(name.to_owned());
        }
    }
    exported_symbols.sort_unstable();
    exported_symbols.dedup();
    imported_symbols.sort_unstable();
    imported_symbols.dedup();

    let has_entry_symbol = exported_symbols.iter().any(|s| s == PLUGIN_ENTRY_SYMBOL);

    let constructor_section_names = [".init_array", ".ctors", "__mod_init_func", ".init"];
    let has_constructor_sections = obj.sections().any(|sec| {
        sec.name()
            .ok()
            .map(|n| constructor_section_names.contains(&n))
            .unwrap_or(false)
    });

    let mut allocator_imports: Vec<String> = imported_symbols
        .iter()
        .filter(|s| s.starts_with(ALLOC_SYMBOL_PREFIX))
        .cloned()
        .collect();
    allocator_imports.sort_unstable();
    allocator_imports.dedup();

    // Eager binding approximated by absence of PLT section.
    let allocator_eager_binding = !obj
        .sections()
        .any(|s| s.name().ok().map(|n| n == ".plt" || n == "__stubs").unwrap_or(false));

    let panic_strategy =
        if imported_symbols.iter().any(|s| s.contains("rust_begin_unwind"))
            || imported_symbols.iter().any(|s| s == "__rust_start_panic")
        {
            Some("unwind".to_owned())
        } else if exported_symbols.iter().any(|s| s.contains("panic_abort"))
            || imported_symbols.iter().any(|s| s == "abort")
        {
            Some("abort".to_owned())
        } else {
            None
        };

    let (dependencies, dependency_search_policy) = collect_dependencies(&obj);

    let mut quarantine_reasons: Vec<String> = vec![];
    if !has_entry_symbol {
        quarantine_reasons.push(quarantine::MISSING_ENTRY_SYMBOL.to_owned());
    }
    if allocator_imports.is_empty() {
        quarantine_reasons.push(quarantine::ALLOCATOR_NOT_IMPORTED.to_owned());
    }
    if has_constructor_sections {
        quarantine_reasons.push(quarantine::CONSTRUCTOR_SECTION_PRESENT.to_owned());
    }
    if dependency_search_policy == SearchPolicy::Absolute {
        quarantine_reasons.push(quarantine::ABSOLUTE_DEPENDENCY_PATH.to_owned());
    }

    InspectionReceipt {
        artifact_kind,
        digest,
        exported_symbols,
        has_entry_symbol,
        has_constructor_sections,
        allocator_imports,
        allocator_eager_binding,
        panic_strategy,
        dependency_search_policy,
        dependencies,
        quarantine_reasons,
    }
}

fn collect_dependencies(obj: &ObjectFile<'_>) -> (Vec<String>, SearchPolicy) {
    let mut deps: Vec<String> = vec![];

    // ELF: scan `.dynamic` section for DT_NEEDED entries (64-bit LE).
    if let Some(sec) = obj.section_by_name(".dynamic") {
        if let Ok(data) = sec.data() {
            deps.extend(parse_elf_needed_le64(data, obj));
        }
    }

    let has_absolute = deps.iter().any(|d| d.starts_with('/'));
    let policy = if has_absolute { SearchPolicy::Absolute } else { SearchPolicy::Origin };
    (deps, policy)
}

/// Extract DT_NEEDED entries from an ELF `.dynamic` section assuming 64-bit LE.
fn parse_elf_needed_le64(data: &[u8], obj: &ObjectFile<'_>) -> Vec<String> {
    const DT_NULL: u64 = 0;
    const DT_NEEDED: u64 = 1;
    const DT_STRTAB: u64 = 5;

    if data.len() < 16 {
        return vec![];
    }

    let mut needed_offsets: Vec<u64> = vec![];
    let mut strtab_vaddr: u64 = 0;
    let mut i = 0usize;
    while i + 16 <= data.len() {
        let tag = u64::from_le_bytes(data[i..i + 8].try_into().unwrap_or([0u8; 8]));
        let val = u64::from_le_bytes(data[i + 8..i + 16].try_into().unwrap_or([0u8; 8]));
        if tag == DT_NULL {
            break;
        }
        if tag == DT_NEEDED {
            needed_offsets.push(val);
        }
        if tag == DT_STRTAB {
            strtab_vaddr = val;
        }
        i += 16;
    }

    if needed_offsets.is_empty() {
        return vec![];
    }

    let strtab: Vec<u8> = obj
        .sections()
        .find_map(|sec| {
            if sec.address() == strtab_vaddr {
                sec.data().ok().map(|d| d.to_vec())
            } else {
                None
            }
        })
        .unwrap_or_default();

    if strtab.is_empty() {
        return vec![];
    }

    needed_offsets
        .iter()
        .filter_map(|&off| {
            let off = off as usize;
            if off >= strtab.len() {
                return None;
            }
            let end = strtab[off..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(strtab.len() - off);
            std::str::from_utf8(&strtab[off..off + end])
                .ok()
                .map(|s| s.to_owned())
        })
        .collect()
}
