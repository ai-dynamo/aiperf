// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural inspection of plugin artifact binaries.
//!
//! `inspect_artifact` uses `nm` to enumerate exported symbols and
//! `artifact_section::extract_record` to retrieve the embedded build record,
//! returning a structured `InspectReport`.

use std::path::Path;
use std::process::Command;

use crate::artifact_section::{self, ArtifactSectionError};
use crate::identity::PluginArtifactBuildRecordV1;

/// Required entry-point symbol that every valid plugin cdylib must export.
pub const REQUIRED_ENTRY_SYMBOL: &str = "aiperf_plugin_entry_v1";

/// Report from inspecting one plugin artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectReport {
    /// Whether the exported symbol set matches requirements.
    pub symbols_ok: bool,
    /// Symbols from `REQUIRED_ENTRY_SYMBOL` that are absent in the artifact.
    pub missing_symbols: Vec<String>,
    /// Exported symbols that are unexpected (informational, not an error).
    pub extra_symbols: Vec<String>,
    /// The build record embedded in the artifact section, if present and valid.
    pub embedded_record: Option<PluginArtifactBuildRecordV1>,
}

/// Errors from `inspect_artifact`.
#[derive(Debug)]
pub enum InspectError {
    ArtifactSection(ArtifactSectionError),
    NmFailed(String),
    Io(std::io::Error),
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectError::ArtifactSection(e) => write!(f, "section read error: {e}"),
            InspectError::NmFailed(e) => write!(f, "nm failed: {e}"),
            InspectError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for InspectError {}

impl From<ArtifactSectionError> for InspectError {
    fn from(e: ArtifactSectionError) -> Self {
        InspectError::ArtifactSection(e)
    }
}

impl From<std::io::Error> for InspectError {
    fn from(e: std::io::Error) -> Self {
        InspectError::Io(e)
    }
}

/// Inspects the plugin artifact at `artifact_path`.
///
/// Runs `nm --dynamic --defined-only --extern-only` to enumerate exported
/// symbols, then calls `extract_record` to retrieve the embedded build record.
pub fn inspect_artifact(artifact_path: &Path) -> Result<InspectReport, InspectError> {
    let out = Command::new("nm")
        .args(["--dynamic", "--defined-only", "--extern-only"])
        .arg(artifact_path)
        .output()?;

    let mut exported: std::collections::HashSet<String> = std::collections::HashSet::new();
    if out.status.success() {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            // nm output: `<addr> <type> <name>` or `<type> <name>` (for undefined).
            // We take the last whitespace-delimited token as the symbol name.
            if let Some(sym) = line.split_whitespace().last() {
                exported.insert(sym.to_owned());
            }
        }
    }
    // nm returning non-zero usually means the file is not a shared library;
    // we treat it as an empty export set and let missing_symbols reflect that.

    let required = [REQUIRED_ENTRY_SYMBOL];
    let missing_symbols: Vec<String> = required
        .iter()
        .filter(|s| !exported.contains(**s))
        .map(|s| s.to_string())
        .collect();

    // Symbols beyond the required set (informational).
    let extra_symbols: Vec<String> = exported
        .iter()
        .filter(|s| !required.contains(&s.as_str()))
        .cloned()
        .collect();

    let symbols_ok = missing_symbols.is_empty();
    let embedded_record = artifact_section::extract_record(artifact_path)?;

    Ok(InspectReport {
        symbols_ok,
        missing_symbols,
        extra_symbols,
        embedded_record,
    })
}
