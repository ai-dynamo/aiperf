// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe discovery contracts for imported Codex CLI and Claude Code sessions.
//!
//! The discovery result is an exact authority which later parsing consumes
//! directly, rather than walking an authored history directory a second time.

mod discovery;

use std::error::Error;
use std::fmt::{self, Display};
use std::path::PathBuf;

pub use discovery::{detect_imported_agent_source, discover_imported_agent_read_set};

/// The provider-native session format selected for an imported recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedAgentSource {
    /// Codex CLI JSONL history.
    Codex,
    /// Claude Code JSONL history.
    ClaudeCode,
}

impl Display for ImportedAgentSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
        })
    }
}

/// The role a selected Claude Code source file plays in a session family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedSessionFamily {
    /// A top-level session file.
    Session,
    /// A direct Claude Code subagent session file.
    Subagent,
}

/// One canonical, root-contained source file selected for import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedAgentSourceFile {
    /// Canonical absolute source path.
    pub path: PathBuf,
    /// Slash-preserving path relative to the read-set root.
    pub relative_path: PathBuf,
    /// Whether this is a main session or direct subagent session.
    pub family: ImportedSessionFamily,
}

/// The complete immutable source authority for an imported session request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedAgentReadSet {
    /// Canonical root used to form each selected relative path.
    pub root: PathBuf,
    /// Canonical selected file or directory path.
    pub selected_path: PathBuf,
    /// The one provider format every selected file validated against.
    pub source: ImportedAgentSource,
    /// Ordered canonical source files. Directories may select no files.
    pub files: Vec<ImportedAgentSourceFile>,
}

/// A privacy-safe imported-session diagnostic.
///
/// Displayed errors contain only a path, line coordinate, fixed source kind,
/// fixed record label, and fixed detail; source values are never retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedAgentError {
    path: PathBuf,
    line: usize,
    source: &'static str,
    record_label: &'static str,
    detail: &'static str,
}

impl ImportedAgentError {
    pub(crate) fn new(
        path: &std::path::Path,
        line: usize,
        source: &'static str,
        record_label: &'static str,
        detail: &'static str,
    ) -> Self {
        Self {
            path: path.to_path_buf(),
            line,
            source,
            record_label,
            detail,
        }
    }
}

impl Display for ImportedAgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: line {}: source {}: record {}: {}",
            self.path.display(),
            self.line,
            self.source,
            self.record_label,
            self.detail
        )
    }
}

impl Error for ImportedAgentError {}
