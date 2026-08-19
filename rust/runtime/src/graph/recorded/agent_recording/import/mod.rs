// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe discovery contracts for imported Codex CLI and Claude Code sessions.
//!
//! The discovery result is an exact authority which later parsing consumes
//! directly, rather than walking an authored history directory a second time.

mod codex;
mod discovery;

use std::error::Error;
use std::fmt::{self, Display};
use std::path::PathBuf;

use bytes::Bytes;

pub use codex::parse_codex_session;
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

/// One canonical JSON request-history message retained as immutable wire bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawJsonMessage {
    /// The validated provider-neutral message role.
    pub role: String,
    /// Canonical serialized JSON object for the message.
    pub wire: Bytes,
}

/// The parent main-session identity for a linked imported subagent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedSubagentParent {
    /// Validated parent session identifier.
    pub session_id: String,
    /// Validated parent tool-use identifier.
    pub tool_use_id: String,
}

/// One inferred model invocation extracted from provider-native history.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedModelCall {
    /// Privacy-safe source coordinate for this inferred invocation.
    pub source_id: String,
    /// Canonical request history sent to the model.
    pub request_messages: Vec<RawJsonMessage>,
    /// Provider model identifier when the source provided one.
    pub model: Option<String>,
    /// Observed tool-bundle duration in microseconds when available.
    pub delay_after_previous_us: Option<f64>,
    /// Whether the source captured the tool schema.
    pub tool_schema_available: bool,
    /// Provider-reported generated token count when available.
    pub output_tokens: Option<u64>,
}

/// One provider-native session normalized into the non-executable import IR.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedAgentSession {
    /// Validated provider session identifier.
    pub session_id: String,
    /// Provider-native source selected during exact discovery.
    pub source: ImportedAgentSource,
    /// Canonical source file path.
    pub source_path: PathBuf,
    /// Lowercase BLAKE3 digest over exact JSONL bytes.
    pub source_digest: String,
    /// Provider model identifier when present.
    pub model: Option<String>,
    /// Codex base instruction message when present.
    pub system_prompt: Option<RawJsonMessage>,
    /// Whether the source recorded a working directory, without retaining it.
    pub cwd_present: bool,
    /// Whether the source recorded a git branch, without retaining it.
    pub git_branch_present: bool,
    /// Optional parent link for imported subagent files.
    pub parent: Option<ImportedSubagentParent>,
    /// Inferred model calls in source order.
    pub calls: Vec<ImportedModelCall>,
    /// Number of observed function-call records.
    pub observed_tool_count: u64,
    /// Number of ignored additive records or unsupported content blocks.
    pub ignored_record_count: u64,
    /// Number of omitted reasoning records.
    pub omitted_reasoning_count: u64,
    /// Whether every retained tool bundle included all results.
    pub tool_results_complete: bool,
}
