// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe discovery contracts for imported Codex CLI and Claude Code sessions.
//!
//! The discovery result is an exact authority which later parsing consumes
//! directly, rather than walking an authored history directory a second time.

mod claude_code;
mod codex;
mod discovery;
mod lowering;

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::path::PathBuf;

use bytes::Bytes;

use crate::config::model::dataset::RecordedAgentSourceFormat;

pub use claude_code::parse_claude_session;
pub use codex::parse_codex_session;
pub use discovery::{detect_imported_agent_source, discover_imported_agent_read_set};
pub use lowering::lower_imported_agent_sessions;

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

/// The complete exact source authority for an imported session request.
///
/// Cellular controllers materialize selected files in a private scratch root, so a
/// run can retain stable source paths without retaining full session histories in RAM.
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

/// Private immutable acquisition backing one imported-session selection.
pub struct AcquiredImportedAgentSelection {
    pub(crate) scratch: tempfile::TempDir,
    pub(crate) read_set: ImportedAgentReadSet,
}

/// A configuration-only request to acquire one imported session selection.
///
/// Constructing this request intentionally performs no filesystem inspection.
/// Acquisition opens every selected source exactly once and retains those
/// descriptors until their contents have been copied into private scratch.
pub struct ImportedAgentSelectionRequest {
    path: PathBuf,
    replay_root: Option<PathBuf>,
    source_format: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
}

impl ImportedAgentSelectionRequest {
    /// Validate source-format compatibility without reading the requested path.
    pub(crate) fn new(
        path: PathBuf,
        replay_root: Option<PathBuf>,
        source_format: RecordedAgentSourceFormat,
        include_subagents: Option<bool>,
    ) -> Result<Self, ImportedAgentError> {
        if source_format == RecordedAgentSourceFormat::MiniSweAgent {
            return Err(ImportedAgentError::new(
                &path,
                0,
                "unknown",
                "unknown",
                "Mini-SWE-Agent is not an imported session source",
            ));
        }
        if include_subagents.is_some() && source_format == RecordedAgentSourceFormat::Codex {
            return Err(ImportedAgentError::new(
                &path,
                0,
                "codex",
                "unknown",
                "include_subagents applies only to Claude Code sources",
            ));
        }
        Ok(Self {
            path,
            replay_root,
            source_format,
            include_subagents,
        })
    }

    /// Acquire into a private temporary directory.
    pub fn acquire(self) -> Result<AcquiredImportedAgentSelection, ImportedAgentError> {
        discovery::acquire_selection(self, None)
    }

    /// Acquire into a private temporary directory below `parent`.
    pub fn acquire_in(
        self,
        parent: &std::path::Path,
    ) -> Result<AcquiredImportedAgentSelection, ImportedAgentError> {
        discovery::acquire_selection(self, Some(parent))
    }
}

impl AcquiredImportedAgentSelection {
    /// Return the scratch-backed exact source authority.
    pub fn read_set(&self) -> &ImportedAgentReadSet {
        debug_assert!(self.read_set.root.starts_with(self.scratch.path()));
        &self.read_set
    }
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

/// One session's canonical request messages and per-call prefix boundaries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportedRequestHistory {
    messages: Vec<RawJsonMessage>,
    call_end_offsets: Vec<usize>,
}

impl ImportedRequestHistory {
    pub(crate) fn push(&mut self, message: RawJsonMessage) {
        self.messages.push(message);
    }

    pub(crate) fn capture_call(&mut self) {
        self.call_end_offsets.push(self.messages.len());
    }

    pub(crate) fn truncate_after_last_call(&mut self) {
        let retained_len = self.call_end_offsets.last().copied().unwrap_or(0);
        self.messages.truncate(retained_len);
    }

    fn messages_for_call(&self, call_index: usize) -> Option<&[RawJsonMessage]> {
        let end = *self.call_end_offsets.get(call_index)?;
        self.messages.get(..end)
    }

    fn is_legacy_fallback(&self) -> bool {
        self.messages.is_empty() && self.call_end_offsets.is_empty()
    }
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
    /// Legacy/direct-construction request history used when session history is absent.
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
    /// Shared canonical request history for parser-produced calls.
    ///
    /// `Default::default()` selects each call's legacy `request_messages` vector.
    pub request_history: ImportedRequestHistory,
    /// Inferred model calls in source order.
    pub calls: Vec<ImportedModelCall>,
    /// Number of observed function-call records.
    pub observed_tool_count: u64,
    /// Number of observed tool uses with a correlated result.
    pub completed_tool_count: u64,
    /// Number of ignored additive records or unsupported content blocks.
    pub ignored_record_count: u64,
    /// Number of omitted reasoning records.
    pub omitted_reasoning_count: u64,
    /// Whether every retained tool bundle included all results.
    pub tool_results_complete: bool,
}

impl ImportedAgentSession {
    /// Borrow the exact request-message prefix for one inferred model call.
    pub fn request_messages(&self, call_index: usize) -> Option<&[RawJsonMessage]> {
        let call = self.calls.get(call_index)?;
        if self.request_history.is_legacy_fallback() {
            return Some(call.request_messages.as_slice());
        }
        self.request_history.messages_for_call(call_index)
    }
}

/// Parse the exact discovered source set into deterministically ordered sessions.
///
/// Claude main sessions are parsed before direct subagent sessions. A subagent is
/// accepted only when its stable parent tool-use identifier resolves to exactly
/// one `Task` call in its declared parent session.
pub fn parse_imported_agent_sessions(
    read_set: &ImportedAgentReadSet,
) -> Result<Vec<ImportedAgentSession>, ImportedAgentError> {
    match read_set.source {
        ImportedAgentSource::Codex => parse_codex_read_set(read_set),
        ImportedAgentSource::ClaudeCode => parse_claude_read_set(read_set),
    }
}

fn parse_codex_read_set(
    read_set: &ImportedAgentReadSet,
) -> Result<Vec<ImportedAgentSession>, ImportedAgentError> {
    let mut sessions = Vec::with_capacity(read_set.files.len());
    for file in &read_set.files {
        if file.family != ImportedSessionFamily::Session {
            return Err(ImportedAgentError::new(
                &file.path,
                0,
                "codex",
                "message",
                "subagent files are invalid for Codex session imports",
            ));
        }
        sessions.push(parse_codex_session(file)?);
    }
    validate_unique_session_ids(sessions)
}

fn parse_claude_read_set(
    read_set: &ImportedAgentReadSet,
) -> Result<Vec<ImportedAgentSession>, ImportedAgentError> {
    let mut main_files = Vec::new();
    let mut subagent_files = Vec::new();
    for file in &read_set.files {
        match file.family {
            ImportedSessionFamily::Session => main_files.push(file),
            ImportedSessionFamily::Subagent => subagent_files.push(file),
        }
    }

    let mut sessions = Vec::with_capacity(read_set.files.len());
    let mut task_parents = HashMap::new();
    let mut observed_tool_parents = HashSet::new();
    for file in main_files {
        let parsed = claude_code::parse_claude_session_details(file)?;
        let session_id = parsed.session.session_id.clone();
        let task_ids = parsed.task_tool_use_ids;
        for tool_use_id in &parsed.all_tool_use_ids {
            observed_tool_parents.insert((session_id.clone(), tool_use_id.clone()));
        }
        for tool_use_id in task_ids {
            let key = (session_id.clone(), tool_use_id);
            *task_parents.entry(key).or_insert(0_usize) += 1;
        }
        sessions.push(parsed.session);
    }

    let mut linked_parents = HashSet::new();
    for file in subagent_files {
        let parsed = claude_code::parse_claude_session_details(file)?;
        let mut session = parsed.session;
        let parent = session
            .parent
            .as_ref()
            .ok_or_else(|| read_set_error(file, "missing parent tool-use identifier"))?;
        let parent_key = (session.session_id.clone(), parent.tool_use_id.clone());
        match task_parents.get(&parent_key).copied() {
            Some(1) => {}
            Some(_) => {
                return Err(read_set_error(
                    file,
                    "parent tool-use identifier matches multiple main sessions",
                ));
            }
            None => {
                let detail = if observed_tool_parents.contains(&parent_key) {
                    "parent tool-use does not identify a Task call"
                } else {
                    "parent tool-use identifier not found"
                };
                return Err(read_set_error(file, detail));
            }
        }
        if !linked_parents.insert(parent_key.clone()) {
            return Err(read_set_error(
                file,
                "multiple subagent files identify one parent Task call",
            ));
        }
        let sibling_id = format!("{}#sa#{}", parent_key.0, parent_key.1);
        if !is_valid_identifier(&sibling_id) {
            return Err(read_set_error(
                file,
                "derived subagent session identifier is invalid",
            ));
        }
        session.session_id = sibling_id;
        session.parent = Some(ImportedSubagentParent {
            session_id: parent_key.0,
            tool_use_id: parent_key.1,
        });
        sessions.push(session);
    }
    validate_unique_session_ids(sessions)
}

fn validate_unique_session_ids(
    sessions: Vec<ImportedAgentSession>,
) -> Result<Vec<ImportedAgentSession>, ImportedAgentError> {
    let mut session_ids = HashSet::new();
    for session in &sessions {
        if !session_ids.insert(&session.session_id) {
            let source = match session.source {
                ImportedAgentSource::Codex => "codex",
                ImportedAgentSource::ClaudeCode => "claude_code",
            };
            return Err(ImportedAgentError::new(
                &session.source_path,
                0,
                source,
                "message",
                "duplicate imported session identifier",
            ));
        }
    }
    Ok(sessions)
}

fn read_set_error(file: &ImportedAgentSourceFile, detail: &'static str) -> ImportedAgentError {
    ImportedAgentError::new(&file.path, 0, "claude_code", "message", detail)
}

fn is_valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'#' | b'-')
        })
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::{ImportedRequestHistory, RawJsonMessage};

    #[test]
    fn imported_request_history_retains_one_message_vector_and_call_end_offsets() {
        let mut history = ImportedRequestHistory::default();
        history.capture_call();
        for index in 0..1_024 {
            history.push(RawJsonMessage {
                role: "user".to_owned(),
                wire: Bytes::from(format!(r#"{{"role":"user","content":"{index}"}}"#)),
            });
            history.capture_call();
        }

        assert_eq!(history.messages.len(), 1_024);
        assert_eq!(history.call_end_offsets.len(), 1_025);
        assert_eq!(history.call_end_offsets[0], 0);
        assert_eq!(history.call_end_offsets[1], 1);
        assert_eq!(history.call_end_offsets[1_024], 1_024);
        assert_eq!(
            history.messages_for_call(1_024).expect("last prefix")[0].wire,
            Bytes::from_static(b"{\"role\":\"user\",\"content\":\"0\"}")
        );
        assert_eq!(
            history.messages_for_call(1_024).expect("last prefix")[1_023].wire,
            Bytes::from_static(b"{\"role\":\"user\",\"content\":\"1023\"}")
        );

        let mut incomplete = ImportedRequestHistory::default();
        incomplete.push(RawJsonMessage {
            role: "user".to_owned(),
            wire: Bytes::from_static(b"{\"role\":\"user\",\"content\":\"x\"}"),
        });
        assert!(!incomplete.is_legacy_fallback());
        assert!(incomplete.messages_for_call(0).is_none());
    }
}
