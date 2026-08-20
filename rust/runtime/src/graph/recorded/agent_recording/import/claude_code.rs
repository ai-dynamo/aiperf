// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-native Claude Code session normalization.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};

use bytes::Bytes;
use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};

use super::{
    ImportedAgentError, ImportedAgentSession, ImportedAgentSource, ImportedAgentSourceFile,
    ImportedModelCall, ImportedSessionFamily, ImportedSubagentParent, RawJsonMessage,
};

/// Parse one exact-discovery Claude Code source file into the provider-native import IR.
pub fn parse_claude_session(
    file: &ImportedAgentSourceFile,
) -> Result<ImportedAgentSession, ImportedAgentError> {
    Ok(parse_claude_session_details(file)?.session)
}

/// Parse one Claude source file and retain correlation metadata for read-set linking.
pub(crate) fn parse_claude_session_details(
    file: &ImportedAgentSourceFile,
) -> Result<ParsedClaudeSession, ImportedAgentError> {
    let source = std::fs::File::open(&file.path)
        .map_err(|_| error(&file.path, 0, "unknown", "cannot read source file"))?;
    let mut reader = BufReader::new(source);
    let mut state = ClaudeState::new(file);
    let mut hasher = blake3::Hasher::new();
    let mut line_bytes = Vec::new();
    let mut line = 0;
    loop {
        line_bytes.clear();
        let read = reader.read_until(b'\n', &mut line_bytes).map_err(|_| {
            error(
                &file.path,
                line.max(1),
                "unknown",
                "cannot read source file",
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&line_bytes);
        line += 1;
        if line_bytes.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value: Value = serde_json::from_slice(&line_bytes)
            .map_err(|_| error(&file.path, line, "unknown", "invalid JSON"))?;
        let record = value
            .as_object()
            .ok_or_else(|| error(&file.path, line, "unknown", "record must be a JSON object"))?;
        state.accept(record, line)?;
    }
    state.finish(hasher.finalize().to_hex().to_string())
}

struct PendingClaudeMessage {
    source_id: String,
    line: usize,
    content: Vec<Value>,
    authored_content: Vec<Value>,
    text_positions: Vec<usize>,
    tool_ids: HashSet<String>,
}

struct NormalizedAssistantBlocks {
    content: Vec<Value>,
    full_content: Vec<Value>,
    fresh_tool_ids: HashSet<String>,
}

pub(crate) struct ParsedClaudeSession {
    pub(crate) session: ImportedAgentSession,
    pub(crate) task_tool_use_ids: Vec<String>,
    pub(crate) all_tool_use_ids: Vec<String>,
}

struct ClaudeState<'a> {
    file: &'a ImportedAgentSourceFile,
    session_id: Option<String>,
    model: Option<String>,
    cwd_present: bool,
    git_branch_present: bool,
    parent_tool_use_id: Option<String>,
    history: Vec<RawJsonMessage>,
    calls: Vec<ImportedModelCall>,
    pending: Option<PendingClaudeMessage>,
    finalized_messages: HashMap<String, Vec<Value>>,
    seen_tool_blocks: HashMap<String, Vec<u8>>,
    seen_result_ids: HashSet<String>,
    open_tool_ids: HashSet<String>,
    first_open_tool_timestamp: Option<Option<DateTime<FixedOffset>>>,
    last_result_timestamp: Option<DateTime<FixedOffset>>,
    next_model_delay_after_previous_us: Option<f64>,
    observed_tool_count: u64,
    completed_tool_count: u64,
    ignored_record_count: u64,
    omitted_reasoning_count: u64,
    tool_results_complete: bool,
    task_tool_use_ids: Vec<String>,
    seen_task_tool_use_ids: HashSet<String>,
    all_tool_use_ids: HashSet<String>,
}

impl<'a> ClaudeState<'a> {
    fn new(file: &'a ImportedAgentSourceFile) -> Self {
        Self {
            file,
            session_id: None,
            model: None,
            cwd_present: false,
            git_branch_present: false,
            parent_tool_use_id: None,
            history: Vec::new(),
            calls: Vec::new(),
            pending: None,
            finalized_messages: HashMap::new(),
            seen_tool_blocks: HashMap::new(),
            seen_result_ids: HashSet::new(),
            open_tool_ids: HashSet::new(),
            first_open_tool_timestamp: None,
            last_result_timestamp: None,
            next_model_delay_after_previous_us: None,
            observed_tool_count: 0,
            completed_tool_count: 0,
            ignored_record_count: 0,
            omitted_reasoning_count: 0,
            tool_results_complete: true,
            task_tool_use_ids: Vec::new(),
            seen_task_tool_use_ids: HashSet::new(),
            all_tool_use_ids: HashSet::new(),
        }
    }

    fn accept(
        &mut self,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let record_type = record.get("type").and_then(Value::as_str);
        if !matches!(record_type, Some("user" | "assistant")) {
            self.ignored_record_count += 1;
            return Ok(());
        }
        let is_sidechain = record
            .get("isSidechain")
            .and_then(Value::as_bool)
            .ok_or_else(|| error(&self.file.path, line, "message", "missing sidechain marker"))?;
        let expected_sidechain = self.file.family == ImportedSessionFamily::Subagent;
        if is_sidechain != expected_sidechain {
            self.ignored_record_count += 1;
            return Ok(());
        }
        self.accept_metadata(record, line)?;
        let message = record
            .get("message")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                error(
                    &self.file.path,
                    line,
                    "message",
                    "message must be an object",
                )
            })?;
        match record_type {
            Some("user") => self.accept_user(message, record, line),
            Some("assistant") => self.accept_assistant(message, record, line),
            _ => Ok(()),
        }
    }

    fn accept_metadata(
        &mut self,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let session_id = required_identifier(
            record,
            "sessionId",
            &self.file.path,
            line,
            "message",
            "invalid session identifier",
        )?;
        if let Some(previous) = &self.session_id {
            if previous != &session_id {
                return Err(error(
                    &self.file.path,
                    line,
                    "message",
                    "inconsistent session identifier",
                ));
            }
        } else {
            self.session_id = Some(session_id);
        }
        if self.file.family == ImportedSessionFamily::Subagent {
            if let Some(value) = record.get("parentToolUseId") {
                let parent = validated_identifier_value(
                    value,
                    &self.file.path,
                    line,
                    "message",
                    "invalid parent tool-use identifier",
                )?;
                if let Some(previous) = &self.parent_tool_use_id {
                    if previous != &parent {
                        return Err(error(
                            &self.file.path,
                            line,
                            "message",
                            "inconsistent parent tool-use identifier",
                        ));
                    }
                } else {
                    self.parent_tool_use_id = Some(parent);
                }
            } else if self.parent_tool_use_id.is_none() {
                return Err(error(
                    &self.file.path,
                    line,
                    "message",
                    "missing parent tool-use identifier",
                ));
            }
        }
        self.cwd_present |= record.contains_key("cwd");
        self.git_branch_present |= record.contains_key("gitBranch");
        Ok(())
    }

    fn accept_user(
        &mut self,
        message: &Map<String, Value>,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        ensure_role(message, "user", &self.file.path, line)?;
        let content = message
            .get("content")
            .ok_or_else(|| error(&self.file.path, line, "user", "missing message content"))?;
        match content {
            Value::String(text) => {
                self.flush_pending()?;
                self.abandon_open_tools();
                self.history.push(raw_value_message(
                    "user",
                    json!({"role": "user", "content": text}),
                    &self.file.path,
                    line,
                )?);
                Ok(())
            }
            Value::Array(blocks) => self.accept_tool_results(blocks, record, line),
            _ => Err(error(
                &self.file.path,
                line,
                "user",
                "unsupported user content",
            )),
        }
    }

    fn accept_tool_results(
        &mut self,
        blocks: &[Value],
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        self.flush_pending()?;
        let mut retained = Vec::new();
        for block in blocks {
            let Some(object) = block.as_object() else {
                self.ignored_record_count += 1;
                continue;
            };
            if object.get("type").and_then(Value::as_str) != Some("tool_result") {
                self.ignored_record_count += 1;
                continue;
            }
            let tool_use_id = required_identifier(
                object,
                "tool_use_id",
                &self.file.path,
                line,
                "tool_result",
                "invalid tool-use identifier",
            )?;
            if !self.seen_result_ids.insert(tool_use_id.clone()) {
                return Err(error(
                    &self.file.path,
                    line,
                    "tool_result",
                    "duplicate result identifier",
                ));
            }
            if !self.open_tool_ids.remove(&tool_use_id) {
                return Err(error(
                    &self.file.path,
                    line,
                    "tool_result",
                    "result does not identify an open tool use",
                ));
            }
            self.completed_tool_count += 1;
            retained.push(Value::Object(object.clone()));
        }
        if retained.is_empty() {
            return Ok(());
        }
        let timestamp = parse_timestamp(record, &self.file.path, line, "tool_result")?;
        self.history.push(raw_value_message(
            "user",
            json!({"role": "user", "content": retained}),
            &self.file.path,
            line,
        )?);
        self.last_result_timestamp = timestamp;
        if self.open_tool_ids.is_empty() {
            self.next_model_delay_after_previous_us = self
                .first_open_tool_timestamp
                .as_ref()
                .and_then(Option::as_ref)
                .zip(self.last_result_timestamp.as_ref())
                .and_then(|(first, last)| last.signed_duration_since(*first).num_microseconds())
                .filter(|micros| *micros > 0)
                .map(|micros| micros as f64);
            self.first_open_tool_timestamp = None;
            self.last_result_timestamp = None;
        }
        Ok(())
    }

    fn accept_assistant(
        &mut self,
        message: &Map<String, Value>,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        ensure_role(message, "assistant", &self.file.path, line)?;
        let source_id = match message.get("id") {
            Some(value) => validated_identifier_value(
                value,
                &self.file.path,
                line,
                "assistant",
                "invalid message identifier",
            )?,
            None => required_identifier(
                record,
                "uuid",
                &self.file.path,
                line,
                "assistant",
                "invalid message identifier",
            )?,
        };
        if self.model.is_none() {
            self.model = message
                .get("model")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
        }
        let blocks = message
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                error(
                    &self.file.path,
                    line,
                    "assistant",
                    "assistant content must be an array",
                )
            })?;
        let incoming = self.normalize_assistant_blocks(blocks, line)?;
        if let Some(finalized) = self.finalized_messages.get(&source_id) {
            if finalized != &incoming.full_content {
                return Err(error(
                    &self.file.path,
                    line,
                    "assistant",
                    "conflicting finalized assistant snapshot",
                ));
            }
            return Ok(());
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.source_id != source_id)
        {
            self.flush_pending()?;
            self.abandon_open_tools();
        }
        if self.pending.is_none() {
            self.reject_reused_task_tool_ids(blocks, line)?;
            let delay = self.next_model_delay_after_previous_us.take();
            self.calls.push(ImportedModelCall {
                source_id: source_id.clone(),
                request_messages: self.history.clone(),
                model: self.model.clone(),
                delay_after_previous_us: delay,
                tool_schema_available: false,
                output_tokens: None,
            });
            self.pending = Some(PendingClaudeMessage {
                source_id,
                line,
                content: Vec::new(),
                authored_content: Vec::new(),
                text_positions: Vec::new(),
                tool_ids: HashSet::new(),
            });
        }
        let timestamp = if incoming.fresh_tool_ids.is_empty() {
            None
        } else {
            parse_timestamp(record, &self.file.path, line, "tool_use")?
        };
        self.merge_pending(incoming, timestamp, line)
    }

    fn reject_reused_task_tool_ids(
        &self,
        blocks: &[Value],
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        for block in blocks {
            let Some(object) = block.as_object() else {
                continue;
            };
            if object.get("type").and_then(Value::as_str) != Some("tool_use")
                || object.get("name").and_then(Value::as_str) != Some("Task")
            {
                continue;
            }
            let id = required_identifier(
                object,
                "id",
                &self.file.path,
                line,
                "tool_use",
                "invalid tool-use identifier",
            )?;
            if self.seen_task_tool_use_ids.contains(&id) {
                return Err(error(
                    &self.file.path,
                    line,
                    "tool_use",
                    "duplicate Task tool-use identifier",
                ));
            }
        }
        Ok(())
    }

    fn normalize_assistant_blocks(
        &mut self,
        blocks: &[Value],
        line: usize,
    ) -> Result<NormalizedAssistantBlocks, ImportedAgentError> {
        let mut retained = Vec::new();
        let mut full_content = Vec::new();
        let mut fresh_tool_ids = HashSet::new();
        for block in blocks {
            let Some(object) = block.as_object() else {
                self.ignored_record_count += 1;
                continue;
            };
            match object.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if !object.get("text").is_some_and(Value::is_string) {
                        return Err(error(
                            &self.file.path,
                            line,
                            "assistant",
                            "invalid text block",
                        ));
                    }
                    let block = Value::Object(object.clone());
                    full_content.push(block.clone());
                    retained.push(block);
                }
                Some("tool_use") => {
                    let id = required_identifier(
                        object,
                        "id",
                        &self.file.path,
                        line,
                        "tool_use",
                        "invalid tool-use identifier",
                    )?;
                    if object
                        .get("name")
                        .and_then(Value::as_str)
                        .is_none_or(str::is_empty)
                        || !object.contains_key("input")
                    {
                        return Err(error(
                            &self.file.path,
                            line,
                            "tool_use",
                            "invalid tool-use block",
                        ));
                    }
                    let canonical = serde_json::to_vec(object).map_err(|_| {
                        error(
                            &self.file.path,
                            line,
                            "tool_use",
                            "cannot canonicalize tool block",
                        )
                    })?;
                    let block = Value::Object(object.clone());
                    full_content.push(block.clone());
                    if let Some(previous) = self.seen_tool_blocks.get(&id) {
                        if previous != &canonical {
                            return Err(error(
                                &self.file.path,
                                line,
                                "tool_use",
                                "conflicting tool-use identifier reuse",
                            ));
                        }
                    } else {
                        self.seen_tool_blocks.insert(id.clone(), canonical);
                        fresh_tool_ids.insert(id);
                        retained.push(block);
                    }
                }
                Some("thinking" | "redacted_thinking") => self.omitted_reasoning_count += 1,
                _ => self.ignored_record_count += 1,
            }
        }
        Ok(NormalizedAssistantBlocks {
            content: retained,
            full_content,
            fresh_tool_ids,
        })
    }

    fn merge_pending(
        &mut self,
        incoming: NormalizedAssistantBlocks,
        timestamp: Option<DateTime<FixedOffset>>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let mut fresh_tools = Vec::new();
        {
            let Some(pending) = self.pending.as_mut() else {
                return Ok(());
            };
            pending.authored_content =
                merge_authored_content(&pending.authored_content, &incoming.full_content)
                    .map_err(|detail| error(&self.file.path, line, "assistant", detail))?;
            let mut incoming_text = 0;
            for block in incoming.content {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(position) = pending.text_positions.get(incoming_text).copied() {
                            let existing = pending.content[position]
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            let merged = merge_text(existing, text).map_err(|detail| {
                                error(&self.file.path, line, "assistant", detail)
                            })?;
                            pending.content[position]["text"] = Value::String(merged);
                        } else {
                            pending.text_positions.push(pending.content.len());
                            pending.content.push(block);
                        }
                        incoming_text += 1;
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        let is_task = block.get("name").and_then(Value::as_str) == Some("Task");
                        if pending.tool_ids.insert(id.clone()) {
                            if incoming.fresh_tool_ids.contains(&id) {
                                fresh_tools.push((id.clone(), is_task));
                            }
                            pending.content.push(block);
                        }
                    }
                    _ => {}
                }
            }
        }
        for (id, is_task) in fresh_tools {
            self.open_tool_ids.insert(id.clone());
            if self.first_open_tool_timestamp.is_none() {
                self.first_open_tool_timestamp = Some(timestamp);
            }
            self.observed_tool_count += 1;
            self.all_tool_use_ids.insert(id.clone());
            if is_task {
                self.seen_task_tool_use_ids.insert(id.clone());
                self.task_tool_use_ids.push(id);
            }
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<(), ImportedAgentError> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        self.finalized_messages
            .insert(pending.source_id.clone(), pending.authored_content.clone());
        self.history.push(raw_value_message(
            "assistant",
            json!({"role": "assistant", "content": pending.content}),
            &self.file.path,
            pending.line,
        )?);
        Ok(())
    }

    fn abandon_open_tools(&mut self) {
        if !self.open_tool_ids.is_empty() {
            self.tool_results_complete = false;
            self.open_tool_ids.clear();
            self.first_open_tool_timestamp = None;
            self.last_result_timestamp = None;
        }
    }

    fn finish(mut self, source_digest: String) -> Result<ParsedClaudeSession, ImportedAgentError> {
        self.flush_pending()?;
        self.abandon_open_tools();
        let session_id = self
            .session_id
            .ok_or_else(|| error(&self.file.path, 0, "message", "missing session identifier"))?;
        if self.calls.is_empty() {
            return Err(error(
                &self.file.path,
                0,
                "assistant",
                "no inferred model calls",
            ));
        }
        let parent = if self.file.family == ImportedSessionFamily::Subagent {
            let tool_use_id = self.parent_tool_use_id.ok_or_else(|| {
                error(
                    &self.file.path,
                    0,
                    "message",
                    "missing parent tool-use identifier",
                )
            })?;
            Some(ImportedSubagentParent {
                session_id: session_id.clone(),
                tool_use_id,
            })
        } else {
            None
        };
        let session = ImportedAgentSession {
            session_id,
            source: ImportedAgentSource::ClaudeCode,
            source_path: self.file.path.clone(),
            source_digest,
            model: self.model,
            system_prompt: None,
            cwd_present: self.cwd_present,
            git_branch_present: self.git_branch_present,
            parent,
            calls: self.calls,
            observed_tool_count: self.observed_tool_count,
            completed_tool_count: self.completed_tool_count,
            ignored_record_count: self.ignored_record_count,
            omitted_reasoning_count: self.omitted_reasoning_count,
            tool_results_complete: self.tool_results_complete,
        };
        Ok(ParsedClaudeSession {
            session,
            task_tool_use_ids: self.task_tool_use_ids,
            all_tool_use_ids: self.all_tool_use_ids.into_iter().collect(),
        })
    }
}

fn merge_authored_content(
    existing: &[Value],
    incoming: &[Value],
) -> Result<Vec<Value>, &'static str> {
    let mut merged = existing.to_vec();
    let mut text_positions = Vec::new();
    let mut tool_positions = HashMap::new();
    for (position, block) in merged.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text_positions.push(position),
            Some("tool_use") => {
                if let Some(id) = block.get("id").and_then(Value::as_str) {
                    tool_positions.insert(id.to_owned(), position);
                }
            }
            _ => {}
        }
    }
    let mut incoming_text = 0;
    for block in incoming {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(position) = text_positions.get(incoming_text).copied() {
                    let existing = merged[position]
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    merged[position]["text"] = Value::String(merge_text(existing, text)?);
                } else {
                    text_positions.push(merged.len());
                    merged.push(block.clone());
                }
                incoming_text += 1;
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if let Some(position) = tool_positions.get(&id).copied() {
                    if merged[position] != *block {
                        return Err("conflicting tool-use identifier reuse");
                    }
                } else {
                    tool_positions.insert(id, merged.len());
                    merged.push(block.clone());
                }
            }
            _ => {}
        }
    }
    Ok(merged)
}

fn merge_text(existing: &str, incoming: &str) -> Result<String, &'static str> {
    if existing == incoming || incoming.starts_with(existing) {
        Ok(incoming.to_owned())
    } else if existing.starts_with(incoming) {
        Ok(existing.to_owned())
    } else {
        Err("conflicting repeated assistant text block")
    }
}

fn ensure_role(
    message: &Map<String, Value>,
    expected: &str,
    path: &std::path::Path,
    line: usize,
) -> Result<(), ImportedAgentError> {
    if message.get("role").and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(error(path, line, "message", "unexpected message role"))
    }
}

fn raw_value_message(
    role: &str,
    value: Value,
    path: &std::path::Path,
    line: usize,
) -> Result<RawJsonMessage, ImportedAgentError> {
    let wire = serde_json::to_vec(&value)
        .map_err(|_| error(path, line, "message", "cannot canonicalize message"))?;
    Ok(RawJsonMessage {
        role: role.to_owned(),
        wire: Bytes::from(wire),
    })
}

fn required_identifier(
    object: &Map<String, Value>,
    field: &str,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
    detail: &'static str,
) -> Result<String, ImportedAgentError> {
    let value = object.get(field).unwrap_or(&Value::Null);
    validated_identifier_value(value, path, line, label, detail)
}

fn validated_identifier_value(
    value: &Value,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
    detail: &'static str,
) -> Result<String, ImportedAgentError> {
    let value = value.as_str().unwrap_or_default();
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'#' | b'-')
        })
    {
        return Err(error(path, line, label, detail));
    }
    Ok(value.to_owned())
}

fn parse_timestamp(
    record: &Map<String, Value>,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
) -> Result<Option<DateTime<FixedOffset>>, ImportedAgentError> {
    match record.get("timestamp") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => DateTime::parse_from_rfc3339(value)
            .map(Some)
            .map_err(|_| error(path, line, label, "invalid timestamp")),
        Some(_) => Err(error(path, line, label, "invalid timestamp")),
    }
}

fn error(
    path: &std::path::Path,
    line: usize,
    label: &'static str,
    detail: &'static str,
) -> ImportedAgentError {
    ImportedAgentError::new(path, line, "claude_code", label, detail)
}
