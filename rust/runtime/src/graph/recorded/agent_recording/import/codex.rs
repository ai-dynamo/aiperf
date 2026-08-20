// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Causal normalization of Codex CLI JSONL sessions.

use std::collections::HashSet;
use std::io::{BufRead, BufReader};

use bytes::Bytes;
use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};

use super::{
    ImportedAgentError, ImportedAgentSession, ImportedAgentSource, ImportedAgentSourceFile,
    ImportedModelCall, RawJsonMessage,
};

/// Parse one exact-discovery Codex source file into the provider-neutral IR.
pub fn parse_codex_session(
    file: &ImportedAgentSourceFile,
) -> Result<ImportedAgentSession, ImportedAgentError> {
    let source = std::fs::File::open(&file.path)
        .map_err(|_| error(&file.path, 0, "unknown", "cannot read source file"))?;
    let mut reader = BufReader::new(source);
    let mut hasher = blake3::Hasher::new();
    let mut state = CodexState::new(file);
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

struct ToolCall {
    name: String,
    arguments: String,
    call_id: String,
}

struct ToolResult {
    call_id: String,
    output: String,
    timestamp: Option<DateTime<FixedOffset>>,
}

struct PendingBundle {
    first_line: usize,
    first_timestamp: Option<DateTime<FixedOffset>>,
    calls: Vec<ToolCall>,
    results: Vec<ToolResult>,
    had_result: bool,
}

struct CodexState<'a> {
    file: &'a ImportedAgentSourceFile,
    session_id: Option<String>,
    model: Option<String>,
    system_prompt: Option<RawJsonMessage>,
    cwd_present: bool,
    git_branch_present: bool,
    history: Vec<RawJsonMessage>,
    calls: Vec<ImportedModelCall>,
    pending: Option<PendingBundle>,
    seen_call_ids: HashSet<String>,
    seen_result_ids: HashSet<String>,
    observed_tool_count: u64,
    completed_tool_count: u64,
    ignored_record_count: u64,
    omitted_reasoning_count: u64,
    tool_results_complete: bool,
    next_model_delay_after_previous_us: Option<f64>,
}

impl<'a> CodexState<'a> {
    fn new(file: &'a ImportedAgentSourceFile) -> Self {
        Self {
            file,
            session_id: None,
            model: None,
            system_prompt: None,
            cwd_present: false,
            git_branch_present: false,
            history: Vec::new(),
            calls: Vec::new(),
            pending: None,
            seen_call_ids: HashSet::new(),
            seen_result_ids: HashSet::new(),
            observed_tool_count: 0,
            completed_tool_count: 0,
            ignored_record_count: 0,
            omitted_reasoning_count: 0,
            tool_results_complete: true,
            next_model_delay_after_previous_us: None,
        }
    }

    fn accept(
        &mut self,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let record_type = record.get("type").and_then(Value::as_str);
        let payload = record.get("payload").and_then(Value::as_object);
        let payload_type = payload
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str);
        match (record_type, payload_type) {
            (Some("session_meta"), _) => self.accept_session_meta(payload, line),
            (Some("turn_context"), _) => {
                self.accept_model_fallback(payload);
                Ok(())
            }
            (Some("response_item"), Some("message")) => self.accept_message(payload, line),
            (Some("response_item"), Some("function_call")) => {
                self.accept_function_call(payload, record, line)
            }
            (Some("response_item"), Some("function_call_output")) => {
                self.accept_function_result(payload, record, line)
            }
            (Some("response_item"), Some("reasoning")) => {
                self.omitted_reasoning_count += 1;
                Ok(())
            }
            _ => {
                self.ignored_record_count += 1;
                Ok(())
            }
        }
    }

    fn accept_session_meta(
        &mut self,
        payload: Option<&Map<String, Value>>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let payload = payload.ok_or_else(|| {
            error(
                &self.file.path,
                line,
                "session_meta",
                "payload must be an object",
            )
        })?;
        let id = required_identifier(
            payload,
            "id",
            &self.file.path,
            line,
            "session_meta",
            "invalid session identifier",
        )?;
        if let Some(previous) = &self.session_id {
            if previous != &id {
                return Err(error(
                    &self.file.path,
                    line,
                    "session_meta",
                    "inconsistent session identifier",
                ));
            }
            return Ok(());
        }
        self.session_id = Some(id);
        self.cwd_present = payload.get("cwd").is_some();
        self.git_branch_present = payload
            .get("git")
            .and_then(Value::as_object)
            .is_some_and(|git| git.get("branch").is_some());
        self.model = optional_string(payload, "model");
        let system = payload
            .get("base_instructions")
            .and_then(Value::as_object)
            .and_then(|base| optional_string(base, "text"));
        if let Some(system) = system {
            let message = raw_message("system", system, &self.file.path, line)?;
            self.system_prompt = Some(message.clone());
            self.history.push(message);
        }
        Ok(())
    }

    fn accept_model_fallback(&mut self, payload: Option<&Map<String, Value>>) {
        if self.model.is_none() {
            self.model = payload.and_then(|payload| optional_string(payload, "model"));
        }
    }

    fn accept_message(
        &mut self,
        payload: Option<&Map<String, Value>>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        self.flush_bundle()?;
        let payload = payload.ok_or_else(|| {
            error(
                &self.file.path,
                line,
                "response_item",
                "payload must be an object",
            )
        })?;
        let role = payload
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| error(&self.file.path, line, "message", "missing message role"))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant") {
            return Err(error(
                &self.file.path,
                line,
                "message",
                "unsupported message role",
            ));
        }
        let (text, ignored) = normalize_text(payload.get("content"));
        self.ignored_record_count += ignored;
        if text.is_none() {
            return Ok(());
        }
        let message = raw_message(role, text.unwrap_or_default(), &self.file.path, line)?;
        if role == "assistant" {
            let delay = self.next_model_delay_after_previous_us.take();
            self.calls
                .push(self.model_call(format!("codex-line-{line}"), delay));
        }
        self.history.push(message);
        Ok(())
    }

    fn accept_function_call(
        &mut self,
        payload: Option<&Map<String, Value>>,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        if self
            .pending
            .as_ref()
            .is_some_and(|bundle| bundle.had_result)
        {
            self.flush_bundle()?;
        }
        let payload = payload.ok_or_else(|| {
            error(
                &self.file.path,
                line,
                "response_item",
                "payload must be an object",
            )
        })?;
        let call_id = required_identifier(
            payload,
            "call_id",
            &self.file.path,
            line,
            "function_call",
            "invalid call identifier",
        )?;
        if !self.seen_call_ids.insert(call_id.clone()) {
            return Err(error(
                &self.file.path,
                line,
                "function_call",
                "duplicate call identifier",
            ));
        }
        let name =
            required_non_empty_string(payload, "name", &self.file.path, line, "function_call")?;
        let arguments =
            required_string_field(payload, "arguments", &self.file.path, line, "function_call")?;
        let timestamp = parse_timestamp(record, &self.file.path, line, "function_call")?;
        if self.pending.is_none() {
            let delay = self.next_model_delay_after_previous_us.take();
            self.calls
                .push(self.model_call(format!("codex-line-{line}"), delay));
        }
        let bundle = self.pending.get_or_insert_with(|| PendingBundle {
            first_line: line,
            first_timestamp: timestamp,
            calls: Vec::new(),
            results: Vec::new(),
            had_result: false,
        });
        bundle.calls.push(ToolCall {
            name,
            arguments,
            call_id,
        });
        self.observed_tool_count += 1;
        Ok(())
    }

    fn accept_function_result(
        &mut self,
        payload: Option<&Map<String, Value>>,
        record: &Map<String, Value>,
        line: usize,
    ) -> Result<(), ImportedAgentError> {
        let payload = payload.ok_or_else(|| {
            error(
                &self.file.path,
                line,
                "response_item",
                "payload must be an object",
            )
        })?;
        let call_id = required_identifier(
            payload,
            "call_id",
            &self.file.path,
            line,
            "function_call_output",
            "invalid call identifier",
        )?;
        let output = required_string_field(
            payload,
            "output",
            &self.file.path,
            line,
            "function_call_output",
        )?;
        let timestamp = parse_timestamp(record, &self.file.path, line, "function_call_output")?;
        if !self.seen_result_ids.insert(call_id.clone()) {
            return Err(error(
                &self.file.path,
                line,
                "function_call_output",
                "duplicate result identifier",
            ));
        }
        let Some(bundle) = self.pending.as_mut() else {
            return Err(error(
                &self.file.path,
                line,
                "function_call_output",
                "result does not identify an open call",
            ));
        };
        if !bundle.calls.iter().any(|call| call.call_id == call_id) {
            return Err(error(
                &self.file.path,
                line,
                "function_call_output",
                "result does not identify an open call",
            ));
        }
        bundle.had_result = true;
        self.completed_tool_count += 1;
        bundle.results.push(ToolResult {
            call_id,
            output,
            timestamp,
        });
        Ok(())
    }

    fn flush_bundle(&mut self) -> Result<(), ImportedAgentError> {
        let Some(bundle) = self.pending.take() else {
            return Ok(());
        };
        let complete = bundle.calls.len() == bundle.results.len();
        self.tool_results_complete &= complete;
        let delay_after_previous_us = bundle
            .first_timestamp
            .as_ref()
            .zip(
                bundle
                    .results
                    .last()
                    .and_then(|result| result.timestamp.as_ref()),
            )
            .and_then(|(first, last)| last.signed_duration_since(*first).num_microseconds())
            .filter(|micros| *micros > 0)
            .map(|micros| micros as f64);
        let tool_calls = bundle
            .calls
            .iter()
            .map(|call| {
                json!({
                    "type": "function",
                    "id": call.call_id,
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect::<Vec<_>>();
        self.history.push(raw_value_message(
            "assistant",
            json!({"role": "assistant", "tool_calls": tool_calls}),
            &self.file.path,
            bundle.first_line,
        )?);
        for result in bundle.results {
            self.history.push(raw_value_message(
                "tool",
                json!({"role": "tool", "tool_call_id": result.call_id, "content": result.output}),
                &self.file.path,
                bundle.first_line,
            )?);
        }
        self.next_model_delay_after_previous_us = delay_after_previous_us;
        Ok(())
    }

    fn model_call(
        &self,
        source_id: String,
        delay_after_previous_us: Option<f64>,
    ) -> ImportedModelCall {
        ImportedModelCall {
            source_id,
            request_messages: self.history.clone(),
            model: self.model.clone(),
            delay_after_previous_us,
            tool_schema_available: false,
            output_tokens: None,
        }
    }

    fn finish(mut self, source_digest: String) -> Result<ImportedAgentSession, ImportedAgentError> {
        self.flush_bundle()?;
        let session_id = self.session_id.ok_or_else(|| {
            error(
                &self.file.path,
                0,
                "session_meta",
                "missing session identifier",
            )
        })?;
        if self.calls.is_empty() {
            return Err(error(
                &self.file.path,
                0,
                "response_item",
                "no inferred model calls",
            ));
        }
        Ok(ImportedAgentSession {
            session_id,
            source: ImportedAgentSource::Codex,
            source_path: self.file.path.clone(),
            source_digest,
            model: self.model,
            system_prompt: self.system_prompt,
            cwd_present: self.cwd_present,
            git_branch_present: self.git_branch_present,
            parent: None,
            request_history: Default::default(),
            calls: self.calls,
            observed_tool_count: self.observed_tool_count,
            completed_tool_count: self.completed_tool_count,
            ignored_record_count: self.ignored_record_count,
            omitted_reasoning_count: self.omitted_reasoning_count,
            tool_results_complete: self.tool_results_complete,
        })
    }
}

fn normalize_text(content: Option<&Value>) -> (Option<String>, u64) {
    let Some(content) = content.and_then(Value::as_array) else {
        return (None, 1);
    };
    let mut text = Vec::new();
    let mut ignored = 0;
    for block in content {
        let Some(block) = block.as_object() else {
            ignored += 1;
            continue;
        };
        let supported = matches!(
            block.get("type").and_then(Value::as_str),
            Some("input_text" | "output_text" | "text")
        );
        if supported {
            if let Some(value) = block.get("text").and_then(Value::as_str) {
                text.push(value);
            } else {
                ignored += 1;
            }
        } else {
            ignored += 1;
        }
    }
    let text = if text.is_empty() {
        None
    } else {
        Some(text.join("\n"))
    };
    (text, ignored)
}

fn raw_message(
    role: &str,
    content: String,
    path: &std::path::Path,
    line: usize,
) -> Result<RawJsonMessage, ImportedAgentError> {
    raw_value_message(role, json!({"role": role, "content": content}), path, line)
}

fn raw_value_message(
    role: &str,
    value: Value,
    path: &std::path::Path,
    line: usize,
) -> Result<RawJsonMessage, ImportedAgentError> {
    if role.is_empty() {
        return Err(error(path, line, "message", "empty message role"));
    }
    let wire = serde_json::to_vec(&value)
        .map_err(|_| error(path, line, "message", "cannot canonicalize message"))?;
    Ok(RawJsonMessage {
        role: role.to_owned(),
        wire: Bytes::from(wire),
    })
}

fn required_identifier(
    payload: &Map<String, Value>,
    field: &str,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
    invalid_detail: &'static str,
) -> Result<String, ImportedAgentError> {
    let value = payload
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'#' | b'-')
        })
    {
        return Err(error(path, line, label, invalid_detail));
    }
    Ok(value)
}

fn required_non_empty_string(
    payload: &Map<String, Value>,
    field: &str,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
) -> Result<String, ImportedAgentError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| error(path, line, label, "missing required string"))
}

fn required_string_field(
    payload: &Map<String, Value>,
    field: &str,
    path: &std::path::Path,
    line: usize,
    label: &'static str,
) -> Result<String, ImportedAgentError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| error(path, line, label, "missing required string"))
}

fn optional_string(payload: &Map<String, Value>, field: &str) -> Option<String> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
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
    ImportedAgentError::new(path, line, "codex", label, detail)
}
