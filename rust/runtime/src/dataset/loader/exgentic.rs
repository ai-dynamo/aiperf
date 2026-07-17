// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exgentic v1/v2 agent LLM trace replay.
//!
//! Recorded request snapshots are normalized into OpenAI message/tool shapes,
//! sorted by span time, and replayed either as one delayed session or as
//! independent absolute fixed-schedule requests.

use std::collections::HashSet;

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{Map, Value};

use crate::dataset::compose::{ComposeConfig, Composer};
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::loader::public::load_public_rows;
use crate::dataset::loader::{DatasetLoader, DatasetProbe, DatasetSource, LoadConfig, RawRow};
use crate::dataset::model::{Conversation, ConversationContextMode, Turn};
use crate::dataset::segment::SegmentPool;
use crate::dataset::tokenizer::TextTokenizer;

/// Exgentic v1 loader.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExgenticDatasetLoader;
/// Exgentic v2 loader with benchmark-filter support.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExgenticV2DatasetLoader;
/// Shared Exgentic composer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExgenticComposer;

/// Pinned Exgentic v1 dataset revision.
pub const EXGENTIC_V1_REVISION: &str = "70036b93a04e61b0ea2706a68b962f4f26774587";
/// Pinned Exgentic v2 dataset revision.
pub const EXGENTIC_V2_REVISION: &str = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814";

#[async_trait]
impl DatasetLoader for ExgenticDatasetLoader {
    fn name(&self) -> &str {
        "exgentic"
    }
    fn can_load(&self, probe: &DatasetProbe) -> bool {
        is_exgentic_probe(probe)
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        load_exgentic(config, false).await
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::MessageArrayWithResponses)
    }
}

#[async_trait]
impl DatasetLoader for ExgenticV2DatasetLoader {
    fn name(&self) -> &str {
        "exgentic_v2"
    }
    fn can_load(&self, _probe: &DatasetProbe) -> bool {
        false
    }
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        load_exgentic(config, true).await
    }
    fn preferred_sampling_strategy(&self) -> &str {
        "sequential"
    }
    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::MessageArrayWithResponses)
    }
}

fn is_exgentic_probe(probe: &DatasetProbe) -> bool {
    probe.value.as_ref().is_some_and(|value| {
        value.get("models").is_some_and(Value::is_array)
            && value.get("spans").is_some_and(Value::is_array)
    })
}

async fn load_exgentic(config: &LoadConfig, v2: bool) -> Result<Vec<RawRow>> {
    let harness_filter = option_string(&config.options, "harness");
    let model_filter = option_string(&config.options, "source_model").map(canonical_source_model);
    let benchmark_filter = option_string(&config.options, "benchmark");
    if !v2 && benchmark_filter.is_some() {
        return Err(DatasetError::Validation(
            "Exgentic benchmark filter is supported only for v2 traces".into(),
        ));
    }
    validate_filter_pair(harness_filter.as_deref(), model_filter.as_deref(), v2)?;
    let fixed_schedule = config
        .options
        .get("fixed_schedule")
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| DatasetError::Validation("fixed_schedule must be boolean".into()))
        })
        .transpose()?
        .unwrap_or(false);
    let revision = if v2 {
        EXGENTIC_V2_REVISION
    } else {
        EXGENTIC_V1_REVISION
    };
    let mut pinned = config.clone();
    if let DatasetSource::HuggingFace {
        revision: configured,
        ..
    } = &mut pinned.source
    {
        if configured
            .as_deref()
            .is_some_and(|configured| configured != revision)
        {
            return Err(DatasetError::Validation(format!(
                "Exgentic {} is pinned to revision {revision}, not {:?}",
                if v2 { "v2" } else { "v1" },
                configured.as_deref()
            )));
        }
        *configured = Some(revision.into());
    }
    let mut rows = load_public_rows(&pinned).await?;
    rows.retain_mut(|row| {
        let Some(object) = row.value.as_object_mut() else {
            return true;
        };
        object.insert(
            "__aiperf_fixed_schedule".into(),
            Value::Bool(fixed_schedule),
        );
        let harness = object.get("harness").and_then(Value::as_str).unwrap_or("");
        let models = object
            .get("models")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(Value::as_str)
                    .map(canonical_source_model)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let benchmark = object.get("benchmark").and_then(Value::as_str);
        harness_filter
            .as_ref()
            .is_none_or(|filter| harness.eq_ignore_ascii_case(filter))
            && model_filter.as_ref().is_none_or(|filter| {
                models
                    .iter()
                    .any(|model| model.eq_ignore_ascii_case(filter))
            })
            && benchmark_filter
                .as_ref()
                .is_none_or(|filter| benchmark == Some(filter.as_str()))
    });
    Ok(rows)
}

impl Composer for ExgenticComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        _tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let max_conversations = config
            .format_options
            .get("max_conversations")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                DatasetError::Validation(
                    "Exgentic requires positive max_conversations to bound remote streaming".into(),
                )
            })?;
        let mut seen = HashSet::new();
        let mut finalizer = config.finalizer()?;
        let mut conversations = Vec::new();
        for (row_index, row) in rows.into_iter().enumerate() {
            let object = row.value.as_object().ok_or_else(|| {
                DatasetError::Validation(format!("{}: Exgentic row must be object", row.origin))
            })?;
            let harness = nonempty_string(object, "harness", row_index + 1)?;
            let session_id = nonempty_string(object, "session_id", row_index + 1)?;
            let models = object
                .get("models")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "Exgentic row {} models must be an array",
                        row_index + 1
                    ))
                })?;
            if models
                .iter()
                .any(|model| model.as_str().is_none_or(str::is_empty))
            {
                return Err(DatasetError::Validation(format!(
                    "Exgentic row {} models must contain non-empty strings",
                    row_index + 1
                )));
            }
            let spans = object
                .get("spans")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "Exgentic row {} spans must be an array",
                        row_index + 1
                    ))
                })?;
            let fixed = object
                .get("__aiperf_fixed_schedule")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let mut parsed = Vec::new();
            for (span_index, span) in spans.iter().enumerate() {
                if let Some(span) = parse_span(session_id, span_index, span)? {
                    parsed.push(span);
                }
            }
            parsed.sort_by(|left, right| {
                left.start_ms
                    .total_cmp(&right.start_ms)
                    .then(left.span_index.cmp(&right.span_index))
            });
            if parsed.is_empty() {
                continue;
            }
            if !seen.insert(session_id.to_string()) {
                return Err(DatasetError::DuplicateSession(session_id.to_string()));
            }
            let _ = harness;
            if fixed {
                let session_start = parsed[0].start_ms;
                for span in parsed {
                    let id = format!("{session_id}:{}", span.span_index);
                    let mut turn = compose_span_turn(
                        span,
                        session_id,
                        Some(session_start),
                        None,
                        &mut finalizer,
                        segments,
                    )?;
                    turn.delay_ms = None;
                    let mut conversation = Conversation::new(id);
                    conversation.context_mode =
                        Some(ConversationContextMode::MessageArrayWithResponses);
                    conversation.turns.push(turn);
                    conversations.push(conversation);
                }
            } else {
                let mut conversation = Conversation::new(session_id);
                conversation.context_mode =
                    Some(ConversationContextMode::MessageArrayWithResponses);
                let mut previous_end = None;
                for span in parsed {
                    let delay = previous_end.map(|end: f64| (span.start_ms - end).max(0.0));
                    previous_end = Some(span.end_ms);
                    conversation.turns.push(compose_span_turn(
                        span,
                        session_id,
                        None,
                        delay,
                        &mut finalizer,
                        segments,
                    )?);
                }
                conversations.push(conversation);
            }
            if seen.len() >= max_conversations {
                break;
            }
        }
        if conversations.is_empty() {
            return Err(DatasetError::Validation(
                "no replayable Exgentic spans matched the selected filters".into(),
            ));
        }
        Ok(conversations)
    }
}

#[derive(Debug)]
struct ParsedSpan {
    start_ms: f64,
    end_ms: f64,
    span_index: usize,
    input_tokens: u64,
    max_tokens: u32,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    extra_body: Option<Map<String, Value>>,
}

fn parse_span(session_id: &str, span_index: usize, value: &Value) -> Result<Option<ParsedSpan>> {
    let span = value.as_object().ok_or_else(|| {
        DatasetError::Validation(format!(
            "Exgentic session {session_id:?} span {span_index} must be an object"
        ))
    })?;
    let attributes = span
        .get("attributes")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let replayable = span.get("type").and_then(Value::as_str) == Some("llm_call")
        || (span.get("type").is_none()
            && attributes
                .get("gen_ai.operation.name")
                .and_then(Value::as_str)
                == Some("chat"));
    if !replayable
        || span
            .get("status")
            .and_then(Value::as_object)
            .and_then(|status| status.get("code"))
            .and_then(Value::as_i64)
            == Some(2)
    {
        return Ok(None);
    }
    let input_tokens = positive_integer(
        &attributes,
        "gen_ai.usage.input_tokens",
        session_id,
        span_index,
    )?;
    let output_tokens = positive_integer(
        &attributes,
        "gen_ai.usage.output_tokens",
        session_id,
        span_index,
    )?;
    if input_tokens == 0 || output_tokens == 0 {
        return Ok(None);
    }
    let start_ms = parse_timestamp(span.get("start_time").and_then(Value::as_str).ok_or_else(
        || {
            DatasetError::Validation(format!(
                "Exgentic session {session_id:?} span {span_index} has no start_time"
            ))
        },
    )?)?;
    let end_ms = parse_timestamp(span.get("end_time").and_then(Value::as_str).ok_or_else(
        || {
            DatasetError::Validation(format!(
                "Exgentic session {session_id:?} span {span_index} has no end_time"
            ))
        },
    )?)?;
    if end_ms < start_ms {
        return Err(DatasetError::Validation(format!(
            "Exgentic session {session_id:?} span {span_index} ends before it starts"
        )));
    }
    let requested = attributes
        .get("gen_ai.request.max_tokens")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "Exgentic session {session_id:?} span {span_index} max_tokens must be positive integer"
                    ))
                })
        })
        .transpose()?;
    let observed_output_tokens = u32::try_from(output_tokens).map_err(|_| {
        DatasetError::Validation(format!(
            "Exgentic session {session_id:?} span {span_index} output token count exceeds u32"
        ))
    })?;
    let mut messages = normalize_system(attributes.get("gen_ai.system_instructions"))?;
    messages.extend(normalize_messages(
        attributes.get("gen_ai.input.messages").ok_or_else(|| {
            DatasetError::Validation(format!(
                "Exgentic session {session_id:?} span {span_index} has no input messages"
            ))
        })?,
    )?);
    let tools = normalize_tools(attributes.get("gen_ai.tool.definitions"))?;
    let extra_body = request_extra_body(&attributes)?;
    Ok(Some(ParsedSpan {
        start_ms,
        end_ms,
        span_index,
        input_tokens,
        max_tokens: requested.unwrap_or(observed_output_tokens),
        messages,
        tools,
        extra_body,
    }))
}

fn compose_span_turn(
    span: ParsedSpan,
    session_id: &str,
    fixed_start: Option<f64>,
    delay: Option<f64>,
    finalizer: &mut crate::dataset::compose::TurnFinalizer<'_>,
    segments: &mut SegmentPool,
) -> Result<Turn> {
    let messages = segments.intern_raw(None, Bytes::from(serde_json::to_vec(&span.messages)?))?;
    let tools = span
        .tools
        .map(|tools| segments.intern_raw(Some(messages), Bytes::from(serde_json::to_vec(&tools)?)))
        .transpose()?;
    let extra_body = span
        .extra_body
        .map(|extra| segments.intern_raw(None, Bytes::from(serde_json::to_vec(&extra)?)))
        .transpose()?;
    let headers = segments.intern_raw(
        None,
        Bytes::from(serde_json::to_vec(&serde_json::json!({
            "x-dynamo-session-id": session_id
        }))?),
    )?;
    let mut turn = Turn {
        max_tokens: Some(span.max_tokens),
        input_tokens: span.input_tokens,
        timestamp_ms: fixed_start.map(|start| span.start_ms - start),
        delay_ms: delay,
        raw_messages: Some(messages),
        tools,
        extra_body,
        extra_headers: Some(headers),
        ..Turn::default()
    };
    finalizer.finalize_turn(&mut turn)?;
    Ok(turn)
}

fn normalize_messages(value: &Value) -> Result<Vec<Value>> {
    let messages = parse_json_array(value, "gen_ai.input.messages")?;
    let mut normalized = Vec::new();
    for message in messages {
        let message = message.as_object().ok_or_else(|| {
            DatasetError::Validation("Exgentic input messages must be objects".into())
        })?;
        let role = message.get("role").and_then(Value::as_str).ok_or_else(|| {
            DatasetError::Validation("Exgentic message requires string role".into())
        })?;
        let parts = message
            .get("parts")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                DatasetError::Validation("Exgentic message requires parts array".into())
            })?;
        normalize_parts(role, parts, &mut normalized)?;
    }
    if normalized.is_empty() {
        return Err(DatasetError::Validation(
            "Exgentic input messages cannot be empty".into(),
        ));
    }
    Ok(normalized)
}

fn normalize_parts(role: &str, parts: &[Value], normalized: &mut Vec<Value>) -> Result<()> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let flush = |normalized: &mut Vec<Value>,
                 content: &mut String,
                 reasoning: &mut String,
                 tool_calls: &mut Vec<Value>| {
        let mut message = Map::new();
        message.insert(
            "role".into(),
            Value::String(if role == "developer" { "system" } else { role }.into()),
        );
        message.insert("content".into(), Value::String(std::mem::take(content)));
        if !reasoning.is_empty() {
            message.insert(
                "reasoning_content".into(),
                Value::String(std::mem::take(reasoning)),
            );
        }
        if !tool_calls.is_empty() {
            message.insert(
                "tool_calls".into(),
                Value::Array(std::mem::take(tool_calls)),
            );
        }
        normalized.push(Value::Object(message));
    };
    for part in parts {
        let part = part.as_object().ok_or_else(|| {
            DatasetError::Validation("Exgentic message parts must be objects".into())
        })?;
        match part.get("type").and_then(Value::as_str) {
            Some("text") => content.push_str(
                part.get("content")
                    .map(json_string)
                    .transpose()?
                    .as_deref()
                    .unwrap_or(""),
            ),
            Some("thinking") => reasoning.push_str(
                part.get("thinking")
                    .map(json_string)
                    .transpose()?
                    .as_deref()
                    .unwrap_or(""),
            ),
            Some("tool_call") => tool_calls.push(serde_json::json!({
                "id": part.get("id").cloned().unwrap_or(Value::Null),
                "type": "function",
                "function": {
                    "name": part.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": json_string(part.get("arguments").unwrap_or(&Value::Null))?,
                }
            })),
            Some("tool_call_response") => {
                if !content.is_empty() || !reasoning.is_empty() || !tool_calls.is_empty() {
                    flush(normalized, &mut content, &mut reasoning, &mut tool_calls);
                }
                normalized.push(serde_json::json!({
                    "role":"tool",
                    "tool_call_id":part.get("id").cloned().unwrap_or(Value::Null),
                    "content":json_string(part.get("result").unwrap_or(&Value::Null))?,
                }));
            }
            other => {
                return Err(DatasetError::Validation(format!(
                    "unsupported Exgentic {role:?} message part {other:?}"
                )));
            }
        }
    }
    if !content.is_empty()
        || !reasoning.is_empty()
        || !tool_calls.is_empty()
        || normalized.is_empty()
    {
        flush(normalized, &mut content, &mut reasoning, &mut tool_calls);
    }
    Ok(())
}

fn normalize_system(value: Option<&Value>) -> Result<Vec<Value>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let text = value.as_str().ok_or_else(|| {
        DatasetError::Validation("gen_ai.system_instructions must be a string".into())
    })?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    if let Ok(parts) = serde_json::from_str::<Value>(text)
        && let Some(parts) = parts.as_array()
    {
        let mut normalized = Vec::new();
        normalize_parts("system", parts, &mut normalized)?;
        return Ok(normalized);
    }
    Ok(vec![serde_json::json!({"role":"system","content":text})])
}

fn normalize_tools(value: Option<&Value>) -> Result<Option<Vec<Value>>> {
    let Some(value) = value else { return Ok(None) };
    let tools = parse_json_array(value, "gen_ai.tool.definitions")?;
    let mut normalized = Vec::new();
    for tool in tools {
        let tool = tool
            .as_object()
            .ok_or_else(|| DatasetError::Validation("Exgentic tools must be objects".into()))?;
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(DatasetError::Validation(format!(
                "unsupported Exgentic tool type {:?}",
                tool.get("type")
            )));
        }
        normalized.push(serde_json::json!({
            "type":"function",
            "function":{
                "name":tool.get("name").cloned().unwrap_or(Value::Null),
                "description":tool.get("description").cloned().unwrap_or(Value::Null),
                "parameters":tool.get("parameters").cloned().unwrap_or(Value::Null),
            }
        }));
    }
    Ok((!normalized.is_empty()).then_some(normalized))
}

fn request_extra_body(attributes: &Map<String, Value>) -> Result<Option<Map<String, Value>>> {
    let mut extra = Map::new();
    if let Some(temperature) = attributes.get("gen_ai.request.temperature") {
        let temperature = temperature
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                DatasetError::Validation("gen_ai.request.temperature must be finite".into())
            })?;
        extra.insert("temperature".into(), Value::from(temperature));
    }
    if let Some(stop) = attributes.get("gen_ai.request.stop_sequences") {
        let stop = stop
            .as_array()
            .filter(|values| values.iter().all(Value::is_string))
            .ok_or_else(|| {
                DatasetError::Validation("gen_ai.request.stop_sequences must be strings".into())
            })?;
        if !stop.is_empty() {
            extra.insert("stop".into(), Value::Array(stop.clone()));
        }
    }
    Ok((!extra.is_empty()).then_some(extra))
}

fn parse_json_array(value: &Value, field: &str) -> Result<Vec<Value>> {
    let value = match value {
        Value::String(value) => serde_json::from_str(value).map_err(|error| {
            DatasetError::Validation(format!("{field} is not valid JSON: {error}"))
        })?,
        value => value.clone(),
    };
    value
        .as_array()
        .cloned()
        .ok_or_else(|| DatasetError::Validation(format!("{field} must be a JSON array")))
}

fn json_string(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        value => serde_json::to_string(value).map_err(DatasetError::from),
    }
}

fn parse_timestamp(value: &str) -> Result<f64> {
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(value) {
        return Ok(parsed.timestamp_micros() as f64 / 1_000.0);
    }
    let parsed =
        chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").map_err(|error| {
            DatasetError::Validation(format!("invalid Exgentic timestamp: {error}"))
        })?;
    Ok(parsed.and_utc().timestamp_micros() as f64 / 1_000.0)
}

fn positive_integer(
    attributes: &Map<String, Value>,
    key: &str,
    session: &str,
    span: usize,
) -> Result<u64> {
    attributes.get(key).and_then(Value::as_u64).ok_or_else(|| {
        DatasetError::Validation(format!(
            "Exgentic session {session:?} span {span} has non-integer {key}"
        ))
    })
}

fn nonempty_string<'a>(object: &'a Map<String, Value>, key: &str, row: usize) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| DatasetError::Validation(format!("Exgentic row {row} has no {key}")))
}

fn canonical_source_model(value: impl AsRef<str>) -> String {
    let value = value.as_ref();
    let lowered = value.to_ascii_lowercase();
    for prefix in ["openai/azure/", "azure/", "aws/", "gcp/"] {
        if lowered.starts_with(prefix) {
            return value[prefix.len()..].to_string();
        }
    }
    value.to_string()
}

fn option_string(options: &Map<String, Value>, key: &str) -> Option<String> {
    options.get(key).and_then(Value::as_str).map(str::to_string)
}

fn validate_filter_pair(harness: Option<&str>, model: Option<&str>, v2: bool) -> Result<()> {
    let Some(harness) = harness else {
        return Ok(());
    };
    let Some(model) = model else { return Ok(()) };
    let unsupported = if v2 {
        matches!(
            (harness, model),
            (
                "claude_code" | "openai_solo" | "smolagents_code" | "tool_calling",
                "gpt-4.1"
            ) | (
                "tool_calling_with_shortlisting",
                "claude-opus-4-5" | "gpt-4.1" | "gpt-5.2-2025-12-11"
            )
        )
    } else {
        matches!(
            (harness, model),
            ("openai_solo" | "tool_calling", "gpt-5.2-2025-12-11")
                | (
                    "smolagents_code",
                    "claude-opus-4-5" | "gemini-3-pro-preview" | "gpt-5.2-2025-12-11"
                )
                | (
                    "tool_calling_with_shortlisting",
                    "claude-opus-4-5" | "gpt-4.1" | "gpt-5.2-2025-12-11"
                )
        )
    };
    if unsupported {
        return Err(DatasetError::Validation(format!(
            "unsupported Exgentic filter combination harness={harness:?}, source_model={model:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::loader::{DatasetFormatRegistration, DatasetSource, LoaderRegistry};
    use crate::dataset::tokenizer::TiktokenTokenizer;

    #[tokio::test]
    async fn normalizes_tools_messages_headers_and_inter_turn_delay() {
        let source = DatasetSource::Inline(json!([{
            "harness":"claude_code","session_id":"s","models":["azure/gpt-4.1"],
            "spans":[
                {"type":"llm_call","start_time":"2026-01-01T00:00:00Z","end_time":"2026-01-01T00:00:01Z","attributes":{
                    "gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":3,
                    "gen_ai.input.messages":"[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"q\"}]}]",
                    "gen_ai.tool.definitions":"[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{}}]"
                }},
                {"type":"llm_call","start_time":"2026-01-01T00:00:02Z","end_time":"2026-01-01T00:00:03Z","attributes":{
                    "gen_ai.usage.input_tokens":12,"gen_ai.usage.output_tokens":4,
                    "gen_ai.input.messages":[{"role":"user","parts":[{"type":"tool_call_response","id":"1","result":{"ok":true}}]}]
                }}
            ]
        }]));
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(ExgenticDatasetLoader),
                Arc::new(ExgenticComposer),
            ))
            .unwrap();
        let mut compose = ComposeConfig::new("target", RngRoot::new(Some(1)));
        compose
            .format_options
            .insert("max_conversations".into(), Value::from(1));
        let dataset = registry
            .build_dataset(
                Some("exgentic"),
                &LoadConfig::new(source),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let conversation = &dataset.conversations()[0];
        assert_eq!(conversation.turns.len(), 2);
        assert_eq!(conversation.turns[1].delay_ms, Some(1000.0));
        assert!(conversation.turns[0].tools.is_some());
        assert!(conversation.turns[0].extra_headers.is_some());
        assert_eq!(conversation.turns[0].input_tokens, 10);
    }

    #[test]
    fn rejects_output_lengths_that_cannot_be_represented_on_the_wire() {
        let output_tokens = u64::from(u32::MAX) + 1;
        let span = json!({
            "type":"llm_call",
            "start_time":"2026-01-01T00:00:00Z",
            "end_time":"2026-01-01T00:00:01Z",
            "attributes":{
                "gen_ai.usage.input_tokens":1,
                "gen_ai.usage.output_tokens":output_tokens,
                "gen_ai.input.messages":[{
                    "role":"user",
                    "parts":[{"type":"text","content":"q"}]
                }]
            }
        });
        let error = parse_span("session", 0, &span).unwrap_err();
        assert!(error.to_string().contains("output token count exceeds u32"));
    }
}
