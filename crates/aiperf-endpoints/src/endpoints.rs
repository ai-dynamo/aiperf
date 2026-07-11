// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier-1 endpoint implementations.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::extraction::{PartTypes, extract_inputs};
use crate::metadata::{EndpointMetadata, EndpointType, metadata_for};
use crate::models::{
    CreditPhase, EndpointError, EndpointResult, ExtractedPayload, Media, ParsedResponse,
    RequestInfo, RequestRecord, ResponseData, ServerResponse, Turn,
};

/// Warmup prefix used by the completions endpoint.
pub const WARMUP_SYSTEM_MESSAGE_PREFIX: &str =
    "You are in warmup mode. This request is used to warm up the benchmark target.";

/// Endpoint adapter contract.
pub trait Endpoint {
    /// Return static capability metadata.
    fn metadata(&self) -> &'static EndpointMetadata;
    /// Build a decoded JSON request body.
    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value>;
    /// Parse a decoded server response.
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    /// Extract tokenizable input and media counts from a built body.
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_inputs(body, &self.part_types())
    }
    /// Build an assistant turn for context replay.
    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        build_plain_assistant_turn(self, record)
    }
    /// Return endpoint content-part names.
    fn part_types(&self) -> PartTypes {
        PartTypes::chat()
    }
    /// Parse every response in a record.
    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        let mut out = Vec::new();
        for response in &record.responses {
            if let Some(parsed) = self.parse_response(response)? {
                out.push(parsed);
            }
        }
        Ok(out)
    }
}

/// OpenAI Chat Completions endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatEndpoint;
/// OpenAI Responses endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResponsesEndpoint;
/// OpenAI Completions endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompletionsEndpoint;
/// OpenAI Embeddings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddingsEndpoint;
/// Chat-shaped embeddings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatEmbeddingsEndpoint;

impl Endpoint for ChatEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Chat)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        let turns = require_turns(request_info, "Chat endpoint requires at least one turn")?;
        let mut messages =
            format_chat_messages(request_info, build_messages(turns, PartShape::Chat)?);
        let last = turns.last().expect("non-empty turns");
        let endpoint = &request_info.model_endpoint.endpoint;
        let mut payload = Map::new();
        payload.insert(
            "messages".into(),
            Value::Array(std::mem::take(&mut messages)),
        );
        payload.insert(
            "model".into(),
            Value::String(
                last.model
                    .clone()
                    .unwrap_or_else(|| request_info.model_endpoint.primary_model_name.clone()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(tools) = latest_turn_attr(turns, |turn| turn.raw_tools.as_ref()) {
            payload.insert("tools".into(), Value::Array(tools.clone()));
        }
        if let Some(max_tokens) = last.max_tokens {
            payload.insert(
                if endpoint.use_legacy_max_tokens {
                    "max_tokens"
                } else {
                    "max_completion_tokens"
                }
                .into(),
                json!(max_tokens),
            );
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, last.extra_body.as_ref());
        if endpoint.streaming && endpoint.use_server_token_count {
            ensure_include_usage(&mut payload);
        }
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = extract_chat_response_data(obj);
        let usage = non_empty_field(obj, "usage");
        Ok(
            (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                perf_ns: response.perf_ns,
                data,
                usage,
            }),
        )
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        let mut content_parts = Vec::new();
        let mut tool_calls_by_index: BTreeMap<i64, Map<String, Value>> = BTreeMap::new();
        for response in &record.responses {
            let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            let Some(choice) = first_choice(obj) else {
                continue;
            };
            absorb_chat_choice(
                obj.get("object").and_then(Value::as_str),
                choice,
                &mut content_parts,
                &mut tool_calls_by_index,
            );
        }
        if tool_calls_by_index.is_empty() {
            return build_plain_assistant_turn(self, record);
        }
        let text = content_parts.concat();
        let mut message = Map::new();
        message.insert("role".into(), Value::String("assistant".into()));
        message.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                Value::String(text)
            },
        );
        message.insert(
            "tool_calls".into(),
            Value::Array(
                tool_calls_by_index
                    .into_values()
                    .map(Value::Object)
                    .collect(),
            ),
        );
        Ok(Some(Turn {
            role: Some("assistant".into()),
            raw_messages: Some(vec![Value::Object(message)]),
            ..Turn::default()
        }))
    }
}

impl Endpoint for ResponsesEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Responses)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        let turns = require_turns(
            request_info,
            "Responses endpoint requires at least one turn",
        )?;
        let endpoint = &request_info.model_endpoint.endpoint;
        let last = turns.last().expect("non-empty turns");
        let mut input = Vec::new();
        if let Some(context) = request_info
            .user_context_message
            .as_ref()
            .filter(|s| !s.is_empty())
        {
            input.push(json!({"type": "message", "role": "user", "content": context}));
        }
        input.extend(build_messages_responses(turns)?);
        let mut payload = Map::new();
        payload.insert("input".into(), Value::Array(input));
        payload.insert(
            "model".into(),
            Value::String(
                last.model
                    .clone()
                    .unwrap_or_else(|| request_info.model_endpoint.primary_model_name.clone()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(system) = request_info
            .system_message
            .as_ref()
            .filter(|s| !s.is_empty())
        {
            payload.insert("instructions".into(), Value::String(system.clone()));
        }
        if let Some(max_tokens) = last.max_tokens {
            payload.insert("max_output_tokens".into(), json!(max_tokens));
        }
        if let Some(tools) = latest_turn_attr(turns, |turn| turn.raw_tools.as_ref()) {
            payload.insert("tools".into(), Value::Array(tools.clone()));
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, last.extra_body.as_ref());
        if endpoint.streaming && endpoint.use_server_token_count {
            ensure_include_usage(&mut payload);
        }
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        if obj.contains_key("type") {
            return Ok(parse_responses_streaming_event(obj, response.perf_ns));
        }
        if obj.get("object").and_then(Value::as_str) == Some("response") {
            let data = extract_responses_full_content(obj);
            let usage = non_empty_field(obj, "usage");
            return Ok(
                (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                    perf_ns: response.perf_ns,
                    data,
                    usage,
                }),
            );
        }
        Ok(None)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let mut result = extract_inputs(body, &PartTypes::responses());
        let Some(obj) = body.as_object() else {
            return result;
        };
        match obj.get("instructions") {
            Some(Value::String(text)) if !text.is_empty() => {
                result.texts.insert(0, text.clone());
                if let Some(messages) = &mut result.messages {
                    messages.insert(0, json!({"role":"system","content":text}));
                }
            }
            Some(Value::Array(parts)) => {
                let mut collected = Vec::new();
                for part in parts {
                    if let Some(text) = part
                        .as_object()
                        .and_then(|obj| obj.get("text"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        collected.push(text.to_string());
                    } else if let Some(text) = part.as_str().filter(|s| !s.is_empty()) {
                        collected.push(text.to_string());
                    }
                }
                for text in collected.iter().rev() {
                    result.texts.insert(0, text.clone());
                }
                if let Some(messages) = &mut result.messages
                    && !collected.is_empty()
                {
                    messages.insert(0, json!({"role":"system","content":collected.concat()}));
                }
            }
            _ => {}
        }
        result
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        let mut items_by_key = Map::new();
        let mut done_items = Vec::new();
        for response in &record.responses {
            let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            if is_responses_failure_event(obj) {
                return build_plain_assistant_turn(self, record);
            }
            collect_response_items(obj, &mut items_by_key, &mut done_items);
        }
        for item in done_items {
            merge_response_item(&mut items_by_key, item);
        }
        if items_by_key.is_empty() {
            return build_plain_assistant_turn(self, record);
        }
        Ok(Some(Turn {
            role: Some("assistant".into()),
            raw_messages: Some(items_by_key.into_values().collect()),
            ..Turn::default()
        }))
    }

    fn part_types(&self) -> PartTypes {
        PartTypes::responses()
    }
}

impl Endpoint for CompletionsEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Completions)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        if request_info.turns.len() != 1 {
            return Err(EndpointError::InvalidRequest(
                "Completions endpoint only supports one turn".into(),
            ));
        }
        let turn = &request_info.turns[0];
        let endpoint = &request_info.model_endpoint.endpoint;
        let mut prompts = turn_texts(turn);
        if request_info.credit_phase == CreditPhase::Warmup {
            prompts = prompts
                .into_iter()
                .map(|prompt| format!("{WARMUP_SYSTEM_MESSAGE_PREFIX}\n{prompt}"))
                .collect();
        }
        let mut payload = Map::new();
        payload.insert(
            "prompt".into(),
            Value::Array(prompts.into_iter().map(Value::String).collect()),
        );
        payload.insert(
            "model".into(),
            Value::String(
                turn.model
                    .clone()
                    .unwrap_or_else(|| request_info.model_endpoint.primary_model_name.clone()),
            ),
        );
        payload.insert("stream".into(), Value::Bool(endpoint.streaming));
        if let Some(max_tokens) = turn.max_tokens {
            payload.insert("max_tokens".into(), json!(max_tokens));
        }
        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, turn.extra_body.as_ref());
        if endpoint.streaming && endpoint.use_server_token_count {
            ensure_include_usage(&mut payload);
        }
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = match obj.get("object").and_then(Value::as_str) {
            Some("completion" | "text_completion") => first_choice(obj)
                .and_then(|choice| choice.get("text"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(|text| ResponseData::Text {
                    text: text.to_string(),
                }),
            _ => None,
        };
        let usage = non_empty_field(obj, "usage");
        Ok(
            (data.is_some() || usage.is_some()).then_some(ParsedResponse {
                perf_ns: response.perf_ns,
                data,
                usage,
            }),
        )
    }
}

impl Endpoint for EmbeddingsEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Embeddings)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        if request_info.turns.len() != 1 {
            return Err(EndpointError::InvalidRequest(
                "Embeddings endpoint only supports one turn".into(),
            ));
        }
        let turn = &request_info.turns[0];
        let mut payload = Map::new();
        payload.insert(
            "model".into(),
            Value::String(
                turn.model
                    .clone()
                    .unwrap_or_else(|| request_info.model_endpoint.primary_model_name.clone()),
            ),
        );
        payload.insert(
            "input".into(),
            Value::Array(turn_texts(turn).into_iter().map(Value::String).collect()),
        );
        merge_extra(
            &mut payload,
            request_info.model_endpoint.endpoint.extra.as_ref(),
        );
        merge_extra(&mut payload, turn.extra_body.as_ref());
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_embeddings_response(response, true)
    }
}

impl Endpoint for ChatEmbeddingsEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::ChatEmbeddings)
    }
    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        ChatEndpoint.format_payload(request_info)
    }
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_embeddings_response(response, false)
    }
}

fn require_turns<'a>(request_info: &'a RequestInfo, message: &str) -> EndpointResult<&'a [Turn]> {
    if request_info.turns.is_empty() {
        Err(EndpointError::InvalidRequest(message.into()))
    } else {
        Ok(&request_info.turns)
    }
}

#[derive(Clone, Copy)]
enum PartShape {
    Chat,
    Responses,
}

fn build_messages(turns: &[Turn], shape: PartShape) -> EndpointResult<Vec<Value>> {
    let mut messages = Vec::new();
    for turn in turns {
        if let Some(raw_messages) = &turn.raw_messages
            && !raw_messages.is_empty()
        {
            messages.extend(raw_messages.clone());
            continue;
        }
        messages.push(render_turn_message(turn, shape)?);
    }
    Ok(messages)
}

fn build_messages_responses(turns: &[Turn]) -> EndpointResult<Vec<Value>> {
    let mut messages = Vec::new();
    for turn in turns {
        if let Some(raw_messages) = &turn.raw_messages
            && !raw_messages.is_empty()
        {
            for item in raw_messages {
                if item
                    .as_object()
                    .and_then(|obj| obj.get("type"))
                    .and_then(Value::as_str)
                    .is_some_and(is_replay_unsafe_output_item)
                {
                    continue;
                }
                messages.push(item.clone());
            }
            continue;
        }
        let mut message = render_turn_message(turn, PartShape::Responses)?;
        if let Value::Object(obj) = &mut message {
            obj.insert("type".into(), Value::String("message".into()));
        }
        messages.push(message);
    }
    Ok(messages)
}

fn render_turn_message(turn: &Turn, shape: PartShape) -> EndpointResult<Value> {
    Ok(
        json!({"role": turn.role.as_deref().unwrap_or("user"), "content": render_turn_content(turn, shape)?}),
    )
}

fn render_turn_content(turn: &Turn, shape: PartShape) -> EndpointResult<Value> {
    if turn.texts.len() == 1
        && turn.texts[0].contents.len() == 1
        && turn.images.is_empty()
        && turn.audios.is_empty()
        && turn.videos.is_empty()
    {
        return Ok(Value::String(turn.texts[0].contents[0].clone()));
    }
    let mut parts = Vec::new();
    extend_parts(&mut parts, &turn.texts, |content| {
        render_text_part(content, shape)
    });
    extend_parts(&mut parts, &turn.images, |content| {
        render_image_part(content, shape)
    });
    for media in &turn.audios {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render_audio_part(content)?);
            }
        }
    }
    for media in &turn.videos {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render_video_part(content, shape)?);
            }
        }
    }
    Ok(Value::Array(parts))
}

fn extend_parts<F>(parts: &mut Vec<Value>, media_items: &[Media], mut render: F)
where
    F: FnMut(&str) -> Value,
{
    for media in media_items {
        for content in &media.contents {
            if !content.is_empty() {
                parts.push(render(content));
            }
        }
    }
}

fn render_text_part(text: &str, shape: PartShape) -> Value {
    match shape {
        PartShape::Chat => json!({"type":"text","text":text}),
        PartShape::Responses => json!({"type":"input_text","text":text}),
    }
}
fn render_image_part(url: &str, shape: PartShape) -> Value {
    match shape {
        PartShape::Chat => json!({"type":"image_url","image_url":{"url":url}}),
        PartShape::Responses => json!({"type":"input_image","image_url":url}),
    }
}
fn render_audio_part(format_and_b64: &str) -> EndpointResult<Value> {
    let Some((left, b64)) = format_and_b64.split_once(',') else {
        return Err(EndpointError::InvalidRequest(
            "audio content must be in the format 'format,b64_audio'".into(),
        ));
    };
    let format = left
        .strip_prefix("data:audio/")
        .and_then(|rest| rest.split_once(';').map(|(fmt, _)| fmt))
        .filter(|fmt| !fmt.is_empty())
        .unwrap_or(left);
    Ok(json!({"type":"input_audio","input_audio":{"data":b64,"format":format}}))
}
fn render_video_part(url: &str, shape: PartShape) -> EndpointResult<Value> {
    match shape {
        PartShape::Chat => Ok(json!({"type":"video_url","video_url":{"url":url}})),
        PartShape::Responses => Err(EndpointError::InvalidRequest(
            "Responses API does not support video input".into(),
        )),
    }
}

fn format_chat_messages(request_info: &RequestInfo, mut rendered: Vec<Value>) -> Vec<Value> {
    let mut messages = Vec::new();
    let first_is_system = rendered
        .first()
        .and_then(Value::as_object)
        .and_then(|obj| obj.get("role"))
        .and_then(Value::as_str)
        == Some("system");
    if let Some(system) = request_info
        .system_message
        .as_ref()
        .filter(|s| !s.is_empty())
    {
        if first_is_system && request_info.credit_phase == CreditPhase::Warmup {
            rendered = prepend_system_message(rendered, system);
        } else if !first_is_system {
            messages.push(json!({"role":"system","content":system}));
        }
    }
    if let Some(context) = request_info
        .user_context_message
        .as_ref()
        .filter(|s| !s.is_empty())
    {
        messages.push(json!({"role":"user","content":context}));
    }
    messages.extend(rendered);
    messages
}

fn prepend_system_message(mut rendered: Vec<Value>, system: &str) -> Vec<Value> {
    if let Some(Value::Object(first)) = rendered.first_mut() {
        match first.get_mut("content") {
            Some(Value::String(content)) if content.is_empty() => *content = system.to_string(),
            Some(Value::String(content)) => *content = format!("{system}\n{content}"),
            Some(Value::Array(parts)) => parts.insert(0, json!({"type":"text","text":system})),
            Some(Value::Null) | None => {
                first.insert("content".into(), Value::String(system.to_string()));
            }
            Some(other) => *other = Value::String(format!("{system}\n{other}")),
        }
    }
    rendered
}

fn latest_turn_attr<'a, T, F>(turns: &'a [Turn], get: F) -> Option<&'a T>
where
    F: Fn(&'a Turn) -> Option<&'a T>,
{
    turns.iter().rev().find_map(get)
}

fn merge_extra(payload: &mut Map<String, Value>, extra: Option<&Map<String, Value>>) {
    if let Some(extra) = extra {
        for (key, value) in extra {
            payload.insert(key.clone(), value.clone());
        }
    }
}
fn ensure_include_usage(payload: &mut Map<String, Value>) {
    match payload.get_mut("stream_options") {
        Some(Value::Object(stream_options)) => {
            stream_options
                .entry("include_usage")
                .or_insert(Value::Bool(true));
        }
        Some(_) | None => {
            payload.insert("stream_options".into(), json!({"include_usage": true}));
        }
    }
}

fn first_choice(obj: &Map<String, Value>) -> Option<&Map<String, Value>> {
    obj.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
}
fn non_empty_field(obj: &Map<String, Value>, field: &str) -> Option<Value> {
    match obj.get(field) {
        Some(Value::Null) | None => None,
        Some(Value::Object(map)) if map.is_empty() => None,
        Some(value) => Some(value.clone()),
    }
}

fn extract_chat_response_data(obj: &Map<String, Value>) -> Option<ResponseData> {
    let key = match obj.get("object").and_then(Value::as_str) {
        Some("chat.completion") => "message",
        Some("chat.completion.chunk") => "delta",
        _ => return None,
    };
    let data = first_choice(obj)?.get(key)?.as_object()?;
    let content = data
        .get("content")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    if let Some(reasoning) = data
        .get("reasoning_content")
        .or_else(|| data.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return Some(ResponseData::Reasoning {
            content,
            reasoning: reasoning.to_string(),
        });
    }
    let mut parts = Vec::new();
    if let Some(tool_calls) = data.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let Some(function) = tc
                .as_object()
                .and_then(|tc| tc.get("function"))
                .and_then(Value::as_object)
            else {
                continue;
            };
            if let Some(name) = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                parts.push(name.to_string());
            }
            if let Some(arguments) = function
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                parts.push(arguments.to_string());
            }
        }
    }
    let tool_call_text = parts.concat();
    if !tool_call_text.is_empty() {
        return Some(ResponseData::ToolCall {
            tool_call_text,
            content: content.filter(|s| !s.is_empty()),
        });
    }
    content
        .filter(|s| !s.is_empty())
        .map(|text| ResponseData::Text { text })
}

fn absorb_chat_choice(
    object_type: Option<&str>,
    choice: &Map<String, Value>,
    content_parts: &mut Vec<String>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    match object_type {
        Some("chat.completion") => {
            let Some(message) = choice.get("message").and_then(Value::as_object) else {
                return;
            };
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                content_parts.push(content.to_string());
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    let Some(tc_obj) = tc.as_object() else {
                        continue;
                    };
                    let idx = tc_obj
                        .get("index")
                        .and_then(Value::as_i64)
                        .unwrap_or(tool_calls_by_index.len() as i64);
                    let mut cloned = tc_obj.clone();
                    cloned.remove("index");
                    tool_calls_by_index.insert(idx, cloned);
                }
            }
            absorb_legacy_function_call(message.get("function_call"), tool_calls_by_index);
        }
        Some("chat.completion.chunk") => {
            let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
                return;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                content_parts.push(content.to_string());
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    if let Some(tc) = tc.as_object() {
                        merge_tool_call_delta(tc, tool_calls_by_index);
                    }
                }
            }
            merge_legacy_function_call_delta(delta.get("function_call"), tool_calls_by_index);
        }
        _ => {}
    }
}

fn absorb_legacy_function_call(
    value: Option<&Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(function_call) = value.and_then(Value::as_object) else {
        return;
    };
    let idx = tool_calls_by_index.len() as i64;
    tool_calls_by_index.insert(idx, json!({"type":"function","function":{"name":function_call.get("name").and_then(Value::as_str).unwrap_or(""),"arguments":function_call.get("arguments").and_then(Value::as_str).unwrap_or("")}}).as_object().unwrap().clone());
}
fn merge_legacy_function_call_delta(
    value: Option<&Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(delta) = value.and_then(Value::as_object) else {
        return;
    };
    let existing = tool_calls_by_index.entry(0).or_insert_with(|| {
        json!({"type":"function","function":{}})
            .as_object()
            .unwrap()
            .clone()
    });
    existing
        .entry("type")
        .or_insert_with(|| Value::String("function".into()));
    let function = ensure_object_field(existing, "function");
    if let Some(name) = delta
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        concat_string_field(function, "name", name);
    }
    if let Some(arguments) = delta.get("arguments") {
        concat_string_field(function, "arguments", arguments.as_str().unwrap_or(""));
    }
}
fn merge_tool_call_delta(
    delta: &Map<String, Value>,
    tool_calls_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let idx = delta
        .get("index")
        .and_then(Value::as_i64)
        .unwrap_or(tool_calls_by_index.len() as i64);
    let existing = tool_calls_by_index.entry(idx).or_default();
    if let Some(id) = delta
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        existing.insert("id".into(), Value::String(id.into()));
    }
    if let Some(ty) = delta
        .get("type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        existing.insert("type".into(), Value::String(ty.into()));
    }
    let Some(fn_delta) = delta.get("function").and_then(Value::as_object) else {
        return;
    };
    let function = ensure_object_field(existing, "function");
    if let Some(name) = fn_delta
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        function.insert("name".into(), Value::String(name.into()));
    }
    if let Some(arguments) = fn_delta.get("arguments") {
        concat_string_field(function, "arguments", arguments.as_str().unwrap_or(""));
    }
}
fn ensure_object_field<'a>(
    map: &'a mut Map<String, Value>,
    field: &str,
) -> &'a mut Map<String, Value> {
    if !matches!(map.get(field), Some(Value::Object(_))) {
        map.insert(field.into(), Value::Object(Map::new()));
    }
    map.get_mut(field).and_then(Value::as_object_mut).unwrap()
}
fn concat_string_field(map: &mut Map<String, Value>, field: &str, suffix: &str) {
    let mut value = map
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    value.push_str(suffix);
    map.insert(field.into(), Value::String(value));
}

fn parse_responses_streaming_event(
    obj: &Map<String, Value>,
    perf_ns: u64,
) -> Option<ParsedResponse> {
    let event_type = obj.get("type").and_then(Value::as_str)?;
    let data = match event_type {
        "response.output_text.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|text| ResponseData::Text { text: text.into() }),
        "response.reasoning_text.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|reasoning| ResponseData::Reasoning {
                content: None,
                reasoning: reasoning.into(),
            }),
        "response.output_text.done" => obj
            .get("text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|text| ResponseData::Text { text: text.into() }),
        "response.function_call_arguments.delta" => obj
            .get("delta")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|tool_call_text| ResponseData::ToolCall {
                tool_call_text: tool_call_text.into(),
                content: None,
            }),
        _ => None,
    };
    if data.is_some() {
        return Some(ParsedResponse {
            perf_ns,
            data,
            usage: None,
        });
    }
    if event_type == "response.completed" {
        let usage = obj
            .get("response")
            .and_then(Value::as_object)
            .and_then(|response| non_empty_field(response, "usage"));
        if usage.is_some() {
            return Some(ParsedResponse {
                perf_ns,
                data: None,
                usage,
            });
        }
    }
    None
}

fn extract_responses_full_content(obj: &Map<String, Value>) -> Option<ResponseData> {
    if let Some(output) = obj.get("output").and_then(Value::as_array) {
        let mut text_parts = Vec::new();
        let mut reasoning_parts = Vec::new();
        let mut tool_parts = Vec::new();
        for item in output {
            let Some(item) = item.as_object() else {
                continue;
            };
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => collect_reasoning_summary(item, &mut reasoning_parts),
                Some("message") => collect_message_content(item, &mut text_parts),
                Some("function_call") => collect_function_call(item, &mut tool_parts),
                _ => {}
            }
        }
        if !reasoning_parts.is_empty() {
            return Some(ResponseData::Reasoning {
                content: (!text_parts.is_empty()).then(|| text_parts.concat()),
                reasoning: reasoning_parts.concat(),
            });
        }
        if !text_parts.is_empty() {
            return Some(ResponseData::Text {
                text: text_parts.concat(),
            });
        }
        if !tool_parts.is_empty() {
            return Some(ResponseData::ToolCall {
                tool_call_text: tool_parts.concat(),
                content: None,
            });
        }
    }
    obj.get("output_text")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|text| ResponseData::Text { text: text.into() })
}
fn collect_reasoning_summary(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for part in summary {
            if let Some(part) = part.as_object()
                && part.get("type").and_then(Value::as_str) == Some("summary_text")
                && let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            {
                out.push(text.into());
            }
        }
    }
}
fn collect_message_content(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(content) = item.get("content").and_then(Value::as_array) {
        for part in content {
            if let Some(part) = part.as_object()
                && part.get("type").and_then(Value::as_str) == Some("output_text")
                && let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            {
                out.push(text.into());
            }
        }
    }
}
fn collect_function_call(item: &Map<String, Value>, out: &mut Vec<String>) {
    if let Some(name) = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        out.push(name.into());
    }
    if let Some(arguments) = item
        .get("arguments")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        out.push(arguments.into());
    }
}

fn is_replay_unsafe_output_item(ty: &str) -> bool {
    matches!(
        ty,
        "web_search_call"
            | "file_search_call"
            | "image_generation_call"
            | "code_interpreter_call"
            | "computer_call"
            | "reasoning"
    )
}
fn is_responses_failure_event(obj: &Map<String, Value>) -> bool {
    matches!(
        obj.get("type").and_then(Value::as_str),
        Some("response.failed" | "response.incomplete" | "response.error" | "error")
    )
}
fn collect_response_items(
    obj: &Map<String, Value>,
    items_by_key: &mut Map<String, Value>,
    done_items: &mut Vec<Value>,
) {
    if obj.get("object").and_then(Value::as_str) == Some("response") {
        merge_output_list(items_by_key, obj.get("output"));
        return;
    }
    match obj.get("type").and_then(Value::as_str) {
        Some("response.completed") => merge_output_list(
            items_by_key,
            obj.get("response")
                .and_then(Value::as_object)
                .and_then(|response| response.get("output")),
        ),
        Some("response.output_item.done") => {
            if let Some(item) = obj.get("item").filter(|item| item.is_object()) {
                done_items.push(item.clone());
            }
        }
        _ => {}
    }
}
fn merge_output_list(items_by_key: &mut Map<String, Value>, output: Option<&Value>) {
    if let Some(output) = output.and_then(Value::as_array) {
        for item in output {
            if item.is_object() {
                merge_response_item(items_by_key, item.clone());
            }
        }
    }
}
fn merge_response_item(items_by_key: &mut Map<String, Value>, item: Value) {
    let Some(obj) = item.as_object() else {
        return;
    };
    let key = obj
        .get("id")
        .or_else(|| obj.get("call_id"))
        .or_else(|| obj.get("item_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| {
            format!(
                "{}::{}",
                obj.get("type").and_then(Value::as_str).unwrap_or("?"),
                serde_json::to_string(obj).unwrap_or_default()
            )
        });
    items_by_key.entry(key).or_insert(item);
}

fn turn_texts(turn: &Turn) -> Vec<String> {
    turn.texts
        .iter()
        .flat_map(|text| text.contents.iter())
        .filter(|content| !content.is_empty())
        .cloned()
        .collect()
}

fn parse_embeddings_response(
    response: &ServerResponse,
    strict_invalid_data: bool,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(obj) = response.json.as_ref().and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(data) = obj.get("data") else {
        if !strict_invalid_data && let Some(embeddings) = try_extract_embeddings(obj) {
            return Ok(Some(ParsedResponse {
                perf_ns: response.perf_ns,
                data: Some(ResponseData::Embeddings { embeddings }),
                usage: None,
            }));
        }
        return Ok(None);
    };
    let Some(data_array) = data.as_array() else {
        return if strict_invalid_data {
            Err(EndpointError::InvalidResponse(format!(
                "received invalid list in response: {}",
                Value::Object(obj.clone())
            )))
        } else {
            Ok(None)
        };
    };
    if data_array.is_empty() {
        return Ok(None);
    }
    if data_array.iter().all(|item| {
        item.as_object()
            .and_then(|obj| obj.get("object"))
            .and_then(Value::as_str)
            == Some("embedding")
    }) {
        let embeddings: Vec<Vec<f64>> = data_array
            .iter()
            .filter_map(|item| item.as_object()?.get("embedding"))
            .filter_map(number_array)
            .collect();
        if embeddings.is_empty() {
            return Ok(None);
        }
        return Ok(Some(ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Embeddings { embeddings }),
            usage: None,
        }));
    }
    if strict_invalid_data {
        Err(EndpointError::InvalidResponse(format!(
            "received invalid list in response: {}",
            Value::Object(obj.clone())
        )))
    } else {
        Ok(None)
    }
}
fn try_extract_embeddings(obj: &Map<String, Value>) -> Option<Vec<Vec<f64>>> {
    for field in ["embeddings", "embedding"] {
        let Some(value) = obj.get(field) else {
            continue;
        };
        if let Some(numbers) = number_array(value) {
            return Some(vec![numbers]);
        }
        if let Some(arrays) = value.as_array() {
            let embeddings: Vec<Vec<f64>> = arrays.iter().filter_map(number_array).collect();
            if !embeddings.is_empty() {
                return Some(embeddings);
            }
        }
    }
    None
}
fn number_array(value: &Value) -> Option<Vec<f64>> {
    let array = value.as_array()?;
    if array.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        out.push(item.as_f64()?);
    }
    Some(out)
}

fn build_plain_assistant_turn<E: Endpoint + ?Sized>(
    endpoint: &E,
    record: &RequestRecord,
) -> EndpointResult<Option<Turn>> {
    let mut output_texts = Vec::new();
    for parsed in endpoint.extract_response_data(record)? {
        let Some(data) = parsed.data else {
            continue;
        };
        match data {
            ResponseData::Reasoning {
                content: Some(content),
                ..
            } => output_texts.push(content),
            ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
                output_texts.push(reasoning)
            }
            other => {
                let text = other.get_text();
                if !text.is_empty() {
                    output_texts.push(text);
                }
            }
        }
    }
    let text = output_texts.concat();
    Ok((!text.is_empty()).then_some(Turn {
        role: Some("assistant".into()),
        texts: vec![Media::new(vec![text])],
        ..Turn::default()
    }))
}
