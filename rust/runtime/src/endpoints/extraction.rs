// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-pass payload input extraction for tokenization and media accounting.

use serde_json::{Map, Value, json};

use crate::endpoints::models::ExtractedPayload;

/// Content-part type names emitted by an endpoint, grouped by media kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartTypes {
    text: &'static [&'static str],
    image: &'static [&'static str],
    audio: &'static [&'static str],
    video: &'static [&'static str],
}

impl PartTypes {
    /// Chat Completions content part names.
    pub fn chat() -> Self {
        Self {
            text: &["text"],
            image: &["image_url"],
            audio: &["input_audio"],
            video: &["video_url"],
        }
    }

    /// Responses API content part names.
    pub fn responses() -> Self {
        Self {
            text: &["input_text"],
            image: &["input_image"],
            audio: &["input_audio"],
            video: &[],
        }
    }

    /// Anthropic Messages content part names.
    pub fn messages() -> Self {
        Self {
            text: &["text"],
            image: &["image"],
            audio: &[],
            video: &[],
        }
    }

    fn kind_for(&self, ty: &str) -> Option<MediaKind> {
        if self.text.contains(&ty) {
            Some(MediaKind::Text)
        } else if self.image.contains(&ty) {
            Some(MediaKind::Image)
        } else if self.audio.contains(&ty) {
            Some(MediaKind::Audio)
        } else if self.video.contains(&ty) {
            Some(MediaKind::Video)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MediaKind {
    Text,
    Image,
    Audio,
    Video,
}

/// Extract payload inputs using chat content-part type names.
pub fn extract_payload(payload: &Value) -> ExtractedPayload {
    extract_inputs(payload, &PartTypes::chat())
}

pub(crate) fn extract_inputs(payload: &Value, part_types: &PartTypes) -> ExtractedPayload {
    let mut result = ExtractedPayload::default();
    let Some(obj) = payload.as_object() else {
        return result;
    };
    let found_items_shape = walk_items_arrays(obj, &mut result, part_types);
    walk_tools_schema(obj, &mut result);
    if !found_items_shape {
        walk_flat_fallbacks(obj, &mut result);
    }
    result
}

fn walk_items_arrays(
    payload: &Map<String, Value>,
    result: &mut ExtractedPayload,
    part_types: &PartTypes,
) -> bool {
    let mut found = false;
    let mut chat_messages = Vec::new();
    for items_field in ["messages", "input"] {
        let Some(items) = payload.get(items_field).and_then(Value::as_array) else {
            continue;
        };
        if items.is_empty()
            || !items.iter().any(|item| {
                item.as_object()
                    .is_some_and(|obj| obj.contains_key("role") || obj.contains_key("type"))
            })
        {
            continue;
        }
        found = true;
        for item in items {
            if let Some(item) = item.as_object() {
                walk_item(item, result, part_types, &mut chat_messages);
            }
        }
    }
    if found {
        result.messages = Some(chat_messages);
    }
    found
}

fn walk_item(
    item: &Map<String, Value>,
    result: &mut ExtractedPayload,
    part_types: &PartTypes,
    chat_messages: &mut Vec<Value>,
) {
    let msg_text_parts = walk_item_content(item, result, part_types);
    // A string `role` means this item rides in the `messages`/`input` array the
    // chat template renders. That routing decides where tool-call text is
    // counted so it is not double-counted (see `walk_item_tool_calls`).
    let in_messages = item.get("role").and_then(Value::as_str).is_some();
    walk_item_tool_calls(item, result, in_messages);
    walk_item_function_call(item, result);
    if let Some(role) = item.get("role").and_then(Value::as_str) {
        let mut msg = json!({"role": role, "content": msg_text_parts.concat()});
        // Pass replayed assistant `tool_calls` through: chat templates render
        // them, so dropping them would undercount the templated ISL for agent
        // replays.
        if let Some(tool_calls) = item
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|calls| !calls.is_empty())
            && let Some(obj) = msg.as_object_mut()
        {
            obj.insert("tool_calls".into(), Value::Array(tool_calls.clone()));
        }
        chat_messages.push(msg);
    }
}

fn walk_item_content(
    item: &Map<String, Value>,
    result: &mut ExtractedPayload,
    part_types: &PartTypes,
) -> Vec<String> {
    let mut msg_text_parts = Vec::new();
    match item.get("content") {
        Some(Value::String(content)) => {
            result.texts.push(content.clone());
            msg_text_parts.push(content.clone());
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                if let Some(part) = part.as_object() {
                    walk_content_part(part, result, part_types, &mut msg_text_parts);
                }
            }
        }
        _ => {}
    }
    msg_text_parts
}

fn walk_content_part(
    part: &Map<String, Value>,
    result: &mut ExtractedPayload,
    part_types: &PartTypes,
    msg_text_parts: &mut Vec<String>,
) {
    let Some(ty) = part.get("type").and_then(Value::as_str) else {
        return;
    };
    match part_types.kind_for(ty) {
        Some(MediaKind::Text) => {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                result.texts.push(text.to_string());
                msg_text_parts.push(text.to_string());
            }
        }
        Some(MediaKind::Image) => result.image_count += 1,
        Some(MediaKind::Audio) => result.audio_count += 1,
        Some(MediaKind::Video) => result.video_count += 1,
        None => {}
    }
}

/// Account a chat-shape assistant message's replayed `tool_calls`.
///
/// `in_messages` routes the collected `function.name`/`arguments` strings: when
/// the item rides in the rendered `messages` array (string role), the tool_calls
/// are passed through in `messages` and the chat template tokenizes them, so
/// they go to `texts` only — adding them to `tool_texts` would double-count them
/// on the chat-template ISL path. Otherwise (Responses `input` items with no
/// role) they go to both ledgers via `append_tool_texts`.
fn walk_item_tool_calls(
    item: &Map<String, Value>,
    result: &mut ExtractedPayload,
    in_messages: bool,
) {
    let Some(tool_calls) = item.get("tool_calls").and_then(Value::as_array) else {
        return;
    };
    for tc in tool_calls {
        let Some(function) = tc
            .as_object()
            .and_then(|tc| tc.get("function"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let mut collected = Vec::new();
        collect_str_fields(function, &["name", "arguments"], &mut collected);
        if in_messages {
            result.texts.extend(collected);
        } else {
            append_tool_texts(result, collected);
        }
    }
}

fn walk_item_function_call(item: &Map<String, Value>, result: &mut ExtractedPayload) {
    let mut collected = Vec::new();
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => collect_str_fields(item, &["name", "arguments"], &mut collected),
        Some("function_call_output") => {
            if let Some(output) = item
                .get("output")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                collected.push(output.to_string());
            }
        }
        _ => {}
    }
    append_tool_texts(result, collected);
}

fn walk_tools_schema(payload: &Map<String, Value>, result: &mut ExtractedPayload) {
    let Some(tools) = payload.get("tools").and_then(Value::as_array) else {
        return;
    };
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if let Some(function) = tool.get("function").and_then(Value::as_object) {
            collect_tool_source(function, result);
        }
        collect_tool_source(tool, result);
    }
}

fn collect_tool_source(source: &Map<String, Value>, result: &mut ExtractedPayload) {
    let mut collected = Vec::new();
    collect_str_fields(source, &["name", "description"], &mut collected);
    if let Some(parameters) = source.get("parameters").and_then(Value::as_object)
        && let Ok(serialized) = serde_json::to_string(parameters)
    {
        collected.push(serialized);
    }
    append_tool_texts(result, collected);
}

fn append_tool_texts(result: &mut ExtractedPayload, collected: Vec<String>) {
    result.texts.extend(collected.iter().cloned());
    result.tool_texts.extend(collected);
}

fn collect_str_fields(source: &Map<String, Value>, keys: &[&str], out: &mut Vec<String>) {
    for key in keys {
        if let Some(value) = source
            .get(*key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            out.push(value.to_string());
        }
    }
}

fn walk_flat_fallbacks(payload: &Map<String, Value>, result: &mut ExtractedPayload) {
    if append_string_or_list(payload, "token_ids", result) {
        return;
    }
    if append_string_or_list(payload, "prompt", result) {
        return;
    }
    if append_string_or_list(payload, "input", result) {
        return;
    }
    if append_query_passages(payload, result) {
        return;
    }
    if let Some(inputs) = payload.get("inputs").and_then(Value::as_str) {
        result.texts.push(inputs.to_string());
    }
}

fn append_string_or_list(
    payload: &Map<String, Value>,
    key: &str,
    result: &mut ExtractedPayload,
) -> bool {
    let Some(value) = payload.get(key) else {
        return false;
    };
    if let Some(text) = value.as_str() {
        result.texts.push(text.to_string());
        return true;
    }
    let Some(list) = value.as_array() else {
        return false;
    };
    if list.iter().all(Value::is_string) {
        result.texts.extend(
            list.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string),
        );
        return true;
    }
    if !list.is_empty() && list.iter().all(|item| item.as_i64().is_some()) {
        result.pretokenised_token_count += list.len() as u64;
        return true;
    }
    if !list.is_empty()
        && list.iter().all(|item| {
            item.as_array()
                .is_some_and(|inner| inner.iter().all(|value| value.as_i64().is_some()))
        })
    {
        result.pretokenised_token_count += list
            .iter()
            .filter_map(Value::as_array)
            .map(|inner| inner.len() as u64)
            .sum::<u64>();
        return true;
    }
    false
}

fn append_query_passages(payload: &Map<String, Value>, result: &mut ExtractedPayload) -> bool {
    let Some(query) = payload.get("query").and_then(Value::as_str) else {
        return false;
    };
    let Some(passages) = payload.get("passages").and_then(Value::as_array) else {
        return false;
    };
    result.texts.push(query.to_string());
    for passage in passages {
        if let Some(text) = passage.as_str() {
            result.texts.push(text.to_string());
        } else if let Some(text) = passage
            .as_object()
            .and_then(|obj| obj.get("text"))
            .and_then(Value::as_str)
        {
            result.texts.push(text.to_string());
        }
    }
    true
}
