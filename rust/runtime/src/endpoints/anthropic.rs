// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages request, response, usage, and replay dialect.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::extraction::{PartTypes, extract_inputs};
use crate::endpoints::implementation::{
    build_plain_assistant_turn, build_reserved_plan, format_messages_array_wires, latest_turn_attr,
    merge_extra, non_empty_field, require_prepared_turns,
};
use crate::endpoints::metadata::{EndpointDescriptor, Modality};
use crate::endpoints::models::{
    EndpointResult, ExtractedPayload, ParsedResponse, RequestInfo, RequestRecord, ResponseData,
    ServerResponse, Turn,
};
use crate::endpoints::registry::{
    PreparedEndpointBehavior, PreparedRequest, format_legacy_payload,
};
use crate::endpoints::{Endpoint, EndpointConfig, RawEndpointConfig};

const ANTHROPIC_VERSION: &str = "2023-06-01";

const MESSAGE: &str = "message";
const MESSAGE_START: &str = "message_start";
const CONTENT_BLOCK_START: &str = "content_block_start";
const CONTENT_BLOCK_DELTA: &str = "content_block_delta";
const CONTENT_BLOCK_STOP: &str = "content_block_stop";
const MESSAGE_DELTA: &str = "message_delta";
const MESSAGE_STOP: &str = "message_stop";
const PING: &str = "ping";
const ERROR: &str = "error";

const TEXT: &str = "text";
const THINKING: &str = "thinking";
const TOOL_USE: &str = "tool_use";
const TOOL_RESULT: &str = "tool_result";

const TEXT_DELTA: &str = "text_delta";
const THINKING_DELTA: &str = "thinking_delta";
const INPUT_JSON_DELTA: &str = "input_json_delta";
const SIGNATURE_DELTA: &str = "signature_delta";

/// Anthropic Messages API dialect (`/v1/messages`).
#[derive(Debug, Clone, Copy, Default)]
pub struct MessagesEndpoint;

const MESSAGES_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "messages",
    aliases: &[],
    description: "Anthropic Messages API",
    endpoint_path: Some("/v1/messages"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

impl Endpoint for MessagesEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &MESSAGES_DESCRIPTOR
    }

    fn format_headers(&self, config: &EndpointConfig) -> BTreeMap<String, String> {
        let mut headers =
            BTreeMap::from([("content-type".to_string(), "application/json".to_string())]);
        headers.extend(config.headers.clone());
        if let Some(api_key) = config.api_key.as_ref().filter(|key| !key.is_empty()) {
            headers.insert("x-api-key".into(), api_key.clone());
        }
        headers
            .entry("anthropic-version".into())
            .or_insert_with(|| ANTHROPIC_VERSION.into());
        headers
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let Some(event_type) = object.get("type").and_then(Value::as_str) else {
            return Ok(None);
        };
        if normalized_eq(event_type, MESSAGE) {
            return Ok(parse_non_streaming(response.perf_ns, object));
        }
        Ok(parse_streaming_event(response.perf_ns, object, event_type))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let mut result = extract_inputs(body, &PartTypes::messages());
        let Some(payload) = body.as_object() else {
            return result;
        };
        walk_system(payload, &mut result);
        walk_tool_schemas(payload, &mut result);
        walk_tool_blocks(payload, &mut result);
        result
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        let mut parsed = Vec::new();
        for response in &record.responses {
            if let Some(item) = self.parse_response(response)? {
                parsed.push(item);
            }
        }
        let usage_indices = parsed
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                item.usage
                    .as_ref()
                    .and_then(Value::as_object)
                    .filter(|usage| !usage.is_empty())
                    .map(|_| index)
            })
            .collect::<Vec<_>>();
        if let Some((&final_index, earlier_indices)) = usage_indices.split_last()
            && !earlier_indices.is_empty()
        {
            let earlier = earlier_indices
                .iter()
                .filter_map(|index| parsed[*index].usage.as_ref()?.as_object().cloned())
                .collect::<Vec<_>>();
            if let Some(final_usage) = parsed[final_index]
                .usage
                .as_mut()
                .and_then(Value::as_object_mut)
            {
                for usage in earlier {
                    for (key, value) in usage {
                        final_usage.entry(key).or_insert(value);
                    }
                }
            }
        }
        Ok(parsed)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        let mut text_parts = Vec::new();
        let mut thinking_by_index = BTreeMap::<i64, Map<String, Value>>::new();
        let mut tool_uses_by_index = BTreeMap::<i64, Map<String, Value>>::new();

        for response in &record.responses {
            let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
                continue;
            };
            absorb_event(
                object,
                &mut text_parts,
                &mut thinking_by_index,
                &mut tool_uses_by_index,
            );
        }

        if thinking_by_index.is_empty() && tool_uses_by_index.is_empty() {
            return build_plain_assistant_turn(self, record);
        }

        let mut content = Vec::new();
        content.extend(thinking_by_index.into_values().map(finalize_thinking));
        let text = text_parts.concat();
        if !text.is_empty() {
            content.push(json!({"type":"text","text":text}));
        }
        content.extend(tool_uses_by_index.into_values().map(finalize_tool_use));
        Ok(Some(Turn {
            role: Some("assistant".into()),
            raw_messages: Some(vec![json!({"role":"assistant","content":content})]),
            ..Turn::default()
        }))
    }

    fn part_types(&self) -> PartTypes {
        PartTypes::messages()
    }

    fn captures_assistant_turn(&self) -> bool {
        true
    }
}

impl PreparedEndpointBehavior for MessagesEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        endpoint: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns = require_prepared_turns(
            request,
            "Anthropic Messages endpoint requires at least one turn.",
        )?;
        let last = turns.last().expect("non-empty turns");

        let message_wires = format_messages_array_wires(request, turns)?;

        let mut payload = Map::new();
        payload.insert(
            "model".into(),
            Value::String(
                last.model
                    .clone()
                    .filter(|model| !model.is_empty())
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        // Reserved slot: the value is discarded, the key fixes the field's
        // insertion position, and `fill_reserved` supplies the real wires.
        payload.insert("messages".into(), Value::Null);
        payload.insert(
            "max_tokens".into(),
            Value::from(last.max_tokens.unwrap_or(1_024)),
        );
        if endpoint.streaming {
            payload.insert("stream".into(), Value::Bool(true));
        }

        if let Some(raw_system) = latest_turn_attr(turns, |turn| turn.raw_system.as_ref()) {
            let system =
                if let Some(system) = request.system_message().filter(|value| !value.is_empty()) {
                    let mut combined = Vec::with_capacity(raw_system.len() + 1);
                    combined.push(json!({"type":"text","text":system}));
                    combined.extend(raw_system.iter().cloned());
                    combined
                } else {
                    raw_system.clone()
                };
            payload.insert("system".into(), Value::Array(system));
        } else if let Some(system) = request.system_message().filter(|value| !value.is_empty()) {
            payload.insert("system".into(), Value::String(system.to_string()));
        }
        if let Some(tools) = latest_turn_attr(turns, |turn| turn.raw_tools.as_ref()) {
            payload.insert("tools".into(), Value::Array(tools.clone()));
        }

        merge_extra(&mut payload, endpoint.extra.as_ref());
        merge_extra(&mut payload, last.extra_body.as_ref());
        build_reserved_plan(&payload, "messages", message_wires)
    }

    fn renders_all_turns(&self) -> bool {
        true
    }

    fn splices_lowered_wires(&self) -> bool {
        true
    }
}

/// Compare a wire event/type name against an already-normalized constant
/// without allocating: `-` is treated as `_` and ASCII case is ignored. Runs on
/// every streamed frame, so it stays alloc-free (the previous `replace` built a
/// short `String` per token).
fn normalized_eq(actual: &str, expected: &str) -> bool {
    actual.len() == expected.len()
        && actual.bytes().zip(expected.bytes()).all(|(a, e)| {
            let a = if a == b'-' { b'_' } else { a };
            a.eq_ignore_ascii_case(&e)
        })
}

fn parse_non_streaming(perf_ns: u64, object: &Map<String, Value>) -> Option<ParsedResponse> {
    let data = extract_content_data(object);
    let usage = object.get("usage").cloned();
    (data.is_some() || usage.as_ref().is_some_and(python_truthy)).then_some(ParsedResponse {
        perf_ns,
        data,
        usage,
        sources: None,
    })
}

fn extract_content_data(object: &Map<String, Value>) -> Option<ResponseData> {
    let blocks = object.get("content").and_then(Value::as_array)?;
    if blocks.is_empty() {
        return None;
    }
    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_call_parts = Vec::new();
    for block in blocks.iter().filter_map(Value::as_object) {
        accumulate_content_block(
            block,
            &mut text_parts,
            &mut thinking_parts,
            &mut tool_call_parts,
        );
    }
    let text = non_empty_concat(text_parts);
    let thinking = non_empty_concat(thinking_parts);
    let tool_call_text = tool_call_parts.concat();
    if let Some(reasoning) = thinking {
        return Some(ResponseData::Reasoning {
            content: text,
            reasoning,
        });
    }
    if !tool_call_text.is_empty() {
        return Some(ResponseData::ToolCall {
            tool_call_text,
            content: text,
        });
    }
    text.map(|text| ResponseData::Text { text })
}

fn accumulate_content_block(
    block: &Map<String, Value>,
    text_parts: &mut Vec<String>,
    thinking_parts: &mut Vec<String>,
    tool_call_parts: &mut Vec<String>,
) {
    let Some(block_type) = block.get("type").and_then(Value::as_str) else {
        return;
    };
    if normalized_eq(block_type, TEXT) {
        if let Some(text) = block
            .get("text")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            text_parts.push(text.into());
        }
    } else if normalized_eq(block_type, THINKING) {
        if let Some(thinking) = block
            .get("thinking")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            thinking_parts.push(thinking.into());
        }
    } else if normalized_eq(block_type, TOOL_USE) {
        if let Some(name) = block
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            tool_call_parts.push(name.into());
        }
        if let Some(input) = block.get("input").filter(|value| python_truthy(value))
            && let Ok(serialized) = serde_json::to_string(input)
        {
            tool_call_parts.push(serialized);
        }
    }
}

fn parse_streaming_event(
    perf_ns: u64,
    object: &Map<String, Value>,
    event_type: &str,
) -> Option<ParsedResponse> {
    if normalized_eq(event_type, MESSAGE_START) {
        let usage = object
            .get("message")
            .and_then(Value::as_object)
            .and_then(|message| non_empty_field(message, "usage"));
        return usage.map(|usage| ParsedResponse {
            perf_ns,
            data: None,
            usage: Some(usage),
            sources: None,
        });
    }
    if normalized_eq(event_type, CONTENT_BLOCK_DELTA) {
        return parse_content_block_delta(perf_ns, object);
    }
    if normalized_eq(event_type, MESSAGE_DELTA) {
        return non_empty_field(object, "usage").map(|usage| ParsedResponse {
            perf_ns,
            data: None,
            usage: Some(usage),
            sources: None,
        });
    }
    if normalized_eq(event_type, PING)
        || normalized_eq(event_type, CONTENT_BLOCK_START)
        || normalized_eq(event_type, CONTENT_BLOCK_STOP)
        || normalized_eq(event_type, MESSAGE_STOP)
    {
        return None;
    }
    if normalized_eq(event_type, ERROR) {
        let error_detail = object.get("error").and_then(Value::as_object);
        let error_type = error_detail
            .and_then(|error| error.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let error_message = error_detail
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        tracing::warn!(error_type, error_message, "Anthropic streaming error");
        return None;
    }
    tracing::debug!(event_type, "unknown Anthropic SSE event type");
    None
}

fn parse_content_block_delta(perf_ns: u64, object: &Map<String, Value>) -> Option<ParsedResponse> {
    let delta = object.get("delta").and_then(Value::as_object)?;
    let delta_type = delta.get("type").and_then(Value::as_str)?;
    let data = if normalized_eq(delta_type, TEXT_DELTA) {
        delta
            .get("text")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(|text| ResponseData::Text { text: text.into() })
    } else if normalized_eq(delta_type, THINKING_DELTA) {
        delta
            .get("thinking")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(|reasoning| ResponseData::Reasoning {
                content: None,
                reasoning: reasoning.into(),
            })
    } else if normalized_eq(delta_type, INPUT_JSON_DELTA) {
        delta
            .get("partial_json")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(|tool_call_text| ResponseData::ToolCall {
                tool_call_text: tool_call_text.into(),
                content: None,
            })
    } else {
        None
    };
    data.map(|data| ParsedResponse {
        perf_ns,
        data: Some(data),
        usage: None,
        sources: None,
    })
}

fn absorb_event(
    object: &Map<String, Value>,
    text_parts: &mut Vec<String>,
    thinking_by_index: &mut BTreeMap<i64, Map<String, Value>>,
    tool_uses_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(event_type) = object.get("type").and_then(Value::as_str) else {
        return;
    };
    if normalized_eq(event_type, MESSAGE) {
        absorb_message(object, text_parts, thinking_by_index, tool_uses_by_index);
    } else if normalized_eq(event_type, CONTENT_BLOCK_START) {
        absorb_content_block_start(object, thinking_by_index, tool_uses_by_index);
    } else if normalized_eq(event_type, CONTENT_BLOCK_DELTA) {
        absorb_content_block_delta(object, text_parts, thinking_by_index, tool_uses_by_index);
    }
}

fn absorb_message(
    object: &Map<String, Value>,
    text_parts: &mut Vec<String>,
    thinking_by_index: &mut BTreeMap<i64, Map<String, Value>>,
    tool_uses_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(blocks) = object.get("content").and_then(Value::as_array) else {
        return;
    };
    for block in blocks.iter().filter_map(Value::as_object) {
        let Some(block_type) = block.get("type").and_then(Value::as_str) else {
            continue;
        };
        if normalized_eq(block_type, TEXT) {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                text_parts.push(text.into());
            }
        } else if normalized_eq(block_type, THINKING) {
            let index = thinking_by_index.len() as i64;
            thinking_by_index.insert(index, without_type(block));
        } else if normalized_eq(block_type, TOOL_USE) {
            let index = tool_uses_by_index.len() as i64;
            tool_uses_by_index.insert(index, without_type(block));
        }
    }
}

fn absorb_content_block_start(
    object: &Map<String, Value>,
    thinking_by_index: &mut BTreeMap<i64, Map<String, Value>>,
    tool_uses_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(block) = object.get("content_block").and_then(Value::as_object) else {
        return;
    };
    let Some(block_type) = block.get("type").and_then(Value::as_str) else {
        return;
    };
    if normalized_eq(block_type, THINKING) {
        let index = object
            .get("index")
            .and_then(Value::as_i64)
            .unwrap_or(thinking_by_index.len() as i64);
        let mut accumulator = without_type(block);
        accumulator
            .entry("thinking")
            .or_insert_with(|| Value::String(String::new()));
        accumulator
            .entry("signature")
            .or_insert_with(|| Value::String(String::new()));
        thinking_by_index.insert(index, accumulator);
    } else if normalized_eq(block_type, TOOL_USE) {
        let index = object
            .get("index")
            .and_then(Value::as_i64)
            .unwrap_or(tool_uses_by_index.len() as i64);
        let mut accumulator = without_type(block);
        accumulator.insert("_input_json".into(), Value::String(String::new()));
        tool_uses_by_index.insert(index, accumulator);
    }
}

fn absorb_content_block_delta(
    object: &Map<String, Value>,
    text_parts: &mut Vec<String>,
    thinking_by_index: &mut BTreeMap<i64, Map<String, Value>>,
    tool_uses_by_index: &mut BTreeMap<i64, Map<String, Value>>,
) {
    let Some(delta) = object.get("delta").and_then(Value::as_object) else {
        return;
    };
    let Some(delta_type) = delta.get("type").and_then(Value::as_str) else {
        return;
    };
    if normalized_eq(delta_type, TEXT_DELTA) {
        if let Some(text) = delta.get("text").and_then(Value::as_str) {
            text_parts.push(text.into());
        }
        return;
    }
    let Some(index) = object.get("index").and_then(Value::as_i64) else {
        return;
    };
    if normalized_eq(delta_type, THINKING_DELTA) {
        append_to_indexed(
            thinking_by_index,
            index,
            "thinking",
            delta.get("thinking").and_then(Value::as_str),
        );
    } else if normalized_eq(delta_type, SIGNATURE_DELTA) {
        append_to_indexed(
            thinking_by_index,
            index,
            "signature",
            delta.get("signature").and_then(Value::as_str),
        );
    } else if normalized_eq(delta_type, INPUT_JSON_DELTA) {
        append_to_indexed(
            tool_uses_by_index,
            index,
            "_input_json",
            Some(
                delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ),
        );
    }
}

fn append_to_indexed(
    blocks: &mut BTreeMap<i64, Map<String, Value>>,
    index: i64,
    field: &str,
    fragment: Option<&str>,
) {
    let (Some(block), Some(fragment)) = (blocks.get_mut(&index), fragment) else {
        return;
    };
    let mut combined = block
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    combined.push_str(fragment);
    block.insert(field.into(), Value::String(combined));
}

fn finalize_thinking(accumulator: Map<String, Value>) -> Value {
    let mut block = Map::new();
    block.insert("type".into(), Value::String(THINKING.into()));
    block.extend(
        accumulator
            .into_iter()
            .filter(|(_, value)| !value.is_null()),
    );
    Value::Object(block)
}

fn finalize_tool_use(mut accumulator: Map<String, Value>) -> Value {
    if let Some(raw) = accumulator.remove("_input_json") {
        let raw = raw.as_str().unwrap_or("");
        if raw.is_empty() {
            if !accumulator.contains_key("input") {
                accumulator.insert("input".into(), Value::Object(Map::new()));
            }
        } else {
            accumulator.insert(
                "input".into(),
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into())),
            );
        }
    }
    let mut block = Map::new();
    block.insert("type".into(), Value::String(TOOL_USE.into()));
    block.extend(
        accumulator
            .into_iter()
            .filter(|(_, value)| !value.is_null()),
    );
    Value::Object(block)
}

fn without_type(block: &Map<String, Value>) -> Map<String, Value> {
    block
        .iter()
        .filter(|(key, _)| key.as_str() != "type")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn walk_system(payload: &Map<String, Value>, result: &mut ExtractedPayload) {
    let Some(system) = payload.get("system") else {
        return;
    };
    if let Some(system) = system.as_str() {
        if !system.is_empty() {
            result.texts.insert(0, system.into());
        }
        return;
    }
    let Some(parts) = system.as_array() else {
        return;
    };
    let mut collected = Vec::new();
    for part in parts {
        if let Some(text) = part.as_str().filter(|value| !value.is_empty()) {
            collected.push(text.to_string());
        } else if let Some(part) = part.as_object()
            && part
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|value| normalized_eq(value, TEXT))
            && let Some(text) = part
                .get("text")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
        {
            collected.push(text.to_string());
        }
    }
    for text in collected.into_iter().rev() {
        result.texts.insert(0, text);
    }
}

fn walk_tool_schemas(payload: &Map<String, Value>, result: &mut ExtractedPayload) {
    let Some(tools) = payload.get("tools").and_then(Value::as_array) else {
        return;
    };
    for tool in tools.iter().filter_map(Value::as_object) {
        if let Some(schema) = tool.get("input_schema").and_then(Value::as_object)
            && let Ok(serialized) = serde_json::to_string(schema)
        {
            result.texts.push(serialized);
        }
    }
}

fn walk_tool_blocks(payload: &Map<String, Value>, result: &mut ExtractedPayload) {
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return;
    };
    for message in messages.iter().filter_map(Value::as_object) {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in content.iter().filter_map(Value::as_object) {
            let Some(part_type) = part.get("type").and_then(Value::as_str) else {
                continue;
            };
            if normalized_eq(part_type, TOOL_USE) {
                collect_tool_use(part, result);
            } else if normalized_eq(part_type, TOOL_RESULT) {
                collect_tool_result(part, result);
            }
        }
    }
}

fn collect_tool_use(part: &Map<String, Value>, result: &mut ExtractedPayload) {
    if let Some(name) = part
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        result.texts.push(name.into());
    }
    if let Some(input) = part.get("input").and_then(Value::as_object)
        && let Ok(serialized) = serde_json::to_string(input)
    {
        result.texts.push(serialized);
    }
}

fn collect_tool_result(part: &Map<String, Value>, result: &mut ExtractedPayload) {
    let Some(content) = part.get("content") else {
        return;
    };
    if let Some(content) = content.as_str() {
        if !content.is_empty() {
            result.texts.push(content.into());
        }
        return;
    }
    let Some(blocks) = content.as_array() else {
        return;
    };
    for block in blocks.iter().filter_map(Value::as_object) {
        if block
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|value| normalized_eq(value, TEXT))
            && let Some(text) = block
                .get("text")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
        {
            result.texts.push(text.into());
        }
    }
}

fn non_empty_concat(parts: Vec<String>) -> Option<String> {
    let combined = parts.concat();
    (!combined.is_empty()).then_some(combined)
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
