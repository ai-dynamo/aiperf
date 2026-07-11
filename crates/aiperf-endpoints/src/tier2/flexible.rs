// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw and Jinja-compatible template endpoint implementations.
//!
//! This ports `src/aiperf/endpoints/raw_endpoint.py:12-49`,
//! `src/aiperf/endpoints/response_mixin.py:9-113`, and
//! `src/aiperf/endpoints/template_endpoint.py:17-194`. JMESPath compilation is
//! deliberately failure-soft for raw responses; an explicitly valid template
//! selector that does not match is a hard parse miss, matching Python.

use std::path::Path;

use minijinja::{AutoEscape, Environment};
use serde_json::{Map, Value, json};

use super::effective_model;
use crate::config::EndpointConfig;
use crate::endpoints::{Endpoint, number_array, try_extract_embeddings};
use crate::metadata::{EndpointMetadata, EndpointType, metadata_for};
use crate::models::{
    EndpointError, EndpointResult, Media, ParsedResponse, RequestInfo, ResponseData, ServerResponse,
};

const NV_EMBEDQA: &str = r#"{"text": {{ texts|tojson }}}"#;

/// Raw request passthrough with JMESPath-first response auto-detection.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawEndpoint;

/// Jinja-compatible JSON request template with optional JMESPath response selection.
#[derive(Debug, Clone, Copy, Default)]
pub struct TemplateEndpoint;

impl Endpoint for RawEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Raw)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        request_info
            .turns
            .last()
            .and_then(|turn| turn.raw_payload.clone())
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "raw endpoint requires raw_payload on the dispatching turn".into(),
                )
            })
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_flexible_response(response, None, MissingSelectorPolicy::Fallback)
    }

    fn parse_response_with_config(
        &self,
        response: &ServerResponse,
        config: &EndpointConfig,
    ) -> EndpointResult<Option<ParsedResponse>> {
        parse_flexible_response(
            response,
            response_field(config),
            MissingSelectorPolicy::Fallback,
        )
    }
}

impl Endpoint for TemplateEndpoint {
    fn metadata(&self) -> &'static EndpointMetadata {
        metadata_for(EndpointType::Template)
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<Value> {
        let turn = request_info.turns.last().ok_or_else(|| {
            EndpointError::InvalidRequest("Template endpoint requires at least one turn".into())
        })?;
        let config = &request_info.model_endpoint.endpoint;
        let (source, legacy_extra_config) = template_source(config)?;
        let source = resolve_template_source(source)?;

        let (texts, texts_by_name) = named_contents(&turn.texts);
        let (images, images_by_name) = named_contents(&turn.images);
        let (audios, audios_by_name) = named_contents(&turn.audios);
        let (videos, videos_by_name) = named_contents(&turn.videos);
        let queries = texts_by_name
            .get("query")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let passages = texts_by_name
            .get("passages")
            .or_else(|| texts_by_name.get("passage"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut variables = Map::new();
        variables.insert("texts".into(), strings_value(&texts));
        variables.insert("images".into(), strings_value(&images));
        variables.insert("audios".into(), strings_value(&audios));
        variables.insert("videos".into(), strings_value(&videos));
        variables.insert("text".into(), first_or_null(&texts));
        variables.insert("image".into(), first_or_null(&images));
        variables.insert("audio".into(), first_or_null(&audios));
        variables.insert("video".into(), first_or_null(&videos));
        variables.insert("queries".into(), Value::Array(queries.clone()));
        variables.insert("passages".into(), Value::Array(passages.clone()));
        variables.insert(
            "query".into(),
            queries.first().cloned().unwrap_or(Value::Null),
        );
        variables.insert(
            "passage".into(),
            passages.first().cloned().unwrap_or(Value::Null),
        );
        variables.insert("texts_by_name".into(), Value::Object(texts_by_name));
        variables.insert("images_by_name".into(), Value::Object(images_by_name));
        variables.insert("audios_by_name".into(), Value::Object(audios_by_name));
        variables.insert("videos_by_name".into(), Value::Object(videos_by_name));
        variables.insert(
            "model".into(),
            Value::String(effective_model(request_info, turn)),
        );
        variables.insert(
            "max_tokens".into(),
            turn.max_tokens.map_or(Value::Null, |tokens| json!(tokens)),
        );
        variables.insert(
            "role".into(),
            turn.role.clone().map_or(Value::Null, Value::String),
        );
        variables.insert(
            "turn".into(),
            serde_json::to_value(turn).map_err(|error| {
                EndpointError::InvalidRequest(format!("serialize template turn: {error}"))
            })?,
        );
        variables.insert(
            "turns".into(),
            serde_json::to_value(&request_info.turns).map_err(|error| {
                EndpointError::InvalidRequest(format!("serialize template turns: {error}"))
            })?,
        );
        variables.insert(
            "request_info".into(),
            serde_json::to_value(request_info).map_err(|error| {
                EndpointError::InvalidRequest(format!("serialize template request: {error}"))
            })?,
        );
        variables.insert("stream".into(), Value::Bool(config.streaming));

        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::Html);
        let rendered = environment
            .render_str(&source, Value::Object(variables))
            .map_err(|error| {
                EndpointError::InvalidRequest(format!("render payload template: {error}"))
            })?;
        let value = serde_json::from_str::<Value>(&rendered).map_err(|error| {
            EndpointError::InvalidRequest(format!(
                "template did not render valid JSON {error}: {}",
                rendered.chars().take(100).collect::<String>()
            ))
        })?;
        let mut payload = value.as_object().cloned().ok_or_else(|| {
            EndpointError::InvalidRequest("template must render a JSON object".into())
        })?;
        if let Some(extra) = config.extra.as_ref() {
            for (key, value) in extra {
                if legacy_extra_config
                    && matches!(key.as_str(), "payload_template" | "response_field")
                {
                    continue;
                }
                payload.insert(key.clone(), value.clone());
            }
        }
        if let Some(extra) = turn.extra_body.as_ref() {
            for (key, value) in extra {
                payload.insert(key.clone(), value.clone());
            }
        }
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_flexible_response(response, None, MissingSelectorPolicy::Fail)
    }

    fn parse_response_with_config(
        &self,
        response: &ServerResponse,
        config: &EndpointConfig,
    ) -> EndpointResult<Option<ParsedResponse>> {
        let field = response_field(config).filter(|field| *field != "text");
        parse_flexible_response(response, field, MissingSelectorPolicy::Fail)
    }
}

fn template_source(config: &EndpointConfig) -> EndpointResult<(&str, bool)> {
    if let Some(template) = config.template.as_deref() {
        return Ok((template, false));
    }
    config
        .extra
        .as_ref()
        .and_then(|extra| extra.get("payload_template"))
        .and_then(Value::as_str)
        .map(|template| (template, true))
        .ok_or_else(|| {
            EndpointError::InvalidConfig(
                "template endpoint requires endpoint.template or endpoint.extra.payload_template"
                    .into(),
            )
        })
}

fn resolve_template_source(source: &str) -> EndpointResult<String> {
    if source == "nv-embedqa" {
        return Ok(NV_EMBEDQA.to_string());
    }
    let path = Path::new(source);
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return Ok(source.to_string());
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Ok(source.to_string());
    }
    std::fs::read_to_string(path).map_err(|error| {
        EndpointError::InvalidConfig(format!(
            "failed to read payload template {}: {error}",
            path.display()
        ))
    })
}

fn named_contents(items: &[Media]) -> (Vec<String>, Map<String, Value>) {
    let mut all = Vec::new();
    let mut named = Map::new();
    for item in items {
        if item.contents.is_empty() {
            continue;
        }
        all.extend(item.contents.iter().cloned());
        if item.name.is_empty() {
            continue;
        }
        let entry = named
            .entry(item.name.clone())
            .or_insert_with(|| Value::Array(Vec::new()));
        entry
            .as_array_mut()
            .expect("named media entries are arrays")
            .extend(item.contents.iter().cloned().map(Value::String));
    }
    (all, named)
}

fn strings_value(values: &[String]) -> Value {
    Value::Array(values.iter().cloned().map(Value::String).collect())
}

fn first_or_null(values: &[String]) -> Value {
    values.first().cloned().map_or(Value::Null, Value::String)
}

fn response_field(config: &EndpointConfig) -> Option<&str> {
    config.response_field.as_deref().or_else(|| {
        config
            .extra
            .as_ref()
            .and_then(|extra| extra.get("response_field"))
            .and_then(Value::as_str)
    })
}

#[derive(Debug, Clone, Copy)]
enum MissingSelectorPolicy {
    Fallback,
    Fail,
}

enum SelectorResult {
    Invalid,
    NoMatch,
    Match(Value),
}

fn parse_flexible_response(
    response: &ServerResponse,
    selector: Option<&str>,
    missing_policy: MissingSelectorPolicy,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(json) = response.json.as_ref() else {
        return Ok(response
            .raw
            .as_ref()
            .filter(|text| !text.is_empty())
            .map(|text| parsed(response.perf_ns, ResponseData::Text { text: text.clone() })));
    };

    let data = if let Some(selector) = selector {
        match select_json(selector, json) {
            SelectorResult::Match(value) => convert_to_response_data(&value),
            SelectorResult::NoMatch if matches!(missing_policy, MissingSelectorPolicy::Fail) => {
                return Ok(None);
            }
            SelectorResult::Invalid | SelectorResult::NoMatch => auto_detect(json),
        }
    } else {
        auto_detect(json)
    };
    Ok(data.map(|data| parsed(response.perf_ns, data)))
}

fn select_json(selector: &str, json: &Value) -> SelectorResult {
    let Ok(expression) = jmespath::compile(selector) else {
        return SelectorResult::Invalid;
    };
    let Ok(result) = expression.search(json) else {
        return SelectorResult::Invalid;
    };
    let Ok(value) = serde_json::to_value(result.as_ref()) else {
        return SelectorResult::Invalid;
    };
    if is_truthy(&value) {
        SelectorResult::Match(value)
    } else {
        SelectorResult::NoMatch
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn auto_detect(value: &Value) -> Option<ResponseData> {
    let object = value.as_object()?;
    if let Some(embeddings) = try_extract_embeddings(object) {
        return Some(ResponseData::Embeddings { embeddings });
    }
    for field in ["rankings", "results"] {
        if let Some(rankings) = object
            .get(field)
            .and_then(Value::as_array)
            .filter(|rankings| !rankings.is_empty())
        {
            return Some(ResponseData::Rankings {
                rankings: rankings.clone(),
            });
        }
    }
    try_extract_text(object)
}

fn try_extract_text(object: &Map<String, Value>) -> Option<ResponseData> {
    for field in ["text", "content", "response", "output", "result"] {
        match object.get(field) {
            Some(Value::String(text)) => {
                return (!text.is_empty()).then(|| ResponseData::Text { text: text.clone() });
            }
            Some(Value::Array(values))
                if !values.is_empty() && values.iter().all(Value::is_string) =>
            {
                let text = values.iter().filter_map(Value::as_str).collect::<String>();
                return (!text.is_empty()).then_some(ResponseData::Text { text });
            }
            _ => {}
        }
    }
    let choice = object
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)?;
    let text = choice
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| {
            choice
                .get("message")
                .and_then(Value::as_object)
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            choice
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
        })?;
    (!text.is_empty()).then(|| ResponseData::Text {
        text: text.to_string(),
    })
}

fn convert_to_response_data(value: &Value) -> Option<ResponseData> {
    if let Some(text) = value.as_str().filter(|text| !text.is_empty()) {
        return Some(ResponseData::Text {
            text: text.to_string(),
        });
    }
    if let Some(numbers) = number_array(value) {
        return Some(ResponseData::Embeddings {
            embeddings: vec![numbers],
        });
    }
    let array = value.as_array()?;
    if array.is_empty() {
        return None;
    }
    if array.iter().all(Value::is_object) {
        return Some(ResponseData::Rankings {
            rankings: array.clone(),
        });
    }
    let embeddings = array.iter().map(number_array).collect::<Option<Vec<_>>>()?;
    (!embeddings.is_empty()).then_some(ResponseData::Embeddings { embeddings })
}

fn parsed(perf_ns: u64, data: ResponseData) -> ParsedResponse {
    ParsedResponse {
        perf_ns,
        data: Some(data),
        usage: None,
        sources: None,
    }
}
