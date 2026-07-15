// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw and Jinja-compatible template endpoint implementations.
//!
//! JMESPath compilation is deliberately failure-soft for raw responses; an
//! explicitly valid template selector that does not match is a hard parse miss,
//! matching Python.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minijinja::{AutoEscape, Environment};
use serde_json::{Map, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig};
use crate::endpoints::endpoints::{Endpoint, number_array, try_extract_embeddings};
use crate::endpoints::metadata::{EndpointDescriptor, EndpointType, Modality};
use crate::endpoints::models::{
    EndpointError, EndpointResult, ExtractedPayload, Media, ParsedResponse, RequestInfo,
    RequestRecord, ResponseData, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedEndpointBehavior, PreparedRequest, ReadinessPolicy,
    format_legacy_payload,
};

const NV_EMBEDQA: &str = r#"{"text": {{ texts|tojson }}}"#;

/// Raw request passthrough with JMESPath-first response auto-detection.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawEndpoint;

/// Jinja-compatible JSON request template with optional JMESPath response selection.
#[derive(Debug, Clone, Copy, Default)]
pub struct TemplateEndpoint;

/// Startup factory that compiles a raw endpoint's response selector once per worker/profile.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawEndpointFactory;

/// Startup factory that compiles template and response selector state once per worker/profile.
#[derive(Debug, Clone, Copy, Default)]
pub struct TemplateEndpointFactory;

const RAW_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "raw",
    aliases: &[],
    description: "Raw JSON passthrough endpoint",
    endpoint_path: None,
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[
        Modality::Text,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
    ],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

const TEMPLATE_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "template",
    aliases: &[],
    description: "Minijinja JSON-template endpoint",
    endpoint_path: None,
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[
        Modality::Text,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
    ],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

impl Endpoint for RawEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &RAW_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
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
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &TEMPLATE_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
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

impl PreparedEndpointBehavior for RawEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        _config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let payload = request
            .turns()
            .last()
            .and_then(|turn| turn.raw_payload.clone())
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "raw endpoint requires raw_payload on the dispatching turn".into(),
                )
            })?;
        // The raw body may be any authored JSON value; the plan model only splices
        // named-field objects, so a non-object body is the hard error the legacy
        // `structured_plan` bridge produced (moved here off the dispatch path).
        let object = payload.as_object().ok_or_else(|| {
            EndpointError::InvalidRequest("endpoint body must be a JSON object".into())
        })?;
        Ok(BodyPlan::from_object(object)?)
    }
}

impl PreparedEndpointBehavior for TemplateEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let (source, legacy_extra_config) = template_source(config)?;
        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::Html);
        environment
            .add_template_owned("payload", resolve_template_source(source))
            .map_err(|error| {
                EndpointError::InvalidConfig(format!("compile payload template: {error}"))
            })?;
        render_template_payload(&environment, request, config, legacy_extra_config)
    }
}

impl EndpointFactory for RawEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &RAW_DESCRIPTOR
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let selector = PreparedSelector::compile(response_field_raw(config.as_raw()));
        let legacy_config = EndpointConfig::from_raw(EndpointType::Raw, config.to_raw());
        let headers = RawEndpoint.format_headers(&legacy_config);
        Ok(Box::new(PreparedRawEndpoint {
            config,
            legacy_config,
            headers,
            selector,
        }))
    }

    fn legacy_endpoint(&self) -> Option<Arc<dyn Endpoint>> {
        Some(Arc::new(RawEndpoint))
    }
}

impl EndpointFactory for TemplateEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &TEMPLATE_DESCRIPTOR
    }

    fn validate_config(&self, config: &mut RawEndpointConfig) -> EndpointResult<()> {
        template_source(config).map(|_| ())
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let (source, legacy_extra_config) = template_source(config.as_raw())?;
        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::Html);
        environment
            .add_template_owned("payload", resolve_template_source(source))
            .map_err(|error| {
                EndpointError::InvalidConfig(format!("compile payload template: {error}"))
            })?;
        let selector = PreparedSelector::compile(
            response_field_raw(config.as_raw()).filter(|field| *field != "text"),
        );
        let legacy_config = EndpointConfig::from_raw(EndpointType::Template, config.to_raw());
        let headers = TemplateEndpoint.format_headers(&legacy_config);
        Ok(Box::new(PreparedTemplateEndpoint {
            config,
            legacy_config,
            headers,
            environment,
            selector,
            legacy_extra_config,
        }))
    }

    fn legacy_endpoint(&self) -> Option<Arc<dyn Endpoint>> {
        Some(Arc::new(TemplateEndpoint))
    }
}

#[derive(Debug)]
struct PreparedRawEndpoint {
    config: EffectiveEndpointConfig,
    legacy_config: EndpointConfig,
    headers: BTreeMap<String, String>,
    selector: PreparedSelector,
}

impl PreparedEndpoint for PreparedRawEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &RAW_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        RawEndpoint.format_prepared_payload(request, self.config.as_raw())
    }

    fn precomputable_body(&self) -> bool {
        // Raw passthrough splices the dispatching turn's authored `raw_payload`;
        // its body is not derivable from static-context turns at bind.
        false
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        RawEndpoint.readiness_policy(&self.legacy_config, model)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_flexible_response_prepared(response, &self.selector, MissingSelectorPolicy::Fallback)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        RawEndpoint.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        extract_prepared_responses(self, record)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        RawEndpoint.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        RawEndpoint.captures_assistant_turn()
    }
}

#[derive(Debug)]
struct PreparedTemplateEndpoint {
    config: EffectiveEndpointConfig,
    legacy_config: EndpointConfig,
    headers: BTreeMap<String, String>,
    environment: Environment<'static>,
    selector: PreparedSelector,
    legacy_extra_config: bool,
}

impl PreparedEndpoint for PreparedTemplateEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &TEMPLATE_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        render_template_payload(
            &self.environment,
            request,
            self.config.as_raw(),
            self.legacy_extra_config,
        )
    }

    fn precomputable_body(&self) -> bool {
        // The Jinja template can reference per-dispatch identity (`x_request_id`,
        // `x_correlation_id`) the cache does not capture, so it must render live.
        false
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        TemplateEndpoint.readiness_policy(&self.legacy_config, model)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_flexible_response_prepared(response, &self.selector, MissingSelectorPolicy::Fail)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        TemplateEndpoint.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        extract_prepared_responses(self, record)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        TemplateEndpoint.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        TemplateEndpoint.captures_assistant_turn()
    }
}

fn extract_prepared_responses(
    endpoint: &dyn PreparedEndpoint,
    record: &RequestRecord,
) -> EndpointResult<Vec<ParsedResponse>> {
    let mut parsed = Vec::new();
    for response in &record.responses {
        if let Some(response) = endpoint.parse_response(response)? {
            parsed.push(response);
        }
    }
    Ok(parsed)
}

/// Lower a `serde_json::Value` into a `minijinja::Value` without routing numbers
/// through serde.
///
/// This directly maps each JSON number to a native integer or float
/// `minijinja::Value`, so template output stays valid JSON regardless of how
/// serde_json is configured. It also guards against the `arbitrary_precision`
/// feature, which the workspace no longer enables: under that feature a
/// `serde_json::Number` serializes through a private `$serde_json::private::Number`
/// marker that only serde_json's own deserializer understands, and handing such a
/// value straight to minijinja's serializer (as `template.render` does) would
/// render the marker verbatim — e.g. `{{ max_tokens }}` emitting
/// `{"$serde_json::private::Number":"12"}`.
fn json_to_minijinja(value: &Value) -> minijinja::Value {
    match value {
        Value::Null => minijinja::Value::from(()),
        Value::Bool(flag) => minijinja::Value::from(*flag),
        Value::Number(number) => {
            if let Some(signed) = number.as_i64() {
                minijinja::Value::from(signed)
            } else if let Some(unsigned) = number.as_u64() {
                minijinja::Value::from(unsigned)
            } else if let Some(float) = number.as_f64() {
                minijinja::Value::from(float)
            } else {
                minijinja::Value::from(number.to_string())
            }
        }
        Value::String(text) => minijinja::Value::from(text.clone()),
        Value::Array(items) => items.iter().map(json_to_minijinja).collect(),
        Value::Object(entries) => entries
            .iter()
            .map(|(key, value)| (key.clone(), json_to_minijinja(value)))
            .collect(),
    }
}

fn render_template_payload(
    environment: &Environment<'_>,
    request: &PreparedRequest<'_>,
    config: &RawEndpointConfig,
    legacy_extra_config: bool,
) -> EndpointResult<BodyPlan> {
    let turn = request.turns().last().ok_or_else(|| {
        EndpointError::InvalidRequest("Template endpoint requires at least one turn".into())
    })?;
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
        Value::String(
            turn.model
                .clone()
                .unwrap_or_else(|| request.primary_model_name().to_string()),
        ),
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
        serde_json::to_value(request.turns()).map_err(|error| {
            EndpointError::InvalidRequest(format!("serialize template turns: {error}"))
        })?,
    );
    variables.insert(
        "request_info".into(),
        prepared_request_value(request, config),
    );
    variables.insert("stream".into(), Value::Bool(config.streaming));

    let rendered = environment
        .get_template("payload")
        .and_then(|template| template.render(json_to_minijinja(&Value::Object(variables))))
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
            if legacy_extra_config && matches!(key.as_str(), "payload_template" | "response_field")
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
    Ok(BodyPlan::from_object(&payload)?)
}

fn prepared_request_value(request: &PreparedRequest<'_>, config: &RawEndpointConfig) -> Value {
    json!({
        "model_endpoint": {
            "primary_model_name": request.primary_model_name(),
            "endpoint": config,
        },
        "turns": request.turns(),
        "system_message": request.system_message(),
        "user_context_message": request.user_context_message(),
        "credit_phase": request.credit_phase(),
        "x_request_id": request.x_request_id(),
        "x_correlation_id": request.x_correlation_id(),
        "conversation_id": request.conversation_id(),
    })
}

fn template_source(config: &RawEndpointConfig) -> EndpointResult<(&str, bool)> {
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

fn resolve_template_source(source: &str) -> String {
    if source == "nv-embedqa" {
        return NV_EMBEDQA.to_string();
    }
    let path = expanded_template_path(source);
    if has_symlink_component(&path) {
        return source.to_string();
    }
    let Ok(resolved) = std::fs::canonicalize(&path) else {
        return source.to_string();
    };
    if !resolved.is_file() {
        return source.to_string();
    }
    std::fs::read_to_string(resolved).unwrap_or_else(|_| source.to_string())
}

/// Home directory for `~` expansion: `HOME` (Unix) with a `USERPROFILE`
/// fallback so the tilde still resolves on Windows, where `HOME` is unset.
fn home_dir_os() -> Option<std::ffi::OsString> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
}

fn expanded_template_path(source: &str) -> PathBuf {
    if source == "~"
        && let Some(home) = home_dir_os()
    {
        return PathBuf::from(home);
    }
    if let Some(relative) = source.strip_prefix("~/")
        && let Some(home) = home_dir_os()
    {
        return PathBuf::from(home).join(relative);
    }
    Path::new(source).to_path_buf()
}

fn has_symlink_component(path: &Path) -> bool {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => return true,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    false
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

fn response_field_raw(config: &RawEndpointConfig) -> Option<&str> {
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

#[derive(Debug)]
enum PreparedSelector {
    Absent,
    Invalid,
    Compiled(jmespath::Expression<'static>),
}

impl PreparedSelector {
    fn compile(selector: Option<&str>) -> Self {
        match selector {
            None => Self::Absent,
            Some(selector) => jmespath::compile(selector)
                .map(Self::Compiled)
                .unwrap_or(Self::Invalid),
        }
    }

    fn select(&self, json: &Value) -> SelectorResult {
        match self {
            Self::Absent => SelectorResult::Invalid,
            Self::Invalid => SelectorResult::Invalid,
            Self::Compiled(expression) => {
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
        }
    }
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

fn parse_flexible_response_prepared(
    response: &ServerResponse,
    selector: &PreparedSelector,
    missing_policy: MissingSelectorPolicy,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(json) = response.json.as_ref() else {
        return Ok(response
            .raw
            .as_ref()
            .filter(|text| !text.is_empty())
            .map(|text| parsed(response.perf_ns, ResponseData::Text { text: text.clone() })));
    };

    let data = match selector {
        PreparedSelector::Absent => auto_detect(json),
        _ => match selector.select(json) {
            SelectorResult::Match(value) => convert_to_response_data(&value),
            SelectorResult::NoMatch if matches!(missing_policy, MissingSelectorPolicy::Fail) => {
                return Ok(None);
            }
            SelectorResult::Invalid | SelectorResult::NoMatch => auto_detect(json),
        },
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
