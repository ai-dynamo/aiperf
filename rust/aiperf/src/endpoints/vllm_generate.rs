// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-native vLLM/Dynamo `/inference/v1/generate` endpoint.
//!
//! The Rust port
//! deliberately tightens the boundary: dataset composition validates and owns
//! the raw token IDs, so formatting maps a typed [`crate::endpoints::Turn`] field to the
//! vLLM wire field without inspecting arbitrary JSON values.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, RawEndpointConfig};
use crate::endpoints::metadata::{EndpointDescriptor, Modality};
use crate::endpoints::models::{
    EndpointError, EndpointResult, ExtractedPayload, ParsedResponse, RequestRecord, ResponseData,
    ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedReadinessRequest, PreparedRequest, ReadinessMethod,
    ReadinessPolicy, ReadinessSuccess,
};

const VLLM_GENERATE_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "vllm_generate",
    aliases: &["vllm_token_generate"],
    description: "vLLM/Dynamo token-in token-out Generate API",
    endpoint_path: Some("/inference/v1/generate"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: true,
    tokenizes_input: false,
    requires_raw_token_ids: true,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Tokens],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "llm",
};

/// Protocol-v2-only factory for vLLM's token-native Generate API.
#[derive(Clone, Copy, Debug, Default)]
pub struct VllmGenerateFactory;

impl EndpointFactory for VllmGenerateFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &VLLM_GENERATE_DESCRIPTOR
    }

    fn validate_config(&self, config: &mut RawEndpointConfig) -> EndpointResult<()> {
        if config.template.is_some() || config.response_field.is_some() {
            return Err(EndpointError::InvalidConfig(
                "vllm_generate does not accept template or response_field configuration".into(),
            ));
        }
        validate_protected_extras(config.extra.as_ref(), "endpoint.extra")
            .and_then(|()| {
                let mut extra = config.extra.clone().unwrap_or_default();
                take_sampling_params(&mut extra, "endpoint.extra").map(|_| ())
            })
            .map_err(|error| match error {
                EndpointError::InvalidRequest(message) => EndpointError::InvalidConfig(message),
                other => other,
            })
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        let mut headers = config.as_raw().headers.clone();
        if let Some(api_key) = &config.as_raw().api_key {
            headers.insert("Authorization".into(), format!("Bearer {api_key}"));
        }
        // The endpoint `extra`/`sampling_params` split depends only on immutable
        // config, so validate and lift it once here instead of on every dispatch.
        let mut endpoint_extra = config.as_raw().extra.clone().unwrap_or_default();
        validate_protected_extras(Some(&endpoint_extra), "endpoint.extra")?;
        let endpoint_sampling = take_sampling_params(&mut endpoint_extra, "endpoint.extra")?;
        Ok(Box::new(PreparedVllmGenerate {
            config,
            headers,
            endpoint_extra,
            endpoint_sampling,
        }))
    }
}

#[derive(Debug)]
struct PreparedVllmGenerate {
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
    endpoint_extra: Map<String, Value>,
    endpoint_sampling: Map<String, Value>,
}

impl PreparedEndpoint for PreparedVllmGenerate {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &VLLM_GENERATE_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn precomputable_body(&self) -> bool {
        // Token-native composition dispatches exact per-turn raw token IDs; there
        // is no reusable message-array body plan to cache.
        false
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        let [turn] = request.turns() else {
            return Err(EndpointError::InvalidRequest(
                "vllm_generate requires exactly one token payload per dispatch".into(),
            ));
        };
        let token_ids = turn.raw_token_ids.as_ref().ok_or_else(|| {
            EndpointError::InvalidRequest(
                "vllm_generate requires validated Turn.raw_token_ids".into(),
            )
        })?;
        if token_ids.is_empty() {
            return Err(EndpointError::InvalidRequest(
                "vllm_generate requires at least one input token ID".into(),
            ));
        }

        let mut turn_extra = turn.extra_body.clone().unwrap_or_default();
        validate_protected_extras(Some(&turn_extra), "turn.extra_body")?;
        let turn_sampling = take_sampling_params(&mut turn_extra, "turn.extra_body")?;

        let mut sampling_params = self.endpoint_sampling.clone();
        sampling_params.extend(turn_sampling);
        if let Some(max_tokens) = turn.max_tokens {
            sampling_params
                .entry("max_tokens")
                .or_insert_with(|| Value::from(max_tokens));
        }

        let mut payload = self.endpoint_extra.clone();
        payload.extend(turn_extra);
        payload.insert(
            "model".into(),
            Value::String(
                turn.model
                    .clone()
                    .unwrap_or_else(|| request.primary_model_name().to_string()),
            ),
        );
        payload.insert("token_ids".into(), json!(token_ids));
        payload.insert("sampling_params".into(), Value::Object(sampling_params));
        payload.insert("stream".into(), Value::Bool(false));
        if let Some(request_id) = request.x_request_id() {
            payload.insert("request_id".into(), Value::String(request_id.to_string()));
        }
        Ok(BodyPlan::from_object(&payload)?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Request(PreparedReadinessRequest {
            method: ReadinessMethod::Post,
            path: self
                .config
                .as_raw()
                .path
                .clone()
                .unwrap_or_else(|| "/inference/v1/generate".into()),
            headers: self.headers.clone(),
            body: Some(json!({
                "model": model,
                "token_ids": [0],
                "sampling_params": {"max_tokens": 1},
                "stream": false
            })),
            success: ReadinessSuccess::SuccessfulStatus,
        }))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_response(response)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let pretokenised_token_count = body
            .get("token_ids")
            .and_then(Value::as_array)
            .and_then(|ids| u64::try_from(ids.len()).ok())
            .unwrap_or(0);
        ExtractedPayload {
            pretokenised_token_count,
            ..ExtractedPayload::default()
        }
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        record
            .responses
            .iter()
            .filter_map(|response| self.parse_response(response).transpose())
            .collect()
    }

    fn build_assistant_turn(&self, _record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        Ok(None)
    }

    fn captures_assistant_turn(&self) -> bool {
        false
    }
}

fn validate_protected_extras(
    extra: Option<&Map<String, Value>>,
    owner: &str,
) -> EndpointResult<()> {
    let Some(extra) = extra else {
        return Ok(());
    };
    for field in ["token_ids", "model"] {
        if extra.contains_key(field) {
            return Err(EndpointError::InvalidRequest(format!(
                "{owner}.{field} cannot override the typed vllm_generate field"
            )));
        }
    }
    if let Some(stream) = extra.get("stream")
        && stream != &Value::Bool(false)
    {
        return Err(EndpointError::InvalidRequest(format!(
            "{owner}.stream must be false for vllm_generate"
        )));
    }
    Ok(())
}

fn take_sampling_params(
    extra: &mut Map<String, Value>,
    owner: &str,
) -> EndpointResult<Map<String, Value>> {
    match extra.remove("sampling_params") {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(value)) => Ok(value),
        Some(value) => Err(EndpointError::InvalidRequest(format!(
            "{owner}.sampling_params must be an object, got {value}"
        ))),
    }
}

fn parse_response(response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
    let Some(payload) = response.json.as_ref().and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(choice) = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    let Some(raw_ids) = choice.get("token_ids").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut token_ids = Vec::with_capacity(raw_ids.len());
    for value in raw_ids {
        let Some(token_id) = value.as_u64().and_then(|value| u32::try_from(value).ok()) else {
            return Ok(None);
        };
        token_ids.push(token_id);
    }

    let completion_tokens = u64::try_from(token_ids.len())
        .map_err(|_| EndpointError::InvalidResponse("completion token count exceeds u64".into()))?;
    let mut usage = payload
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    usage.insert("completion_tokens".into(), Value::from(completion_tokens));
    if let Some(prompt_tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
        usage.insert(
            "total_tokens".into(),
            Value::from(prompt_tokens.saturating_add(completion_tokens)),
        );
    }

    Ok(Some(ParsedResponse {
        perf_ns: response.perf_ns,
        data: Some(ResponseData::TokenIds { token_ids }),
        usage: Some(Value::Object(usage)),
        sources: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_retains_exact_output_ids_and_reconstructs_completion_usage() {
        let response = ServerResponse::from_json(
            123,
            json!({
                "request_id": "req-1",
                "choices": [{"index": 0, "token_ids": [20, 21], "finish_reason": "stop"}]
            }),
        );

        let parsed = parse_response(&response).unwrap().unwrap();
        assert_eq!(
            parsed.data,
            Some(ResponseData::TokenIds {
                token_ids: vec![20, 21]
            })
        );
        assert_eq!(
            parsed
                .usage
                .as_ref()
                .and_then(|usage| usage.get("completion_tokens")),
            Some(&Value::from(2))
        );
    }

    #[test]
    fn parser_degrades_to_none_for_non_u32_output_ids() {
        let response = ServerResponse::from_json(
            123,
            json!({"choices": [{"token_ids": [-1, true, 4_294_967_296_u64]}]}),
        );
        assert_eq!(parse_response(&response).unwrap(), None);
    }
}
