// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-protocol-v2-only KServe endpoint factories.
//!
//! These factories deliberately omit the protocol-v1 [`crate::endpoints::Endpoint`]
//! compatibility hook. They are therefore selectable only through the open
//! endpoint registry and prepared-profile path used by runner protocol v2.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Number, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, EndpointConfig, RawEndpointConfig};
use crate::endpoints::endpoints::{ChatEndpoint, CompletionsEndpoint, EmbeddingsEndpoint};
use crate::endpoints::metadata::{EndpointDescriptor, EndpointType, Modality};
use crate::endpoints::models::{
    EndpointError, EndpointResult, ExtractedPayload, ImageDataItem, ImageResponseData,
    ParsedResponse, RequestRecord, ResponseData, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedEndpointBehavior, PreparedReadinessRequest,
    PreparedRequest, ReadinessMethod, ReadinessPolicy, ReadinessSuccess,
};

const KSERVE_CHAT_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_chat",
    aliases: &[],
    description: "KServe OpenAI-compatible Chat Completions API",
    endpoint_path: Some("/openai/v1/chat/completions"),
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
    service_kind: "kserve",
};

const KSERVE_COMPLETIONS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_completions",
    aliases: &[],
    description: "KServe OpenAI-compatible Completions API",
    endpoint_path: Some("/openai/v1/completions"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "LLM Metrics",
    service_kind: "kserve",
};

const KSERVE_EMBEDDINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_embeddings",
    aliases: &[],
    description: "KServe OpenAI-compatible Embeddings API",
    endpoint_path: Some("/openai/v1/embeddings"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Embeddings],
    metrics_title: "Embeddings Metrics",
    service_kind: "kserve",
};

const KSERVE_V1_PREDICT_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v1_predict",
    aliases: &[],
    description: "KServe V1 instances/predictions inference protocol",
    endpoint_path: Some("/v1/models/{model_name}:predict"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens, Modality::Embeddings, Modality::Rankings],
    metrics_title: "KServe V1 Metrics",
    service_kind: "kserve",
};

const KSERVE_V2_INFER_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v2_infer",
    aliases: &[],
    description: "KServe V2 Open Inference Protocol text endpoint",
    endpoint_path: Some("/v2/models/{model_name}/infer"),
    streaming_path: Some("/v2/models/{model_name}/infer"),
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "KServe V2 Metrics",
    service_kind: "kserve",
};

const KSERVE_V2_EMBEDDINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v2_embeddings",
    aliases: &[],
    description: "KServe V2 Open Inference Protocol embeddings endpoint",
    endpoint_path: Some("/v2/models/{model_name}/infer"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Embeddings],
    metrics_title: "KServe V2 Embeddings Metrics",
    service_kind: "kserve",
};

const KSERVE_V2_RANKINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v2_rankings",
    aliases: &[],
    description: "KServe V2 Open Inference Protocol rankings endpoint",
    endpoint_path: Some("/v2/models/{model_name}/infer"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Rankings],
    metrics_title: "KServe V2 Rankings Metrics",
    service_kind: "kserve",
};

const KSERVE_V2_VLM_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v2_vlm",
    aliases: &[],
    description: "KServe V2 Open Inference Protocol vision-language endpoint",
    endpoint_path: Some("/v2/models/{model_name}/infer"),
    streaming_path: Some("/v2/models/{model_name}/infer"),
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image],
    output_modalities: &[Modality::Tokens],
    metrics_title: "KServe V2 VLM Metrics",
    service_kind: "kserve",
};

const KSERVE_V2_IMAGES_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "kserve_v2_images",
    aliases: &[],
    description: "KServe V2 Open Inference Protocol image-generation endpoint",
    endpoint_path: Some("/v2/models/{model_name}/infer"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Image],
    metrics_title: "KServe V2 Image Generation Metrics",
    service_kind: "kserve",
};

/// V2-only factory for KServe's OpenAI-compatible chat route.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeChatFactory;

/// V2-only factory for KServe's OpenAI-compatible completions route.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeCompletionsFactory;

/// V2-only factory for KServe's OpenAI-compatible embeddings route.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeEmbeddingsFactory;

macro_rules! alias_factory {
    ($factory:ty, $descriptor:expr, $endpoint:expr, $endpoint_type:expr) => {
        impl EndpointFactory for $factory {
            fn descriptor(&self) -> &'static EndpointDescriptor {
                &$descriptor
            }

            fn prepare(
                &self,
                config: EffectiveEndpointConfig,
            ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
                prepare_alias(
                    $endpoint,
                    &$descriptor,
                    $endpoint_type,
                    "/openai/v1/models",
                    config,
                )
            }
        }
    };
}

alias_factory!(
    KServeChatFactory,
    KSERVE_CHAT_DESCRIPTOR,
    ChatEndpoint,
    EndpointType::Chat
);
alias_factory!(
    KServeCompletionsFactory,
    KSERVE_COMPLETIONS_DESCRIPTOR,
    CompletionsEndpoint,
    EndpointType::Completions
);
alias_factory!(
    KServeEmbeddingsFactory,
    KSERVE_EMBEDDINGS_DESCRIPTOR,
    EmbeddingsEndpoint,
    EndpointType::Embeddings
);

fn prepare_alias<E>(
    endpoint: E,
    descriptor: &'static EndpointDescriptor,
    endpoint_type: EndpointType,
    readiness_path: &'static str,
    config: EffectiveEndpointConfig,
) -> EndpointResult<Box<dyn PreparedEndpoint>>
where
    E: PreparedEndpointBehavior + 'static,
{
    let endpoint = Arc::new(endpoint);
    let compatibility_config = EndpointConfig::from_raw(endpoint_type, config.to_raw());
    let headers = endpoint.format_headers(&compatibility_config);
    Ok(Box::new(PreparedAliasEndpoint {
        endpoint,
        descriptor,
        config,
        compatibility_config,
        headers,
        readiness_path,
    }))
}

#[derive(Debug)]
struct PreparedAliasEndpoint<E> {
    endpoint: Arc<E>,
    descriptor: &'static EndpointDescriptor,
    config: EffectiveEndpointConfig,
    compatibility_config: EndpointConfig,
    headers: BTreeMap<String, String>,
    readiness_path: &'static str,
}

impl<E> PreparedEndpoint for PreparedAliasEndpoint<E>
where
    E: PreparedEndpointBehavior + 'static,
{
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.descriptor
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        self.endpoint
            .format_prepared_payload(request, self.config.as_raw())
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, _model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(readiness_request(
            self.readiness_path.to_string(),
            self.headers.clone(),
        ))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.endpoint
            .parse_response_with_config(response, &self.compatibility_config)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.endpoint.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        self.endpoint
            .extract_response_data_with_config(record, &self.compatibility_config)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        self.endpoint.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.endpoint.captures_assistant_turn()
    }
}

/// Runner-v2-only factory for KServe V1 predict payloads.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV1PredictFactory;

/// Runner-v2-only factory for KServe V2 text inference.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2InferFactory;

/// Runner-v2-only factory for KServe V2 embeddings.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2EmbeddingsFactory;

/// Runner-v2-only factory for KServe V2 rankings.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2RankingsFactory;

/// Runner-v2-only factory for KServe V2 vision-language inference.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2VlmFactory;

/// Runner-v2-only factory for KServe V2 image generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct KServeV2ImagesFactory;

macro_rules! kserve_factory {
    ($factory:ty, $descriptor:expr, $prepare:expr) => {
        impl EndpointFactory for $factory {
            fn descriptor(&self) -> &'static EndpointDescriptor {
                &$descriptor
            }

            fn validate_config(&self, config: &mut RawEndpointConfig) -> EndpointResult<()> {
                validate_selector_values(config)
            }

            fn prepare(
                &self,
                config: EffectiveEndpointConfig,
            ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
                let behavior = $prepare(config.as_raw())?;
                Ok(Box::new(PreparedKServeEndpoint::new(
                    &$descriptor,
                    config,
                    behavior,
                )))
            }
        }
    };
}

kserve_factory!(
    KServeV1PredictFactory,
    KSERVE_V1_PREDICT_DESCRIPTOR,
    V1PredictBehavior::prepare
);
kserve_factory!(
    KServeV2InferFactory,
    KSERVE_V2_INFER_DESCRIPTOR,
    V2InferBehavior::prepare
);
kserve_factory!(
    KServeV2EmbeddingsFactory,
    KSERVE_V2_EMBEDDINGS_DESCRIPTOR,
    V2EmbeddingsBehavior::prepare
);
kserve_factory!(
    KServeV2RankingsFactory,
    KSERVE_V2_RANKINGS_DESCRIPTOR,
    V2RankingsBehavior::prepare
);
kserve_factory!(
    KServeV2VlmFactory,
    KSERVE_V2_VLM_DESCRIPTOR,
    V2VlmBehavior::prepare
);
kserve_factory!(
    KServeV2ImagesFactory,
    KSERVE_V2_IMAGES_DESCRIPTOR,
    V2ImagesBehavior::prepare
);

trait KServePreparedBehavior: fmt::Debug {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value>;
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;
    fn readiness_path(&self, model: &str) -> String;
}

#[derive(Debug)]
struct PreparedKServeEndpoint {
    descriptor: &'static EndpointDescriptor,
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
    behavior: Box<dyn KServePreparedBehavior>,
}

impl PreparedKServeEndpoint {
    fn new(
        descriptor: &'static EndpointDescriptor,
        config: EffectiveEndpointConfig,
        behavior: Box<dyn KServePreparedBehavior>,
    ) -> Self {
        let headers = bearer_headers(config.as_raw());
        Self {
            descriptor,
            config,
            headers,
            behavior,
        }
    }
}

impl PreparedEndpoint for PreparedKServeEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.descriptor
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        // KServe behaviors author a complete JSON object (`v2_payload`/`instances`);
        // wrap it once into a plan here so the internal behavior trait stays
        // Value-shaped for the protobuf-wire codec that also reads these bodies.
        let value = self.behavior.format_payload(request)?;
        let object = value.as_object().ok_or_else(|| {
            EndpointError::InvalidRequest("KServe endpoint body must be a JSON object".into())
        })?;
        Ok(BodyPlan::from_object(object)?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(readiness_request(
            self.behavior.readiness_path(model),
            self.headers.clone(),
        ))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.behavior.parse_response(response)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        self.behavior.extract_payload_inputs(body)
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

#[derive(Debug)]
struct V1PredictBehavior {
    input_field: String,
    output_field: String,
}

impl V1PredictBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        Ok(Box::new(Self {
            input_field: take_selector(&mut extra, "v1_input_field", "text")?,
            output_field: take_selector(&mut extra, "v1_output_field", "output")?,
        }))
    }
}

impl KServePreparedBehavior for V1PredictBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "KServe V1 endpoint requires at least one turn.")?;
        Ok(json!({"instances": [{self.input_field.clone(): joined_text(turn)}]}))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let Some(predictions) = object.get("predictions").and_then(Value::as_array) else {
            return Ok(auto_detect_response(response.perf_ns, object));
        };
        let Some(first) = predictions.first() else {
            return Ok(auto_detect_response(response.perf_ns, object));
        };
        let data = if let Some(prediction) = first.as_object() {
            prediction
                .get(&self.output_field)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(text_data)
                .or_else(|| auto_detect_data(prediction))
        } else {
            first
                .as_str()
                .filter(|text| !text.is_empty())
                .map(text_data)
        };
        Ok(data.map(|data| parsed(response.perf_ns, data)))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let texts = body
            .get("instances")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_object)
            .filter_map(|instance| instance.get(&self.input_field))
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect();
        ExtractedPayload {
            texts,
            ..ExtractedPayload::default()
        }
    }

    fn readiness_path(&self, model: &str) -> String {
        format!("/v1/models/{model}")
    }
}

#[derive(Debug)]
struct V2InferBehavior {
    input_name: String,
    output_name: String,
    parameters: Map<String, Value>,
}

impl V2InferBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        let input_name = take_selector(&mut extra, "v2_input_name", "text_input")?;
        let output_name = take_selector(&mut extra, "v2_output_name", "text_output")?;
        Ok(Box::new(Self {
            input_name,
            output_name,
            parameters: extra,
        }))
    }
}

impl KServePreparedBehavior for V2InferBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "KServe V2 endpoint requires at least one turn.")?;
        let mut inputs = vec![bytes_tensor(&self.input_name, vec![joined_text(turn)])];
        if let Some(max_tokens) = turn.max_tokens {
            inputs.push(tensor("max_tokens", "INT32", vec![json!(max_tokens)]));
        }
        Ok(v2_payload(inputs, &self.parameters))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_v2_text_response(response, &self.output_name)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_tensor_texts(body, &[&self.input_name], None)
    }

    fn readiness_path(&self, model: &str) -> String {
        v2_readiness_path(model)
    }
}

#[derive(Debug)]
struct V2EmbeddingsBehavior {
    input_name: String,
    output_name: String,
    parameters: Map<String, Value>,
}

impl V2EmbeddingsBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        let input_name = take_selector(&mut extra, "v2_input_name", "text_input")?;
        let output_name = take_selector(&mut extra, "v2_output_name", "embedding_output")?;
        Ok(Box::new(Self {
            input_name,
            output_name,
            parameters: extra,
        }))
    }
}

impl KServePreparedBehavior for V2EmbeddingsBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = exactly_one_turn(request, "Embeddings endpoint only supports one turn.")?;
        let texts = turn_texts(turn);
        Ok(v2_payload(
            vec![bytes_tensor(&self.input_name, texts)],
            &self.parameters,
        ))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(output) = selected_output(response, &self.output_name) else {
            return Ok(None);
        };
        let Some(data) = output.get("data").and_then(Value::as_array) else {
            return Ok(None);
        };
        if data.is_empty() {
            return Ok(None);
        }
        let flat = data
            .iter()
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    EndpointError::InvalidResponse(format!(
                        "KServe embedding output contains non-numeric value {value}"
                    ))
                })
            })
            .collect::<EndpointResult<Vec<_>>>()?;
        let embeddings = match output.get("shape").and_then(Value::as_array) {
            Some(shape) if shape.len() == 2 => {
                let n = shape[0].as_u64().unwrap_or(0) as usize;
                let width = shape[1].as_u64().unwrap_or(0) as usize;
                (0..n)
                    .map(|index| {
                        flat.get(index.saturating_mul(width)..(index + 1).saturating_mul(width))
                            .unwrap_or_default()
                            .to_vec()
                    })
                    .collect()
            }
            _ => vec![flat],
        };
        Ok(Some(parsed(
            response.perf_ns,
            ResponseData::Embeddings { embeddings },
        )))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_tensor_texts(body, &[&self.input_name], None)
    }

    fn readiness_path(&self, model: &str) -> String {
        v2_readiness_path(model)
    }
}

#[derive(Debug)]
struct V2RankingsBehavior {
    query_name: String,
    passages_name: String,
    output_name: String,
    parameters: Map<String, Value>,
}

impl V2RankingsBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        let query_name = take_selector(&mut extra, "v2_query_name", "query")?;
        let passages_name = take_selector(&mut extra, "v2_passages_name", "passages")?;
        let output_name = take_selector(&mut extra, "v2_output_name", "scores")?;
        Ok(Box::new(Self {
            query_name,
            passages_name,
            output_name,
            parameters: extra,
        }))
    }
}

impl KServePreparedBehavior for V2RankingsBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = exactly_one_turn(request, "Rankings endpoint only supports one turn.")?;
        let (query, passages) = ranking_inputs(turn)?;
        Ok(v2_payload(
            vec![
                bytes_tensor(&self.query_name, vec![query]),
                bytes_tensor(&self.passages_name, passages),
            ],
            &self.parameters,
        ))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(output) = selected_output(response, &self.output_name) else {
            return Ok(None);
        };
        let Some(data) = output.get("data").and_then(Value::as_array) else {
            return Ok(None);
        };
        let rankings = data
            .iter()
            .enumerate()
            .filter_map(|(index, score)| {
                numeric_value(score).map(|score| json!({"index": index, "score": score}))
            })
            .collect::<Vec<_>>();
        Ok((!rankings.is_empty())
            .then(|| parsed(response.perf_ns, ResponseData::Rankings { rankings })))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_tensor_texts(body, &[&self.query_name, &self.passages_name], None)
    }

    fn readiness_path(&self, model: &str) -> String {
        v2_readiness_path(model)
    }
}

#[derive(Debug)]
struct V2VlmBehavior {
    text_name: String,
    image_name: String,
    output_name: String,
    parameters: Map<String, Value>,
}

impl V2VlmBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        let text_name = take_selector(&mut extra, "v2_text_name", "text_input")?;
        let image_name = take_selector(&mut extra, "v2_image_name", "image")?;
        let output_name = take_selector(&mut extra, "v2_output_name", "text_output")?;
        Ok(Box::new(Self {
            text_name,
            image_name,
            output_name,
            parameters: extra,
        }))
    }
}

impl KServePreparedBehavior for V2VlmBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(
            request,
            "KServe V2 VLM endpoint requires at least one turn.",
        )?;
        let mut inputs = vec![bytes_tensor(&self.text_name, vec![joined_text(turn)])];
        let images = turn
            .images
            .iter()
            .flat_map(|image| image.contents.iter())
            .filter(|content| !content.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        if !images.is_empty() {
            inputs.push(bytes_tensor(&self.image_name, images));
        }
        if let Some(max_tokens) = turn.max_tokens {
            inputs.push(tensor("max_tokens", "INT32", vec![json!(max_tokens)]));
        }
        Ok(v2_payload(inputs, &self.parameters))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_v2_text_response(response, &self.output_name)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_tensor_texts(body, &[&self.text_name], Some(&self.image_name))
    }

    fn readiness_path(&self, model: &str) -> String {
        v2_readiness_path(model)
    }
}

#[derive(Debug)]
struct V2ImagesBehavior {
    prompt_name: String,
    output_name: String,
    parameters: Map<String, Value>,
    typed_inputs: Vec<Value>,
}

impl V2ImagesBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Box<dyn KServePreparedBehavior>> {
        let mut extra = endpoint_extra(config);
        let prompt_name = take_selector(&mut extra, "v2_prompt_name", "prompt")?;
        let output_name = take_selector(&mut extra, "v2_output_name", "generated_image")?;
        let mut typed_inputs = Vec::new();
        if let Some(value) = extra.remove("negative_prompt") {
            typed_inputs.push(tensor(
                "negative_prompt",
                "BYTES",
                vec![Value::String(value_to_python_string(&value))],
            ));
        }
        if let Some(value) = extra.remove("num_inference_steps") {
            typed_inputs.push(tensor(
                "num_inference_steps",
                "INT32",
                vec![Value::Number(Number::from(value_to_i64(
                    &value,
                    "num_inference_steps",
                )?))],
            ));
        }
        if let Some(value) = extra.remove("guidance_scale") {
            typed_inputs.push(tensor(
                "guidance_scale",
                "FP32",
                vec![number_value(value_to_f64(&value, "guidance_scale")?)?],
            ));
        }
        if let Some(value) = extra.remove("seed") {
            typed_inputs.push(tensor(
                "seed",
                "INT64",
                vec![Value::Number(Number::from(value_to_i64(&value, "seed")?))],
            ));
        }
        Ok(Box::new(Self {
            prompt_name,
            output_name,
            parameters: extra,
            typed_inputs,
        }))
    }
}

impl KServePreparedBehavior for V2ImagesBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(
            request,
            "KServe V2 images endpoint requires at least one turn.",
        )?;
        let mut inputs = vec![bytes_tensor(&self.prompt_name, vec![joined_text(turn)])];
        inputs.extend(self.typed_inputs.iter().cloned());
        Ok(v2_payload(inputs, &self.parameters))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(output) = selected_output(response, &self.output_name) else {
            return Ok(None);
        };
        let Some(data) = output.get("data").and_then(Value::as_array) else {
            return Ok(None);
        };
        let images = data
            .iter()
            .filter(|item| !item.is_null())
            .map(|item| ImageDataItem {
                b64_json: Some(value_to_python_string(item)),
                ..ImageDataItem::default()
            })
            .collect::<Vec<_>>();
        Ok((!images.is_empty()).then(|| {
            parsed(
                response.perf_ns,
                ResponseData::Images(ImageResponseData {
                    images,
                    ..ImageResponseData::default()
                }),
            )
        }))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        extract_tensor_texts(body, &[&self.prompt_name], None)
    }

    fn readiness_path(&self, model: &str) -> String {
        v2_readiness_path(model)
    }
}

fn validate_selector_values(config: &mut RawEndpointConfig) -> EndpointResult<()> {
    let Some(extra) = config.extra.as_ref() else {
        return Ok(());
    };
    for (name, value) in extra {
        if (name.starts_with("v1_") || name.starts_with("v2_")) && !value.is_string() {
            return Err(EndpointError::InvalidConfig(format!(
                "KServe selector {name:?} must be a string"
            )));
        }
    }
    Ok(())
}

fn endpoint_extra(config: &RawEndpointConfig) -> Map<String, Value> {
    config.extra.clone().unwrap_or_default()
}

fn take_selector(
    extra: &mut Map<String, Value>,
    name: &str,
    default: &str,
) -> EndpointResult<String> {
    match extra.remove(name) {
        None => Ok(default.to_string()),
        Some(Value::String(value)) => Ok(value),
        Some(value) => Err(EndpointError::InvalidConfig(format!(
            "KServe selector {name:?} must be a string, got {value}"
        ))),
    }
}

fn bearer_headers(config: &RawEndpointConfig) -> BTreeMap<String, String> {
    let mut headers = config.headers.clone();
    if let Some(api_key) = &config.api_key {
        headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
    }
    headers
}

fn readiness_request(path: String, headers: BTreeMap<String, String>) -> ReadinessPolicy {
    ReadinessPolicy::Request(PreparedReadinessRequest {
        method: ReadinessMethod::Get,
        path,
        headers,
        body: None,
        success: ReadinessSuccess::SuccessfulStatus,
    })
}

fn first_turn<'a>(request: &'a PreparedRequest<'_>, message: &str) -> EndpointResult<&'a Turn> {
    request
        .turns()
        .first()
        .ok_or_else(|| EndpointError::InvalidRequest(message.to_string()))
}

fn exactly_one_turn<'a>(
    request: &'a PreparedRequest<'_>,
    message: &str,
) -> EndpointResult<&'a Turn> {
    if request.turns().len() != 1 {
        return Err(EndpointError::InvalidRequest(message.to_string()));
    }
    Ok(&request.turns()[0])
}

fn turn_texts(turn: &Turn) -> Vec<String> {
    turn.texts
        .iter()
        .flat_map(|text| text.contents.iter())
        .filter(|content| !content.is_empty())
        .cloned()
        .collect()
}

fn joined_text(turn: &Turn) -> String {
    turn_texts(turn).join(" ")
}

fn tensor(name: &str, datatype: &str, data: Vec<Value>) -> Value {
    json!({
        "name": name,
        "shape": [data.len()],
        "datatype": datatype,
        "data": data,
    })
}

fn bytes_tensor(name: &str, data: Vec<String>) -> Value {
    tensor(name, "BYTES", data.into_iter().map(Value::String).collect())
}

fn v2_payload(inputs: Vec<Value>, parameters: &Map<String, Value>) -> Value {
    let mut body = Map::new();
    body.insert("inputs".to_string(), Value::Array(inputs));
    if !parameters.is_empty() {
        body.insert("parameters".to_string(), Value::Object(parameters.clone()));
    }
    Value::Object(body)
}

fn selected_output<'a>(
    response: &'a ServerResponse,
    output_name: &str,
) -> Option<&'a Map<String, Value>> {
    let outputs = response
        .json
        .as_ref()?
        .as_object()?
        .get("outputs")?
        .as_array()?;
    outputs
        .iter()
        .filter_map(Value::as_object)
        .find(|output| output.get("name").and_then(Value::as_str) == Some(output_name))
        .or_else(|| outputs.iter().find_map(Value::as_object))
}

fn parse_v2_text_response(
    response: &ServerResponse,
    output_name: &str,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(outputs) = response
        .json
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|object| object.get("outputs"))
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    let extract = |output: &Value| {
        output
            .as_object()
            .and_then(|output| output.get("data"))
            .and_then(Value::as_array)
            .and_then(|data| data.first())
            .filter(|value| !value.is_null())
            .map(value_to_python_string)
    };
    let text = outputs
        .iter()
        .find(|output| {
            output
                .as_object()
                .and_then(|output| output.get("name"))
                .and_then(Value::as_str)
                == Some(output_name)
        })
        .and_then(extract)
        .or_else(|| outputs.iter().find_map(extract));
    Ok(text.map(|text| parsed(response.perf_ns, text_data(&text))))
}

fn ranking_inputs(turn: &Turn) -> EndpointResult<(String, Vec<String>)> {
    let mut queries = Vec::new();
    let mut passages = Vec::new();
    for text in &turn.texts {
        match text.name.as_str() {
            "query" | "queries" => queries.extend(text.contents.iter().cloned()),
            "passages" => passages.extend(text.contents.iter().cloned()),
            _ => {}
        }
    }
    let Some(query) = queries.into_iter().next() else {
        return Err(EndpointError::InvalidRequest(
            "Rankings request requires a text with name 'query' or 'queries'. Provide a Text object with name='query' or name='queries' containing the search query."
                .to_string(),
        ));
    };
    Ok((query, passages))
}

fn extract_tensor_texts(
    body: &Value,
    text_names: &[&str],
    image_name: Option<&str>,
) -> ExtractedPayload {
    let mut extracted = ExtractedPayload::default();
    let Some(inputs) = body.get("inputs").and_then(Value::as_array) else {
        return extracted;
    };
    for input in inputs.iter().filter_map(Value::as_object) {
        let Some(name) = input.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some(data) = input.get("data").and_then(Value::as_array) else {
            continue;
        };
        if text_names.contains(&name) {
            extracted.texts.extend(
                data.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string),
            );
        } else if image_name == Some(name) {
            extracted.image_count = extracted
                .image_count
                .saturating_add(u32::try_from(data.len()).unwrap_or(u32::MAX));
        }
    }
    extracted
}

fn v2_readiness_path(model: &str) -> String {
    format!("/v2/models/{model}/ready")
}

fn parsed(perf_ns: u64, data: ResponseData) -> ParsedResponse {
    ParsedResponse {
        perf_ns,
        data: Some(data),
        usage: None,
        sources: None,
    }
}

fn text_data(text: &str) -> ResponseData {
    ResponseData::Text {
        text: text.to_string(),
    }
}

fn auto_detect_response(perf_ns: u64, object: &Map<String, Value>) -> Option<ParsedResponse> {
    auto_detect_data(object).map(|data| parsed(perf_ns, data))
}

fn auto_detect_data(object: &Map<String, Value>) -> Option<ResponseData> {
    if let Some(embeddings) = auto_detect_embeddings(object) {
        return Some(ResponseData::Embeddings { embeddings });
    }
    for field in ["rankings", "results"] {
        if let Some(rankings) = object.get(field).and_then(Value::as_array) {
            return Some(ResponseData::Rankings {
                rankings: rankings.clone(),
            });
        }
    }
    for field in ["text", "content", "response", "output", "result"] {
        match object.get(field) {
            Some(Value::String(text)) if !text.is_empty() => return Some(text_data(text)),
            Some(Value::Array(parts))
                if !parts.is_empty() && parts.iter().all(Value::is_string) =>
            {
                let text = parts.iter().filter_map(Value::as_str).collect::<String>();
                if !text.is_empty() {
                    return Some(text_data(&text));
                }
            }
            _ => {}
        }
    }
    let choice = object
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)?;
    choice
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
        })
        .filter(|text| !text.is_empty())
        .map(text_data)
}

fn auto_detect_embeddings(object: &Map<String, Value>) -> Option<Vec<Vec<f64>>> {
    if let Some(data) = object.get("data").and_then(Value::as_array)
        && data
            .first()
            .and_then(Value::as_object)
            .and_then(|item| item.get("object"))
            == Some(&Value::String("embedding".to_string()))
    {
        let embeddings = data
            .iter()
            .filter_map(Value::as_object)
            .filter_map(|item| item.get("embedding"))
            .filter_map(numeric_array)
            .collect::<Vec<_>>();
        if !embeddings.is_empty() {
            return Some(embeddings);
        }
    }
    for field in ["embeddings", "embedding"] {
        let Some(value) = object.get(field) else {
            continue;
        };
        if let Some(values) = numeric_array(value) {
            return Some(vec![values]);
        }
        if let Some(values) = value.as_array() {
            let embeddings = values.iter().filter_map(numeric_array).collect::<Vec<_>>();
            if !embeddings.is_empty() {
                return Some(embeddings);
            }
        }
    }
    None
}

fn numeric_array(value: &Value) -> Option<Vec<f64>> {
    let values = value.as_array()?;
    if values.is_empty() {
        return None;
    }
    values.iter().map(numeric_value).collect()
}

fn numeric_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn value_to_python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn value_to_i64(value: &Value, name: &str) -> EndpointResult<i64> {
    value
        .as_i64()
        .or_else(|| value.as_bool().map(i64::from))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .ok_or_else(|| {
            EndpointError::InvalidConfig(format!(
                "KServe image parameter {name:?} cannot be converted to an integer"
            ))
        })
}

fn value_to_f64(value: &Value, name: &str) -> EndpointResult<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .ok_or_else(|| {
            EndpointError::InvalidConfig(format!(
                "KServe image parameter {name:?} cannot be converted to a float"
            ))
        })
}

fn number_value(value: f64) -> EndpointResult<Value> {
    Number::from_f64(value).map(Value::Number).ok_or_else(|| {
        EndpointError::InvalidConfig("KServe image parameters must be finite".to_string())
    })
}
