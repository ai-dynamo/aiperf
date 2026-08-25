// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extended endpoint dialects.
//!
//! Raw/template behavior lives in the
//! sibling `flexible` module so its Jinja/JMESPath dependency boundary remains
//! explicit. HTTP multipart, polling, and inline-media execution is deliberately
//! implemented against transport traits outside this decoded-JSON layer.

mod flexible;

use serde_json::{Map, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EndpointConfig, RawEndpointConfig};
use crate::endpoints::implementation::{
    Endpoint, merge_extra, parse_embeddings_response, turn_texts,
};
use crate::endpoints::metadata::{EndpointDescriptor, Modality};
use crate::endpoints::models::{
    EndpointError, EndpointResult, ExtractedPayload, ImageDataItem, ImageResponseData,
    ParsedResponse, RequestInfo, ResponseData, ServerResponse, VideoResponseData,
};
use crate::endpoints::registry::{
    PreparedEndpointBehavior, PreparedRequest, format_legacy_payload,
};

pub use flexible::{RawEndpoint, RawEndpointFactory, TemplateEndpoint, TemplateEndpointFactory};

/// NVIDIA NIM multimodal embeddings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct NimEmbeddingsEndpoint;

/// NVIDIA NIM rankings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct NimRankingsEndpoint;

/// Cohere rankings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct CohereRankingsEndpoint;

/// Hugging Face TEI rankings endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfTeiRankingsEndpoint;

/// Hugging Face TGI generate endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuggingFaceGenerateEndpoint;

/// OpenAI-compatible image generation endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageGenerationEndpoint;

/// OpenAI-compatible multipart image edit endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageEditEndpoint;

/// OpenAI-compatible audio transcription endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioTranscriptionEndpoint;

/// OpenAI/SGLang async video generation endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct VideoGenerationEndpoint;

/// NVIDIA NIM image retrieval endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageRetrievalEndpoint;

/// SOLIDO retrieval-augmented generation endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolidoRagEndpoint;

const NIM_EMBEDDINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "nim_embeddings",
    aliases: &[],
    description: "NVIDIA NIM multimodal embeddings API",
    endpoint_path: Some("/v1/embeddings"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image],
    output_modalities: &[Modality::Embeddings],
    metrics_title: "NIM Embeddings Metrics",
    service_kind: "embeddings",
};

const NIM_RANKINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "nim_rankings",
    aliases: &[],
    description: "NVIDIA NIM rankings API",
    endpoint_path: Some("/v1/ranking"),
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
    metrics_title: "Rankings Metrics",
    service_kind: "rankings",
};

const COHERE_RANKINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "cohere_rankings",
    aliases: &[],
    description: "Cohere rerank API",
    endpoint_path: Some("/v2/rerank"),
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
    metrics_title: "Ranking Metrics",
    service_kind: "rankings",
};

const HF_TEI_RANKINGS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "hf_tei_rankings",
    aliases: &[],
    description: "Hugging Face TEI rerank API",
    endpoint_path: Some("/rerank"),
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
    metrics_title: "Ranking Metrics",
    service_kind: "rankings",
};

const HUGGINGFACE_GENERATE_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "huggingface_generate",
    aliases: &[],
    description: "Hugging Face text-generation inference API",
    endpoint_path: Some("/generate"),
    streaming_path: Some("/generate_stream"),
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
    service_kind: "llm",
};

const IMAGE_GENERATION_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "image_generation",
    aliases: &[],
    description: "OpenAI-compatible image generation API",
    endpoint_path: Some("/v1/images/generations"),
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Image],
    metrics_title: "Image Generation Metrics",
    service_kind: "image_generation",
};

const IMAGE_EDIT_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "image_edit",
    aliases: &[],
    description: "OpenAI-compatible image edit API",
    endpoint_path: Some("/v1/images/edits"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: true,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text, Modality::Image],
    output_modalities: &[Modality::Image],
    metrics_title: "Image Edit Metrics",
    service_kind: "image_edit",
};

const AUDIO_TRANSCRIPTION_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "audio_transcription",
    aliases: &[],
    description: "OpenAI audio transcription API",
    endpoint_path: Some("/v1/audio/transcriptions"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: true,
    requires_polling: false,
    requires_inline_media: true,
    input_modalities: &[Modality::Audio],
    output_modalities: &[Modality::Text],
    metrics_title: "Audio Transcription Metrics",
    service_kind: "audio_transcription",
};

const VIDEO_GENERATION_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "video_generation",
    aliases: &[],
    description: "OpenAI-compatible asynchronous video generation API",
    endpoint_path: Some("/v1/videos"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: true,
    requires_polling: true,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Video],
    metrics_title: "Video Generation Metrics",
    service_kind: "video_generation",
};

const IMAGE_RETRIEVAL_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "image_retrieval",
    aliases: &[],
    description: "NVIDIA NIM image retrieval API",
    endpoint_path: Some("/v1/infer"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: false,
    tokenizes_input: false,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: true,
    input_modalities: &[Modality::Image],
    output_modalities: &[Modality::Rankings],
    metrics_title: "Image Retrieval Metrics",
    service_kind: "image_retrieval",
};

const SOLIDO_RAG_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "solido_rag",
    aliases: &[],
    description: "SOLIDO retrieval-augmented generation API",
    endpoint_path: Some("/rag/api/prompt"),
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
    metrics_title: "SOLIDO RAG Metrics",
    service_kind: "llm",
};

impl Endpoint for NimEmbeddingsEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &NIM_EMBEDDINGS_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_embeddings_response(response, true)
    }
}

impl PreparedEndpointBehavior for NimEmbeddingsEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turn = prepared_single_turn(request, "Embeddings endpoint only supports one turn")?;
        let texts = turn_texts(turn);
        let images = turn
            .images
            .iter()
            .flat_map(|image| image.contents.iter())
            .filter(|content| !content.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        let inputs = match (texts.is_empty(), images.is_empty()) {
            (false, false) => {
                if texts.len() != images.len() {
                    return Err(EndpointError::InvalidRequest(format!(
                        "when both texts and images are provided, they must have the same length; got {} texts and {} images",
                        texts.len(),
                        images.len()
                    )));
                }
                texts
                    .into_iter()
                    .zip(images)
                    .map(|(text, image)| format!("{text} {image}"))
                    .collect()
            }
            (true, false) => images,
            _ => texts,
        };
        let mut payload = Map::new();
        payload.insert(
            "model".into(),
            Value::String(prepared_effective_model(request, turn)),
        );
        payload.insert(
            "input".into(),
            Value::Array(inputs.into_iter().map(Value::String).collect()),
        );
        merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

#[derive(Debug, Clone, Copy)]
enum RankingFlavor {
    Nim,
    Cohere,
    HfTei,
}

fn format_rankings(
    flavor: RankingFlavor,
    request: &PreparedRequest<'_>,
    config: &RawEndpointConfig,
) -> EndpointResult<BodyPlan> {
    let turn = prepared_single_turn(request, "Rankings endpoint only supports one turn")?;
    let mut queries = Vec::new();
    let mut passages = Vec::new();
    for text in &turn.texts {
        match text.name.as_str() {
            "query" | "queries" => queries.extend(text.contents.iter().cloned()),
            "passages" => passages.extend(text.contents.iter().cloned()),
            _ => {}
        }
    }
    let query = queries.first().ok_or_else(|| {
        EndpointError::InvalidRequest(
            "rankings request requires a text with name 'query' or 'queries'".into(),
        )
    })?;
    let model = prepared_effective_model(request, turn);
    let mut payload = match flavor {
        RankingFlavor::Nim => json!({
            "model": model,
            "query": {"text": query},
            "passages": passages.iter().map(|text| json!({"text": text})).collect::<Vec<_>>()
        }),
        RankingFlavor::Cohere => {
            json!({"model": model, "query": query, "documents": passages})
        }
        RankingFlavor::HfTei => json!({"query": query, "texts": passages}),
    }
    .as_object()
    .expect("ranking payload is an object")
    .clone();
    merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
    Ok(BodyPlan::from_object(&payload)?)
}

fn parse_rankings(
    flavor: RankingFlavor,
    response: &ServerResponse,
) -> EndpointResult<Option<ParsedResponse>> {
    let rankings = match flavor {
        RankingFlavor::HfTei if response.json.as_ref().is_some_and(Value::is_array) => response
            .json
            .as_ref()
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        RankingFlavor::HfTei => response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("results"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        RankingFlavor::Nim => response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("rankings"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        RankingFlavor::Cohere => response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("results"))
            .and_then(Value::as_array)
            .map(|results| {
                results
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|result| {
                        json!({
                            "index": result.get("index").cloned().unwrap_or(Value::Null),
                            "score": result.get("relevance_score").cloned().unwrap_or(Value::Null)
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    };
    Ok((!rankings.is_empty()).then_some(ParsedResponse {
        perf_ns: response.perf_ns,
        data: Some(ResponseData::Rankings { rankings }),
        usage: None,
        sources: None,
    }))
}

fn ranking_inputs(body: &Value, flavor: RankingFlavor) -> ExtractedPayload {
    let mut extracted = ExtractedPayload::default();
    let Some(object) = body.as_object() else {
        return extracted;
    };
    match flavor {
        RankingFlavor::Nim => {
            if let Some(query) = object
                .get("query")
                .and_then(Value::as_object)
                .and_then(|query| query.get("text"))
                .and_then(Value::as_str)
            {
                extracted.texts.push(query.to_string());
            }
            append_object_texts(object.get("passages"), &mut extracted.texts);
        }
        RankingFlavor::Cohere => {
            append_string(object.get("query"), &mut extracted.texts);
            append_string_list(object.get("documents"), &mut extracted.texts);
        }
        RankingFlavor::HfTei => {
            append_string(object.get("query"), &mut extracted.texts);
            append_string_list(object.get("texts"), &mut extracted.texts);
        }
    }
    extracted
}

macro_rules! ranking_endpoint {
    ($ty:ty, $flavor:ident, $descriptor:ident) => {
        impl Endpoint for $ty {
            fn descriptor(&self) -> &'static EndpointDescriptor {
                &$descriptor
            }

            fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
                format_legacy_payload(self, request_info)
            }

            fn parse_response(
                &self,
                response: &ServerResponse,
            ) -> EndpointResult<Option<ParsedResponse>> {
                parse_rankings(RankingFlavor::$flavor, response)
            }

            fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
                ranking_inputs(body, RankingFlavor::$flavor)
            }
        }

        impl PreparedEndpointBehavior for $ty {
            fn format_prepared_payload(
                &self,
                request: &PreparedRequest<'_>,
                config: &RawEndpointConfig,
            ) -> EndpointResult<BodyPlan> {
                format_rankings(RankingFlavor::$flavor, request, config)
            }
        }
    };
}

ranking_endpoint!(NimRankingsEndpoint, Nim, NIM_RANKINGS_DESCRIPTOR);
ranking_endpoint!(CohereRankingsEndpoint, Cohere, COHERE_RANKINGS_DESCRIPTOR);
ranking_endpoint!(HfTeiRankingsEndpoint, HfTei, HF_TEI_RANKINGS_DESCRIPTOR);

impl Endpoint for HuggingFaceGenerateEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &HUGGINGFACE_GENERATE_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_tgi_response(response, None)
    }

    fn parse_response_with_config(
        &self,
        response: &ServerResponse,
        config: &EndpointConfig,
    ) -> EndpointResult<Option<ParsedResponse>> {
        parse_tgi_response(response, Some(config.streaming))
    }
}

impl PreparedEndpointBehavior for HuggingFaceGenerateEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turn =
            prepared_single_turn(request, "TGI endpoint supports a single turn per request")?;
        let inputs = turn_texts(turn).join(" ");
        let mut parameters = Map::new();
        if let Some(max_tokens) = turn.max_tokens {
            parameters.insert("max_new_tokens".into(), json!(max_tokens));
        }
        merge_extra(&mut parameters, config.extra.as_ref());
        let mut payload = Map::new();
        payload.insert("inputs".into(), Value::String(inputs));
        payload.insert("parameters".into(), Value::Object(parameters));
        merge_extra(&mut payload, turn.extra_body.as_ref());
        Ok(BodyPlan::from_object(&payload)?)
    }
}

fn parse_tgi_response(
    response: &ServerResponse,
    streaming: Option<bool>,
) -> EndpointResult<Option<ParsedResponse>> {
    let value = match response.json.as_ref() {
        Some(value) => value,
        None => return Ok(None),
    };
    let inferred_streaming = value
        .as_object()
        .is_some_and(|object| object.contains_key("token"));
    let text = if streaming.unwrap_or(inferred_streaming) {
        value
            .as_object()
            .and_then(|object| object.get("token"))
            .and_then(Value::as_object)
            .and_then(|token| token.get("text"))
            .and_then(Value::as_str)
    } else if let Some(items) = value.as_array() {
        items
            .first()
            .and_then(Value::as_object)
            .and_then(|object| object.get("generated_text"))
            .and_then(Value::as_str)
    } else {
        value
            .as_object()
            .and_then(|object| object.get("generated_text"))
            .and_then(Value::as_str)
    };
    Ok(text
        .filter(|text| !text.is_empty())
        .map(|text| ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Text {
                text: text.to_string(),
            }),
            usage: None,
            sources: None,
        }))
}

impl Endpoint for ImageGenerationEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &IMAGE_GENERATION_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_image_response(response, true)
    }
}

impl PreparedEndpointBehavior for ImageGenerationEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns = prepared_require_turns(
            request,
            "Image generation endpoint requires at least one turn",
        )?;
        let turn = turns
            .last()
            .expect("require_turns returned a non-empty slice");
        let prompt = first_text(turn).ok_or_else(|| {
            EndpointError::InvalidRequest("Image generation endpoint requires a text prompt".into())
        })?;
        let mut payload = json!({
            "prompt": prompt,
            "model": prepared_effective_model(request, turn),
            "response_format": "b64_json",
            "n": 1
        })
        .as_object()
        .expect("image payload is an object")
        .clone();
        if config.streaming {
            payload.insert("stream".into(), Value::Bool(true));
        }
        merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

impl Endpoint for ImageEditEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &IMAGE_EDIT_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_image_response(response, false)
    }
}

impl Endpoint for AudioTranscriptionEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &AUDIO_TRANSCRIPTION_DESCRIPTOR
    }
    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let text = response
            .json
            .as_ref()
            .and_then(|v| v.get("text"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| response.raw.clone().filter(|value| !value.is_empty()))
            .ok_or_else(|| {
                EndpointError::InvalidResponse("audio transcription response has no text".into())
            })?;
        let usage = response.json.as_ref().and_then(|v| v.get("usage")).cloned();
        Ok(Some(ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Text { text }),
            usage,
            sources: None,
        }))
    }
}

impl PreparedEndpointBehavior for AudioTranscriptionEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turn = request.turns().last().ok_or_else(|| {
            EndpointError::InvalidRequest("audio transcription requires an audio turn".into())
        })?;
        let audio = turn
            .images
            .first()
            .and_then(|_| None)
            .or_else(|| turn.audios.first().and_then(|m| m.contents.first()))
            .ok_or_else(|| {
                EndpointError::InvalidRequest("audio transcription requires audio content".into())
            })?;
        let mut payload = json!({
            "file": build_audio_file_field(audio)?,
            "model": request.primary_model_name(),
        })
        .as_object()
        .expect("audio transcription payload is an object")
        .clone();
        merge_audio_transcription_extra(&mut payload, config.extra.as_ref());
        merge_audio_transcription_extra(&mut payload, turn.extra_body.as_ref());
        Ok(BodyPlan::from_object(&payload)?)
    }
}

#[cfg(test)]
mod audio_transcription_tests {
    use super::*;

    #[test]
    fn rejects_empty_transcription_responses() {
        let endpoint = AudioTranscriptionEndpoint;
        for response in [
            ServerResponse::from_json(1, serde_json::json!({})),
            ServerResponse {
                perf_ns: 1,
                json: None,
                raw: Some(String::new()),
            },
        ] {
            assert!(endpoint.parse_response(&response).is_err());
        }
    }
}

impl PreparedEndpointBehavior for ImageEditEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns =
            prepared_require_turns(request, "Image edit endpoint requires at least one turn")?;
        let turn = turns
            .last()
            .expect("require_turns returned a non-empty slice");
        let prompt = first_text(turn).ok_or_else(|| {
            EndpointError::InvalidRequest("Image edit endpoint requires a text prompt".into())
        })?;
        let image = turn
            .images
            .first()
            .and_then(|image| image.contents.first())
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "Image edit endpoint requires a reference image in turn.images[0]".into(),
                )
            })?;
        if image.is_empty() {
            return Err(EndpointError::InvalidRequest(
                "reference image content is empty".into(),
            ));
        }
        let mut payload = json!({
            "prompt": prompt,
            "model": prepared_effective_model(request, turn),
            "response_format": "b64_json",
            "n": 1
        })
        .as_object()
        .expect("image edit payload is an object")
        .clone();
        if image.to_ascii_lowercase().starts_with("http://")
            || image.to_ascii_lowercase().starts_with("https://")
        {
            payload.insert("url".into(), Value::String(image.clone()));
        } else {
            payload.insert("image".into(), build_image_file_field(image)?);
        }
        merge_image_edit_extra(&mut payload, config.extra.as_ref());
        merge_image_edit_extra(&mut payload, turn.extra_body.as_ref());
        Ok(BodyPlan::from_object(&payload)?)
    }
}

fn parse_image_response(
    response: &ServerResponse,
    allow_streaming_item: bool,
) -> EndpointResult<Option<ParsedResponse>> {
    let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
        return Ok(None);
    };
    if object.is_empty() {
        return Ok(None);
    }
    let mut images = Vec::new();
    if allow_streaming_item && object.contains_key("b64_json") {
        images.push(ImageDataItem {
            b64_json: optional_string(object, "b64_json"),
            partial_image_index: object.get("partial_image_index").and_then(Value::as_u64),
            ..ImageDataItem::default()
        });
    } else if let Some(data) = object.get("data").and_then(Value::as_array) {
        images.extend(
            data.iter()
                .filter_map(Value::as_object)
                .map(|item| ImageDataItem {
                    url: optional_string(item, "url"),
                    b64_json: optional_string(item, "b64_json"),
                    revised_prompt: optional_string(item, "revised_prompt"),
                    partial_image_index: None,
                }),
        );
    }
    Ok(Some(ParsedResponse {
        perf_ns: response.perf_ns,
        data: Some(ResponseData::Images(ImageResponseData {
            images,
            size: optional_string(object, "size"),
            quality: optional_string(object, "quality"),
            output_format: optional_string(object, "output_format"),
            background: optional_string(object, "background"),
        })),
        usage: non_empty(object.get("usage")),
        sources: None,
    }))
}

const RESERVED_IMAGE_EDIT_KEYS: [&str; 4] = ["prompt", "image", "url", "mask"];
const RESERVED_AUDIO_TRANSCRIPTION_KEYS: [&str; 2] = ["file", "model"];

fn merge_image_edit_extra(payload: &mut Map<String, Value>, extra: Option<&Map<String, Value>>) {
    let Some(extra) = extra else {
        return;
    };
    for (key, value) in extra {
        if !RESERVED_IMAGE_EDIT_KEYS.contains(&key.as_str()) {
            payload.insert(key.clone(), value.clone());
        }
    }
}

fn merge_audio_transcription_extra(
    payload: &mut Map<String, Value>,
    extra: Option<&Map<String, Value>>,
) {
    let Some(extra) = extra else {
        return;
    };
    for (key, value) in extra {
        if !RESERVED_AUDIO_TRANSCRIPTION_KEYS.contains(&key.as_str()) {
            payload.insert(key.clone(), value.clone());
        }
    }
}

fn build_image_file_field(content: &str) -> EndpointResult<Value> {
    let (explicit_mime, b64) = if let Some(rest) = content.strip_prefix("data:") {
        let (header, b64) = rest.split_once(',').ok_or_else(|| {
            EndpointError::InvalidRequest(
                "malformed data URL for image content (missing comma)".into(),
            )
        })?;
        let mime = header
            .split_once(';')
            .map(|(mime, _)| mime)
            .filter(|mime| mime.starts_with("image/") && !mime.is_empty());
        (mime, b64)
    } else {
        (None, content)
    };
    let mime = explicit_mime.or_else(|| sniff_image_mime(b64)).ok_or_else(|| {
        EndpointError::InvalidRequest(
            "image content is not a recognized image format; expected a data URL or raw base64 image (PNG/JPEG/WebP/GIF/BMP)".into(),
        )
    })?;
    let subtype = mime
        .split_once('/')
        .map_or("png", |(_, subtype)| subtype)
        .split_once('+')
        .map_or_else(
            || mime.split_once('/').map_or("png", |(_, value)| value),
            |(base, _)| base,
        );
    let filename_subtype = if subtype == "jpeg" { "jpg" } else { subtype };
    let content_type = match subtype {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        _ => mime,
    };
    Ok(json!({
        "b64_data": b64,
        "filename": format!("image.{filename_subtype}"),
        "content_type": content_type
    }))
}

fn build_audio_file_field(content: &str) -> EndpointResult<Value> {
    let (content_type, filename, b64) = if let Some(rest) = content.strip_prefix("data:") {
        let (header, b64) = rest.split_once(',').ok_or_else(|| {
            EndpointError::InvalidRequest(
                "malformed data URL for audio content (missing comma)".into(),
            )
        })?;
        let mime = header
            .split_once(';')
            .map(|(mime, _)| mime)
            .filter(|mime| mime.starts_with("audio/") && !mime.is_empty())
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "audio data URL must declare an audio/* media type".into(),
                )
            })?;
        let extension = mime
            .split_once('/')
            .map_or("bin", |(_, subtype)| subtype)
            .split_once('+')
            .map_or_else(
                || mime.split_once('/').map_or("bin", |(_, value)| value),
                |(base, _)| base,
            );
        (
            mime.to_string(),
            format!("audio.{extension}"),
            b64.to_string(),
        )
    } else if let Some((format, b64)) = content.split_once(',') {
        let (content_type, filename) = match format.to_ascii_lowercase().as_str() {
            "wav" => ("audio/wav", "audio.wav"),
            "mp3" => ("audio/mpeg", "audio.mp3"),
            _ => ("application/octet-stream", "audio.bin"),
        };
        (
            content_type.to_string(),
            filename.to_string(),
            b64.to_string(),
        )
    } else {
        (
            "application/octet-stream".to_string(),
            "audio.bin".to_string(),
            content.to_string(),
        )
    };
    Ok(json!({
        "b64_data": b64,
        "filename": filename,
        "content_type": content_type
    }))
}

fn sniff_image_mime(b64: &str) -> Option<&'static str> {
    [
        ("iVBORw0KGgo", "image/png"),
        ("/9j/", "image/jpeg"),
        ("R0lGODlh", "image/gif"),
        ("R0lGODdh", "image/gif"),
        ("UklGR", "image/webp"),
        ("Qk", "image/bmp"),
    ]
    .into_iter()
    .find_map(|(prefix, mime)| b64.starts_with(prefix).then_some(mime))
}

impl Endpoint for VideoGenerationEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &VIDEO_GENERATION_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        if object.is_empty() {
            return Ok(None);
        }
        Ok(Some(ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Video(Box::new(VideoResponseData {
                video_id: optional_string(object, "id"),
                object: optional_string(object, "object"),
                status: optional_string(object, "status"),
                progress: object
                    .get("progress")
                    .cloned()
                    .filter(|value| !value.is_null()),
                url: optional_string(object, "url"),
                size: optional_string(object, "size"),
                seconds: object
                    .get("seconds")
                    .cloned()
                    .filter(|value| !value.is_null()),
                quality: optional_string(object, "quality"),
                model: optional_string(object, "model"),
                created_at: object
                    .get("created_at")
                    .cloned()
                    .filter(|value| !value.is_null()),
                completed_at: object
                    .get("completed_at")
                    .cloned()
                    .filter(|value| !value.is_null()),
                expires_at: object
                    .get("expires_at")
                    .cloned()
                    .filter(|value| !value.is_null()),
                inference_time_s: object.get("inference_time_s").and_then(Value::as_f64),
                peak_memory_mb: object.get("peak_memory_mb").and_then(Value::as_f64),
                error: object
                    .get("error")
                    .cloned()
                    .filter(|value| !value.is_null()),
            }))),
            usage: non_empty(object.get("usage")),
            sources: None,
        }))
    }
}

impl PreparedEndpointBehavior for VideoGenerationEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns = prepared_require_turns(
            request,
            "Video generation endpoint requires at least one turn",
        )?;
        let turn = turns
            .last()
            .expect("require_turns returned a non-empty slice");
        let prompt = first_text(turn).ok_or_else(|| {
            EndpointError::InvalidRequest("Video generation endpoint requires a text prompt".into())
        })?;
        let mut payload = json!({
            "prompt": prompt,
            "model": prepared_effective_model(request, turn)
        })
        .as_object()
        .expect("video payload is an object")
        .clone();
        merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

impl Endpoint for ImageRetrievalEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &IMAGE_RETRIEVAL_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let data = object
            .get("data")
            .and_then(Value::as_array)
            .filter(|data| !data.is_empty())
            .cloned();
        Ok(data.map(|data| ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::ImageRetrieval { data }),
            usage: None,
            sources: None,
        }))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        // A body without this dialect's `input` array was never walked, so it
        // establishes nothing — reporting an exact zero for it would let a caller
        // skip its own accounting on a body no one inspected.
        let Some(items) = body.get("input").and_then(Value::as_array) else {
            return ExtractedPayload::default();
        };
        ExtractedPayload {
            image_count: items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("image_url"))
                .count() as u32,
            // Counted straight off the array this dialect posts, so an empty
            // `input` is an exact zero rather than an absent answer.
            owns_image_count: true,
            ..ExtractedPayload::default()
        }
    }
}

impl PreparedEndpointBehavior for ImageRetrievalEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turn =
            prepared_single_turn(request, "Image Retrieval endpoint only supports one turn")?;
        if turn.images.is_empty() {
            return Err(EndpointError::InvalidRequest(
                "Image Retrieval request requires at least one image".into(),
            ));
        }
        let input = turn
            .images
            .iter()
            .flat_map(|image| image.contents.iter())
            .filter(|content| !content.is_empty())
            .map(|content| json!({"type":"image_url", "url":content}))
            .collect::<Vec<_>>();
        if input.is_empty() {
            return Err(EndpointError::InvalidRequest(
                "no valid image content found; all images have empty contents".into(),
            ));
        }
        let mut payload = Map::new();
        payload.insert("input".into(), Value::Array(input));
        merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

impl Endpoint for SolidoRagEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &SOLIDO_RAG_DESCRIPTOR
    }

    fn format_payload(&self, request_info: &RequestInfo) -> EndpointResult<BodyPlan> {
        format_legacy_payload(self, request_info)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(object) = response.json.as_ref().and_then(Value::as_object) else {
            return Ok(None);
        };
        let Some(content) = object
            .get("content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        else {
            return Ok(None);
        };
        Ok(Some(ParsedResponse {
            perf_ns: response.perf_ns,
            data: Some(ResponseData::Text {
                text: content.to_string(),
            }),
            usage: None,
            sources: object
                .get("sources")
                .cloned()
                .filter(|sources| !sources.is_null()),
        }))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        let mut extracted = ExtractedPayload::default();
        append_string_list(body.get("query"), &mut extracted.texts);
        extracted
    }
}

impl PreparedEndpointBehavior for SolidoRagEndpoint {
    fn format_prepared_payload(
        &self,
        request: &PreparedRequest<'_>,
        config: &RawEndpointConfig,
    ) -> EndpointResult<BodyPlan> {
        let turns = prepared_require_turns(request, "SOLIDO endpoint requires at least one turn")?;
        let turn = turns
            .last()
            .expect("require_turns returned a non-empty slice");
        let mut payload = json!({
            "query": turn_texts(turn),
            "filters": {"family":"Solido", "tool":"SDE"},
            "inference_model": prepared_effective_model(request, turn)
        })
        .as_object()
        .expect("SOLIDO payload is an object")
        .clone();
        merge_prepared_endpoint_and_turn_extra(&mut payload, request, config);
        Ok(BodyPlan::from_object(&payload)?)
    }
}

fn prepared_single_turn<'a>(
    request: &'a PreparedRequest<'_>,
    message: &str,
) -> EndpointResult<&'a crate::endpoints::Turn> {
    if request.turns().len() == 1 {
        Ok(&request.turns()[0])
    } else {
        Err(EndpointError::InvalidRequest(message.into()))
    }
}

fn prepared_require_turns<'a>(
    request: &'a PreparedRequest<'_>,
    message: &str,
) -> EndpointResult<&'a [crate::endpoints::Turn]> {
    if request.turns().is_empty() {
        Err(EndpointError::InvalidRequest(message.into()))
    } else {
        Ok(request.turns())
    }
}

fn prepared_effective_model(
    request: &PreparedRequest<'_>,
    turn: &crate::endpoints::Turn,
) -> String {
    turn.model
        .clone()
        .unwrap_or_else(|| request.primary_model_name().to_string())
}

fn first_text(turn: &crate::endpoints::Turn) -> Option<&str> {
    turn.texts
        .first()
        .and_then(|text| text.contents.first())
        .map(String::as_str)
}

fn merge_prepared_endpoint_and_turn_extra(
    payload: &mut Map<String, Value>,
    request: &PreparedRequest<'_>,
    config: &RawEndpointConfig,
) {
    merge_extra(payload, config.extra.as_ref());
    merge_extra(
        payload,
        request
            .turns()
            .last()
            .and_then(|turn| turn.extra_body.as_ref()),
    );
}

fn optional_string(object: &Map<String, Value>, field: &str) -> Option<String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn non_empty(value: Option<&Value>) -> Option<Value> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::Object(object)) if object.is_empty() => None,
        Some(value) => Some(value.clone()),
    }
}

fn append_string(value: Option<&Value>, output: &mut Vec<String>) {
    if let Some(value) = value.and_then(Value::as_str) {
        output.push(value.to_string());
    }
}

fn append_string_list(value: Option<&Value>, output: &mut Vec<String>) {
    if let Some(values) = value.and_then(Value::as_array) {
        output.extend(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string),
        );
    }
}

fn append_object_texts(value: Option<&Value>, output: &mut Vec<String>) {
    if let Some(values) = value.and_then(Value::as_array) {
        output.extend(values.iter().filter_map(|value| {
            value
                .as_object()
                .and_then(|object| object.get("text"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        }));
    }
}
