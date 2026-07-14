// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-protocol-v2-only NVIDIA Riva endpoint factories.
//!
//! The factories stay on the open [`crate::endpoints::EndpointFactory`] seam; protobuf RPC
//! bindings live in `aiperf-transport-grpc`.

use std::collections::BTreeMap;
use std::fmt;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value, json};

use crate::body_plan::BodyPlan;
use crate::endpoints::config::{EffectiveEndpointConfig, RawEndpointConfig};
use crate::endpoints::metadata::{EndpointDescriptor, Modality};
use crate::endpoints::models::{
    AudioResponseData, EndpointError, EndpointResult, ExtractedPayload, ParsedResponse,
    RequestRecord, ResponseData, ServerResponse, Turn,
};
use crate::endpoints::registry::{
    EndpointFactory, PreparedEndpoint, PreparedRequest, ReadinessPolicy,
};

const RIVA_ASR_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "riva_asr",
    aliases: &[],
    description: "NVIDIA Riva automatic speech recognition",
    endpoint_path: None,
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: false,
    tokenizes_input: false,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Audio],
    output_modalities: &[Modality::Text],
    metrics_title: "Riva ASR Metrics",
    service_kind: "riva",
};

const RIVA_TTS_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "riva_tts",
    aliases: &[],
    description: "NVIDIA Riva text-to-speech synthesis",
    endpoint_path: None,
    streaming_path: None,
    supports_streaming: true,
    produces_tokens: false,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Audio],
    metrics_title: "Riva TTS Metrics",
    service_kind: "riva",
};

macro_rules! nlp_descriptor {
    ($name:ident, $id:literal, $description:literal, $title:literal) => {
        const $name: EndpointDescriptor = EndpointDescriptor {
            id: $id,
            aliases: &[],
            description: $description,
            endpoint_path: None,
            streaming_path: None,
            supports_streaming: false,
            produces_tokens: false,
            tokenizes_input: true,
            requires_raw_token_ids: false,
            requires_form_data: false,
            requires_polling: false,
            requires_inline_media: false,
            input_modalities: &[Modality::Text],
            output_modalities: &[Modality::Text],
            metrics_title: $title,
            service_kind: "riva",
        };
    };
}

nlp_descriptor!(
    RIVA_TEXT_CLASSIFY_DESCRIPTOR,
    "riva_text_classify",
    "NVIDIA Riva text classification",
    "Riva Text Classification Metrics"
);
nlp_descriptor!(
    RIVA_TOKEN_CLASSIFY_DESCRIPTOR,
    "riva_token_classify",
    "NVIDIA Riva token classification",
    "Riva Token Classification Metrics"
);
nlp_descriptor!(
    RIVA_TRANSFORM_TEXT_DESCRIPTOR,
    "riva_transform_text",
    "NVIDIA Riva text transformation",
    "Riva Text Transform Metrics"
);
nlp_descriptor!(
    RIVA_PUNCTUATE_TEXT_DESCRIPTOR,
    "riva_punctuate_text",
    "NVIDIA Riva punctuation and capitalization",
    "Riva Punctuate Text Metrics"
);
nlp_descriptor!(
    RIVA_NATURAL_QUERY_DESCRIPTOR,
    "riva_natural_query",
    "NVIDIA Riva natural-language question answering",
    "Riva Natural Query Metrics"
);
nlp_descriptor!(
    RIVA_ANALYZE_INTENT_DESCRIPTOR,
    "riva_analyze_intent",
    "NVIDIA Riva intent and slot analysis",
    "Riva Analyze Intent Metrics"
);
nlp_descriptor!(
    RIVA_ANALYZE_ENTITIES_DESCRIPTOR,
    "riva_analyze_entities",
    "NVIDIA Riva named-entity analysis",
    "Riva Analyze Entities Metrics"
);

/// Protocol-v2 Riva ASR endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaAsrFactory;
/// Protocol-v2 Riva TTS endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaTtsFactory;
/// Protocol-v2 Riva text-classification endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaTextClassifyFactory;
/// Protocol-v2 Riva token-classification endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaTokenClassifyFactory;
/// Protocol-v2 Riva text-transformation endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaTransformTextFactory;
/// Protocol-v2 Riva punctuation endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaPunctuateTextFactory;
/// Protocol-v2 Riva natural-query endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaNaturalQueryFactory;
/// Protocol-v2 Riva intent-analysis endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaAnalyzeIntentFactory;
/// Protocol-v2 Riva entity-analysis endpoint factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct RivaAnalyzeEntitiesFactory;

trait RivaPreparedBehavior: fmt::Debug {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value>;
    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>>;
    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload;
}

trait RivaBehaviorFactory: RivaPreparedBehavior + Sized + 'static {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self>;
}

macro_rules! riva_factory {
    ($factory:ty, $descriptor:ident, $behavior:ty) => {
        impl EndpointFactory for $factory {
            fn descriptor(&self) -> &'static EndpointDescriptor {
                &$descriptor
            }

            fn prepare(
                &self,
                config: EffectiveEndpointConfig,
            ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
                let behavior = <$behavior as RivaBehaviorFactory>::prepare(config.as_raw())?;
                Ok(Box::new(PreparedRivaEndpoint::new(
                    &$descriptor,
                    config,
                    Box::new(behavior),
                )))
            }
        }
    };
}

riva_factory!(RivaAsrFactory, RIVA_ASR_DESCRIPTOR, AsrBehavior);
riva_factory!(RivaTtsFactory, RIVA_TTS_DESCRIPTOR, TtsBehavior);
riva_factory!(
    RivaTextClassifyFactory,
    RIVA_TEXT_CLASSIFY_DESCRIPTOR,
    TextClassifyBehavior
);
riva_factory!(
    RivaTokenClassifyFactory,
    RIVA_TOKEN_CLASSIFY_DESCRIPTOR,
    TokenClassifyBehavior
);
riva_factory!(
    RivaTransformTextFactory,
    RIVA_TRANSFORM_TEXT_DESCRIPTOR,
    TransformTextBehavior
);
riva_factory!(
    RivaPunctuateTextFactory,
    RIVA_PUNCTUATE_TEXT_DESCRIPTOR,
    PunctuateTextBehavior
);
riva_factory!(
    RivaNaturalQueryFactory,
    RIVA_NATURAL_QUERY_DESCRIPTOR,
    NaturalQueryBehavior
);
riva_factory!(
    RivaAnalyzeIntentFactory,
    RIVA_ANALYZE_INTENT_DESCRIPTOR,
    AnalyzeIntentBehavior
);
riva_factory!(
    RivaAnalyzeEntitiesFactory,
    RIVA_ANALYZE_ENTITIES_DESCRIPTOR,
    AnalyzeEntitiesBehavior
);

#[derive(Debug)]
struct PreparedRivaEndpoint {
    descriptor: &'static EndpointDescriptor,
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
    behavior: Box<dyn RivaPreparedBehavior>,
}

impl PreparedRivaEndpoint {
    fn new(
        descriptor: &'static EndpointDescriptor,
        config: EffectiveEndpointConfig,
        behavior: Box<dyn RivaPreparedBehavior>,
    ) -> Self {
        let mut headers = config.as_raw().headers.clone();
        if let Some(api_key) = &config.as_raw().api_key {
            headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
        }
        Self {
            descriptor,
            config,
            headers,
            behavior,
        }
    }
}

impl PreparedEndpoint for PreparedRivaEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.descriptor
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<BodyPlan> {
        // Riva behaviors author a complete JSON object; wrap it once into a plan
        // here so the internal behavior trait stays Value-shaped for the gRPC codec.
        let value = self.behavior.format_payload(request)?;
        let object = value.as_object().ok_or_else(|| {
            EndpointError::InvalidRequest("Riva endpoint body must be a JSON object".into())
        })?;
        Ok(BodyPlan::from_object(object)?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, _model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Unsupported {
            reason: "Riva endpoints do not define a model-readiness RPC",
        })
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
struct AsrBehavior {
    language_code: String,
    sample_rate_hertz: u32,
    encoding: String,
    chunk_size: usize,
    streaming: bool,
}

impl RivaBehaviorFactory for AsrBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
        let extra = endpoint_extra(config);
        Ok(Self {
            language_code: string_extra(extra, "language_code", "en-US")?,
            sample_rate_hertz: u32_extra(extra, "sample_rate_hertz", 16_000)?,
            encoding: string_extra(extra, "encoding", "LINEAR_PCM")?,
            chunk_size: usize_extra(extra, "chunk_size", 8_000)?,
            streaming: config.streaming,
        })
    }
}

impl RivaPreparedBehavior for AsrBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva ASR endpoint requires at least one turn.")?;
        let audio = turn
            .audios
            .first()
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "Riva ASR endpoint requires audio data in the turn.".to_string(),
                )
            })?
            .contents
            .first()
            .filter(|content| !content.is_empty())
            .ok_or_else(|| {
                EndpointError::InvalidRequest(
                    "Riva ASR endpoint requires non-empty audio content.".to_string(),
                )
            })?;
        let audio = decode_audio_content(audio);
        if self.streaming {
            let chunks = audio
                .chunks(self.chunk_size)
                .map(|chunk| Value::String(STANDARD.encode(chunk)))
                .collect::<Vec<_>>();
            Ok(json!({
                "language_code": self.language_code,
                "sample_rate_hertz": self.sample_rate_hertz,
                "encoding": self.encoding,
                "interim_results": true,
                "audio_chunks": chunks,
            }))
        } else {
            Ok(json!({
                "audio": STANDARD.encode(audio),
                "language_code": self.language_code,
                "sample_rate_hertz": self.sample_rate_hertz,
                "encoding": self.encoding,
            }))
        }
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(transcript) = response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("transcript"))
            .and_then(Value::as_str)
            .filter(|transcript| !transcript.is_empty())
        else {
            return Ok(None);
        };
        Ok(Some(parsed(
            response.perf_ns,
            Some(ResponseData::Text {
                text: transcript.to_string(),
            }),
        )))
    }

    fn extract_payload_inputs(&self, _body: &Value) -> ExtractedPayload {
        ExtractedPayload {
            audio_count: 1,
            ..ExtractedPayload::default()
        }
    }
}

#[derive(Debug)]
struct TtsBehavior {
    voice_name: String,
    language_code: String,
    encoding: String,
    sample_rate_hz: u32,
}

impl RivaBehaviorFactory for TtsBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
        let extra = endpoint_extra(config);
        Ok(Self {
            voice_name: string_extra(extra, "voice_name", "")?,
            language_code: string_extra(extra, "language_code", "en-US")?,
            encoding: string_extra(extra, "encoding", "LINEAR_PCM")?,
            sample_rate_hz: u32_extra(extra, "sample_rate_hz", 22_050)?,
        })
    }
}

impl RivaPreparedBehavior for TtsBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva TTS endpoint requires at least one turn.")?;
        Ok(json!({
            "text": joined_text(turn),
            "voice_name": self.voice_name,
            "language_code": self.language_code,
            "encoding": self.encoding,
            "sample_rate_hz": self.sample_rate_hz,
        }))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(audio) = response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("audio"))
        else {
            return Ok(None);
        };
        let audio_bytes = decode_response_audio(audio)?;
        if audio_bytes.is_empty() {
            return Ok(None);
        }
        let duration_ms = (self.encoding == "LINEAR_PCM" && self.sample_rate_hz > 0)
            .then(|| audio_bytes.len() as f64 / 2.0 / f64::from(self.sample_rate_hz) * 1_000.0);
        Ok(Some(parsed(
            response.perf_ns,
            Some(ResponseData::Audio(AudioResponseData {
                audio_bytes,
                sample_rate_hz: self.sample_rate_hz,
                encoding: self.encoding.clone(),
                duration_ms,
            })),
        )))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        ExtractedPayload {
            texts: body
                .get("text")
                .and_then(Value::as_str)
                .map(|text| vec![text.to_string()])
                .unwrap_or_default(),
            ..ExtractedPayload::default()
        }
    }
}

#[derive(Debug)]
struct TextListConfig {
    language_code: String,
}

impl TextListConfig {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
        Ok(Self {
            language_code: string_extra(endpoint_extra(config), "language_code", "en-US")?,
        })
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva NLP endpoint requires at least one turn.")?;
        Ok(json!({
            "texts": turn_texts(turn),
            "language_code": self.language_code,
        }))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        ExtractedPayload {
            texts: body
                .get("texts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            ..ExtractedPayload::default()
        }
    }
}

macro_rules! text_list_behavior {
    ($name:ident, $parser:ident) => {
        #[derive(Debug)]
        struct $name(TextListConfig);

        impl RivaBehaviorFactory for $name {
            fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
                TextListConfig::prepare(config).map(Self)
            }
        }

        impl RivaPreparedBehavior for $name {
            fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
                self.0.format_payload(request)
            }

            fn parse_response(
                &self,
                response: &ServerResponse,
            ) -> EndpointResult<Option<ParsedResponse>> {
                $parser(response)
            }

            fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
                self.0.extract_payload_inputs(body)
            }
        }
    };
}

text_list_behavior!(TextClassifyBehavior, parse_json_response);
text_list_behavior!(TokenClassifyBehavior, parse_json_response);
text_list_behavior!(TransformTextBehavior, parse_texts_response);
text_list_behavior!(PunctuateTextBehavior, parse_texts_response);

#[derive(Debug)]
struct NaturalQueryBehavior {
    context: String,
    top_n: u32,
}

impl RivaBehaviorFactory for NaturalQueryBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
        let extra = endpoint_extra(config);
        Ok(Self {
            context: string_extra(extra, "context", "")?,
            top_n: nonnegative_u32_extra(extra, "top_n", 1)?,
        })
    }
}

impl RivaPreparedBehavior for NaturalQueryBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva NLP endpoint requires at least one turn.")?;
        Ok(json!({
            "query": joined_text(turn),
            "context": self.context,
            "top_n": self.top_n,
        }))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        let Some(first) = response
            .json
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.get("results"))
            .and_then(Value::as_array)
            .and_then(|results| results.first())
        else {
            return Ok(None);
        };
        let answer = first
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Ok(Some(parsed(
            response.perf_ns,
            Some(ResponseData::Text { text: answer }),
        )))
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        ExtractedPayload {
            texts: ["query", "context"]
                .into_iter()
                .filter_map(|name| body.get(name).and_then(Value::as_str))
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
                .collect(),
            ..ExtractedPayload::default()
        }
    }
}

#[derive(Debug)]
struct AnalyzeIntentBehavior {
    domain: String,
}

impl RivaBehaviorFactory for AnalyzeIntentBehavior {
    fn prepare(config: &RawEndpointConfig) -> EndpointResult<Self> {
        Ok(Self {
            domain: string_extra(endpoint_extra(config), "domain", "")?,
        })
    }
}

impl RivaPreparedBehavior for AnalyzeIntentBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva NLP endpoint requires at least one turn.")?;
        let mut payload = Map::from_iter([("query".to_string(), Value::String(joined_text(turn)))]);
        if !self.domain.is_empty() {
            payload.insert("domain".to_string(), Value::String(self.domain.clone()));
        }
        Ok(Value::Object(payload))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_json_response(response)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        query_inputs(body)
    }
}

#[derive(Debug, Default)]
struct AnalyzeEntitiesBehavior;

impl RivaBehaviorFactory for AnalyzeEntitiesBehavior {
    fn prepare(_config: &RawEndpointConfig) -> EndpointResult<Self> {
        Ok(Self)
    }
}

impl RivaPreparedBehavior for AnalyzeEntitiesBehavior {
    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<Value> {
        let turn = first_turn(request, "Riva NLP endpoint requires at least one turn.")?;
        Ok(json!({"query": joined_text(turn)}))
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        parse_json_response(response)
    }

    fn extract_payload_inputs(&self, body: &Value) -> ExtractedPayload {
        query_inputs(body)
    }
}

fn endpoint_extra(config: &RawEndpointConfig) -> &Map<String, Value> {
    static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
    config.extra.as_ref().unwrap_or(&EMPTY)
}

fn string_extra(extra: &Map<String, Value>, name: &str, default: &str) -> EndpointResult<String> {
    match extra.get(name) {
        None => Ok(default.to_string()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(value) => Err(EndpointError::InvalidConfig(format!(
            "Riva option {name:?} must be a string, got {value}"
        ))),
    }
}

fn u32_extra(extra: &Map<String, Value>, name: &str, default: u32) -> EndpointResult<u32> {
    let value = match extra.get(name) {
        None => return Ok(default),
        Some(Value::Number(value)) => value.as_u64(),
        Some(Value::String(value)) => value.parse::<u64>().ok(),
        Some(_) => None,
    }
    .and_then(|value| u32::try_from(value).ok())
    .filter(|value| *value > 0)
    .ok_or_else(|| {
        EndpointError::InvalidConfig(format!("Riva option {name:?} must be a positive u32"))
    })?;
    Ok(value)
}

fn nonnegative_u32_extra(
    extra: &Map<String, Value>,
    name: &str,
    default: u32,
) -> EndpointResult<u32> {
    let value = match extra.get(name) {
        None => return Ok(default),
        Some(Value::Number(value)) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        Some(Value::String(value)) => value.parse::<u32>().ok(),
        Some(_) => None,
    }
    .ok_or_else(|| {
        EndpointError::InvalidConfig(format!("Riva option {name:?} must be a non-negative u32"))
    })?;
    Ok(value)
}

fn usize_extra(extra: &Map<String, Value>, name: &str, default: usize) -> EndpointResult<usize> {
    let value = match extra.get(name) {
        None => return Ok(default),
        Some(Value::Number(value)) => value.as_u64(),
        Some(Value::String(value)) => value.parse::<u64>().ok(),
        Some(_) => None,
    }
    .and_then(|value| usize::try_from(value).ok())
    .filter(|value| *value > 0)
    .ok_or_else(|| {
        EndpointError::InvalidConfig(format!("Riva option {name:?} must be a positive integer"))
    })?;
    Ok(value)
}

fn first_turn<'a>(request: &'a PreparedRequest<'_>, message: &str) -> EndpointResult<&'a Turn> {
    request
        .turns()
        .first()
        .ok_or_else(|| EndpointError::InvalidRequest(message.to_string()))
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

fn decode_audio_content(content: &str) -> Vec<u8> {
    let encoded = if content.starts_with("data:") {
        content
            .split_once(',')
            .map(|(_, value)| value)
            .unwrap_or(content)
    } else {
        content
            .split_once(',')
            .filter(|(format, _)| matches!(format.to_ascii_lowercase().as_str(), "wav" | "mp3"))
            .map_or(content, |(_, value)| value)
    };
    STANDARD
        .decode(encoded)
        .unwrap_or_else(|_| content.as_bytes().to_vec())
}

fn decode_response_audio(value: &Value) -> EndpointResult<Vec<u8>> {
    if let Some(encoded) = value.as_str() {
        return STANDARD.decode(encoded).map_err(|error| {
            EndpointError::InvalidResponse(format!("Riva TTS audio is not valid base64: {error}"))
        });
    }
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| {
                        EndpointError::InvalidResponse(
                            "Riva TTS audio byte array contains a non-u8 value".to_string(),
                        )
                    })
            })
            .collect();
    }
    Err(EndpointError::InvalidResponse(
        "Riva TTS audio must be base64 text or a byte array".to_string(),
    ))
}

fn parse_json_response(response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
    let Some(value) = response
        .json
        .as_ref()
        .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
    else {
        return Ok(None);
    };
    let text = serde_json::to_string(value).map_err(|error| {
        EndpointError::InvalidResponse(format!("serialize Riva NLP response: {error}"))
    })?;
    Ok(Some(parsed(
        response.perf_ns,
        Some(ResponseData::Text { text }),
    )))
}

fn parse_texts_response(response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
    let Some(texts) = response
        .json
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|object| object.get("texts"))
        .and_then(Value::as_array)
        .filter(|texts| !texts.is_empty())
    else {
        return Ok(None);
    };
    let texts = texts
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    if texts.is_empty() {
        return Ok(None);
    }
    Ok(Some(parsed(
        response.perf_ns,
        Some(ResponseData::Text { text: texts }),
    )))
}

fn query_inputs(body: &Value) -> ExtractedPayload {
    ExtractedPayload {
        texts: body
            .get("query")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(|text| vec![text.to_string()])
            .unwrap_or_default(),
        ..ExtractedPayload::default()
    }
}

fn parsed(perf_ns: u64, data: Option<ResponseData>) -> ParsedResponse {
    ParsedResponse {
        perf_ns,
        data,
        usage: None,
        sources: None,
    }
}
