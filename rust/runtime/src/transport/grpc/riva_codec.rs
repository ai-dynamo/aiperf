// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Riva canonical-JSON/protobuf conversion.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use prost::Message;
use serde_json::{Map, Number, Value};

use crate::transport::grpc::binding::GrpcStreamChunk;
use crate::transport::grpc::codec::CodecError;
use crate::transport::grpc::riva_proto::streaming_recognize_request::StreamingRequest;
use crate::transport::grpc::riva_proto::{
    AnalyzeEntitiesRequest, AnalyzeIntentOptions, AnalyzeIntentRequest, AnalyzeIntentResponse,
    AudioEncoding, Classification, NaturalQueryRequest, NaturalQueryResponse, NlpModelParams,
    RecognitionConfig, RecognizeRequest, RecognizeResponse, RequestId,
    SpeechRecognitionAlternative, StreamingRecognitionConfig, StreamingRecognizeRequest,
    StreamingRecognizeResponse, SynthesizeSpeechRequest, SynthesizeSpeechResponse,
    TextClassRequest, TextClassResponse, TextTransformRequest, TextTransformResponse,
    TokenClassRequest, TokenClassResponse, TokenClassValue,
};

pub(crate) fn encode_asr_request(
    payload: &Value,
    model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva ASR payload")?;
    encode(RecognizeRequest {
        config: Some(recognition_config(object, model_name)?),
        audio: object
            .get("audio")
            .map(audio_bytes)
            .transpose()?
            .unwrap_or_default(),
        id: optional_id(request_id),
    })
}

pub(crate) fn encode_asr_stream_requests(
    payload: &Value,
    model_name: &str,
    request_id: &str,
) -> Result<Vec<Bytes>, CodecError> {
    let object = object(payload, "Riva streaming ASR payload")?;
    let mut messages = Vec::new();
    messages.push(encode(StreamingRecognizeRequest {
        streaming_request: Some(StreamingRequest::StreamingConfig(
            StreamingRecognitionConfig {
                config: Some(recognition_config(object, model_name)?),
                interim_results: bool_field(object, "interim_results", true)?,
            },
        )),
        id: optional_id(request_id),
    })?);
    for chunk in object
        .get("audio_chunks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        messages.push(encode(StreamingRecognizeRequest {
            streaming_request: Some(StreamingRequest::AudioContent(audio_bytes(chunk)?)),
            id: None,
        })?);
    }
    Ok(messages)
}

pub(crate) fn decode_asr_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = RecognizeResponse::decode(bytes)
        .map_err(|error| decode_error("Riva ASR response", error))?;
    let transcript = top_transcripts(
        response
            .results
            .iter()
            .map(|result| result.alternatives.as_slice()),
    );
    let results = response
        .results
        .iter()
        .map(|result| {
            Ok(Value::Object(Map::from_iter([(
                "alternatives".to_string(),
                alternatives(&result.alternatives)?,
            )])))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    Ok(Value::Object(Map::from_iter([
        ("transcript".to_string(), Value::String(transcript)),
        ("results".to_string(), Value::Array(results)),
    ])))
}

pub(crate) fn decode_asr_stream_response(bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
    let response = StreamingRecognizeResponse::decode(bytes)
        .map_err(|error| decode_error("Riva streaming ASR response", error))?;
    let transcript = top_transcripts(
        response
            .results
            .iter()
            .map(|result| result.alternatives.as_slice()),
    );
    let is_final = response.results.iter().any(|result| result.is_final);
    let results = response
        .results
        .iter()
        .map(|result| {
            Ok(Value::Object(Map::from_iter([
                (
                    "alternatives".to_string(),
                    alternatives(&result.alternatives)?,
                ),
                ("is_final".to_string(), Value::Bool(result.is_final)),
                (
                    "stability".to_string(),
                    finite_number(f64::from(result.stability), "Riva ASR stability")?,
                ),
            ])))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    Ok(GrpcStreamChunk {
        error_message: None,
        response: Some(Value::Object(Map::from_iter([
            ("transcript".to_string(), Value::String(transcript)),
            ("is_final".to_string(), Value::Bool(is_final)),
            ("results".to_string(), Value::Array(results)),
        ]))),
        response_size: bytes.len(),
    })
}

pub(crate) fn encode_tts_request(
    payload: &Value,
    _model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva TTS payload")?;
    encode(SynthesizeSpeechRequest {
        text: string_field(object, "text", "")?,
        language_code: string_field(object, "language_code", "en-US")?,
        encoding: audio_encoding(&string_field(object, "encoding", "LINEAR_PCM")?) as i32,
        sample_rate_hz: i32_field(object, "sample_rate_hz", 22_050)?,
        voice_name: string_field(object, "voice_name", "")?,
        id: optional_id(request_id),
    })
}

pub(crate) fn decode_tts_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = SynthesizeSpeechResponse::decode(bytes)
        .map_err(|error| decode_error("Riva TTS response", error))?;
    let mut object = Map::from_iter([(
        "audio".to_string(),
        Value::String(STANDARD.encode(response.audio)),
    )]);
    if let Some(meta) = response.meta.filter(|meta| !meta.text.is_empty()) {
        object.insert(
            "meta".to_string(),
            Value::Object(Map::from_iter([
                ("text".to_string(), Value::String(meta.text)),
                (
                    "processed_text".to_string(),
                    Value::String(meta.processed_text),
                ),
            ])),
        );
    }
    Ok(Value::Object(object))
}

pub(crate) fn decode_tts_stream_response(bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
    Ok(GrpcStreamChunk {
        error_message: None,
        response: Some(decode_tts_response(bytes)?),
        response_size: bytes.len(),
    })
}

pub(crate) fn encode_text_classify_request(
    payload: &Value,
    model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva text-classification payload")?;
    let (text, top_n, model, id) = text_list_fields(object, model_name, request_id)?;
    encode(TextClassRequest {
        text,
        top_n,
        model: Some(model),
        id,
    })
}

pub(crate) fn decode_text_classify_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = TextClassResponse::decode(bytes)
        .map_err(|error| decode_error("Riva text-classification response", error))?;
    let results = response
        .results
        .iter()
        .map(|result| {
            Ok(Value::Object(Map::from_iter([(
                "labels".to_string(),
                classifications(&result.labels)?,
            )])))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    Ok(Value::Object(Map::from_iter([(
        "results".to_string(),
        Value::Array(results),
    )])))
}

pub(crate) fn encode_token_classify_request(
    payload: &Value,
    model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva token-classification payload")?;
    let (text, top_n, model, id) = text_list_fields(object, model_name, request_id)?;
    encode(TokenClassRequest {
        text,
        top_n,
        model: Some(model),
        id,
    })
}

pub(crate) fn decode_token_classify_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = TokenClassResponse::decode(bytes)
        .map_err(|error| decode_error("Riva token-classification response", error))?;
    token_class_response(&response)
}

pub(crate) fn encode_transform_text_request(
    payload: &Value,
    model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva text-transformation payload")?;
    let (text, top_n, model, id) = text_list_fields(object, model_name, request_id)?;
    encode(TextTransformRequest {
        text,
        top_n,
        model: Some(model),
        id,
    })
}

pub(crate) fn decode_transform_text_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = TextTransformResponse::decode(bytes)
        .map_err(|error| decode_error("Riva text-transformation response", error))?;
    Ok(Value::Object(Map::from_iter([(
        "texts".to_string(),
        Value::Array(response.text.into_iter().map(Value::String).collect()),
    )])))
}

pub(crate) fn encode_natural_query_request(
    payload: &Value,
    _model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva natural-query payload")?;
    encode(NaturalQueryRequest {
        query: string_field(object, "query", "")?,
        top_n: u32_field(object, "top_n", 1)?,
        context: string_field(object, "context", "")?,
        id: optional_id(request_id),
    })
}

pub(crate) fn decode_natural_query_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = NaturalQueryResponse::decode(bytes)
        .map_err(|error| decode_error("Riva natural-query response", error))?;
    let results = response
        .results
        .iter()
        .map(|result| {
            Ok(Value::Object(Map::from_iter([
                ("answer".to_string(), Value::String(result.answer.clone())),
                (
                    "score".to_string(),
                    finite_number(f64::from(result.score), "Riva natural-query score")?,
                ),
            ])))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    Ok(Value::Object(Map::from_iter([(
        "results".to_string(),
        Value::Array(results),
    )])))
}

pub(crate) fn encode_analyze_intent_request(
    payload: &Value,
    _model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva intent-analysis payload")?;
    let domain = string_field(object, "domain", "")?;
    encode(AnalyzeIntentRequest {
        query: string_field(object, "query", "")?,
        options: (!domain.is_empty()).then(|| AnalyzeIntentOptions {
            domain,
            lang: String::new(),
        }),
        id: optional_id(request_id),
    })
}

pub(crate) fn decode_analyze_intent_response(bytes: &[u8]) -> Result<Value, CodecError> {
    let response = AnalyzeIntentResponse::decode(bytes)
        .map_err(|error| decode_error("Riva intent-analysis response", error))?;
    let mut object = Map::from_iter([
        (
            "intent".to_string(),
            response
                .intent
                .as_ref()
                .map(classification)
                .transpose()?
                .unwrap_or_else(empty_classification),
        ),
        ("slots".to_string(), token_values(&response.slots)?),
    ]);
    if let Some(domain) = response
        .domain
        .as_ref()
        .filter(|domain| !domain.class_name.is_empty())
    {
        object.insert("domain".to_string(), classification(domain)?);
    }
    Ok(Value::Object(object))
}

pub(crate) fn encode_analyze_entities_request(
    payload: &Value,
    _model_name: &str,
    request_id: &str,
) -> Result<Bytes, CodecError> {
    let object = object(payload, "Riva entity-analysis payload")?;
    encode(AnalyzeEntitiesRequest {
        query: string_field(object, "query", "")?,
        options: None,
        id: optional_id(request_id),
    })
}

pub(crate) fn decode_analyze_entities_response(bytes: &[u8]) -> Result<Value, CodecError> {
    decode_token_classify_response(bytes)
}

fn recognition_config(
    object: &Map<String, Value>,
    model_name: &str,
) -> Result<RecognitionConfig, CodecError> {
    let payload_model = string_field(object, "model", "")?;
    Ok(RecognitionConfig {
        encoding: audio_encoding(&string_field(object, "encoding", "LINEAR_PCM")?) as i32,
        sample_rate_hertz: i32_field(object, "sample_rate_hertz", 16_000)?,
        language_code: string_field(object, "language_code", "en-US")?,
        max_alternatives: i32_field(object, "max_alternatives", 1)?,
        enable_automatic_punctuation: bool_field(object, "enable_automatic_punctuation", true)?,
        model: if payload_model.is_empty() {
            model_name.to_string()
        } else {
            payload_model
        },
    })
}

fn text_list_fields(
    object: &Map<String, Value>,
    model_name: &str,
    request_id: &str,
) -> Result<(Vec<String>, u32, NlpModelParams, Option<RequestId>), CodecError> {
    let text = object
        .get("texts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| CodecError::new("Riva NLP texts must contain only strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        text,
        u32_field(object, "top_n", 0)?,
        NlpModelParams {
            model_name: string_field(object, "model_name", model_name)?,
            language_code: string_field(object, "language_code", "en-US")?,
        },
        optional_id(request_id),
    ))
}

fn token_class_response(response: &TokenClassResponse) -> Result<Value, CodecError> {
    let results = response
        .results
        .iter()
        .map(|sequence| {
            Ok(Value::Object(Map::from_iter([(
                "tokens".to_string(),
                token_values(&sequence.results)?,
            )])))
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    Ok(Value::Object(Map::from_iter([(
        "results".to_string(),
        Value::Array(results),
    )])))
}

fn token_values(values: &[TokenClassValue]) -> Result<Value, CodecError> {
    values
        .iter()
        .map(|value| {
            Ok(Value::Object(Map::from_iter([
                ("token".to_string(), Value::String(value.token.clone())),
                ("labels".to_string(), classifications(&value.label)?),
            ])))
        })
        .collect::<Result<Vec<_>, CodecError>>()
        .map(Value::Array)
}

fn alternatives(values: &[SpeechRecognitionAlternative]) -> Result<Value, CodecError> {
    values
        .iter()
        .map(|value| {
            Ok(Value::Object(Map::from_iter([
                (
                    "transcript".to_string(),
                    Value::String(value.transcript.clone()),
                ),
                (
                    "confidence".to_string(),
                    finite_number(f64::from(value.confidence), "Riva ASR confidence")?,
                ),
            ])))
        })
        .collect::<Result<Vec<_>, CodecError>>()
        .map(Value::Array)
}

fn top_transcripts<'a>(
    results: impl IntoIterator<Item = &'a [SpeechRecognitionAlternative]>,
) -> String {
    results
        .into_iter()
        .filter_map(|alternatives| alternatives.first())
        .map(|alternative| alternative.transcript.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn classifications(values: &[Classification]) -> Result<Value, CodecError> {
    values
        .iter()
        .map(classification)
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn classification(value: &Classification) -> Result<Value, CodecError> {
    Ok(Value::Object(Map::from_iter([
        (
            "class_name".to_string(),
            Value::String(value.class_name.clone()),
        ),
        (
            "score".to_string(),
            finite_number(f64::from(value.score), "Riva classification score")?,
        ),
    ])))
}

fn empty_classification() -> Value {
    Value::Object(Map::from_iter([
        ("class_name".to_string(), Value::String(String::new())),
        (
            "score".to_string(),
            Value::Number(Number::from_f64(0.0).expect("zero is finite")),
        ),
    ]))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, CodecError> {
    value
        .as_object()
        .ok_or_else(|| CodecError::new(format!("{label} must be a JSON object")))
}

fn string_field(
    object: &Map<String, Value>,
    name: &str,
    default: &str,
) -> Result<String, CodecError> {
    match object.get(name) {
        None => Ok(default.to_string()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(CodecError::new(format!(
            "Riva field {name:?} must be a string"
        ))),
    }
}

fn u32_field(object: &Map<String, Value>, name: &str, default: u32) -> Result<u32, CodecError> {
    match object.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| CodecError::new(format!("Riva field {name:?} must be a u32"))),
    }
}

fn i32_field(object: &Map<String, Value>, name: &str, default: i32) -> Result<i32, CodecError> {
    match object.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| CodecError::new(format!("Riva field {name:?} must be an i32"))),
    }
}

fn bool_field(object: &Map<String, Value>, name: &str, default: bool) -> Result<bool, CodecError> {
    match object.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| CodecError::new(format!("Riva field {name:?} must be a boolean"))),
    }
}

fn audio_bytes(value: &Value) -> Result<Vec<u8>, CodecError> {
    if let Some(encoded) = value.as_str() {
        // Reject malformed base64 rather than reinterpreting the raw string bytes
        // as audio, which would silently smuggle invalid input into the RPC.
        return STANDARD
            .decode(encoded)
            .map_err(|error| CodecError::new(format!("Riva audio is not valid base64: {error}")));
    }
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| CodecError::new("Riva audio array contains a non-u8 value"))
            })
            .collect();
    }
    Err(CodecError::new(
        "Riva audio must be base64 text or a byte array",
    ))
}

fn audio_encoding(value: &str) -> AudioEncoding {
    match value {
        "FLAC" => AudioEncoding::Flac,
        "MULAW" => AudioEncoding::Mulaw,
        "OGGOPUS" => AudioEncoding::Oggopus,
        "ALAW" => AudioEncoding::Alaw,
        _ => AudioEncoding::LinearPcm,
    }
}

fn optional_id(value: &str) -> Option<RequestId> {
    (!value.is_empty()).then(|| RequestId {
        value: value.to_string(),
    })
}

fn encode(message: impl Message) -> Result<Bytes, CodecError> {
    let mut bytes = Vec::with_capacity(message.encoded_len());
    message
        .encode(&mut bytes)
        .map_err(|error| CodecError::new(format!("encode Riva protobuf: {error}")))?;
    Ok(Bytes::from(bytes))
}

fn decode_error(label: &str, error: prost::DecodeError) -> CodecError {
    CodecError::new(format!("decode {label}: {error}"))
}

fn finite_number(value: f64, label: &str) -> Result<Value, CodecError> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| CodecError::new(format!("{label} is non-finite")))
}
