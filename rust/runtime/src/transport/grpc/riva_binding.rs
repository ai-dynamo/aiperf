// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open gRPC bindings for NVIDIA Riva ASR, TTS, and NLP services.

use std::fmt;

use bytes::Bytes;
use http::uri::PathAndQuery;
use serde_json::Value;

use crate::endpoints::EndpointId;

use crate::transport::grpc::binding::{
    GrpcBindingRegistryBuilder, GrpcEndpointBinding, GrpcEndpointBindingFactory,
    GrpcEndpointBindingRegistryError, GrpcStreamChunk,
};
use crate::transport::grpc::codec::CodecError;
use crate::transport::grpc::riva_codec::{
    decode_analyze_entities_response, decode_analyze_intent_response, decode_asr_response,
    decode_asr_stream_response, decode_natural_query_response, decode_text_classify_response,
    decode_token_classify_response, decode_transform_text_response, decode_tts_response,
    decode_tts_stream_response, encode_analyze_entities_request, encode_analyze_intent_request,
    encode_asr_request, encode_asr_stream_requests, encode_natural_query_request,
    encode_text_classify_request, encode_token_classify_request, encode_transform_text_request,
    encode_tts_request,
};

type EncodeFn = fn(&Value, &str, &str) -> Result<Bytes, CodecError>;
type DecodeFn = fn(&[u8]) -> Result<Value, CodecError>;
type DecodeStreamFn = fn(&[u8]) -> Result<GrpcStreamChunk, CodecError>;
type EncodeBidiFn = fn(&Value, &str, &str) -> Result<Vec<Bytes>, CodecError>;

trait RivaWireBehavior: fmt::Debug {
    fn unary_method(&self) -> &'static PathAndQuery;
    fn streaming_method(&self) -> Option<&'static PathAndQuery> {
        None
    }
    fn bidi_streaming_method(&self) -> Option<&'static PathAndQuery> {
        None
    }
    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError>;
    fn encode_bidi_requests(
        &self,
        _payload: &Value,
        _model_name: &str,
        _request_id: &str,
    ) -> Result<Vec<Bytes>, CodecError> {
        Err(CodecError::new(
            "this Riva RPC does not support bidirectional streaming",
        ))
    }
    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError>;
    fn decode_stream_response(&self, _bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
        Err(CodecError::new(
            "this Riva RPC does not support server streaming",
        ))
    }
}

#[derive(Debug)]
struct PreparedRivaGrpcBinding {
    endpoint_id: EndpointId,
    behavior: Box<dyn RivaWireBehavior>,
}

impl GrpcEndpointBinding for PreparedRivaGrpcBinding {
    fn endpoint_id(&self) -> &EndpointId {
        &self.endpoint_id
    }

    fn unary_method(&self) -> &'static PathAndQuery {
        self.behavior.unary_method()
    }

    fn streaming_method(&self) -> Option<&'static PathAndQuery> {
        self.behavior.streaming_method()
    }

    fn bidi_streaming_method(&self) -> Option<&'static PathAndQuery> {
        self.behavior.bidi_streaming_method()
    }

    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError> {
        self.behavior
            .encode_request(payload, model_name, request_id)
    }

    fn encode_bidi_requests(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Vec<Bytes>, CodecError> {
        self.behavior
            .encode_bidi_requests(payload, model_name, request_id)
    }

    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError> {
        self.behavior.decode_response(bytes)
    }

    fn decode_stream_response(&self, bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
        self.behavior.decode_stream_response(bytes)
    }
}

#[derive(Clone, Copy)]
struct UnaryBehavior {
    method: &'static PathAndQuery,
    encode: EncodeFn,
    decode: DecodeFn,
}

impl fmt::Debug for UnaryBehavior {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnaryBehavior")
            .field("method", &self.method)
            .finish_non_exhaustive()
    }
}

impl RivaWireBehavior for UnaryBehavior {
    fn unary_method(&self) -> &'static PathAndQuery {
        self.method
    }

    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError> {
        (self.encode)(payload, model_name, request_id)
    }

    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError> {
        (self.decode)(bytes)
    }
}

#[derive(Clone, Copy)]
struct ServerStreamingBehavior {
    unary_method: &'static PathAndQuery,
    streaming_method: &'static PathAndQuery,
    encode: EncodeFn,
    decode: DecodeFn,
    decode_stream: DecodeStreamFn,
}

impl fmt::Debug for ServerStreamingBehavior {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerStreamingBehavior")
            .field("unary_method", &self.unary_method)
            .field("streaming_method", &self.streaming_method)
            .finish_non_exhaustive()
    }
}

impl RivaWireBehavior for ServerStreamingBehavior {
    fn unary_method(&self) -> &'static PathAndQuery {
        self.unary_method
    }

    fn streaming_method(&self) -> Option<&'static PathAndQuery> {
        Some(self.streaming_method)
    }

    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError> {
        (self.encode)(payload, model_name, request_id)
    }

    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError> {
        (self.decode)(bytes)
    }

    fn decode_stream_response(&self, bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
        (self.decode_stream)(bytes)
    }
}

#[derive(Clone, Copy)]
struct BidiStreamingBehavior {
    unary_method: &'static PathAndQuery,
    bidi_method: &'static PathAndQuery,
    encode: EncodeFn,
    encode_bidi: EncodeBidiFn,
    decode: DecodeFn,
    decode_stream: DecodeStreamFn,
}

impl fmt::Debug for BidiStreamingBehavior {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BidiStreamingBehavior")
            .field("unary_method", &self.unary_method)
            .field("bidi_method", &self.bidi_method)
            .finish_non_exhaustive()
    }
}

impl RivaWireBehavior for BidiStreamingBehavior {
    fn unary_method(&self) -> &'static PathAndQuery {
        self.unary_method
    }

    fn bidi_streaming_method(&self) -> Option<&'static PathAndQuery> {
        Some(self.bidi_method)
    }

    fn encode_request(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Bytes, CodecError> {
        (self.encode)(payload, model_name, request_id)
    }

    fn encode_bidi_requests(
        &self,
        payload: &Value,
        model_name: &str,
        request_id: &str,
    ) -> Result<Vec<Bytes>, CodecError> {
        (self.encode_bidi)(payload, model_name, request_id)
    }

    fn decode_response(&self, bytes: &[u8]) -> Result<Value, CodecError> {
        (self.decode)(bytes)
    }

    fn decode_stream_response(&self, bytes: &[u8]) -> Result<GrpcStreamChunk, CodecError> {
        (self.decode_stream)(bytes)
    }
}

macro_rules! binding_factory {
    ($name:ident, $doc:literal, $endpoint:literal, $prepare:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl GrpcEndpointBindingFactory for $name {
            fn endpoint_ids(&self) -> &'static [&'static str] {
                &[$endpoint]
            }

            fn prepare(
                &self,
                endpoint_id: &EndpointId,
            ) -> Result<Box<dyn GrpcEndpointBinding>, CodecError> {
                if endpoint_id.as_str() != $endpoint {
                    return Err(CodecError::new(format!(
                        "Riva binding factory for {:?} received endpoint {}",
                        $endpoint, endpoint_id
                    )));
                }
                Ok(Box::new(PreparedRivaGrpcBinding {
                    endpoint_id: endpoint_id.clone(),
                    behavior: Box::new($prepare),
                }))
            }
        }
    };
}

static ASR_UNARY: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.asr.RivaSpeechRecognition/Recognize");
static ASR_BIDI: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize");
static TTS_UNARY: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.tts.RivaSpeechSynthesis/Synthesize");
static TTS_STREAM: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.tts.RivaSpeechSynthesis/SynthesizeOnline");
static TEXT_CLASSIFY: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyText");
static TOKEN_CLASSIFY: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyTokens");
static TRANSFORM_TEXT: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/TransformText");
static PUNCTUATE_TEXT: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/PunctuateText");
static NATURAL_QUERY: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/NaturalQuery");
static ANALYZE_INTENT: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeIntent");
static ANALYZE_ENTITIES: PathAndQuery =
    PathAndQuery::from_static("/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeEntities");

binding_factory!(
    RivaAsrGrpcBindingFactory,
    "gRPC binding factory for unary and bidirectional Riva ASR.",
    "riva_asr",
    BidiStreamingBehavior {
        unary_method: &ASR_UNARY,
        bidi_method: &ASR_BIDI,
        encode: encode_asr_request,
        encode_bidi: encode_asr_stream_requests,
        decode: decode_asr_response,
        decode_stream: decode_asr_stream_response,
    }
);
binding_factory!(
    RivaTtsGrpcBindingFactory,
    "gRPC binding factory for unary and server-streaming Riva TTS.",
    "riva_tts",
    ServerStreamingBehavior {
        unary_method: &TTS_UNARY,
        streaming_method: &TTS_STREAM,
        encode: encode_tts_request,
        decode: decode_tts_response,
        decode_stream: decode_tts_stream_response,
    }
);
binding_factory!(
    RivaTextClassifyGrpcBindingFactory,
    "gRPC binding factory for Riva text classification.",
    "riva_text_classify",
    UnaryBehavior {
        method: &TEXT_CLASSIFY,
        encode: encode_text_classify_request,
        decode: decode_text_classify_response,
    }
);
binding_factory!(
    RivaTokenClassifyGrpcBindingFactory,
    "gRPC binding factory for Riva token classification.",
    "riva_token_classify",
    UnaryBehavior {
        method: &TOKEN_CLASSIFY,
        encode: encode_token_classify_request,
        decode: decode_token_classify_response,
    }
);
binding_factory!(
    RivaTransformTextGrpcBindingFactory,
    "gRPC binding factory for Riva text transformation.",
    "riva_transform_text",
    UnaryBehavior {
        method: &TRANSFORM_TEXT,
        encode: encode_transform_text_request,
        decode: decode_transform_text_response,
    }
);
binding_factory!(
    RivaPunctuateTextGrpcBindingFactory,
    "gRPC binding factory for Riva punctuation.",
    "riva_punctuate_text",
    UnaryBehavior {
        method: &PUNCTUATE_TEXT,
        encode: encode_transform_text_request,
        decode: decode_transform_text_response,
    }
);
binding_factory!(
    RivaNaturalQueryGrpcBindingFactory,
    "gRPC binding factory for Riva natural query.",
    "riva_natural_query",
    UnaryBehavior {
        method: &NATURAL_QUERY,
        encode: encode_natural_query_request,
        decode: decode_natural_query_response,
    }
);
binding_factory!(
    RivaAnalyzeIntentGrpcBindingFactory,
    "gRPC binding factory for Riva intent analysis.",
    "riva_analyze_intent",
    UnaryBehavior {
        method: &ANALYZE_INTENT,
        encode: encode_analyze_intent_request,
        decode: decode_analyze_intent_response,
    }
);
binding_factory!(
    RivaAnalyzeEntitiesGrpcBindingFactory,
    "gRPC binding factory for Riva entity analysis.",
    "riva_analyze_entities",
    UnaryBehavior {
        method: &ANALYZE_ENTITIES,
        encode: encode_analyze_entities_request,
        decode: decode_analyze_entities_response,
    }
);

pub(crate) fn register_builtins(
    builder: &mut GrpcBindingRegistryBuilder,
) -> Result<(), GrpcEndpointBindingRegistryError> {
    builder.register(RivaAsrGrpcBindingFactory)?;
    builder.register(RivaTtsGrpcBindingFactory)?;
    builder.register(RivaTextClassifyGrpcBindingFactory)?;
    builder.register(RivaTokenClassifyGrpcBindingFactory)?;
    builder.register(RivaTransformTextGrpcBindingFactory)?;
    builder.register(RivaPunctuateTextGrpcBindingFactory)?;
    builder.register(RivaNaturalQueryGrpcBindingFactory)?;
    builder.register(RivaAnalyzeIntentGrpcBindingFactory)?;
    builder.register(RivaAnalyzeEntitiesGrpcBindingFactory)?;
    Ok(())
}
