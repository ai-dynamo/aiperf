// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact NVIDIA Riva protobuf and RPC binding parity tests.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use aiperf_runtime::endpoints::EndpointId;
use aiperf_runtime::transport_grpc::riva_proto::streaming_recognize_request::StreamingRequest;
use aiperf_runtime::transport_grpc::riva_proto::{
    AnalyzeEntitiesRequest, AnalyzeIntentRequest, AnalyzeIntentResponse, AudioEncoding,
    Classification, ClassificationResult, NaturalQueryRequest, NaturalQueryResponse,
    NaturalQueryResult, RecognizeRequest, RecognizeResponse, SpeechRecognitionAlternative,
    SpeechRecognitionResult, StreamingRecognitionResult, StreamingRecognizeRequest,
    StreamingRecognizeResponse, SynthesizeSpeechRequest, SynthesizeSpeechResponse,
    SynthesizeSpeechResponseMetadata, TextClassRequest, TextClassResponse, TextTransformRequest,
    TextTransformResponse, TokenClassRequest, TokenClassResponse, TokenClassSequence,
    TokenClassValue,
};
use aiperf_runtime::transport_grpc::{GrpcBindingRegistry, GrpcEndpointBinding};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use prost::Message;
use serde_json::json;

fn binding(id: &str) -> Box<dyn GrpcEndpointBinding> {
    GrpcBindingRegistry::builtin()
        .unwrap()
        .prepare(&EndpointId::new(id).unwrap())
        .unwrap()
}

fn wire_hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").unwrap();
        output
    })
}

#[test]
fn all_riva_bindings_use_the_exact_service_paths_and_cardinalities() {
    let cases = BTreeMap::from([
        (
            "riva_analyze_entities",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeEntities",
        ),
        (
            "riva_analyze_intent",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/AnalyzeIntent",
        ),
        (
            "riva_asr",
            "/nvidia.riva.asr.RivaSpeechRecognition/Recognize",
        ),
        (
            "riva_natural_query",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/NaturalQuery",
        ),
        (
            "riva_punctuate_text",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/PunctuateText",
        ),
        (
            "riva_text_classify",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyText",
        ),
        (
            "riva_token_classify",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/ClassifyTokens",
        ),
        (
            "riva_transform_text",
            "/nvidia.riva.nlp.RivaLanguageUnderstanding/TransformText",
        ),
        (
            "riva_tts",
            "/nvidia.riva.tts.RivaSpeechSynthesis/Synthesize",
        ),
    ]);
    for (id, unary_path) in cases {
        let binding = binding(id);
        assert_eq!(binding.unary_method().as_str(), unary_path);
        assert!(binding.readiness_method().is_none());
        match id {
            "riva_asr" => {
                assert!(binding.streaming_method().is_none());
                assert_eq!(
                    binding.bidi_streaming_method().unwrap().as_str(),
                    "/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize"
                );
            }
            "riva_tts" => {
                assert_eq!(
                    binding.streaming_method().unwrap().as_str(),
                    "/nvidia.riva.tts.RivaSpeechSynthesis/SynthesizeOnline"
                );
                assert!(binding.bidi_streaming_method().is_none());
            }
            _ => {
                assert!(binding.streaming_method().is_none());
                assert!(binding.bidi_streaming_method().is_none());
            }
        }
    }
}

#[test]
fn asr_unary_codec_matches_reference_defaults_overrides_and_response_shape() {
    // Python parity: riva_asr_serializers.py:20-117,147-181.
    let binding = binding("riva_asr");
    let encoded = binding
        .encode_request(
            &json!({
                "audio": STANDARD.encode(b"abc"),
                "language_code": "fr-FR",
                "sample_rate_hertz": 8000,
                "encoding": "FLAC",
                "max_alternatives": 2,
                "enable_automatic_punctuation": false,
                "model": "explicit-model",
            }),
            "fallback-model",
            "request-1",
        )
        .unwrap();
    // Serialized by the reference Python serializer at commit a391cfe27.
    assert_eq!(
        wire_hex(&encoded),
        "0a1e080210c03e1a0566722d465220026a0e6578706c696369742d6d6f64656c1203616263a2060b0a09726571756573742d31"
    );
    let request = RecognizeRequest::decode(encoded).unwrap();
    assert_eq!(request.audio, b"abc");
    assert_eq!(request.id.unwrap().value, "request-1");
    let config = request.config.unwrap();
    assert_eq!(config.encoding, AudioEncoding::Flac as i32);
    assert_eq!(config.sample_rate_hertz, 8000);
    assert_eq!(config.language_code, "fr-FR");
    assert_eq!(config.max_alternatives, 2);
    assert!(!config.enable_automatic_punctuation);
    assert_eq!(config.model, "explicit-model");

    let response = RecognizeResponse {
        results: vec![
            SpeechRecognitionResult {
                alternatives: vec![
                    SpeechRecognitionAlternative {
                        transcript: "hello".to_string(),
                        confidence: 0.75,
                    },
                    SpeechRecognitionAlternative {
                        transcript: "ignored".to_string(),
                        confidence: 0.5,
                    },
                ],
            },
            SpeechRecognitionResult {
                alternatives: vec![SpeechRecognitionAlternative {
                    transcript: "world".to_string(),
                    confidence: 1.0,
                }],
            },
        ],
        id: None,
    };
    assert_eq!(
        binding.decode_response(&response.encode_to_vec()).unwrap(),
        json!({
            "transcript": "hello world",
            "results": [
                {"alternatives": [
                    {"transcript": "hello", "confidence": 0.75},
                    {"transcript": "ignored", "confidence": 0.5},
                ]},
                {"alternatives": [{"transcript": "world", "confidence": 1.0}]},
            ],
        })
    );
}

#[test]
fn asr_bidi_codec_emits_config_first_then_audio_and_decodes_stream_chunks() {
    // Python parity: riva_asr_serializers.py:119-145,183-224.
    let binding = binding("riva_asr");
    let messages = binding
        .encode_bidi_requests(
            &json!({
                "language_code": "en-US",
                "sample_rate_hertz": 16000,
                "encoding": "LINEAR_PCM",
                "interim_results": true,
                "audio_chunks": [STANDARD.encode([1_u8, 2]), STANDARD.encode([3_u8])],
            }),
            "asr-model",
            "stream-id",
        )
        .unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        wire_hex(&messages[0]),
        "0a1f0a1b080110807d1a05656e2d5553200158016a096173722d6d6f64656c1001a2060b0a0973747265616d2d6964"
    );

    let config_message = StreamingRecognizeRequest::decode(messages[0].clone()).unwrap();
    assert_eq!(config_message.id.unwrap().value, "stream-id");
    let Some(StreamingRequest::StreamingConfig(streaming)) = config_message.streaming_request
    else {
        panic!("first ASR stream message was not configuration")
    };
    assert!(streaming.interim_results);
    let recognition = streaming.config.unwrap();
    assert_eq!(recognition.model, "asr-model");
    assert_eq!(recognition.encoding, AudioEncoding::LinearPcm as i32);

    for (message, expected) in messages[1..].iter().zip([&[1_u8, 2][..], &[3_u8][..]]) {
        let message = StreamingRecognizeRequest::decode(message.clone()).unwrap();
        assert!(message.id.is_none());
        assert!(matches!(
            message.streaming_request,
            Some(StreamingRequest::AudioContent(value)) if value == expected
        ));
    }

    let response = StreamingRecognizeResponse {
        results: vec![StreamingRecognitionResult {
            alternatives: vec![SpeechRecognitionAlternative {
                transcript: "partial".to_string(),
                confidence: 0.25,
            }],
            is_final: true,
            stability: 0.8,
        }],
        id: None,
    };
    let wire = response.encode_to_vec();
    let chunk = binding.decode_stream_response(&wire).unwrap();
    assert_eq!(chunk.response_size, wire.len());
    assert_eq!(chunk.error_message, None);
    assert_eq!(
        chunk.response,
        Some(json!({
            "transcript": "partial",
            "is_final": true,
            "results": [{
                "alternatives": [{"transcript": "partial", "confidence": 0.25}],
                "is_final": true,
                "stability": 0.800000011920929,
            }],
        }))
    );
}

#[test]
fn tts_codec_supports_unary_and_server_streaming_audio() {
    // Python parity: riva_tts_serializers.py:20-94.
    let binding = binding("riva_tts");
    let encoded = binding
        .encode_request(
            &json!({
                "text": "speak this",
                "language_code": "de-DE",
                "encoding": "ALAW",
                "sample_rate_hz": 44100,
                "voice_name": "voice-a",
            }),
            "ignored-model",
            "tts-id",
        )
        .unwrap();
    assert_eq!(
        wire_hex(&encoded),
        "0a0a737065616b2074686973120564652d4445181420c4d8022a07766f6963652d61a206080a067474732d6964"
    );
    let request = SynthesizeSpeechRequest::decode(encoded).unwrap();
    assert_eq!(request.text, "speak this");
    assert_eq!(request.language_code, "de-DE");
    assert_eq!(request.encoding, AudioEncoding::Alaw as i32);
    assert_eq!(request.sample_rate_hz, 44100);
    assert_eq!(request.voice_name, "voice-a");
    assert_eq!(request.id.unwrap().value, "tts-id");

    let response = SynthesizeSpeechResponse {
        audio: vec![4, 5, 6],
        meta: Some(SynthesizeSpeechResponseMetadata {
            text: "speak this".to_string(),
            processed_text: "Speak this.".to_string(),
            predicted_durations: vec![0.1],
        }),
        id: None,
    };
    let wire = response.encode_to_vec();
    let expected = json!({
        "audio": STANDARD.encode([4_u8, 5, 6]),
        "meta": {"text": "speak this", "processed_text": "Speak this."},
    });
    assert_eq!(binding.decode_response(&wire).unwrap(), expected);
    let chunk = binding.decode_stream_response(&wire).unwrap();
    assert_eq!(chunk.response, Some(expected));
    assert_eq!(chunk.response_size, wire.len());
}

#[test]
fn all_nlp_request_codecs_preserve_reference_fields() {
    // Python parity: riva_nlp_serializers.py:38-271.
    let payload = json!({
        "texts": ["one", "two"],
        "top_n": 3,
        "model_name": "explicit-nlp",
        "language_code": "ja-JP",
    });
    let text_wire = binding("riva_text_classify")
        .encode_request(&payload, "fallback", "text-id")
        .unwrap();
    assert_eq!(
        wire_hex(&text_wire),
        "0a036f6e650a0374776f10031a150a0c6578706c696369742d6e6c701a056a612d4a50a206090a07746578742d6964"
    );
    let text = TextClassRequest::decode(text_wire).unwrap();
    assert_eq!(text.text, ["one", "two"]);
    assert_eq!(text.top_n, 3);
    assert_eq!(text.model.unwrap().model_name, "explicit-nlp");
    assert_eq!(text.id.unwrap().value, "text-id");

    let token = TokenClassRequest::decode(
        binding("riva_token_classify")
            .encode_request(&payload, "fallback", "token-id")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(token.text, ["one", "two"]);
    assert_eq!(token.top_n, 3);
    assert_eq!(token.model.unwrap().language_code, "ja-JP");

    for id in ["riva_transform_text", "riva_punctuate_text"] {
        let request = TextTransformRequest::decode(
            binding(id)
                .encode_request(&payload, "fallback", "transform-id")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(request.text, ["one", "two"]);
        assert_eq!(request.top_n, 3);
        assert_eq!(request.id.unwrap().value, "transform-id");
    }

    let natural = NaturalQueryRequest::decode(
        binding("riva_natural_query")
            .encode_request(
                &json!({"query": "question", "context": "document", "top_n": 2}),
                "ignored",
                "natural-id",
            )
            .unwrap(),
    )
    .unwrap();
    assert_eq!(natural.query, "question");
    assert_eq!(natural.context, "document");
    assert_eq!(natural.top_n, 2);
    assert_eq!(natural.id.unwrap().value, "natural-id");

    let intent = AnalyzeIntentRequest::decode(
        binding("riva_analyze_intent")
            .encode_request(
                &json!({"query": "weather tomorrow", "domain": "weather"}),
                "ignored",
                "intent-id",
            )
            .unwrap(),
    )
    .unwrap();
    assert_eq!(intent.query, "weather tomorrow");
    assert_eq!(intent.options.unwrap().domain, "weather");

    let entities = AnalyzeEntitiesRequest::decode(
        binding("riva_analyze_entities")
            .encode_request(&json!({"query": "NVIDIA"}), "ignored", "entity-id")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(entities.query, "NVIDIA");
    assert!(entities.options.is_none());
}

#[test]
fn all_nlp_response_codecs_match_the_reference_json_contract() {
    let label = Classification {
        class_name: "positive".to_string(),
        score: 0.75,
    };
    let classify = TextClassResponse {
        results: vec![ClassificationResult {
            labels: vec![label.clone()],
        }],
        id: None,
    };
    assert_eq!(
        binding("riva_text_classify")
            .decode_response(&classify.encode_to_vec())
            .unwrap(),
        json!({"results": [{"labels": [{"class_name": "positive", "score": 0.75}]}]})
    );

    let token = TokenClassResponse {
        results: vec![TokenClassSequence {
            results: vec![TokenClassValue {
                token: "NVIDIA".to_string(),
                label: vec![label.clone()],
                span: Vec::new(),
            }],
        }],
        id: None,
    };
    let expected_tokens = json!({"results": [{"tokens": [{
        "token": "NVIDIA",
        "labels": [{"class_name": "positive", "score": 0.75}],
    }]}]});
    for id in ["riva_token_classify", "riva_analyze_entities"] {
        assert_eq!(
            binding(id).decode_response(&token.encode_to_vec()).unwrap(),
            expected_tokens
        );
    }

    let transformed = TextTransformResponse {
        text: vec!["Hello.".to_string(), "World!".to_string()],
        id: None,
    };
    for id in ["riva_transform_text", "riva_punctuate_text"] {
        assert_eq!(
            binding(id)
                .decode_response(&transformed.encode_to_vec())
                .unwrap(),
            json!({"texts": ["Hello.", "World!"]})
        );
    }

    let natural = NaturalQueryResponse {
        results: vec![NaturalQueryResult {
            answer: "answer".to_string(),
            score: 0.5,
        }],
        id: None,
    };
    assert_eq!(
        binding("riva_natural_query")
            .decode_response(&natural.encode_to_vec())
            .unwrap(),
        json!({"results": [{"answer": "answer", "score": 0.5}]})
    );

    let intent = AnalyzeIntentResponse {
        intent: Some(label.clone()),
        slots: vec![TokenClassValue {
            token: "tomorrow".to_string(),
            label: vec![label],
            span: Vec::new(),
        }],
        domain_str: String::new(),
        domain: Some(Classification {
            class_name: "weather".to_string(),
            score: 1.0,
        }),
        id: None,
    };
    assert_eq!(
        binding("riva_analyze_intent")
            .decode_response(&intent.encode_to_vec())
            .unwrap(),
        json!({
            "intent": {"class_name": "positive", "score": 0.75},
            "slots": [{
                "token": "tomorrow",
                "labels": [{"class_name": "positive", "score": 0.75}],
            }],
            "domain": {"class_name": "weather", "score": 1.0},
        })
    );
}

#[test]
fn unsupported_audio_encoding_falls_back_to_linear_pcm() {
    for id in ["riva_asr", "riva_tts"] {
        let payload = if id == "riva_asr" {
            json!({"audio": "", "encoding": "UNKNOWN"})
        } else {
            json!({"text": "hello", "encoding": "UNKNOWN"})
        };
        let bytes = binding(id).encode_request(&payload, "model", "").unwrap();
        let encoding = if id == "riva_asr" {
            RecognizeRequest::decode(bytes)
                .unwrap()
                .config
                .unwrap()
                .encoding
        } else {
            SynthesizeSpeechRequest::decode(bytes).unwrap().encoding
        };
        assert_eq!(encoding, AudioEncoding::LinearPcm as i32);
    }
}
