// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native endpoint parity tests for NVIDIA Riva ASR, TTS, and NLP.

use std::collections::BTreeSet;

use aiperf_runtime::endpoints::{
    AudioResponseData, CreditPhase, EndpointId, EndpointRegistry, EndpointRegistryError, Media,
    PreparedEndpoint, PreparedRequest, RawEndpointConfig, ReadinessPolicy, ResponseData,
    ServerResponse, Turn,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value, json};

/// Materialize a prepared endpoint's [`BodyPlan`] into a decoded JSON value so
/// the structural assertions below keep comparing against `json!` objects.
fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn prepared(
    id: &str,
    streaming: bool,
    extra: Option<Map<String, Value>>,
) -> Box<dyn PreparedEndpoint> {
    EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new(id).unwrap(),
            RawEndpointConfig {
                urls: vec!["grpc://127.0.0.1:50051".to_string()],
                streaming,
                extra,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap()
}

fn request<'a>(model: &'a str, turns: &'a [Turn]) -> PreparedRequest<'a> {
    PreparedRequest::new(
        model,
        turns,
        None,
        None,
        CreditPhase::Profiling,
        Some("request-id"),
        Some("correlation-id"),
        Some("conversation-id"),
    )
}

fn text_turn(contents: &[&str]) -> Turn {
    Turn {
        texts: vec![Media::new(
            contents
                .iter()
                .map(|content| (*content).to_string())
                .collect::<Vec<_>>(),
        )],
        ..Turn::default()
    }
}

#[test]
fn all_nine_riva_endpoints_are_open_protocol_v2_dialects() {
    let registry = EndpointRegistry::builtin().unwrap();
    let expected = BTreeSet::from([
        "riva_analyze_entities",
        "riva_analyze_intent",
        "riva_asr",
        "riva_natural_query",
        "riva_punctuate_text",
        "riva_text_classify",
        "riva_token_classify",
        "riva_transform_text",
        "riva_tts",
    ]);
    let present = registry
        .canonical_ids()
        .map(EndpointId::as_str)
        .filter(|id| id.starts_with("riva_"))
        .collect::<BTreeSet<_>>();
    assert_eq!(present, expected);

    for id in expected {
        let endpoint = prepared(id, false, None);
        assert_eq!(endpoint.descriptor().service_kind, "riva");
        assert!(!endpoint.descriptor().produces_tokens);
        assert!(matches!(
            endpoint.readiness_policy("model").unwrap(),
            ReadinessPolicy::Unsupported { reason }
                if reason == "Riva endpoints do not define a model-readiness RPC"
        ));
        assert!(matches!(
            registry.legacy_endpoint(&EndpointId::new(id).unwrap()),
            Err(EndpointRegistryError::NoLegacyAdapter(_))
        ));
    }

    assert!(
        prepared("riva_asr", false, None)
            .descriptor()
            .supports_streaming
    );
    assert!(
        prepared("riva_tts", false, None)
            .descriptor()
            .supports_streaming
    );
    assert!(
        !prepared("riva_text_classify", false, None)
            .descriptor()
            .supports_streaming
    );
}

#[test]
fn asr_formats_unary_and_configured_chunked_audio_and_parses_transcripts() {
    let audio = [0_u8, 1, 2, 3, 4];
    let turns = [Turn {
        audios: vec![Media::new(vec![STANDARD.encode(audio)])],
        ..Turn::default()
    }];
    let unary = prepared("riva_asr", false, None);
    assert_eq!(
        plan_body(unary.format_payload(&request("asr", &turns)).unwrap()),
        json!({
            "audio": STANDARD.encode(audio),
            "language_code": "en-US",
            "sample_rate_hertz": 16000,
            "encoding": "LINEAR_PCM",
        })
    );
    assert_eq!(
        unary
            .extract_payload_inputs(&json!({"audio": "ignored"}))
            .audio_count,
        1
    );

    let streaming = prepared(
        "riva_asr",
        true,
        Some(
            json!({
                "language_code": "de-DE",
                "sample_rate_hertz": "8000",
                "encoding": "FLAC",
                "chunk_size": 2,
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    );
    assert_eq!(
        plan_body(streaming.format_payload(&request("asr", &turns)).unwrap()),
        json!({
            "language_code": "de-DE",
            "sample_rate_hertz": 8000,
            "encoding": "FLAC",
            "interim_results": true,
            "audio_chunks": ["AAE=", "AgM=", "BA=="],
        })
    );

    let parsed = streaming
        .parse_response(&ServerResponse::from_json(
            17,
            json!({"transcript": "hello world", "is_final": true}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(parsed.perf_ns, 17);
    assert_eq!(
        parsed.data,
        Some(ResponseData::Text {
            text: "hello world".to_string()
        })
    );
    assert!(
        streaming
            .parse_response(&ServerResponse::from_json(18, json!({"transcript": ""})))
            .unwrap()
            .is_none()
    );
}

#[test]
fn asr_rejects_missing_audio_and_non_positive_chunk_sizes() {
    let asr = prepared("riva_asr", false, None);
    assert!(asr.format_payload(&request("asr", &[])).is_err());
    assert!(
        asr.format_payload(&request("asr", &[Turn::default()]))
            .is_err()
    );

    let registry = EndpointRegistry::builtin().unwrap();
    let error = registry
        .prepare(
            &EndpointId::new("riva_asr").unwrap(),
            RawEndpointConfig {
                urls: vec!["grpc://127.0.0.1:50051".to_string()],
                streaming: true,
                extra: json!({"chunk_size": 0}).as_object().cloned(),
                ..RawEndpointConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("positive integer"));
}

#[test]
fn tts_joins_first_turn_text_and_preserves_audio_geometry() {
    let endpoint = prepared(
        "riva_tts",
        true,
        Some(
            json!({
                "voice_name": "English-US.Female-1",
                "language_code": "en-US",
                "encoding": "LINEAR_PCM",
                "sample_rate_hz": 8000,
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    );
    let turns = [Turn {
        texts: vec![
            Media::new(vec!["hello".to_string(), String::new()]),
            Media::new(vec!["world".to_string()]),
        ],
        ..Turn::default()
    }];
    let payload = plan_body(
        endpoint
            .format_payload(&request("unused-by-riva-tts", &turns))
            .unwrap(),
    );
    assert_eq!(
        payload,
        json!({
            "text": "hello world",
            "voice_name": "English-US.Female-1",
            "language_code": "en-US",
            "encoding": "LINEAR_PCM",
            "sample_rate_hz": 8000,
        })
    );
    // Golden-byte gate (stage B): materialized Riva plan bytes are identical to
    // `to_vec` of the equivalent hand-built object.
    let expected = json!({
        "text": "hello world",
        "voice_name": "English-US.Female-1",
        "language_code": "en-US",
        "encoding": "LINEAR_PCM",
        "sample_rate_hz": 8000,
    });
    let bytes = endpoint
        .format_payload(&request("unused-by-riva-tts", &turns))
        .unwrap()
        .materialize_standalone()
        .unwrap();
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );
    assert_eq!(
        endpoint.extract_payload_inputs(&payload).texts,
        ["hello world"]
    );

    let audio = vec![1_u8; 16];
    let parsed = endpoint
        .parse_response(&ServerResponse::from_json(
            23,
            json!({"audio": STANDARD.encode(&audio)}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Audio(AudioResponseData {
            audio_bytes: audio,
            sample_rate_hz: 8000,
            encoding: "LINEAR_PCM".to_string(),
            duration_ms: Some(1.0),
        }))
    );
}

#[test]
fn nlp_endpoints_match_text_list_query_and_response_rules() {
    let turns = [text_turn(&["hello", "", "world"])];
    for id in [
        "riva_text_classify",
        "riva_token_classify",
        "riva_transform_text",
        "riva_punctuate_text",
    ] {
        let endpoint = prepared(
            id,
            false,
            json!({"language_code": "es-ES"}).as_object().cloned(),
        );
        assert_eq!(
            plan_body(endpoint.format_payload(&request("nlp", &turns)).unwrap()),
            json!({"texts": ["hello", "world"], "language_code": "es-ES"})
        );
    }

    let classify = prepared("riva_text_classify", false, None);
    let parsed = classify
        .parse_response(&ServerResponse::from_json(
            31,
            json!({"results": [{"labels": [{"class_name": "positive", "score": 0.9}]}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Text {
            text: "{\"results\":[{\"labels\":[{\"class_name\":\"positive\",\"score\":0.9}]}]}"
                .to_string(),
        })
    );

    let transform = prepared("riva_transform_text", false, None);
    assert_eq!(
        transform
            .parse_response(&ServerResponse::from_json(
                32,
                json!({"texts": ["Hello,", "world!"]}),
            ))
            .unwrap()
            .unwrap()
            .data,
        Some(ResponseData::Text {
            text: "Hello, world!".to_string()
        })
    );

    let natural = prepared(
        "riva_natural_query",
        false,
        json!({"context": "reference document", "top_n": "2"})
            .as_object()
            .cloned(),
    );
    assert_eq!(
        plan_body(natural.format_payload(&request("qa", &turns)).unwrap()),
        json!({"query": "hello world", "context": "reference document", "top_n": 2})
    );
    assert_eq!(
        natural
            .parse_response(&ServerResponse::from_json(
                33,
                json!({"results": [{"answer": "first", "score": 0.8}, {"answer": "second"}]}),
            ))
            .unwrap()
            .unwrap()
            .data,
        Some(ResponseData::Text {
            text: "first".to_string()
        })
    );
    assert_eq!(
        natural
            .parse_response(&ServerResponse::from_json(34, json!({"results": [{}]})))
            .unwrap()
            .unwrap()
            .data,
        Some(ResponseData::Text {
            text: String::new()
        })
    );

    let intent = prepared(
        "riva_analyze_intent",
        false,
        json!({"domain": "weather"}).as_object().cloned(),
    );
    assert_eq!(
        plan_body(intent.format_payload(&request("intent", &turns)).unwrap()),
        json!({"query": "hello world", "domain": "weather"})
    );
    let entities = prepared("riva_analyze_entities", false, None);
    assert_eq!(
        plan_body(entities.format_payload(&request("ner", &turns)).unwrap()),
        json!({"query": "hello world"})
    );
}

#[test]
fn natural_query_accepts_protobuf_zero_top_n_but_rejects_negative_values() {
    let endpoint = prepared(
        "riva_natural_query",
        false,
        json!({"context": "document", "top_n": 0})
            .as_object()
            .cloned(),
    );
    let turns = [text_turn(&["question"])];
    assert_eq!(
        plan_body(endpoint.format_payload(&request("qa", &turns)).unwrap()),
        json!({"query": "question", "context": "document", "top_n": 0})
    );

    let error = EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new("riva_natural_query").unwrap(),
            RawEndpointConfig {
                extra: Some(json!({"top_n": -1}).as_object().unwrap().clone()),
                ..RawEndpointConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("non-negative u32"));
}
