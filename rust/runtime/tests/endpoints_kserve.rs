// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KServe endpoint wire and behavior contracts.

use std::collections::BTreeSet;

use aiperf_runtime::endpoints::{
    CreditPhase, EndpointId, EndpointRegistry, EndpointRegistryError, ImageResponseData, Media,
    PreparedEndpoint, PreparedRequest, RawEndpointConfig, ReadinessPolicy, ResponseData,
    ServerResponse, Turn,
};
use serde_json::{Map, Value, json};

fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn prepared(id: &str, extra: Option<Map<String, Value>>) -> Box<dyn PreparedEndpoint> {
    let registry = EndpointRegistry::builtin().unwrap();
    registry
        .prepare(
            &EndpointId::new(id).unwrap(),
            RawEndpointConfig {
                urls: vec!["http://127.0.0.1:8000".to_string()],
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

fn text_turn(contents: &[&str], max_tokens: Option<u32>) -> Turn {
    Turn {
        max_tokens,
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
fn all_kserve_dialects_are_open_registry_only() {
    let registry = EndpointRegistry::builtin().unwrap();
    let expected = BTreeSet::from([
        "kserve_chat",
        "kserve_completions",
        "kserve_embeddings",
        "kserve_v1_predict",
        "kserve_v2_embeddings",
        "kserve_v2_images",
        "kserve_v2_infer",
        "kserve_v2_rankings",
        "kserve_v2_vlm",
    ]);
    let present = registry
        .canonical_ids()
        .map(EndpointId::as_str)
        .filter(|id| id.starts_with("kserve_"))
        .collect::<BTreeSet<_>>();
    assert_eq!(present, expected);

    for id in expected {
        assert!(matches!(
            registry.legacy_endpoint(&EndpointId::new(id).unwrap()),
            Err(EndpointRegistryError::NoLegacyAdapter(_))
        ));
    }
}

#[test]
fn openai_compatible_kserve_factories_override_paths_without_v1_adapters() {
    let cases = [
        ("kserve_chat", "/openai/v1/chat/completions"),
        ("kserve_completions", "/openai/v1/completions"),
        ("kserve_embeddings", "/openai/v1/embeddings"),
    ];
    for (id, path) in cases {
        let endpoint = prepared(id, None);
        assert_eq!(endpoint.descriptor().endpoint_path, Some(path));
        assert_eq!(endpoint.descriptor().service_kind, "kserve");
        assert!(matches!(
            endpoint.readiness_policy("model").unwrap(),
            ReadinessPolicy::Request(request) if request.path == "/openai/v1/models"
        ));
    }
}

#[test]
fn kserve_v1_predict_matches_instances_predictions_and_autodetection() {
    let endpoint = prepared(
        "kserve_v1_predict",
        Some(
            json!({"v1_input_field": "sentence", "v1_output_field": "answer"})
                .as_object()
                .unwrap()
                .clone(),
        ),
    );
    let turns = [text_turn(&["hello", "", "world"], None)];
    assert_eq!(
        plan_body(
            endpoint
                .format_payload(&request("classifier", &turns))
                .unwrap(),
        ),
        json!({"instances": [{"sentence": "hello world"}]})
    );
    assert!(matches!(
        endpoint.readiness_policy("classifier").unwrap(),
        ReadinessPolicy::Request(request) if request.path == "/v1/models/classifier"
    ));

    let parsed = endpoint
        .parse_response(&ServerResponse::from_json(
            10,
            json!({"predictions": [{"answer": "yes"}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Text {
            text: "yes".to_string()
        })
    );
    let fallback = endpoint
        .parse_response(&ServerResponse::from_json(
            11,
            json!({"predictions": [{"embedding": [1, 2.5]}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        fallback.data,
        Some(ResponseData::Embeddings {
            embeddings: vec![vec![1.0, 2.5]]
        })
    );
}

#[test]
fn kserve_v2_infer_and_vlm_match_tensor_payloads_and_text_fallback() {
    let infer = prepared(
        "kserve_v2_infer",
        Some(
            json!({"v2_input_name": "prompt", "v2_output_name": "answer", "temperature": 0.2})
                .as_object()
                .unwrap()
                .clone(),
        ),
    );
    let turns = [text_turn(&["one", "two"], Some(17))];
    let expected = json!({
        "inputs": [
            {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["one two"]},
            {"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [17]},
        ],
        "parameters": {"temperature": 0.2},
    });
    let bytes = infer
        .format_payload(&request("llm", &turns))
        .unwrap()
        .materialize_standalone()
        .unwrap();
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
    let parsed = infer
        .parse_response(&ServerResponse::from_json(
            20,
            json!({"outputs": [
                {"name": "other", "data": ["fallback"]},
                {"name": "answer", "data": [42]},
            ]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Text {
            text: "42".to_string()
        })
    );

    let vlm = prepared("kserve_v2_vlm", None);
    let turns = [Turn {
        max_tokens: Some(8),
        texts: vec![Media::new(vec!["describe".to_string()])],
        images: vec![Media::new(vec![
            "image-a".to_string(),
            "image-b".to_string(),
        ])],
        ..Turn::default()
    }];
    assert_eq!(
        plan_body(vlm.format_payload(&request("vlm", &turns)).unwrap()),
        json!({"inputs": [
            {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["describe"]},
            {"name": "image", "shape": [2], "datatype": "BYTES", "data": ["image-a", "image-b"]},
            {"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [8]},
        ]})
    );
}

#[test]
fn kserve_v2_embeddings_and_rankings_preserve_shape_and_numeric_rules() {
    let embeddings = prepared("kserve_v2_embeddings", None);
    let turns = [text_turn(&["a", "b"], Some(4))];
    assert_eq!(
        plan_body(
            embeddings
                .format_payload(&request("embed", &turns))
                .unwrap(),
        ),
        json!({"inputs": [{
            "name": "text_input", "shape": [2], "datatype": "BYTES", "data": ["a", "b"]
        }]})
    );
    let parsed = embeddings
        .parse_response(&ServerResponse::from_json(
            30,
            json!({"outputs": [{
                "name": "embedding_output", "shape": [2, 3],
                "data": [1, 2, 3, 4, 5, 6]
            }]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Embeddings {
            embeddings: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        })
    );

    let rankings = prepared("kserve_v2_rankings", None);
    let turns = [Turn {
        texts: vec![
            Media {
                name: "queries".to_string(),
                contents: vec!["question".to_string(), "ignored".to_string()],
            },
            Media {
                name: "passages".to_string(),
                contents: vec!["p0".to_string(), "p1".to_string()],
            },
        ],
        ..Turn::default()
    }];
    assert_eq!(
        plan_body(
            rankings
                .format_payload(&request("reranker", &turns))
                .unwrap(),
        ),
        json!({"inputs": [
            {"name": "query", "shape": [1], "datatype": "BYTES", "data": ["question"]},
            {"name": "passages", "shape": [2], "datatype": "BYTES", "data": ["p0", "p1"]},
        ]})
    );
    let parsed = rankings
        .parse_response(&ServerResponse::from_json(
            31,
            json!({"outputs": [{"name": "scores", "data": [0.9, "bad", "0.3"]}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Rankings {
            rankings: vec![
                json!({"index": 0, "score": 0.9}),
                json!({"index": 2, "score": 0.3}),
            ]
        })
    );
}

#[test]
fn kserve_v2_images_separates_typed_tensors_and_generic_parameters() {
    let endpoint = prepared(
        "kserve_v2_images",
        Some(
            json!({
                "negative_prompt": "blurred",
                "num_inference_steps": "25",
                "guidance_scale": 7,
                "seed": 123,
                "scheduler": "euler",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    );
    let turns = [text_turn(&["a", "cat"], Some(99))];
    assert_eq!(
        plan_body(
            endpoint
                .format_payload(&request("diffusion", &turns))
                .unwrap(),
        ),
        json!({
            "inputs": [
                {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["a cat"]},
                {"name": "negative_prompt", "shape": [1], "datatype": "BYTES", "data": ["blurred"]},
                {"name": "num_inference_steps", "shape": [1], "datatype": "INT32", "data": [25]},
                {"name": "guidance_scale", "shape": [1], "datatype": "FP32", "data": [7.0]},
                {"name": "seed", "shape": [1], "datatype": "INT64", "data": [123]},
            ],
            "parameters": {"scheduler": "euler"},
        })
    );
    let parsed = endpoint
        .parse_response(&ServerResponse::from_json(
            40,
            json!({"outputs": [{"name": "generated_image", "data": ["YWJj", null]}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Images(ImageResponseData {
            images: vec![aiperf_runtime::endpoints::ImageDataItem {
                b64_json: Some("YWJj".to_string()),
                ..aiperf_runtime::endpoints::ImageDataItem::default()
            }],
            ..ImageResponseData::default()
        }))
    );
}

#[test]
fn identity_free_config_accepts_http_and_grpc_schemes() {
    let registry = EndpointRegistry::builtin().unwrap();
    for url in [
        "http://localhost:8000",
        "https://localhost:8443",
        "grpc://localhost:8001",
        "grpcs://localhost:8444",
    ] {
        registry
            .prepare(
                &EndpointId::new("kserve_v2_infer").unwrap(),
                RawEndpointConfig {
                    urls: vec![url.to_string()],
                    ..RawEndpointConfig::default()
                },
            )
            .unwrap();
    }
}
