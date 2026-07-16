// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::endpoints::{
    CreditPhase, EndpointId, EndpointRegistry, RawEndpointConfig, ResponseData, ServerResponse,
    Turn,
};
use serde_json::{Value, json};

/// Materialize a prepared endpoint's [`BodyPlan`] into a decoded JSON value so
/// the structural assertions below keep comparing against `json!` objects.
fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn prepared() -> Box<dyn aiperf_runtime::endpoints::PreparedEndpoint> {
    EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new("vllm_generate").unwrap(),
            RawEndpointConfig::default(),
        )
        .unwrap()
}

#[test]
fn descriptor_and_payload_are_token_native() {
    let endpoint = prepared();
    let descriptor = endpoint.descriptor();
    assert_eq!(descriptor.endpoint_path, Some("/inference/v1/generate"));
    assert!(!descriptor.supports_streaming);
    assert!(descriptor.produces_tokens);
    assert!(!descriptor.tokenizes_input);
    assert!(descriptor.requires_raw_token_ids);

    let turns = [Turn {
        model: Some("turn-model".into()),
        max_tokens: Some(17),
        raw_token_ids: Some(vec![1, 2, 3]),
        extra_body: Some(
            json!({
                "sampling_params": {"temperature": 0},
                "priority": 4
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
        ..Turn::default()
    }];
    let request = aiperf_runtime::endpoints::PreparedRequest::new(
        "default-model",
        &turns,
        None,
        None,
        CreditPhase::Profiling,
        Some("req-1"),
        None,
        Some("session-1"),
    );

    assert_eq!(
        plan_body(endpoint.format_payload(&request).unwrap()),
        json!({
            "model": "turn-model",
            "token_ids": [1, 2, 3],
            "sampling_params": {"temperature": 0, "max_tokens": 17},
            "stream": false,
            "request_id": "req-1",
            "priority": 4
        })
    );
}

#[test]
fn formatting_rejects_missing_ids_and_streaming_override() {
    let endpoint = prepared();
    for turn in [
        Turn::default(),
        Turn {
            raw_token_ids: Some(vec![1]),
            extra_body: Some(json!({"stream": true}).as_object().unwrap().clone()),
            ..Turn::default()
        },
    ] {
        let turns = [turn];
        let request = aiperf_runtime::endpoints::PreparedRequest::new(
            "model",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );
        assert!(endpoint.format_payload(&request).is_err());
    }
}

#[test]
fn preparation_rejects_malformed_endpoint_sampling_params() {
    let registry = EndpointRegistry::builtin().unwrap();
    let result = registry.prepare(
        &EndpointId::new("vllm_generate").unwrap(),
        RawEndpointConfig {
            extra: Some(json!({"sampling_params": []}).as_object().unwrap().clone()),
            ..RawEndpointConfig::default()
        },
    );

    assert!(result.is_err());
}

#[test]
fn response_retains_exact_ids_and_counts_the_array() {
    let endpoint = prepared();
    let parsed = endpoint
        .parse_response(&ServerResponse::from_json(
            123,
            json!({
                "request_id": "req-1",
                "choices": [{"token_ids": [20, 21], "finish_reason": "stop"}]
            }),
        ))
        .unwrap()
        .unwrap();
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
