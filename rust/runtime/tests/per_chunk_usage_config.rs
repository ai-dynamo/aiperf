// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-policy validation for cumulative streaming usage.

use aiperf_runtime::endpoints::{EndpointConfig, EndpointType, RawEndpointConfig};

fn config(endpoint_type: EndpointType) -> EndpointConfig {
    EndpointConfig::from_raw(
        endpoint_type,
        RawEndpointConfig {
            urls: vec!["http://127.0.0.1:8000".to_string()],
            streaming: true,
            use_server_token_count: true,
            per_chunk_usage: true,
            ..RawEndpointConfig::default()
        },
    )
}

#[test]
fn per_chunk_usage_requires_server_token_count() {
    let mut endpoint = config(EndpointType::Chat);
    endpoint.use_server_token_count = false;
    let error = endpoint
        .validate()
        .expect_err("server counts are mandatory");
    assert!(
        error
            .to_string()
            .contains("requires --use-server-token-count")
    );
}

#[test]
fn per_chunk_usage_requires_streaming_chat() {
    let mut non_streaming = config(EndpointType::Chat);
    non_streaming.streaming = false;
    let error = non_streaming
        .validate()
        .expect_err("streaming is mandatory");
    assert!(error.to_string().contains("requires --streaming"));

    let error = config(EndpointType::Completions)
        .validate()
        .expect_err("chat endpoint is mandatory");
    assert!(error.to_string().contains("requires endpoint type 'chat'"));
}

#[test]
fn per_chunk_usage_valid_tuple_and_camel_alias_round_trip() {
    let endpoint = config(EndpointType::Chat)
        .validate()
        .expect("streaming chat with server counts is valid");
    assert!(endpoint.per_chunk_usage);

    let decoded: EndpointConfig = serde_json::from_value(serde_json::json!({
        "type": "chat",
        "urls": ["http://127.0.0.1:8000"],
        "streaming": true,
        "use_server_token_count": true,
        "perChunkUsage": true,
        "timeout_seconds": 60.0,
        "polling_interval_seconds": 0.1,
        "download_video_content": false,
        "wait_for_model_timeout": 0.0,
        "wait_for_model_interval": 5.0,
        "wait_for_model_mode": "inference",
        "wait_for_model_interval_set": false,
        "wait_for_model_mode_set": false,
        "headers": {},
        "extra": null
    }))
    .expect("decode camel-case alias");
    assert!(decoded.per_chunk_usage);
}
