// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public WebSocket transport-configuration contract.
#![cfg(feature = "engine")]

use aiperf_runtime::config::model::transport::{
    Transport, WebSocketTransportConfig, WebSocketTransportConfigError,
};

#[test]
fn websocket_transport_defaults_serialize_and_round_trip() {
    let transport: Transport =
        serde_json::from_str(r#"{"type":"websocket"}"#).expect("websocket defaults parse");
    assert!(transport.is_websocket());

    let expected = serde_json::json!({
        "type": "websocket",
        "fallback": "disabled",
        "ping_interval_seconds": 30.0,
        "stream_idle_timeout_seconds": 900.0,
        "max_queued_commands": 64,
        "max_queued_bytes": 1_048_576,
        "max_frame_bytes": 1_048_576,
        "max_message_bytes": 8_388_608,
        "max_response_bytes": 67_108_864,
    });
    let serialized = serde_json::to_value(&transport).expect("websocket defaults serialize");
    assert_eq!(serialized, expected);

    let round_tripped: Transport =
        serde_json::from_value(serialized).expect("serialized websocket transport parses");
    assert_eq!(
        serde_json::to_value(round_tripped).expect("round-tripped transport serializes"),
        expected
    );
}

#[test]
fn websocket_transport_rejects_invalid_durations() {
    for field in ["ping_interval_seconds", "stream_idle_timeout_seconds"] {
        for value in ["0", "-1", ".nan", ".inf", "-.inf"] {
            let yaml = format!("type: websocket\n{field}: {value}\n");
            assert!(
                serde_yaml::from_str::<Transport>(&yaml).is_err(),
                "{field}={value} must be rejected"
            );
        }
    }
}

#[test]
fn websocket_transport_rejects_zero_counts_and_byte_limits() {
    for field in [
        "max_queued_commands",
        "max_queued_bytes",
        "max_frame_bytes",
        "max_message_bytes",
        "max_response_bytes",
    ] {
        let value = serde_json::json!({"type": "websocket", field: 0});
        assert!(
            serde_json::from_value::<Transport>(value).is_err(),
            "{field}=0 must be rejected"
        );
    }
}

#[test]
fn websocket_transport_rejects_inconsistent_size_hierarchy() {
    for value in [
        serde_json::json!({
            "type": "websocket",
            "max_frame_bytes": 2,
            "max_message_bytes": 1,
        }),
        serde_json::json!({
            "type": "websocket",
            "max_message_bytes": 2,
            "max_response_bytes": 1,
        }),
    ] {
        assert!(serde_json::from_value::<Transport>(value).is_err());
    }
}

#[test]
fn websocket_transport_rejects_unknown_field() {
    assert!(
        serde_json::from_value::<Transport>(serde_json::json!({
            "type": "websocket",
            "unknown": true,
        }))
        .is_err()
    );
}

#[test]
fn websocket_transport_validation_returns_a_typed_error() {
    let config = WebSocketTransportConfig {
        max_queued_bytes: 0,
        ..WebSocketTransportConfig::default()
    };

    let error = config
        .validate()
        .expect_err("zero queue byte limit must be rejected");
    assert_eq!(
        error,
        WebSocketTransportConfigError::NonPositiveLimit {
            field: "max_queued_bytes",
        }
    );
    assert_eq!(
        error.to_string(),
        "websocket max_queued_bytes must be positive"
    );
}
