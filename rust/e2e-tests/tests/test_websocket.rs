// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native WebSocket transport product coverage.

mod common;

use aiperf_mock_server::config::WebSocketMode;
use common::{AIPerfHarness, DEFAULT_MODEL, MockServerConfig};
use serde_json::Value;

const REQUESTS: u32 = 2;

fn response_event_types(record: &Value) -> Vec<String> {
    record["responses"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|response| response["text"].as_str())
        .filter_map(|text| serde_json::from_str::<Value>(text).ok())
        .filter_map(|event| event["type"].as_str().map(str::to_owned))
        .collect()
}

fn request_event_types(record: &Value) -> Vec<String> {
    record["payload"]["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|message| message["payload"].as_str())
        .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
        .filter_map(|event| event["type"].as_str().map(str::to_owned))
        .collect()
}

fn websocket_config(url: &str, endpoint_type: &str, path: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{url}\"]\n\
        \x20   path: {path}\n\
        \x20   type: {endpoint_type}\n\
        \x20   streaming: true\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUESTS}\n\
        \x20   prompts:\n\
        \x20     isl: 8\n\
        \x20     osl: 1\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUESTS}\n\
        \x20     concurrency: 1\n\
        \x20 artifacts:\n\
        \x20   raw: true\n\
        \x20   records: [jsonl]\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport:\n\
        \x20   type: websocket\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

#[tokio::test]
async fn websocket_responses_profile_records_application_events() {
    let config = MockServerConfig {
        websocket_mode: WebSocketMode::TurnSerialized,
        websocket_content_events: 1,
        websocket_first_content_delay_ms: 20.0,
        websocket_content_interval_ms: 5.0,
        no_tokenizer: true,
        ..MockServerConfig::default()
    };
    let harness = AIPerfHarness::new_with(config).await;
    let url = harness.mock.url.replacen("http://", "ws://", 1);
    let config_path = harness.artifact_path().join("websocket-responses.yaml");
    std::fs::write(
        &config_path,
        websocket_config(&url, "responses", "/mock/websocket/turns"),
    )
    .expect("write WebSocket profile config");

    let result = harness.run(&format!("--config {}", config_path.display()));
    assert!(
        result.success(),
        "WebSocket Responses profile failed:\nstdout:\n{}\nstderr:\n{}",
        result.stdout,
        result.stderr
    );
    assert_eq!(result.artifacts.request_count(), f64::from(REQUESTS));
    let records = result.artifacts.raw_records();
    assert_eq!(records.len(), REQUESTS as usize);
    for record in &records {
        assert_eq!(record["metadata"]["actual_transport_route"], "websocket");
        assert_eq!(request_event_types(record), ["response.create"]);
        let events = response_event_types(record);
        assert!(
            events
                .iter()
                .any(|event| event == "response.output_text.delta")
        );
        assert!(events.iter().any(|event| event == "response.completed"));
    }
    for record in result.artifacts.jsonl() {
        assert!(
            record["metrics"]["time_to_last_round_trip"]["value"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        );
        assert!(
            record["metrics"]["avg_round_trip_time"]["value"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        );
    }
}

#[tokio::test]
async fn websocket_realtime_profile_records_text_and_audio_events() {
    let config = MockServerConfig {
        websocket_mode: WebSocketMode::Realtime,
        websocket_content_events: 1,
        no_tokenizer: true,
        ..MockServerConfig::default()
    };
    let harness = AIPerfHarness::new_with(config).await;
    let url = harness.mock.url.replacen("http://", "ws://", 1);
    let config_path = harness.artifact_path().join("websocket-realtime.yaml");
    std::fs::write(
        &config_path,
        websocket_config(&url, "realtime", "/mock/websocket/realtime"),
    )
    .expect("write Realtime WebSocket profile config");

    let result = harness.run(&format!("--config {}", config_path.display()));
    assert!(
        result.success(),
        "WebSocket Realtime profile failed:\nstdout:\n{}\nstderr:\n{}",
        result.stdout,
        result.stderr
    );
    assert_eq!(result.artifacts.request_count(), f64::from(REQUESTS));
    let records = result.artifacts.raw_records();
    assert_eq!(records.len(), REQUESTS as usize);
    for record in &records {
        assert_eq!(record["metadata"]["actual_transport_route"], "websocket");
        assert_eq!(
            request_event_types(record),
            [
                "conversation.item.create",
                "input_audio_buffer.commit",
                "response.create"
            ]
        );
        let events = response_event_types(record);
        assert!(events.iter().any(|event| event == "response.text.delta"));
        assert!(events.iter().any(|event| event == "response.audio.delta"));
        assert!(events.iter().any(|event| event == "response.done"));
    }
    for record in result.artifacts.jsonl() {
        assert!(record["metrics"].get("time_to_last_round_trip").is_none());
        assert!(record["metrics"].get("avg_round_trip_time").is_none());
    }
}
