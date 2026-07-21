// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Messages endpoint wire and behavior contracts.

use std::collections::BTreeMap;

use aiperf_runtime::endpoints::{
    CreditPhase, Endpoint, EndpointConfig, EndpointType, Media, MessagesEndpoint, Modality,
    ModelEndpoint, RequestInfo, RequestRecord, ResponseData, ServerResponse, Turn,
};
use serde_json::{Map, Value, json};

fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn config(streaming: bool) -> EndpointConfig {
    EndpointConfig {
        endpoint_type: EndpointType::Messages,
        urls: vec!["http://localhost:8000".into()],
        streaming,
        ..EndpointConfig::default()
    }
    .validate()
    .unwrap()
}

fn request(turns: Vec<Turn>, streaming: bool) -> RequestInfo {
    RequestInfo {
        model_endpoint: ModelEndpoint {
            primary_model_name: "test-model".into(),
            endpoint: config(streaming),
        },
        turns,
        system_message: None,
        user_context_message: None,
        credit_phase: CreditPhase::Profiling,
        x_request_id: None,
        x_correlation_id: None,
        conversation_id: None,
    }
}

fn text_turn(text: &str) -> Turn {
    Turn {
        texts: vec![Media::new(vec![text.into()])],
        ..Turn::default()
    }
}

fn response(perf_ns: u64, value: Value) -> ServerResponse {
    ServerResponse::from_json(perf_ns, value)
}

#[test]
fn simple_non_streaming_request_is_byte_exact() {
    let mut turn = text_turn("Hello!");
    turn.model = Some("claude-sonnet-4-20250514".into());
    let body = MessagesEndpoint
        .format_payload(&request(vec![turn], false))
        .unwrap()
        .materialize_standalone()
        .unwrap();
    assert_eq!(
        &body[..],
        br#"{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}"#
            as &[u8]
    );
}

#[test]
fn extra_numeric_values_use_orjson_equivalent_wire_spelling() {
    let mut turn = text_turn("numbers");
    turn.extra_body = Some(Map::from_iter([(
        "values".into(),
        json!([0.2, 1e-7, 1e20, -0.0, 1.2345678901234567]),
    )]));
    let body = MessagesEndpoint
        .format_payload(&request(vec![turn], false))
        .unwrap()
        .materialize_standalone()
        .unwrap();
    assert_eq!(
        &body[..],
        br#"{"model":"test-model","messages":[{"role":"user","content":"numbers"}],"max_tokens":1024,"values":[0.2,1e-7,1e+20,-0.0,1.2345678901234567]}"#
            as &[u8]
    );
}

#[test]
fn authored_empty_raw_messages_renders_the_synthetic_turn() {
    let mut turn = text_turn("must be rendered");
    turn.role = Some(String::new());
    turn.model = Some(String::new());
    turn.raw_messages = Some(Vec::new());
    let body = plan_body(
        MessagesEndpoint
            .format_payload(&request(vec![turn], false))
            .unwrap(),
    );
    assert_eq!(body["model"], "test-model");
    assert_eq!(
        body["messages"],
        json!([{"role":"user","content":"must be rendered"}])
    );
}

#[test]
fn full_request_merge_order_and_omitted_false_stream_are_byte_exact() {
    let mut first = text_turn("first");
    first.raw_system = Some(vec![json!({
        "type":"text",
        "text":"system",
        "cache_control":{"type":"ephemeral"}
    })]);
    first.raw_tools = Some(vec![json!({
        "name":"lookup",
        "description":"look up",
        "input_schema":{"type":"object","properties":{"q":{"type":"string"}}}
    })]);
    let mut last = text_turn("second");
    last.model = Some("claude".into());
    last.max_tokens = Some(12);
    last.extra_body = Some(Map::from_iter([
        ("temperature".into(), json!(0.2)),
        ("top_k".into(), json!(7)),
    ]));
    let mut request = request(vec![first, last], false);
    request.user_context_message = Some("ctx".into());
    request.system_message = Some("ignored".into());
    request.model_endpoint.endpoint.extra = Some(Map::from_iter([
        ("temperature".into(), json!(0.8)),
        ("top_p".into(), json!(0.9)),
    ]));
    let bytes = MessagesEndpoint
        .format_payload(&request)
        .unwrap()
        .materialize_standalone()
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(body.get("stream").is_none());
    assert_eq!(body["temperature"], json!(0.2));
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(
        &bytes[..],
        br#"{"model":"claude","messages":[{"role":"user","content":"ctx"},{"role":"user","content":"first"},{"role":"user","content":"second"}],"max_tokens":12,"system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}],"tools":[{"name":"lookup","description":"look up","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],"temperature":0.2,"top_p":0.9,"top_k":7}"#
            as &[u8]
    );
}

#[test]
fn streaming_and_headers_match_messages_wire_contract() {
    let mut request = request(vec![text_turn("hi")], true);
    request.model_endpoint.endpoint.headers = BTreeMap::from([
        ("anthropic-version".into(), "custom-version".into()),
        ("anthropic-beta".into(), "thinking".into()),
    ]);
    request.model_endpoint.endpoint.api_key = Some("sk-ant-test".into());
    let headers = MessagesEndpoint.format_headers(&request.model_endpoint.endpoint);
    assert_eq!(headers["x-api-key"], "sk-ant-test");
    assert_eq!(headers["anthropic-version"], "custom-version");
    assert_eq!(headers["content-type"], "application/json");
    assert!(!headers.contains_key("Authorization"));
    let debug = format!("{:?}", request.model_endpoint.endpoint);
    assert!(!debug.contains("sk-ant-test"));
    assert!(!debug.contains("thinking"));
    let serialized = serde_json::to_string(&request.model_endpoint.endpoint).unwrap();
    assert!(!serialized.contains("sk-ant-test"));
    assert!(!serialized.contains("anthropic-beta"));
    let body = plan_body(MessagesEndpoint.format_payload(&request).unwrap());
    assert_eq!(body["stream"], true);

    let mut empty_key = config(false);
    empty_key.api_key = Some(String::new());
    assert!(
        !MessagesEndpoint
            .format_headers(&empty_key)
            .contains_key("x-api-key")
    );
}

#[test]
fn image_shapes_and_unsupported_media_match_contract() {
    let mut turn = Turn {
        texts: vec![Media::new(vec!["describe".into()])],
        images: vec![Media::new(vec![
            "https://example/cat.jpg".into(),
            "data:image/png;base64,QUJD".into(),
            "data:;base64,REVG".into(),
        ])],
        ..Turn::default()
    };
    let body = plan_body(
        MessagesEndpoint
            .format_payload(&request(vec![turn.clone()], false))
            .unwrap(),
    );
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type":"image","source":{"type":"url","url":"https://example/cat.jpg"}})
    );
    assert_eq!(
        body["messages"][0]["content"][2],
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}})
    );
    assert_eq!(
        body["messages"][0]["content"][3]["source"]["media_type"],
        "image/png"
    );
    turn.audios = vec![Media::new(vec!["wav,AA==".into()])];
    assert!(
        MessagesEndpoint
            .format_payload(&request(vec![turn.clone()], false))
            .unwrap_err()
            .to_string()
            .contains("does not support audio")
    );
    turn.audios.clear();
    turn.videos = vec![Media::new(vec!["https://example/video.mp4".into()])];
    assert!(
        MessagesEndpoint
            .format_payload(&request(vec![turn], false))
            .unwrap_err()
            .to_string()
            .contains("does not support video")
    );
}

#[test]
fn non_streaming_parse_uses_reasoning_text_tool_precedence() {
    let parsed = MessagesEndpoint
        .parse_response(&response(
            7,
            json!({
                "type":"message",
                "content":[
                    {"type":"thinking","thinking":"why"},
                    {"type":"text","text":"answer"},
                    {"type":"tool_use","name":"calc","input":{"a":1}}
                ],
                "usage":{"input_tokens":10,"output_tokens":4}
            }),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Reasoning {
            content: Some("answer".into()),
            reasoning: "why".into(),
        })
    );
    assert_eq!(parsed.usage.unwrap()["input_tokens"], 10);

    let parsed = MessagesEndpoint
        .parse_response(&response(
            8,
            json!({
                "type":"message",
                "content":[
                    {"type":"text","text":"calling"},
                    {"type":"tool_use","name":"calc","input":{"a":1}}
                ]
            }),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::ToolCall {
            tool_call_text: "calc{\"a\":1}".into(),
            content: Some("calling".into()),
        })
    );

    let parsed = MessagesEndpoint
        .parse_response(&response(
            9,
            json!({
                "type":"message",
                "content":[{"type":"text","text":"answer"}],
                "usage":{}
            }),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(parsed.usage, Some(json!({})));
}

#[test]
fn streaming_event_map_includes_text_reasoning_tool_and_usage() {
    let cases = [
        (
            json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}),
            Some(ResponseData::Text { text: "hi".into() }),
        ),
        (
            json!({"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"why"}}),
            Some(ResponseData::Reasoning {
                content: None,
                reasoning: "why".into(),
            }),
        ),
        (
            json!({"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}),
            Some(ResponseData::ToolCall {
                tool_call_text: "{\"q\":".into(),
                content: None,
            }),
        ),
    ];
    for (index, (event, expected)) in cases.into_iter().enumerate() {
        let parsed = MessagesEndpoint
            .parse_response(&response(index as u64 + 1, event))
            .unwrap()
            .unwrap();
        assert_eq!(parsed.data, expected);
    }
    for event in [
        json!({"type":"ping"}),
        json!({"type":"content_block_start"}),
        json!({"type":"content_block_stop"}),
        json!({"type":"content_block_delta","delta":{"type":"signature_delta","signature":"sig"}}),
        json!({"type":"message_stop"}),
        // No error/type/message fields: exercises the "<missing>" logging fallback.
        json!({"type":"error"}),
        json!({"type":"error","error":{"type":"overloaded_error","message":"try again"}}),
        // Unknown event/delta types are dropped, not treated as errors.
        json!({"type":"some_future_event"}),
        json!({"type":"content_block_delta","delta":{"type":"some_future_delta"}}),
    ] {
        assert!(
            MessagesEndpoint
                .parse_response(&response(1, event))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn docs_split_streaming_usage_is_folded_into_final_usage() {
    let record = RequestRecord {
        responses: vec![
            response(
                1,
                json!({"type":"message_start","message":{"usage":{"input_tokens":25,"cache_read_input_tokens":7,"output_tokens":1}}}),
            ),
            response(
                2,
                json!({"type":"message_delta","usage":{"output_tokens":15}}),
            ),
        ],
    };
    let parsed = MessagesEndpoint.extract_response_data(&record).unwrap();
    let final_usage = parsed.last().unwrap().usage.as_ref().unwrap();
    assert_eq!(
        serde_json::to_vec(final_usage).unwrap(),
        br#"{"output_tokens":15,"input_tokens":25,"cache_read_input_tokens":7}"#
    );
}

#[test]
fn payload_extraction_covers_system_tools_history_and_anthropic_images() {
    let payload = json!({
        "system":[{"type":"text","text":"first"},"second"],
        "messages":[
            {"role":"user","content":[{"type":"text","text":"ask"},{"type":"image","source":{"type":"url","url":"u"}},{"type":"image_url"}]},
            {"role":"assistant","content":[{"type":"tool_use","name":"calc","input":{"b":2,"a":1}}]},
            {"role":"user","content":[{"type":"tool_result","content":[{"type":"text","text":"done"}]}]}
        ],
        "tools":[{"name":"calc","description":"calculate","input_schema":{"type":"object","properties":{"x":{"type":"number"}}}}]
    });
    let extracted = MessagesEndpoint.extract_payload_inputs(&payload);
    assert_eq!(&extracted.texts[..3], ["first", "second", "ask"]);
    assert_eq!(extracted.image_count, 1);
    assert!(extracted.texts.contains(&"calc".into()));
    assert!(extracted.texts.contains(&"calculate".into()));
    assert!(extracted.texts.contains(&"{\"b\":2,\"a\":1}".into()));
    assert!(extracted.texts.contains(&"done".into()));
    assert!(
        extracted
            .texts
            .contains(&"{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"}}}".into())
    );
}

#[test]
fn assistant_replay_reassembles_thinking_signature_text_tool_and_unknown_fields() {
    let record = RequestRecord {
        responses: vec![
            response(
                1,
                json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            ),
            response(
                2,
                json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
            ),
            response(
                3,
                json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
            ),
            response(
                4,
                json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}),
            ),
            response(
                5,
                json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"Bash","input":{},"caller":{"type":"direct"}}}),
            ),
            response(
                6,
                json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}),
            ),
            response(
                7,
                json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}),
            ),
        ],
    };
    let turn = MessagesEndpoint
        .build_assistant_turn(&record)
        .unwrap()
        .unwrap();
    let content = turn.raw_messages.unwrap().remove(0)["content"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        content[0],
        json!({"type":"thinking","thinking":"think","signature":"sig"})
    );
    assert_eq!(content[1], json!({"type":"text","text":"answer"}));
    assert_eq!(content[2]["input"], json!({"command":"ls"}));
    assert_eq!(content[2]["caller"], json!({"type":"direct"}));
}

#[test]
fn text_only_replay_falls_back_to_plain_turn_and_metadata_is_registered() {
    let record = RequestRecord {
        responses: vec![response(
            1,
            json!({"type":"message","content":[{"type":"text","text":"hello"}]}),
        )],
    };
    let turn = MessagesEndpoint
        .build_assistant_turn(&record)
        .unwrap()
        .unwrap();
    assert!(turn.raw_messages.is_none());
    assert_eq!(turn.texts[0].contents, ["hello"]);
    let metadata = MessagesEndpoint.descriptor();
    assert_eq!(metadata.endpoint_path, Some("/v1/messages"));
    assert!(metadata.supports_streaming);
    assert!(metadata.supports_input(Modality::Image));
    assert!(!metadata.supports_input(Modality::Audio));
}
