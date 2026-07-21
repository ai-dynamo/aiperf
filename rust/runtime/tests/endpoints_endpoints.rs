// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::endpoints::{
    ChatEndpoint, CompletionsEndpoint, CreditPhase, EmbeddingsEndpoint, Endpoint, EndpointConfig,
    EndpointType, ImageEditEndpoint, ImageRetrievalEndpoint, Media, ModelEndpoint,
    RequestContentType, RequestInfo, RequestRecord, ResponseData, ResponsesEndpoint,
    ServerResponse, Turn, VideoGenerationEndpoint,
};
use serde_json::{Map, Value, json};

fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn cfg(endpoint_type: EndpointType) -> EndpointConfig {
    EndpointConfig {
        endpoint_type,
        urls: vec!["http://localhost:8000".to_string()],
        ..EndpointConfig::default()
    }
    .validate()
    .unwrap()
}

fn request(endpoint_type: EndpointType, turns: Vec<Turn>) -> RequestInfo {
    RequestInfo {
        model_endpoint: ModelEndpoint {
            primary_model_name: "model-a".to_string(),
            endpoint: cfg(endpoint_type),
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
        texts: vec![Media::new(vec![text.to_string()])],
        ..Turn::default()
    }
}

#[test]
fn metadata_and_config_validation_cover_registry_rules() {
    assert!(VideoGenerationEndpoint.descriptor().requires_polling);
    assert!(ImageEditEndpoint.descriptor().requires_form_data);
    assert!(!ImageRetrievalEndpoint.descriptor().tokenizes_input);
    assert_eq!(EndpointConfig::default().polling_interval_seconds, 0.1);

    let mut embeddings = cfg(EndpointType::Embeddings);
    embeddings.streaming = true;
    assert!(!embeddings.validate().unwrap().streaming);

    assert_eq!(
        EndpointConfig {
            endpoint_type: EndpointType::ImageEdit,
            urls: vec!["http://h".into()],
            ..EndpointConfig::default()
        }
        .validate()
        .unwrap()
        .request_content_type,
        Some(RequestContentType::MultipartFormData)
    );
    assert!(
        EndpointConfig {
            endpoint_type: EndpointType::ImageEdit,
            urls: vec!["http://h".into()],
            request_content_type: Some(RequestContentType::ApplicationJson),
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        EndpointConfig {
            polling_interval_seconds: 0.000_9,
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        EndpointConfig {
            urls: vec![" http://h".into()],
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        EndpointConfig {
            urls: vec!["ftp://h".into()],
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        EndpointConfig {
            urls: vec!["http://:8000".into()],
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        EndpointConfig {
            urls: vec!["http://h".into()],
            wait_for_model_interval: 1.0,
            wait_for_model_interval_set: true,
            ..EndpointConfig::default()
        }
        .validate()
        .is_err()
    );
    assert_eq!(
        EndpointConfig {
            urls: vec!["http://h".into()],
            template: Some("{{ body }}".into()),
            ..EndpointConfig::default()
        }
        .validate()
        .unwrap()
        .endpoint_type,
        EndpointType::Template
    );
    assert!(
        EndpointConfig {
            endpoint_type: EndpointType::Template,
            urls: vec!["http://h".into()],
            extra: Some(Map::from_iter([(
                "payload_template".into(),
                json!(r#"{"text":{{ text|tojson }}}"#),
            )])),
            ..EndpointConfig::default()
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn chat_formatting_merges_and_preserves_usage_override() {
    let mut req = request(
        EndpointType::Chat,
        vec![Turn {
            max_tokens: Some(8),
            raw_tools: Some(vec![json!({"type":"function","function":{"name":"f"}})]),
            texts: vec![Media::new(vec!["hi".to_string()])],
            extra_body: Some(Map::from_iter([("temperature".to_string(), json!(0.1))])),
            ..Turn::default()
        }],
    );
    req.system_message = Some("sys".to_string());
    req.user_context_message = Some("ctx".to_string());
    req.model_endpoint.endpoint.streaming = true;
    req.model_endpoint.endpoint.use_server_token_count = true;
    req.model_endpoint.endpoint.extra = Some(Map::from_iter([
        ("temperature".to_string(), json!(1.0)),
        (
            "stream_options".to_string(),
            json!({"include_usage": false, "x": 1}),
        ),
    ]));

    let body = plan_body(ChatEndpoint.format_payload(&req).unwrap());
    assert_eq!(
        body["messages"][0],
        json!({"role":"system","content":"sys"})
    );
    assert_eq!(body["messages"][1], json!({"role":"user","content":"ctx"}));
    assert_eq!(body["messages"][2], json!({"role":"user","content":"hi"}));
    assert_eq!(body["max_completion_tokens"], json!(8));
    assert_eq!(body["temperature"], json!(0.1));
    assert_eq!(
        body["stream_options"],
        json!({"include_usage": false, "x": 1})
    );
}

#[test]
fn responses_formatting_rejects_video_and_filters_replay_unsafe() {
    let req = request(
        EndpointType::Responses,
        vec![Turn {
            raw_messages: Some(vec![
                json!({"type":"reasoning","summary":[]}),
                json!({"type":"message","role":"assistant","content":"ok"}),
            ]),
            ..Turn::default()
        }],
    );
    let body = plan_body(ResponsesEndpoint.format_payload(&req).unwrap());
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
    assert_eq!(body["input"][0]["type"], json!("message"));

    let err = ResponsesEndpoint
        .format_payload(&request(
            EndpointType::Responses,
            vec![Turn {
                videos: vec![Media::new(vec!["file://v".to_string()])],
                ..Turn::default()
            }],
        ))
        .unwrap_err();
    assert!(err.to_string().contains("does not support video"));
}

#[test]
fn completions_and_embeddings_formatting() {
    let mut req = request(EndpointType::Completions, vec![text_turn("p")]);
    req.credit_phase = CreditPhase::Warmup;
    req.turns[0].max_tokens = Some(4);
    let body = plan_body(CompletionsEndpoint.format_payload(&req).unwrap());
    assert!(body["prompt"][0].as_str().unwrap().ends_with("\np"));
    assert_eq!(body["max_tokens"], json!(4));

    let mut req = request(EndpointType::Embeddings, vec![text_turn("embed")]);
    req.turns[0].max_tokens = Some(9);
    let body = plan_body(EmbeddingsEndpoint.format_payload(&req).unwrap());
    assert_eq!(body["input"], json!(["embed"]));
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn body_plan_materializes_byte_identical_to_hand_built_objects() {
    let mut req = request(
        EndpointType::Chat,
        vec![Turn {
            max_tokens: Some(8),
            texts: vec![Media::new(vec!["hi".to_string()])],
            extra_body: Some(Map::from_iter([("temperature".to_string(), json!(0.1))])),
            ..Turn::default()
        }],
    );
    req.model_endpoint.endpoint.streaming = true;
    req.model_endpoint.endpoint.use_server_token_count = true;
    req.model_endpoint.endpoint.extra = Some(Map::from_iter([(
        "stream_options".to_string(),
        json!({"include_usage": false, "x": 1}),
    )]));
    let bytes = ChatEndpoint
        .format_payload(&req)
        .unwrap()
        .materialize_standalone()
        .unwrap();
    let expected = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "model-a",
        "stream": true,
        "max_completion_tokens": 8,
        "stream_options": {"include_usage": false, "x": 1},
        "temperature": 0.1
    });
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );

    let mut req = request(EndpointType::Completions, vec![text_turn("p")]);
    req.turns[0].max_tokens = Some(4);
    let bytes = CompletionsEndpoint
        .format_payload(&req)
        .unwrap()
        .materialize_standalone()
        .unwrap();
    let expected = json!({
        "prompt": ["p"],
        "model": "model-a",
        "stream": false,
        "max_tokens": 4
    });
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );

    let req = request(EndpointType::Embeddings, vec![text_turn("embed")]);
    let bytes = EmbeddingsEndpoint
        .format_payload(&req)
        .unwrap()
        .materialize_standalone()
        .unwrap();
    let expected = json!({"model": "model-a", "input": ["embed"]});
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );

    let mut req = request(EndpointType::Responses, vec![text_turn("q")]);
    req.system_message = Some("sys".to_string());
    let bytes = ResponsesEndpoint
        .format_payload(&req)
        .unwrap()
        .materialize_standalone()
        .unwrap();
    let expected = json!({
        "input": [{"role": "user", "content": "q", "type": "message"}],
        "model": "model-a",
        "stream": false,
        "instructions": "sys"
    });
    assert_eq!(
        &bytes[..],
        serde_json::to_vec(&expected).unwrap().as_slice()
    );
}

#[test]
fn chat_parse_precedence_usage_and_assistant_reassembly() {
    let parsed = ChatEndpoint.parse_response(&ServerResponse::from_json(7, json!({"object":"chat.completion.chunk","choices":[{"delta":{"content":"hello","tool_calls":[{"function":{"name":"fn","arguments":"{}"}}]}}]}))).unwrap().unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::ToolCall {
            tool_call_text: "fn{}".to_string(),
            content: Some("hello".to_string())
        })
    );

    let parsed = ChatEndpoint.parse_response(&ServerResponse::from_json(8, json!({"object":"chat.completion.chunk","choices":[{"delta":{}}],"usage":{"completion_tokens":3}}))).unwrap().unwrap();
    assert!(parsed.data.is_none());
    assert_eq!(parsed.usage.unwrap()["completion_tokens"], json!(3));
    assert!(
        ChatEndpoint
            .parse_response(&ServerResponse::from_json(9, json!({"object":"error"})))
            .unwrap()
            .is_none()
    );

    let record = RequestRecord {
        responses: vec![
            ServerResponse::from_json(
                1,
                json!({"object":"chat.completion.chunk","choices":[{"delta":{"content":"A","tool_calls":[{"function":{"name":"first","arguments":"{\"a"}}]}}]}),
            ),
            ServerResponse::from_json(
                2,
                json!({"object":"chat.completion.chunk","choices":[{"delta":{"tool_calls":[{"function":{"name":"second","arguments":"\":1}"}}]}}]}),
            ),
            ServerResponse::from_json(
                3,
                json!({"object":"chat.completion.chunk","choices":[{"delta":{"function_call":{"name":"leg","arguments":"acy"}}}]}),
            ),
        ],
    };
    let turn = ChatEndpoint.build_assistant_turn(&record).unwrap().unwrap();
    let message = &turn.raw_messages.unwrap()[0];
    assert_eq!(message["content"], json!("A"));
    assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(
        message["tool_calls"][0]["function"]["name"],
        json!("firstleg")
    );
    assert_eq!(
        message["tool_calls"][1]["function"]["name"],
        json!("second")
    );
}

#[test]
fn responses_parse_event_map_full_precedence_and_replay_union() {
    let parsed = ResponsesEndpoint
        .parse_response(&ServerResponse::from_json(
            1,
            json!({"type":"response.function_call_arguments.delta","delta":"{\"x\""}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::ToolCall {
            tool_call_text: "{\"x\"".to_string(),
            content: None
        })
    );

    let parsed = ResponsesEndpoint
        .parse_response(&ServerResponse::from_json(
            2,
            json!({"object":"response","output":[
        {"type":"message","content":[{"type":"output_text","text":"msg"}]},
        {"type":"reasoning","summary":[{"type":"summary_text","text":"why"}]},
        {"type":"function_call","name":"fn","arguments":"{}"}
    ],"usage":{"output_tokens":5}}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Reasoning {
            content: Some("msg".to_string()),
            reasoning: "why".to_string()
        })
    );

    let record = RequestRecord {
        responses: vec![
            ServerResponse::from_json(
                1,
                json!({"type":"response.completed","response":{"output":[{"id":"a","type":"message","content":[]}]}}),
            ),
            ServerResponse::from_json(
                2,
                json!({"type":"response.output_item.done","item":{"id":"a","type":"message","content":[{"text":"dupe"}]}}),
            ),
            ServerResponse::from_json(
                3,
                json!({"type":"response.output_item.done","item":{"call_id":"b","type":"function_call","name":"f","arguments":"{}"}}),
            ),
        ],
    };
    let raw = ResponsesEndpoint
        .build_assistant_turn(&record)
        .unwrap()
        .unwrap()
        .raw_messages
        .unwrap();
    assert_eq!(raw.len(), 2);
    assert_eq!(raw[0]["id"], json!("a"));
    assert_eq!(raw[1]["call_id"], json!("b"));
}

#[test]
fn responses_extract_response_data_dedups_streamed_output_text() {
    // A part that streamed via deltas: its terminal `done` must not
    // re-contribute the full text (that would double-count output tokens),
    // but must still surface an empty-text ParsedResponse at the `done`
    // timestamp so content-timing metrics see the same event count.
    let record = RequestRecord {
        responses: vec![
            ServerResponse::from_json(
                1,
                json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hel"}),
            ),
            ServerResponse::from_json(
                2,
                json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"lo"}),
            ),
            ServerResponse::from_json(
                3,
                json!({"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello"}),
            ),
            // A sibling part that never streamed deltas: its `done` is the
            // sole carrier and must still contribute its full text.
            ServerResponse::from_json(
                4,
                json!({"type":"response.output_text.done","output_index":1,"content_index":0,"text":"World"}),
            ),
        ],
    };
    let parsed = ResponsesEndpoint.extract_response_data(&record).unwrap();
    assert_eq!(parsed.len(), 4);
    assert_eq!(
        parsed[0].data,
        Some(ResponseData::Text {
            text: "Hel".to_string()
        })
    );
    assert_eq!(
        parsed[1].data,
        Some(ResponseData::Text {
            text: "lo".to_string()
        })
    );
    // Deduped: empty text, not "Hello" again.
    assert_eq!(
        parsed[2].data,
        Some(ResponseData::Text {
            text: String::new()
        })
    );
    assert_eq!(parsed[2].perf_ns, 3);
    // Sole carrier: full text preserved.
    assert_eq!(
        parsed[3].data,
        Some(ResponseData::Text {
            text: "World".to_string()
        })
    );
}

#[test]
fn completions_and_embeddings_parse_policies() {
    let parsed = CompletionsEndpoint
        .parse_response(&ServerResponse::from_json(
            1,
            json!({"object":"text_completion","choices":[{"text":"done"}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Text {
            text: "done".to_string()
        })
    );
    assert!(
        CompletionsEndpoint
            .parse_response(&ServerResponse::from_json(
                1,
                json!({"object":"proxy_error"})
            ))
            .unwrap()
            .is_none()
    );

    let parsed = EmbeddingsEndpoint
        .parse_response(&ServerResponse::from_json(
            1,
            json!({"data":[{"object":"embedding","embedding":[1.0,2.0]}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Embeddings {
            embeddings: vec![vec![1.0, 2.0]]
        })
    );
    assert!(
        EmbeddingsEndpoint
            .parse_response(&ServerResponse::from_json(
                1,
                json!({"data":[{"object":"not_embedding"}]})
            ))
            .unwrap_err()
            .to_string()
            .contains("invalid list")
    );
}

#[test]
fn payload_extraction_covers_items_tools_flat_and_pretokenized() {
    let payload = json!({
        "messages": [
            {"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"u"}}]},
            {"role":"assistant","content":"ok","tool_calls":[{"function":{"name":"fn","arguments":"{}"}}]}
        ],
        "tools": [{"type":"function","function":{"name":"tool","description":"desc","parameters":{"b":1,"a":2}}}],
        "input": ["should-not-double-count"]
    });
    let extracted = ChatEndpoint.extract_payload_inputs(&payload);
    assert_eq!(extracted.image_count, 1);
    assert_eq!(extracted.messages.as_ref().unwrap().len(), 2);
    assert!(extracted.tool_texts.contains(&"fn".to_string()));
    assert!(extracted.tool_texts.contains(&"{}".to_string()));
    assert!(
        extracted
            .tool_texts
            .contains(&"{\"b\":1,\"a\":2}".to_string())
    );
    assert!(
        !extracted
            .texts
            .contains(&"should-not-double-count".to_string())
    );

    assert_eq!(
        ChatEndpoint
            .extract_payload_inputs(&json!({"input": [[1,2,3], [4]]}))
            .pretokenised_token_count,
        4
    );
    assert_eq!(
        ChatEndpoint
            .extract_payload_inputs(&json!({"query":"q","passages":["p",{"text":"p2"}]}))
            .texts,
        vec!["q", "p", "p2"]
    );
}

#[test]
fn responses_extraction_prepends_instructions() {
    let payload = json!({
        "instructions": [{"text":"sys"}],
        "input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_audio","input_audio":{}}]}]
    });
    let extracted = ResponsesEndpoint.extract_payload_inputs(&payload);
    assert_eq!(extracted.texts[0], "sys");
    assert_eq!(extracted.audio_count, 1);
    assert_eq!(
        extracted.messages.unwrap()[0],
        json!({"role":"system","content":"sys"})
    );
}
