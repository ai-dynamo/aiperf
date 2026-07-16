// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::endpoints::{
    CohereRankingsEndpoint, CreditPhase, Endpoint, EndpointConfig, EndpointType,
    HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint, ImageEditEndpoint, ImageGenerationEndpoint,
    ImageRetrievalEndpoint, Media, ModelEndpoint, NimEmbeddingsEndpoint, NimRankingsEndpoint,
    RawEndpoint, RequestInfo, ResponseData, ServerResponse, SolidoRagEndpoint, TemplateEndpoint,
    Turn, VideoGenerationEndpoint,
};
use serde_json::{Map, Value, json};

/// Materialize a formatter's [`BodyPlan`] into a decoded JSON value so the
/// structural assertions below keep inspecting fields as before stage B.
fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

fn config(endpoint_type: EndpointType) -> EndpointConfig {
    EndpointConfig {
        endpoint_type,
        urls: vec!["http://localhost:8000".into()],
        ..EndpointConfig::default()
    }
}

fn request(endpoint_type: EndpointType, turns: Vec<Turn>) -> RequestInfo {
    RequestInfo {
        model_endpoint: ModelEndpoint {
            primary_model_name: "default-model".into(),
            endpoint: config(endpoint_type),
        },
        turns,
        system_message: None,
        user_context_message: None,
        credit_phase: CreditPhase::Profiling,
        x_request_id: Some("req-1".into()),
        x_correlation_id: Some("corr-1".into()),
        conversation_id: Some("conv-1".into()),
    }
}

fn text(value: &str) -> Media {
    Media::new(vec![value.to_string()])
}

fn named(name: &str, values: &[&str]) -> Media {
    Media {
        name: name.into(),
        contents: values.iter().map(|value| (*value).to_string()).collect(),
    }
}

#[test]
fn ranking_dialects_format_parse_and_account_vendor_shapes() {
    let turn = Turn {
        model: Some("reranker".into()),
        texts: vec![
            named("queries", &["query", "ignored-query"]),
            named("passages", &["first", "second"]),
            named("ignored", &["not-on-wire"]),
        ],
        extra_body: Some(Map::from_iter([("top_n".into(), json!(2))])),
        ..Turn::default()
    };

    let nim = plan_body(
        NimRankingsEndpoint
            .format_payload(&request(EndpointType::NimRankings, vec![turn.clone()]))
            .unwrap(),
    );
    assert_eq!(nim["model"], "reranker");
    assert_eq!(nim["query"], json!({"text":"query"}));
    assert_eq!(nim["passages"][1], json!({"text":"second"}));
    assert_eq!(nim["top_n"], 2);
    assert_eq!(
        NimRankingsEndpoint.extract_payload_inputs(&nim).texts,
        vec!["query", "first", "second"]
    );

    let cohere = plan_body(
        CohereRankingsEndpoint
            .format_payload(&request(EndpointType::CohereRankings, vec![turn.clone()]))
            .unwrap(),
    );
    assert_eq!(cohere["query"], "query");
    assert_eq!(cohere["documents"], json!(["first", "second"]));
    let parsed = CohereRankingsEndpoint
        .parse_response(&ServerResponse::from_json(
            7,
            json!({"results":[{"index":1,"relevance_score":0.75,"document":"ignored"}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        parsed.data,
        Some(ResponseData::Rankings {
            rankings: vec![json!({"index":1,"score":0.75})]
        })
    );

    let hf = plan_body(
        HfTeiRankingsEndpoint
            .format_payload(&request(EndpointType::HfTeiRankings, vec![turn]))
            .unwrap(),
    );
    assert_eq!(hf["texts"], json!(["first", "second"]));
    let parsed = HfTeiRankingsEndpoint
        .parse_response(&ServerResponse::from_json(
            8,
            json!([{"index":0,"score":0.9}]),
        ))
        .unwrap()
        .unwrap();
    assert!(matches!(parsed.data, Some(ResponseData::Rankings { .. })));

    let authored_empty = plan_body(
        NimRankingsEndpoint
            .format_payload(&request(
                EndpointType::NimRankings,
                vec![Turn {
                    texts: vec![named("query", &[""]), named("passages", &["", "kept"])],
                    ..Turn::default()
                }],
            ))
            .unwrap(),
    );
    assert_eq!(authored_empty["query"]["text"], "");
    assert_eq!(
        authored_empty["passages"],
        json!([{"text":""},{"text":"kept"}])
    );
}

#[test]
fn nim_embeddings_pairs_multimodal_inputs_and_preserves_strict_parser() {
    let mut req = request(
        EndpointType::NimEmbeddings,
        vec![Turn {
            texts: vec![Media::new(vec![
                "describe one".into(),
                "describe two".into(),
            ])],
            images: vec![Media::new(vec![
                "data:image/png;base64,one".into(),
                "data:image/png;base64,two".into(),
            ])],
            ..Turn::default()
        }],
    );
    let body = plan_body(NimEmbeddingsEndpoint.format_payload(&req).unwrap());
    assert_eq!(
        body["input"],
        json!([
            "describe one data:image/png;base64,one",
            "describe two data:image/png;base64,two"
        ])
    );
    req.turns[0].images[0].contents.pop();
    assert!(
        NimEmbeddingsEndpoint
            .format_payload(&req)
            .unwrap_err()
            .to_string()
            .contains("same length")
    );
    assert!(
        NimEmbeddingsEndpoint
            .parse_response(&ServerResponse::from_json(
                9,
                json!({"data":[{"object":"wrong","embedding":[1.0]}]})
            ))
            .is_err()
    );
    for empty_data in [Value::Null, json!([]), json!({})] {
        assert!(
            NimEmbeddingsEndpoint
                .parse_response(&ServerResponse::from_json(9, json!({"data":empty_data})))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn huggingface_generate_uses_streaming_path_and_incremental_token_text() {
    let mut req = request(
        EndpointType::HuggingfaceGenerate,
        vec![Turn {
            texts: vec![Media::new(vec!["hello".into(), "world".into()])],
            max_tokens: Some(17),
            extra_body: Some(Map::from_iter([("details".into(), Value::Bool(true))])),
            ..Turn::default()
        }],
    );
    req.model_endpoint.endpoint.extra = Some(Map::from_iter([("temperature".into(), json!(0.2))]));
    let body = plan_body(HuggingFaceGenerateEndpoint.format_payload(&req).unwrap());
    assert_eq!(body["inputs"], "hello world");
    assert_eq!(body["parameters"]["max_new_tokens"], 17);
    assert_eq!(body["parameters"]["temperature"], 0.2);
    assert_eq!(body["details"], true);
    assert_eq!(
        HuggingFaceGenerateEndpoint.descriptor().streaming_path,
        Some("/generate_stream")
    );

    req.model_endpoint.endpoint.streaming = true;
    let parsed = HuggingFaceGenerateEndpoint
        .parse_response_with_config(
            &ServerResponse::from_json(
                10,
                json!({"token":{"text":"!"},"generated_text":"hello world!"}),
            ),
            &req.model_endpoint.endpoint,
        )
        .unwrap()
        .unwrap();
    assert_eq!(parsed.data, Some(ResponseData::Text { text: "!".into() }));

    req.model_endpoint.endpoint.streaming = false;
    let parsed = HuggingFaceGenerateEndpoint
        .parse_response_with_config(
            &ServerResponse::from_json(11, json!([{"generated_text":"complete"}])),
            &req.model_endpoint.endpoint,
        )
        .unwrap()
        .unwrap();
    assert_eq!(parsed.data.unwrap().get_text(), "complete");
}

#[test]
fn image_generation_and_edit_cover_streaming_full_and_multipart_descriptor_shapes() {
    let mut generation = request(
        EndpointType::ImageGeneration,
        vec![Turn {
            model: Some("flux".into()),
            texts: vec![text("draw a fox")],
            ..Turn::default()
        }],
    );
    generation.model_endpoint.endpoint.streaming = true;
    let body = plan_body(ImageGenerationEndpoint.format_payload(&generation).unwrap());
    assert_eq!(
        body,
        json!({"prompt":"draw a fox","model":"flux","response_format":"b64_json","n":1,"stream":true})
    );
    let partial = ImageGenerationEndpoint
        .parse_response(&ServerResponse::from_json(
            12,
            json!({"b64_json":"partial","partial_image_index":2}),
        ))
        .unwrap()
        .unwrap();
    let Some(ResponseData::Images(partial)) = partial.data else {
        panic!("expected images")
    };
    assert_eq!(partial.images[0].partial_image_index, Some(2));

    let mut edit = request(
        EndpointType::ImageEdit,
        vec![Turn {
            texts: vec![text("make it blue")],
            images: vec![Media::new(vec!["iVBORw0KGgoAAAA".into()])],
            extra_body: Some(Map::from_iter([
                ("prompt".into(), json!("hijack")),
                ("seed".into(), json!(5)),
            ])),
            ..Turn::default()
        }],
    );
    edit.model_endpoint.endpoint.extra = Some(Map::from_iter([("size".into(), json!("512x512"))]));
    let body = plan_body(ImageEditEndpoint.format_payload(&edit).unwrap());
    assert_eq!(body["prompt"], "make it blue");
    assert_eq!(body["image"]["content_type"], "image/png");
    assert_eq!(body["image"]["filename"], "image.png");
    assert_eq!(body["seed"], 5);
    assert!(ImageEditEndpoint.descriptor().requires_form_data);

    let full = ImageEditEndpoint
        .parse_response(&ServerResponse::from_json(
            13,
            json!({"data":[{"url":"https://cdn/result.png","revised_prompt":"blue"}],"usage":{"prompt_tokens":4}}),
        ))
        .unwrap()
        .unwrap();
    let Some(ResponseData::Images(full)) = full.data else {
        panic!("expected images")
    };
    assert_eq!(full.images[0].revised_prompt.as_deref(), Some("blue"));
}

#[test]
fn video_image_retrieval_and_solido_preserve_non_text_response_data() {
    let video = plan_body(
        VideoGenerationEndpoint
            .format_payload(&request(
                EndpointType::VideoGeneration,
                vec![Turn {
                    texts: vec![text("animate")],
                    extra_body: Some(Map::from_iter([("seconds".into(), json!(8))])),
                    ..Turn::default()
                }],
            ))
            .unwrap(),
    );
    assert_eq!(video["seconds"], 8);
    assert!(VideoGenerationEndpoint.descriptor().requires_polling);
    let parsed = VideoGenerationEndpoint
        .parse_response(&ServerResponse::from_json(
            14,
            json!({"id":"video-1","status":"completed","progress":100,"url":"/content","inference_time_s":1.5}),
        ))
        .unwrap()
        .unwrap();
    let Some(ResponseData::Video(video)) = parsed.data else {
        panic!("expected video")
    };
    assert_eq!(video.video_id.as_deref(), Some("video-1"));
    assert_eq!(video.status.as_deref(), Some("completed"));

    let retrieval_body = plan_body(
        ImageRetrievalEndpoint
            .format_payload(&request(
                EndpointType::ImageRetrieval,
                vec![Turn {
                    images: vec![Media::new(vec![
                        "data:image/png;base64,a".into(),
                        "data:image/png;base64,b".into(),
                    ])],
                    ..Turn::default()
                }],
            ))
            .unwrap(),
    );
    assert_eq!(
        ImageRetrievalEndpoint
            .extract_payload_inputs(&retrieval_body)
            .image_count,
        2
    );
    assert!(ImageRetrievalEndpoint.descriptor().requires_inline_media);
    let parsed = ImageRetrievalEndpoint
        .parse_response(&ServerResponse::from_json(
            15,
            json!({"data":[{"index":0,"bounding_boxes":{}}]}),
        ))
        .unwrap()
        .unwrap();
    assert!(matches!(
        parsed.data,
        Some(ResponseData::ImageRetrieval { .. })
    ));

    let solido = plan_body(
        SolidoRagEndpoint
            .format_payload(&request(
                EndpointType::SolidoRag,
                vec![Turn {
                    texts: vec![Media::new(vec!["one".into(), "two".into()])],
                    ..Turn::default()
                }],
            ))
            .unwrap(),
    );
    assert_eq!(solido["query"], json!(["one", "two"]));
    let parsed = SolidoRagEndpoint
        .parse_response(&ServerResponse::from_json(
            16,
            json!({"content":"answer","sources":[{"id":"doc"}]}),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(parsed.data.unwrap().get_text(), "answer");
    assert_eq!(parsed.sources, Some(json!([{"id":"doc"}])));
}

#[test]
fn raw_and_template_support_jmespath_autodetect_and_full_jinja_context() {
    let raw_body = json!({"custom":true});
    let raw_req = request(
        EndpointType::Raw,
        vec![Turn {
            raw_payload: Some(raw_body.clone()),
            ..Turn::default()
        }],
    );
    assert_eq!(
        plan_body(RawEndpoint.format_payload(&raw_req).unwrap()),
        raw_body
    );

    let mut raw_config = config(EndpointType::Raw);
    raw_config.response_field = Some("result.items".into());
    let parsed = RawEndpoint
        .parse_response_with_config(
            &ServerResponse::from_json(17, json!({"result":{"items":[{"index":0,"score":1.0}]}})),
            &raw_config,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(parsed.data, Some(ResponseData::Rankings { .. })));

    raw_config.response_field = Some("!!! invalid !!!".into());
    let parsed = RawEndpoint
        .parse_response_with_config(
            &ServerResponse::from_json(18, json!({"content":"fallback"})),
            &raw_config,
        )
        .unwrap()
        .unwrap();
    assert_eq!(parsed.data.unwrap().get_text(), "fallback");

    let mut template_req = request(
        EndpointType::Template,
        vec![Turn {
            model: Some("templated-model".into()),
            max_tokens: Some(12),
            texts: vec![named("query", &["what?"]), named("passages", &["a", "b"])],
            images: vec![named("main", &["data:image/png;base64,x"])],
            extra_body: Some(Map::from_iter([("seed".into(), json!(3))])),
            ..Turn::default()
        }],
    );
    template_req.model_endpoint.endpoint.template = Some(
        r#"{"query":{{ query|tojson }},"passages":{{ passages|tojson }},"model":{{ model|tojson }},"max_tokens":{{ max_tokens }},"correlation":{{ request_info.x_correlation_id|tojson }}}"#.into(),
    );
    template_req.model_endpoint.endpoint.extra =
        Some(Map::from_iter([("temperature".into(), json!(0.4))]));
    let rendered = plan_body(TemplateEndpoint.format_payload(&template_req).unwrap());
    assert_eq!(rendered["query"], "what?");
    assert_eq!(rendered["passages"], json!(["a", "b"]));
    assert_eq!(rendered["model"], "templated-model");
    assert_eq!(rendered["correlation"], "corr-1");
    assert_eq!(rendered["temperature"], 0.4);
    assert_eq!(rendered["seed"], 3);

    template_req.model_endpoint.endpoint.response_field = Some("result.text".into());
    let parsed = TemplateEndpoint
        .parse_response_with_config(
            &ServerResponse::from_json(19, json!({"result":{"text":"selected"}})),
            &template_req.model_endpoint.endpoint,
        )
        .unwrap()
        .unwrap();
    assert_eq!(parsed.data.unwrap().get_text(), "selected");
    assert!(
        TemplateEndpoint
            .parse_response_with_config(
                &ServerResponse::from_json(20, json!({"content":"must-not-fallback"})),
                &template_req.model_endpoint.endpoint,
            )
            .unwrap()
            .is_none()
    );
}
