// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry, preparation, and protocol-v1 compatibility invariants.

use std::collections::BTreeMap;

use aiperf_runtime::endpoints::{
    CreditPhase, EffectiveEndpointConfig, EndpointConfig, EndpointDescriptor, EndpointFactory,
    EndpointId, EndpointRegistry, EndpointRegistryBuilder, EndpointRegistryError, EndpointResult,
    EndpointType, Media, Modality, PreparedEndpoint, PreparedEndpointTable, PreparedRequest,
    RawEndpointConfig, ReadinessPolicy, ResponseData, ServerResponse, Turn,
};
use serde_json::{Value, json};

/// Materialize a prepared endpoint's [`BodyPlan`] into a decoded JSON value so
/// the structural assertions below keep inspecting fields as before stage B.
fn plan_body(plan: aiperf_runtime::body_plan::BodyPlan) -> Value {
    serde_json::from_slice(&plan.materialize_standalone().unwrap()).unwrap()
}

#[derive(Debug, Clone, Copy)]
struct RegistrationOnlyFactory(&'static EndpointDescriptor);

impl EndpointFactory for RegistrationOnlyFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        self.0
    }

    fn prepare(
        &self,
        _config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        unreachable!("registration-only test factory is never prepared")
    }
}

const ALPHA: EndpointDescriptor = descriptor("alpha", &["alpha_v1", "shared"]);
const BETA_WITH_COLLIDING_ALIAS: EndpointDescriptor = descriptor("beta", &["shared"]);
const SHARED_CANONICAL: EndpointDescriptor = descriptor("shared", &[]);
const INTERNAL_DUPLICATE: EndpointDescriptor = descriptor("duplicate", &["duplicate"]);

const fn descriptor(id: &'static str, aliases: &'static [&'static str]) -> EndpointDescriptor {
    EndpointDescriptor {
        id,
        aliases,
        description: "test-only endpoint",
        endpoint_path: Some("/test"),
        streaming_path: None,
        supports_streaming: true,
        produces_tokens: true,
        tokenizes_input: true,
        requires_raw_token_ids: false,
        requires_form_data: false,
        requires_polling: false,
        requires_inline_media: false,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Tokens],
        metrics_title: "Test Metrics",
        service_kind: "test",
    }
}

fn prepared_request<'a>(turns: &'a [Turn]) -> PreparedRequest<'a> {
    PreparedRequest::new(
        "model-a",
        turns,
        Some("system"),
        Some("context"),
        CreditPhase::Profiling,
        Some("request-1"),
        Some("correlation-1"),
        Some("conversation-1"),
    )
}

#[test]
fn endpoint_id_accepts_only_the_open_canonical_grammar() {
    for accepted in ["a", "chat", "chat_2", "a0_b9"] {
        assert_eq!(EndpointId::new(accepted).unwrap().as_str(), accepted);
        assert_eq!(
            serde_json::from_value::<EndpointId>(json!(accepted))
                .unwrap()
                .as_str(),
            accepted
        );
    }
    for rejected in [
        "",
        "Chat",
        "2chat",
        "_chat",
        "chat-completions",
        "chat ",
        "chät",
    ] {
        let error = EndpointId::new(rejected).unwrap_err();
        assert_eq!(error.value(), rejected);
        assert!(error.to_string().contains("[a-z][a-z0-9_]*"));
    }
}

#[test]
fn registry_order_aliases_and_unknown_diagnostics_are_deterministic() {
    let registry = EndpointRegistry::builtin().unwrap();
    let ids = registry
        .canonical_ids()
        .map(EndpointId::as_str)
        .collect::<Vec<_>>();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    assert!(!ids.contains(&"chat_completions"));

    let alias = EndpointId::new("chat_completions").unwrap();
    assert_eq!(registry.canonical_id(&alias).unwrap().as_str(), "chat");

    let unknown = EndpointId::new("compiled_elsewhere").unwrap();
    let EndpointRegistryError::UnknownEndpoint { available, .. } =
        registry.resolve_factory(&unknown).unwrap_err()
    else {
        panic!("expected an unknown-endpoint diagnostic");
    };
    assert_eq!(
        available.iter().map(EndpointId::as_str).collect::<Vec<_>>(),
        ids
    );
}

#[test]
fn duplicate_ids_and_aliases_fail_atomically() {
    let mut builder = EndpointRegistryBuilder::new();
    builder
        .register_factory(RegistrationOnlyFactory(&ALPHA))
        .unwrap();

    for descriptor in [
        &BETA_WITH_COLLIDING_ALIAS,
        &SHARED_CANONICAL,
        &INTERNAL_DUPLICATE,
    ] {
        let error = builder
            .register_factory(RegistrationOnlyFactory(descriptor))
            .unwrap_err();
        assert!(matches!(error, EndpointRegistryError::DuplicateName { .. }));
    }

    let registry = builder.freeze();
    assert_eq!(
        registry
            .canonical_ids()
            .map(EndpointId::as_str)
            .collect::<Vec<_>>(),
        ["alpha"]
    );
    for alias in ["alpha_v1", "shared"] {
        assert_eq!(
            registry
                .canonical_id(&EndpointId::new(alias).unwrap())
                .unwrap()
                .as_str(),
            "alpha"
        );
    }
}

#[test]
fn raw_config_has_no_identity_and_round_trips_through_v1_policy() {
    let raw = RawEndpointConfig {
        urls: vec!["https://example.test".into()],
        path: Some("/custom".into()),
        streaming: true,
        response_field: Some("answer.text".into()),
        headers: BTreeMap::from([("x-test".into(), "value".into())]),
        api_key: Some("secret".into()),
        extra: Some(serde_json::Map::from_iter([(
            "temperature".into(),
            json!(0.2),
        )])),
        ..RawEndpointConfig::default()
    };
    let serialized = serde_json::to_value(&raw).unwrap();
    assert!(serialized.get("type").is_none());
    assert!(serialized.get("endpoint_type").is_none());
    assert!(serialized.get("headers").is_none());
    assert!(serialized.get("api_key").is_none());

    let legacy = EndpointConfig::from_raw(EndpointType::Messages, raw.clone());
    assert_eq!(legacy.endpoint_type, EndpointType::Messages);
    assert_eq!(RawEndpointConfig::from(&legacy), raw);
    assert_eq!(RawEndpointConfig::from(legacy), raw);
}

#[test]
fn prepared_dispatch_uses_only_its_bound_config_and_dense_key() {
    let registry = EndpointRegistry::builtin().unwrap();
    let chat = EndpointId::new("chat").unwrap();
    let mut authored = RawEndpointConfig {
        streaming: true,
        use_server_token_count: true,
        ..RawEndpointConfig::default()
    };
    let prepared = registry.prepare(&chat, authored.clone()).unwrap();
    authored.streaming = false;
    authored.use_server_token_count = false;

    let turns = [Turn {
        texts: vec![Media::new(vec!["hello".into()])],
        ..Turn::default()
    }];
    let request = prepared_request(&turns);
    let payload = plan_body(prepared.format_payload(&request).unwrap());
    assert_eq!(payload["stream"], Value::Bool(true));
    assert_eq!(
        payload["stream_options"]["include_usage"],
        Value::Bool(true)
    );

    // The chat endpoint now honors `wait_for_model_mode` (default "inference"),
    // so its prepared readiness policy is a concrete probe request rather than
    // the historical `Unsupported` placeholder.
    assert!(matches!(
        prepared.readiness_policy("model-a").unwrap(),
        ReadinessPolicy::Request(_)
    ));

    let mut table = PreparedEndpointTable::new();
    let key = table.push(prepared).unwrap();
    assert_eq!(key.index(), 0);
    assert_eq!(table.get(key).unwrap().descriptor().id, "chat");
}

#[test]
fn flexible_endpoints_compile_and_reuse_profile_state_during_preparation() {
    let registry = EndpointRegistry::builtin().unwrap();
    let template = EndpointId::new("template").unwrap();
    let error = registry
        .prepare(
            &template,
            RawEndpointConfig {
                template: Some(r#"{"broken": {{"#.into()),
                ..RawEndpointConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("compile payload template"));

    let prepared_template = registry
        .prepare(
            &template,
            RawEndpointConfig {
                template: Some(
                    r#"{"model": {{ model|tojson }}, "text": {{ text|tojson }}}"#.into(),
                ),
                response_field: Some("answer".into()),
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let turns = [Turn {
        texts: vec![Media::new(vec!["hello".into()])],
        ..Turn::default()
    }];
    let request = prepared_request(&turns);
    for _ in 0..2 {
        assert_eq!(
            plan_body(prepared_template.format_payload(&request).unwrap()),
            json!({"model":"model-a","text":"hello"})
        );
    }
    assert_eq!(
        prepared_template
            .parse_response(&ServerResponse::from_json(
                7,
                json!({"answer":"selected","text":"fallback"}),
            ))
            .unwrap()
            .unwrap()
            .data,
        Some(ResponseData::Text {
            text: "selected".into()
        })
    );

    let raw = EndpointId::new("raw").unwrap();
    let prepared_raw = registry
        .prepare(
            &raw,
            RawEndpointConfig {
                response_field: Some("[invalid".into()),
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    assert_eq!(
        prepared_raw
            .parse_response(&ServerResponse::from_json(8, json!({"text":"fallback"})))
            .unwrap()
            .unwrap()
            .data,
        Some(ResponseData::Text {
            text: "fallback".into()
        })
    );
}

#[test]
fn adapter_descriptors_round_trip_closed_endpoint_types() {
    let registry = EndpointRegistry::builtin().unwrap();
    let endpoint_types = [
        EndpointType::Chat,
        EndpointType::Completions,
        EndpointType::Responses,
        EndpointType::Messages,
        EndpointType::Embeddings,
        EndpointType::ChatEmbeddings,
        EndpointType::NimEmbeddings,
        EndpointType::CohereRankings,
        EndpointType::HfTeiRankings,
        EndpointType::NimRankings,
        EndpointType::HuggingfaceGenerate,
        EndpointType::ImageGeneration,
        EndpointType::ImageEdit,
        EndpointType::VideoGeneration,
        EndpointType::ImageRetrieval,
        EndpointType::SolidoRag,
        EndpointType::Raw,
        EndpointType::Template,
    ];

    for endpoint_type in endpoint_types {
        let id = EndpointId::new(endpoint_type.canonical_id()).unwrap();
        let descriptor = registry.resolve_factory(&id).unwrap().descriptor();
        assert_eq!(descriptor.legacy_type(), Some(endpoint_type), "{id}");
        assert_eq!(descriptor.id, endpoint_type.canonical_id(), "{id}");
        assert!(!descriptor.description.is_empty(), "{id}");
        assert!(!descriptor.metrics_title.is_empty(), "{id}");
        assert!(!descriptor.service_kind.is_empty(), "{id}");
    }
}
