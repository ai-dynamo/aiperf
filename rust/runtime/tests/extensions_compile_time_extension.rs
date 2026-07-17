// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate extension registration contracts.

use std::collections::BTreeMap;

use aiperf_runtime::dataset::{ConversationMetadata, Sampler, SamplerFactory, SessionId};
use aiperf_runtime::endpoints::{
    CreditPhase, EffectiveEndpointConfig, EndpointDescriptor, EndpointFactory, EndpointId,
    EndpointResult, ExtractedPayload, Media, Modality, ParsedResponse, PreparedEndpoint,
    PreparedRequest, RawEndpointConfig, ReadinessPolicy, RequestRecord, ResponseData,
    ServerResponse, Turn,
};
use aiperf_runtime::extensions::{
    AIPerfExtension, AIPerfRegistry, AIPerfRegistryFactory, ExtensionError,
};
use aiperf_runtime::rng::RngRoot;

struct ExternalSampler {
    id: SessionId,
}

impl Sampler for ExternalSampler {
    fn next(&mut self) -> SessionId {
        self.id.clone()
    }
}

#[derive(Clone, Copy)]
struct ExternalSamplerFactory {
    name: &'static str,
}

impl SamplerFactory for ExternalSamplerFactory {
    fn name(&self) -> &str {
        self.name
    }

    fn create(
        &self,
        metadata: &[ConversationMetadata],
        _root: RngRoot,
    ) -> aiperf_runtime::dataset::Result<Box<dyn Sampler>> {
        let id = metadata
            .first()
            .map(|metadata| metadata.conversation_id.clone())
            .unwrap_or_else(|| SessionId::from("external"));
        Ok(Box::new(ExternalSampler { id }))
    }
}

struct ExternalExtension;

static EXTERNAL_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: "external_echo",
    aliases: &["external_echo_v1"],
    description: "Test-only compiled echo dialect",
    endpoint_path: Some("/v1/external/echo"),
    streaming_path: None,
    supports_streaming: false,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "External Echo Metrics",
    service_kind: "external_echo",
};

#[derive(Debug, Clone, Copy)]
struct ExternalEndpointFactory;

impl EndpointFactory for ExternalEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &EXTERNAL_DESCRIPTOR
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        Ok(Box::new(ExternalPreparedEndpoint {
            config,
            headers: BTreeMap::from([("x-external".into(), "echo".into())]),
        }))
    }
}

#[derive(Debug)]
struct ExternalPreparedEndpoint {
    config: EffectiveEndpointConfig,
    headers: BTreeMap<String, String>,
}

impl PreparedEndpoint for ExternalPreparedEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &EXTERNAL_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        &self.config
    }

    fn format_payload(
        &self,
        request: &PreparedRequest<'_>,
    ) -> EndpointResult<aiperf_runtime::body_plan::BodyPlan> {
        let text = request
            .turns()
            .last()
            .and_then(|turn| turn.texts.first())
            .and_then(|media| media.contents.first())
            .cloned()
            .unwrap_or_default();
        let payload = serde_json::json!({
            "model": request.primary_model_name(),
            "echo": text,
            "stream": self.config.streaming(),
        });
        Ok(aiperf_runtime::body_plan::BodyPlan::from_object(
            payload.as_object().expect("external payload is an object"),
        )?)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    fn readiness_policy(&self, _model: &str) -> EndpointResult<ReadinessPolicy> {
        Ok(ReadinessPolicy::Unsupported {
            reason: "test dialect has no readiness request",
        })
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        Ok(response
            .json
            .as_ref()
            .and_then(|value| value.get("echo"))
            .and_then(serde_json::Value::as_str)
            .map(|text| ParsedResponse {
                perf_ns: response.perf_ns,
                data: Some(ResponseData::Text { text: text.into() }),
                usage: None,
                sources: None,
            }))
    }

    fn extract_payload_inputs(&self, body: &serde_json::Value) -> ExtractedPayload {
        ExtractedPayload {
            texts: body
                .get("echo")
                .and_then(serde_json::Value::as_str)
                .map(|text| vec![text.into()])
                .unwrap_or_default(),
            ..ExtractedPayload::default()
        }
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        record
            .responses
            .iter()
            .filter_map(|response| self.parse_response(response).transpose())
            .collect()
    }

    fn build_assistant_turn(&self, _record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        Ok(None)
    }

    fn captures_assistant_turn(&self) -> bool {
        false
    }
}

impl AIPerfExtension for ExternalExtension {
    fn name(&self) -> &str {
        "external-test"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "external" })?;
        registry.register_endpoint_factory(ExternalEndpointFactory)?;
        Ok(())
    }
}

struct ExternalRegistryFactory;

impl AIPerfRegistryFactory for ExternalRegistryFactory {
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError> {
        AIPerfRegistry::builtin()?.with_extensions([&ExternalExtension as &dyn AIPerfExtension])
    }
}

struct PartiallyFailingExtension;

impl AIPerfExtension for PartiallyFailingExtension {
    fn name(&self) -> &str {
        "partial"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.register_endpoint_factory(ExternalEndpointFactory)?;
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "staged" })?;
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "random" })?;
        Ok(())
    }
}

fn metadata() -> Vec<ConversationMetadata> {
    vec![ConversationMetadata {
        conversation_id: SessionId::from("conversation-1"),
        turns: Vec::new(),
        context_mode: None,
        accuracy: None,
        dag: None,
    }]
}

#[test]
fn linked_extension_registers_and_resolves_a_trait_implementation() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let mut sampler = registry
        .samplers()
        .create("external", &metadata(), RngRoot::new(Some(7)))
        .unwrap();
    assert_eq!(sampler.next().as_str(), "conversation-1");
    let alias = EndpointId::new("external_echo_v1").unwrap();
    assert_eq!(
        registry.endpoints().canonical_id(&alias).unwrap().as_str(),
        "external_echo"
    );
    let prepared = registry
        .endpoints()
        .prepare(&alias, RawEndpointConfig::default())
        .unwrap();
    let turn = Turn {
        texts: vec![Media::new(vec!["hello".into()])],
        ..Turn::default()
    };
    let request = PreparedRequest::new(
        "external-model",
        std::slice::from_ref(&turn),
        None,
        None,
        CreditPhase::Profiling,
        None,
        None,
        None,
    );
    let body: serde_json::Value = serde_json::from_slice(
        &prepared
            .format_payload(&request)
            .unwrap()
            .materialize_standalone()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        body,
        serde_json::json!({"model":"external-model","echo":"hello","stream":false})
    );
    assert_eq!(prepared.headers()["x-external"], "echo");
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        ["external-test"]
    );
}

#[test]
fn custom_distribution_builds_its_registry_through_the_factory_seam() {
    let registry = ExternalRegistryFactory.build().unwrap();
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        ["external-test"]
    );
    assert!(
        registry
            .samplers()
            .create("external", &metadata(), RngRoot::new(Some(7)))
            .is_ok()
    );
}

#[test]
fn duplicate_extension_name_is_rejected() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let error = registry.register_extension(&ExternalExtension).unwrap_err();
    assert!(error.to_string().contains("duplicate AIPerf extension"));
}

#[test]
fn failed_extension_does_not_leak_earlier_registrations() {
    let mut registry = AIPerfRegistry::builtin().unwrap();
    let error = registry
        .register_extension(&PartiallyFailingExtension)
        .unwrap_err();
    assert!(error.to_string().contains("duplicate sampler strategy"));
    assert!(
        registry
            .samplers()
            .create("staged", &metadata(), RngRoot::new(Some(7)))
            .is_err()
    );
    assert!(
        registry
            .endpoints()
            .canonical_id(&EndpointId::new("external_echo").unwrap())
            .is_err()
    );
    assert_eq!(registry.extension_names().len(), 0);
}
