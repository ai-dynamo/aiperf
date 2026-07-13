// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Separately packaged static extension linked only into runner process tests.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_dataset::{ConversationMetadata, Sampler, SamplerFactory, SessionId};
use aiperf_endpoints::{
    ChatEndpoint, EffectiveEndpointConfig, EndpointDescriptor, EndpointFactory, EndpointResult,
    ExtractedPayload, Modality, ParsedResponse, PreparedEndpoint, PreparedRequest, ReadinessPolicy,
    RequestRecord, ServerResponse, StatelessEndpointFactory, Turn,
};
use aiperf_extensions::{AiperfExtension, AiperfRegistry, AiperfRegistryFactory, ExtensionError};
use aiperf_rng::RngRoot;

/// Stable package identity advertised by the fixture extension.
pub const EXTENSION_NAME: &str = "runner-static-extension-process-proof";
/// Sampler identity deliberately absent from every built-in registry.
pub const SAMPLER_NAME: &str = "linked_pinned";
/// Prepared-only endpoint identity deliberately absent from the legacy enum.
pub const PREPARED_ONLY_ENDPOINT_ID: &str = "linked_prepared_chat";

static REGISTRY_BUILDS: AtomicUsize = AtomicUsize::new(0);
static SAMPLER_CREATIONS: AtomicUsize = AtomicUsize::new(0);
static SAMPLER_NEXT_CALLS: AtomicUsize = AtomicUsize::new(0);
static ENDPOINT_PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
static ENDPOINT_FORMAT_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Calls observed inside the fresh process that owns this linked package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionEvidence {
    /// Number of times the composition root invoked the registry factory.
    pub registry_builds: usize,
    /// Number of stateful samplers constructed through the registered trait.
    pub sampler_creations: usize,
    /// Number of sample selections served by the custom implementation.
    pub sampler_next_calls: usize,
    /// Number of prepared-only endpoint bindings constructed.
    pub endpoint_preparations: usize,
    /// Number of request bodies materialized through the prepared-only endpoint.
    pub endpoint_format_calls: usize,
}

/// Clear process-local evidence before one coordinator acceptance run.
pub fn reset_evidence() {
    REGISTRY_BUILDS.store(0, Ordering::SeqCst);
    SAMPLER_CREATIONS.store(0, Ordering::SeqCst);
    SAMPLER_NEXT_CALLS.store(0, Ordering::SeqCst);
    ENDPOINT_PREPARATIONS.store(0, Ordering::SeqCst);
    ENDPOINT_FORMAT_CALLS.store(0, Ordering::SeqCst);
}

/// Snapshot process-local factory and trait invocation evidence.
pub fn evidence() -> ExtensionEvidence {
    ExtensionEvidence {
        registry_builds: REGISTRY_BUILDS.load(Ordering::SeqCst),
        sampler_creations: SAMPLER_CREATIONS.load(Ordering::SeqCst),
        sampler_next_calls: SAMPLER_NEXT_CALLS.load(Ordering::SeqCst),
        endpoint_preparations: ENDPOINT_PREPARATIONS.load(Ordering::SeqCst),
        endpoint_format_calls: ENDPOINT_FORMAT_CALLS.load(Ordering::SeqCst),
    }
}

static PREPARED_ONLY_ENDPOINT_DESCRIPTOR: EndpointDescriptor = EndpointDescriptor {
    id: PREPARED_ONLY_ENDPOINT_ID,
    aliases: &["linked_prepared_chat_alias"],
    description: "prepared-only scheduled runner process fixture",
    endpoint_path: Some("/v1/chat/completions"),
    streaming_path: Some("/v1/chat/completions"),
    supports_streaming: true,
    produces_tokens: true,
    tokenizes_input: true,
    requires_raw_token_ids: false,
    requires_form_data: false,
    requires_polling: false,
    requires_inline_media: false,
    input_modalities: &[Modality::Text],
    output_modalities: &[Modality::Tokens],
    metrics_title: "Prepared-only Fixture Metrics",
    service_kind: "prepared-only-fixture",
};

#[derive(Clone, Copy, Debug)]
struct PreparedOnlyEndpointFactory;

impl EndpointFactory for PreparedOnlyEndpointFactory {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &PREPARED_ONLY_ENDPOINT_DESCRIPTOR
    }

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
        ENDPOINT_PREPARATIONS.fetch_add(1, Ordering::SeqCst);
        let inner = StatelessEndpointFactory::new(ChatEndpoint).prepare(config)?;
        Ok(Box::new(PreparedOnlyEndpoint { inner }))
    }
}

#[derive(Debug)]
struct PreparedOnlyEndpoint {
    inner: Box<dyn PreparedEndpoint>,
}

impl PreparedEndpoint for PreparedOnlyEndpoint {
    fn descriptor(&self) -> &'static EndpointDescriptor {
        &PREPARED_ONLY_ENDPOINT_DESCRIPTOR
    }

    fn config(&self) -> &EffectiveEndpointConfig {
        self.inner.config()
    }

    fn format_payload(&self, request: &PreparedRequest<'_>) -> EndpointResult<serde_json::Value> {
        ENDPOINT_FORMAT_CALLS.fetch_add(1, Ordering::SeqCst);
        self.inner.format_payload(request)
    }

    fn headers(&self) -> &BTreeMap<String, String> {
        self.inner.headers()
    }

    fn readiness_policy(&self, model: &str) -> EndpointResult<ReadinessPolicy> {
        self.inner.readiness_policy(model)
    }

    fn parse_response(&self, response: &ServerResponse) -> EndpointResult<Option<ParsedResponse>> {
        self.inner.parse_response(response)
    }

    fn extract_payload_inputs(&self, body: &serde_json::Value) -> ExtractedPayload {
        self.inner.extract_payload_inputs(body)
    }

    fn extract_response_data(&self, record: &RequestRecord) -> EndpointResult<Vec<ParsedResponse>> {
        self.inner.extract_response_data(record)
    }

    fn build_assistant_turn(&self, record: &RequestRecord) -> EndpointResult<Option<Turn>> {
        self.inner.build_assistant_turn(record)
    }

    fn captures_assistant_turn(&self) -> bool {
        self.inner.captures_assistant_turn()
    }
}

#[derive(Clone, Copy, Debug)]
struct LinkedPinnedSamplerFactory;

impl SamplerFactory for LinkedPinnedSamplerFactory {
    fn name(&self) -> &str {
        SAMPLER_NAME
    }

    fn create(
        &self,
        metadata: &[ConversationMetadata],
        _root: RngRoot,
    ) -> aiperf_dataset::Result<Box<dyn Sampler>> {
        SAMPLER_CREATIONS.fetch_add(1, Ordering::SeqCst);
        let id = metadata
            .last()
            .map(|conversation| conversation.conversation_id.clone())
            .ok_or(aiperf_dataset::DatasetError::EmptySampler)?;
        Ok(Box::new(LinkedPinnedSampler { id }))
    }
}

struct LinkedPinnedSampler {
    id: SessionId,
}

impl Sampler for LinkedPinnedSampler {
    fn next(&mut self) -> SessionId {
        SAMPLER_NEXT_CALLS.fetch_add(1, Ordering::SeqCst);
        self.id.clone()
    }
}

#[derive(Clone, Copy, Debug)]
struct LinkedTestExtension;

impl AiperfExtension for LinkedTestExtension {
    fn name(&self) -> &str {
        EXTENSION_NAME
    }

    fn register(&self, registry: &mut AiperfRegistry) -> Result<(), ExtensionError> {
        registry
            .samplers_mut()
            .register(LinkedPinnedSamplerFactory)?;
        registry.register_endpoint_factory(PreparedOnlyEndpointFactory)?;
        Ok(())
    }
}

/// Custom-distribution registry factory supplied to the real v2 coordinator.
#[derive(Clone, Copy, Debug)]
pub struct StaticTestRegistryFactory;

impl AiperfRegistryFactory for StaticTestRegistryFactory {
    fn build(&self) -> Result<AiperfRegistry, ExtensionError> {
        REGISTRY_BUILDS.fetch_add(1, Ordering::SeqCst);
        AiperfRegistry::builtin()?.with_extensions([&LinkedTestExtension as &dyn AiperfExtension])
    }
}
