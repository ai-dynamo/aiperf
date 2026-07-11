// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate proof that a linked package can implement and register a leaf trait.

use aiperf_dataset::{ConversationMetadata, EndpointResolver, Sampler, SamplerFactory, SessionId};
use aiperf_endpoints::{ChatEndpoint, EndpointType};
use aiperf_extensions::{AiperfExtension, AiperfRegistry, ExtensionError};
use aiperf_rng::RngRoot;

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
    ) -> aiperf_dataset::Result<Box<dyn Sampler>> {
        let id = metadata
            .first()
            .map(|metadata| metadata.conversation_id.clone())
            .unwrap_or_else(|| SessionId::from("external"));
        Ok(Box::new(ExternalSampler { id }))
    }
}

struct ExternalExtension;

impl AiperfExtension for ExternalExtension {
    fn name(&self) -> &str {
        "external-test"
    }

    fn register(&self, registry: &mut AiperfRegistry) -> Result<(), ExtensionError> {
        registry
            .samplers_mut()
            .register(ExternalSamplerFactory { name: "external" })?;
        registry
            .endpoints_mut()
            .register("external-chat", ChatEndpoint)?;
        Ok(())
    }
}

struct PartiallyFailingExtension;

impl AiperfExtension for PartiallyFailingExtension {
    fn name(&self) -> &str {
        "partial"
    }

    fn register(&self, registry: &mut AiperfRegistry) -> Result<(), ExtensionError> {
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
    let mut registry = AiperfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let mut sampler = registry
        .samplers()
        .create("external", &metadata(), RngRoot::new(Some(7)))
        .unwrap();
    assert_eq!(sampler.next().as_str(), "conversation-1");
    assert_eq!(
        registry
            .endpoints()
            .resolve(Some("external-chat"))
            .unwrap()
            .metadata()
            .endpoint_type,
        EndpointType::Chat
    );
    assert_eq!(
        registry.extension_names().collect::<Vec<_>>(),
        ["external-test"]
    );
}

#[test]
fn duplicate_extension_name_is_rejected() {
    let mut registry = AiperfRegistry::builtin().unwrap();
    registry.register_extension(&ExternalExtension).unwrap();

    let error = registry.register_extension(&ExternalExtension).unwrap_err();
    assert!(error.to_string().contains("duplicate AIPerf extension"));
}

#[test]
fn failed_extension_does_not_leak_earlier_registrations() {
    let mut registry = AiperfRegistry::builtin().unwrap();
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
    assert_eq!(registry.extension_names().len(), 0);
}
