// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Separately packaged static extension linked only into runner process tests.

use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_dataset::{ConversationMetadata, Sampler, SamplerFactory, SessionId};
use aiperf_extensions::{AiperfExtension, AiperfRegistry, AiperfRegistryFactory, ExtensionError};
use aiperf_rng::RngRoot;

/// Stable package identity advertised by the fixture extension.
pub const EXTENSION_NAME: &str = "runner-static-extension-process-proof";
/// Sampler identity deliberately absent from every built-in registry.
pub const SAMPLER_NAME: &str = "linked_pinned";

static REGISTRY_BUILDS: AtomicUsize = AtomicUsize::new(0);
static SAMPLER_CREATIONS: AtomicUsize = AtomicUsize::new(0);
static SAMPLER_NEXT_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Calls observed inside the fresh process that owns this linked package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionEvidence {
    /// Number of times the composition root invoked the registry factory.
    pub registry_builds: usize,
    /// Number of stateful samplers constructed through the registered trait.
    pub sampler_creations: usize,
    /// Number of sample selections served by the custom implementation.
    pub sampler_next_calls: usize,
}

/// Clear process-local evidence before one coordinator acceptance run.
pub fn reset_evidence() {
    REGISTRY_BUILDS.store(0, Ordering::SeqCst);
    SAMPLER_CREATIONS.store(0, Ordering::SeqCst);
    SAMPLER_NEXT_CALLS.store(0, Ordering::SeqCst);
}

/// Snapshot process-local factory and trait invocation evidence.
pub fn evidence() -> ExtensionEvidence {
    ExtensionEvidence {
        registry_builds: REGISTRY_BUILDS.load(Ordering::SeqCst),
        sampler_creations: SAMPLER_CREATIONS.load(Ordering::SeqCst),
        sampler_next_calls: SAMPLER_NEXT_CALLS.load(Ordering::SeqCst),
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
