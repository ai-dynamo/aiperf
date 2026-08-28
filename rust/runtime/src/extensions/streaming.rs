// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in streaming source, format, and checkpoint-backend registration.
//!
//! The streaming registries hold two kinds of implementation. Most compiled
//! adapters are pure startup values (`local`, `s3`, `baseten`) and register
//! unconditionally. The rest need authority the registry cannot invent — a
//! credentialed Hugging Face transport, the resolved run tokenizer, a frozen
//! synthesis-profile receipt — and are registered only when the composition
//! root supplies it. An unsupplied binding leaves its identifier absent, so
//! selecting it fails closed at validation instead of at preparation.

use std::sync::Arc;

use super::{AIPerfExtension, AIPerfRegistry, ExtensionError};
use crate::streaming::{
    checkpoints::local::LocalCheckpointBackendFactory,
    formats::{streaming_dynamo::StreamingDynamoFormatFactory, synthesis::SynthesisFormatFactory},
    identity::ContentDigest,
    sources::{builtin_source_factories, hf_rows::HfPageTransportFactory},
};

/// Built-in streaming dataset adapters and the local checkpoint backend.
///
/// Constructed empty for the stock distribution; a composition root that has
/// already resolved a host binding attaches it with the `with_*` methods before
/// applying the extension.
#[derive(Debug, Default)]
pub struct BuiltinStreamingExtension {
    hf_page_transport: Option<Arc<dyn HfPageTransportFactory>>,
    synthesis: Option<Arc<SynthesisFormatFactory>>,
    streaming_dynamo_profile_digest: Option<ContentDigest>,
}

impl BuiltinStreamingExtension {
    /// The stock set: every adapter that needs no run-scoped host authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hf_page_transport: None,
            synthesis: None,
            streaming_dynamo_profile_digest: None,
        }
    }

    /// Enable the `hf_rows` source against the host's credential and HTTP
    /// authority.
    #[must_use]
    pub fn with_hf_page_transport(mut self, factory: Arc<dyn HfPageTransportFactory>) -> Self {
        self.hf_page_transport = Some(factory);
        self
    }

    /// Enable the `synthesis` format against the resolved run tokenizer.
    ///
    /// The factory is built by the host because it owns the tokenizer, its
    /// semantic receipt, and the budgets every decoder charges against.
    #[must_use]
    pub fn with_synthesis_format(mut self, factory: SynthesisFormatFactory) -> Self {
        self.synthesis = Some(Arc::new(factory));
        self
    }

    /// Enable the `streaming_dynamo` format against one frozen
    /// synthesis-profile receipt.
    #[must_use]
    pub fn with_streaming_dynamo_profile(mut self, profile_digest: ContentDigest) -> Self {
        self.streaming_dynamo_profile_digest = Some(profile_digest);
        self
    }
}

impl AIPerfExtension for BuiltinStreamingExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.streaming"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        for factory in builtin_source_factories(self.hf_page_transport.clone()) {
            registry
                .register_stream_source(Arc::from(factory))
                .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        }

        #[cfg(feature = "parquet")]
        registry
            .register_stream_format(Arc::new(
                crate::streaming::formats::baseten::BasetenFormatFactory,
            ))
            .map_err(|error| ExtensionError::rejected(error.to_string()))?;

        if let Some(profile_digest) = self.streaming_dynamo_profile_digest {
            registry
                .register_stream_format(Arc::new(StreamingDynamoFormatFactory::new(profile_digest)))
                .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        }

        if let Some(synthesis) = &self.synthesis {
            registry
                .register_stream_format(Arc::clone(synthesis) as Arc<_>)
                .map_err(|error| ExtensionError::rejected(error.to_string()))?;
        }

        registry
            .register_stream_checkpoint_backend(Arc::new(LocalCheckpointBackendFactory))
            .map_err(|error| ExtensionError::rejected(error.to_string()))
    }
}
