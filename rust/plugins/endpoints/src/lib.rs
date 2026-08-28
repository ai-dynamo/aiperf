// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint factory candidate plugin.
//!
//! Registers Chat, Responses, Completions, Embeddings, and SageMaker endpoint
//! factories with canonical IDs and effective aliases. Static production
//! endpoint authority remains unchanged until Task 39a.

use std::sync::LazyLock;

use aiperf_plugin_api::{
    descriptor::PluginPackageDescriptor,
    error::ExtensionError,
    extension::{AIPerfExtension, PluginRegistrar},
};
use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

/// The source API version exposed by this plugin candidate.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored(
        "nvidia/endpoints",
        env!("CARGO_PKG_VERSION"),
        "Endpoint factory candidate: Chat, Responses, Completions, Embeddings, SageMaker",
    )
    .expect("nvidia/endpoints id must normalize")
});

struct EndpointExtension;

impl AIPerfExtension for EndpointExtension {
    fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        // Candidate shell: capabilities declared in plugins.yaml.in.
        // Static production registration remains until Task 39a.
        Ok(())
    }
}

static EXT: EndpointExtension = EndpointExtension;

#[aiperf_plugin]
fn endpoints_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
