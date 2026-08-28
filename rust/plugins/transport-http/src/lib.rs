// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP transport candidate plugin.
//!
//! Registers HTTP/1, h2c, UDS, TLS, and SSE transport capabilities with
//! canonical ID and effective aliases. Static production HTTP transport
//! authority remains unchanged until Task 39a.

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
        "nvidia/transport-http",
        env!("CARGO_PKG_VERSION"),
        "HTTP transport candidate: HTTP/1, h2c, UDS, TLS, SSE",
    )
    .expect("nvidia/transport-http id must normalize")
});

struct HttpTransportExtension;

impl AIPerfExtension for HttpTransportExtension {
    fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        // Candidate shell: capabilities declared in plugins.yaml.in.
        // Static production registration remains until Task 39a.
        Ok(())
    }
}

static EXT: HttpTransportExtension = HttpTransportExtension;

#[aiperf_plugin]
fn transport_http_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
