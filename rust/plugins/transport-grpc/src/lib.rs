// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport candidate plugin.
//!
//! Registers KServe OIP and NVIDIA Riva ASR/TTS/NLP gRPC transport
//! capabilities with canonical ID and effective aliases. Static production
//! gRPC transport authority remains unchanged until Task 39a.

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
        "nvidia/transport-grpc",
        env!("CARGO_PKG_VERSION"),
        "gRPC transport candidate: KServe OIP, Riva ASR/TTS/NLP",
    )
    .expect("nvidia/transport-grpc id must normalize")
});

struct GrpcTransportExtension;

impl AIPerfExtension for GrpcTransportExtension {
    fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        // Candidate shell: capabilities declared in plugins.yaml.in.
        // Static production registration remains until Task 39a.
        Ok(())
    }
}

static EXT: GrpcTransportExtension = GrpcTransportExtension;

#[aiperf_plugin]
fn transport_grpc_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
