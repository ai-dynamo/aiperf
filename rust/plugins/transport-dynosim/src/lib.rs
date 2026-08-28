// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynosim transport candidate plugin.
//!
//! Stages the socket-free DES-clock Dynamo replay transports as one loadable
//! native package. Two capabilities are declared: canonical ID
//! `dynosim_offline` with effective aliases `dynosim` and `dynamo.offline`, and
//! canonical ID `dynosim_online` with effective alias `dynamo.online`. The
//! authoritative capability list lives in this package's `plugins.yaml.in`,
//! which the Task 37a assembler locks against the built artifact digest.
//!
//! The runtime gates the Dynosim implementations behind the `dynosim` Cargo
//! feature, but this candidate shell carries no Dynamo dependency: it declares
//! both capabilities and builds unconditionally. Static production Dynosim
//! transport authority remains unchanged until Task 39a moves the
//! implementation out of `rust/runtime/src/transport/`.

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
        "nvidia/transport-dynosim",
        env!("CARGO_PKG_VERSION"),
        "Dynosim transport candidate: dynosim_offline and dynosim_online replay",
    )
    .expect("nvidia/transport-dynosim id must normalize")
});

struct DynosimTransportExtension;

impl AIPerfExtension for DynosimTransportExtension {
    fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        // Candidate shell: both capabilities declared in plugins.yaml.in.
        // Static production registration remains until Task 39a.
        Ok(())
    }
}

static EXT: DynosimTransportExtension = DynosimTransportExtension;

#[aiperf_plugin]
fn transport_dynosim_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
