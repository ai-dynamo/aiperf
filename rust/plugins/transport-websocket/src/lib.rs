// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket transport plugin candidate shell (Task 33).

use std::sync::LazyLock;

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

static PKG: LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
    LazyLock::new(|| {
        aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
            "aiperf-plugin-transport-websocket",
            env!("CARGO_PKG_VERSION"),
            "WebSocket transport plugin candidate",
        )
        .expect("transport-websocket id must normalize")
    });

struct WebSocketTransportExtension;

impl aiperf_plugin_api::extension::AIPerfExtension for WebSocketTransportExtension {
    fn register(
        &self,
        _registrar: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>,
    ) -> Result<(), aiperf_plugin_api::error::ExtensionError> {
        Ok(())
    }
}

static EXT: WebSocketTransportExtension = WebSocketTransportExtension;

#[aiperf_plugin]
fn websocket_transport_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
