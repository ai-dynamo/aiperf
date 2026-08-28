// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dry-run transport plugin candidate shell (Task 34).

use std::sync::LazyLock;

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

static PKG: LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
    LazyLock::new(|| {
        aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
            "nvidia/transport-dry-run",
            env!("CARGO_PKG_VERSION"),
            "Dry-run transport plugin candidate",
        )
        .expect("transport-dry-run id must normalize")
    });

struct DryRunTransportExtension;

impl aiperf_plugin_api::extension::AIPerfExtension for DryRunTransportExtension {
    fn register(
        &self,
        _registrar: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>,
    ) -> Result<(), aiperf_plugin_api::error::ExtensionError> {
        Ok(())
    }
}

static EXT: DryRunTransportExtension = DryRunTransportExtension;

#[aiperf_plugin]
fn dry_run_transport_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
