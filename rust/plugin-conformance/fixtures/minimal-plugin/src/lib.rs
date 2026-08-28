// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal plugin fixture for Task 36 native-boundary conformance tests.

use std::sync::LazyLock;

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

static PKG: LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
    LazyLock::new(|| {
        aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
            "minimal-plugin",
            env!("CARGO_PKG_VERSION"),
            "Minimal native-boundary conformance fixture",
        )
        .expect("minimal-plugin id must normalize")
    });

struct MinimalExtension;

impl aiperf_plugin_api::extension::AIPerfExtension for MinimalExtension {
    fn register(
        &self,
        _registrar: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>,
    ) -> Result<(), aiperf_plugin_api::error::ExtensionError> {
        Ok(())
    }
}

static EXT: MinimalExtension = MinimalExtension;

#[aiperf_plugin]
fn minimal_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
