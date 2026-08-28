// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! W&B exporter plugin candidate (Task 27).

use std::sync::LazyLock;

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

/// The source API version this plugin crate is authored against.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

static PKG: LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
    LazyLock::new(|| {
        aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
            "aiperf-plugin-export-wandb",
            env!("CARGO_PKG_VERSION"),
            "W&B exporter candidate",
        )
        .expect("aiperf-plugin-export-wandb id must normalize")
    });

struct ExportWandbExtension;

impl aiperf_plugin_api::extension::AIPerfExtension for ExportWandbExtension {
    fn register(
        &self,
        _registrar: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>,
    ) -> Result<(), aiperf_plugin_api::error::ExtensionError> {
        Ok(())
    }
}

static EXT: ExportWandbExtension = ExportWandbExtension;

#[aiperf_plugin]
fn export_wandb_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
