// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic-exporter plugin candidate (Tasks 24-25: JSON, CSV, console, timeslice).

use std::sync::LazyLock;

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

static PKG: LazyLock<aiperf_plugin_api::descriptor::PluginPackageDescriptor> =
    LazyLock::new(|| {
        aiperf_plugin_api::descriptor::PluginPackageDescriptor::from_authored(
            "aiperf-plugin-export-basic",
            env!("CARGO_PKG_VERSION"),
            "Basic exporter candidate (JSON, CSV, console, timeslice)",
        )
        .expect("aiperf-plugin-export-basic id must normalize")
    });

struct ExportBasicExtension;

impl aiperf_plugin_api::extension::AIPerfExtension for ExportBasicExtension {
    fn register(
        &self,
        _registrar: &mut aiperf_plugin_api::extension::PluginRegistrar<'_>,
    ) -> Result<(), aiperf_plugin_api::error::ExtensionError> {
        Ok(())
    }
}

static EXT: ExportBasicExtension = ExportBasicExtension;

#[aiperf_plugin]
fn export_basic_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
