// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parquet exporter candidate plugin (Task 26).
//!
//! Stages `rust/plugins/export-parquet` as a loadable `cdylib` exporting
//! `aiperf_plugin_entry_v1`. The package declares the `export.parquet`
//! capability and its manifest aliases; the Parquet writing implementation
//! still lives in the static runtime exporter and moves here in Task 39b.
//!
//! The candidate builds without the workspace `parquet` feature on purpose:
//! staging must not pull the Arrow/Parquet stack into the plugin artifact
//! before the implementation migrates.

use std::sync::LazyLock;

use aiperf_plugin_api::{
    descriptor::PluginPackageDescriptor,
    error::ExtensionError,
    extension::{AIPerfExtension, PluginRegistrar},
    id::{REGISTRY_ID_NORMALIZATION_VERSION, RegistryId},
};
use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

/// The source API version exposed by this plugin candidate.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

/// Capability identifiers this package registers, matching `plugins.yaml.in`.
///
/// The manifest spells the canonical identifier `export.parquet`; `.` is
/// outside the version-1 registry-id grammar, so the normalized registry
/// spelling is `export_parquet`. The remaining entries are the manifest
/// aliases.
pub const CAPABILITIES: &[&str] = &["export_parquet", "parquet", "parquet_records"];

// The manifest `package_id` is `nvidia/export-parquet`; `/` is outside the
// version-1 registry-id grammar, so the descriptor carries the normalizable
// spelling of the same package.
static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored(
        "export-parquet",
        env!("CARGO_PKG_VERSION"),
        "Parquet exporter candidate: Parquet per-record output as a plugin package",
    )
    .expect("export-parquet id must normalize")
});

struct ExportParquetExtension;

impl AIPerfExtension for ExportParquetExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        for capability in CAPABILITIES {
            let id = RegistryId::new(capability, REGISTRY_ID_NORMALIZATION_VERSION)
                .map_err(|e| ExtensionError::registration_failed(format!("{capability}: {e}")))?;
            registrar.record_registration(id)?;
        }
        Ok(())
    }
}

static EXT: ExportParquetExtension = ExportParquetExtension;

#[aiperf_plugin]
fn export_parquet_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
