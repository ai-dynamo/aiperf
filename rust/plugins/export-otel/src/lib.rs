// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry exporter candidate plugin (Task 28).
//!
//! Stages `rust/plugins/export-otel` as a loadable `cdylib` exporting
//! `aiperf_plugin_entry_v1`. The package declares the `otel` capability its
//! `plugins.yaml.in` manifest names, together with that manifest's
//! `FoldedProjectionV1(GenAiClientHistogramsV1)` requirement: the candidate
//! consumes the host-folded GenAI histogram projection rather than a
//! per-record callback. The static production `otel` exporter keeps sole
//! authority until Task 39b migrates the implementation here.

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
/// `"otel"` is the canonical capability id for the OTel exporter — the same
/// short-form id the static production exporter registers. The manifest
/// `package_id` (`"nvidia/export-otel"`) is a distinct vendor-namespace
/// identifier; it does not replace the capability id because `/` is outside
/// the version-1 registry-id grammar.
pub const CAPABILITIES: &[&str] = &["otel"];

// The manifest `package_id` is `nvidia/export-otel`; `/` is outside the
// version-1 registry-id grammar, so the descriptor carries the normalizable
// spelling of the same package.
static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored(
        "export-otel",
        env!("CARGO_PKG_VERSION"),
        "OpenTelemetry exporter candidate: decorates host-folded GenAI client histograms",
    )
    .expect("export-otel id must normalize")
});

struct ExportOtelExtension;

impl AIPerfExtension for ExportOtelExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        for capability in CAPABILITIES {
            let id = RegistryId::new(capability, REGISTRY_ID_NORMALIZATION_VERSION)
                .map_err(|e| ExtensionError::registration_failed(format!("{capability}: {e}")))?;
            registrar.record_registration(id)?;
        }
        Ok(())
    }
}

static EXT: ExportOtelExtension = ExportOtelExtension;

#[aiperf_plugin]
fn export_otel_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
