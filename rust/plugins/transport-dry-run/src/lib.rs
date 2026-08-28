// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dry-run transport candidate plugin.
//!
//! Stages the native socket-free dry-run transport as a loadable native
//! package. The declared capability is canonical ID `dry_run` with effective
//! aliases `dry-run` and `dryrun`; the authoritative capability list lives in
//! this package's `plugins.yaml.in`, which the Task 37a assembler locks against
//! the built artifact digest.
//!
//! This package builds and exports the entry symbol only. Static production
//! dry-run transport authority remains unchanged until Task 39a moves the
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
        "nvidia/transport-dry-run",
        env!("CARGO_PKG_VERSION"),
        "Dry-run transport candidate: socket-free dry_run transport",
    )
    .expect("nvidia/transport-dry-run id must normalize")
});

struct DryRunTransportExtension;

impl AIPerfExtension for DryRunTransportExtension {
    fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        // Candidate shell: capabilities declared in plugins.yaml.in.
        // Static production registration remains until Task 39a.
        Ok(())
    }
}

static EXT: DryRunTransportExtension = DryRunTransportExtension;

#[aiperf_plugin]
fn transport_dry_run_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}
