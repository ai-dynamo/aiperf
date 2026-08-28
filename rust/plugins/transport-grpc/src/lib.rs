// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport candidate plugin.
//!
//! Declares the KServe OIP and NVIDIA Riva ASR/TTS/NLP gRPC transport
//! capability under the canonical ID `grpc`. This is the staging milestone:
//! the candidate builds as a `cdylib`, exports `aiperf_plugin_entry_v1`, and
//! declares the capability set its manifest advertises. The transport
//! implementation still lives in `aiperf-runtime`, and static production gRPC
//! transport authority is unchanged until Task 39a makes the transport ID
//! exclusively dynamic.
//!
//! Only the canonical identifier is registered. The generation-1
//! [`PluginRegistrar`] records one [`RegistryId`] per capability and has no
//! alias seam, and the manifest's dotted aliases (`grpc.tonic`, `grpc.kserve`,
//! `grpc.riva`) are not expressible under the version-1 grammar
//! `^[a-z0-9][a-z0-9_]{0,127}$` anyway. Aliases therefore remain
//! manifest-declared facts until the registrar grows an alias-binding method.

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

/// The authored package identifier.
///
/// Spelled with `-` rather than `/`: a slash is rejected by both the
/// [`RegistryId`] version-1 grammar and the host manifest's own identifier
/// normalizer, so a `nvidia/transport-grpc` spelling could never resolve on
/// either side of the boundary.
const PACKAGE_ID: &str = "nvidia-transport-grpc";

/// Canonical transport capability identifiers declared by this package.
///
/// Order is the reported registration order and mirrors `plugins.yaml.in`.
const CAPABILITIES: &[&str] = &["grpc"];

static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    // Every byte of PACKAGE_ID satisfies the version-1 grammar, so this
    // normalization cannot fail; a typo in the const would abort the process
    // at the entry call, which is why the id is a checked const and not
    // caller-supplied.
    PluginPackageDescriptor::from_authored(
        PACKAGE_ID,
        env!("CARGO_PKG_VERSION"),
        "gRPC transport candidate: KServe OIP, Riva ASR/TTS/NLP",
    )
    .expect("PACKAGE_ID is a compile-time constant that satisfies the version-1 id grammar")
});

/// Registers this package's declared transport capabilities.
struct GrpcTransportExtension;

impl AIPerfExtension for GrpcTransportExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        for authored in CAPABILITIES {
            let id = RegistryId::new(authored, REGISTRY_ID_NORMALIZATION_VERSION).map_err(|e| {
                ExtensionError::registration_failed(format!("capability `{authored}`: {e}"))
            })?;
            registrar.record_registration(id)?;
        }
        Ok(())
    }
}

static EXT: GrpcTransportExtension = GrpcTransportExtension;

/// The version-1 entry point the host calls once to obtain this package's
/// identity and registration seam.
#[aiperf_plugin]
fn transport_grpc_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG,
        extension: &EXT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The package descriptor must normalize; otherwise the entry call aborts.
    #[test]
    fn package_id_normalizes() {
        assert_eq!(PKG.id().as_str(), "nvidia_transport_grpc");
    }

    /// Registration declares every canonical capability exactly once.
    #[test]
    fn register_declares_all_canonical_capabilities() {
        let mut registrar = PluginRegistrar::new(&PKG);
        GrpcTransportExtension
            .register(&mut registrar)
            .expect("registration must succeed");

        let observed: Vec<&str> = registrar.observed().iter().map(|id| id.as_str()).collect();
        assert_eq!(observed, ["grpc"]);
    }
}
