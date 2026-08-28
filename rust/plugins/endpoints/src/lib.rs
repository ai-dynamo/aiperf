// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint factory candidate plugin.
//!
//! Declares the Chat, Responses, Completions, Embeddings, and SageMaker
//! endpoint capabilities (Task 29) together with the KServe and Riva gRPC
//! companion bindings (Task 30). This is the staging milestone: the candidate
//! builds as a `cdylib`, exports `aiperf_plugin_entry_v1`, and declares the
//! capability set its manifest advertises. The endpoint dialect
//! implementations still live in `aiperf-runtime`, and the static production
//! endpoint and `GrpcBindingRegistry::builtin()` authority is unchanged until
//! Task 39a makes an identifier exclusively dynamic.
//!
//! Only canonical identifiers are registered. The generation-1
//! [`PluginRegistrar`] records one [`RegistryId`] per capability and has no
//! alias seam, and the authored aliases are not all expressible as registry
//! identifiers anyway: the version-1 grammar is
//! `^[a-z0-9][a-z0-9_]{0,127}$`, which rejects the dotted spellings
//! (`openai.chat`, `riva.asr`) that `plugins.yaml.in` declares. Aliases
//! therefore remain manifest-declared facts until the registrar grows an
//! alias-binding method.

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
/// normalizer, so a `nvidia/endpoints` spelling could never resolve on either
/// side of the boundary.
const PACKAGE_ID: &str = "nvidia-endpoints";

/// Canonical endpoint capability identifiers declared by this package.
///
/// Order is the reported registration order and mirrors `plugins.yaml.in`.
const ENDPOINT_CAPABILITIES: &[&str] = &[
    "chat",
    "responses",
    "completions",
    "embeddings",
    "sagemaker_chat",
];

/// Canonical gRPC companion binding identifiers declared by this package.
///
/// These are companion bindings to the endpoint capabilities above, not
/// standalone endpoints; they select the KServe OIP and Riva codec families.
const GRPC_BINDING_CAPABILITIES: &[&str] = &["kserve", "riva_asr", "riva_tts", "riva_nlp"];

static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    // Every byte of PACKAGE_ID satisfies the version-1 grammar, so this
    // normalization cannot fail; a typo in the const would abort the process
    // at the entry call, which is why the id is a checked const and not
    // caller-supplied.
    PluginPackageDescriptor::from_authored(
        PACKAGE_ID,
        env!("CARGO_PKG_VERSION"),
        "Endpoint factory candidate: Chat, Responses, Completions, Embeddings, SageMaker",
    )
    .expect("PACKAGE_ID is a compile-time constant that satisfies the version-1 id grammar")
});

/// Registers this package's declared endpoint and gRPC binding capabilities.
struct EndpointExtension;

impl AIPerfExtension for EndpointExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        for authored in ENDPOINT_CAPABILITIES
            .iter()
            .chain(GRPC_BINDING_CAPABILITIES.iter())
        {
            let id = RegistryId::new(authored, REGISTRY_ID_NORMALIZATION_VERSION)
                .map_err(|e| {
                    ExtensionError::registration_failed(format!("capability `{authored}`: {e}"))
                })?;
            registrar.record_registration(id)?;
        }
        Ok(())
    }
}

static EXT: EndpointExtension = EndpointExtension;

/// The version-1 entry point the host calls once to obtain this package's
/// identity and registration seam.
#[aiperf_plugin]
fn endpoints_plugin() -> PluginDeclarationV1 {
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
        assert_eq!(PKG.id().as_str(), "nvidia_endpoints");
    }

    /// Registration declares every canonical capability exactly once, in
    /// manifest order, with the gRPC companions following the endpoints.
    #[test]
    fn register_declares_all_canonical_capabilities() {
        let mut registrar = PluginRegistrar::new(&PKG);
        EndpointExtension
            .register(&mut registrar)
            .expect("registration must succeed");

        let observed: Vec<&str> = registrar.observed().iter().map(|id| id.as_str()).collect();
        assert_eq!(
            observed,
            [
                "chat",
                "responses",
                "completions",
                "embeddings",
                "sagemaker_chat",
                "kserve",
                "riva_asr",
                "riva_tts",
                "riva_nlp",
            ]
        );
    }
}
