// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for plugin registration and universe freeze (Task 15).

use std::sync::LazyLock;

use aiperf_plugin_api::{
    AIPerfExtension, ExtensionError, PluginPackageDescriptor, PluginRegistrar, RegistryId,
    REGISTRY_ID_NORMALIZATION_VERSION,
};
use aiperf_plugin_host::{
    freeze::freeze_universe,
    register::{PluginRegistrationContext, RegistrationError, register_plugin},
};

fn id(input: &str) -> RegistryId {
    RegistryId::new(input, REGISTRY_ID_NORMALIZATION_VERSION)
        .unwrap_or_else(|e| panic!("test id must normalize: {e}"))
}

static PACKAGE_EXPORTER: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored("AIPerf-Export-OTLP", "0.3.0", "OTLP exporter")
        .expect("test package id must normalize")
});

static PACKAGE_TRANSPORT: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored("AIPerf-Transport-H2C", "1.0.0", "h2c transport")
        .expect("test package id must normalize")
});

struct SingleCapabilityExtension {
    capability_id: &'static str,
}

impl AIPerfExtension for SingleCapabilityExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        registrar.record_registration(id(self.capability_id))?;
        Ok(())
    }
}

struct DuplicateRegistrationExtension;

impl AIPerfExtension for DuplicateRegistrationExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        registrar.record_registration(id("otlp"))?;
        registrar.record_registration(id("OTLP"))?; // same after normalization
        Ok(())
    }
}

struct MultiCapabilityExtension;

impl AIPerfExtension for MultiCapabilityExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        registrar.record_registration(id("h2c"))?;
        registrar.record_registration(id("h2c-tls"))?;
        Ok(())
    }
}

#[test]
fn register_plugin_succeeds_and_frozen_universe_has_correct_registrations() {
    let ctx = PluginRegistrationContext::new(&PACKAGE_EXPORTER);
    let extension = SingleCapabilityExtension { capability_id: "otlp" };
    let frozen = register_plugin(&ctx, &extension).expect("registration must succeed");

    assert_eq!(frozen.package().id.as_str(), "aiperf_export_otlp");
    assert_eq!(frozen.len(), 1);
    assert_eq!(frozen.registrations()[0].id().as_str(), "otlp");
    // The descriptor carries the manifest-bound package.
    assert!(std::ptr::eq(frozen.registrations()[0].package(), &*PACKAGE_EXPORTER));
}

#[test]
fn register_plugin_rejects_duplicate_registration_within_one_package() {
    let ctx = PluginRegistrationContext::new(&PACKAGE_EXPORTER);
    let extension = DuplicateRegistrationExtension;
    let err = register_plugin(&ctx, &extension).expect_err("duplicate must be rejected");
    assert!(
        matches!(err, RegistrationError::DuplicateCapability(_)),
        "wrong variant: {err:?}"
    );
}

#[test]
fn freeze_universe_merges_multiple_packages_in_load_order() {
    let ctx_export = PluginRegistrationContext::new(&PACKAGE_EXPORTER);
    let ctx_transport = PluginRegistrationContext::new(&PACKAGE_TRANSPORT);

    let u1 = register_plugin(&ctx_export, &SingleCapabilityExtension { capability_id: "otlp" })
        .expect("exporter registration succeeds");
    let u2 = register_plugin(&ctx_transport, &MultiCapabilityExtension)
        .expect("transport registration succeeds");

    let registry = freeze_universe(vec![u1, u2]);
    assert_eq!(registry.universes().len(), 2);
    assert_eq!(registry.registration_count(), 3);

    let all: Vec<&str> = registry
        .all_registrations()
        .map(|d| d.id().as_str())
        .collect();
    assert_eq!(all, ["otlp", "h2c", "h2c_tls"]);
}

#[test]
fn frozen_plugin_universe_is_empty_for_no_op_extension() {
    struct NoOpExtension;
    impl AIPerfExtension for NoOpExtension {
        fn register(&self, _registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
            Ok(())
        }
    }

    let ctx = PluginRegistrationContext::new(&PACKAGE_EXPORTER);
    let frozen = register_plugin(&ctx, &NoOpExtension).expect("no-op registration succeeds");
    assert!(frozen.is_empty());
    assert_eq!(frozen.len(), 0);
}
