// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior tests for the generation-1 plugin source API.
//!
//! These cover the externally observable properties: identifier normalization
//! is exactly version 1, the source API version string round-trips, every
//! boundary item is documented in the ownership spec, and the plugin-authoring
//! surface compiles from outside the crate under the visibility rules a real
//! `cdylib` sees. Only the registrar tests live in `src/extension.rs`, because
//! `PluginRegistrar::new` is `pub(crate)` — the host binds the origin, a plugin
//! never can.

use std::sync::LazyLock;

use aiperf_plugin_api::{
    AIPerfExtension, ExtensionError, GENERATION_1_SURFACE, PLUGIN_ENTRY_SYMBOL_V1,
    PLUGIN_SOURCE_API_VERSION, PluginDeclarationV1, PluginEntryV1, PluginPackageDescriptor,
    PluginRegistrar, PluginSourceApiVersion, REGISTRY_ID_NORMALIZATION_VERSION, RegistryId,
    RegistryIdError, SourceApiVersionError, ownership::render_surface_table,
};

/// Normalize under the only supported version.
fn id(input: &str) -> Result<RegistryId, RegistryIdError> {
    RegistryId::new(input, REGISTRY_ID_NORMALIZATION_VERSION)
}

/// A package descriptor built once and borrowed for `'static`, exactly as the
/// SDK macro will emit it from a plugin crate.
static PACKAGE: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored("AIPerf-Export-OTLP", "0.12.0", "OpenTelemetry exporter")
        .unwrap_or_else(|error| panic!("test package id must normalize: {error}"))
});

/// Stands in for an out-of-crate plugin's extension implementation.
struct TestExtension;

impl AIPerfExtension for TestExtension {
    fn register(&self, registrar: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        registrar.record_registration(id("otel")?)?;
        registrar.record_registration(id("OTEL-Console")?)?;
        Ok(())
    }
}

static EXTENSION: TestExtension = TestExtension;

/// The exact shape the SDK macro exports under `aiperf_plugin_entry_v1`.
unsafe fn entry() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &PACKAGE,
        extension: &EXTENSION,
    }
}

#[test]
fn normalization_version_1_accepts_and_folds_exactly_the_documented_forms() {
    // Trim ASCII space and tab, ASCII-lowercase, fold `-` to `_`.
    assert_eq!(
        id(" \tExport-OTLP\t ").map(RegistryId::into_string),
        Ok("export_otlp".to_owned())
    );
    // Already-normalized input is a fixed point.
    assert_eq!(
        id("export_otlp").map(RegistryId::into_string),
        Ok("export_otlp".to_owned())
    );
    // Digits are legal both leading and interior.
    assert_eq!(
        id("H2C-9").map(RegistryId::into_string),
        Ok("h2c_9".to_owned())
    );
    // Two authored spellings that fold to one identifier compare equal.
    assert_eq!(id("Export-OTLP"), id("export_otlp"));
    // The 128-byte maximum is inclusive.
    let at_limit = "a".repeat(128);
    assert_eq!(id(&at_limit).map(RegistryId::into_string), Ok(at_limit));
}

#[test]
fn normalization_version_1_rejects_each_violated_rule_distinctly() {
    assert_eq!(id("exporté"), Err(RegistryIdError::NonAscii));
    assert_eq!(id("  \t "), Err(RegistryIdError::Empty));
    assert_eq!(id(""), Err(RegistryIdError::Empty));
    // Consecutive authored separators are caught before the `-` to `_` fold, so
    // `a--b`, `a__b`, and `a-_b` cannot collapse onto one identifier.
    assert_eq!(
        id("a--b"),
        Err(RegistryIdError::ConsecutiveSeparators { offset: 2 })
    );
    assert_eq!(
        id("a__b"),
        Err(RegistryIdError::ConsecutiveSeparators { offset: 2 })
    );
    assert_eq!(
        id("a-_b"),
        Err(RegistryIdError::ConsecutiveSeparators { offset: 2 })
    );
    // Leading separator and leading underscore both fail the start rule.
    assert_eq!(
        id("-otlp"),
        Err(RegistryIdError::InvalidStart { character: '_' })
    );
    assert_eq!(
        id("_otlp"),
        Err(RegistryIdError::InvalidStart { character: '_' })
    );
    // Interior bytes outside [a-z0-9_].
    assert_eq!(
        id("ex.port"),
        Err(RegistryIdError::InvalidCharacter {
            character: '.',
            offset: 2
        })
    );
    assert_eq!(
        id("ex port"),
        Err(RegistryIdError::InvalidCharacter {
            character: ' ',
            offset: 2
        })
    );
    // One byte past the inclusive maximum.
    assert_eq!(
        id(&"a".repeat(129)),
        Err(RegistryIdError::TooLong { len: 129 })
    );
    // Only version 1 exists; version 0 and 2 are typed rejections, not folds.
    assert_eq!(
        RegistryId::new("export_otlp", 2),
        Err(RegistryIdError::UnsupportedVersion { requested: 2 })
    );
    assert_eq!(
        RegistryId::new("export_otlp", 0),
        Err(RegistryIdError::UnsupportedVersion { requested: 0 })
    );
}

#[test]
fn the_plugin_authoring_surface_is_reachable_from_outside_the_crate() {
    // This target is compiled as an external consumer, so it sees exactly the
    // visibility a `cdylib` plugin sees. Narrowing any item used here to
    // `pub(crate)` — or dropping it — breaks this test at compile time, which
    // is the only detector the crate has for that drift.
    let entry_point: PluginEntryV1 = entry;
    // SAFETY: `entry` is this test's own Rust-ABI function with the exact
    // `PluginEntryV1` signature, standing in for a validated library symbol.
    let declaration = unsafe { entry_point() };

    assert_eq!(declaration.package.id.as_str(), "aiperf_export_otlp");
    assert_eq!(declaration.package.version, "0.12.0");
    assert_eq!(declaration.package.description, "OpenTelemetry exporter");
    assert_eq!(PLUGIN_ENTRY_SYMBOL_V1, "aiperf_plugin_entry_v1");

    // `AIPerfExtension: Send + Sync` is what lets the host move a declaration
    // onto another thread; a `Box<dyn AIPerfExtension>` must satisfy it too.
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<Box<dyn AIPerfExtension>>();

    // The trait is object-safe from out of crate, and the declaration borrows
    // the plugin's own static rather than a copy of it.
    let _boxed: Box<dyn AIPerfExtension> = Box::new(TestExtension);
    assert!(std::ptr::eq(
        declaration.extension as *const dyn AIPerfExtension as *const u8,
        &EXTENSION as *const TestExtension as *const u8
    ));
}

#[test]
fn every_typed_boundary_error_crosses_threads() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<ExtensionError>();
    assert_send_sync::<RegistryIdError>();
    assert_send_sync::<SourceApiVersionError>();
    assert_send_sync::<RegistryId>();
}

#[test]
fn source_api_version_string_matches_parsed_current_and_rejects_non_canonical() {
    assert_eq!(
        PluginSourceApiVersion::parse(PLUGIN_SOURCE_API_VERSION),
        Ok(PluginSourceApiVersion::CURRENT)
    );
    assert_eq!(
        PluginSourceApiVersion::CURRENT.to_string(),
        PLUGIN_SOURCE_API_VERSION
    );

    assert_eq!(
        PluginSourceApiVersion::parse("1.0"),
        Err(SourceApiVersionError::ComponentCount { found: 2 })
    );
    assert_eq!(
        PluginSourceApiVersion::parse("01.0.0"),
        Err(SourceApiVersionError::Component { index: 0 })
    );

    // Same major, host at least as new in minor.
    let host = PluginSourceApiVersion::new(1, 3, 0);
    assert!(PluginSourceApiVersion::new(1, 2, 9).is_compatible_with(&host));
    assert!(!PluginSourceApiVersion::new(1, 4, 0).is_compatible_with(&host));
    assert!(!PluginSourceApiVersion::new(2, 0, 0).is_compatible_with(&host));
}

#[test]
fn every_boundary_item_is_documented_in_the_ownership_spec() {
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/specs/plugin-api-ownership.md");
    let spec = std::fs::read_to_string(&spec_path).unwrap_or_else(|error| {
        panic!("ownership spec {} must exist: {error}", spec_path.display())
    });

    let missing: Vec<&str> = GENERATION_1_SURFACE
        .iter()
        .map(|row| row.item)
        .filter(|item| !spec.contains(*item))
        .collect();
    assert!(
        missing.is_empty(),
        "boundary items absent from docs/specs/plugin-api-ownership.md: {missing:?}"
    );

    // Generation 1 crosses the boundary only at startup, and every artifact is
    // `panic = abort`. Both facts are load-bearing for the ownership argument.
    assert!(!GENERATION_1_SURFACE.is_empty());
    assert!(GENERATION_1_SURFACE.iter().all(|row| row.panic_abort));
    assert!(
        GENERATION_1_SURFACE
            .iter()
            .all(|row| row.phase == aiperf_plugin_api::CallPhase::Startup)
    );

    // The rendered table is what the guard binary prints; it must name items.
    let rendered = render_surface_table(GENERATION_1_SURFACE);
    assert!(rendered.contains("AIPerfExtension::register"));
    assert!(rendered.lines().count() == GENERATION_1_SURFACE.len() + 1);
}
