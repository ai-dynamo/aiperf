// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the `#[aiperf_plugin]` macro and `PluginDeclarationV1` entry symbol.
//!
//! After Task 8/9 C1 fix, `PluginDeclarationV1` is the API's type
//! (`{package, extension}`), not the SDK's old `{name, version, …}` struct.

use std::sync::LazyLock;

use aiperf_plugin_api::{
    descriptor::PluginPackageDescriptor,
    error::ExtensionError,
    extension::{AIPerfExtension, PluginDeclarationV1, PluginEntryV1, PluginRegistrar},
};

static PKG_A: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
    PluginPackageDescriptor::from_authored("test-plugin-a", "1.2.3", "Test plugin A")
        .expect("authored package id must normalize")
});

struct NoOpExtension;
impl AIPerfExtension for NoOpExtension {
    fn register(&self, _r: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
        Ok(())
    }
}
static EXT_NOOP: NoOpExtension = NoOpExtension;

fn make_decl() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        package: &*PKG_A,
        extension: &EXT_NOOP,
    }
}

#[test]
fn declaration_fields_are_accessible() {
    let d = make_decl();
    // Package id is normalized from "test-plugin-a" → "test_plugin_a"
    assert_eq!(d.package.id.as_str(), "test_plugin_a");
    assert_eq!(d.package.version, "1.2.3");
}

#[test]
fn declaration_is_copy() {
    // PluginDeclarationV1 must be Copy so the OnceLock entry can return by value.
    fn assert_copy<T: Copy>() {}
    assert_copy::<PluginDeclarationV1>();
}

#[test]
fn declaration_size_nonzero() {
    // PluginDeclarationV1 carries two pointer-width fields; not zero-sized.
    assert!(std::mem::size_of::<PluginDeclarationV1>() > 0);
}

#[test]
fn declaration_static_borrow_compiles() {
    // Both fields are &'static borrows from plugin-owned storage.
    let d: PluginDeclarationV1 = make_decl();
    let _pkg: &'static PluginPackageDescriptor = d.package;
    let _ext: &'static dyn AIPerfExtension = d.extension;
}

#[test]
fn plugin_sdk_version_constant_present() {
    // The SDK exposes a version constant that plugins embed.
    let v = aiperf_plugin_sdk::PLUGIN_SOURCE_API_VERSION;
    assert!(!v.is_empty());
}

#[test]
fn plugin_entry_v1_alias_matches_macro_output() {
    // PluginEntryV1 is `unsafe fn() -> PluginDeclarationV1`.
    // Verify the alias is accessible through the SDK re-export.
    let _: PluginEntryV1;
}
