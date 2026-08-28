// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RED test: verifies that `#[aiperf_plugin]` generates a function whose type
//! is exactly `PluginEntryV1` from `aiperf_plugin_api`.
//!
//! Before the fix the macro generates:
//!   `pub extern "Rust" fn aiperf_plugin_entry_v1() -> &'static sdk::PluginDeclarationV1`
//! which is NOT the same as:
//!   `unsafe fn() -> api::PluginDeclarationV1`  (= PluginEntryV1)
//!
//! This file fails to compile before the fix because `my_plugin::aiperf_plugin_entry_v1`
//! cannot be coerced to `PluginEntryV1`.

use std::sync::LazyLock;

use aiperf_plugin_api::extension::{PluginDeclarationV1, PluginEntryV1};

mod my_plugin {
    use std::sync::LazyLock;

    use aiperf_plugin_api::{
        descriptor::PluginPackageDescriptor,
        error::ExtensionError,
        extension::{AIPerfExtension, PluginDeclarationV1, PluginRegistrar},
    };
    use aiperf_plugin_sdk_macros::aiperf_plugin;

    static PKG: LazyLock<PluginPackageDescriptor> = LazyLock::new(|| {
        PluginPackageDescriptor::from_authored("abi-test-plugin", "0.1.0", "ABI type test")
            .expect("authored package id must normalize")
    });

    struct NoOpExtension;
    impl AIPerfExtension for NoOpExtension {
        fn register(&self, _r: &mut PluginRegistrar<'_>) -> Result<(), ExtensionError> {
            Ok(())
        }
    }
    static EXT: NoOpExtension = NoOpExtension;

    #[aiperf_plugin]
    pub fn init() -> PluginDeclarationV1 {
        PluginDeclarationV1 {
            package: &*PKG,
            extension: &EXT,
        }
    }
}

/// The generated entry symbol must be coercible to `PluginEntryV1`.
///
/// Before fix: compile error — generated function has type
///   `extern "Rust" fn() -> &'static sdk::PluginDeclarationV1`
/// After fix: compiles — generated function has type
///   `unsafe fn() -> api::PluginDeclarationV1`
#[test]
fn entry_symbol_is_plugin_entry_v1() {
    // Type-level assertion: coerce to the exact host entry type.
    let entry_fn: PluginEntryV1 = my_plugin::aiperf_plugin_entry_v1;
    // Calling it must yield the API PluginDeclarationV1 by value.
    let decl: PluginDeclarationV1 = unsafe { entry_fn() };
    assert_eq!(decl.package.id.as_str(), "abi_test_plugin");
}
