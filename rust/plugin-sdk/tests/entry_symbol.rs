// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the `#[aiperf_plugin]` macro and `PluginDeclarationV1` entry symbol.

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;

fn make_decl() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        name: "test-plugin",
        version: "1.2.3",
        aiperf_sdk_version: "0.13.0",
        capabilities: &["endpoint", "transport"],
    }
}

#[test]
fn declaration_fields_are_accessible() {
    let d = make_decl();
    assert_eq!(d.name, "test-plugin");
    assert_eq!(d.version, "1.2.3");
    assert_eq!(d.aiperf_sdk_version, "0.13.0");
    assert_eq!(d.capabilities, &["endpoint", "transport"]);
}

#[test]
fn declaration_empty_capabilities() {
    let d = PluginDeclarationV1 {
        name: "minimal",
        version: "0.1.0",
        aiperf_sdk_version: "0.13.0",
        capabilities: &[],
    };
    assert!(d.capabilities.is_empty());
}

#[test]
fn declaration_is_repr_c() {
    // PluginDeclarationV1 must be #[repr(C)] for stable layout.
    // We can't directly test the attribute, but we can verify the struct's
    // fields are accessible and that it has a stable size (not zero-sized).
    let d = make_decl();
    assert!(std::mem::size_of_val(&d) > 0);
}

#[test]
fn declaration_static_lifetime() {
    // Entry function returns &'static PluginDeclarationV1; verify the
    // lifetime constraint holds with a static declaration.
    static DECL: PluginDeclarationV1 = PluginDeclarationV1 {
        name: "static-plugin",
        version: "0.0.1",
        aiperf_sdk_version: "0.13.0",
        capabilities: &[],
    };
    let r: &'static PluginDeclarationV1 = &DECL;
    assert_eq!(r.name, "static-plugin");
}

#[test]
fn plugin_sdk_version_constant_present() {
    // The SDK exposes a version constant that plugins embed.
    let v = aiperf_plugin_sdk::PLUGIN_SOURCE_API_VERSION;
    assert!(!v.is_empty());
}
