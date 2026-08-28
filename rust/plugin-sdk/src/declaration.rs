// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin declaration type returned by the native entry symbol.

/// The static declaration a plugin returns from `aiperf_plugin_entry_v1`.
///
/// Must be `#[repr(C)]` for stable memory layout; all fields are `'static`
/// string slices backed by the plugin's own binary so the caller never frees
/// them.
#[repr(C)]
pub struct PluginDeclarationV1 {
    /// Plugin name as registered with the host.
    pub name: &'static str,
    /// SemVer version string of this plugin build.
    pub version: &'static str,
    /// Version of the `aiperf-plugin-sdk` crate this plugin was compiled against.
    pub aiperf_sdk_version: &'static str,
    /// Declared capability identifiers (e.g. `"endpoint"`, `"transport"`).
    pub capabilities: &'static [&'static str],
}
