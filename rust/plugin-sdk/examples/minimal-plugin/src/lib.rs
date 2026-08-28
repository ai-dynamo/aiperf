// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal plugin fixture demonstrating `#[aiperf_plugin]` usage.
//!
//! Not part of the workspace — built standalone as a fixture for tests.

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

static CAPABILITIES: &[&str] = &[];

#[aiperf_plugin]
fn plugin_init() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        name: "minimal",
        version: env!("CARGO_PKG_VERSION"),
        aiperf_sdk_version: aiperf_plugin_sdk::PLUGIN_SOURCE_API_VERSION,
        capabilities: CAPABILITIES,
    }
}
