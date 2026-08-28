// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_plugin_sdk::declaration::PluginDeclarationV1;
use aiperf_plugin_sdk_macros::aiperf_plugin;

#[aiperf_plugin]
fn minimal_plugin() -> PluginDeclarationV1 {
    PluginDeclarationV1 {
        name: "minimal",
        version: "0.1.0",
        aiperf_sdk_version: env!("CARGO_PKG_VERSION"),
        capabilities: &[],
    }
}
