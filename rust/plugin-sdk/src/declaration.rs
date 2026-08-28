// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin declaration type returned by the native entry symbol.
//!
//! Re-exports [`PluginDeclarationV1`] and [`PluginEntryV1`] from
//! `aiperf_plugin_api` so SDK users do not need to depend on the API crate
//! directly. The host-authoritative definition lives in `aiperf_plugin_api`;
//! the SDK provides a single canonical import path for plugin authors.

pub use aiperf_plugin_api::extension::{PluginDeclarationV1, PluginEntryV1};
