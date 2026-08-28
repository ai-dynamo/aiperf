// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packaging-test boundary for native AIPerf plugins.
//!
//! The integration suites in `tests/` exercise the distribution lifecycle the
//! host crate owns: authenticated inventory publication
//! ([`aiperf_plugin_host::inventory`]) and atomic generation installation
//! ([`aiperf_plugin_host::install`]).

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

/// The inventory schema version this packaging boundary publishes.
pub const PLUGIN_INVENTORY_SCHEMA_VERSION: u32 = 1;
