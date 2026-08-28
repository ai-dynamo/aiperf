// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static-comparator boundary for native AIPerf plugins.
//!
//! The comparator is the statically-linked baseline the dynamic plugin
//! distribution is measured against. [`static_inventory`] owns the census both
//! sides must agree on; the `aiperf-plugin-static-comparator` binary asserts
//! that agreement and refuses to stand in as a baseline without it.

pub mod static_inventory;

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
