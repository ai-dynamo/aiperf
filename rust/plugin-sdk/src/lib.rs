// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin authoring boundary for native AIPerf plugins.

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

pub mod abi_closure;
pub mod artifact_section;
pub mod build;
pub mod canonical;
pub mod conformance;
pub mod declaration;
pub mod identity;
pub mod inspect;
pub mod manifest;
pub mod sandbox;
