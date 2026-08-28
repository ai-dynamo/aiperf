// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host boundary for native AIPerf plugins.

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

pub mod error;
pub mod manifest;
pub mod normalize;
