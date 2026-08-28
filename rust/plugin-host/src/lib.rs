// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host boundary for native AIPerf plugins.

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

pub mod acquire;
pub mod authority;
pub mod bundle;
pub mod catalog;
pub mod closure;
pub mod diff;
pub mod discovery;
pub mod error;
pub mod freeze;
pub mod inspect;
pub mod loader;
pub mod lock;
pub mod manifest;
pub mod normalize;
pub mod platform;
pub mod priority;
pub mod register;
pub mod residency;
pub mod stage;
