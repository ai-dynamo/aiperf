// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Clock abstraction for real and deterministic virtual execution.
//!
//! The trait itself is boundary-owned and lives in
//! [`aiperf_core::clock`]; this module is the compatibility path for runtime
//! code and downstream crates that already import `crate::clock::Clock`.

pub use aiperf_core::clock::Clock;
