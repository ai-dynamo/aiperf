// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AIPerf HTTP transport, endpoint binding, and timing recording.
//!
//! All timing uses [`crate::clock::Clock`] for both live and virtual-time runs.
//! [`transport::endpoint_binding`] owns HTTP wire translation for
//! transport-agnostic endpoint dialects.

pub mod client;
pub mod config;
pub mod models;
pub mod sink;
pub mod sse;
pub mod transport;

pub use crate::clock::{Clock, RealClock, SimClock};
pub use sink::*;
