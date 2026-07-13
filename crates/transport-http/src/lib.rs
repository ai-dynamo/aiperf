// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf HTTP transport, endpoint binding, and timing recording.
//!
//! A behavioral port of AIPerf's Python `aiohttp` transport layer, driven
//! entirely off the [`aiperf_clock::Clock`] abstraction so it runs identically
//! live ([`aiperf_clock::RealClock`]) or under virtual time
//! ([`aiperf_clock::SimClock`]).
//!
//! Transport-agnostic endpoint dialects remain in `aiperf-endpoints`; the
//! [`transport::endpoint_binding`] module owns their HTTP wire translation.

pub mod client;
pub mod config;
pub mod models;
pub mod sse;
pub mod transport;

pub use aiperf_clock::{Clock, RealClock, SimClock};
