// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf CLI (slim).
//!
//! Measurement wire types + the collector observer live in [`aiperf_core`]; the
//! Graph-IR engine lives in `aiperf_graph`. This crate owns the CLI surface: the
//! online HTTP sink over `aiperf-transport` ([`http`]), workload shaping
//! ([`workload`]), the online run loop ([`run`]), reporting ([`report`]), and
//! logging setup ([`logging`]).

pub mod http;
pub mod logging;
pub mod report;
pub mod run;
pub mod timing;
pub mod workload;

#[cfg(test)]
mod test_util;
