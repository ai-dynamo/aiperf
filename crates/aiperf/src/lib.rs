// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native AIPerf CLI (slim).
//!
//! The shared HTTP client + measurement layer lives in [`aiperf_core`] (wire
//! types, streaming sink, collector observer); the Graph-IR engine lives in
//! `aiperf_graph`. This crate is just the CLI surface: workload shaping
//! ([`workload`]), the online run loop ([`run`]), and reporting ([`report`]).

pub mod http;
pub mod logging;
pub mod report;
pub mod run;
pub mod workload;

#[cfg(test)]
mod test_util;
