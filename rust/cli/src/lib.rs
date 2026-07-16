// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native Rust front door for AIPerf.
//!
//! This crate owns the `aiperf profile`/`config` commands natively AND the
//! execution engine: it parses the profile CLI flags and YAML Config v2 into
//! idiomatic Rust types, projects one run onto the protocol-v2 request schema,
//! and drives execution by re-execing the same `aiperf` binary in its internal
//! `--execute` mode ([`execute_mode`]) over stdio. Every other subcommand is
//! delegated to the Python frontend.
//!
//! The library target exists so both the `aiperf` binary and the integration
//! tests share the same wire DTOs, projection code, and execution shell.

pub mod analyze_trace;
#[cfg(feature = "search-pyo3")]
pub mod bayes;
pub mod chat;
pub mod config;
pub mod delegate;
pub mod dispatch;
pub mod exec_bin;
pub mod execute;
pub mod execute_mode;
pub mod expand;
pub mod flags;
#[cfg(feature = "search-pyo3")]
pub mod isotonic;
pub mod load;
pub mod model;
pub mod profile;
#[cfg(feature = "search-pyo3")]
pub mod pyfit;
#[cfg(feature = "search-pyo3")]
pub mod pyopt;
pub mod render;
pub mod search;
pub mod signals;
pub mod speed_bench;
pub mod sweep;
pub mod synthesize;
pub mod validate;
pub mod yaml;
