// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native Rust front door for AIPerf.
//!
//! This crate owns the `aiperf profile` command for a **single run**: it parses
//! the profile CLI flags and YAML Config v2 into idiomatic Rust types, projects
//! one run onto the runner's protocol-v2 request schema, and drives the
//! unchanged `aiperf-runner` over stdio. Every other subcommand is delegated to
//! the Python frontend. Multi-run / sweeps / orchestration are out of scope for
//! this increment (rejected with a clear error, never silently degraded).
//!
//! The library target exists so both the `aiperf` binary and the integration
//! tests share the same wire DTOs and projection code (see [`wire`]).

pub mod config;
pub mod delegate;
pub mod dispatch;
pub mod execute;
pub mod expand;
pub mod flags;
pub mod load;
pub mod model;
pub mod profile;
#[cfg(feature = "search-pyo3")]
pub mod pyfit;
pub mod render;
pub mod runner_install;
pub mod search;
pub mod signals;
pub mod sweep;
pub mod yaml;
