// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! AIPerf CLI and execution entry point.
//!
//! The crate parses CLI flags and Config v2, projects protocol-v2 requests, and
//! re-execs `aiperf --execute` over stdio. Python-backed commands are delegated
//! through [`delegate`].

pub mod analyze_trace;
#[cfg(feature = "search-pyo3")]
pub mod bayes;
pub mod cellular_role;
pub mod chat;
pub mod compare;
pub mod config;
pub mod control_hooks;
pub mod delegate;
pub mod diagnostics;
pub mod dispatch;
pub mod exec_bin;
pub mod execute;
pub mod execute_mode;
pub mod expand;
pub mod flags;
#[cfg(feature = "search-pyo3")]
pub mod isotonic;
pub mod jsonnum;
pub mod k8s;
pub mod load;
pub mod logging;
pub mod metrics_list;
pub mod model;
pub mod profile;
#[cfg(feature = "search-pyo3")]
pub mod pyfit;
#[cfg(feature = "search-pyo3")]
pub mod pyopt;
pub mod render;
pub mod results_sidecar;
pub mod search;
pub mod search_history;
pub mod signals;
pub mod slurm;
pub mod speed_bench;
pub mod stats;
pub mod sweep;
pub mod synthesize;
pub mod validate;
pub mod yaml;
