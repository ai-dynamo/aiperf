// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed Config-v2 model modules shared as the benchmark wire type.
//!
//! These leaf modules define the typed benchmark domain object and its
//! protocol-v2 wire projection. They were moved verbatim out of `aiperf-cli`
//! so the runtime and CLI can share one typed model; `aiperf-cli` re-exports
//! them so its existing call sites are unchanged.

pub mod artifacts;
pub mod config;
pub mod dataset;
pub mod endpoint;
pub mod export;
pub mod metrics;
pub mod models;
pub mod phase;
pub mod public_catalog;
pub mod rate_series;
pub mod resolved;
pub mod run;
pub mod runtime;
pub mod telemetry;
pub mod tokenizer;
pub mod transport;

pub use config::BenchmarkConfig;
pub use resolved::Resolved;
pub use run::BenchmarkRun;
