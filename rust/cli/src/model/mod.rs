// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed benchmark domain object and protocol-v2 wire request.
//!
//! The typed model now lives in `aiperf-config`; this module re-exports it so
//! all `crate::model::…` call sites in the CLI resolve unchanged.

// Leaf model modules, `BenchmarkConfig`, and `BenchmarkRun` now live in
// `aiperf-config`; re-export them so `crate::model::…` call sites (and intra-model
// `super::…` paths) resolve unchanged.
pub use aiperf_config::model::{
    artifacts, config, dataset, endpoint, export, metrics, models, phase, public_catalog,
    rate_series, resolved, run, runtime, telemetry, tokenizer, transport,
};

pub use aiperf_config::model::resolved::Resolved;
pub use aiperf_config::model::{BenchmarkConfig, BenchmarkRun};
