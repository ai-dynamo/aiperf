// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `runtime` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::dump_benchmark_run`
//! (the runtime dict is filtered to `{workers, workers_min, workers_max,
//! cells}`). `workers` / `workers_min` are always present (emitting `null` when
//! unset — Python does not exclude them); `workers_max` is present only when set.

use serde::{Deserialize, Serialize};

/// The typed worker/cell runtime policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Runtime {
    /// Fixed worker count (`null` = runner auto-selects).
    pub workers: Option<u32>,
    /// Minimum worker count for adaptive worker scaling (`null` = unset).
    pub workers_min: Option<u32>,
    /// Maximum worker count for adaptive worker scaling (present only when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers_max: Option<u32>,
    /// Cellular (multi-process) cell count; `1` is the single-process path.
    pub cells: u32,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            workers: None,
            workers_min: None,
            workers_max: None,
            cells: 1,
        }
    }
}
