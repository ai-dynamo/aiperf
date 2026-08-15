// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded trace-terminal facts returned by pluggable graph drivers.

use serde::{Deserialize, Serialize};

/// Versioned terminal facts that placement can fold without reading agent state.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TraceTerminalSupplement {
    /// Stable schema version for future cellular folding.
    pub schema_version: u32,
    /// Run-wide correlation identity.
    pub run_id: String,
    /// Trace-local trajectory document identity.
    pub trajectory_id: String,
    /// Graph trace identity.
    pub trace_id: String,
    /// Owning worker ordinal.
    pub worker_id: usize,
    /// Registered driver that emitted this bounded result.
    pub driver_kind: String,
}

impl TraceTerminalSupplement {
    /// Construct the stock version-one terminal supplement.
    pub fn new(
        run_id: String,
        trajectory_id: String,
        trace_id: String,
        worker_id: usize,
        driver_kind: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: 1,
            run_id,
            trajectory_id,
            trace_id,
            worker_id,
            driver_kind: driver_kind.into(),
        }
    }
}
