// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded trace-terminal facts returned by pluggable graph drivers.

use serde::{Deserialize, Serialize};

use crate::graph::replay::{ReplayCallMeasurement, ToolCallMeasurement};

/// Versioned terminal facts that placement can fold without reading agent state.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
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
    /// Whether the full profiling graph executor completed successfully.
    pub completed: bool,
    /// Injected-clock profiling wall duration in milliseconds.
    pub trace_wall_ms: f64,
    /// Ordered completed LLM measurements; warmup calls are never retained here.
    pub calls: Vec<ReplayCallMeasurement>,
    /// Ordered attempted tool measurements; command/output bytes are never retained.
    pub tools: Vec<ToolCallMeasurement>,
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
            completed: true,
            trace_wall_ms: 0.0,
            calls: Vec::new(),
            tools: Vec::new(),
        }
    }

    /// Attach bounded profiling facts after the graph executor reaches success.
    pub fn with_profiling_measurements(
        mut self,
        trace_wall_ms: f64,
        calls: Vec<ReplayCallMeasurement>,
        tools: Vec<ToolCallMeasurement>,
    ) -> Self {
        self.trace_wall_ms = trace_wall_ms;
        self.calls = calls;
        self.tools = tools;
        self
    }
}

/// Controller-owned fold of one graph phase's terminal replay facts.
///
/// Workers append only their bounded terminal supplement through the phase event
/// stream. The controller owns ordering and every final artifact write.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphPhaseSupplement {
    /// Stable schema version for phase/cell compatibility checks.
    pub schema_version: u32,
    /// Terminal facts in the phase completion order.
    pub traces: Vec<TraceTerminalSupplement>,
}

impl GraphPhaseSupplement {
    /// Construct an empty stock version-one phase fold.
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            traces: Vec::new(),
        }
    }

    /// Append one compatible worker terminal fact in controller-selected order.
    pub fn push(
        &mut self,
        supplement: TraceTerminalSupplement,
    ) -> Result<(), GraphSupplementError> {
        if supplement.schema_version != self.schema_version {
            return Err(GraphSupplementError::new(format!(
                "cannot fold trace supplement schema {} into phase schema {}",
                supplement.schema_version, self.schema_version
            )));
        }
        self.traces.push(supplement);
        Ok(())
    }
}

impl Default for GraphPhaseSupplement {
    fn default() -> Self {
        Self::new()
    }
}

/// Failure while validating or folding terminal graph supplements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSupplementError(String);

impl GraphSupplementError {
    /// Build an explicit supplement-fold failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for GraphSupplementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GraphSupplementError {}
