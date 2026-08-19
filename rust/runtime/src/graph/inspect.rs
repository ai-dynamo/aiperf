// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared graph-inspection compatibility vocabulary.

use std::collections::BTreeMap;

/// Validates graph structure and returns detailed inspection findings.
pub use crate::graph::validate::validate_detailed;

/// Severity assigned to an inspection finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphInspectionSeverity {
    /// The graph cannot be executed as authored.
    Error,
    /// The graph can be executed, but has a notable condition.
    Warning,
}

/// Execution-plan phase to which an inspection finding is scoped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphPlanPhase {
    /// The profiling phase.
    Profiling,
    /// The warmup phase.
    Warmup,
}

/// A machine-readable graph-inspection finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphInspectionIssue {
    /// Stable identifier for the kind of finding.
    pub code: String,
    /// Severity of the finding.
    pub severity: GraphInspectionSeverity,
    /// Trace scope, when the finding applies to one trace.
    pub trace_id: Option<String>,
    /// Plan-phase scope, when the finding applies to one phase.
    pub phase: Option<GraphPlanPhase>,
    /// Source location within the graph payload, when available.
    pub location: Option<String>,
    /// Human-readable compatibility message.
    pub message: String,
    /// Deterministic, machine-readable details for the finding.
    pub context: BTreeMap<String, String>,
}
