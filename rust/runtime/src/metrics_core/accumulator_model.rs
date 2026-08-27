// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary configuration values for metric accumulation.

use crate::metrics_core::catalog::MetricTag;
use crate::metrics_core::definition::Definition;
use crate::metrics_core::steady_state::SteadyStateConfig;
use crate::metrics_core::store::MetricsStorageMode;

/// One configured goodput service-level threshold in native metric units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SloThreshold {
    /// Metric evaluated per request.
    pub tag: MetricTag,
    /// Threshold converted to the metric's native unit.
    pub native_value: f64,
    /// Whether the metric passes when the value is `>=` the threshold (larger is
    /// better) versus `<=` it. Resolved from the static catalog at construction so
    /// the per-record good-request path never re-scans it.
    pub larger_is_better: bool,
    /// The metric's static definition, captured once at construction (config-time)
    /// so the per-record path can route through the shared
    /// [`Definition::passes_threshold`] direction logic without a registry lookup.
    pub definition: &'static Definition,
}

/// Runtime-independent configuration for the metrics engine.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsConfig {
    /// Optional timeslice duration in nanoseconds.
    pub slice_duration_ns: Option<i64>,
    /// Per-request SLOs used by good-request count and goodput.
    pub slos: Vec<SloThreshold>,
    /// Usage client/server discrepancy threshold in percent.
    pub usage_diff_threshold_pct: f64,
    /// Requested-vs-actual OSL percentage threshold.
    pub osl_mismatch_threshold_pct: f64,
    /// Absolute OSL mismatch cap in tokens.
    pub osl_mismatch_max_tokens: f64,
    /// Source visible token accounting from the endpoint's `usage` fields
    /// instead of client-side tokenization. When enabled, `TokenCounts.input`
    /// is `usage.prompt_tokens` and `output`/`reasoning` come from
    /// `usage.completion_tokens`/`usage.reasoning_tokens`; otherwise all three
    /// are the client-tokenized counts. `metrics.rs` applies that per-mode
    /// choice, so the accumulator only ever sees the resolved `TokenCounts`.
    pub use_server_token_count: bool,
    /// Per-record retention mode. [`MetricsStorageMode::Sketch`] streams each value
    /// into a bounded-memory t-digest instead of retaining it, trading exact
    /// percentiles for O(1) memory. Off by default.
    pub storage_mode: MetricsStorageMode,
    /// Closed-loop steady-state windowing for concurrency-target runs. Disabled
    /// by default; when enabled with a positive concurrency target the metrics
    /// plane also emits a steady-state summary over the auto-detected window.
    pub steady_state: SteadyStateConfig,
}
