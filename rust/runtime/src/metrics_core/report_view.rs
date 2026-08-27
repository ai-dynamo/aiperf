// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow read-only projection of a finalized report for exporters.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::export::otel::OtelRecordAccumulator;
use crate::metrics_core::AccuracyAnalysis;
use crate::metrics_core::catalog::MetricTag;
use crate::metrics_core::report::{MetricEntry, ReportError, ReportSteadyState, ReportSummary};

/// Read-only accessors an exporter needs from a finalized report.
pub trait ReportView {
    /// AIPerf package version that produced the report.
    fn aiperf_version(&self) -> &str;

    /// Run-level timing, cancellation, endpoint, and server-metrics facts.
    fn run_summary(&self) -> &ReportSummary;

    /// One profiling metric entry, absent when the run produced none.
    fn metric(&self, tag: MetricTag) -> Option<&MetricEntry>;

    /// Names of every profiling metric present, in stable report order.
    fn metric_names(&self) -> Vec<Arc<str>>;

    /// Profiling metrics keyed by stable report name.
    fn metrics(&self) -> &BTreeMap<String, MetricEntry>;

    /// Warmup metrics when a warmup phase was retained.
    fn warmup_metrics(&self) -> Option<&BTreeMap<String, MetricEntry>>;

    /// Profiling server telemetry keyed by Prometheus family name.
    fn server_metrics(&self) -> &BTreeMap<String, MetricEntry>;

    /// Warmup server telemetry keyed by Prometheus family name.
    fn warmup_server_metrics(&self) -> &BTreeMap<String, MetricEntry>;

    /// Optional accuracy analysis.
    fn accuracy(&self) -> Option<&AccuracyAnalysis>;

    /// Grouped run errors.
    fn errors(&self) -> &[ReportError];

    /// Optional closed-loop steady-state summary.
    fn steady_state(&self) -> Option<&ReportSteadyState>;

    /// Exact pooled speculative-decode acceptance counts.
    fn pooled_spec_decode_acceptance_histogram(&self) -> Option<&BTreeMap<u64, u128>>;

    /// Per-record OTLP capture when exact records were retained.
    fn per_record(&self) -> Option<&OtelRecordAccumulator>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporters_read_reports_through_the_narrow_view() {
        // An exporter plugin must be writable against ReportView alone, with no
        // access to NativeReport's internal structure.
        fn summarize(view: &dyn ReportView) -> usize {
            view.metric_names().len()
        }

        let report = crate::metrics_core::report::test_util::two_metric_report();
        assert_eq!(summarize(&report), 2);
    }

    #[test]
    fn exporter_boundary_accepts_the_narrow_view() {
        fn export(
            exporter: &dyn crate::export::Exporter,
            view: &dyn ReportView,
            artifact_dir: &std::path::Path,
            cfg: &crate::export::ExportConfig,
        ) -> anyhow::Result<()> {
            exporter.export(view, artifact_dir, cfg)
        }

        let _ = export;
    }
}
