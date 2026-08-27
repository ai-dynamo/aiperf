// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow read-only projection of a finalized report for exporters.

use std::sync::Arc;

/// Read-only accessors an exporter needs from a finalized report.
pub trait ReportView {
    /// Names of every profiling metric present, in stable report order.
    fn metric_names(&self) -> Vec<Arc<str>>;
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
