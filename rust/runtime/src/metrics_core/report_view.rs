// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow read-only projection of a finalized report for exporters.

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
}
