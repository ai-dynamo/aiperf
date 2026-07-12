// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native report persistence for the subprocess runner.

use std::path::Path;

use aiperf_metrics::{NativeReport, ReportPairRunFacts, ReportRunProvenance};
use anyhow::{Context, Result};
use serde::Serialize;

/// Write the unified native-v2 report as pretty JSON to `path`.
pub fn write_native_report_json(report: &NativeReport, path: impl AsRef<Path>) -> Result<()> {
    write_json(report, path)
}

/// Finalize coordinator-owned and pair-owned run provenance, then perform the
/// sole native-v2 JSON write.
///
/// Backend/workload adapters return [`ReportPairRunFacts`]; the process
/// coordinator constructs [`ReportRunProvenance`] from its exact executable and
/// frozen registries. This function joins those typed values before
/// serialization and never parses or mutates raw report JSON.
pub fn finalize_and_write_native_report_json(
    report: NativeReport,
    provenance: ReportRunProvenance,
    facts: ReportPairRunFacts,
    path: impl AsRef<Path>,
) -> Result<()> {
    let report = report
        .finalize_run(provenance, facts)
        .context("finalizing native report run provenance")?;
    write_json(&report, path)
}

fn write_json(value: &impl Serialize, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(value).context("serializing summary report")?;
    std::fs::write(path, json)
        .with_context(|| format!("writing summary report {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ReportRunProvenance {
        ReportRunProvenance::new(
            format!("blake3:{}", "a".repeat(64)),
            "online_http",
            "scheduled",
            Vec::new(),
            vec![aiperf_metrics::ReportEndpointProfileIdentity::new("default", "chat").unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn native_report_json_uses_the_metrics_first_v2_shape() {
        let mut summary = aiperf_metrics::AccumulatorSummary::new();
        summary.insert_finite(aiperf_metrics::MetricTag::RequestCount, 1.0);
        let report = NativeReport::new(&summary, None);
        let path =
            std::env::temp_dir().join(format!("aiperf_native_sum_{}.json", std::process::id()));
        write_native_report_json(&report, &path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["schema_version"], "2.0");
        assert_eq!(value["metrics"]["request_count"]["type"], "counter");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn coordinator_finalizes_typed_run_before_the_only_write() {
        let report = NativeReport::new(&aiperf_metrics::AccumulatorSummary::new(), None);
        let path = std::env::temp_dir().join(format!(
            "aiperf_finalized_native_sum_{}.json",
            std::process::id()
        ));
        finalize_and_write_native_report_json(
            report,
            provenance(),
            ReportPairRunFacts::new(),
            &path,
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["run"]["backend"], "online_http");
        assert_eq!(value["run"]["workload"], "scheduled");
        assert_eq!(value["run"]["endpoint_profiles"][0]["endpoint_id"], "chat");
        let _ = std::fs::remove_file(&path);
    }
}
