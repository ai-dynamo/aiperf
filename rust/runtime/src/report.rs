// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native report persistence for the subprocess runner.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::metrics_core::{NativeReport, ReportPairRunFacts, ReportRunMetadata};
use anyhow::{Context, Result};
use serde::Serialize;

/// Atomically commit the unified native-v2 report as pretty JSON to `path`.
pub fn write_native_report_json(report: &NativeReport, path: impl AsRef<Path>) -> Result<()> {
    write_json(report, path)
}

/// Finalize coordinator-owned and pair-owned run metadata, then perform the
/// sole native-v2 JSON write.
///
/// Backend/workload adapters return [`ReportPairRunFacts`]; the process
/// coordinator constructs [`ReportRunMetadata`] from its exact executable and
/// frozen registries. This function joins those typed values before
/// serialization and never parses or mutates raw report JSON.
/// Returns the finalized [`NativeReport`] so the caller can drive the native
/// post-report [`crate::export`] exporter plane over the exact committed report
/// (the report is otherwise consumed by the write).
pub fn finalize_and_write_native_report_json(
    report: NativeReport,
    run_metadata: ReportRunMetadata,
    facts: ReportPairRunFacts,
    path: impl AsRef<Path>,
) -> Result<NativeReport> {
    let report = report
        .finalize_run(run_metadata, facts)
        .context("finalizing native report run metadata")?;
    write_json(&report, path)?;
    Ok(report)
}

fn write_json(value: &impl Serialize, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(value).context("serializing summary report")?;
    let (temporary_path, mut temporary) = create_temporary_report(path)?;
    let result = (|| {
        temporary
            .write_all(json.as_bytes())
            .with_context(|| format!("writing temporary report {}", temporary_path.display()))?;
        temporary
            .sync_all()
            .with_context(|| format!("syncing temporary report {}", temporary_path.display()))?;
        drop(temporary);
        if path.exists() {
            anyhow::bail!(
                "authoritative native report already exists: {}",
                path.display()
            );
        }
        std::fs::rename(&temporary_path, path).with_context(|| {
            format!(
                "committing temporary report {} to {}",
                temporary_path.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

fn create_temporary_report(path: &Path) -> Result<(PathBuf, std::fs::File)> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        anyhow::anyhow!("native report path has no file name: {}", path.display())
    })?;
    for sequence in 0..1_024_u16 {
        let temporary_path = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("creating temporary report {}", temporary_path.display())
                });
            }
        }
    }
    anyhow::bail!(
        "could not reserve a temporary report beside {}",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_metadata() -> ReportRunMetadata {
        ReportRunMetadata::new(
            format!("blake3:{}", "a".repeat(64)),
            "online_http",
            "scheduled",
            Vec::new(),
            vec![
                crate::metrics_core::ReportEndpointProfileIdentity::new("default", "chat").unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn native_report_json_uses_the_metrics_first_v2_shape() {
        let mut summary = crate::metrics_core::AccumulatorSummary::new();
        summary.insert_finite(crate::metrics_core::MetricTag::RequestCount, 1.0);
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
        let report = NativeReport::new(&crate::metrics_core::AccumulatorSummary::new(), None);
        let path = std::env::temp_dir().join(format!(
            "aiperf_finalized_native_sum_{}.json",
            std::process::id()
        ));
        finalize_and_write_native_report_json(
            report,
            run_metadata(),
            ReportPairRunFacts::new(),
            &path,
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["run"]["transport"], "online_http");
        assert_eq!(value["run"]["workload"], "scheduled");
        assert_eq!(value["run"]["endpoint_profiles"][0]["endpoint_id"], "chat");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn report_commit_never_replaces_existing_authority() {
        let report = NativeReport::new(&crate::metrics_core::AccumulatorSummary::new(), None);
        let path = std::env::temp_dir().join(format!(
            "aiperf_existing_native_sum_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"existing-authority").unwrap();

        let error = write_native_report_json(&report, &path).unwrap_err();
        assert!(error.to_string().contains("already exists"), "{error:#}");
        assert_eq!(std::fs::read(&path).unwrap(), b"existing-authority");
        let _ = std::fs::remove_file(&path);
    }
}
