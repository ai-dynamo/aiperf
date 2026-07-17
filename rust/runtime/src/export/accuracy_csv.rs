// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accuracy CSV sink for `accuracy_results.csv`.
//!
//! The header is `task,correct,total,unparsed,accuracy`. A leading `OVERALL` row
//! precedes task rows in alphabetical task-id order:
//! - `task` — `"OVERALL"` for the overall rollup, else the task id.
//! - `correct`, `total`, and `unparsed` are integer rollup counts.
//! - `accuracy` is the correct/total ratio to four decimals, or empty when absent.
//!
//! No file is written without accuracy analysis or when the overall population is empty.

use std::fs::File;
use std::path::Path;

use crate::export::{ExportConfig, Exporter, crlf_csv_writer};
use crate::metrics_core::{AccuracyRollup, NativeReport};

/// Fixed output file name, joined onto the run's artifact directory. A constant
/// literal, so no path-traversal component can enter the join.
const ACCURACY_CSV_FILE: &str = "accuracy_results.csv";

/// Accuracy CSV export policy. Enabled in accuracy mode.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AccuracyCsvExportConfig {
    /// Emit `accuracy_results.csv`.
    pub enabled: bool,
}

/// The accuracy CSV [`Exporter`].
pub struct AccuracyCsvExporter;

impl Exporter for AccuracyCsvExporter {
    fn name(&self) -> &'static str {
        "accuracy_csv"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.accuracy_csv.enabled
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let Some(analysis) = report.accuracy.as_ref() else {
            // No analysis produces no artifact.
            return Ok(());
        };
        let summary = &analysis.summary;
        if summary.overall.n == 0 {
            // Empty populations produce no artifact.
            return Ok(());
        }

        std::fs::create_dir_all(artifact_dir)?;
        let path = artifact_dir.join(ACCURACY_CSV_FILE);
        let file = File::create(&path)?;
        let mut writer = crlf_csv_writer(file);

        writer.write_record(["task", "correct", "total", "unparsed", "accuracy"])?;
        write_row(&mut writer, "OVERALL", &summary.overall)?;
        for (task, rollup) in &summary.per_task {
            write_row(&mut writer, task.as_str(), rollup)?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Write one CSV data row for a rollup.
fn write_row<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    task: &str,
    rollup: &AccuracyRollup,
) -> anyhow::Result<()> {
    // Missing accuracy is an empty cell; present values use four decimals.
    let accuracy = match rollup.accuracy {
        Some(value) => format!("{value:.4}"),
        None => String::new(),
    };
    writer.write_record([
        task,
        &rollup.correct_count.to_string(),
        &rollup.n.to_string(),
        &rollup.unparsed_count.to_string(),
        &accuracy,
    ])?;
    Ok(())
}

#[cfg(test)]
mod tests;
