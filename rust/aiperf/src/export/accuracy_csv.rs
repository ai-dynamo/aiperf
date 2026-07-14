// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust accuracy sink: `accuracy_results.csv`.
//!
//! Ports the Python `accuracy/accuracy_data_exporter.py` (`AccuracyDataExporter.export`,
//! `src/aiperf/accuracy/accuracy_data_exporter.py:53-108`). That exporter consumes the
//! per-task accuracy `MetricResult` rows produced by
//! `accuracy/accuracy_results_processor.py:101-168` (`summarize`) and writes one CSV
//! row per task plus a leading OVERALL row.
//!
//! The native-v2 report carries the same facts directly on
//! [`AccuracyAnalysis::summary`](crate::metrics_core::AccuracyAnalysis): the `overall`
//! [`AccuracyRollup`](crate::metrics_core::AccuracyRollup) and the `per_task`
//! rollups (a `BTreeMap`, i.e. already sorted by task id — the same alphabetical
//! order the Python processor emits via `sorted(self._task_total.keys())`). This
//! sink writes that summary byte-for-byte against the Python output; there is no
//! intermediate metric-record projection.
//!
//! # Column ↔ source mapping (Python `path:line`)
//! Header `task,correct,total,unparsed,accuracy`
//! (`accuracy_data_exporter.py:106`). Per row
//! (`accuracy_data_exporter.py:89-97`):
//! - `task` — `"OVERALL"` for the overall rollup, else the task id.
//! - `correct` — `int(m.sum)` = rollup `correct_count` (processor line 123/138).
//! - `total` — `int(m.count)` = rollup `n` (processor line 121/136).
//! - `unparsed` — the paired `accuracy.unparsed[.task.<name>]` sum = rollup
//!   `unparsed_count` (processor line 150/164).
//! - `accuracy` — `f"{m.current:.4f}"` where `m.current` is the correct/total
//!   ratio (processor line 122/137); empty string when absent. The native
//!   `AccuracyRollup::accuracy` is the identical `correct_count as f64 / n as f64`
//!   quotient (`metrics_core::accuracy::RollupBuilder::finish`), so `{:.4}` and
//!   Python `:.4f` round the same IEEE-754 value identically.
//!
//! # Emission gate
//! Only fires in accuracy mode (`cfg.accuracy_csv.enabled`). When the report
//! carries no accuracy analysis, or the overall population is empty, the Python
//! exporter returns without writing any file
//! (`accuracy_data_exporter.py:60-67`; the processor emits an empty list for an
//! empty population, `accuracy_results_processor.py:114`). This sink matches: it
//! writes nothing in those cases.

use std::fs::File;
use std::path::Path;

use crate::export::{ExportConfig, Exporter};
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
            // No accuracy analysis: the Python exporter returns without writing.
            return Ok(());
        };
        let summary = &analysis.summary;
        if summary.overall.n == 0 {
            // Empty population: the Python processor emits no metric rows, so the
            // exporter writes no file.
            return Ok(());
        }

        std::fs::create_dir_all(artifact_dir)?;
        let path = artifact_dir.join(ACCURACY_CSV_FILE);
        let file = File::create(&path)?;
        let mut writer = csv::WriterBuilder::new()
            .terminator(csv::Terminator::CRLF)
            .from_writer(file);

        writer.write_record(["task", "correct", "total", "unparsed", "accuracy"])?;
        write_row(&mut writer, "OVERALL", &summary.overall)?;
        for (task, rollup) in &summary.per_task {
            write_row(&mut writer, task.as_str(), rollup)?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Write one CSV data row for a rollup, matching the Python column formatting.
fn write_row<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    task: &str,
    rollup: &AccuracyRollup,
) -> anyhow::Result<()> {
    // `accuracy` mirrors Python `f"{m.current:.4f}" if m.current is not None else ""`.
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
