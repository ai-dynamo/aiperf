// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON and CSV sinks for the `--dry-run` dataset analysis.
//!
//! [`write_dataset_analysis_json`] emits the full [`DatasetAnalysis`] as pretty
//! JSON; [`write_dataset_analysis_csv`] emits one row per distribution metric
//! using the genai_perf stat-key columns so the file loads with the same schema
//! as the primary metrics export.

use std::io::Write;
use std::path::Path;

use crate::dataset::analysis::{DatasetAnalysis, StatSummary};
use crate::metrics_core::definition::definition;

/// Distribution stat-key columns, matching the genai_perf CSV schema.
const STAT_KEYS: &[&str] = &[
    "avg", "min", "max", "sum", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "std",
];

/// Return the CSV row-name token for an analyzer base concept, read from the
/// metric registry rather than a hardcoded literal.
///
/// The token is the base def's `short_header` (e.g. `analyzer.per_turn_isl` →
/// `isl`); the concrete emitted names (`isl`, `turn0_isl`, `conv[a]_osl`, …) are
/// composed from these tokens so the CSV stays byte-compatible with prior output.
fn token(id: &str) -> &'static str {
    definition(id)
        .and_then(|d| d.short_header)
        .unwrap_or_else(|| panic!("analyzer base definition {id} missing short_header"))
}

/// Write the analysis as pretty-printed JSON to `path`.
///
/// Every serialized `f64` is already finite (guaranteed by `analyze`), so the
/// output never contains `NaN`/`Infinity` tokens.
pub fn write_dataset_analysis_json(a: &DatasetAnalysis, path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    write_dataset_analysis_json_writer(a, file)
}

/// Serialize the analysis as pretty JSON and commit it to `writer`.
fn write_dataset_analysis_json_writer<W: Write>(
    a: &DatasetAnalysis,
    writer: W,
) -> std::io::Result<()> {
    let mut writer = std::io::BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, a).map_err(std::io::Error::other)?;
    writer.flush()
}

/// Write the analysis's distribution metrics as a CSV to `path`.
///
/// The header is `metric` followed by the [`STAT_KEYS`] columns; each subsequent
/// row is one named distribution (`StatSummary`) drawn from the length,
/// per-turn-index, and timeline sections.
pub fn write_dataset_analysis_csv(a: &DatasetAnalysis, path: &Path) -> std::io::Result<()> {
    let mut rows: Vec<(String, &StatSummary)> = Vec::new();
    if let Some(s) = a.shape.turns_per_conversation.as_ref() {
        rows.push((token("analyzer.turns_per_conversation").to_string(), s));
    }
    if let Some(s) = a.lengths.isl.as_ref() {
        rows.push((token("analyzer.isl").to_string(), s));
    }
    if let Some(s) = a.lengths.osl.as_ref() {
        rows.push((token("analyzer.osl").to_string(), s));
    }
    if let Some(s) = a.lengths.total.as_ref() {
        rows.push((token("analyzer.total").to_string(), s));
    }
    if let Some(s) = a.lengths.isl_osl_ratio.as_ref() {
        rows.push((token("analyzer.isl_osl_ratio").to_string(), s));
    }
    for row in &a.turns.by_index {
        let ti = row.turn_index;
        if let Some(s) = row.isl.as_ref() {
            rows.push((format!("turn{ti}_{}", token("analyzer.per_turn_isl")), s));
        }
        if let Some(s) = row.osl.as_ref() {
            rows.push((format!("turn{ti}_{}", token("analyzer.per_turn_osl")), s));
        }
        if let Some(s) = row.authored_think_time_ms.as_ref() {
            rows.push((
                format!("turn{ti}_{}", token("analyzer.per_turn_think_time_ms")),
                s,
            ));
        }
    }
    if let Some(t) = a.timeline.as_ref()
        && let Some(s) = t.queue.queue_delay_ms.as_ref()
    {
        rows.push((token("analyzer.queue_delay_ms").to_string(), s));
    }
    // Per-conversation length rows, emitted only when the breakdown was requested.
    if let Some(conversations) = a.conversations.as_ref() {
        for summary in conversations {
            let id = &summary.conversation_id;
            if let Some(s) = summary.lengths.isl.as_ref() {
                rows.push((format!("conv[{id}]_{}", token("analyzer.isl")), s));
            }
            if let Some(s) = summary.lengths.osl.as_ref() {
                rows.push((format!("conv[{id}]_{}", token("analyzer.osl")), s));
            }
        }
    }

    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    let mut header = String::from("metric");
    for key in STAT_KEYS {
        header.push(',');
        header.push_str(key);
    }
    writeln!(writer, "{header}")?;

    for (name, s) in rows {
        let mut line = name;
        for key in STAT_KEYS {
            line.push(',');
            line.push_str(&format_stat(s, key));
        }
        writeln!(writer, "{line}")?;
    }
    writer.flush()
}

/// Select and format one stat field from a [`StatSummary`] as a plain decimal.
fn format_stat(s: &StatSummary, key: &str) -> String {
    let value = match key {
        "avg" => s.mean,
        "min" => s.min,
        "max" => s.max,
        "sum" => s.sum,
        "p1" => s.p1,
        "p5" => s.p5,
        "p10" => s.p10,
        "p25" => s.p25,
        "p50" => s.p50,
        "p75" => s.p75,
        "p90" => s.p90,
        "p95" => s.p95,
        "p99" => s.p99,
        "std" => s.std,
        _ => 0.0,
    };
    format!("{value}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::analysis::*;

    #[derive(Default)]
    struct FlushFailWriter(Vec<u8>);

    impl Write for FlushFailWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("flush failed"))
        }
    }

    fn tiny() -> DatasetAnalysis {
        let turns = vec![AnalyzedTurn {
            conversation_id: "a".into(),
            turn_index: 0,
            input_tokens: 32,
            max_output_tokens: 8,
            delay_ms: None,
            block_ids: Some(vec![1, 2]),
            system_handle: None,
        }];
        let records = vec![AnalyzedRecord {
            conversation_id: "a".into(),
            turn_index: 0,
            start_ns: 0,
            end_ns: 1_000_000_000,
            admit_ns: Some(0),
            first_token_ns: Some(0),
            input_tokens: 32,
            output_tokens: 8,
            token_arrival_ns: vec![],
        }];
        analyze(&turns, &records, &AnalysisOptions::default())
    }

    #[test]
    fn json_and_csv_write_finite() {
        let dir = tempfile::tempdir().unwrap();
        let a = tiny();
        let jp = dir.path().join("dataset_analysis.json");
        write_dataset_analysis_json(&a, &jp).unwrap();
        let json = std::fs::read_to_string(&jp).unwrap();
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(json.contains("\"cache\""));
        assert!(!json.contains("NaN"));
        let cp = dir.path().join("dataset_analysis.csv");
        write_dataset_analysis_csv(&a, &cp).unwrap();
        let csv = std::fs::read_to_string(&cp).unwrap();
        assert!(csv.lines().next().unwrap().contains("p50"));
    }

    #[test]
    fn json_writer_propagates_flush_error() {
        let error = write_dataset_analysis_json_writer(&tiny(), FlushFailWriter::default())
            .expect_err("flush failure must be returned");
        assert_eq!(error.to_string(), "flush failed");
    }
}
