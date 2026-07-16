// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming per-record artifact lane: `records.jsonl`, `raw.jsonl`, and the
//! per-record CSV, written one row per record at completion.
//!
//! The exact-fold path (`execute::RunCapture`, task S1) folds each completed
//! record's metric scalars into the exact accumulator and drops the heavy
//! per-record data mid-run, so it cannot hold every record until end-of-run for the
//! legacy batch writers ([`crate::engine::records::write_records_jsonl`] /
//! `write_raw_records_jsonl` / `write_records_csv`). This lane holds each enabled
//! writer open for the whole run and appends one row per record as it completes,
//! then the fold drops the record — bounding peak memory while still emitting the
//! artifacts.
//!
//! Byte-parity contract: every row is produced by the exact same shared builders the
//! batch writers use ([`crate::engine::records::write_record_jsonl_row`],
//! `write_raw_record_jsonl_row`, `record_csv_header`, `record_csv_row`), so a lane
//! that sees the same record sequence emits byte-identical files. The CSV keeps the
//! batch writer's **lazy header** (the file and header are created only at the first
//! non-skipped row) and **skip-empty** rule (a record with no projected metric and no
//! error contributes no row; an all-skipped run writes no file at all).
//!
//! Modeled on [`crate::engine::heartbeat_lane::HeartbeatLane`]: a held-open `BufWriter` fed
//! one line per event and flushed at the end.
//!
//! Ordering note: the batch writers emit `captured` in dispatch (identity) order,
//! whereas the lane appends in the order records **complete**. For a serial / low-
//! concurrency run (e.g. the canonical single-turn parity scenario) completion order
//! equals dispatch order and the files are byte-identical to the legacy retain path;
//! under real concurrency the lane streams in completion (arrival) order, which the
//! order-independent exact-fold accumulator absorbs for the report but which the row
//! stream cannot reorder without reintroducing the O(records) buffer this lane exists
//! to avoid.

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::metrics_core::MetricsConfig;
use anyhow::{Context, Result};

#[cfg(feature = "parquet")]
use crate::export::per_record_parquet::{
    DEFAULT_ROW_GROUP_ROWS, StreamingPerRecordParquetWriter, record_metric_columns,
};

#[cfg(feature = "parquet")]
use crate::engine::records::per_record_parquet_row;
use crate::engine::records::{
    CapturedRecord, OUTPUTS_PREFIX, outputs_entry_indented, record_csv_header, record_csv_row,
    write_raw_record_jsonl_row, write_record_jsonl_row,
};

/// Create (truncating) one export file, creating its parent directory first, exactly
/// as the batch writers do before their `File::create`.
fn create_export_file(path: &Path, what: &str) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {what} directory {}", parent.display()))?;
    }
    let file = File::create(path).with_context(|| format!("creating {what} {}", path.display()))?;
    Ok(BufWriter::new(file))
}

/// The lazy CSV sub-writer: it defers `File::create` and the header until the first
/// non-skipped row, so an all-skipped run leaves no file — matching the batch
/// [`crate::engine::records::write_records_csv`].
struct CsvLaneWriter {
    path: PathBuf,
    include_trace: bool,
    writer: Option<BufWriter<File>>,
}

impl CsvLaneWriter {
    /// Append one record's CSV row, skipping records with no metric and no error and
    /// creating the file + header on the first non-skipped row.
    fn write(&mut self, captured: &CapturedRecord, config: &MetricsConfig) -> Result<()> {
        let Some(row) = record_csv_row(captured, config, self.include_trace) else {
            return Ok(());
        };
        if self.writer.is_none() {
            let mut writer = create_export_file(&self.path, "records CSV export")?;
            writer
                .write_all(record_csv_header(self.include_trace).as_bytes())
                .and_then(|()| writer.write_all(b"\n"))
                .context("writing records CSV header")?;
            self.writer = Some(writer);
        }
        let writer = self
            .writer
            .as_mut()
            .expect("csv writer created on first non-skipped row");
        writer
            .write_all(row.as_bytes())
            .and_then(|()| writer.write_all(b"\n"))
            .context("writing records CSV row")
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush().context("flushing records CSV export")?;
        }
        Ok(())
    }
}

/// The held-open `outputs.json` streaming sub-writer (task S4). It writes the pretty
/// document prefix eagerly (so an all-warmup / empty run still leaves a valid
/// `{"schema_version":"1.1","data":[]}`, matching the batch
/// [`crate::engine::records::write_outputs_json`]), appends each PROFILING record's pretty
/// entry at completion in **completion order**, then closes the array + object on
/// finish. Non-profiling records are skipped. The response text is dropped by the
/// fold immediately after this append, so retention stays O(in-flight).
///
/// Byte-format: prefix + (`\n` before the first entry, `,\n` before each later one) +
/// each [`outputs_entry_indented`] entry + the closing `\n  ]\n}` (or `]\n}` when no
/// entry was written), with NO trailing newline — exactly the bytes
/// `serde_json::to_writer_pretty` emits for the whole document. A set-comparison of
/// this stream against the batch document (both sorted by `(session_num, turn_index)`)
/// is therefore byte-identical.
struct OutputsLaneWriter {
    writer: BufWriter<File>,
    wrote_any: bool,
}

impl OutputsLaneWriter {
    fn create(path: &Path) -> Result<Self> {
        let mut writer = create_export_file(path, "native outputs export")?;
        writer
            .write_all(OUTPUTS_PREFIX.as_bytes())
            .context("writing outputs export prefix")?;
        Ok(Self {
            writer,
            wrote_any: false,
        })
    }

    fn write(&mut self, captured: &CapturedRecord, config: &MetricsConfig) -> Result<()> {
        let Some(entry) = outputs_entry_indented(captured, config)? else {
            return Ok(());
        };
        self.writer
            .write_all(if self.wrote_any { b",\n" } else { b"\n" })
            .context("writing outputs export entry separator")?;
        self.writer
            .write_all(entry.as_bytes())
            .context("writing outputs export entry")?;
        self.wrote_any = true;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let suffix: &[u8] = if self.wrote_any { b"\n  ]\n}" } else { b"]\n}" };
        self.writer
            .write_all(suffix)
            .context("writing outputs export suffix")?;
        self.writer.flush().context("flushing outputs export")
    }
}

/// A run-held lane that streams enabled per-record artifacts one row at a time.
///
/// Built once (when the exact-fold path requests any of records/raw/CSV), fed
/// [`Self::write`] per completed record, and flushed with [`Self::finish`] at run
/// end. Each writer is optional; a `None` slot is an artifact the run did not
/// request.
pub(crate) struct RecordArtifactLane {
    records: Option<RefCell<BufWriter<File>>>,
    raw: Option<RefCell<BufWriter<File>>>,
    csv: Option<RefCell<CsvLaneWriter>>,
    /// Held-open `outputs.json` streaming writer (task S4): each completed profiling
    /// record's pretty entry is appended in completion order, then the fold drops its
    /// response text. `None` when no `outputs.json` artifact is requested.
    outputs: Option<RefCell<OutputsLaneWriter>>,
    /// Held-open incremental Parquet writer (task S3): each completed record's wide
    /// row is buffered and flushed as a bounded row group, so the columnar sidecar
    /// streams without retaining every record. `None` when no Parquet artifact is
    /// requested. Only compiled under the `parquet` feature — a lite runner never
    /// streams Parquet (and such a run is not exact-fold-eligible for it).
    #[cfg(feature = "parquet")]
    parquet: RefCell<Option<StreamingPerRecordParquetWriter>>,
    include_trace: bool,
}

impl RecordArtifactLane {
    /// Build the lane from the resolved artifact paths, opening the records/raw JSONL
    /// writers eagerly (matching the batch writers' unconditional `File::create`, so
    /// an empty run still leaves an empty file) and deferring the CSV file to its
    /// first non-skipped row. Returns `None` when no lane artifact is requested, so a
    /// caller need not special-case the empty lane.
    pub(crate) fn new(
        records_path: Option<PathBuf>,
        raw_path: Option<PathBuf>,
        csv_path: Option<PathBuf>,
        records_parquet_path: Option<PathBuf>,
        outputs_path: Option<PathBuf>,
        include_trace: bool,
    ) -> Result<Option<Rc<Self>>> {
        // On a lite build the Parquet path never streams (the exact-fold gate keeps
        // Parquet disqualifying), so it does not count toward the "any artifact
        // requested" decision and is ignored here.
        #[cfg(feature = "parquet")]
        let any_parquet = records_parquet_path.is_some();
        #[cfg(not(feature = "parquet"))]
        let any_parquet = {
            let _ = &records_parquet_path;
            false
        };
        if records_path.is_none()
            && raw_path.is_none()
            && csv_path.is_none()
            && outputs_path.is_none()
            && !any_parquet
        {
            return Ok(None);
        }
        let records = records_path
            .map(|path| create_export_file(&path, "native record export"))
            .transpose()?
            .map(RefCell::new);
        let raw = raw_path
            .map(|path| create_export_file(&path, "native raw record export"))
            .transpose()?
            .map(RefCell::new);
        let csv = csv_path.map(|path| {
            RefCell::new(CsvLaneWriter {
                path,
                include_trace,
                writer: None,
            })
        });
        // The outputs writer opens its file and writes the document prefix eagerly,
        // matching the batch writer's unconditional `File::create` (an empty run still
        // leaves a valid `{"schema_version":"1.1","data":[]}`).
        let outputs = outputs_path
            .map(|path| OutputsLaneWriter::create(&path))
            .transpose()?
            .map(RefCell::new);
        // The Parquet writer defers file creation to its first row (matching the
        // batch writer's empty-rows-no-file contract), so it is built here without
        // touching the filesystem.
        #[cfg(feature = "parquet")]
        let parquet = RefCell::new(records_parquet_path.map(|path| {
            StreamingPerRecordParquetWriter::new(
                path,
                record_metric_columns(),
                include_trace,
                DEFAULT_ROW_GROUP_ROWS,
            )
        }));
        Ok(Some(Rc::new(Self {
            records,
            raw,
            csv,
            outputs,
            #[cfg(feature = "parquet")]
            parquet,
            include_trace,
        })))
    }

    /// Append one completed record's row to each enabled artifact. The caller drops
    /// the record afterward.
    pub(crate) fn write(&self, captured: &CapturedRecord, config: &MetricsConfig) -> Result<()> {
        if let Some(records) = &self.records {
            write_record_jsonl_row(
                &mut *records.borrow_mut(),
                captured,
                config,
                self.include_trace,
            )
            .context("streaming record export row")?;
        }
        if let Some(raw) = &self.raw {
            write_raw_record_jsonl_row(&mut *raw.borrow_mut(), captured)
                .context("streaming raw record export row")?;
        }
        if let Some(csv) = &self.csv {
            csv.borrow_mut().write(captured, config)?;
        }
        if let Some(outputs) = &self.outputs {
            outputs.borrow_mut().write(captured, config)?;
        }
        #[cfg(feature = "parquet")]
        if let Some(parquet) = self.parquet.borrow_mut().as_mut() {
            parquet
                .push(per_record_parquet_row(captured, config, self.include_trace))
                .context("streaming per-record parquet row")?;
        }
        Ok(())
    }

    /// Flush every open writer at run end. A CSV that never saw a non-skipped row was
    /// never created and stays absent, matching the batch writer.
    pub(crate) fn finish(&self) -> Result<()> {
        if let Some(records) = &self.records {
            records
                .borrow_mut()
                .flush()
                .context("flushing record export")?;
        }
        if let Some(raw) = &self.raw {
            raw.borrow_mut()
                .flush()
                .context("flushing raw record export")?;
        }
        if let Some(csv) = &self.csv {
            csv.borrow_mut().finish()?;
        }
        if let Some(outputs) = &self.outputs {
            outputs.borrow_mut().finish()?;
        }
        // `StreamingPerRecordParquetWriter::finish` consumes the writer (closing the
        // Arrow file footer), so take it out of the cell. A writer that saw no row
        // never created its file, matching the batch writer.
        #[cfg(feature = "parquet")]
        if let Some(parquet) = self.parquet.borrow_mut().take() {
            parquet
                .finish()
                .context("finalizing streaming per-record parquet")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::metrics_core::{Phase, RecordIngest, TokenCounts};
    use crate::transport_http::models::{RequestRecord, Response, SseMessage};
    use uuid::Uuid;

    use super::*;
    use crate::engine::records::{
        CapturedHttpExchange, CapturedModelOutput, write_raw_records_jsonl, write_records_csv,
        write_records_jsonl,
    };

    /// A representative mixed record slice: a normal streaming success carrying a raw
    /// HTTP exchange, plus a cancelled request (HTTP 499, error triple populated).
    fn sample_records() -> Vec<CapturedRecord> {
        let mut ok = RecordIngest::minimal(1_000_000, 11_000_000, Phase::Profiling);
        ok.session_num = 7;
        ok.turn_index = 0;
        ok.first_token_ns = Some(6_000_000);
        ok.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
        ok.tokens = TokenCounts {
            input: Some(8),
            output: Some(3),
            requested_output: Some(3),
            ..TokenCounts::default()
        };
        let payload = br#"{"model":"m","messages":[]}"#.to_vec();
        let transport = RequestRecord {
            start_ns: 1_000_000,
            end_ns: Some(11_000_000),
            request_body: payload.clone().into(),
            request_headers: BTreeMap::from([
                ("Authorization".into(), "Bearer super-secret".into()),
                ("X-Custom-Tracking".into(), "trace-123".into()),
            ]),
            status: Some(200),
            response_headers: BTreeMap::from([("content-type".into(), "text/event-stream".into())]),
            responses: vec![
                Response::Sse(SseMessage::parse(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
                    6_000_000,
                )),
                Response::Sse(SseMessage::parse("data: [DONE]", 11_000_000)),
            ],
            ..RequestRecord::default()
        };
        let success = CapturedRecord {
            uuid: Uuid::from_u128(7),
            x_correlation_id: "session-7".into(),
            output: CapturedModelOutput::from_parts("hi", None, None),
            raw: Some(CapturedHttpExchange {
                request_payload: payload,
                record: transport,
            }),
            ingest: ok,
        };

        let mut cancel = RecordIngest::minimal(1_000_000, 5_000_000, Phase::Profiling);
        cancel.session_num = 3;
        cancel.turn_index = 0;
        cancel.canceled = true;
        cancel.tokens.input = Some(128);
        let cancelled = CapturedRecord {
            uuid: Uuid::from_u128(3),
            x_correlation_id: "session-3".into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest: cancel,
        };

        vec![success, cancelled]
    }

    /// Drive a record slice through the lane, one `write` per record then `finish`.
    fn drive_lane(
        dir: &Path,
        records: &[CapturedRecord],
        include_trace: bool,
        config: &MetricsConfig,
    ) {
        let lane = RecordArtifactLane::new(
            Some(dir.join("profile_export.jsonl")),
            Some(dir.join("profile_export_raw.jsonl")),
            Some(dir.join("profile_export_records.csv")),
            None,
            None,
            include_trace,
        )
        .unwrap()
        .expect("lane requested three artifacts");
        for captured in records {
            lane.write(captured, config).unwrap();
        }
        lane.finish().unwrap();
    }

    /// Write the same slice through the legacy batch writers.
    fn drive_batch(
        dir: &Path,
        records: &[CapturedRecord],
        include_trace: bool,
        config: &MetricsConfig,
    ) {
        write_records_jsonl(
            &dir.join("profile_export.jsonl"),
            records,
            config,
            include_trace,
        )
        .unwrap();
        write_raw_records_jsonl(&dir.join("profile_export_raw.jsonl"), records).unwrap();
        write_records_csv(
            &dir.join("profile_export_records.csv"),
            records,
            config,
            include_trace,
        )
        .unwrap();
    }

    fn read_opt(path: &Path) -> Option<Vec<u8>> {
        std::fs::read(path).ok()
    }

    fn assert_lane_matches_batch(include_trace: bool) {
        let config = MetricsConfig::default();
        let records = sample_records();

        let lane_dir = tempfile::tempdir().unwrap();
        let batch_dir = tempfile::tempdir().unwrap();
        drive_lane(lane_dir.path(), &records, include_trace, &config);
        drive_batch(batch_dir.path(), &records, include_trace, &config);

        for name in [
            "profile_export.jsonl",
            "profile_export_raw.jsonl",
            "profile_export_records.csv",
        ] {
            let lane_bytes = read_opt(&lane_dir.path().join(name));
            let batch_bytes = read_opt(&batch_dir.path().join(name));
            assert_eq!(
                lane_bytes, batch_bytes,
                "lane vs batch mismatch for {name} (include_trace={include_trace})"
            );
        }
        // The mixed slice always yields at least one non-skipped CSV row, so the CSV
        // file exists in both.
        assert!(lane_dir.path().join("profile_export_records.csv").exists());
    }

    #[test]
    fn lane_matches_batch_without_trace() {
        assert_lane_matches_batch(false);
    }

    #[test]
    fn lane_matches_batch_with_trace() {
        assert_lane_matches_batch(true);
    }

    #[test]
    fn empty_slice_matches_batch_and_leaves_no_csv() {
        let config = MetricsConfig::default();
        let records: Vec<CapturedRecord> = Vec::new();

        let lane_dir = tempfile::tempdir().unwrap();
        let batch_dir = tempfile::tempdir().unwrap();
        drive_lane(lane_dir.path(), &records, false, &config);
        drive_batch(batch_dir.path(), &records, false, &config);

        // JSONL writers create an (empty) file eagerly in both paths.
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            let lane_bytes = read_opt(&lane_dir.path().join(name));
            let batch_bytes = read_opt(&batch_dir.path().join(name));
            assert_eq!(
                lane_bytes,
                Some(Vec::new()),
                "{name} should be an empty file"
            );
            assert_eq!(lane_bytes, batch_bytes, "lane vs batch mismatch for {name}");
        }
        // An all-skipped (here: empty) CSV writes no file in either path.
        assert!(
            !lane_dir.path().join("profile_export_records.csv").exists(),
            "lane must not create a CSV with no rows"
        );
        assert!(!batch_dir.path().join("profile_export_records.csv").exists());
    }

    #[test]
    fn new_returns_none_when_no_artifact_requested() {
        assert!(
            RecordArtifactLane::new(None, None, None, None, None, false)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn csv_only_lane_writes_only_csv() {
        let config = MetricsConfig::default();
        let records = sample_records();
        let dir = tempfile::tempdir().unwrap();
        let lane = RecordArtifactLane::new(
            None,
            None,
            Some(dir.path().join("profile_export_records.csv")),
            None,
            None,
            false,
        )
        .unwrap()
        .expect("csv artifact requested");
        for captured in &records {
            lane.write(captured, &config).unwrap();
        }
        lane.finish().unwrap();

        assert!(dir.path().join("profile_export_records.csv").exists());
        assert!(!dir.path().join("profile_export.jsonl").exists());
        assert!(!dir.path().join("profile_export_raw.jsonl").exists());
    }

    /// The lane wires the streaming Parquet sidecar (task S3): a parquet-only lane
    /// pushes each record's wide row and, on finish, materializes a non-empty file
    /// while emitting none of the other artifacts. The columnar row-set parity vs the
    /// batch writer is proven in `crate::export::per_record_parquet` (the runner crate
    /// has no direct `arrow`/`parquet` dependency to read the file back here).
    #[cfg(feature = "parquet")]
    #[test]
    fn lane_streams_parquet_only() {
        let config = MetricsConfig::default();
        let records = sample_records();
        let dir = tempfile::tempdir().unwrap();
        let lane_path = dir.path().join("profile_export.parquet");

        let lane = RecordArtifactLane::new(None, None, None, Some(lane_path.clone()), None, false)
            .unwrap()
            .expect("parquet artifact requested");
        for captured in &records {
            lane.write(captured, &config).unwrap();
        }
        lane.finish().unwrap();

        assert!(lane_path.exists(), "the streamed parquet sidecar exists");
        assert!(
            std::fs::metadata(&lane_path).unwrap().len() > 0,
            "the streamed parquet sidecar is non-empty"
        );
        assert!(!dir.path().join("profile_export.jsonl").exists());
        assert!(!dir.path().join("profile_export_raw.jsonl").exists());
        assert!(!dir.path().join("profile_export_records.csv").exists());
    }

    /// A parquet-only lane that never sees a row leaves no file, matching the batch
    /// writer's empty-rows contract.
    #[cfg(feature = "parquet")]
    #[test]
    fn empty_parquet_lane_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let lane_path = dir.path().join("profile_export.parquet");
        let lane = RecordArtifactLane::new(None, None, None, Some(lane_path.clone()), None, false)
            .unwrap()
            .expect("parquet artifact requested");
        lane.finish().unwrap();
        assert!(!lane_path.exists());
    }

    /// A mixed slice for the outputs stream: three profiling records (out of
    /// `(session_num, turn_index)` order) with distinct response/reasoning text, plus a
    /// warmup record that `outputs.json` must exclude.
    fn outputs_records() -> Vec<CapturedRecord> {
        use crate::engine::records::CapturedModelOutput;

        let mut record = |session_num: u64,
                          turn_index: u32,
                          visible: &str,
                          reasoning: Option<&str>,
                          phase: Phase| {
            let mut ingest = RecordIngest::minimal(1_000_000, 11_000_000, phase);
            ingest.session_num = session_num;
            ingest.turn_index = turn_index;
            ingest.conversation_id = Some(format!("conversation-{session_num}"));
            ingest.first_token_ns = Some(6_000_000);
            ingest.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
            ingest.tokens = TokenCounts {
                input: Some(8),
                output: Some(3),
                requested_output: Some(3),
                ..TokenCounts::default()
            };
            CapturedRecord {
                uuid: Uuid::from_u128(u128::from(session_num) * 10 + u128::from(turn_index)),
                x_correlation_id: format!("session-{session_num}"),
                output: CapturedModelOutput::from_parts(visible, Some(visible), reasoning),
                raw: None,
                ingest,
            }
        };

        vec![
            record(2, 1, "second answer", Some("second why"), Phase::Profiling),
            record(1, 0, "first answer", None, Phase::Profiling),
            record(2, 0, "middle answer", Some("mid why"), Phase::Profiling),
            record(9, 0, "warmup answer", None, Phase::Warmup),
        ]
    }

    fn drive_outputs_lane(dir: &Path, records: &[CapturedRecord], config: &MetricsConfig) {
        let lane = RecordArtifactLane::new(
            None,
            None,
            None,
            None,
            Some(dir.join("outputs.json")),
            false,
        )
        .unwrap()
        .expect("outputs artifact requested");
        for captured in records {
            lane.write(captured, config).unwrap();
        }
        lane.finish().unwrap();
    }

    /// Feeding the outputs stream in already-sorted `(session_num, turn_index)` order
    /// yields bytes byte-identical to the batch `write_outputs_json` document, proving
    /// the streamed prefix/entry-indentation/suffix match `to_writer_pretty` exactly.
    #[test]
    fn outputs_stream_in_sorted_order_matches_batch_bytes() {
        use crate::engine::records::write_outputs_json;

        let config = MetricsConfig::default();
        let mut records = outputs_records();
        // Only profiling records reach the batch document; sort them the same way.
        records.sort_by_key(|r| (r.ingest.session_num, r.ingest.turn_index));

        let lane_dir = tempfile::tempdir().unwrap();
        let batch_dir = tempfile::tempdir().unwrap();
        drive_outputs_lane(lane_dir.path(), &records, &config);
        write_outputs_json(&batch_dir.path().join("outputs.json"), &records, &config).unwrap();

        let lane_bytes = std::fs::read(lane_dir.path().join("outputs.json")).unwrap();
        let batch_bytes = std::fs::read(batch_dir.path().join("outputs.json")).unwrap();
        assert_eq!(
            String::from_utf8(lane_bytes).unwrap(),
            String::from_utf8(batch_bytes).unwrap()
        );
    }

    /// In completion (arrival) order the stream is a reordered SET of the batch
    /// document's entries: sorting both `data` arrays by `(session_num, turn_index)`
    /// yields byte-identical documents, the warmup record is excluded, and the schema
    /// version matches.
    #[test]
    fn outputs_stream_is_sorted_set_equal_to_batch() {
        use crate::engine::records::write_outputs_json;
        use serde_json::Value;

        let config = MetricsConfig::default();
        let records = outputs_records(); // deliberately out of order, with a warmup row

        let lane_dir = tempfile::tempdir().unwrap();
        let batch_dir = tempfile::tempdir().unwrap();
        drive_outputs_lane(lane_dir.path(), &records, &config);
        write_outputs_json(&batch_dir.path().join("outputs.json"), &records, &config).unwrap();

        let sort_data = |path: &Path| -> Value {
            let mut doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let data = doc["data"].as_array_mut().unwrap();
            data.sort_by_key(|row| {
                (
                    row["session_num"].as_u64().unwrap(),
                    row["turn_index"].as_u64().unwrap(),
                )
            });
            doc
        };
        let lane_doc = sort_data(&lane_dir.path().join("outputs.json"));
        let batch_doc = sort_data(&batch_dir.path().join("outputs.json"));

        assert_eq!(lane_doc["schema_version"], "1.1");
        // Warmup is excluded: only the three profiling records survive.
        assert_eq!(lane_doc["data"].as_array().unwrap().len(), 3);
        assert_eq!(lane_doc, batch_doc);
    }

    /// An all-warmup (or empty) run still leaves a valid empty document byte-identical
    /// to the batch writer's `{"schema_version":"1.1","data":[]}`.
    #[test]
    fn outputs_stream_empty_matches_batch() {
        use crate::engine::records::write_outputs_json;

        let config = MetricsConfig::default();
        let warmup: Vec<CapturedRecord> = outputs_records()
            .into_iter()
            .filter(|r| r.ingest.phase == Phase::Warmup)
            .collect();

        let lane_dir = tempfile::tempdir().unwrap();
        let batch_dir = tempfile::tempdir().unwrap();
        drive_outputs_lane(lane_dir.path(), &warmup, &config);
        write_outputs_json(&batch_dir.path().join("outputs.json"), &warmup, &config).unwrap();

        let lane_bytes = std::fs::read(lane_dir.path().join("outputs.json")).unwrap();
        let batch_bytes = std::fs::read(batch_dir.path().join("outputs.json")).unwrap();
        assert_eq!(
            String::from_utf8(lane_bytes).unwrap(),
            String::from_utf8(batch_bytes).unwrap()
        );
    }
}
