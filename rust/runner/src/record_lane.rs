// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming per-record artifact lane: `records.jsonl`, `raw.jsonl`, and the
//! per-record CSV, written one row per record at completion.
//!
//! The exact-fold path (`execute::RunCapture`, task S1) folds each completed
//! record's metric scalars into the exact accumulator and drops the heavy
//! per-record data mid-run, so it cannot hold every record until end-of-run for the
//! legacy batch writers ([`crate::records::write_records_jsonl`] /
//! `write_raw_records_jsonl` / `write_records_csv`). This lane holds each enabled
//! writer open for the whole run and appends one row per record as it completes,
//! then the fold drops the record — bounding peak memory while still emitting the
//! artifacts.
//!
//! Byte-parity contract: every row is produced by the exact same shared builders the
//! batch writers use ([`crate::records::write_record_jsonl_row`],
//! `write_raw_record_jsonl_row`, `record_csv_header`, `record_csv_row`), so a lane
//! that sees the same record sequence emits byte-identical files. The CSV keeps the
//! batch writer's **lazy header** (the file and header are created only at the first
//! non-skipped row) and **skip-empty** rule (a record with no projected metric and no
//! error contributes no row; an all-skipped run writes no file at all).
//!
//! Modeled on [`crate::heartbeat_lane::HeartbeatLane`]: a held-open `BufWriter` fed
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

use aiperf::metrics_core::MetricsConfig;
use anyhow::{Context, Result};

use crate::records::{
    CapturedRecord, record_csv_header, record_csv_row, write_raw_record_jsonl_row,
    write_record_jsonl_row,
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
/// [`crate::records::write_records_csv`].
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
        include_trace: bool,
    ) -> Result<Option<Rc<Self>>> {
        if records_path.is_none() && raw_path.is_none() && csv_path.is_none() {
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
        Ok(Some(Rc::new(Self {
            records,
            raw,
            csv,
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aiperf::metrics_core::{Phase, RecordIngest, TokenCounts};
    use aiperf::transport_http::models::{RequestRecord, Response, SseMessage};
    use uuid::Uuid;

    use super::*;
    use crate::records::{
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
            RecordArtifactLane::new(None, None, None, false)
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
}
