// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-shard artifact files for sharded exact-fold (Stage B).
//!
//! Stage A gave a `workers > 1` scheduled run per-shard EXACT accumulators merged
//! into one within-tolerance summary, but a run WITH per-record file artifacts still
//! fell back to the retain path because the shards had nowhere to stream their
//! records/raw/CSV/parquet/outputs. Stage B closes that: each thread-per-core shard
//! opens its OWN [`crate::engine::record_lane::RecordArtifactLane`] writing
//! to a per-shard temp directory (`<artifact_dir>/.shard-<id>/…`), streams one row
//! per completed record, and drops it — exactly like the single-thread lane. This
//! module fuses those per-shard files into the single final artifact at the
//! coordinator once every shard has finished.
//!
//! # Why a plain concatenation is correct
//!
//! Streamed per-record artifacts are emitted in COMPLETION order (the accepted
//! decision — the fold-and-drop path cannot buffer the whole run to re-sort), so the
//! final artifact is compared as a SORTED SET, never byte-for-byte against a specific
//! dispatch order. A shard's rows are a disjoint subset of the run's records (the
//! two-level partition tiles `0..total` exactly once — see
//! [`crate::engine::sharded_scheduled`]), and each shard builds its rows
//! with the SAME shared row builders the batch writers use, so a byte-append of the
//! shard files yields the union of rows, which is set-identical to the batch writer
//! over the union. No cross-shard ordering is needed.
//!
//! # Per-artifact merge rules
//!
//! - `records.jsonl` / `raw.jsonl`: byte-append each shard file (one compact JSON
//!   object + `\n` per row). The JSONL lane creates its file eagerly, so an empty
//!   shard contributes an empty file; the final file is always created (matching the
//!   batch writer's unconditional `File::create`).
//! - records CSV: lazy (no file when zero displayable rows across ALL shards). Write
//!   the header once (from the first shard file's first physical line), then append
//!   every shard file's DATA bytes (everything after its first `\n`). Skipping the
//!   header by the first newline is byte-safe even though a quoted error-message cell
//!   may itself contain an embedded newline.
//! - `profile_export.parquet`: row-group concatenation via
//!   [`crate::export::per_record_parquet::concat_per_record_parquet`] (schema + KV
//!   metadata from the first shard file). Parquet-feature only.
//! - `outputs.json`: parse each shard's `{schema_version, data:[…]}`, concatenate the
//!   `data` arrays, and write the final document once. Set-compared, so shard order
//!   is irrelevant.
//!
//! `inputs.json` is NOT merged here: under sharded exact-fold it is generated ONCE at
//! the coordinator from the resident dataset (the S4 up-front path), so the shards
//! never capture it.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::engine::protocol::ArtifactSpec;

/// The per-shard temp directory under the run's artifact tree. Each shard streams its
/// artifact lane here; the coordinator concatenates and then removes these.
pub(crate) fn shard_dir(artifact_dir: &Path, shard_id: usize) -> PathBuf {
    artifact_dir.join(format!(".shard-{shard_id}"))
}

/// This shard's temp path for the artifact whose FINAL relative path is `relative`
/// (its file name is reused inside the per-shard directory).
pub(crate) fn shard_artifact_path(
    artifact_dir: &Path,
    shard_id: usize,
    relative: &Path,
) -> PathBuf {
    let name = relative
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(relative));
    shard_dir(artifact_dir, shard_id).join(name)
}

/// The `0..workers` shard temp paths for one artifact's FINAL relative path.
fn shard_paths_for(artifact_dir: &Path, relative: &Path, workers: usize) -> Vec<PathBuf> {
    (0..workers)
        .map(|id| shard_artifact_path(artifact_dir, id, relative))
        .collect()
}

/// Fuse every per-shard artifact file into the single final artifact and remove the
/// per-shard temp directories. Called once at the coordinator after all shards of a
/// sharded exact-fold run finish. `inputs.json` is handled separately (up-front at
/// the coordinator), so it is not concatenated here.
pub(crate) fn concatenate_shard_artifacts(
    artifact_dir: &Path,
    artifacts: &ArtifactSpec,
    workers: usize,
) -> Result<()> {
    // Best-effort cleanup of the `.shard-<id>` temp dirs on EVERY exit path, including
    // an early `?` return from a failed concat below. The success path removes them
    // loudly (with error context) at the end; this guard is the belt-and-braces for the
    // error path, where the explicit loop is skipped — without it a failed concat would
    // leak the per-shard temp tree. On the success path the dirs are already gone, so
    // the guard's `remove_dir_all` is a no-op.
    struct ShardTempCleanup<'a> {
        artifact_dir: &'a Path,
        workers: usize,
    }
    impl Drop for ShardTempCleanup<'_> {
        fn drop(&mut self) {
            for id in 0..self.workers {
                let _ = std::fs::remove_dir_all(shard_dir(self.artifact_dir, id));
            }
        }
    }
    let _cleanup = ShardTempCleanup {
        artifact_dir,
        workers,
    };

    concatenate_artifacts(artifacts, artifact_dir, |relative| {
        shard_paths_for(artifact_dir, relative, workers)
    })?;
    // Drop the per-shard temp directories now that every file has been fused.
    for id in 0..workers {
        let dir = shard_dir(artifact_dir, id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).with_context(|| {
                format!("removing per-shard artifact directory {}", dir.display())
            })?;
        }
    }
    Ok(())
}

/// Fuse each cell's per-record artifact files into the single final artifact in the
/// run's real artifact dir (Stage D, same-host cellular path).
///
/// Unlike the sharded path, a cell's `artifact_dir` IS its `temp_root/cell-{id}` dir,
/// so under exact-fold (or the retain batch tail) each cell already wrote its merged
/// per-record artifacts at `cell_dir.join(relative)` — the FULL relative path, not the
/// flattened per-shard file name. The controller has every cell dir locally (same
/// host), so Stage D is exactly the Stage B concat with the per-cell dirs as the
/// shards: byte-append records/raw JSONL, header-once + data-append CSV, row-group
/// concat parquet, and data-array merge outputs.json. Set-compared (completion order
/// accepted), identical to the in-process sharded merge.
///
/// Cleanup of the cell dirs is owned by the controller's `ScratchTreeGuard` (it removes
/// the whole `temp_root`), so this fn — unlike [`concatenate_shard_artifacts`] — never
/// deletes its sources. `inputs.json` is NOT merged here: it is a single FULL-dataset
/// document (not per-record rows), generated identically by every cell, so the controller
/// COPIES one cell's copy verbatim via [`copy_cell_inputs_json`] rather than concatenating.
pub(crate) fn concatenate_cell_artifacts(
    cell_dirs: &[PathBuf],
    artifact_dir: &Path,
    artifacts: &ArtifactSpec,
) -> Result<()> {
    concatenate_artifacts(artifacts, artifact_dir, |relative| {
        cell_dirs.iter().map(|dir| dir.join(relative)).collect()
    })
}

/// Fuse each per-record artifact from N source dirs into the single final
/// artifact, choosing the merge rule per format (byte-append records/raw JSONL,
/// header-once + data-append CSV, row-group concat parquet, data-array merge
/// outputs.json). `sources_for` maps one artifact's relative path to the N
/// source files; the sharded and cellular paths differ only in that map (and in
/// who owns temp-dir cleanup), so the dispatch table lives here once.
fn concatenate_artifacts(
    artifacts: &ArtifactSpec,
    artifact_dir: &Path,
    sources_for: impl Fn(&Path) -> Vec<PathBuf>,
) -> Result<()> {
    if let Some(relative) = &artifacts.records_path {
        concat_jsonl(&sources_for(relative), &artifact_dir.join(relative))?;
    }
    if let Some(relative) = &artifacts.raw_path {
        concat_jsonl(&sources_for(relative), &artifact_dir.join(relative))?;
    }
    if let Some(relative) = &artifacts.records_csv_path {
        concat_csv(&sources_for(relative), &artifact_dir.join(relative))?;
    }
    if let Some(relative) = &artifacts.outputs_path {
        concat_outputs_json(&sources_for(relative), &artifact_dir.join(relative))?;
    }
    #[cfg(feature = "parquet")]
    if let Some(relative) = &artifacts.records_parquet_path {
        crate::export::per_record_parquet::concat_per_record_parquet(
            &sources_for(relative),
            &artifact_dir.join(relative),
        )?;
    }
    Ok(())
}

/// Copy one cell's `inputs.json` verbatim into the real artifact dir.
///
/// Unlike the per-record artifacts, `inputs.json` is a single FULL-dataset document —
/// the up-front (S4) `write_inputs_json` output over the whole resident dataset — that
/// every cell generates IDENTICALLY (same dataset, same seed): each cell's
/// `cell_dir.join(relative)` is byte-identical, so there is nothing to merge. The
/// controller simply copies the FIRST cell dir that produced the file into
/// `artifact_dir.join(relative)`, so a cellular run emits the exact same `inputs.json`
/// as the single-cell run (GenAI-Perf compat, always-on per `rust_wire`). A no-op when
/// no cell wrote the file (e.g. inputs disabled). Sources are owned by the controller's
/// `ScratchTreeGuard`, so — like [`concatenate_cell_artifacts`] — this never deletes them.
pub(crate) fn copy_cell_inputs_json(
    cell_dirs: &[PathBuf],
    artifact_dir: &Path,
    artifacts: &ArtifactSpec,
) -> Result<()> {
    let Some(relative) = &artifacts.inputs_path else {
        return Ok(());
    };
    let Some(source) = cell_dirs
        .iter()
        .map(|dir| dir.join(relative))
        .find(|path| path.exists())
    else {
        // No cell produced an inputs.json (e.g. inputs export disabled); nothing to copy.
        return Ok(());
    };
    let final_path = artifact_dir.join(relative);
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating inputs.json directory {}", parent.display()))?;
    }
    std::fs::copy(&source, &final_path).with_context(|| {
        format!(
            "copying cell inputs.json {} -> {}",
            source.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

/// Byte-append each existing shard JSONL file into `final_path` in shard order. The
/// final file is always created (matching the eager batch writer), so an all-empty
/// run leaves an empty file rather than none.
fn concat_jsonl(shard_paths: &[PathBuf], final_path: &Path) -> Result<()> {
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating merged JSONL directory {}", parent.display()))?;
    }
    let mut out = std::fs::File::create(final_path)
        .with_context(|| format!("creating merged JSONL {}", final_path.display()))?;
    for shard in shard_paths {
        if !shard.exists() {
            continue;
        }
        let bytes = std::fs::read(shard)
            .with_context(|| format!("reading shard JSONL {}", shard.display()))?;
        out.write_all(&bytes)
            .with_context(|| format!("appending shard JSONL {}", shard.display()))?;
    }
    out.flush()
        .with_context(|| format!("flushing merged JSONL {}", final_path.display()))
}

/// Merge shard records-CSV files: header once, then every shard's data bytes. No
/// final file is written when no shard produced one (the zero-displayable-row
/// contract). The header is skipped per shard by the first newline, which is
/// byte-safe even when a quoted cell holds an embedded newline.
fn concat_csv(shard_paths: &[PathBuf], final_path: &Path) -> Result<()> {
    let existing: Vec<&PathBuf> = shard_paths.iter().filter(|path| path.exists()).collect();
    if existing.is_empty() {
        return Ok(());
    }
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating merged CSV directory {}", parent.display()))?;
    }
    let mut out = std::fs::File::create(final_path)
        .with_context(|| format!("creating merged CSV {}", final_path.display()))?;
    let mut wrote_header = false;
    for shard in existing {
        let bytes = std::fs::read(shard)
            .with_context(|| format!("reading shard CSV {}", shard.display()))?;
        let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') else {
            // A shard file with no newline is header-only/malformed; nothing to append.
            continue;
        };
        if !wrote_header {
            // Header line including its trailing newline (identical across shards).
            out.write_all(&bytes[..=newline])
                .with_context(|| format!("writing merged CSV header {}", final_path.display()))?;
            wrote_header = true;
        }
        // Data portion: everything after the header line (already `\n`-terminated per
        // row by the shard writer).
        out.write_all(&bytes[newline + 1..])
            .with_context(|| format!("appending shard CSV data {}", shard.display()))?;
    }
    out.flush()
        .with_context(|| format!("flushing merged CSV {}", final_path.display()))
}

/// Merge shard `outputs.json` documents: concatenate their `data` arrays into one
/// final `{schema_version, data:[…]}` document. Set-compared downstream, so the
/// shard order of the concatenated entries does not matter.
fn concat_outputs_json(shard_paths: &[PathBuf], final_path: &Path) -> Result<()> {
    use crate::engine::records::OUTPUTS_SCHEMA_VERSION;

    let mut data: Vec<serde_json::Value> = Vec::new();
    for shard in shard_paths {
        if !shard.exists() {
            continue;
        }
        let bytes = std::fs::read(shard)
            .with_context(|| format!("reading shard outputs.json {}", shard.display()))?;
        let doc: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing shard outputs.json {}", shard.display()))?;
        if let Some(entries) = doc.get("data").and_then(serde_json::Value::as_array) {
            data.extend(entries.iter().cloned());
        }
    }
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating merged outputs.json directory {}",
                parent.display()
            )
        })?;
    }
    let document = serde_json::json!({
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "data": data,
    });
    let file = std::fs::File::create(final_path)
        .with_context(|| format!("creating merged outputs.json {}", final_path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &document)
        .with_context(|| format!("serializing merged outputs.json {}", final_path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing merged outputs.json {}", final_path.display()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::engine::record_lane::RecordArtifactLane;
    use crate::engine::records::{
        CapturedModelOutput, CapturedRecord, write_outputs_json, write_raw_records_jsonl,
        write_records_csv, write_records_jsonl,
    };
    use crate::metrics_core::{MetricsConfig, Phase, RecordIngest, TokenCounts};
    use uuid::Uuid;

    /// One synthetic captured record. `visible`/`reasoning` populate the outputs
    /// stream; `phase` gates warmup exclusion; a cancelled record exercises the CSV
    /// error tail. Distinct `session_num` per record keys the set comparisons.
    fn record(session_num: u64, phase: Phase, cancelled: bool) -> CapturedRecord {
        let mut ingest = RecordIngest::minimal(1_000_000, 11_000_000, phase);
        ingest.session_num = session_num;
        ingest.turn_index = 0;
        ingest.conversation_id = Some(format!("conversation-{session_num}"));
        ingest.canceled = cancelled;
        ingest.first_token_ns = Some(6_000_000);
        ingest.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
        ingest.tokens = TokenCounts {
            input: Some(8),
            output: Some(3),
            requested_output: Some(3),
            ..TokenCounts::default()
        };
        CapturedRecord {
            uuid: Uuid::from_u128(u128::from(session_num)),
            x_correlation_id: format!("session-{session_num}"),
            output: CapturedModelOutput::from_parts(
                &format!("answer-{session_num}"),
                Some(&format!("answer-{session_num}")),
                Some(&format!("why-{session_num}")),
            ),
            raw: None,
            ingest,
        }
    }

    /// A representative run: six profiling records (one cancelled) plus a warmup
    /// record `outputs.json` must exclude.
    fn union_records() -> Vec<CapturedRecord> {
        vec![
            record(1, Phase::Profiling, false),
            record(2, Phase::Profiling, false),
            record(3, Phase::Profiling, true),
            record(4, Phase::Profiling, false),
            record(5, Phase::Profiling, false),
            record(6, Phase::Profiling, false),
            record(9, Phase::Warmup, false),
        ]
    }

    /// The final relative artifact paths under a run directory.
    fn all_artifacts() -> ArtifactSpec {
        ArtifactSpec {
            records_path: Some(PathBuf::from("profile_export.jsonl")),
            raw_path: Some(PathBuf::from("profile_export_raw.jsonl")),
            records_csv_path: Some(PathBuf::from("profile_export_records.csv")),
            records_parquet_path: Some(PathBuf::from("profile_export.parquet")),
            outputs_path: Some(PathBuf::from("outputs.json")),
            inputs_path: None,
            trace: false,
        }
    }

    /// Drive one shard slice through its own per-shard lane (all artifacts enabled).
    fn drive_shard_lane(
        artifact_dir: &Path,
        shard_id: usize,
        slice: &[CapturedRecord],
        artifacts: &ArtifactSpec,
        config: &MetricsConfig,
    ) {
        let per_shard = |relative: &Option<PathBuf>| -> Option<PathBuf> {
            relative
                .as_ref()
                .map(|path| shard_artifact_path(artifact_dir, shard_id, path))
        };
        let lane = RecordArtifactLane::new(
            per_shard(&artifacts.records_path),
            per_shard(&artifacts.raw_path),
            per_shard(&artifacts.records_csv_path),
            per_shard(&artifacts.records_parquet_path),
            per_shard(&artifacts.outputs_path),
            artifacts.trace,
        )
        .unwrap()
        .expect("all artifacts requested");
        for captured in slice {
            lane.write(captured, config).unwrap();
        }
        lane.finish().unwrap();
    }

    /// The line SET (order-independent) of a text file, or empty when absent.
    fn line_set(path: &Path) -> BTreeMap<String, usize> {
        let mut set = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                *set.entry(line.to_string()).or_insert(0) += 1;
            }
        }
        set
    }

    /// Per-shard lanes over disjoint slices + concat == the batch writer over the
    /// union, as a SET, for records/raw/CSV/outputs (and parquet under its feature).
    /// One shard is deliberately empty (no rows) to exercise the empty-shard path.
    #[test]
    fn per_shard_concat_matches_batch_over_union() {
        let config = MetricsConfig::default();
        let records = union_records();
        let artifacts = all_artifacts();
        let workers = 3usize;

        // Disjoint shard slices: [0..3], [] (empty), [3..7].
        let shard_dir_root = tempfile::tempdir().unwrap();
        let slices: [&[CapturedRecord]; 3] = [&records[0..3], &[], &records[3..7]];
        for (id, slice) in slices.iter().enumerate() {
            drive_shard_lane(shard_dir_root.path(), id, slice, &artifacts, &config);
        }
        // The empty shard created its eager JSONL files but no CSV/parquet.
        assert!(
            !shard_artifact_path(
                shard_dir_root.path(),
                1,
                artifacts.records_csv_path.as_ref().unwrap()
            )
            .exists(),
            "an empty shard writes no CSV"
        );

        concatenate_shard_artifacts(shard_dir_root.path(), &artifacts, workers).unwrap();

        // The per-shard temp dirs are cleaned up.
        for id in 0..workers {
            assert!(
                !shard_dir(shard_dir_root.path(), id).exists(),
                "shard temp dir {id} removed"
            );
        }

        // Batch writers over the union into a separate dir.
        let batch_dir = tempfile::tempdir().unwrap();
        write_records_jsonl(
            &batch_dir.path().join("profile_export.jsonl"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_raw_records_jsonl(&batch_dir.path().join("profile_export_raw.jsonl"), &records)
            .unwrap();
        write_records_csv(
            &batch_dir.path().join("profile_export_records.csv"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_outputs_json(&batch_dir.path().join("outputs.json"), &records, &config).unwrap();

        // records.jsonl / raw.jsonl: line SET parity.
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            assert_eq!(
                line_set(&shard_dir_root.path().join(name)),
                line_set(&batch_dir.path().join(name)),
                "{name} line set must equal the batch writer over the union"
            );
        }

        // CSV: identical header + identical data-row SET.
        let merged_csv =
            std::fs::read_to_string(shard_dir_root.path().join("profile_export_records.csv"))
                .unwrap();
        let batch_csv =
            std::fs::read_to_string(batch_dir.path().join("profile_export_records.csv")).unwrap();
        assert_eq!(
            merged_csv.lines().next(),
            batch_csv.lines().next(),
            "CSV header identical"
        );
        assert_eq!(
            line_set(&shard_dir_root.path().join("profile_export_records.csv")),
            line_set(&batch_dir.path().join("profile_export_records.csv")),
            "CSV line set (header once + data rows) must equal the batch writer"
        );

        // outputs.json: same schema_version and same data SET after sorting by
        // (session_num, turn_index); warmup excluded (six profiling records).
        let sort_data = |path: &Path| -> serde_json::Value {
            let mut doc: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let data = doc["data"].as_array_mut().unwrap();
            data.sort_by_key(|row| {
                (
                    row["session_num"].as_u64().unwrap(),
                    row["turn_index"].as_u64().unwrap(),
                )
            });
            doc
        };
        let merged_outputs = sort_data(&shard_dir_root.path().join("outputs.json"));
        let batch_outputs = sort_data(&batch_dir.path().join("outputs.json"));
        assert_eq!(merged_outputs["schema_version"], "1.1");
        assert_eq!(merged_outputs["data"].as_array().unwrap().len(), 6);
        assert_eq!(merged_outputs, batch_outputs);

        // parquet: row-count parity (SET/schema parity proven in
        // `crate::export::per_record_parquet::concat_shards_matches_batch_over_union`).
        #[cfg(feature = "parquet")]
        {
            use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
            let count_rows = |path: &Path| -> usize {
                let file = std::fs::File::open(path).unwrap();
                ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .metadata()
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows() as usize)
                    .sum()
            };
            crate::export::per_record_parquet::write_per_record_parquet(
                &batch_dir.path().join("profile_export.parquet"),
                &records
                    .iter()
                    .map(|captured| {
                        crate::engine::records::per_record_parquet_row(captured, &config, false)
                    })
                    .collect::<Vec<_>>(),
                &crate::export::per_record_parquet::record_metric_columns(),
                false,
            )
            .unwrap();
            assert_eq!(
                count_rows(&shard_dir_root.path().join("profile_export.parquet")),
                count_rows(&batch_dir.path().join("profile_export.parquet")),
                "merged parquet row count equals the batch writer over the union"
            );
        }
    }

    /// Drive one cell slice through a lane pointed straight at the cell's own dir
    /// (`cell_dir.join(relative)`), exactly as a cell's execute path does when its
    /// `artifact_dir` is its `temp_root/cell-{id}` dir. Mirrors [`drive_shard_lane`] but
    /// with the cell-dir (full relative path) layout, not the flattened per-shard file.
    fn drive_cell_lane(
        cell_dir: &Path,
        slice: &[CapturedRecord],
        artifacts: &ArtifactSpec,
        config: &MetricsConfig,
    ) {
        std::fs::create_dir_all(cell_dir).unwrap();
        let in_cell = |relative: &Option<PathBuf>| -> Option<PathBuf> {
            relative.as_ref().map(|path| cell_dir.join(path))
        };
        let lane = RecordArtifactLane::new(
            in_cell(&artifacts.records_path),
            in_cell(&artifacts.raw_path),
            in_cell(&artifacts.records_csv_path),
            in_cell(&artifacts.records_parquet_path),
            in_cell(&artifacts.outputs_path),
            artifacts.trace,
        )
        .unwrap()
        .expect("all artifacts requested");
        for captured in slice {
            lane.write(captured, config).unwrap();
        }
        lane.finish().unwrap();
    }

    /// Stage D: per-cell lanes over disjoint slices (each writing to its own
    /// `cell-{id}` dir) + the controller's cellular concat == the batch writer over the
    /// union, as a SET, for records/raw/CSV/outputs (and parquet under its feature). One
    /// cell is deliberately empty. This is the same SET-parity bar the sharded concat
    /// meets, but exercising the cell-dir (full relative path) source layout and the
    /// controller-owned (no source deletion) cleanup contract.
    #[test]
    fn per_cell_concat_matches_batch_over_union() {
        let config = MetricsConfig::default();
        let records = union_records();
        let artifacts = all_artifacts();
        let cell_count = 3usize;

        // The controller's throwaway scratch tree with one dir per cell.
        let temp_root = tempfile::tempdir().unwrap();
        let cell_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|id| temp_root.path().join(format!("cell-{id}")))
            .collect();
        // Disjoint cell slices: [0..3], [] (empty), [3..7].
        let slices: [&[CapturedRecord]; 3] = [&records[0..3], &[], &records[3..7]];
        for (dir, slice) in cell_dirs.iter().zip(slices) {
            drive_cell_lane(dir, slice, &artifacts, &config);
        }

        // The real run artifact dir the controller fuses into (separate from the scratch).
        let run_dir = tempfile::tempdir().unwrap();
        concatenate_cell_artifacts(&cell_dirs, run_dir.path(), &artifacts).unwrap();

        // The controller does NOT delete cell dirs (ScratchTreeGuard owns the scratch).
        for dir in &cell_dirs {
            assert!(
                dir.exists(),
                "cell dir {} must survive concat",
                dir.display()
            );
        }

        // Batch writers over the union into a separate dir.
        let batch_dir = tempfile::tempdir().unwrap();
        write_records_jsonl(
            &batch_dir.path().join("profile_export.jsonl"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_raw_records_jsonl(&batch_dir.path().join("profile_export_raw.jsonl"), &records)
            .unwrap();
        write_records_csv(
            &batch_dir.path().join("profile_export_records.csv"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_outputs_json(&batch_dir.path().join("outputs.json"), &records, &config).unwrap();

        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            assert_eq!(
                line_set(&run_dir.path().join(name)),
                line_set(&batch_dir.path().join(name)),
                "{name} line set must equal the batch writer over the union"
            );
        }
        assert_eq!(
            line_set(&run_dir.path().join("profile_export_records.csv")),
            line_set(&batch_dir.path().join("profile_export_records.csv")),
            "CSV line set (header once + data rows) must equal the batch writer"
        );
        let sort_data = |path: &Path| -> serde_json::Value {
            let mut doc: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let data = doc["data"].as_array_mut().unwrap();
            data.sort_by_key(|row| {
                (
                    row["session_num"].as_u64().unwrap(),
                    row["turn_index"].as_u64().unwrap(),
                )
            });
            doc
        };
        assert_eq!(
            sort_data(&run_dir.path().join("outputs.json")),
            sort_data(&batch_dir.path().join("outputs.json")),
            "outputs.json data SET must equal the batch writer (warmup excluded)"
        );
        assert_eq!(
            sort_data(&run_dir.path().join("outputs.json"))["data"]
                .as_array()
                .unwrap()
                .len(),
            6,
            "six profiling records, warmup excluded"
        );

        #[cfg(feature = "parquet")]
        {
            use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
            let count_rows = |path: &Path| -> usize {
                let file = std::fs::File::open(path).unwrap();
                ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .metadata()
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows() as usize)
                    .sum()
            };
            crate::export::per_record_parquet::write_per_record_parquet(
                &batch_dir.path().join("profile_export.parquet"),
                &records
                    .iter()
                    .map(|captured| {
                        crate::engine::records::per_record_parquet_row(captured, &config, false)
                    })
                    .collect::<Vec<_>>(),
                &crate::export::per_record_parquet::record_metric_columns(),
                false,
            )
            .unwrap();
            assert_eq!(
                count_rows(&run_dir.path().join("profile_export.parquet")),
                count_rows(&batch_dir.path().join("profile_export.parquet")),
                "merged parquet row count equals the batch writer over the union"
            );
        }
    }

    /// An all-empty run (every shard saw zero displayable rows): the JSONL finals are
    /// empty files, and the lazy CSV/parquet/outputs behave like the batch writer.
    #[test]
    fn all_empty_shards_leave_empty_jsonl_and_no_csv() {
        let config = MetricsConfig::default();
        let artifacts = all_artifacts();
        let workers = 2usize;
        let dir = tempfile::tempdir().unwrap();
        for id in 0..workers {
            drive_shard_lane(dir.path(), id, &[], &artifacts, &config);
        }
        concatenate_shard_artifacts(dir.path(), &artifacts, workers).unwrap();

        // JSONL finals exist and are empty.
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            let bytes = std::fs::read(dir.path().join(name)).unwrap();
            assert!(bytes.is_empty(), "{name} is an empty file");
        }
        // No CSV (zero displayable rows), no parquet.
        assert!(!dir.path().join("profile_export_records.csv").exists());
        #[cfg(feature = "parquet")]
        assert!(!dir.path().join("profile_export.parquet").exists());
        // outputs.json still a valid empty document.
        let outputs: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("outputs.json")).unwrap())
                .unwrap();
        assert_eq!(outputs["data"].as_array().unwrap().len(), 0);
    }
}
