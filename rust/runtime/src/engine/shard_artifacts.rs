// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-shard artifact merging for sharded exact-fold execution.
//!
//! Streamed per-record artifacts are emitted in completion order, so the
//! final artifact is compared as a SORTED SET, never byte-for-byte against a specific
//! dispatch order. A shard's rows are a disjoint subset of the run's records (the
//! two-level partition tiles `0..total` exactly once — see
//! [`crate::engine::sharded_scheduled`]), and each shard builds its rows
//! with the SAME shared row builders the batch writers use, so a byte-append of the
//! shard files yields the union of rows, which is set-identical to the batch writer
//! over the union. No cross-shard ordering is needed.
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
//! `inputs.json` is generated once at the coordinator from the resident dataset, so shards
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
/// run's artifact directory.
///
/// Unlike the sharded path, a cell's `artifact_dir` IS its `temp_root/cell-{id}` dir,
/// Under exact-fold or the retain batch tail, each cell writes its merged
/// per-record artifacts at `cell_dir.join(relative)` — the FULL relative path, not the
/// flattened per-shard file name. The controller has every cell dir locally (same
/// host), so per-cell directories are merged as shards: byte-append records/raw
/// JSONL, header-once + data-append CSV, row-group
/// concat parquet, and data-array merge outputs.json. Set-compared (completion order
/// accepted), identical to the in-process sharded merge.
///
/// Cleanup of the cell dirs is owned by the controller's `ScratchTreeGuard` (it removes
/// the whole `temp_root`), so this fn — unlike [`concatenate_shard_artifacts`] — never
/// deletes its sources. `inputs.json` is NOT merged here: it is a session document rather
/// than per-record rows, so the controller merges it separately (union of the per-cell
/// slices, re-interleaved round-robin) via [`merge_cell_inputs_json`].
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

/// Merge every cell's `inputs.json` into one full-dataset document.
///
/// A cell's `inputs.json` covers only the conversations THAT cell owns: the round-robin
/// partition (`position % cell_count == cell_id`) slices the resident dataset before the
/// document is generated, so cell `k` lists sessions `k, k+C, k+2C, …`. The slices are
/// disjoint and tile the dataset exactly once, so their union is the single-process
/// document — which is why this merges rather than copying one cell's file (copying cell
/// 0's would emit `ceil(n/C)` of `n` sessions with a stride-`C` id gap).
///
/// The slices are re-INTERLEAVED round-robin (cell 0 row 0, cell 1 row 0, …, cell 0 row 1,
/// …) rather than sorted by `session_id`: interleaving is the exact inverse of the
/// `position % cell_count == cell_id` partition, so it reproduces the single-process
/// document's dataset order for ANY id scheme. Sorting only works when ids are ordinal
/// (`session_000012`); real datasets carry random UUID session ids, for which the sort
/// order is arbitrary and need not match the single-cell document (GenAI-Perf compat,
/// always-on per `rust_wire`). `cell_dirs` is indexed by cell id at the call site, which is
/// what makes the interleave well-defined. A no-op when no cell wrote the file (e.g. inputs
/// export disabled). Sources are owned by the controller's `ScratchTreeGuard`, so — like
/// [`concatenate_cell_artifacts`] — this never deletes them.
pub(crate) fn merge_cell_inputs_json(
    cell_dirs: &[PathBuf],
    artifact_dir: &Path,
    artifacts: &ArtifactSpec,
) -> Result<()> {
    let Some(relative) = &artifacts.inputs_path else {
        return Ok(());
    };
    // Keep each cell's rows in their own list, indexed by cell id, so the interleave below
    // can walk them in lockstep. A cell that wrote no file contributes an empty list, which
    // the interleave skips without shifting the other cells' positions.
    let mut per_cell: Vec<Vec<serde_json::Value>> = Vec::with_capacity(cell_dirs.len());
    let mut any_source = false;
    for source in cell_dirs.iter().map(|dir| dir.join(relative)) {
        if !source.exists() {
            per_cell.push(Vec::new());
            continue;
        }
        any_source = true;
        let bytes = std::fs::read(&source)
            .with_context(|| format!("reading cell inputs.json {}", source.display()))?;
        let doc: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing cell inputs.json {}", source.display()))?;
        per_cell.push(match doc.get("data").and_then(serde_json::Value::as_array) {
            Some(rows) => rows.clone(),
            None => Vec::new(),
        });
    }
    if !any_source {
        // No cell produced an inputs.json (e.g. inputs export disabled); nothing to write.
        return Ok(());
    }
    // Undo the round-robin partition: dataset position `p` went to cell `p % cell_count` as
    // that cell's row `p / cell_count`, so reading row `r` from every cell in cell-id order
    // walks positions `r*count .. r*count+count-1` — the original dataset order.
    let longest = per_cell.iter().map(Vec::len).max().unwrap_or(0);
    let mut sessions: Vec<serde_json::Value> =
        Vec::with_capacity(per_cell.iter().map(Vec::len).sum());
    for row in 0..longest {
        for cell in &per_cell {
            if let Some(session) = cell.get(row) {
                sessions.push(session.clone());
            }
        }
    }
    let final_path = artifact_dir.join(relative);
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating inputs.json directory {}", parent.display()))?;
    }
    let document = serde_json::json!({ "data": sessions });
    let file = std::fs::File::create(&final_path)
        .with_context(|| format!("creating merged inputs.json {}", final_path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &document)
        .with_context(|| format!("serializing merged inputs.json {}", final_path.display()))?;
    writer
        .write_all(b"\n")
        .with_context(|| format!("writing merged inputs.json {}", final_path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing merged inputs.json {}", final_path.display()))
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

    fn all_artifacts() -> ArtifactSpec {
        ArtifactSpec {
            records_path: Some(PathBuf::from("profile_export.jsonl")),
            raw_path: Some(PathBuf::from("profile_export_raw.jsonl")),
            records_csv_path: Some(PathBuf::from("profile_export_records.csv")),
            records_parquet_path: Some(PathBuf::from("profile_export.parquet")),
            outputs_path: Some(PathBuf::from("outputs.json")),
            inputs_path: None,
            trace: false,
            dataset_analysis_path: None,
            ..Default::default()
        }
    }

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

    fn line_set(path: &Path) -> BTreeMap<String, usize> {
        let mut set = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                *set.entry(line.to_string()).or_insert(0) += 1;
            }
        }
        set
    }

    #[test]
    fn per_shard_concat_matches_batch_over_union() {
        let config = MetricsConfig::default();
        let records = union_records();
        let artifacts = all_artifacts();
        let workers = 3usize;

        let shard_dir_root = tempfile::tempdir().unwrap();
        let slices: [&[CapturedRecord]; 3] = [&records[0..3], &[], &records[3..7]];
        for (id, slice) in slices.iter().enumerate() {
            drive_shard_lane(shard_dir_root.path(), id, slice, &artifacts, &config);
        }
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

        for id in 0..workers {
            assert!(
                !shard_dir(shard_dir_root.path(), id).exists(),
                "shard temp dir {id} removed"
            );
        }

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
                line_set(&shard_dir_root.path().join(name)),
                line_set(&batch_dir.path().join(name)),
                "{name} line set must equal the batch writer over the union"
            );
        }

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

    #[test]
    fn per_cell_concat_matches_batch_over_union() {
        let config = MetricsConfig::default();
        let records = union_records();
        let artifacts = all_artifacts();
        let cell_count = 3usize;

        let temp_root = tempfile::tempdir().unwrap();
        let cell_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|id| temp_root.path().join(format!("cell-{id}")))
            .collect();
        let slices: [&[CapturedRecord]; 3] = [&records[0..3], &[], &records[3..7]];
        for (dir, slice) in cell_dirs.iter().zip(slices) {
            drive_cell_lane(dir, slice, &artifacts, &config);
        }

        let run_dir = tempfile::tempdir().unwrap();
        concatenate_cell_artifacts(&cell_dirs, run_dir.path(), &artifacts).unwrap();

        for dir in &cell_dirs {
            assert!(
                dir.exists(),
                "cell dir {} must survive concat",
                dir.display()
            );
        }

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

        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            let bytes = std::fs::read(dir.path().join(name)).unwrap();
            assert!(bytes.is_empty(), "{name} is an empty file");
        }
        assert!(!dir.path().join("profile_export_records.csv").exists());
        #[cfg(feature = "parquet")]
        assert!(!dir.path().join("profile_export.parquet").exists());
        let outputs: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("outputs.json")).unwrap())
                .unwrap();
        assert_eq!(outputs["data"].as_array().unwrap().len(), 0);
    }

    /// Each cell's file covers only its round-robin slice (`position % C == cell_id`),
    /// so the controller must union them and restore `session_id` order — copying one
    /// cell's file would emit a stride-`C` subset of the sessions.
    #[test]
    fn merged_inputs_json_unions_cell_slices_in_session_order() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = ArtifactSpec {
            inputs_path: Some(PathBuf::from("inputs.json")),
            ..ArtifactSpec::default()
        };
        let cell_count = 3usize;
        let total = 7usize;
        let cell_dirs: Vec<PathBuf> = (0..cell_count)
            .map(|cell_id| dir.path().join(format!("cell-{cell_id}")))
            .collect();
        for (cell_id, cell_dir) in cell_dirs.iter().enumerate() {
            std::fs::create_dir_all(cell_dir).unwrap();
            let rows: Vec<serde_json::Value> = (0..total)
                .filter(|position| position % cell_count == cell_id)
                .map(|position| {
                    serde_json::json!({
                        "session_id": format!("session_{position:06}"),
                        "payloads": [{"prompt": format!("p{position}")}],
                    })
                })
                .collect();
            std::fs::write(
                cell_dir.join("inputs.json"),
                serde_json::to_vec(&serde_json::json!({ "data": rows })).unwrap(),
            )
            .unwrap();
        }

        merge_cell_inputs_json(&cell_dirs, dir.path(), &artifacts).unwrap();

        let merged: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("inputs.json")).unwrap()).unwrap();
        let ids: Vec<&str> = merged["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["session_id"].as_str().unwrap())
            .collect();
        let expected: Vec<String> = (0..total)
            .map(|position| format!("session_{position:06}"))
            .collect();
        assert_eq!(ids, expected, "union of the slices, ordered by session_id");
    }

    #[test]
    fn merged_inputs_json_is_a_noop_without_cell_files() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = ArtifactSpec {
            inputs_path: Some(PathBuf::from("inputs.json")),
            ..ArtifactSpec::default()
        };
        let cell_dirs = vec![dir.path().join("cell-0"), dir.path().join("cell-1")];
        merge_cell_inputs_json(&cell_dirs, dir.path(), &artifacts).unwrap();
        assert!(!dir.path().join("inputs.json").exists());
    }
}
