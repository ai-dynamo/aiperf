// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Format-owned source acquisition for recorded graph inputs.

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};

use crate::dataset::{DatasetSource, LoadConfig, load_raw_rows};
use flate2::read::MultiGzDecoder;
use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::value::RawValue;

use super::RecordedTraceError;

/// Load WEKA source documents as untouched JSON text.
///
/// The WEKA/Dynamo formats carry cache-block hash ids that exceed `u64::MAX`, so
/// the documents are handed downstream as [`RawValue`]s: the format schemas
/// re-parse the enclosing object into a [`Value`] for ordinary field logic while
/// pulling the wide hash tokens straight from the raw text (see
/// `scalar::hash_i128_from_raw_text`). Decoding to `Value` here would round those
/// hashes through `f64` and lose their low digits without the globally
/// side-effecting `arbitrary_precision` feature.
pub(crate) async fn load_weka_documents(
    config: &LoadConfig,
) -> Result<Vec<Box<RawValue>>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) if path.is_dir() => {
            let paths = json_documents_in_dir(path, "WEKA trace")?;
            paths
                .iter()
                .map(|candidate| {
                    parse_whole_json_raw(
                        &fs::read(candidate).map_err(|error| source_error(candidate, error))?,
                        &candidate.display().to_string(),
                    )
                })
                .collect()
        }
        DatasetSource::Path(path) => {
            // A WEKA file is either one trace as a whole JSON document or a JSONL
            // corpus with one
            // trace per line (the published `semianalysisai/cc-traces-weka-*`
            // `traces.jsonl` shape). A trailing-characters failure switches to
            // line-delimited parsing.
            let bytes = fs::read(path).map_err(|error| source_error(path, error))?;
            match parse_whole_json_raw(&bytes, &path.display().to_string()) {
                Ok(value) => Ok(vec![value]),
                Err(_) => read_json_lines_raw(path),
            }
        }
        DatasetSource::Bytes(bytes) => {
            parse_whole_json_raw(bytes, "in-memory WEKA trace").map(|value| vec![value])
        }
        DatasetSource::Inline(value) => Ok(raws_from_inline(value)),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| {
                rows.into_iter()
                    .map(|row| raw_from_value(&row.value))
                    .collect()
            })
            .map_err(Into::into),
    }
}

/// Load Dynamo request-trace documents as untouched JSON text. See
/// [`load_weka_documents`] for why the wide-hash formats stay as [`RawValue`]s.
pub(crate) async fn load_dynamo_documents(
    config: &LoadConfig,
) -> Result<Vec<Box<RawValue>>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) => {
            let paths = discover_dynamo_segments(path)?;
            let mut values = Vec::new();
            for segment in paths {
                values.extend(read_json_lines_raw(&segment)?);
            }
            Ok(values)
        }
        DatasetSource::Bytes(bytes) => parse_json_lines_raw(bytes, "in-memory Dynamo trace"),
        DatasetSource::Inline(value) => Ok(raws_from_inline(value)),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| {
                rows.into_iter()
                    .map(|row| raw_from_value(&row.value))
                    .collect()
            })
            .map_err(Into::into),
    }
}

pub(crate) async fn load_aiperf_documents(
    config: &LoadConfig,
) -> Result<Vec<Value>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) if path.is_dir() => {
            let paths = json_documents_in_dir(path, "aiperf.trace.v1")?;
            let mut values = Vec::new();
            for candidate in &paths {
                let bytes = fs::read(candidate).map_err(|error| source_error(candidate, error))?;
                values.extend(parse_aiperf_documents(
                    &bytes,
                    &candidate.display().to_string(),
                )?);
            }
            Ok(values)
        }
        DatasetSource::Path(path) => {
            let bytes = fs::read(path).map_err(|error| source_error(path, error))?;
            parse_aiperf_documents(&bytes, &path.display().to_string())
        }
        DatasetSource::Bytes(bytes) => parse_aiperf_documents(bytes, "in-memory aiperf trace"),
        DatasetSource::Inline(value) => values_from_inline(value),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| rows.into_iter().map(|row| row.value).collect())
            .map_err(Into::into),
    }
}

/// One session object, an array of them, or JSONL (one compact object per line).
fn parse_aiperf_documents(bytes: &[u8], label: &str) -> Result<Vec<Value>, RecordedTraceError> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Array(values)) => Ok(values),
        Ok(value) => Ok(vec![value]),
        Err(_) => parse_json_lines(bytes, label),
    }
}

fn values_from_inline(value: &Value) -> Result<Vec<Value>, RecordedTraceError> {
    Ok(match value {
        Value::Array(values) => values.clone(),
        value => vec![value.clone()],
    })
}

/// Fan an already-decoded inline [`Value`] out into per-document raw tokens.
///
/// Inline sources are used by tests and authored configs, never by the wide-hash
/// product path, so serializing the `Value` back to raw text is lossless for
/// every hash that already fit through `Value` decoding.
fn raws_from_inline(value: &Value) -> Vec<Box<RawValue>> {
    match value {
        Value::Array(values) => values.iter().map(raw_from_value).collect(),
        value => vec![raw_from_value(value)],
    }
}

/// Re-serialize an in-memory [`Value`] into an opaque raw JSON token.
fn raw_from_value(value: &Value) -> Box<RawValue> {
    serde_json::value::to_raw_value(value)
        .expect("serializing an in-memory Value into raw JSON cannot fail")
}

fn parse_whole_json_raw(bytes: &[u8], label: &str) -> Result<Box<RawValue>, RecordedTraceError> {
    serde_json::from_slice(bytes)
        .map_err(|error| RecordedTraceError(format!("{label}: invalid JSON: {error}")))
}

fn parse_json_lines(bytes: &[u8], label: &str) -> Result<Vec<Value>, RecordedTraceError> {
    parse_json_lines_from(Cursor::new(bytes), label)
}

fn parse_json_lines_raw(
    bytes: &[u8],
    label: &str,
) -> Result<Vec<Box<RawValue>>, RecordedTraceError> {
    parse_json_lines_from(Cursor::new(bytes), label)
}

/// JSONL line reader deserializing each non-blank line into `T`.
///
/// The aiperf-trace path uses `T = Value` (hashes fit in `u64`); the Dynamo
/// schema uses `T = Box<RawValue>` to capture each record as untouched raw JSON
/// text so wide `input_sequence_hashes` survive before any `f64` coercion.
fn parse_json_lines_from<T: DeserializeOwned>(
    mut reader: impl BufRead,
    label: &str,
) -> Result<Vec<T>, RecordedTraceError> {
    let mut values = Vec::new();
    let mut buffer = Vec::new();
    let mut index = 0_usize;
    loop {
        buffer.clear();
        let read = reader.read_until(b'\n', &mut buffer).map_err(|error| {
            RecordedTraceError(format!(
                "{label}: truncated, corrupt, or unreadable JSONL stream: {error}"
            ))
        })?;
        if read == 0 {
            break;
        }
        index += 1;
        let line = std::str::from_utf8(&buffer).map_err(|error| {
            RecordedTraceError(format!("{label}: not valid UTF-8 JSONL: {error}"))
        })?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: T = serde_json::from_str(line).map_err(|error| {
            RecordedTraceError(format!("{label}: invalid JSON line {index}: {error}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn read_json_lines_raw(path: &Path) -> Result<Vec<Box<RawValue>>, RecordedTraceError> {
    let file = File::open(path).map_err(|error| source_error(path, error))?;
    let label = path.display().to_string();
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gz"))
    {
        return parse_json_lines_from(BufReader::new(MultiGzDecoder::new(file)), &label);
    }
    parse_json_lines_from(BufReader::new(file), &label)
}

/// The ordered `.json` files the WEKA / `aiperf.trace.v1` directory loaders read
/// from `path`, sorted lexicographically (the loaders' own `paths.sort()` order),
/// filtered to case-insensitive `.json` files (non-recursive, matching
/// [`fs::read_dir`]). `kind` names the format for the empty-directory error. This
/// is the single source of truth for the directory read set: both the loaders
/// above and the cross-host shipping enumerator ([`enumerate_recorded_trace_files`])
/// call it, so the shipped file set can never diverge from the read set.
fn json_documents_in_dir(path: &Path, kind: &str) -> Result<Vec<PathBuf>, RecordedTraceError> {
    let mut paths = fs::read_dir(path)
        .map_err(|error| source_error(path, error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate.is_file()
                && candidate
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(RecordedTraceError(format!(
            "{}: {kind} directory contains no .json files",
            path.display()
        )));
    }
    Ok(paths)
}

/// The on-disk layout a recorded-trace `path` resolves to. Governs how the
/// cross-host cell rewrites `datasets/0.path` after reconstructing the shipped
/// files: a [`File`](Self::File) or [`SegmentedPrefix`](Self::SegmentedPrefix)
/// points `path` back at a single (re-globbed) stem, while a
/// [`Directory`](Self::Directory) points it at the reconstructed directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordedTracePathKind {
    /// A single recorded-trace file.
    File,
    /// A directory the loader enumerates (WEKA/aiperf `.json`; Dynamo
    /// `.jsonl`/`.jsonl.gz`).
    Directory,
    /// A segmented-prefix stem (`parent/prefix`) whose `prefix.NNNNNN.jsonl.gz`
    /// shards live beside it (Dynamo only).
    SegmentedPrefix,
}

/// The exact ordered on-disk file set the recorded-trace loader for `format`
/// reads for a `Path` source, plus the layout kind and the original path's file
/// name. This is the shipping-side mirror of the loaders in this module — it
/// reuses their own enumeration ([`json_documents_in_dir`] /
/// [`discover_dynamo_segments`]), so the set the controller ships is byte-for-byte
/// the set a 1-cell run reads (no over/under-ship).
///
/// `format` must be one of the graph recorded formats:
/// - `weka_trace` / `aiperf_trace`: a single file, or a directory of `.json`
///   files (see [`load_weka_documents`] / [`load_aiperf_documents`]);
/// - `dynamo_trace`: a single file, a directory of `.jsonl`/`.jsonl.gz`, or a
///   segmented-prefix stem (see [`discover_dynamo_segments`]);
/// - `dag_jsonl`: a single file ONLY — its loader (`load_raw_rows`) reads one file
///   via `std::fs::read`, so a directory/prefix is unreadable and rejected here.
///
/// Fails closed on a missing path, an empty directory, an unmatched prefix, or an
/// unsupported format — the same errors the loader would raise, surfaced before
/// the run launches cells.
pub fn enumerate_recorded_trace_files(
    format: &str,
    path: &Path,
) -> Result<(RecordedTracePathKind, String, Vec<PathBuf>), RecordedTraceError> {
    let base_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            RecordedTraceError(format!("{}: trace path has no file name", path.display()))
        })?
        .to_owned();
    match format {
        "weka_trace" | "aiperf_trace" => {
            let kind = if format == "weka_trace" {
                "WEKA trace"
            } else {
                "aiperf.trace.v1"
            };
            if path.is_dir() {
                Ok((
                    RecordedTracePathKind::Directory,
                    base_name,
                    json_documents_in_dir(path, kind)?,
                ))
            } else if path.is_file() {
                Ok((
                    RecordedTracePathKind::File,
                    base_name,
                    vec![path.to_path_buf()],
                ))
            } else {
                Err(RecordedTraceError(format!(
                    "{}: {kind} path is not a file or directory",
                    path.display()
                )))
            }
        }
        "dynamo_trace" => {
            let kind = if path.is_file() {
                RecordedTracePathKind::File
            } else if path.is_dir() {
                RecordedTracePathKind::Directory
            } else {
                RecordedTracePathKind::SegmentedPrefix
            };
            // `discover_dynamo_segments` owns all three shapes and fails closed on
            // a missing path / empty dir / unmatched prefix.
            Ok((kind, base_name, discover_dynamo_segments(path)?))
        }
        "dag_jsonl" => {
            if path.is_file() {
                Ok((
                    RecordedTracePathKind::File,
                    base_name,
                    vec![path.to_path_buf()],
                ))
            } else {
                Err(RecordedTraceError(format!(
                    "{}: dag_jsonl reads a single file; a directory or segmented-prefix path is \
                     not supported",
                    path.display()
                )))
            }
        }
        other => Err(RecordedTraceError(format!(
            "{other:?}: not a directory/prefix-capable recorded graph trace format"
        ))),
    }
}

pub(crate) fn discover_dynamo_segments(path: &Path) -> Result<Vec<PathBuf>, RecordedTraceError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        let mut paths = fs::read_dir(path)
            .map_err(|error| source_error(path, error))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|candidate| {
                let name = candidate
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default();
                candidate.is_file() && (name.ends_with(".jsonl") || name.ends_with(".jsonl.gz"))
            })
            .collect::<Vec<_>>();
        paths.sort_by_key(|candidate| segment_sort_key(candidate));
        if paths.is_empty() {
            return Err(RecordedTraceError(format!(
                "{}: no .jsonl or .jsonl.gz files in directory",
                path.display()
            )));
        }
        return Ok(paths);
    }

    let parent = path
        .parent()
        .filter(|parent| parent.is_dir())
        .ok_or_else(|| {
            RecordedTraceError(format!(
                "{}: not a file or directory, and parent is not a directory",
                path.display()
            ))
        })?;
    let mut prefix = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_string();
    for suffix in [".jsonl.gz", ".jsonl"] {
        if let Some(stripped) = prefix.strip_suffix(suffix) {
            prefix = stripped.to_string();
            break;
        }
    }
    let mut paths = fs::read_dir(parent)
        .map_err(|error| source_error(parent, error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate
                .file_name()
                .and_then(|value| value.to_str())
                .and_then(parse_segment_name)
                .is_some_and(|(candidate_prefix, _)| candidate_prefix == prefix)
        })
        .collect::<Vec<_>>();
    paths.sort_by_key(|candidate| {
        candidate
            .file_name()
            .and_then(|value| value.to_str())
            .and_then(parse_segment_name)
            .map(|(_, index)| index)
            .unwrap_or(u64::MAX)
    });
    if paths.is_empty() {
        return Err(RecordedTraceError(format!(
            "{}: no matching segments found ({prefix}.*.jsonl.gz)",
            path.display()
        )));
    }
    Ok(paths)
}

fn segment_sort_key(path: &Path) -> (String, i128) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    parse_segment_name(name)
        .map(|(prefix, index)| (prefix.to_string(), i128::from(index)))
        .unwrap_or_else(|| (name.to_string(), -1))
}

fn parse_segment_name(name: &str) -> Option<(&str, u64)> {
    let stem = name.strip_suffix(".jsonl.gz")?;
    let (prefix, index) = stem.rsplit_once('.')?;
    if prefix.is_empty() || index.len() < 6 || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((prefix, index.parse().ok()?))
}

fn source_error(path: &Path, error: std::io::Error) -> RecordedTraceError {
    RecordedTraceError(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use serde_json::json;

    use super::*;

    fn gzip_member(lines: &[Value]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        for line in lines {
            encoder
                .write_all(&serde_json::to_vec(line).unwrap())
                .unwrap();
            encoder.write_all(b"\n").unwrap();
        }
        encoder.finish().unwrap()
    }

    #[test]
    fn segmented_names_sort_numerically_across_width_rollover() {
        let mut paths = [
            PathBuf::from("trace.1000000.jsonl.gz"),
            PathBuf::from("trace.999999.jsonl.gz"),
        ];
        paths.sort_by_key(|path| segment_sort_key(path));
        assert_eq!(paths[0], PathBuf::from("trace.999999.jsonl.gz"));
    }

    #[test]
    fn explicit_gzip_reads_every_concatenated_member_and_sink_envelope() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.JSONL.GZ");
        let first = gzip_member(&[
            json!({"verification": "trace-s3-uploader"}),
            json!({"timestamp": 7, "event": {"schema": "dynamo.request.trace.v1", "event_type": "request_end", "event_time_unix_ms": 1}}),
        ]);
        let second = gzip_member(&[json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 2
        })]);
        let mut bytes = first;
        bytes.extend(second);
        fs::write(&path, bytes).unwrap();

        let raws = read_json_lines_raw(&path).unwrap();
        let values = raws
            .iter()
            .map(|raw| serde_json::from_str::<Value>(raw.get()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0]["verification"], "trace-s3-uploader");
        assert_eq!(values[1]["event"]["event_time_unix_ms"], 1);
        assert_eq!(values[2]["event_time_unix_ms"], 2);
    }

    #[test]
    fn directory_and_nonexistent_prefix_follow_python_discovery_order() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "trace.1000000.jsonl.gz",
            "trace.999999.jsonl.gz",
            "other.jsonl",
            "ignored.JSONL.GZ",
        ] {
            fs::write(directory.path().join(name), b"\n").unwrap();
        }
        let discovered = discover_dynamo_segments(directory.path()).unwrap();
        let names = discovered
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "other.jsonl",
                "trace.999999.jsonl.gz",
                "trace.1000000.jsonl.gz"
            ]
        );

        let prefix = directory.path().join("trace.jsonl.gz");
        let discovered = discover_dynamo_segments(&prefix).unwrap();
        let names = discovered
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, ["trace.999999.jsonl.gz", "trace.1000000.jsonl.gz"]);
        assert!(parse_segment_name(".000000.jsonl.gz").is_none());
    }

    #[test]
    fn enumerate_single_file_returns_just_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("trace.jsonl");
        fs::write(&file, b"{}\n").unwrap();
        for format in ["weka_trace", "aiperf_trace", "dynamo_trace", "dag_jsonl"] {
            let (kind, base, files) = enumerate_recorded_trace_files(format, &file).unwrap();
            assert_eq!(kind, RecordedTracePathKind::File, "{format}");
            assert_eq!(base, "trace.jsonl", "{format}");
            assert_eq!(files, vec![file.clone()], "{format}");
        }
    }

    #[test]
    fn enumerate_weka_directory_matches_loader_json_read_set() {
        let dir = tempfile::tempdir().unwrap();
        // The loader reads only .json (case-insensitive), non-recursive, sorted.
        for name in ["b.json", "a.json", "c.JSON"] {
            fs::write(dir.path().join(name), b"{}").unwrap();
        }
        fs::write(dir.path().join("ignored.txt"), b"x").unwrap();
        let nested = dir.path().join("sub");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("deep.json"), b"{}").unwrap();

        let (kind, _base, files) =
            enumerate_recorded_trace_files("weka_trace", dir.path()).unwrap();
        assert_eq!(kind, RecordedTracePathKind::Directory);
        let names = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        // Sorted lexicographically; the nested dir and non-json are excluded (the
        // loader is non-recursive and .json-only) — exactly the loader's read set.
        assert_eq!(names, ["a.json", "b.json", "c.JSON"]);
    }

    #[test]
    fn enumerate_dynamo_directory_and_prefix_match_discovery_order() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "trace.1000000.jsonl.gz",
            "trace.999999.jsonl.gz",
            "other.jsonl",
        ] {
            fs::write(dir.path().join(name), b"\n").unwrap();
        }
        let (dir_kind, _b, dir_files) =
            enumerate_recorded_trace_files("dynamo_trace", dir.path()).unwrap();
        assert_eq!(dir_kind, RecordedTracePathKind::Directory);
        assert_eq!(
            dir_files,
            discover_dynamo_segments(dir.path()).unwrap(),
            "directory enumeration must equal the loader's discovery"
        );

        let prefix = dir.path().join("trace.jsonl.gz");
        let (prefix_kind, base, prefix_files) =
            enumerate_recorded_trace_files("dynamo_trace", &prefix).unwrap();
        assert_eq!(prefix_kind, RecordedTracePathKind::SegmentedPrefix);
        assert_eq!(base, "trace.jsonl.gz");
        let names = prefix_files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(names, ["trace.999999.jsonl.gz", "trace.1000000.jsonl.gz"]);
    }

    #[test]
    fn enumerate_rejects_dag_jsonl_directory_and_missing_paths() {
        let dir = tempfile::tempdir().unwrap();
        // dag_jsonl cannot read a directory (single-file loader): must fail closed.
        assert!(enumerate_recorded_trace_files("dag_jsonl", dir.path()).is_err());
        // A missing path fails closed for every format.
        let missing = dir.path().join("nope");
        for format in ["weka_trace", "aiperf_trace", "dynamo_trace", "dag_jsonl"] {
            assert!(
                enumerate_recorded_trace_files(format, &missing).is_err(),
                "{format} missing path must fail closed"
            );
        }
        // An empty WEKA directory fails closed (no .json files).
        assert!(enumerate_recorded_trace_files("weka_trace", dir.path()).is_err());
    }

    #[test]
    fn corrupt_gzip_fails_with_source_context() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl.gz");
        fs::write(&path, b"not gzip").unwrap();
        let error = read_json_lines_raw(&path).unwrap_err().to_string();
        assert!(error.contains("trace.jsonl.gz"));
        assert!(error.contains("corrupt") || error.contains("unreadable"));
    }
}
