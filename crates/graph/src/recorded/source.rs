// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Format-owned source acquisition for recorded graph inputs.

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};

use aiperf_dataset::{DatasetSource, LoadConfig, load_raw_rows};
use flate2::read::MultiGzDecoder;
use serde_json::Value;

use super::RecordedTraceError;

pub(crate) async fn load_weka_documents(
    config: &LoadConfig,
) -> Result<Vec<Value>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) if path.is_dir() => {
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
                    "{}: WEKA trace directory contains no .json files",
                    path.display()
                )));
            }
            paths
                .iter()
                .map(|candidate| {
                    parse_whole_json(
                        &fs::read(candidate).map_err(|error| source_error(candidate, error))?,
                        &candidate.display().to_string(),
                    )
                })
                .collect()
        }
        DatasetSource::Path(path) => {
            let bytes = fs::read(path).map_err(|error| source_error(path, error))?;
            parse_whole_json(&bytes, &path.display().to_string()).map(|value| vec![value])
        }
        DatasetSource::Bytes(bytes) => {
            parse_whole_json(bytes, "in-memory WEKA trace").map(|value| vec![value])
        }
        DatasetSource::Inline(value) => values_from_inline(value),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| rows.into_iter().map(|row| row.value).collect())
            .map_err(Into::into),
    }
}

pub(crate) async fn load_dynamo_documents(
    config: &LoadConfig,
) -> Result<Vec<Value>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) => {
            let paths = discover_dynamo_segments(path)?;
            let mut values = Vec::new();
            for segment in paths {
                values.extend(read_json_lines(&segment)?);
            }
            Ok(values)
        }
        DatasetSource::Bytes(bytes) => parse_json_lines(bytes, "in-memory Dynamo trace"),
        DatasetSource::Inline(value) => Ok(match value {
            Value::Array(values) => values.clone(),
            value => vec![value.clone()],
        }),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => load_raw_rows(config)
            .await
            .map(|rows| rows.into_iter().map(|row| row.value).collect())
            .map_err(Into::into),
    }
}

pub(crate) async fn load_aiperf_documents(
    config: &LoadConfig,
) -> Result<Vec<Value>, RecordedTraceError> {
    match &config.source {
        DatasetSource::Path(path) if path.is_dir() => {
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
                    "{}: aiperf.trace.v1 directory contains no .json files",
                    path.display()
                )));
            }
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

fn parse_whole_json(bytes: &[u8], label: &str) -> Result<Value, RecordedTraceError> {
    serde_json::from_slice(bytes)
        .map_err(|error| RecordedTraceError(format!("{label}: invalid JSON: {error}")))
}

fn parse_json_lines(bytes: &[u8], label: &str) -> Result<Vec<Value>, RecordedTraceError> {
    parse_json_lines_from(Cursor::new(bytes), label)
}

fn parse_json_lines_from(
    mut reader: impl BufRead,
    label: &str,
) -> Result<Vec<Value>, RecordedTraceError> {
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
        let value: Value = serde_json::from_str(line).map_err(|error| {
            RecordedTraceError(format!("{label}: invalid JSON line {index}: {error}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn read_json_lines(path: &Path) -> Result<Vec<Value>, RecordedTraceError> {
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

fn discover_dynamo_segments(path: &Path) -> Result<Vec<PathBuf>, RecordedTraceError> {
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

        let values = read_json_lines(&path).unwrap();
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
    fn corrupt_gzip_fails_with_source_context() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl.gz");
        fs::write(&path, b"not gzip").unwrap();
        let error = read_json_lines(&path).unwrap_err().to_string();
        assert!(error.contains("trace.jsonl.gz"));
        assert!(error.contains("corrupt") || error.contains("unreadable"));
    }
}
