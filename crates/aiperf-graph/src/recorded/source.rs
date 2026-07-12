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
    use super::*;

    #[test]
    fn segmented_names_sort_numerically_across_width_rollover() {
        let mut paths = [
            PathBuf::from("trace.1000000.jsonl.gz"),
            PathBuf::from("trace.999999.jsonl.gz"),
        ];
        paths.sort_by_key(|path| segment_sort_key(path));
        assert_eq!(paths[0], PathBuf::from("trace.999999.jsonl.gz"));
    }
}
