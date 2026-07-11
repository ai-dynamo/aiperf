// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset-source extension seam and JSON implementations.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{AccuracyError, DatasetSplit};

/// Provides decoded dataset rows to an accuracy benchmark.
///
/// Filesystem, cached-Hugging-Face, embedded-fixture, and future object-store
/// implementations all satisfy this same interface.
pub trait DatasetSource {
    /// Load the decoded rows for one split.
    fn load_rows(&self, split: DatasetSplit) -> Result<Vec<Value>, AccuracyError>;
}

/// A pair of local JSON/JSONL split files.
#[derive(Debug, Clone)]
pub struct JsonDatasetSource {
    paths: HashMap<DatasetSplit, PathBuf>,
}

impl JsonDatasetSource {
    /// Builds a source from explicit validation and test paths.
    pub fn new(validation_path: impl Into<PathBuf>, test_path: impl Into<PathBuf>) -> Self {
        Self {
            paths: HashMap::from([
                (DatasetSplit::Validation, validation_path.into()),
                (DatasetSplit::Test, test_path.into()),
            ]),
        }
    }

    /// Builds a source from `<split>.json` files below `directory`.
    pub fn from_directory(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        Self {
            paths: [
                DatasetSplit::Train,
                DatasetSplit::Dev,
                DatasetSplit::Validation,
                DatasetSplit::Test,
            ]
            .into_iter()
            .map(|split| (split, directory.join(format!("{}.json", split.as_str()))))
            .collect(),
        }
    }

    /// Overrides or adds the path for one split.
    pub fn with_split(mut self, split: DatasetSplit, path: impl Into<PathBuf>) -> Self {
        self.paths.insert(split, path.into());
        self
    }

    fn path_for(&self, split: DatasetSplit) -> Option<&Path> {
        self.paths.get(&split).map(PathBuf::as_path)
    }
}

impl DatasetSource for JsonDatasetSource {
    fn load_rows(&self, split: DatasetSplit) -> Result<Vec<Value>, AccuracyError> {
        let path = self
            .path_for(split)
            .ok_or_else(|| AccuracyError::ReadDataset {
                path: PathBuf::from(format!("<unconfigured:{}>", split.as_str())),
                message: "dataset source has no path for this split".to_string(),
            })?;
        let text = fs::read_to_string(path).map_err(|error| AccuracyError::ReadDataset {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        parse_rows(&text, &path.display().to_string())
    }
}

/// In-memory rows used by tests and embedding applications.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDatasetSource {
    rows: HashMap<DatasetSplit, Vec<Value>>,
}

impl InMemoryDatasetSource {
    /// Builds a source from validation and test rows.
    pub fn new(validation: Vec<Value>, test: Vec<Value>) -> Self {
        Self {
            rows: HashMap::from([
                (DatasetSplit::Validation, validation),
                (DatasetSplit::Test, test),
            ]),
        }
    }

    /// Builds a source from arbitrary upstream splits.
    pub fn from_splits(rows: impl IntoIterator<Item = (DatasetSplit, Vec<Value>)>) -> Self {
        Self {
            rows: rows.into_iter().collect(),
        }
    }

    /// Adds or replaces one split.
    pub fn with_split(mut self, split: DatasetSplit, rows: Vec<Value>) -> Self {
        self.rows.insert(split, rows);
        self
    }
}

impl DatasetSource for InMemoryDatasetSource {
    fn load_rows(&self, split: DatasetSplit) -> Result<Vec<Value>, AccuracyError> {
        Ok(self.rows.get(&split).cloned().unwrap_or_default())
    }
}

fn parse_rows(text: &str, source: &str) -> Result<Vec<Value>, AccuracyError> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return rows_from_value(value, source);
    }

    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value =
            serde_json::from_str::<Value>(line).map_err(|error| AccuracyError::ParseDataset {
                source: source.to_string(),
                message: format!("JSON Lines row {}: {error}", line_index + 1),
            })?;
        rows.push(value);
    }
    if rows.is_empty() {
        return Err(AccuracyError::ParseDataset {
            source: source.to_string(),
            message: "expected a JSON array, a Hugging Face rows envelope, or JSON Lines"
                .to_string(),
        });
    }
    Ok(rows)
}

fn rows_from_value(value: Value, source: &str) -> Result<Vec<Value>, AccuracyError> {
    match value {
        Value::Array(rows) => Ok(rows),
        Value::Object(mut object) => {
            let Some(Value::Array(rows)) = object.remove("rows") else {
                return Err(AccuracyError::ParseDataset {
                    source: source.to_string(),
                    message: "top-level JSON object has no rows array".to_string(),
                });
            };
            rows.into_iter()
                .map(|row| match row {
                    Value::Object(mut envelope) => {
                        envelope
                            .remove("row")
                            .ok_or_else(|| AccuracyError::ParseDataset {
                                source: source.to_string(),
                                message: "Hugging Face row envelope has no row field".to_string(),
                            })
                    }
                    _ => Err(AccuracyError::ParseDataset {
                        source: source.to_string(),
                        message: "Hugging Face rows entry is not an object".to_string(),
                    }),
                })
                .collect()
        }
        _ => Err(AccuracyError::ParseDataset {
            source: source.to_string(),
            message: "top-level JSON must be an array or rows envelope".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_rows;

    #[test]
    fn parses_array_jsonl_and_hugging_face_envelope() {
        assert_eq!(parse_rows("[{\"x\":1}]", "array").unwrap().len(), 1);
        assert_eq!(
            parse_rows("{\"x\":1}\n{\"x\":2}\n", "jsonl").unwrap().len(),
            2
        );
        assert_eq!(
            parse_rows("{\"rows\":[{\"row_idx\":0,\"row\":{\"x\":1}}]}", "hf")
                .unwrap()
                .len(),
            1
        );
    }
}
