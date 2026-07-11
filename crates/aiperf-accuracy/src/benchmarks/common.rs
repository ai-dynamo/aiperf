// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared row validation and typed-problem construction for benchmark ports.

use std::collections::BTreeMap;

use aiperf_metrics::{CorrelationId, TaskId};
use serde_json::Value;

use crate::{AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage, GenerationConfig};

pub(super) fn required_string(
    row: &Value,
    field: &str,
    row_index: usize,
) -> Result<String, AccuracyError> {
    row.get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            invalid_row(
                row_index,
                format!("field {field:?} must be a non-empty string"),
            )
        })
}

pub(super) fn optional_string(row: &Value, field: &str) -> String {
    row.get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn scalar_string(
    row: &Value,
    field: &str,
    row_index: usize,
) -> Result<String, AccuracyError> {
    match row.get(field) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Number(value)) => Ok(value.to_string()),
        Some(Value::Bool(value)) => Ok(value.to_string()),
        _ => Err(invalid_row(
            row_index,
            format!("field {field:?} must be a string, number, or boolean"),
        )),
    }
}

pub(super) fn string_array(
    row: &Value,
    field: &str,
    row_index: usize,
) -> Result<Vec<String>, AccuracyError> {
    row.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_row(row_index, format!("field {field:?} must be an array")))?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                invalid_row(
                    row_index,
                    format!("field {field:?} contains a non-string choice"),
                )
            })
        })
        .collect()
}

pub(super) fn integer(row: &Value, field: &str, row_index: usize) -> Result<i64, AccuracyError> {
    if let Some(value) = row.get(field).and_then(Value::as_i64) {
        return Ok(value);
    }
    row.get(field)
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| invalid_row(row_index, format!("field {field:?} must be an integer")))
}

pub(super) fn invalid_row(row_index: usize, message: String) -> AccuracyError {
    AccuracyError::InvalidRow {
        question_id: None,
        message: format!("row {row_index}: {message}"),
    }
}

pub(super) fn generation(max_tokens: usize, stop: Vec<String>) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        stop,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn problem(
    benchmark: &str,
    item_id: impl AsRef<str>,
    task: impl AsRef<str>,
    messages: Vec<ChatMessage>,
    ground_truth: String,
    generation: GenerationConfig,
    metadata: BTreeMap<String, Value>,
) -> BenchmarkProblem {
    let id = format!("{}:{}", benchmark, item_id.as_ref());
    BenchmarkProblem {
        correlation_id: CorrelationId::new(id.clone()),
        id,
        task: TaskId::new(task.as_ref().to_string()),
        messages,
        ground_truth,
        generation,
        metadata,
    }
}

pub(super) fn item_id(row: &Value, row_index: usize, fields: &[&str]) -> String {
    for field in fields {
        if let Some(value) = row.get(field) {
            match value {
                Value::String(value) if !value.is_empty() => return value.clone(),
                Value::Number(value) => return value.to_string(),
                _ => {}
            }
        }
    }
    row_index.to_string()
}

pub(super) fn metadata(
    pairs: impl IntoIterator<Item = (&'static str, Value)>,
) -> BTreeMap<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

pub(super) fn finish_selection(
    _benchmark: &str,
    config: &BenchmarkConfig,
    mut problems: Vec<BenchmarkProblem>,
) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
    if let Some(limit) = config.max_problems {
        problems.truncate(limit);
    }
    Ok(problems)
}

pub(super) fn normalized_task(prefix: &str, task: &str) -> String {
    let suffix = task
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    format!("{prefix}.{suffix}")
}
