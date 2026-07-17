// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Arrow/Parquet writer boilerplate.
//!
//! The aggregated server-metrics sink ([`super::parquet`]) and the per-request
//! sidecar ([`super::per_record_parquet`]) build nullable Arrow columns and
//! Snappy-plus-metadata writer properties identically. Those primitives live
//! here so both sinks share one definition; each sink keeps its own
//! `build_schema`/`build_record_batch` and its own `write_parquet` wrapper
//! (which differ in their `with_context` messages and directory handling).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

/// Build a nullable UTF-8 column.
pub(crate) fn string_column<I: Iterator<Item = Option<String>>>(values: I) -> ArrayRef {
    Arc::new(StringArray::from_iter(values)) as ArrayRef
}

/// Build a nullable float64 column.
pub(crate) fn float_column<I: Iterator<Item = Option<f64>>>(values: I) -> ArrayRef {
    Arc::new(Float64Array::from_iter(values)) as ArrayRef
}

/// Snappy compression with schema key-value metadata copied to the file
/// metadata, so every Parquet file this crate writes carries identical
/// `aiperf.*` metadata and codec.
pub(crate) fn writer_properties(schema: &Arc<Schema>) -> WriterProperties {
    let kv: Vec<KeyValue> = schema
        .metadata()
        .iter()
        .map(|(key, value)| KeyValue::new(key.clone(), value.clone()))
        .collect();
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build()
}

/// Write one `RecordBatch` to a Parquet file at `path` under the shared Snappy +
/// metadata writer properties, creating the parent directory (idempotent).
///
/// `subject` labels the `with_context` diagnostics — e.g. `"parquet export"` for
/// the aggregated sink, `"per-record parquet"` for the sidecar — so each sink's
/// error context stays specific while the one-shot writer sequence lives once.
pub(crate) fn write_parquet_table(
    path: &Path,
    schema: Arc<Schema>,
    batch: &RecordBatch,
    subject: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {subject} directory {}", parent.display()))?;
    }
    let file =
        File::create(path).with_context(|| format!("creating {subject} {}", path.display()))?;
    let props = writer_properties(&schema);
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .with_context(|| format!("constructing {subject} arrow writer"))?;
    writer
        .write(batch)
        .with_context(|| format!("writing {subject} batch"))?;
    writer
        .close()
        .with_context(|| format!("finalizing {subject} file"))?;
    Ok(())
}
