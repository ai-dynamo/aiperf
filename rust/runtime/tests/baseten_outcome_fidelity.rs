// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-registry integration coverage for Baseten recorded outcomes.

#![cfg(feature = "parquet")]

use std::sync::Arc;

use aiperf_runtime::dataset::{
    ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, Payload, TiktokenTokenizer,
};
use aiperf_runtime::rng::RngRoot;
use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::Value;

fn write_fixture(directory: &std::path::Path) -> std::path::PathBuf {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp_start_unix_ms", DataType::Int64, false),
        Field::new("prompt", DataType::Utf8, false),
        Field::new("input_tokens", DataType::Int64, false),
        Field::new("output_tokens", DataType::Int64, false),
        Field::new("provided_session_id", DataType::Utf8, false),
        Field::new("duration_e2e_ms", DataType::Int64, false),
        Field::new("duration_ttft_ms", DataType::Int64, false),
        Field::new("cached_tokens_reference", DataType::Int64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1_000])),
        Arc::new(StringArray::from(vec!["recorded prompt"])),
        Arc::new(Int64Array::from(vec![256])),
        Arc::new(Int64Array::from(vec![32])),
        Arc::new(StringArray::from(vec!["session-a"])),
        Arc::new(Int64Array::from(vec![900])),
        Arc::new(Int64Array::from(vec![125])),
        Arc::new(Int64Array::from(vec![192])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let path = directory.join("baseten-outcomes.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

#[tokio::test]
async fn builtin_baseten_pipeline_preserves_recorded_outcomes_without_dispatching_them() {
    let directory = tempfile::tempdir().unwrap();
    let path = write_fixture(directory.path());
    let registry = LoaderRegistry::with_builtin_formats().unwrap();
    let dataset = registry
        .build_dataset(
            Some("baseten_trace"),
            &LoadConfig::new(DatasetSource::Path(path)),
            &ComposeConfig::new("model", RngRoot::new(Some(7))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();

    let turn = &dataset.conversations()[0].turns[0];
    let outcome = turn.recorded_outcome.as_ref().unwrap();
    assert_eq!(outcome.duration_e2e_ms, Some(900.0));
    assert_eq!(outcome.duration_ttft_ms, Some(125.0));
    assert_eq!(outcome.cached_tokens_reference, Some(192));

    let Payload::Raw { wire } = dataset.segments().get(turn.extra_body.unwrap()).unwrap() else {
        panic!("expected Baseten request hints");
    };
    let body: Value = serde_json::from_slice(wire).unwrap();
    assert!(body.get("duration_e2e_ms").is_none());
    assert!(body.get("duration_ttft_ms").is_none());
    assert!(body.get("cached_tokens_reference").is_none());
}
