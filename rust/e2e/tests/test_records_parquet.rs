// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use std::collections::HashMap;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use common::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const REQUEST_COUNT: u32 = 12;
const CONCURRENCY: u32 = 3;

fn read_parquet(
    path: &Path,
) -> (
    Vec<arrow::record_batch::RecordBatch>,
    HashMap<String, String>,
) {
    let file = std::fs::File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader builder");
    let metadata: HashMap<String, String> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .map(|kv| {
            kv.iter()
                .filter_map(|e| e.value.clone().map(|v| (e.key.clone(), v)))
                .collect()
        })
        .unwrap_or_default();
    // Drain every batch so row checks are independent of row-group splitting.
    let reader = builder.build().expect("parquet reader");
    let batches: Vec<_> = reader.map(|b| b.expect("parquet batch")).collect();
    (batches, metadata)
}

fn parquet_config(url: &str, records_line: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         \n\
         benchmark:\n\
        \x20 model: Qwen/Qwen2.5-Coder-32B-Instruct\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   requests: {REQUEST_COUNT}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        {records_line}",
    )
}

#[tokio::test]
async fn test_records_parquet_sidecar_mirrors_jsonl() {
    if cfg!(target_os = "macos") {
        return;
    }

    let h = AIPerfHarness::new().await;
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("records_parquet.yaml");
    std::fs::write(
        &cfg_file,
        parquet_config(
            &h.mock.url,
            "   records:\n     - jsonl\n     - csv\n     - parquet\n",
        ),
    )
    .unwrap();

    let r = h.run(&format!("--config {}", cfg_file.display()));
    assert!(
        r.success(),
        "run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let jsonl = r.artifacts.jsonl();
    assert!(
        !jsonl.is_empty(),
        "profile_export.jsonl should have records"
    );

    let parquet_path = r
        .artifacts
        .find_file("**/profile_export.parquet")
        .expect("profile_export.parquet should exist");
    let (batches, metadata) = read_parquet(&parquet_path);
    assert!(
        !batches.is_empty(),
        "parquet should contain at least one batch"
    );

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows,
        jsonl.len(),
        "one parquet row per JSONL record (parquet={total_rows}, jsonl={})",
        jsonl.len()
    );

    let schema = batches[0].schema();
    for column in [
        "session_num",
        "x_request_id",
        "benchmark_phase",
        "request_start_ns",
        "request_end_ns",
        "request_latency",
    ] {
        assert!(
            schema.index_of(column).is_ok(),
            "parquet schema should have column `{column}`; got {:?}",
            schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
        );
    }

    let batch = &batches[0];
    let ids = batch
        .column(schema.index_of("x_request_id").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("x_request_id is Utf8");
    assert!(!ids.value(0).is_empty(), "x_request_id should be populated");

    let phase = batch
        .column(schema.index_of("benchmark_phase").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("benchmark_phase is Utf8");
    assert!(
        (0..phase.len()).all(|i| matches!(phase.value(i), "profiling" | "warmup")),
        "benchmark_phase values should be profiling/warmup"
    );

    let session = batch
        .column(schema.index_of("session_num").unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("session_num is Int64");
    assert!(session.len() > 0);

    let latency = batch
        .column(schema.index_of("request_latency").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("request_latency is Float64");
    assert!(
        (0..latency.len()).any(|i| latency.is_valid(i) && latency.value(i) > 0.0),
        "at least one request_latency should be finite and positive"
    );

    assert_eq!(
        metadata.get("aiperf.schema_version").map(String::as_str),
        Some("1.0"),
        "parquet should carry aiperf.schema_version metadata"
    );
    assert!(
        metadata.contains_key("aiperf.units"),
        "parquet should carry the aiperf.units metadata map"
    );

    let csv_path = r
        .artifacts
        .find_file("**/profile_export_records.csv")
        .expect("profile_export_records.csv should exist");
    let csv_text = std::fs::read_to_string(&csv_path).unwrap();
    let mut csv_lines = csv_text.lines();
    let header: Vec<&str> = csv_lines.next().expect("csv header").split(',').collect();
    for column in [
        "session_num",
        "x_request_id",
        "benchmark_phase",
        "Request Latency (ms)",
        "error_code",
        "error_type",
        "error_message",
    ] {
        assert!(
            header.contains(&column),
            "records CSV header should have `{column}`; got {header:?}"
        );
    }
    assert!(
        !header.iter().any(|c| c.ends_with("_unit")),
        "records CSV should carry units in the header, not in _unit columns"
    );
    let data_rows: Vec<&str> = csv_lines.filter(|l| !l.is_empty()).collect();
    assert_eq!(
        data_rows.len(),
        jsonl.len(),
        "one CSV row per JSONL record (csv={}, jsonl={})",
        data_rows.len(),
        jsonl.len()
    );
    let latency_col = header
        .iter()
        .position(|c| *c == "Request Latency (ms)")
        .unwrap();
    assert!(
        data_rows.iter().any(|row| {
            row.split(',')
                .nth(latency_col)
                .and_then(|cell| cell.parse::<f64>().ok())
                .is_some_and(|v| v > 0.0)
        }),
        "at least one CSV request_latency_value should be a positive number"
    );
}
