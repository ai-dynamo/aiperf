// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::Path;

const API_KEY: &str = "sk-integration-secret-REDACT-12345";
const REDACTED_VALUE: &str = "<redacted>";

const COMMON_FLAGS: &str = "--endpoint-type chat --streaming --request-count 5 --concurrency 2 \
     --workers-max 1 --extra-verbose";

fn assert_api_key_not_in_logs(dir: &Path) {
    walk_files(dir, &|path| {
        if path.extension().and_then(|e| e.to_str()) == Some("log") {
            let content = read_lossy(path);
            assert!(
                !content.contains(API_KEY),
                "API key leaked into log file: {}",
                path.display()
            );
        }
    });
}

fn assert_api_key_not_in_any_artifact(dir: &Path) {
    let text_exts = ["json", "jsonl", "csv", "log", "yaml", "yml", "txt"];
    walk_files(dir, &|path| {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if text_exts.contains(&ext) {
                let content = read_lossy(path);
                assert!(
                    !content.contains(API_KEY),
                    "API key leaked into artifact: {}",
                    path.display()
                );
            }
        }
    });
}

fn walk_files(dir: &Path, f: &dyn Fn(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, f);
        } else if path.is_file() {
            f(&path);
        }
    }
}

fn read_lossy(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

#[tokio::test]
async fn test_api_key_redacted_in_raw_records_http() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} --export-level raw",
        h.mock.url
    ));

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 5);

    for record in &records {
        let headers = &record["request_headers"];
        assert!(!headers.is_null(), "raw record missing request_headers");
        let headers_str = serde_json::to_string(headers).unwrap();
        assert!(
            !headers_str.contains(API_KEY),
            "API key leaked into raw record headers: {headers_str}"
        );
        assert_eq!(headers["Authorization"], REDACTED_VALUE);
    }

    assert_api_key_not_in_logs(&r.artifacts.dir);
}

#[tokio::test]
async fn test_raw_file_and_logs_do_not_contain_api_key_http() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} --export-level raw",
        h.mock.url
    ));

    let raw_file = r
        .artifacts
        .find_file("**/*profile_export_raw.jsonl")
        .expect("raw records JSONL file present");
    let content = read_lossy(&raw_file);
    assert!(
        !content.contains(API_KEY),
        "Real API key found in raw records JSONL file"
    );

    assert_api_key_not_in_logs(&r.artifacts.dir);
}

#[tokio::test]
async fn test_api_key_redacted_in_http_trace() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} --export-http-trace",
        h.mock.url
    ));

    let records = r.artifacts.jsonl();
    assert_eq!(records.len(), 5);

    for record in &records {
        let headers = &record["trace_data"]["request_headers"];
        if !headers.is_null() {
            let headers_str = serde_json::to_string(headers).unwrap();
            assert!(
                !headers_str.contains(API_KEY),
                "API key leaked into trace request_headers: {headers_str}"
            );
            assert_eq!(headers["Authorization"], REDACTED_VALUE);
        }
    }

    assert_api_key_not_in_logs(&r.artifacts.dir);
}

#[tokio::test]
async fn test_jsonl_file_and_logs_do_not_contain_api_key_with_trace() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} --export-http-trace",
        h.mock.url
    ));

    let jsonl_file = r
        .artifacts
        .find_file("**/*profile_export.jsonl")
        .expect("profile export JSONL file present");
    let content = read_lossy(&jsonl_file);
    assert!(
        !content.contains(API_KEY),
        "API key found in JSONL file with --export-http-trace enabled"
    );

    assert_api_key_not_in_logs(&r.artifacts.dir);
}

#[tokio::test]
async fn test_combined_raw_and_trace_redaction() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} \
         --export-level raw --export-http-trace",
        h.mock.url
    ));

    for record in &r.artifacts.raw_records() {
        let headers = &record["request_headers"];
        if !headers.is_null() {
            let headers_str = serde_json::to_string(headers).unwrap();
            assert!(!headers_str.contains(API_KEY));
        }
    }

    for record in &r.artifacts.jsonl() {
        let headers = &record["trace_data"]["request_headers"];
        if !headers.is_null() {
            let headers_str = serde_json::to_string(headers).unwrap();
            assert!(!headers_str.contains(API_KEY));
        }
    }

    assert_api_key_not_in_any_artifact(&r.artifacts.dir);
}

#[tokio::test]
async fn test_no_artifact_contains_api_key() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY} \
         --export-level raw --export-http-trace",
        h.mock.url
    ));

    assert_api_key_not_in_any_artifact(&r.artifacts.dir);
}

#[tokio::test]
async fn test_benchmark_succeeds_with_api_key_http() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} --api-key {API_KEY}",
        h.mock.url
    ));

    assert_eq!(r.artifacts.request_count() as u32, 5);
    let json = r.artifacts.json();
    assert!(!json.is_null());
    assert!(!json["request_latency"].is_null());

    assert_api_key_not_in_logs(&r.artifacts.dir);
}

#[tokio::test]
async fn test_non_sensitive_headers_preserved_http() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} {COMMON_FLAGS} \
         --header \"X-Custom-Tracking:trace-abc-123\" --export-level raw",
        h.mock.url
    ));

    let records = r.artifacts.raw_records();
    assert!(!records.is_empty());
    for record in &records {
        let headers = &record["request_headers"];
        assert!(!headers.is_null(), "raw record missing request_headers");
        assert_eq!(headers["X-Custom-Tracking"], "trace-abc-123");
    }
}
