// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::fs;
use std::path::{Path, PathBuf};

fn is_valid_png(path: &Path) -> bool {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return false,
    };
    if bytes.len() < 8 {
        return false;
    }
    const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    bytes[..8] == PNG_SIGNATURE
}

fn validate_png_ihdr_chunk(path: &Path) -> Option<(u32, u32)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 8 + 4 + 4 + 8 {
        return None;
    }
    let mut cursor = 8usize;

    let chunk_length = u32::from_be_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ]) as usize;
    cursor += 4;

    let chunk_type = &bytes[cursor..cursor + 4];
    if chunk_type != b"IHDR" {
        return None;
    }
    cursor += 4;

    if bytes.len() < cursor + chunk_length || chunk_length < 8 {
        return None;
    }
    let ihdr = &bytes[cursor..cursor + chunk_length];
    let width = u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]);
    let height = u32::from_be_bytes([ihdr[4], ihdr[5], ihdr[6], ihdr[7]]);
    Some((width, height))
}

fn png_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(Result::ok) {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("png") {
                out.push(p);
            }
        }
    }
    out
}

#[tokio::test]
#[ignore] // requires: matplotlib plotting backend
async fn test_profile_then_plot_single_run() {
    let h = AIPerfHarness::new().await;

    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --request-count {DEFAULT_REQUEST_COUNT} \
         --concurrency {DEFAULT_CONCURRENCY} --streaming",
        h.mock.url
    ));
    assert_eq!(profile_result.exit_code, 0);
    assert_eq!(
        profile_result.artifacts.request_count() as u32,
        DEFAULT_REQUEST_COUNT
    );

    let artifacts_dir = h.artifact_path().to_path_buf();

    let plot_result = h.run_no_server(&format!("plot --paths {}", artifacts_dir.display()));
    assert_eq!(plot_result.exit_code, 0);

    let plot_dir = artifacts_dir.join("plots");
    assert!(
        plot_dir.exists(),
        "Plot directory not created at {plot_dir:?}"
    );

    let pngs = png_files(&plot_dir);
    assert!(!pngs.is_empty(), "No PNG files were generated");

    for png_path in &pngs {
        assert!(
            is_valid_png(png_path),
            "Plot {:?} is not a valid PNG file",
            png_path.file_name()
        );
        let dimensions = validate_png_ihdr_chunk(png_path);
        assert!(
            dimensions.is_some(),
            "Plot {:?} has invalid IHDR chunk",
            png_path.file_name()
        );
        let (width, height) = dimensions.unwrap();
        assert!(
            width > 0 && height > 0,
            "Plot {:?} has invalid dimensions: {width}x{height}",
            png_path.file_name()
        );
    }

    let summary_path = plot_dir.join("summary.txt");
    assert!(summary_path.exists(), "Plot summary.txt was not created");
    let summary_content = fs::read_to_string(&summary_path).expect("read summary.txt");
    assert!(summary_content.contains("Generated"));
    assert!(summary_content.contains("plots:"));
}

#[tokio::test]
#[ignore] // requires: matplotlib plotting backend
async fn test_profile_then_plot_with_timeslices() {
    let h = AIPerfHarness::new().await;

    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --concurrency {DEFAULT_CONCURRENCY} --streaming \
         --benchmark-duration 3 --benchmark-grace-period 0 --slice-duration 1",
        h.mock.url
    ));
    assert_eq!(profile_result.exit_code, 0);

    let artifacts_dir = h.artifact_path().to_path_buf();

    let plot_result = h.run_no_server(&format!("plot --paths {}", artifacts_dir.display()));
    assert_eq!(plot_result.exit_code, 0);

    let plot_dir = artifacts_dir.join("plots");
    assert!(plot_dir.exists(), "Plot directory not created");

    let pngs = png_files(&plot_dir);
    assert!(!pngs.is_empty(), "No PNG files were generated");

    for png_path in &pngs {
        assert!(
            is_valid_png(png_path),
            "Plot {:?} is not valid",
            png_path.file_name()
        );
    }
}

#[tokio::test]
#[ignore] // requires: matplotlib plotting backend
async fn test_plot_with_nonexistent_directory_fails() {
    let h = AIPerfHarness::new().await;

    let plot_result = h.run_no_server("plot --paths /nonexistent/path/to/artifacts");
    assert_ne!(plot_result.exit_code, 0);
}

#[tokio::test]
#[ignore] // requires: matplotlib plotting backend
async fn test_plot_with_server_metrics_parquet_and_json() {
    let h = AIPerfHarness::new().await;
    let vllm_url = h.mock.server_metrics_urls()["vllm"].clone();

    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --request-count 50 --concurrency 2 --streaming \
         --server-metrics {vllm_url} --server-metrics-formats parquet json",
        h.mock.url
    ));
    assert_eq!(profile_result.exit_code, 0);
    assert_eq!(profile_result.artifacts.request_count() as u32, 50);

    let artifacts_dir = h.artifact_path().to_path_buf();

    let parquet_file = artifacts_dir.join("server_metrics_export.parquet");
    let json_file = artifacts_dir.join("server_metrics_export.json");
    assert!(parquet_file.exists(), "Parquet file should exist");
    assert!(json_file.exists(), "JSON file should exist");

    let plot_result = h.run_no_server(&format!("plot --paths {}", artifacts_dir.display()));
    assert_eq!(plot_result.exit_code, 0);

    let plot_dir = artifacts_dir.join("plots");
    assert!(plot_dir.exists(), "Plot directory should exist");

    let pngs = png_files(&plot_dir);
    assert!(!pngs.is_empty(), "At least some plots should be created");

    for png_path in &pngs {
        assert!(
            is_valid_png(png_path),
            "{:?} is not a valid PNG",
            png_path.file_name()
        );
        let dimensions = validate_png_ihdr_chunk(png_path);
        assert!(
            dimensions.is_some(),
            "{:?} has invalid IHDR chunk",
            png_path.file_name()
        );
        let (width, height) = dimensions.unwrap();
        assert!(
            width >= 800 && height >= 600,
            "{:?} has unexpectedly small dimensions: {width}x{height}",
            png_path.file_name()
        );
    }
}

#[tokio::test]
#[ignore] // requires: matplotlib plotting backend
async fn test_plot_with_server_metrics_parquet_only() {
    let h = AIPerfHarness::new().await;
    let vllm_url = h.mock.server_metrics_urls()["vllm"].clone();

    let profile_result = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --request-count 50 --concurrency 2 --streaming \
         --server-metrics {vllm_url} --server-metrics-formats parquet",
        h.mock.url
    ));
    assert_eq!(profile_result.exit_code, 0);

    let artifacts_dir = h.artifact_path().to_path_buf();

    let parquet_file = artifacts_dir.join("server_metrics_export.parquet");
    let json_file = artifacts_dir.join("server_metrics_export.json");
    assert!(parquet_file.exists(), "Parquet file should exist");
    assert!(
        !json_file.exists(),
        "JSON file should NOT exist (Parquet-only test)"
    );

    let plot_result = h.run_no_server(&format!("plot --paths {}", artifacts_dir.display()));
    assert_eq!(plot_result.exit_code, 0);

    let plot_dir = artifacts_dir.join("plots");
    let pngs = png_files(&plot_dir);
    assert!(
        !pngs.is_empty(),
        "Plots should be generated even with Parquet-only export"
    );

    for png_path in &pngs {
        assert!(
            is_valid_png(png_path),
            "{:?} is not a valid PNG",
            png_path.file_name()
        );
    }
}
