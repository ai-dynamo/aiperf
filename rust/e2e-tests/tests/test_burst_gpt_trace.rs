// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::{Path, PathBuf};

fn sample_rows() -> Vec<Vec<String>> {
    let raw: &[(&str, &str, u32, u32, u32, &str)] = &[
        ("0.0", "ChatGPT", 472, 18, 490, "Conversation log"),
        ("0.1", "ChatGPT", 1087, 230, 1317, "Conversation log"),
        ("0.2", "GPT-4", 417, 276, 693, "Conversation log"),
        ("0.3", "ChatGPT", 1360, 647, 2007, "Conversation log"),
        ("0.4", "ChatGPT", 185, 215, 400, "Conversation log"),
    ];
    raw.iter()
        .map(|(ts, model, req, resp, total, log)| {
            vec![
                ts.to_string(),
                model.to_string(),
                req.to_string(),
                resp.to_string(),
                total.to_string(),
                log.to_string(),
            ]
        })
        .collect()
}

fn create_burst_gpt_csv_file(dir: &Path, rows: &[Vec<String>]) -> PathBuf {
    let headers = [
        "Timestamp",
        "Model",
        "Request tokens",
        "Response tokens",
        "Total tokens",
        "Log Type",
    ];
    write_csv(dir, "burst_gpt.csv", &headers, rows)
}

#[tokio::test]
async fn test_fixed_schedule_with_explicit_dataset_type() {
    let h = AIPerfHarness::new().await;
    let rows = sample_rows();
    let csv_file = create_burst_gpt_csv_file(h.artifact_dir.path(), &rows);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type burst_gpt_trace \
         --request-count {} --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        csv_file.display(),
        rows.len(),
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, rows.len());
}

#[tokio::test]
async fn test_fixed_schedule_auto_detected() {
    let h = AIPerfHarness::new().await;
    let rows = sample_rows();
    let csv_file = create_burst_gpt_csv_file(h.artifact_dir.path(), &rows);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --request-count {} \
         --fixed-schedule --fixed-schedule-auto-offset \
         --workers-max 1 --ui simple",
        h.mock.url,
        csv_file.display(),
        rows.len(),
    ));

    assert_eq!(r.exit_code, 0);
    assert_eq!(r.artifacts.request_count() as usize, rows.len());
}
