// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live HuggingFace fetch of the real AgentX corpus. Ignored by default (hits
//! the network); run with `--ignored`. Downloads `semianalysisai/cc-traces-weka-062126`
//! (393 public v7 traces), validates rows as `WekaTrace`, and reconstructs.
#![cfg(feature = "engine")]
#![cfg(feature = "parquet")]

use aiperf_runtime::agentx::hf_dataset::{HfDatasetRef, fetch_hf_weka_rows};

#[tokio::test]
#[ignore = "hits the Hugging Face hub"]
async fn fetch_semianalysis_062126() {
    let rows = fetch_hf_weka_rows(HfDatasetRef::new("semianalysisai/cc-traces-weka-062126"))
        .await
        .expect("download 062126 corpus");
    eprintln!("fetched {} rows", rows.len());
    assert_eq!(rows.len(), 393, "expected 393 v7 traces");

    // Every row must validate as a WekaTrace via the same path file replay uses.
    let (traces, stats) = aiperf_runtime::agentx::loader::load_hf_traces_from_rows(
        rows,
        "semianalysisai/cc-traces-weka-062126",
        Some(3),
        None,
        None,
    )
    .expect("rows validate + select");
    assert_eq!(traces.len(), 3, "filter-then-cap kept 3");
    eprintln!(
        "scanned={} kept={} first_id={:?} first_requests={}",
        stats.scanned,
        traces.len(),
        traces[0].0,
        traces[0].1.requests.len()
    );
    assert!(!traces[0].1.requests.is_empty(), "trace has requests");
}
