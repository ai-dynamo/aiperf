// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed artifact output policy.
//!
//! `inputs_path` and `trace` are always present. Per-record paths are present
//! only when their formats are selected.

use serde::{Deserialize, Serialize};

/// A pre-rendered UTF-8 file for the runner to materialize.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserFile {
    pub path: String,
    pub format: String,
    pub content: String,
}

/// The typed native output policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Artifacts {
    pub trace: bool,
    pub inputs_path: String,
    /// Per-record JSONL (present when `jsonl`/`raw` selected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records_path: Option<String>,
    /// Per-record Parquet (present when `parquet` selected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records_parquet_path: Option<String>,
    /// Per-record CSV (present when `csv` selected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records_csv_path: Option<String>,
    /// Per-request outputs JSON (present when enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs_path: Option<String>,
    /// Raw per-record JSONL (present when `raw` enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_path: Option<String>,
    /// Once-rendered user files (present when authored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_files: Option<Vec<UserFile>>,
    /// Base path for the `--dry-run` dataset-analysis artifact family, relative to
    /// the run directory. Present only when the dry-run analysis is requested; the
    /// runtime emits `dataset_analysis.{txt,json,csv,html}` beside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_analysis_path: Option<String>,
    /// KV-cache block size (tokens) for the dry-run cache-reuse analysis. Absent →
    /// the runtime default (16).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_analysis_block_size: Option<u32>,
    /// Explicit realized-LRU cache capacity (blocks) to add as a sweep point in the
    /// dry-run analysis. Absent → capacity sweep only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_analysis_cache_blocks: Option<u64>,
    /// Emit per-conversation breakdowns in the dry-run analysis.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dataset_analysis_per_conversation: bool,
    /// Recorded-agent tool timing output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_tool_time_path: Option<String>,
    /// Recorded-agent trace summary output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_trace_summary_path: Option<String>,
    /// Recorded-agent normalized replay metrics JSON output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_replay_metrics_path: Option<String>,
    /// Optional recorded-agent normalized replay metrics CSV output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_replay_metrics_csv_path: Option<String>,
    /// Recorded-agent failed-task output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_replay_failures_path: Option<String>,
    /// Recorded-agent replay provenance output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_replay_provenance_path: Option<String>,
    /// Recorded-agent replay backend metadata output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_replay_backend_metadata_path: Option<String>,
}
