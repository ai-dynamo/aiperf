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
}
