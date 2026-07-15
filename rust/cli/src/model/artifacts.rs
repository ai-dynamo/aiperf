// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `artifacts` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_authored_artifacts`.
//! `inputs_path` and `trace` are always present; each per-record output path is
//! present only when its format is selected. `user_files` carries once-rendered
//! UTF-8 content the runner materializes.

use serde::{Deserialize, Serialize};

/// One pre-rendered user file (Python owns Jinja/serialization; the runner only
/// materializes the bytes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserFile {
    /// Output-relative path.
    pub path: String,
    /// File format token.
    pub format: String,
    /// Rendered UTF-8 content.
    pub content: String,
}

/// The typed native output policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Artifacts {
    /// Emit per-request trace columns.
    pub trace: bool,
    /// Per-session formatted request payloads (always requested).
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
