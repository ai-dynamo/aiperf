// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact serde schema for the Weka KV-cache-tester agentic coding trace
//! format, ported from `src/aiperf/dataset/loader/weka_trace_models.py`.
//!
//! Each trace file is one JSON object; `requests` is an ordered list interleaving
//! normal API calls (`type: "n"`), streaming API calls (`type: "s"`), and
//! subagent markers (`type: "subagent"`) with their own nested request lists.
//! The Python models use `extra="forbid"`, mirrored here with
//! `deny_unknown_fields`, and alias `in`/`out` to `input_length`/`output_length`.

use serde::Deserialize;

/// One normal (`type: "n"`) API call in a Weka trace.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WekaNormalRequest {
    /// Request timestamp in seconds from conversation start.
    pub t: f64,
    /// Model identifier for this request.
    pub model: String,
    /// Input token count (wire key `in`).
    #[serde(rename = "in")]
    pub input_length: i64,
    /// Output token count (wire key `out`).
    #[serde(rename = "out")]
    pub output_length: i64,
    /// KV-cache block hash IDs.
    #[serde(default)]
    pub hash_ids: Vec<i64>,
    /// Content-type annotations for input.
    #[serde(default)]
    pub input_types: Vec<String>,
    /// Content-type annotations for output.
    #[serde(default)]
    pub output_types: Vec<String>,
    /// Stop reason: `""`, `"tool_use"`, `"end_turn"`.
    #[serde(default)]
    pub stop: String,
    /// Server processing time in seconds.
    #[serde(default)]
    pub api_time: Option<f64>,
    /// Client delay in seconds before this request.
    #[serde(default)]
    pub think_time: Option<f64>,
}

/// One streaming (`type: "s"`) API call. Structurally identical to
/// [`WekaNormalRequest`] plus an optional recorded time-to-first-token.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WekaStreamingRequest {
    /// Request timestamp in seconds from conversation start.
    pub t: f64,
    /// Model identifier for this request.
    pub model: String,
    /// Input token count (wire key `in`).
    #[serde(rename = "in")]
    pub input_length: i64,
    /// Output token count (wire key `out`).
    #[serde(rename = "out")]
    pub output_length: i64,
    /// KV-cache block hash IDs.
    #[serde(default)]
    pub hash_ids: Vec<i64>,
    /// Content-type annotations for input.
    #[serde(default)]
    pub input_types: Vec<String>,
    /// Content-type annotations for output.
    #[serde(default)]
    pub output_types: Vec<String>,
    /// Stop reason: `""`, `"tool_use"`, `"end_turn"`.
    #[serde(default)]
    pub stop: String,
    /// Server processing time in seconds.
    #[serde(default)]
    pub api_time: Option<f64>,
    /// Client delay in seconds before this request.
    #[serde(default)]
    pub think_time: Option<f64>,
    /// Recorded time-to-first-token in seconds.
    #[serde(default)]
    pub ttft: Option<f64>,
}

/// An inner request of a subagent: a normal or streaming API call (never a
/// nested subagent).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum WekaInnerRequest {
    /// Normal inner call.
    #[serde(rename = "n")]
    Normal(WekaNormalRequest),
    /// Streaming inner call.
    #[serde(rename = "s")]
    Streaming(WekaStreamingRequest),
}

/// A `type: "subagent"` marker with its nested inner requests. The parent's next
/// request in the outer list is understood to occur after this subagent
/// completes.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WekaSubagentEntry {
    /// Spawn timestamp in seconds from conversation start.
    pub t: f64,
    /// Opaque subagent identifier, e.g. `"agent_001"`.
    pub agent_id: String,
    /// Subagent type, e.g. `"Explore"`.
    pub subagent_type: String,
    /// Wall-clock duration; `None` for `status = "async_launched"`.
    #[serde(default)]
    pub duration_ms: Option<i64>,
    /// Total tokens across inner requests; `None` for `async_launched`.
    #[serde(default)]
    pub total_tokens: Option<i64>,
    /// Tool calls made by the subagent; `None` for `async_launched`.
    #[serde(default)]
    pub tool_use_count: Option<i64>,
    /// `"completed"` or another terminal status.
    pub status: String,
    /// Inner requests (normal or streaming API calls).
    pub requests: Vec<WekaInnerRequest>,
    /// Models used by the subagent.
    pub models: Vec<String>,
    /// Subagent's tools prefix token count.
    #[serde(default)]
    pub tool_tokens: i64,
    /// Subagent's system prefix token count.
    #[serde(default)]
    pub system_tokens: i64,
}

/// An entry in a trace's top-level `requests` list.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum WekaRequest {
    /// Normal top-level API call.
    #[serde(rename = "n")]
    Normal(WekaNormalRequest),
    /// Streaming top-level API call.
    #[serde(rename = "s")]
    Streaming(WekaStreamingRequest),
    /// Subagent marker with nested inner requests.
    #[serde(rename = "subagent")]
    Subagent(WekaSubagentEntry),
}

/// Hash-ID namespace scope. The v1 loader supports only `"local"` (per-trace
/// hashes); `"global"` cross-trace sharing is rejected at schema level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HashIdScope {
    /// Per-trace hash namespace.
    Local,
}

/// A single Weka trace file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WekaTrace {
    /// Trace identifier (session ID).
    pub id: String,
    /// Models used in the trace.
    pub models: Vec<String>,
    /// Cache block size in tokens (must be `> 0`).
    pub block_size: i64,
    /// Hash-ID namespace scope (`"local"` only).
    pub hash_id_scope: HashIdScope,
    /// Tools prefix token count.
    #[serde(default)]
    pub tool_tokens: i64,
    /// System prefix token count.
    #[serde(default)]
    pub system_tokens: i64,
    /// Interleaved normal/streaming requests and subagent markers.
    pub requests: Vec<WekaRequest>,
    /// Optional trace-level summary; opaque.
    #[serde(default)]
    pub totals: Option<serde_json::Value>,
}

impl WekaTrace {
    /// Parse a trace from JSON bytes, rejecting unknown fields and non-`local`
    /// hash scope (matching the Python pydantic `extra="forbid"` contract).
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let trace: WekaTrace = serde_json::from_slice(bytes)?;
        Ok(trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixture_root() -> std::path::PathBuf {
        // `CARGO_MANIFEST_DIR` is `rust/runtime`; the repo root is two parents up.
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/weka_traces")
    }

    #[test]
    fn parses_one_subagent_fixture() {
        let path = fixture_root().join("one_subagent.json");
        let bytes = std::fs::read(&path).expect("read fixture");
        let trace = WekaTrace::from_json_bytes(&bytes).expect("parse");
        assert_eq!(trace.hash_id_scope, HashIdScope::Local);
        assert!(trace.block_size > 0);
        // Interleaves at least one normal and one subagent entry.
        let has_normal = trace
            .requests
            .iter()
            .any(|r| matches!(r, WekaRequest::Normal(_)));
        let has_subagent = trace
            .requests
            .iter()
            .any(|r| matches!(r, WekaRequest::Subagent(_)));
        assert!(has_normal && has_subagent);
    }

    #[test]
    fn parses_all_valid_fixtures() {
        let dir = fixture_root();
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("read dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            WekaTrace::from_json_bytes(&bytes)
                .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
            count += 1;
        }
        assert!(count > 0, "no fixtures found in {}", dir.display());
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let json = br#"{"id":"x","models":["m"],"block_size":8,"hash_id_scope":"local","requests":[],"bogus":1}"#;
        assert!(WekaTrace::from_json_bytes(json).is_err());
    }
}
