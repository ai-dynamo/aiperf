// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict `aiperf.trace.v1` schema.
//!
//! One object per session: a content-addressed **segment pool** (`segments`,
//! each a message with an explicit `role` and its block `hash_ids`) plus
//! **`inference_calls`** that reference pooled segments by index and carry
//! timing, usage, and the conversation graph (`previous_ref`, `compaction`,
//! hashed agent ids). The block ids are opaque prefix-encoded rolling ids.

use serde::Deserialize;
use serde_json::Value;

use crate::graph::recorded::RecordedTraceError;

const SCHEMA_TAG: &str = "aiperf.trace.v1";

/// One parsed session trace.
#[derive(Debug)]
pub(super) struct AIPerfTrace {
    pub id: String,
    pub block_size: usize,
    pub segments: Vec<AIPerfSegment>,
    pub calls: Vec<AIPerfCall>,
}

/// A pooled message segment: its role and the opaque block ids it covers.
#[derive(Debug)]
pub(super) struct AIPerfSegment {
    pub role: String,
    pub hash_ids: Vec<i128>,
    pub tokens: usize,
}

/// One model call: pooled prompt segments + response, timing, usage, and graph.
#[derive(Debug)]
pub(super) struct AIPerfCall {
    pub segment_refs: Vec<usize>,
    /// Pooled segments that make up the response. A response may span multiple
    /// segments (e.g. an assistant turn plus its tool-call segments), so this is a
    /// list; a single-segment response is just a one-element list.
    pub response_refs: Vec<usize>,
    pub ts_ms: f64,
    pub ttft_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    pub model: Option<String>,
    pub request_kind: Option<String>,
    pub agent_id: Option<u64>,
    pub parent_agent_id: Option<u64>,
    pub previous_ref: Option<usize>,
    pub compaction: Option<AIPerfCompaction>,
    pub output_tokens: Option<usize>,
}

/// A compaction edge: this turn reset (summarized) the context of `prior_ref`.
#[derive(Debug)]
pub(super) struct AIPerfCompaction {
    pub prior_ref: Option<usize>,
    pub prior_segments: Option<usize>,
}

// --- wire (serde) structs; deny unknown top-level keys for a strict parse ---

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceWire {
    schema: String,
    block_size: usize,
    session_id: u64,
    segments: Vec<SegWire>,
    inference_calls: Vec<CallWire>,
    // Present but not consumed by the adapter — accepted, ignored.
    #[serde(default)]
    provenance: Value,
    #[serde(default)]
    hash_id_salt: Value,
    #[serde(default)]
    time_anchor_ms: Value,
    #[serde(default)]
    role_counts: Value,
    #[serde(default)]
    meta: Value,
}

#[derive(Deserialize)]
struct SegWire {
    role: String,
    hash_ids: Vec<u64>,
    tokens: usize,
    // `kind` / `tool_calls` present in the format but not needed by lowering; the
    // struct is non-strict, so serde ignores them.
}

#[derive(Deserialize)]
struct CallWire {
    #[serde(default)]
    ts: f64,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    request_kind: Option<String>,
    #[serde(default)]
    agent_id: Option<u64>,
    #[serde(default)]
    parent_agent_id: Option<u64>,
    #[serde(default)]
    ttft_ms: Option<f64>,
    #[serde(default)]
    e2e_latency_ms: Option<f64>,
    #[serde(default)]
    previous_ref: Option<usize>,
    #[serde(default)]
    compaction: Option<CompactionWire>,
    #[serde(default)]
    segment_refs: Vec<usize>,
    // A response may cover several pooled segments. `response_refs` is the
    // canonical list; `response_ref` is accepted as a single-segment shorthand
    // and folded in when the list form is absent.
    #[serde(default)]
    response_refs: Vec<usize>,
    #[serde(default)]
    response_ref: Option<usize>,
    #[serde(default)]
    usage: UsageWire,
}

#[derive(Deserialize, Default)]
struct UsageWire {
    #[serde(default)]
    output_tokens: Option<usize>,
    // `input_tokens` / `cache_read_input_tokens` / `tool_usage` are ignored — the
    // adapter derives input length block-granularly from the segments' hash_ids.
}

#[derive(Deserialize)]
struct CompactionWire {
    #[serde(default)]
    prior_ref: Option<usize>,
    #[serde(default)]
    prior_segments: Option<usize>,
}

/// Strictly parse one `aiperf.trace.v1` object.
pub(super) fn parse_trace(value: Value) -> Result<AIPerfTrace, RecordedTraceError> {
    let wire: TraceWire = serde_json::from_value(value)
        .map_err(|error| RecordedTraceError(format!("invalid aiperf.trace.v1 object: {error}")))?;
    let _ = (
        &wire.provenance,
        &wire.hash_id_salt,
        &wire.time_anchor_ms,
        &wire.role_counts,
        &wire.meta,
    );
    if wire.schema != SCHEMA_TAG {
        return Err(RecordedTraceError(format!(
            "aiperf trace schema must be {SCHEMA_TAG:?}, got {:?}",
            wire.schema
        )));
    }
    if wire.block_size == 0 {
        return Err(RecordedTraceError(
            "aiperf trace block_size must be > 0".into(),
        ));
    }

    let segments = wire
        .segments
        .into_iter()
        .map(|seg| AIPerfSegment {
            role: seg.role,
            hash_ids: seg.hash_ids.into_iter().map(i128::from).collect(),
            tokens: seg.tokens,
        })
        .collect();

    let calls = wire
        .inference_calls
        .into_iter()
        .map(|call| AIPerfCall {
            segment_refs: call.segment_refs,
            response_refs: if call.response_refs.is_empty() {
                call.response_ref.into_iter().collect()
            } else {
                call.response_refs
            },
            ts_ms: call.ts,
            ttft_ms: call.ttft_ms,
            e2e_latency_ms: call.e2e_latency_ms,
            model: call.model,
            request_kind: call.request_kind,
            agent_id: call.agent_id,
            parent_agent_id: call.parent_agent_id,
            previous_ref: call.previous_ref,
            compaction: call.compaction.map(|c| AIPerfCompaction {
                prior_ref: c.prior_ref,
                prior_segments: c.prior_segments,
            }),
            output_tokens: call.usage.output_tokens,
        })
        .collect();

    Ok(AIPerfTrace {
        id: wire.session_id.to_string(),
        block_size: wire.block_size,
        segments,
        calls,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn minimal() -> Value {
        json!({
            "schema": SCHEMA_TAG,
            "session_id": 1,
            "block_size": 16,
            "segments": [{"role": "user", "hash_ids": [1], "tokens": 4}],
            "inference_calls": [{"ts": 0.0, "segment_refs": [0], "usage": {}}]
        })
    }

    #[test]
    fn parses_minimal_session() {
        let trace = parse_trace(minimal()).unwrap();
        assert_eq!(trace.id, "1");
        assert_eq!(trace.block_size, 16);
        assert_eq!(trace.segments.len(), 1);
        assert_eq!(trace.calls.len(), 1);
    }

    #[test]
    fn rejects_wrong_schema_tag() {
        let mut value = minimal();
        value["schema"] = json!("weka.trace.v1");
        assert!(parse_trace(value).is_err());
    }

    #[test]
    fn rejects_zero_block_size() {
        let mut value = minimal();
        value["block_size"] = json!(0);
        assert!(parse_trace(value).is_err());
    }

    #[test]
    fn rejects_unknown_top_level_key() {
        let mut value = minimal();
        value["surprise"] = json!(true);
        assert!(parse_trace(value).is_err());
    }
}
