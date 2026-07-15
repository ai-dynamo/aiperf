// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Deterministic per-trace `t*` sampling for snapshot-at-`t*` warmup partition.
//!
//! Byte-exact port of the agentx `t*` draw from
//! `src/aiperf/timing/graph_ir_source.py:113-150` (`_sample_t_star` and
//! `_seed_for_trace_lane`) plus the trace-duration helper from
//! `src/aiperf/graph/analysis/snapshot.py:65` (`trace_duration_us`).
//!
//! A sampling instant `t*` splits a trace's firings into a warmup prefix
//! (`arrival_offset_us < t*`) and a profiled set. `t*` is drawn uniformly over
//! `[start_min_ratio, start_max_ratio] * trace_duration`, with a LANE-salted,
//! numpy-bit-compatible RNG so seeded Rust and Python produce identical draws.
//! Python numpy is the fixed reference; this Rust conforms to it.

use crate::graph::model::{ParsedGraph, TraceRecord};
use crate::rng::numpy_pcg64::NumpyPcg64;
use sha2::{Digest, Sha256};

/// Draws a deterministic per-trace `t*` (microseconds) for warmup partition.
///
/// Extension seam: the window policy is one implementation of `t*` selection;
/// alternative schedules (fixed instants, empirical distributions) can supply a
/// different sampler without touching the snapshot partition.
pub trait TStarSampler {
    /// Draw `t*` in microseconds for `trace_id` on `lane`, given the trace's
    /// intrinsic `duration_us` (see [`trace_duration_us`]).
    fn sample_t_star(&self, trace_id: &str, lane: u64, duration_us: f64) -> f64;
}

/// Window-based `t*` sampler: `t* = uniform(min*dur, max*dur)`.
///
/// Byte parity with `graph_ir_source.py:_sample_t_star` (lines 113-138): a
/// zero-or-negative duration or a collapsed window (`hi <= lo`, e.g. the
/// default `[0, 0]`) yields a no-draw exact instant; otherwise a lane-salted
/// numpy-compatible uniform draw. The result is a float (no integer-microsecond
/// truncation), matching agentx.
pub struct WindowTStarSampler {
    /// Lower window bound as a fraction of the trace duration.
    pub start_min_ratio: f64,
    /// Upper window bound as a fraction of the trace duration.
    pub start_max_ratio: f64,
    /// Base RNG seed salted per (trace, lane) via [`seed_for_trace_lane`].
    pub random_seed: u64,
}

impl TStarSampler for WindowTStarSampler {
    fn sample_t_star(&self, trace_id: &str, lane: u64, duration_us: f64) -> f64 {
        // graph_ir_source.py:132-135 -- non-positive duration collapses to 0.
        if duration_us <= 0.0 {
            return 0.0;
        }
        let lo = self.start_min_ratio * duration_us;
        let hi = self.start_max_ratio * duration_us;
        // graph_ir_source.py:137-138 -- a collapsed window draws nothing.
        if hi <= lo {
            return lo.max(0.0);
        }
        let seed = seed_for_trace_lane(self.random_seed, trace_id, lane);
        let t = NumpyPcg64::from_u64_seed(seed).uniform(lo, hi);
        // graph_ir_source.py:_plan_trace clamps with max(t_star, 0.0).
        t.max(0.0)
    }
}

/// Derive a per-(trace, lane) RNG seed (agentx `_seed_for_trace_lane`).
///
/// Byte-exact port of `graph_ir_source.py:138-150`: SHA-256 the ASCII string
/// `"{base_seed}:{trace_id}:{lane}"` and take the low 8 bytes big-endian. This
/// keeps per-(trace, lane) `t*` values deterministic given `base_seed` yet
/// decorrelated across both traces and lanes.
pub fn seed_for_trace_lane(base_seed: u64, trace_id: &str, lane: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("{base_seed}:{trace_id}:{lane}").as_bytes());
    let digest = hasher.finalize();
    let mut low8 = [0u8; 8];
    low8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(low8)
}

/// Return the trace's intrinsic wall-clock span in microseconds.
///
/// Port of `graph/analysis/snapshot.py:65` (`trace_duration_us`): the largest
/// `arrival_offset_us` across the trace's node firings, `0.0` when no node
/// carries timing. The Python side reads the offset from each elaborated
/// firing; the Rust recorded-graph adapters store it as
/// `node.metadata["arrival_offset_us"]` (a u64; see
/// `graph/recorded/trie/mod.rs:190`). Every node in the resolved graph fires
/// exactly once in the static elaboration, so the max over nodes equals the max
/// over firings.
pub fn trace_duration_us(parsed: &ParsedGraph, trace: &TraceRecord) -> f64 {
    let graph = parsed.resolve_trace_graph(trace);
    graph
        .nodes
        .values()
        .filter_map(|node| node.metadata.get("arrival_offset_us"))
        .filter_map(|value| value.as_f64())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{GraphRecord, LlmNode};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn node_with_offset(offset: Option<u64>) -> LlmNode {
        let mut metadata = BTreeMap::new();
        if let Some(off) = offset {
            metadata.insert("arrival_offset_us".to_string(), json!(off));
        }
        LlmNode {
            output: "out".to_string(),
            streaming: true,
            inputs: Vec::new(),
            min_start_delay_us: None,
            max_tokens: None,
            items: Vec::new(),
            metadata,
        }
    }

    fn parsed_with_offsets(offsets: &[Option<u64>]) -> (ParsedGraph, TraceRecord) {
        let mut nodes = BTreeMap::new();
        for (i, off) in offsets.iter().enumerate() {
            nodes.insert(format!("n{i}"), node_with_offset(*off));
        }
        let graph = GraphRecord {
            nodes,
            ..Default::default()
        };
        let parsed = ParsedGraph {
            graph,
            ..Default::default()
        };
        let trace = TraceRecord {
            id: "trace-7".to_string(),
            graph_ref: None,
            initial_state: BTreeMap::new(),
        };
        (parsed, trace)
    }

    #[test]
    fn seed_matches_python_sha256_low8_be() {
        // python3 -c "import hashlib; print(int.from_bytes(
        //   hashlib.sha256(b'0:trace-7:0').digest()[:8],'big'))"
        assert_eq!(seed_for_trace_lane(0, "trace-7", 0), 5561269195474234662);
        // "42:abc:3"
        assert_eq!(seed_for_trace_lane(42, "abc", 3), 12694478397425876729);
    }

    #[test]
    fn trace_duration_is_max_offset_default_zero() {
        let (parsed, trace) = parsed_with_offsets(&[Some(0), Some(1_000_000), Some(500_000)]);
        assert_eq!(trace_duration_us(&parsed, &trace), 1_000_000.0);

        // No node carries timing -> 0.0.
        let (parsed, trace) = parsed_with_offsets(&[None, None]);
        assert_eq!(trace_duration_us(&parsed, &trace), 0.0);
    }

    #[test]
    fn nonpositive_duration_collapses_to_zero() {
        let sampler = WindowTStarSampler {
            start_min_ratio: 0.2,
            start_max_ratio: 0.8,
            random_seed: 0,
        };
        assert_eq!(sampler.sample_t_star("trace-7", 0, 0.0), 0.0);
        assert_eq!(sampler.sample_t_star("trace-7", 0, -5.0), 0.0);
    }

    #[test]
    fn collapsed_window_returns_lo_no_draw() {
        // Default [0, 0] window collapses to the exact instant lo.
        let sampler = WindowTStarSampler {
            start_min_ratio: 0.0,
            start_max_ratio: 0.0,
            random_seed: 0,
        };
        assert_eq!(sampler.sample_t_star("trace-7", 0, 1000.0), 0.0);

        // hi < lo also collapses to lo (no draw).
        let sampler = WindowTStarSampler {
            start_min_ratio: 0.5,
            start_max_ratio: 0.3,
            random_seed: 0,
        };
        assert_eq!(sampler.sample_t_star("trace-7", 0, 1000.0), 500.0);
    }

    #[test]
    fn window_draw_matches_numpy_reference() {
        // seed_for_trace_lane(0,"trace-7",0)=5561269195474234662; lo=200, hi=800.
        // python: float(np.random.default_rng(seed).uniform(200.0, 800.0))
        let sampler = WindowTStarSampler {
            start_min_ratio: 0.2,
            start_max_ratio: 0.8,
            random_seed: 0,
        };
        let t = sampler.sample_t_star("trace-7", 0, 1000.0);
        assert_eq!(t, 611.7386827809453);
    }
}
