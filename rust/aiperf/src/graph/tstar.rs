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

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

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

/// Derive a per-pass RNG seed for the shuffle/random dataset-sampling draw.
///
/// Byte-exact port of `graph_ir_replay.py:_seed_for_draw_pass` (lines 205-216,
/// branch `ajc/aiperf-graph-ir`): SHA-256 the ASCII string
/// `"{base_seed}:dataset-draw:{pass_index}"` and take the low 8 bytes
/// big-endian. This mirrors [`seed_for_trace_lane`]'s derivation so each recycle
/// pass re-permutes under a distinct-yet-deterministic seed drawn from the run's
/// `t_star_random_seed`: the same base seed + pass index always yields the same
/// permutation (cross-run reproducibility), while different passes decorrelate.
pub fn seed_for_draw_pass(base_seed: u64, pass_index: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("{base_seed}:dataset-draw:{pass_index}").as_bytes());
    let digest = hasher.finalize();
    let mut low8 = [0u8; 8];
    low8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(low8)
}

/// Return the seeded permutation of `range(total)` for one draw pass.
///
/// Byte-exact port of `graph_ir_replay.py:_draw_permutation` (lines 837-855,
/// branch `ajc/aiperf-graph-ir`): a pass-salted numpy RNG
/// ([`seed_for_draw_pass`] -> `np.random.default_rng` -> in-place Fisher-Yates
/// `shuffle`, reproduced by [`NumpyPcg64::permutation`]) permutes `range(total)`
/// without replacement. Each pass of `total` draws covers every index exactly
/// once, then a fresh seeded permutation begins — the music-shuffle contract the
/// conversation-plane `ShuffleSampler` provides. Callers cache the result per
/// `(total, pass_index)`; the derivation is pure, so caching is a pure
/// optimization (the permutation is identical whether cached or recomputed).
pub fn draw_permutation(base_seed: u64, pass_index: u64, total: usize) -> Vec<usize> {
    NumpyPcg64::from_u64_seed(seed_for_draw_pass(base_seed, pass_index)).permutation(total)
}

/// Strategy-aware corpus-index remap shared by every graph recycle draw site.
///
/// Faithful port of `graph_ir_replay.py:_draw_index`/`_draw_permutation`
/// (lines 792-855, branch `ajc/aiperf-graph-ir`): the SINGLE choke point every
/// cross-trace draw in the pressure lane fan-out, the pass-0 lane resolve, AND
/// the profiling recycle draw routes through, so `--dataset-sampling-strategy`
/// governs WHICH corpus template a freed lane serves without changing the draw
/// COUNTERS (only the counter -> index remap changes). `Sequential` (the
/// default) returns `x % total` unchanged; `Shuffle`/`Random` map `x` to
/// `perm[pass][x % total]` where `pass = x / total`, each pass drawing a
/// distinct seeded permutation ([`draw_permutation`]).
///
/// The permutation is cached per `(total, pass_index)` in a `RefCell` (single
/// event-loop mutation, mirroring Python's per-instance `_draw_perm_cache`); the
/// cache is a pure optimization since [`draw_permutation`] is deterministic.
///
/// Reused by both the runner's `PressureDraw` (pressure/pass-0 draws) and the
/// [`crate::graph::workload::CyclingGraphTraceSource`] /
/// `PartitionedGraphTraceSource` profiling recycle, so the profiling recycle
/// continues the SAME per-pass permutation contract the pressure stage replays
/// under (a freed profiling lane never re-serves a template the pressure stage
/// already drew under a different order).
pub struct PermutationDraw {
    /// Whether the resolved strategy permutes (shuffle/random) vs. sequential.
    shuffled: bool,
    /// Base seed for [`seed_for_draw_pass`] (the run's `t_star_random_seed`).
    base_seed: u64,
    /// Per-`(total, pass_index)` permutation cache (`_draw_perm_cache`).
    cache: RefCell<HashMap<(usize, u64), Rc<Vec<usize>>>>,
}

impl PermutationDraw {
    /// Build a draw for a resolved strategy: `shuffled` selects the per-pass
    /// permutation remap; `base_seed` salts each pass's permutation seed.
    pub fn new(shuffled: bool, base_seed: u64) -> Self {
        Self {
            shuffled,
            base_seed,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// The byte-unchanged sequential draw (`x % total`); no permutation, no seed.
    pub fn sequential() -> Self {
        Self::new(false, 0)
    }

    /// Remap draw counter `x` to a corpus index in `[0, total)`.
    ///
    /// `graph_ir_replay.py:_draw_index`: `total <= 0` yields `0`; sequential
    /// returns `x % total`; shuffle returns `perm[x // total][x % total]`.
    pub fn index(&self, x: u64, total: usize) -> usize {
        if total == 0 {
            return 0;
        }
        let total_u64 = total as u64;
        if !self.shuffled {
            return usize::try_from(x % total_u64).unwrap_or(0);
        }
        let pass_index = x / total_u64;
        let offset = usize::try_from(x % total_u64).unwrap_or(0);
        self.permutation(pass_index, total)[offset]
    }

    /// Return the cached seeded permutation of `range(total)` for a draw pass,
    /// building it once per `(total, pass_index)` (`_draw_permutation`).
    fn permutation(&self, pass_index: u64, total: usize) -> Rc<Vec<usize>> {
        let key = (total, pass_index);
        if let Some(cached) = self.cache.borrow().get(&key) {
            return cached.clone();
        }
        let perm = Rc::new(draw_permutation(self.base_seed, pass_index, total));
        self.cache.borrow_mut().insert(key, perm.clone());
        perm
    }
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
    fn draw_pass_seed_matches_python_sha256_low8_be() {
        // python3 -c "import hashlib; print(int.from_bytes(
        //   hashlib.sha256(b'0:dataset-draw:0').digest()[:8],'big'))"
        assert_eq!(seed_for_draw_pass(0, 0), 14221486954297044610);
        assert_eq!(seed_for_draw_pass(0, 1), 10278907799327951431);
        assert_eq!(seed_for_draw_pass(42, 3), 991418308715691445);
    }

    #[test]
    fn draw_permutation_matches_numpy_and_covers_every_index_once() {
        // python: list(np.random.default_rng(_seed_for_draw_pass(0, p)).permutation(5))
        assert_eq!(draw_permutation(0, 0, 5), vec![4, 3, 0, 2, 1]);
        assert_eq!(draw_permutation(0, 1, 5), vec![1, 2, 0, 4, 3]);
        // Distinct per-pass seed => pass 1 differs from pass 0.
        assert_ne!(draw_permutation(0, 0, 5), draw_permutation(0, 1, 5));
        // A full pass is a permutation: every index in [0, total) exactly once.
        let mut sorted = draw_permutation(0, 0, 5);
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn permutation_draw_sequential_is_modulo_wrap() {
        let draw = PermutationDraw::sequential();
        for x in 0u64..20 {
            for total in 1usize..7 {
                assert_eq!(draw.index(x, total), (x % total as u64) as usize);
            }
        }
        // Degenerate empty corpus never panics.
        assert_eq!(draw.index(5, 0), 0);
    }

    #[test]
    fn permutation_draw_shuffle_matches_seeded_permutation_each_pass() {
        // Shuffle: index(x) == draw_permutation(base, x/total, total)[x%total],
        // each pass covers every index exactly once, and distinct passes differ.
        let draw = PermutationDraw::new(true, 0);
        let total = 5usize;
        for pass in 0u64..2 {
            let expected = draw_permutation(0, pass, total);
            let mut seen = Vec::new();
            for offset in 0..total as u64 {
                let x = pass * total as u64 + offset;
                assert_eq!(draw.index(x, total), expected[offset as usize]);
                seen.push(draw.index(x, total));
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..total).collect::<Vec<_>>());
        }
        assert_ne!(draw.index(0, total), draw.index(total as u64, total));
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
