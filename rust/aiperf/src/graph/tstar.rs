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

/// Derive the legacy agentx `ShuffleSampler` child RNG seed from the RUN root.
///
/// Byte-exact port of `_RNGManager.derive` (agentx
/// `common/random_generator.py:392-410`, the `sha256` low-8-big-endian child
/// seed) specialized to the label `ShuffleSampler` requests at
/// `dataset/dataset_samplers.py:76` (`rng.derive("dataset.sampler.shuffle")`):
/// SHA-256 the ASCII string `"{root_seed}:dataset.sampler.shuffle"` and take the
/// low 8 bytes big-endian. The argument is the RUN root seed
/// (`rng.init(config.random_seed)`), NOT `t_star_random_seed`.
pub fn legacy_shuffle_seed(root_seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("{root_seed}:dataset.sampler.shuffle").as_bytes());
    let digest = hasher.finalize();
    let mut low8 = [0u8; 8];
    low8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(low8)
}

/// Persistent-epoch shuffle state for one corpus size (the legacy sampler model).
///
/// agentx `ShuffleSampler` (`dataset/dataset_samplers.py:66`) shuffles
/// `arange(total)` in place at init (pass 0) with ONE generator, then re-shuffles
/// that SAME persistent generator each time the cursor wraps (pass 1, 2, ...).
/// This is a CONTINUOUS-STATE generator: pass `k` is `arange(total)` after
/// `k + 1` in-place `numpy Generator.shuffle` calls, NOT a fresh per-pass seed.
/// We keep the generator and the running array alive and snapshot each pass, so
/// producing pass `k` is O(total) amortized (one shuffle) rather than replaying
/// `k + 1` shuffles from scratch.
struct ShuffleEpochs {
    /// The single persistent generator (`np.random.default_rng(child_seed)`).
    generator: NumpyPcg64,
    /// The running array, mutated in place by each pass's shuffle.
    running: Vec<usize>,
    /// Snapshot of the array AFTER pass `k`'s shuffle (`passes[k]` == pass `k`).
    passes: Vec<Rc<Vec<usize>>>,
}

impl ShuffleEpochs {
    fn new(child_seed: u64, total: usize) -> Self {
        Self {
            generator: NumpyPcg64::from_u64_seed(child_seed),
            running: (0..total).collect(),
            passes: Vec::new(),
        }
    }

    /// Return the array after pass `pass_index`, advancing the generator only as
    /// far as needed (each additional pass is exactly one more in-place shuffle).
    fn pass(&mut self, pass_index: usize) -> Rc<Vec<usize>> {
        while self.passes.len() <= pass_index {
            self.generator.shuffle(&mut self.running);
            self.passes.push(Rc::new(self.running.clone()));
        }
        self.passes[pass_index].clone()
    }
}

/// Strategy-aware corpus-index remap shared by every graph recycle draw site.
///
/// Reproduces legacy agentx `ShuffleSampler`/`SequentialSampler`
/// (`dataset/dataset_samplers.py`) BYTE-EXACT: the SINGLE choke point every
/// cross-trace draw in the pressure lane fan-out, the pass-0 lane resolve, AND
/// the profiling recycle draw routes through, so `--dataset-sampling-strategy`
/// governs WHICH corpus template a freed lane serves without changing the draw
/// COUNTERS (only the counter -> index remap changes). `Sequential`
/// (`SequentialSampler`, the default) returns `x % total` unchanged;
/// `Shuffle`/`Random` return `epoch[x / total][x % total]` where each epoch is
/// the running array after one more in-place shuffle of the SAME persistent
/// generator (the continuous-state model — see [`ShuffleEpochs`]), seeded once
/// with [`legacy_shuffle_seed`]`(run_root)`.
///
/// The epoch snapshots are cached per `total` in a `RefCell` (single event-loop
/// mutation); the derivation is deterministic given `(base_seed, total)`, so two
/// instances with the same base seed produce identical draws regardless of call
/// order (the cache is a pure optimization).
///
/// Reused by both the runner's `PressureDraw` (pressure/pass-0 draws) and the
/// [`crate::graph::workload::CyclingGraphTraceSource`] /
/// `PartitionedGraphTraceSource` profiling recycle, so the profiling recycle
/// continues the SAME persistent-epoch order the pressure stage replays under (a
/// freed profiling lane never re-serves a template the pressure stage already
/// drew under a different order).
pub struct PermutationDraw {
    /// Whether the resolved strategy permutes (shuffle/random) vs. sequential.
    shuffled: bool,
    /// The legacy `ShuffleSampler` child seed ([`legacy_shuffle_seed`] of the run
    /// root), NOT `t_star_random_seed`.
    base_seed: u64,
    /// Per-`total` persistent-epoch shuffle state (mirrors the sampler's own
    /// single-generator, running-array state across wraps).
    cache: RefCell<HashMap<usize, ShuffleEpochs>>,
}

impl PermutationDraw {
    /// Build a draw for a resolved strategy: `shuffled` selects the persistent
    /// shuffle-epoch remap; `base_seed` is the legacy `ShuffleSampler` child seed.
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
    /// `total == 0` yields `0`; `SequentialSampler` returns `x % total`;
    /// `ShuffleSampler` returns `epoch[x / total][x % total]`.
    pub fn index(&self, x: u64, total: usize) -> usize {
        if total == 0 {
            return 0;
        }
        let total_u64 = total as u64;
        if !self.shuffled {
            return usize::try_from(x % total_u64).unwrap_or(0);
        }
        let pass_index = usize::try_from(x / total_u64).unwrap_or(0);
        let offset = usize::try_from(x % total_u64).unwrap_or(0);
        let mut cache = self.cache.borrow_mut();
        let epochs = cache
            .entry(total)
            .or_insert_with(|| ShuffleEpochs::new(self.base_seed, total));
        epochs.pass(pass_index)[offset]
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
    fn legacy_shuffle_seed_matches_python_sha256_low8_be() {
        // python3 -c "import hashlib; print(int.from_bytes(
        //   hashlib.sha256(b'0:dataset.sampler.shuffle').digest()[:8],'big'))"
        // These are exactly the `seed` fields of the committed golden vectors.
        assert_eq!(legacy_shuffle_seed(0), 5203359018791016587);
        assert_eq!(legacy_shuffle_seed(42), 7029856620319297634);
        assert_eq!(legacy_shuffle_seed(12345), 9928324691828912718);
    }

    #[test]
    fn legacy_shuffle_sampler_golden_vectors() {
        // Authoritative parity gate: for each committed vector, the persistent-
        // epoch `PermutationDraw` seeded with the legacy `ShuffleSampler` child
        // seed must reproduce the exact recycle-order `sequence` (pass 0, pass 1,
        // into pass 2) that agentx `ShuffleSampler` emits for the same run root.
        #[derive(serde::Deserialize)]
        struct Vector {
            root_seed: u64,
            n: usize,
            seed: u64,
            sequence: Vec<usize>,
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/legacy_shuffle_sampler_vectors.json"
        );
        let raw = std::fs::read_to_string(path).expect("read legacy shuffle vectors");
        let vectors: Vec<Vector> =
            serde_json::from_str(&raw).expect("parse legacy shuffle vectors");
        assert!(!vectors.is_empty(), "golden vectors must not be empty");
        for vector in &vectors {
            // The child-seed derivation itself is part of the parity contract.
            assert_eq!(
                legacy_shuffle_seed(vector.root_seed),
                vector.seed,
                "child seed for root {}",
                vector.root_seed
            );
            let draw = PermutationDraw::new(true, vector.seed);
            for (x, &expected) in vector.sequence.iter().enumerate() {
                assert_eq!(
                    draw.index(x as u64, vector.n),
                    expected,
                    "root {} n {} at draw {x}",
                    vector.root_seed,
                    vector.n
                );
            }
        }
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
    fn permutation_draw_shuffle_covers_each_pass_and_is_call_order_independent() {
        // Every full pass of the persistent-epoch shuffle covers each corpus index
        // exactly once (music-shuffle contract), and the draw is a pure function of
        // (base_seed, x, total): two instances agree regardless of the order in
        // which passes are requested (the incremental cache is only optimization).
        let draw = PermutationDraw::new(true, 5203359018791016587);
        let total = 8usize;
        for pass in 0u64..3 {
            let mut seen = Vec::new();
            for offset in 0..total as u64 {
                seen.push(draw.index(pass * total as u64 + offset, total));
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..total).collect::<Vec<_>>(), "pass {pass} coverage");
        }
        // A second instance drawing passes in reverse order agrees index-for-index.
        let reverse = PermutationDraw::new(true, 5203359018791016587);
        for pass in (0u64..3).rev() {
            for offset in 0..total as u64 {
                let x = pass * total as u64 + offset;
                assert_eq!(reverse.index(x, total), draw.index(x, total), "draw {x}");
            }
        }
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
