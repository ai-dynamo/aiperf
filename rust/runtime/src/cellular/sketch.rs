// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, mergeable t-digest quantile sketch.
//!
//! The cellular runtime's live lane reports **sketch-derived** percentiles; the
//! final report stays exact from record partitions.
//!
//! This is the "merging" t-digest (Dunning): values become weight-1 centroids;
//! [`compress`](TDigest::compress) sorts by mean and greedily clusters adjacent
//! centroids while each cluster spans at most one unit of the K1 scale function
//! `k(q) = compression·asin(2q−1)/2π`, which keeps clusters small (fine resolution)
//! at the tails and large in the body. Because compression sorts, the result is
//! order-independent up to floating point, and [`merge`](TDigest::merge) is just
//! "concatenate centroids, compress" — associative and deterministic at a fixed
//! topology. Quantiles interpolate centroid means by cumulative quantile, anchored
//! at the exact `min`/`max`.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Default compression (δ) — the standard t-digest default. Larger keeps more
/// centroids: finer quantiles, more bytes on the wire (~`δ/2` centroids after
/// compression). At δ=100 live percentiles track the exact report to well under a
/// percent on the broad distributions (TTFT, latency) and within a few percent at
/// the extreme tail of tight ones — expected for an approximate live sketch.
pub const DEFAULT_COMPRESSION: f64 = 100.0;

/// A cluster of ingested values summarized by their mean and total weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Centroid {
    mean: f64,
    weight: f64,
}

/// A deterministic, associatively-mergeable t-digest for streaming quantiles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TDigest {
    /// Centroids, sorted by mean after every [`compress`](Self::compress). May hold
    /// transient unsorted weight-1 centroids between compressions.
    centroids: Vec<Centroid>,
    /// Sum of all centroid weights (the ingested value count).
    total_weight: f64,
    /// Exact minimum ingested value (`+inf` when empty), anchoring quantile 0.
    min: f64,
    /// Exact maximum ingested value (`-inf` when empty), anchoring quantile 1.
    max: f64,
    /// Compression parameter δ.
    compression: f64,
    /// Uncompressed-centroid count that triggers an eager compress, bounding memory
    /// deterministically (a fixed function of `compression`).
    compress_threshold: usize,
}

impl TDigest {
    /// Builds an empty digest with the [default compression](DEFAULT_COMPRESSION).
    pub fn new() -> Self {
        Self::with_compression(DEFAULT_COMPRESSION)
    }

    /// Builds an empty digest with an explicit compression δ (clamped to `>= 1`).
    pub fn with_compression(compression: f64) -> Self {
        let compression = compression.max(1.0);
        Self {
            centroids: Vec::new(),
            total_weight: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            compression,
            // Bound transient centroids to a small multiple of the compressed size.
            compress_threshold: (compression as usize).saturating_mul(10).max(64),
        }
    }

    /// Ingests one finite value. Non-finite values are ignored (a distribution
    /// sketch never stores NaN/±inf sentinels).
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.centroids.push(Centroid {
            mean: value,
            weight: 1.0,
        });
        self.total_weight += 1.0;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        if self.centroids.len() > self.compress_threshold {
            self.compress();
        }
    }

    /// Ingests every finite value in `values`.
    pub fn extend_from(&mut self, values: impl IntoIterator<Item = f64>) {
        for value in values {
            self.add(value);
        }
    }

    /// The number of ingested values.
    pub fn count(&self) -> u64 {
        self.total_weight as u64
    }

    /// Whether no value has been ingested.
    pub fn is_empty(&self) -> bool {
        self.total_weight == 0.0
    }

    /// The exact minimum ingested value, or `None` when empty.
    pub fn min(&self) -> Option<f64> {
        (!self.is_empty()).then_some(self.min)
    }

    /// The exact maximum ingested value, or `None` when empty.
    pub fn max(&self) -> Option<f64> {
        (!self.is_empty()).then_some(self.max)
    }

    /// Merges `other` into `self`: concatenate centroids, then compress. The min and
    /// max stay exact. Associative and deterministic at a fixed topology.
    pub fn merge(&mut self, other: &TDigest) {
        if other.is_empty() {
            return;
        }
        self.centroids.extend_from_slice(&other.centroids);
        self.total_weight += other.total_weight;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.compress();
    }

    /// Estimates the value at quantile `q` (clamped to `[0, 1]`), or `None` when
    /// empty. `q = 0` returns the exact min and `q = 1` the exact max; interior
    /// quantiles linearly interpolate centroid means by cumulative quantile. For
    /// several quantiles of one snapshot prefer [`quantiles`](Self::quantiles),
    /// which clusters once instead of per call.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.quantile_from(&self.clustered(), q))
    }

    /// Estimates several quantiles from one clustering, returning `None` per entry
    /// when empty. Clusters once rather than per quantile — the projection path a
    /// heartbeat snapshot takes over the fixed percentile band.
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<Option<f64>> {
        if self.is_empty() {
            return vec![None; quantiles.len()];
        }
        let centroids = self.clustered();
        quantiles
            .iter()
            .map(|&q| Some(self.quantile_from(&centroids, q)))
            .collect()
    }

    /// Interpolates one quantile over already-clustered centroids. Anchor points in
    /// (quantile, value) space are `(0, min)`, each centroid at its center quantile,
    /// and `(1, max)`; the value interpolates linearly between the bracketing pair.
    fn quantile_from(&self, centroids: &[Centroid], q: f64) -> f64 {
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.min;
        }
        if q >= 1.0 {
            return self.max;
        }
        let total = self.total_weight;
        let mut cumulative = 0.0;
        let mut prev_q = 0.0;
        let mut prev_value = self.min;
        for centroid in centroids {
            let center_q = (cumulative + centroid.weight / 2.0) / total;
            if q < center_q {
                return interpolate(prev_q, prev_value, center_q, centroid.mean, q);
            }
            prev_q = center_q;
            prev_value = centroid.mean;
            cumulative += centroid.weight;
        }
        interpolate(prev_q, prev_value, 1.0, self.max, q)
    }

    /// Compacts centroids in place: sort by mean, then greedily cluster adjacent
    /// centroids while each cluster spans at most one K1-scale unit.
    pub fn compress(&mut self) {
        self.centroids = self.clustered();
    }

    fn clustered(&self) -> Vec<Centroid> {
        if self.centroids.len() <= 1 {
            return self.centroids.clone();
        }
        let mut sorted = self.centroids.clone();
        sorted.sort_unstable_by(|a, b| a.mean.total_cmp(&b.mean));

        let total = self.total_weight;
        let mut clustered: Vec<Centroid> = Vec::new();
        let mut cumulative_before = 0.0;
        let mut current = sorted[0];
        for centroid in &sorted[1..] {
            let q_start = cumulative_before / total;
            let proposed_weight = current.weight + centroid.weight;
            let q_end = (cumulative_before + proposed_weight) / total;
            if k_scale(q_end, self.compression) - k_scale(q_start, self.compression) <= 1.0 {
                // Merge `centroid` into the current cluster (weighted mean).
                let combined = current.weight + centroid.weight;
                current.mean =
                    (current.mean * current.weight + centroid.mean * centroid.weight) / combined;
                current.weight = combined;
            } else {
                clustered.push(current);
                cumulative_before += current.weight;
                current = *centroid;
            }
        }
        clustered.push(current);
        clustered
    }
}

impl Default for TDigest {
    fn default() -> Self {
        Self::new()
    }
}

/// The K1 scale function `k(q) = compression·asin(2q−1)/2π`, mapping a quantile to
/// scale space where a cluster may span at most one unit.
fn k_scale(q: f64, compression: f64) -> f64 {
    compression * (2.0 * q - 1.0).clamp(-1.0, 1.0).asin() / (2.0 * PI)
}

/// Linear interpolation of `value` at `q` between the brackets `(q0, v0)` and
/// `(q1, v1)`. Falls back to `v0` when the bracket is degenerate.
fn interpolate(q0: f64, v0: f64, q1: f64, v1: f64, q: f64) -> f64 {
    if q1 <= q0 {
        return v0;
    }
    v0 + (v1 - v0) * (q - q0) / (q1 - q0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report's exact type-7 linear-interpolation percentile (`kernel.rs`).
    fn exact_percentile(sorted: &[f64], percentile: f64) -> f64 {
        let count = sorted.len();
        let virtual_idx = percentile / 100.0 * (count - 1) as f64;
        let lo = virtual_idx.floor() as usize;
        let hi = (lo + 1).min(count - 1);
        let frac = virtual_idx - lo as f64;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }

    /// Deterministic pseudo-random values (a small LCG — no `rand`, no wall clock)
    /// over `[0, range)`, so tests are reproducible.
    fn samples(count: usize, range: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                unit * range
            })
            .collect()
    }

    #[test]
    fn empty_and_single_value() {
        let mut digest = TDigest::new();
        assert!(digest.is_empty());
        assert_eq!(digest.quantile(0.5), None);
        digest.add(7.5);
        assert_eq!(digest.count(), 1);
        assert_eq!(digest.quantile(0.0), Some(7.5));
        assert_eq!(digest.quantile(0.5), Some(7.5));
        assert_eq!(digest.quantile(1.0), Some(7.5));
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let mut digest = TDigest::new();
        digest.extend_from([1.0, f64::NAN, 2.0, f64::INFINITY, 3.0, f64::NEG_INFINITY]);
        assert_eq!(digest.count(), 3);
        assert_eq!(digest.min(), Some(1.0));
        assert_eq!(digest.max(), Some(3.0));
    }

    #[test]
    fn quantiles_converge_to_the_exact_report_percentiles() {
        let mut values = samples(20_000, 1000.0, 0xC0FFEE);
        let mut digest = TDigest::new();
        digest.extend_from(values.iter().copied());
        values.sort_by(f64::total_cmp);

        // t-digest is tail-accurate; assert each report percentile is within a small
        // fraction of the value range of the exact linear-interpolation percentile.
        let tolerance = 1000.0 * 0.01; // 1% of range
        for percentile in crate::metrics_core::PERCENTILES {
            let exact = exact_percentile(&values, percentile as f64);
            let sketch = digest.quantile(percentile as f64 / 100.0).unwrap();
            assert!(
                (sketch - exact).abs() <= tolerance,
                "p{percentile}: sketch {sketch:.3} vs exact {exact:.3} exceeds {tolerance}"
            );
        }
        assert_eq!(digest.min(), Some(values[0]));
        assert_eq!(digest.max(), Some(*values.last().unwrap()));
    }

    #[test]
    fn merge_matches_a_single_digest_of_the_whole() {
        let all = samples(12_000, 500.0, 0xABCDEF);
        let mut whole = TDigest::new();
        whole.extend_from(all.iter().copied());

        // Split across three shards, merge — quantiles must match the single digest
        // closely (both are sketches of the same data).
        let mut shards = [TDigest::new(), TDigest::new(), TDigest::new()];
        for (index, &value) in all.iter().enumerate() {
            shards[index % 3].add(value);
        }
        let mut merged = TDigest::new();
        for shard in &shards {
            merged.merge(shard);
        }

        assert_eq!(merged.count(), whole.count());
        assert_eq!(merged.min(), whole.min());
        assert_eq!(merged.max(), whole.max());
        for percentile in crate::metrics_core::PERCENTILES {
            let q = percentile as f64 / 100.0;
            let a = whole.quantile(q).unwrap();
            let b = merged.quantile(q).unwrap();
            assert!(
                (a - b).abs() <= 500.0 * 0.02,
                "p{percentile}: whole {a:.3} vs merged {b:.3}"
            );
        }
    }

    #[test]
    fn merge_is_deterministic_regardless_of_shard_order() {
        let all = samples(6_000, 100.0, 42);
        let mut shards = [TDigest::new(), TDigest::new()];
        for (index, &value) in all.iter().enumerate() {
            shards[index % 2].add(value);
        }
        let (a, b) = (shards[0].clone(), shards[1].clone());

        let mut forward = a.clone();
        forward.merge(&b);
        let mut backward = b.clone();
        backward.merge(&a);
        // Both compress the same centroid set (sorted), so results are identical.
        forward.compress();
        backward.compress();
        for percentile in crate::metrics_core::PERCENTILES {
            let q = percentile as f64 / 100.0;
            assert_eq!(forward.quantile(q), backward.quantile(q));
        }
    }

    #[test]
    fn serde_round_trip_preserves_quantiles() {
        let mut digest = TDigest::new();
        digest.extend_from(samples(8_000, 250.0, 7));
        digest.compress();

        let bytes = rmp_serde::to_vec(&digest).expect("encode");
        let restored: TDigest = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(restored, digest);
        for percentile in crate::metrics_core::PERCENTILES {
            let q = percentile as f64 / 100.0;
            assert_eq!(restored.quantile(q), digest.quantile(q));
        }
    }

    /// Emit, or verify, the cross-language golden fixture for the TypeScript port.
    ///
    /// `apps/aiperf-flow` carries a hand-written port of this digest so an explainer page can show
    /// real numbers rather than illustrative ones. A port with no pin against the original is a
    /// liability: it can drift silently and then teach something false with total confidence.
    ///
    /// This test owns the Rust side of that pin. It replays the inputs recorded in the committed
    /// fixture and asserts this implementation still produces the recorded outputs, so changing
    /// the algorithm fails here until the fixture is regenerated with `UPDATE_SKETCH_GOLDEN=1`.
    /// The TypeScript test performs the identical replay, which is what makes the two comparable.
    ///
    /// Only outputs are compared, never the input arrays: whether a 5000-element `f64` array
    /// round-trips through JSON byte-for-byte is a serde question, not a t-digest one.
    #[test]
    fn tdigest_golden_fixture_matches_this_implementation() {
        use std::path::PathBuf;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/parity/sketch_golden/tdigest.json");

        let quantile_band = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0];

        // Inputs come from the fixture when it exists so both languages replay the same values;
        // the generator seeds them the first time.
        let generate = std::env::var("UPDATE_SKETCH_GOLDEN").is_ok();
        let (broad, cells, tiny) = if generate {
            (
                samples(5_000, 1_000.0, 7),
                (0..3)
                    .map(|cell| samples(1_500, 1_000.0, 100 + cell as u64))
                    .collect::<Vec<Vec<f64>>>(),
                vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0],
            )
        } else {
            let committed: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&path)
                    .expect("sketch golden fixture missing — run with UPDATE_SKETCH_GOLDEN=1"),
            )
            .expect("fixture parses");
            let floats = |value: &serde_json::Value| -> Vec<f64> {
                value
                    .as_array()
                    .expect("array")
                    .iter()
                    .map(|v| v.as_f64().expect("f64"))
                    .collect()
            };
            (
                floats(&committed["cases"]["broad"]["input"]),
                committed["cases"]["folded"]["cells"]
                    .as_array()
                    .expect("cells")
                    .iter()
                    .map(floats)
                    .collect(),
                floats(&committed["cases"]["tiny"]["input"]),
            )
        };

        let digest_of = |values: &[f64]| {
            let mut digest = TDigest::new();
            digest.extend_from(values.iter().copied());
            digest.compress();
            digest
        };

        let broad_digest = digest_of(&broad);
        let mut folded = TDigest::new();
        for slice in &cells {
            folded.merge(&digest_of(slice));
        }
        let tiny_digest = digest_of(&tiny);

        let describe = |digest: &TDigest| {
            serde_json::json!({
                "count": digest.count(),
                "min": digest.min(),
                "max": digest.max(),
                "centroid_count": digest.centroids.len(),
                "centroid_means": digest.centroids.iter().map(|c| c.mean).collect::<Vec<_>>(),
                "centroid_weights": digest.centroids.iter().map(|c| c.weight).collect::<Vec<_>>(),
                "quantiles": quantile_band
                    .iter()
                    .map(|&q| digest.quantile(q))
                    .collect::<Vec<_>>(),
            })
        };

        if generate {
            let fixture = serde_json::json!({
                "compression": DEFAULT_COMPRESSION,
                "quantile_band": quantile_band,
                "cases": {
                    "broad": { "input": broad, "digest": describe(&broad_digest) },
                    "folded": { "cells": cells, "digest": describe(&folded) },
                    "tiny": { "input": tiny, "digest": describe(&tiny_digest) },
                },
            });
            std::fs::create_dir_all(path.parent().expect("fixture dir")).expect("create dir");
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&fixture).expect("serialize"),
            )
            .expect("write fixture");
            return;
        }

        let committed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");

        // Numeric comparison rather than `Value` equality: the inputs are replayed through JSON,
        // and whether a 5000-element f64 array survives that byte-for-byte is a serde property,
        // not a t-digest one. A 1e-9 relative tolerance is far tighter than any behaviour change
        // this test exists to catch, and is the same bound the TypeScript side uses.
        fn close(a: f64, b: f64) -> bool {
            // Finiteness first: a relative bound against a non-finite value degenerates, since
            // `1e-9 * INFINITY` is `INFINITY` and `INFINITY <= INFINITY` holds. A digest that
            // lost its min would otherwise compare equal to one that kept it.
            if !a.is_finite() || !b.is_finite() {
                return a == b;
            }
            (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
        }
        fn assert_field(expected: &serde_json::Value, actual: &serde_json::Value, what: &str) {
            match (expected, actual) {
                (serde_json::Value::Array(want), serde_json::Value::Array(got)) => {
                    assert_eq!(want.len(), got.len(), "{what}: length");
                    for (i, (w, g)) in want.iter().zip(got).enumerate() {
                        assert_field(w, g, &format!("{what}[{i}]"));
                    }
                }
                (serde_json::Value::Null, serde_json::Value::Null) => {}
                _ => {
                    let want = expected
                        .as_f64()
                        .unwrap_or_else(|| panic!("{what}: not a number"));
                    let got = actual
                        .as_f64()
                        .unwrap_or_else(|| panic!("{what}: not a number"));
                    assert!(close(want, got), "{what}: expected {want}, got {got}");
                }
            }
        }

        for (case, digest) in [
            ("broad", &broad_digest),
            ("folded", &folded),
            ("tiny", &tiny_digest),
        ] {
            let want = &committed["cases"][case]["digest"];
            let got = describe(digest);
            for field in [
                "count",
                "min",
                "max",
                "centroid_count",
                "centroid_means",
                "centroid_weights",
                "quantiles",
            ] {
                assert_field(&want[field], &got[field], &format!("{case}.{field}"));
            }
        }
    }

    #[test]
    fn centroid_count_stays_bounded_by_compression() {
        let mut digest = TDigest::with_compression(100.0);
        digest.extend_from(samples(100_000, 1000.0, 99));
        digest.compress();
        // ~compression/2 centroids after compression, well under the raw count.
        assert!(
            digest.centroids.len() < 200,
            "compressed to {} centroids",
            digest.centroids.len()
        );
    }
}
