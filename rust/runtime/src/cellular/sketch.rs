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
