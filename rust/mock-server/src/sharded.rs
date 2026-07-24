// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-sharded Prometheus histogram.
//!
//! tikv `prometheus::Histogram::observe` updates a single shared `sample_sum`
//! (an `f64` packed in an `AtomicU64`, mutated with a compare-and-swap loop)
//! plus shared per-bucket atomics. When many worker threads observe one global
//! series — the mock's per-request vLLM/SGLang/TRT-LLM latency/queue/iteration
//! histograms are hit by every request on every worker — those atomics become a
//! cache-line-ping-pong bottleneck. Profiling under saturating load showed
//! `Histogram::observe` as the single largest CPU cost (~14% self-time at 16
//! workers), and it did not scale with added cores.
//!
//! [`ShardedHistogram`] gives each OS thread its own inner histogram shard, so
//! `observe` is contention-free, and merges all shards into one identical
//! Histogram exposition at scrape time. It is a drop-in for the hot global
//! histograms: it exposes the same `observe(f64)` method and implements
//! `Collector`, so registration and the emitted metric family are unchanged.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use prometheus::HistogramOpts;
use prometheus::Histogram;
use prometheus::core::{Collector, Desc};
use prometheus::proto::MetricFamily;

/// Global monotonic source of per-thread shard ids. Each thread claims one id
/// the first time it observes any `ShardedHistogram`, then reuses it for all of
/// them, so a thread always lands on the same shard (no cross-thread sharing).
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static SHARD_ID: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// This thread's shard index, modulo the shard count. Assigned lazily and cached.
#[inline]
fn thread_shard(shard_count: usize) -> usize {
    SHARD_ID.with(|c| {
        let mut id = c.get();
        if id == usize::MAX {
            id = NEXT_SHARD.fetch_add(1, Ordering::Relaxed);
            c.set(id);
        }
        id % shard_count
    })
}

/// A histogram sharded per thread to eliminate `observe` contention. Cloning
/// shares the underlying shards (each inner `Histogram` is `Arc`-backed), so the
/// register-a-clone pattern used by the metric structs keeps observing and
/// scraping the same data.
#[derive(Clone)]
pub struct ShardedHistogram {
    shards: Vec<Histogram>,
}

impl ShardedHistogram {
    /// Build a sharded histogram with one shard per CPU. Every shard carries the
    /// same name/help/buckets/const-labels as `opts`, so the merged exposition
    /// is identical to a single `Histogram::with_opts(opts)`.
    pub fn new(opts: HistogramOpts) -> Self {
        let n = num_cpus::get().max(1);
        let shards = (0..n)
            .map(|_| Histogram::with_opts(opts.clone()).expect("valid histogram opts"))
            .collect();
        Self { shards }
    }

    /// Fallible constructor mirroring `prometheus::Histogram::with_opts`, so the
    /// metric structs can swap `Histogram::with_opts(..).unwrap()` for
    /// `ShardedHistogram::with_opts(..).unwrap()` with no other call-site change.
    pub fn with_opts(opts: HistogramOpts) -> Result<Self, prometheus::Error> {
        let n = num_cpus::get().max(1);
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(Histogram::with_opts(opts.clone())?);
        }
        Ok(Self { shards })
    }

    /// Observe a value into this thread's shard. Contention-free on the hot path.
    #[inline]
    pub fn observe(&self, v: f64) {
        let n = self.shards.len();
        self.shards[thread_shard(n)].observe(v);
    }
}

impl Collector for ShardedHistogram {
    fn desc(&self) -> Vec<&Desc> {
        // All shards share one Desc (same name/help/const-labels); expose one.
        self.shards[0].desc()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        // Start from shard 0's exposition, then fold in the remaining shards'
        // bucket counts, sample sum, and sample count. Prometheus histogram
        // buckets are cumulative, and summing cumulative counts across disjoint
        // shards yields the correct merged cumulative counts.
        let mut families = self.shards[0].collect();
        {
            let family = &mut families[0];
            let metric = &mut family.mut_metric()[0];
            let merged = metric.mut_histogram();
            for shard in &self.shards[1..] {
                let shard_families = shard.collect();
                let shard_hist = shard_families[0].get_metric()[0].get_histogram();
                merged.set_sample_count(merged.get_sample_count() + shard_hist.get_sample_count());
                merged.set_sample_sum(merged.get_sample_sum() + shard_hist.get_sample_sum());
                let shard_buckets = shard_hist.get_bucket();
                for (i, bucket) in merged.mut_bucket().iter_mut().enumerate() {
                    bucket.set_cumulative_count(
                        bucket.get_cumulative_count() + shard_buckets[i].get_cumulative_count(),
                    );
                }
            }
        }
        families
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_exposition_matches_single_histogram() {
        let bounds = vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];
        let sh = ShardedHistogram::new(
            HistogramOpts::new("test_seconds", "help").buckets(bounds.clone()),
        );
        // Reference single histogram fed the identical sequence.
        let single = Histogram::with_opts(HistogramOpts::new("test_seconds", "help").buckets(bounds))
            .unwrap();
        let samples = [0.001, 0.02, 0.2, 0.7, 3.0, 12.0, 0.006, 0.5];
        for &v in &samples {
            sh.observe(v);
            single.observe(v);
        }

        let merged = &sh.collect()[0].get_metric()[0].get_histogram().clone();
        let expect = &single.collect()[0].get_metric()[0].get_histogram().clone();

        assert_eq!(merged.get_sample_count(), expect.get_sample_count());
        assert!((merged.get_sample_sum() - expect.get_sample_sum()).abs() < 1e-9);
        assert_eq!(merged.get_bucket().len(), expect.get_bucket().len());
        for (m, e) in merged.get_bucket().iter().zip(expect.get_bucket()) {
            assert_eq!(m.get_cumulative_count(), e.get_cumulative_count());
            assert_eq!(m.get_upper_bound(), e.get_upper_bound());
        }
    }
}
