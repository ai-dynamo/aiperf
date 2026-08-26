// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live cellular metrics heartbeats.
//!
//! A [`MetricsHeartbeat`] is a phase-progress-cadence live snapshot: monotonic
//! counters, concurrency saturation, and associatively-mergeable t-digest sketches
//! of the key latency distributions (TTFT / ITL / request latency). Live
//! percentiles are sketch-derived; final reports remain exact from record
//! partitions.
//!
//! [`HeartbeatAccumulator`] ingests per-record latency facts into the sketches; the
//! runner snapshots it on the phase-progress cadence, pairing the sketches with the
//! authoritative counts from the monotonic issuer. [`MetricsHeartbeat::merge`]
//! sums counters and merges t-digests across cells.

use serde::{Deserialize, Serialize};

use crate::cellular::sketch::TDigest;

/// Monotonic phase counters at a heartbeat tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatCounters {
    /// Requests issued (admitted) so far.
    pub issued: u64,
    /// Requests that reached a successful terminal.
    pub completed: u64,
    /// Requests that terminated in error or cancellation.
    pub errored: u64,
}

/// Concurrency saturation at a heartbeat tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatSaturation {
    /// Requests in flight (issued but not yet terminal).
    pub in_flight: u64,
    /// The configured concurrency limit the cell is admitting against.
    pub concurrency_limit: u64,
}

/// A live snapshot of the run: counters, saturation, and mergeable latency sketches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricsHeartbeat {
    /// Observation time in nanoseconds (Clock-derived; the cadence tick).
    pub observed_at_ns: i64,
    /// Monotonic counters (authoritative live counts from the issuer).
    pub counters: HeartbeatCounters,
    /// Concurrency saturation at the tick.
    pub saturation: HeartbeatSaturation,
    /// Time-to-first-token distribution, in milliseconds.
    pub ttft_ms: TDigest,
    /// Inter-token-latency distribution, in milliseconds.
    pub itl_ms: TDigest,
    /// Request-latency distribution, in milliseconds.
    pub latency_ms: TDigest,
}

impl MetricsHeartbeat {
    /// Folds another cell's heartbeat into this one: counters and saturation by
    /// sum, sketches by t-digest merge, observation time by max (the latest tick).
    /// Associative, so the controller can reduce any number of cell heartbeats.
    pub fn merge(&mut self, other: &MetricsHeartbeat) {
        self.observed_at_ns = self.observed_at_ns.max(other.observed_at_ns);
        self.counters.issued += other.counters.issued;
        self.counters.completed += other.counters.completed;
        self.counters.errored += other.counters.errored;
        self.saturation.in_flight += other.saturation.in_flight;
        self.saturation.concurrency_limit += other.saturation.concurrency_limit;
        self.ttft_ms.merge(&other.ttft_ms);
        self.itl_ms.merge(&other.itl_ms);
        self.latency_ms.merge(&other.latency_ms);
    }
}

/// Ingests per-record latency facts into the live t-digest sketches.
///
/// Counters and saturation are supplied by the caller at [`snapshot`](Self::snapshot)
/// time from the monotonic issuer/phase progress, so the accumulator owns only the
/// distribution sketches. Merging two accumulators (t-digest merge per metric) is
/// the in-process analogue of the cross-cell heartbeat merge.
#[derive(Clone, Debug, Default)]
pub struct HeartbeatAccumulator {
    ttft_ms: TDigest,
    itl_ms: TDigest,
    latency_ms: TDigest,
}

impl HeartbeatAccumulator {
    /// Builds an empty accumulator with default sketch compression.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one completed request's latency facts. Absent TTFT/latency and an
    /// empty ITL list contribute nothing; non-finite values are dropped by the
    /// sketch. All values are milliseconds.
    pub fn observe(
        &mut self,
        ttft_ms: Option<f64>,
        inter_token_ms: impl IntoIterator<Item = f64>,
        latency_ms: Option<f64>,
    ) {
        if let Some(ttft) = ttft_ms {
            self.ttft_ms.add(ttft);
        }
        self.itl_ms.extend_from(inter_token_ms);
        if let Some(latency) = latency_ms {
            self.latency_ms.add(latency);
        }
    }

    /// Snapshots the current sketches into a heartbeat, pairing them with the
    /// caller-supplied observation time, counters, and saturation.
    pub fn snapshot(
        &self,
        observed_at_ns: i64,
        counters: HeartbeatCounters,
        saturation: HeartbeatSaturation,
    ) -> MetricsHeartbeat {
        MetricsHeartbeat {
            observed_at_ns,
            counters,
            saturation,
            ttft_ms: self.ttft_ms.clone(),
            itl_ms: self.itl_ms.clone(),
            latency_ms: self.latency_ms.clone(),
        }
    }

    /// Merges another accumulator's sketches into this one (per-metric t-digest
    /// merge) — the in-process shard merge.
    pub fn merge(&mut self, other: &HeartbeatAccumulator) {
        self.ttft_ms.merge(&other.ttft_ms);
        self.itl_ms.merge(&other.itl_ms);
        self.latency_ms.merge(&other.latency_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::PERCENTILES;

    fn samples(count: usize, base: f64, spread: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                base + unit * spread
            })
            .collect()
    }

    #[test]
    fn snapshot_carries_counters_saturation_and_sketch_quantiles() {
        let latencies = samples(5_000, 40.0, 20.0, 1);
        let mut accumulator = HeartbeatAccumulator::new();
        for &latency in &latencies {
            accumulator.observe(
                Some(latency * 0.4),
                [latency * 0.1, latency * 0.1],
                Some(latency),
            );
        }
        let counters = HeartbeatCounters {
            issued: 5_000,
            completed: 4_900,
            errored: 3,
        };
        let saturation = HeartbeatSaturation {
            in_flight: 97,
            concurrency_limit: 128,
        };
        let heartbeat = accumulator.snapshot(1_700_000_000, counters, saturation);

        assert_eq!(heartbeat.counters, counters);
        assert_eq!(heartbeat.saturation, saturation);
        assert_eq!(heartbeat.observed_at_ns, 1_700_000_000);
        // Sketch medians land in the sampled bands.
        let p50 = heartbeat.latency_ms.quantile(0.5).unwrap();
        assert!((40.0..=60.0).contains(&p50), "latency p50 {p50}");
        assert_eq!(heartbeat.itl_ms.count(), 10_000);
    }

    #[test]
    fn merged_shard_heartbeats_match_a_single_shard_snapshot() {
        // The live merge: shards accumulate disjoint records, merge, and the sketch
        // quantiles match a single accumulator over the whole — the same associative
        // t-digest merge a controller applies to cross-cell heartbeats.
        let all = samples(9_000, 10.0, 90.0, 7);
        let mut whole = HeartbeatAccumulator::new();
        for &value in &all {
            whole.observe(Some(value), std::iter::empty(), Some(value * 2.0));
        }

        let mut shards = [
            HeartbeatAccumulator::new(),
            HeartbeatAccumulator::new(),
            HeartbeatAccumulator::new(),
        ];
        for (index, &value) in all.iter().enumerate() {
            shards[index % 3].observe(Some(value), std::iter::empty(), Some(value * 2.0));
        }
        let mut merged = HeartbeatAccumulator::new();
        for shard in &shards {
            merged.merge(shard);
        }

        let whole_snapshot = whole.snapshot(
            0,
            HeartbeatCounters::default(),
            HeartbeatSaturation::default(),
        );
        let merged_snapshot = merged.snapshot(
            0,
            HeartbeatCounters::default(),
            HeartbeatSaturation::default(),
        );
        for percentile in PERCENTILES {
            let q = percentile as f64 / 100.0;
            let a = whole_snapshot.latency_ms.quantile(q).unwrap();
            let b = merged_snapshot.latency_ms.quantile(q).unwrap();
            assert!(
                (a - b).abs() <= 180.0 * 0.02,
                "p{percentile}: {a:.2} vs {b:.2}"
            );
        }
    }

    #[test]
    fn heartbeat_merge_sums_counters_and_merges_sketches() {
        let mut left = HeartbeatAccumulator::new();
        left.observe(Some(10.0), [1.0], Some(20.0));
        let mut right = HeartbeatAccumulator::new();
        right.observe(Some(30.0), [3.0], Some(40.0));

        let mut a = left.snapshot(
            100,
            HeartbeatCounters {
                issued: 5,
                completed: 4,
                errored: 1,
            },
            HeartbeatSaturation {
                in_flight: 1,
                concurrency_limit: 8,
            },
        );
        let b = right.snapshot(
            200,
            HeartbeatCounters {
                issued: 7,
                completed: 6,
                errored: 0,
            },
            HeartbeatSaturation {
                in_flight: 2,
                concurrency_limit: 8,
            },
        );
        a.merge(&b);

        assert_eq!(a.observed_at_ns, 200);
        assert_eq!(
            a.counters,
            HeartbeatCounters {
                issued: 12,
                completed: 10,
                errored: 1
            }
        );
        assert_eq!(
            a.saturation,
            HeartbeatSaturation {
                in_flight: 3,
                concurrency_limit: 16
            }
        );
        assert_eq!(a.latency_ms.count(), 2);
        assert_eq!(a.latency_ms.min(), Some(20.0));
        assert_eq!(a.latency_ms.max(), Some(40.0));
    }

    #[test]
    fn heartbeat_serde_round_trips() {
        let mut accumulator = HeartbeatAccumulator::new();
        for value in samples(2_000, 5.0, 50.0, 3) {
            accumulator.observe(Some(value), [value * 0.2], Some(value));
        }
        let heartbeat = accumulator.snapshot(
            42,
            HeartbeatCounters {
                issued: 2_000,
                completed: 1_999,
                errored: 1,
            },
            HeartbeatSaturation {
                in_flight: 1,
                concurrency_limit: 64,
            },
        );
        let bytes = rmp_serde::to_vec(&heartbeat).expect("encode");
        let restored: MetricsHeartbeat = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(restored, heartbeat);
    }

    #[test]
    fn empty_sketch_heartbeat_round_trips_over_messagepack() {
        // An osl==1 run observes no inter-token gaps, so a cell ships an all-empty
        // itl_ms sketch whose min/max carry the +inf / -inf sentinels. MessagePack must
        // round-trip those sentinels because JSON cannot.
        let heartbeat = HeartbeatAccumulator::new().snapshot(
            0,
            HeartbeatCounters::default(),
            HeartbeatSaturation::default(),
        );
        assert_eq!(heartbeat.itl_ms.count(), 0);
        assert_eq!(heartbeat.itl_ms.min(), None);
        assert_eq!(heartbeat.itl_ms.max(), None);

        let bytes = rmp_serde::to_vec(&heartbeat).expect("encode");
        let restored: MetricsHeartbeat = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(restored, heartbeat);
        assert_eq!(restored.itl_ms.count(), 0);
        assert_eq!(restored.itl_ms.min(), None);
        assert_eq!(restored.itl_ms.max(), None);
    }
}
