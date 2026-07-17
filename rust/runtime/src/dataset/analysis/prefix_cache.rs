// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ideal (unbounded) prefix-cache reuse analysis for the `--dry-run` dataset
//! report.
//!
//! Block identifiers are prefix-dependent (chained): an identical id means the
//! whole prefix up to it matched, so per-block set membership is the exact
//! reuse test — this matches vLLM chained-hash automatic prefix caching and
//! SGLang radix longest-prefix reuse. A block is a cache hit when that exact id
//! was seen in an *earlier* request. Reuse is classified as intra- or
//! cross-conversation by the conversation that first introduced the id.
//!
//! This models an ideal cache with no eviction (unbounded capacity); realized
//! LRU behavior is a separate concern.

use std::collections::{BTreeMap, HashMap, HashSet};

/// How the block identifiers feeding [`ideal_reuse`] were derived. Serialized as
/// snake_case for report emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentitySource {
    /// Ids come directly from precomputed content hashes.
    HashIds,
    /// Ids were derived by hashing token blocks.
    TokenBlocks,
    /// Ids were derived from sequence-length structure only (no content).
    LengthStructure,
}

/// The block-id sequence for a single request, tagged with its conversation and
/// turn so reuse can be classified as intra- or cross-conversation.
#[derive(Debug, Clone)]
pub struct RequestBlocks {
    /// Conversation the request belongs to.
    pub conversation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Prefix-dependent (chained) block identifiers for this request, ordered
    /// from the root of the prefix outward.
    pub block_ids: Vec<i64>,
}

/// Aggregate ideal prefix-cache reuse statistics over a set of requests.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct IdealReuse {
    /// Total block occurrences across all requests.
    pub total_blocks: u64,
    /// Number of block occurrences that were cache hits (id seen earlier).
    pub cached_blocks: u64,
    /// `cached_blocks / total_blocks`, or `0.0` when there are no blocks.
    pub hit_rate: f64,
    /// Number of distinct block ids observed.
    pub unique_blocks: u64,
    /// Number of distinct first-block (root) ids across requests.
    pub unique_roots: u64,
    /// Cache hits whose id was first introduced by the same conversation.
    pub intra_conversation_cached: u64,
    /// Cache hits whose id was first introduced by a different conversation.
    pub cross_conversation_cached: u64,
    /// Count of requests whose per-request shared-prefix rate
    /// (`cached_blocks / total_blocks`) is at least the keyed threshold percent.
    /// Keys are `"25"`, `"50"`, `"75"`, `"90"`, `"100"`.
    pub shared_prefix_rate_ge: BTreeMap<String, u64>,
}

/// Compute ideal (unbounded, no-eviction) prefix-cache reuse over `requests`,
/// processed in the given order.
///
/// A block id is a hit if that exact id appeared in an earlier request. The
/// conversation that first introduced an id determines whether later reuse is
/// intra- or cross-conversation. `unique_roots` counts distinct first-block ids;
/// `shared_prefix_rate_ge` buckets each request by its cached-block fraction.
pub fn ideal_reuse(requests: &[RequestBlocks]) -> IdealReuse {
    // Block id -> conversation that first introduced it.
    let mut seen: HashMap<i64, String> = HashMap::new();
    let mut roots: HashSet<i64> = HashSet::new();

    let mut total_blocks: u64 = 0;
    let mut cached_blocks: u64 = 0;
    let mut intra_conversation_cached: u64 = 0;
    let mut cross_conversation_cached: u64 = 0;

    let thresholds: [(&str, f64); 5] = [
        ("25", 0.25),
        ("50", 0.50),
        ("75", 0.75),
        ("90", 0.90),
        ("100", 1.0),
    ];
    let mut shared_prefix_rate_ge: BTreeMap<String, u64> = BTreeMap::new();
    for (key, _) in thresholds {
        shared_prefix_rate_ge.insert(key.to_string(), 0);
    }

    for req in requests {
        if let Some(&first) = req.block_ids.first() {
            roots.insert(first);
        }

        let mut req_cached: u64 = 0;
        let req_total = req.block_ids.len() as u64;

        for &id in &req.block_ids {
            total_blocks += 1;
            match seen.get(&id) {
                Some(first_conversation) => {
                    cached_blocks += 1;
                    req_cached += 1;
                    if first_conversation == &req.conversation_id {
                        intra_conversation_cached += 1;
                    } else {
                        cross_conversation_cached += 1;
                    }
                }
                None => {
                    seen.insert(id, req.conversation_id.clone());
                }
            }
        }

        if req_total > 0 {
            let rate = req_cached as f64 / req_total as f64;
            for (key, threshold) in thresholds {
                if rate >= threshold {
                    *shared_prefix_rate_ge.get_mut(key).expect("seeded key") += 1;
                }
            }
        }
    }

    let hit_rate = if total_blocks > 0 {
        cached_blocks as f64 / total_blocks as f64
    } else {
        0.0
    };

    IdealReuse {
        total_blocks,
        cached_blocks,
        hit_rate,
        unique_blocks: seen.len() as u64,
        unique_roots: roots.len() as u64,
        intra_conversation_cached,
        cross_conversation_cached,
        shared_prefix_rate_ge,
    }
}

/// A single point on the realized cache hit-rate curve: the hit rate and
/// eviction count achieved by a finite-capacity LRU of `capacity_blocks`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CacheCurvePoint {
    /// LRU capacity in blocks that produced this point.
    pub capacity_blocks: u64,
    /// `hits / total_blocks`, or `0.0` when there are no blocks.
    pub hit_rate: f64,
    /// Number of block evictions performed to stay within capacity.
    pub evictions: u64,
}

/// Simulate a block-granular LRU cache of `capacity_blocks` over `requests`
/// processed in the given (arrival) order.
///
/// A block is a hit iff it is currently resident; a hit moves it to the MRU
/// position; a miss inserts it, evicting the least-recently-used block first
/// when the cache is full. Recency is tracked by a monotonically increasing
/// tick stored per resident block; eviction finds the minimum-tick entry by a
/// linear scan, so eviction is O(capacity) per miss. This is acceptable for the
/// modest capacities used by [`realized_sweep`].
pub fn realized_reuse(
    requests_in_arrival_order: &[RequestBlocks],
    capacity_blocks: u64,
) -> CacheCurvePoint {
    // Resident block id -> last-use tick.
    let mut resident: HashMap<i64, u64> = HashMap::new();
    let mut tick: u64 = 0;
    let mut total_blocks: u64 = 0;
    let mut hits: u64 = 0;
    let mut evictions: u64 = 0;

    let capacity = capacity_blocks.max(1) as usize;

    for req in requests_in_arrival_order {
        for &id in &req.block_ids {
            total_blocks += 1;
            tick += 1;
            if let Some(last_use) = resident.get_mut(&id) {
                // Hit: refresh recency to MRU.
                *last_use = tick;
                hits += 1;
            } else {
                // Miss: evict the LRU block if at capacity, then insert.
                if resident.len() >= capacity {
                    if let Some((&lru_id, _)) =
                        resident.iter().min_by_key(|&(_, &last_use)| last_use)
                    {
                        resident.remove(&lru_id);
                        evictions += 1;
                    }
                }
                resident.insert(id, tick);
            }
        }
    }

    let hit_rate = if total_blocks > 0 {
        hits as f64 / total_blocks as f64
    } else {
        0.0
    };

    CacheCurvePoint {
        capacity_blocks,
        hit_rate,
        evictions,
    }
}

/// Sweep realized LRU reuse across capacities at `ceil(unique_blocks * f)` for
/// `f in [0.1, 0.25, 0.5, 1.0, 2.0]`, with each capacity clamped to at least 1
/// and duplicate capacities removed (preserving ascending order).
pub fn realized_sweep(
    requests_in_arrival_order: &[RequestBlocks],
    unique_blocks: u64,
) -> Vec<CacheCurvePoint> {
    let fractions = [0.1_f64, 0.25, 0.5, 1.0, 2.0];
    let mut capacities: Vec<u64> = fractions
        .iter()
        .map(|f| (unique_blocks as f64 * f).ceil() as u64)
        .map(|c| c.max(1))
        .collect();
    capacities.sort_unstable();
    capacities.dedup();

    capacities
        .into_iter()
        .map(|capacity| realized_reuse(requests_in_arrival_order, capacity))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_reuse_shared_prefix() {
        // Two requests share the first two blocks [1,2]; second adds [7].
        let reqs = vec![
            RequestBlocks {
                conversation_id: "a".into(),
                turn_index: 0,
                block_ids: vec![1, 2, 3],
            },
            RequestBlocks {
                conversation_id: "a".into(),
                turn_index: 1,
                block_ids: vec![1, 2, 7],
            },
        ];
        let r = ideal_reuse(&reqs);
        assert_eq!(r.total_blocks, 6);
        // second request's blocks 1 and 2 are hits → 2 cached
        assert_eq!(r.cached_blocks, 2);
        assert!((r.hit_rate - 2.0 / 6.0).abs() < 1e-9);
        assert_eq!(r.unique_blocks, 4); // {1,2,3,7}
        assert_eq!(r.unique_roots, 1); // both start at block 1
        assert_eq!(r.intra_conversation_cached, 2);
        assert_eq!(r.cross_conversation_cached, 0);
        // second request cached 2/3 = 0.667 → ≥25/50 yes, ≥75/90/100 no
        assert_eq!(r.shared_prefix_rate_ge["50"], 1);
        assert_eq!(r.shared_prefix_rate_ge["75"], 0);
    }

    #[test]
    fn ideal_reuse_cross_conversation() {
        let reqs = vec![
            RequestBlocks {
                conversation_id: "a".into(),
                turn_index: 0,
                block_ids: vec![10, 11],
            },
            RequestBlocks {
                conversation_id: "b".into(),
                turn_index: 0,
                block_ids: vec![10, 11],
            },
        ];
        let r = ideal_reuse(&reqs);
        assert_eq!(r.cached_blocks, 2);
        assert_eq!(r.cross_conversation_cached, 2);
        assert_eq!(r.intra_conversation_cached, 0);
    }

    #[test]
    fn realized_lru_evicts_by_recency() {
        // Arrival order: r0 [1,2], r1 [3,4], r2 [1,2] again.
        let reqs = vec![
            RequestBlocks {
                conversation_id: "x".into(),
                turn_index: 0,
                block_ids: vec![1, 2],
            },
            RequestBlocks {
                conversation_id: "x".into(),
                turn_index: 1,
                block_ids: vec![3, 4],
            },
            RequestBlocks {
                conversation_id: "x".into(),
                turn_index: 2,
                block_ids: vec![1, 2],
            },
        ];
        // capacity 2 blocks: after r0 cache={1,2}; r1 evicts to {3,4}; r2 misses both.
        let c2 = realized_reuse(&reqs, 2);
        assert_eq!(c2.hit_rate, 0.0);
        // capacity 4 blocks: r2's [1,2] still resident → 2 hits of 6 total.
        let c4 = realized_reuse(&reqs, 4);
        assert!((c4.hit_rate - 2.0 / 6.0).abs() < 1e-9);
        // unbounded-ish capacity matches ideal for this trace.
        let big = realized_reuse(&reqs, 100);
        assert!((big.hit_rate - 2.0 / 6.0).abs() < 1e-9);
    }
}
