// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared theoretical prefix-cache pre-pass for Weka traces, ported from
//! `src/aiperf/dataset/loader/weka_metric_prepass.py` (spec §5.5).
//!
//! `hash_id_scope: "local"` means one hash namespace per trace FILE, so a block
//! first sent by any conversation of a trace (root, subagent child, or detected
//! flat chain) is a cache hit when any other conversation of the same trace
//! re-sends it. This computes those values over ONE shared per-trace seen-set
//! consumed in global time order; emission then looks them up per
//! `(session_id, k)`.

use std::collections::{HashMap, HashSet};

/// Deterministic global-order sort key: `(absolute_t, outer_idx, stream_idx, k)`.
///
/// `absolute_t` is a wall-clock instant; the remaining integer fields break ties
/// so the ordering is total and float comparison never sees equal-`t` ambiguity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SortKey {
    /// Absolute request timestamp.
    pub absolute_t: f64,
    /// Index in the trace's top-level request list.
    pub outer_idx: i64,
    /// Index within a subagent's inner stream (0 for top-level).
    pub stream_idx: i64,
    /// Turn index within the conversation.
    pub k: i64,
}

/// One request's contribution to the per-trace shared seen-set
/// (Python `MetricRecord`).
#[derive(Debug, Clone)]
pub struct MetricRecord {
    /// Global-order key.
    pub sort_key: SortKey,
    /// Conversation the value is looked up under at emission time.
    pub session_id: String,
    /// Turn index within that conversation.
    pub k: i64,
    /// The request's input hash blocks.
    pub hash_ids: Vec<i64>,
}

/// `{(session_id, k): (hit_blocks, total_blocks)}` over ONE shared per-trace
/// seen-set, consumed in global time order (spec §5.5).
///
/// A request's `hit_blocks` is the length of the longest prefix of its
/// `hash_ids` all already in the seen-set (break at the first miss); the request
/// then contributes ALL its blocks to the seen-set for later requests.
pub fn compute_shared_prefix_cache_metrics(
    mut records: Vec<MetricRecord>,
) -> HashMap<(String, i64), (i64, i64)> {
    records.sort_by(|a, b| {
        // Total order on the fully-specified key; timestamps are finite.
        (
            a.sort_key.absolute_t,
            a.sort_key.outer_idx,
            a.sort_key.stream_idx,
            a.sort_key.k,
        )
            .partial_cmp(&(
                b.sort_key.absolute_t,
                b.sort_key.outer_idx,
                b.sort_key.stream_idx,
                b.sort_key.k,
            ))
            .expect("weka prefix-cache sort key must be finite")
    });

    let mut out: HashMap<(String, i64), (i64, i64)> = HashMap::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for rec in &records {
        let mut hits: i64 = 0;
        for hid in &rec.hash_ids {
            if !seen.contains(hid) {
                break;
            }
            hits += 1;
        }
        out.insert(
            (rec.session_id.clone(), rec.k),
            (hits, rec.hash_ids.len() as i64),
        );
        seen.extend(rec.hash_ids.iter().copied());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(t: f64, oi: i64, si: i64, sid: &str, k: i64, hs: &[i64]) -> MetricRecord {
        MetricRecord {
            sort_key: SortKey {
                absolute_t: t,
                outer_idx: oi,
                stream_idx: si,
                k,
            },
            session_id: sid.to_string(),
            k,
            hash_ids: hs.to_vec(),
        }
    }

    #[test]
    fn shared_seen_set_prefix_hits_in_time_order() {
        // r0 (t=0) sends [1,2,3]: nothing seen -> 0 hits.
        // r1 (t=1) sends [1,2,9]: prefix [1,2] seen -> 2 hits (breaks at 9).
        // r2 (t=2, other session) sends [9,1]: 9 now seen -> prefix [9,1] -> 2 hits.
        let recs = vec![
            rec(2.0, 2, 0, "b", 0, &[9, 1]),
            rec(0.0, 0, 0, "a", 0, &[1, 2, 3]),
            rec(1.0, 1, 0, "a", 1, &[1, 2, 9]),
        ];
        let m = compute_shared_prefix_cache_metrics(recs);
        assert_eq!(m[&("a".to_string(), 0)], (0, 3));
        assert_eq!(m[&("a".to_string(), 1)], (2, 3));
        assert_eq!(m[&("b".to_string(), 0)], (2, 2));
    }
}
