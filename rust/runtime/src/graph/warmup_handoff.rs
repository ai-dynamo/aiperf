// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extended-warmup handoff payload.
//!
//! Warmup builds one [`GraphWarmupHandoff`] after every credit return lands.
//! Profiling consumes it once and resumes each lane at its recorded frontier.
//!
//! The per-`(trace, lane)` t* plan is not carried because both phases derive it
//! from the seeded sampler. The handoff
//! carries only what determinism cannot reproduce: which template each lane was
//! mid-flight on at drain, which nodes actually executed, and when their
//! returns landed.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One lane's live-at-drain execution state.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaneHandoff {
    /// Template trace id the lane was mid-flight on at drain (may differ from
    /// the lane's pass-0 template when pressure warmup recycled the lane).
    pub template_trace_id: String,
    /// The live pressure instance id at drain (e.g. `t-1#0.p2`). The profiling
    /// resume reuses it verbatim as the resumed instance's id so the
    /// per-instance cache-bust marker (digest of `credit.trace_id`; see
    /// `build_trace_instance_marker`) is continuous across the handoff and the
    /// KV built during pressure transfers instead of cold-prefilling behind a
    /// fresh `.0` marker.
    pub instance_id: String,
    /// The instance's t* (lane-salted plan for the pressure pass-0 instance;
    /// `0.0` for recycled full-replay instances). Pre-t* nodes are warmup
    /// history and are dropped by the frontier chop alongside the executed set.
    pub t_star_us: f64,
    /// Node ids of the live instance that dispatched AND returned during
    /// warmup/pressure -- the server holds their KV; profiling must not refire.
    pub executed_node_ids: BTreeSet<String>,
    /// Monotonic return wall times (microseconds, strategy ledger clock) used
    /// to compute residual re-root delays; includes the lane's boundary-priming
    /// returns merged in for pass-0 instances.
    pub return_wall_us: BTreeMap<String, f64>,
}

/// The full warmup -> profiling handoff, one entry per live lane.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphWarmupHandoff {
    /// Lanes live at drain, keyed by lane index. Lanes absent here (template
    /// completed exactly at drain, or lane index beyond the pressure lane
    /// count) resume the normal pass-0 t* path in profiling.
    pub lanes: BTreeMap<u64, LaneHandoff>,
    /// Drain-end instant on the same monotonic clock as the return walls,
    /// stamped at warmup teardown (after all returns landed).
    pub drain_end_wall_us: f64,
    /// Next corpus draw index after pressure warmup's last recycle draw.
    /// Profiling's BOUNDED recycle loop continues the wrap from here so freed
    /// lanes don't re-serve templates pressure warmup just replayed. One
    /// sampler is shared across pressure, handoff, and profiling draws. Single-
    /// pass profiling (no stop conditions) deliberately ignores it -- full-corpus
    /// coverage takes precedence over cursor continuity there.
    pub corpus_cursor: u64,
    /// Number of pressure lanes (`0..K-1`) the warmup ran. A lane below this
    /// count with NO entry in `lanes` completed at drain: profiling must
    /// fresh-start it (next cursor template, full `t*=0` replay) instead of
    /// re-running a t* resume pressure warmup already executed.
    pub pressure_lane_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GraphWarmupHandoff {
        let lane0 = LaneHandoff {
            template_trace_id: "t-1".to_string(),
            instance_id: "t-1#0.p2".to_string(),
            t_star_us: 1234.5,
            executed_node_ids: BTreeSet::from(["n0".to_string(), "n1".to_string()]),
            return_wall_us: BTreeMap::from([("n0".to_string(), 100.0), ("n1".to_string(), 250.0)]),
        };
        let lane3 = LaneHandoff {
            template_trace_id: "t-7".to_string(),
            instance_id: "t-7#0.p0".to_string(),
            t_star_us: 0.0,
            executed_node_ids: BTreeSet::new(),
            return_wall_us: BTreeMap::new(),
        };
        GraphWarmupHandoff {
            lanes: BTreeMap::from([(0u64, lane0), (3u64, lane3)]),
            drain_end_wall_us: 9999.0,
            corpus_cursor: 42,
            pressure_lane_count: 4,
        }
    }

    #[test]
    fn field_values_are_carried() {
        let h = sample();
        assert_eq!(h.lanes.len(), 2);
        assert_eq!(h.drain_end_wall_us, 9999.0);
        assert_eq!(h.corpus_cursor, 42);
        assert_eq!(h.pressure_lane_count, 4);

        let lane0 = &h.lanes[&0];
        assert_eq!(lane0.template_trace_id, "t-1");
        assert_eq!(lane0.instance_id, "t-1#0.p2");
        assert_eq!(lane0.t_star_us, 1234.5);
        assert!(lane0.executed_node_ids.contains("n0"));
        assert_eq!(lane0.return_wall_us["n1"], 250.0);

        // Lane index is load-bearing: lane 3 exists, lanes 1 and 2 (below
        // pressure_lane_count with no entry) are the "completed at drain" case.
        assert!(!h.lanes.contains_key(&1));
        assert!(h.lanes.contains_key(&3));
    }

    #[test]
    fn serde_round_trips() {
        let h = sample();
        let json = serde_json::to_string(&h).expect("serialize");
        let back: GraphWarmupHandoff = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(h, back);
    }

    #[test]
    fn default_constructs_empty() {
        let h = GraphWarmupHandoff::default();
        assert!(h.lanes.is_empty());
        assert_eq!(h.corpus_cursor, 0);
        assert_eq!(h.pressure_lane_count, 0);
    }
}
