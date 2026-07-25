// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AgentX metric accumulators (Slice 4), ported from
//! `src/aiperf/metrics/theoretical_prefix_cache.py` and
//! `src/aiperf/metrics/types/context_overflow_count_metric.py`.
//!
//! These are the pure accumulation cores; the metrics-framework wiring
//! (`BaseMetricsProcessor`, phase-scoped export) lives in the runtime metrics
//! plane. Phases are modeled as opaque string keys.

use std::collections::HashMap;

/// Phase-scoped theoretical prefix-cache hit-rate accumulator (Python
/// `TheoreticalPrefixCacheAccumulator`). Consumes the per-turn
/// `theoretical_prefix_cache_hit_blocks` / `_total_blocks` the loader stamps.
#[derive(Debug, Default)]
pub struct TheoreticalPrefixCacheAccumulator {
    hit_by_phase: HashMap<String, i64>,
    total_by_phase: HashMap<String, i64>,
}

impl TheoreticalPrefixCacheAccumulator {
    /// New empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate one record's `(hit_blocks, total_blocks)` under `phase`.
    /// Records with `total <= 0` are skipped; `hit` is clamped to `[0, total]`
    /// (a loader miscount must not push the rate above 100% or below 0%).
    pub fn add(&mut self, phase: &str, hit_blocks: i64, total_blocks: i64) {
        if total_blocks <= 0 {
            return;
        }
        let hit = hit_blocks.clamp(0, total_blocks);
        *self.hit_by_phase.entry(phase.to_string()).or_insert(0) += hit;
        *self.total_by_phase.entry(phase.to_string()).or_insert(0) += total_blocks;
    }

    /// Hit rate as a percentage for `phase` (or all phases when `None`).
    /// `None` when no blocks were accumulated (Python returns no `MetricResult`).
    pub fn hit_rate_pct(&self, phase: Option<&str>) -> Option<f64> {
        let (hit, total) = match phase {
            None => (
                self.hit_by_phase.values().sum::<i64>(),
                self.total_by_phase.values().sum::<i64>(),
            ),
            Some(p) => (
                self.hit_by_phase.get(p).copied().unwrap_or(0),
                self.total_by_phase.get(p).copied().unwrap_or(0),
            ),
        };
        if total <= 0 {
            None
        } else {
            Some(100.0 * hit as f64 / total as f64)
        }
    }
}

/// Count of records classified as context-overflow (Python
/// `ContextOverflowCountMetric`): counts records with `context_overflow == true`.
#[derive(Debug, Default)]
pub struct ContextOverflowCount {
    count: i64,
}

impl ContextOverflowCount {
    /// New zeroed counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Account one record; increments only when `context_overflow` is true.
    pub fn add(&mut self, context_overflow: bool) {
        if context_overflow {
            self.count += 1;
        }
    }

    /// The accumulated count.
    pub fn count(&self) -> i64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_cache_rate_clamps_and_scopes_by_phase() {
        let mut acc = TheoreticalPrefixCacheAccumulator::new();
        acc.add("warmup", 1, 2);
        acc.add("profiling", 3, 4);
        acc.add("profiling", 9, 4); // hit clamped to 4
        acc.add("profiling", 0, 0); // skipped (total 0)
        // profiling: (3 + 4) / (4 + 4) = 7/8 = 87.5%.
        assert_eq!(acc.hit_rate_pct(Some("profiling")), Some(87.5));
        // warmup: 1/2 = 50%.
        assert_eq!(acc.hit_rate_pct(Some("warmup")), Some(50.0));
        // all phases: (1+3+4)/(2+4+4) = 8/10 = 80%.
        assert_eq!(acc.hit_rate_pct(None), Some(80.0));
        // unknown phase / no data.
        assert_eq!(acc.hit_rate_pct(Some("missing")), None);
    }

    #[test]
    fn context_overflow_counts_true_only() {
        let mut c = ContextOverflowCount::new();
        c.add(true);
        c.add(false);
        c.add(true);
        assert_eq!(c.count(), 2);
    }
}
