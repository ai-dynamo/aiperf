// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Summary window and phase filters shared by accumulators and analyzers.
//!
//! Authoritative phase selection and half-open timeslice construction live here.

use serde::{Deserialize, Serialize};

/// Credit phase attached to a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Warmup traffic that should not count toward profiling-only summaries.
    Warmup,
    /// Profiling traffic used for the primary benchmark summary.
    Profiling,
}

/// A half-open time range used for timeslice summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Timeslice {
    /// Inclusive lower bound in nanoseconds.
    pub start_ns: i64,
    /// Exclusive upper bound in nanoseconds.
    pub end_ns: i64,
}

impl Timeslice {
    /// Builds a half-open timeslice.
    pub fn new(start_ns: i64, end_ns: i64) -> Self {
        Self { start_ns, end_ns }
    }

    /// Returns true when `timestamp_ns` lands inside this half-open timeslice.
    pub fn contains_start(self, timestamp_ns: i64) -> bool {
        timestamp_ns >= self.start_ns && timestamp_ns < self.end_ns
    }

    /// Returns true when the record interval is contained in this half-open timeslice.
    pub fn contains_interval(self, start_ns: i64, end_ns: i64) -> bool {
        start_ns >= self.start_ns && end_ns <= self.end_ns
    }
}

/// Context selecting which records an accumulator should summarize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct ExportContext {
    /// Optional inclusive lower bound in nanoseconds.
    pub start_ns: Option<i64>,
    /// Optional exclusive upper bound in nanoseconds.
    pub end_ns: Option<i64>,
    /// Optional phase filter. Phase masks are authoritative over wall-clock bounds.
    pub phase: Option<Phase>,
    /// Optional concrete instance within the selected phase.
    pub phase_index: Option<usize>,
}

impl ExportContext {
    /// Summarizes all records.
    pub fn all() -> Self {
        Self::default()
    }

    /// Summarizes records whose start timestamp is in a half-open time range.
    pub fn time_range(start_ns: i64, end_ns: i64) -> Self {
        Self {
            start_ns: Some(start_ns),
            end_ns: Some(end_ns),
            phase: None,
            phase_index: None,
        }
    }

    /// Summarizes records in one phase.
    pub fn phase(phase: Phase) -> Self {
        Self {
            start_ns: None,
            end_ns: None,
            phase: Some(phase),
            phase_index: None,
        }
    }

    /// Summarizes records in one concrete phase instance.
    pub fn phase_index(phase: Phase, phase_index: usize) -> Self {
        Self {
            start_ns: None,
            end_ns: None,
            phase: Some(phase),
            phase_index: Some(phase_index),
        }
    }

    /// Returns this context as a timeslice when both bounds are present.
    pub fn timeslice(self) -> Option<Timeslice> {
        Some(Timeslice::new(self.start_ns?, self.end_ns?))
    }

    /// Returns true when the record dimensions pass this context.
    pub fn contains(self, phase: Phase, start_ns: i64, _end_ns: i64) -> bool {
        if let Some(expected) = self.phase {
            return expected == phase;
        }
        self.start_ns.is_none_or(|lower| start_ns >= lower)
            && self.end_ns.is_none_or(|upper| start_ns < upper)
    }
}

#[cfg(test)]
mod tests {
    use super::{ExportContext, Phase, Timeslice};

    #[test]
    fn timeslice_is_half_open_for_start_timestamps() {
        let window = Timeslice::new(10, 20);
        assert!(window.contains_start(10));
        assert!(window.contains_start(19));
        assert!(!window.contains_start(9));
        assert!(!window.contains_start(20));
    }

    #[test]
    fn timeslice_is_half_open_for_contained_intervals() {
        let window = Timeslice::new(10, 20);
        assert!(window.contains_interval(10, 19));
        assert!(window.contains_interval(11, 19));
        assert!(!window.contains_interval(9, 19));
        assert!(window.contains_interval(10, 20));
        assert!(!window.contains_interval(10, 21));
    }

    #[test]
    fn export_context_phase_is_authoritative_over_window() {
        let ctx = ExportContext {
            start_ns: Some(500),
            end_ns: Some(600),
            phase: Some(Phase::Warmup),
            phase_index: None,
        };
        assert!(ctx.contains(Phase::Warmup, 100, 200));
        assert!(!ctx.contains(Phase::Profiling, 550, 560));
    }

    #[test]
    fn export_context_time_range_is_half_open_on_start_timestamp_only() {
        let ctx = ExportContext::time_range(100, 200);
        assert!(ctx.contains(Phase::Profiling, 100, 150));
        assert!(ctx.contains(Phase::Warmup, 199, 200));
        assert!(!ctx.contains(Phase::Profiling, 99, 150));
        assert!(ctx.contains(Phase::Profiling, 199, 201));
        assert!(!ctx.contains(Phase::Profiling, 200, 201));
    }
}
