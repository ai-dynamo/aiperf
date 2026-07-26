// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-trace idle-gap time-warp: compress request-start gaps larger than a cap so
//! a long dead-air stretch does not stall replay. Byte-exact port of Python
//! `_IdleGapTimeWarp` / `_build_trace_idle_timing` (`dataset/loader/weka_trace.py`).
//!
//! The warp is per root trace (not global). It collects every request submission
//! timestamp from the parent and all subagents, compresses any gap between
//! consecutive starts above `cap_seconds`, then maps raw seconds to adjusted
//! seconds. All downstream timing (t\* sampling, dispatch delays, subagent
//! spawn/join placement) then operates on the warped timeline — which is what
//! makes the sampled t\* land on the same turn the Python oracle resumes at.

/// One compressed idle gap `[raw_start, raw_end]` whose excess beyond the cap is
/// collapsed; requests at/after `raw_end` shift left by the accumulated excess.
#[derive(Debug, Clone)]
struct IdleGap {
    raw_start: f64,
    raw_end: f64,
    shift_before: f64,
    cap_seconds: f64,
    excess_seconds: f64,
}

/// Maps raw request-start seconds to the idle-gap-capped timeline.
#[derive(Debug, Clone, Default)]
pub struct IdleGapTimeWarp {
    gaps: Vec<IdleGap>,
}

impl IdleGapTimeWarp {
    /// Build the warp from a trace's request-start timestamps (seconds). Gaps
    /// between consecutive *sorted* starts exceeding `cap_seconds` are recorded
    /// with their excess. Mirrors Python `_IdleGapTimeWarp.__init__`.
    pub fn new(mut request_starts: Vec<f64>, cap_seconds: f64) -> Self {
        request_starts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut gaps = Vec::new();
        let Some(&first) = request_starts.first() else {
            return Self { gaps };
        };
        let mut prev_start = first;
        let mut cumulative_shift = 0.0;
        for &start in &request_starts[1..] {
            let gap_seconds = start - prev_start;
            if gap_seconds > cap_seconds {
                let excess = gap_seconds - cap_seconds;
                gaps.push(IdleGap {
                    raw_start: prev_start,
                    raw_end: start,
                    shift_before: cumulative_shift,
                    cap_seconds,
                    excess_seconds: excess,
                });
                cumulative_shift += excess;
            }
            prev_start = start;
        }
        Self { gaps }
    }

    /// Map a raw timestamp (seconds) to the idle-gap-capped timeline. Mirrors
    /// Python `_IdleGapTimeWarp.map`: keep the first `cap_seconds` of each long
    /// gap intact, collapse the remainder to the cap boundary, and shift later
    /// requests left by the accumulated excess.
    pub fn map(&self, t_seconds: f64) -> f64 {
        let mut shift = 0.0;
        for gap in &self.gaps {
            if t_seconds < gap.raw_start {
                return t_seconds - gap.shift_before;
            }
            if t_seconds < gap.raw_end {
                let local = t_seconds - gap.raw_start;
                if local <= gap.cap_seconds {
                    return t_seconds - gap.shift_before;
                }
                return gap.raw_start - gap.shift_before + gap.cap_seconds;
            }
            shift = gap.shift_before + gap.excess_seconds;
        }
        t_seconds - shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_gap_above_cap_and_shifts_later_requests() {
        // starts 0, 20, 220 with cap 60: gap 20->220 is 200s, excess 140.
        let warp = IdleGapTimeWarp::new(vec![0.0, 20.0, 220.0], 60.0);
        assert_eq!(warp.map(0.0), 0.0);
        assert_eq!(warp.map(20.0), 20.0);
        // request at 220 shifts left by 140 -> 80.
        assert_eq!(warp.map(220.0), 80.0);
        // inside the collapsed tail (e.g. 200) maps to the cap boundary 20+60=80.
        assert_eq!(warp.map(200.0), 80.0);
    }

    #[test]
    fn no_gaps_is_identity() {
        let warp = IdleGapTimeWarp::new(vec![0.0, 1.0, 2.0], 10.0);
        assert_eq!(warp.map(1.5), 1.5);
    }
}
