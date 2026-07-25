// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trajectory t*-sampling helpers, ported from
//! `src/aiperf/timing/trajectory_source.py`.
//!
//! The pure, deterministic core: per-trace / per-lane RNG seed derivation, the
//! dataset-wrap over-subscription policy, and the legacy timestamp-less
//! warmup/profile turn-count window (`k_i` candidate range). The stateful
//! `TrajectorySource` (dataset-metadata- and reconstruction-coupled) and the
//! final numpy `rng.choice` pick are not ported here.

use sha2::{Digest, Sha256};

/// Derive a per-trace RNG seed by hashing `trace_id` with the base seed
/// (Python `_seed_for_trace`): `sha256("{base}:{trace}")[:8]` big-endian.
pub fn seed_for_trace(base_seed: u64, trace_id: &str) -> u64 {
    let digest = Sha256::digest(format!("{base_seed}:{trace_id}").as_bytes());
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first8)
}

/// Derive a per-`(trace, lane)` RNG seed (Python `_seed_for_trace_lane`):
/// `sha256("{base}:{trace}:{lane}")[:8]` big-endian. Wrap-fill lanes share a
/// conversation id but must produce different start turns.
pub fn seed_for_trace_lane(base_seed: u64, trace_id: &str, lane_index: i64) -> u64 {
    let digest = Sha256::digest(format!("{base_seed}:{trace_id}:{lane_index}").as_bytes());
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first8)
}

/// Dataset-wrap over-subscription policy violation (Python raises `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrapPolicyError {
    /// Requested concurrency.
    pub concurrency: i64,
    /// Distinct loaded traces.
    pub distinct: i64,
}

impl std::fmt::Display for WrapPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "concurrency {} exceeds the {} distinct loaded traces while dataset wrapping is disabled",
            self.concurrency, self.distinct
        )
    }
}

impl std::error::Error for WrapPolicyError {}

/// Fail loud when concurrency over-subscribes a non-wrapping corpus (Python
/// `validate_dataset_wrap_policy`). Wrapping is opt-in; one-pass runs and session
/// budgets that fit the distinct pool are allowed.
pub fn validate_dataset_wrap_policy(
    distinct: i64,
    concurrency: i64,
    allow_dataset_wrap: bool,
    expected_num_sessions: Option<i64>,
    total_expected_requests: Option<i64>,
    expected_duration_sec: Option<f64>,
) -> Result<(), WrapPolicyError> {
    if allow_dataset_wrap {
        return Ok(());
    }
    let one_pass = expected_num_sessions.is_none()
        && total_expected_requests.is_none()
        && expected_duration_sec.is_none();
    if one_pass || matches!(expected_num_sessions, Some(n) if n <= distinct) {
        return Ok(());
    }
    if concurrency > distinct {
        return Err(WrapPolicyError {
            concurrency,
            distinct,
        });
    }
    Ok(())
}

/// The legacy timestamp-less warmup/profile `k_i` candidate window (Python
/// `_build_trajectory_for_lane`, pre-sendability-filter). Returns the candidate
/// start-turn indices; empty for `n <= 1` (no profile turn after warmup).
///
/// - `n == 2`: only `[0]` (turn 1 remains as the profile turn).
/// - `n >= 3`: `[k_min ..= k_max]` where `k_min = min(int(min_ratio*n), n-2)`,
///   `k_max = min(int(max_ratio*n), n-2)`, capped so `k_i + 1 < n` always holds.
pub fn legacy_start_turn_candidates(n: i64, start_min_ratio: f64, start_max_ratio: f64) -> Vec<i64> {
    if n <= 1 {
        return Vec::new();
    }
    if n == 2 {
        return vec![0];
    }
    let nf = n as f64;
    let mut k_min = ((start_min_ratio * nf) as i64).min(n - 2);
    let k_max = ((start_max_ratio * nf) as i64).min(n - 2);
    if k_min > k_max {
        k_min = k_max;
    }
    (k_min..=k_max).collect()
}

/// First turn index whose recorded timestamp is at/after `t_star_ms` (Python
/// `_next_turn_index_at_or_after`) — the PROFILING resume index. `None` when no
/// turn starts at/after t* (the whole stream is pre-t* history).
pub fn next_turn_index_at_or_after(turn_timestamps_ms: &[Option<f64>], t_star_ms: f64) -> Option<i64> {
    for (idx, ts) in turn_timestamps_ms.iter().enumerate() {
        if let Some(t) = ts {
            if t.is_finite() && *t >= t_star_ms {
                return Some(idx as i64);
            }
        }
    }
    None
}

/// The turn to warm for a stream (Python `ConversationState.warmup_turn_index`):
/// the last request before t* (`next_turn_index - 1`), or `None` when the
/// stream's first request is at/after t* (nothing to warm).
pub fn warmup_turn_index(next_turn_index: i64) -> Option<i64> {
    if next_turn_index >= 1 {
        Some(next_turn_index - 1)
    } else {
        None
    }
}

/// A PROFILING turn's dispatch offset from t* (Python `_offset_ms`): the
/// recorded gap after t*, floored at 0 (a turn recorded before t* dispatches
/// immediately). Missing timestamp → 0.
pub fn offset_ms(timestamp_ms: Option<f64>, t_star_ms: f64) -> f64 {
    match timestamp_ms {
        Some(ts) => (ts - t_star_ms).max(0.0),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_star_split_and_offsets() {
        // Turn timestamps (ms): [0, 100, 250, 400]; t* = 200.
        let ts = [Some(0.0), Some(100.0), Some(250.0), Some(400.0)];
        // First turn at/after 200 -> index 2 (t=250).
        assert_eq!(next_turn_index_at_or_after(&ts, 200.0), Some(2));
        // Warmup turn = resume - 1 = 1.
        assert_eq!(warmup_turn_index(2), Some(1));
        // A stream entirely at/after t* (resume 0) has nothing to warm.
        assert_eq!(warmup_turn_index(0), None);
        // Profiling offsets from t*: turn 2 -> 50, turn 3 -> 200; a pre-t* turn -> 0.
        assert_eq!(offset_ms(Some(250.0), 200.0), 50.0);
        assert_eq!(offset_ms(Some(400.0), 200.0), 200.0);
        assert_eq!(offset_ms(Some(100.0), 200.0), 0.0);
        // t* past every turn -> whole stream is history.
        assert_eq!(next_turn_index_at_or_after(&ts, 500.0), None);
    }

    #[test]
    fn seed_helpers_match_python() {
        assert_eq!(seed_for_trace(42, "trace_x"), 14945228459415978572);
        assert_eq!(seed_for_trace_lane(42, "trace_x", 0), 17254358379366083059);
    }

    #[test]
    fn legacy_candidates_window() {
        assert_eq!(legacy_start_turn_candidates(10, 0.25, 0.75), vec![2, 3, 4, 5, 6, 7]);
        assert_eq!(legacy_start_turn_candidates(2, 0.25, 0.75), vec![0]);
        assert!(legacy_start_turn_candidates(1, 0.25, 0.75).is_empty());
        // n=3: k_min=min(0,1)=0, k_max=min(2,1)=1 -> [0,1].
        assert_eq!(legacy_start_turn_candidates(3, 0.25, 0.75), vec![0, 1]);
    }

    #[test]
    fn wrap_policy_gates_oversubscription() {
        // Bounded run over-subscribing a non-wrapping corpus -> error.
        assert!(validate_dataset_wrap_policy(4, 8, false, Some(100), None, None).is_err());
        // allow_dataset_wrap -> ok.
        assert!(validate_dataset_wrap_policy(4, 8, true, Some(100), None, None).is_ok());
        // one-pass (no bounds) -> ok even when concurrency > distinct.
        assert!(validate_dataset_wrap_policy(4, 8, false, None, None, None).is_ok());
        // sessions fit the pool -> ok.
        assert!(validate_dataset_wrap_policy(4, 8, false, Some(3), None, None).is_ok());
    }
}
