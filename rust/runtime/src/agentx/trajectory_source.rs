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
pub fn legacy_start_turn_candidates(
    n: i64,
    start_min_ratio: f64,
    start_max_ratio: f64,
) -> Vec<i64> {
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

/// Pick the legacy timestamp-less start turn `k_i` from `candidates` using a
/// numpy `default_rng(seed).choice(candidates)` draw (Python
/// `_build_trajectory_for_lane`'s `rng.choice`). `choice` on a 1-d array reduces
/// to `candidates[integers(0, len)]`, reproduced bit-for-bit via the runtime's
/// numpy compat. `None` for an empty candidate list.
pub fn legacy_pick_start_turn(seed: u64, candidates: &[i64]) -> Option<i64> {
    if candidates.is_empty() {
        return None;
    }
    let mut rng = crate::rng::compat::numpy_generator::NumpyGenerator::from_seed(seed);
    let idx = rng.integers(0, candidates.len() as i64) as usize;
    Some(candidates[idx])
}

/// Sample the timestamped-trace t* in `[lo, hi)` via numpy
/// `default_rng(seed).uniform(lo, hi)` (Python `_build_timestamped_trajectory`),
/// reproduced as `lo + random()*(hi-lo)`. Returns `lo` when `hi == lo`.
pub fn timestamped_t_star_ms(seed: u64, lo: f64, hi: f64) -> f64 {
    if hi == lo {
        return lo;
    }
    let mut rng = crate::rng::compat::numpy_generator::NumpyGenerator::from_seed(seed);
    lo + rng.random() * (hi - lo)
}

/// First turn index whose recorded timestamp is at/after `t_star_ms` (Python
/// `_next_turn_index_at_or_after`) — the PROFILING resume index. `None` when no
/// turn starts at/after t* (the whole stream is pre-t* history).
pub fn next_turn_index_at_or_after(
    turn_timestamps_ms: &[Option<f64>],
    t_star_ms: f64,
) -> Option<i64> {
    for (idx, ts) in turn_timestamps_ms.iter().enumerate() {
        if let Some(t) = ts
            && t.is_finite()
            && *t >= t_star_ms
        {
            return Some(idx as i64);
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

/// A WARMUP turn's lead — how far before t* its recorded turn sits — capped at
/// the idle-gap cap (Python `_capped_warmup_lead_ms`, AR:416-431). `lead` is
/// `t* - warm_turn_ts`; the cap bounds priming spacing (which carries no replay
/// meaning). No cap → uncapped lead.
pub fn capped_warmup_lead_ms(lead_ms: f64, cap_ms: Option<f64>) -> f64 {
    match cap_ms {
        Some(cap) => lead_ms.min(cap),
        None => lead_ms,
    }
}

/// Per-stream WARMUP dispatch offsets from warmup-phase start under the SPREAD
/// policy (Python `_execute_warmup`, AR:565-583): global t*-alignment — the turn
/// with the largest (capped) lead fires at 0, every stream's t* lands at the
/// common `max_lead`, so a stream's offset is `max_lead - lead`. A leadless
/// stream (`None`, e.g. a subagent with no pre-t* turn) fires at 0. The total
/// spread is `(max_lead - min_lead)`. Returns offsets in ms, index-aligned to
/// `leads`.
pub fn warmup_dispatch_offsets_ms(leads: &[Option<f64>]) -> Vec<f64> {
    let max_lead = leads.iter().filter_map(|l| *l).fold(0.0_f64, f64::max);
    leads
        .iter()
        .map(|l| match l {
            Some(lead) => (max_lead - lead).max(0.0),
            None => 0.0,
        })
        .collect()
}

/// Excess to subtract uniformly from every PROFILING dispatch offset so a
/// trajectory's leading idle (t* → its earliest post-t* request) is capped
/// (Python `_leading_idle_shift_ms`). `max(0, min(offsets) - cap)`; 0 when no cap
/// or already within it. A single uniform shift preserves relative spacing.
pub fn leading_idle_shift_ms(offsets: &[f64], cap_ms: Option<f64>) -> f64 {
    match cap_ms {
        None => 0.0,
        Some(cap) => {
            if offsets.is_empty() {
                0.0
            } else {
                (offsets.iter().copied().fold(f64::INFINITY, f64::min) - cap).max(0.0)
            }
        }
    }
}

/// Per-stream PROFILING dispatch delays from profiling-start (Python's
/// phase-start anchoring around `_dispatch_profiling`): apply the leading-idle
/// shift, then anchor at t0 = the lane's min shifted offset under
/// `--burst-phase-starts` (lane bursts together) or 0 under spread (each lane
/// waits its recorded gap). Delays are floored at 0 (a non-positive delay fires
/// immediately).
pub fn profiling_dispatch_delays_ms(offsets: &[f64], burst: bool, cap_ms: Option<f64>) -> Vec<f64> {
    let shift = leading_idle_shift_ms(offsets, cap_ms);
    let shifted: Vec<f64> = offsets.iter().map(|o| o - shift).collect();
    let t0 = if burst && !shifted.is_empty() {
        shifted.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        0.0
    };
    shifted.iter().map(|s| (s - t0).max(0.0)).collect()
}

/// A stream turn's replay phase under a sampled t*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayPhase {
    /// Pre-t* turn back-seeded as history (not dispatched).
    History,
    /// The last pre-t* turn, dispatched as the warmup/session-start.
    Warmup,
    /// A turn at/after t*, dispatched during PROFILING at its offset.
    Profiling,
}

/// One scheduled turn: its phase and (for PROFILING) its dispatch offset from t*.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledTurn {
    /// Turn index within the stream.
    pub k: i64,
    /// Replay phase.
    pub phase: ReplayPhase,
    /// PROFILING dispatch offset from t* in ms (`None` for history/warmup).
    pub offset_ms: Option<f64>,
}

/// Compute a stream's agentic-replay execution schedule from its recorded turn
/// timestamps and a sampled t*: which turns are history / warmup / profiling, and
/// each profiling turn's dispatch offset from t*. This is the deterministic
/// execution-order timing schedule the async dispatch loop fires against.
pub fn replay_schedule(turn_timestamps_ms: &[Option<f64>], t_star_ms: f64) -> Vec<ScheduledTurn> {
    let n = turn_timestamps_ms.len() as i64;
    let resume = next_turn_index_at_or_after(turn_timestamps_ms, t_star_ms);
    let warmup_idx = resume.and_then(warmup_turn_index);
    (0..n)
        .map(|k| {
            let (phase, offset) = match resume {
                None => {
                    if k == n - 1 {
                        (ReplayPhase::Warmup, None)
                    } else {
                        (ReplayPhase::History, None)
                    }
                }
                Some(resume) => {
                    if Some(k) == warmup_idx {
                        (ReplayPhase::Warmup, None)
                    } else if k >= resume {
                        (
                            ReplayPhase::Profiling,
                            Some(offset_ms(turn_timestamps_ms[k as usize], t_star_ms)),
                        )
                    } else {
                        (ReplayPhase::History, None)
                    }
                }
            };
            ScheduledTurn {
                k,
                phase,
                offset_ms: offset,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiling_phase_start_anchoring() {
        let offs = [500.0, 50.0, 650.0];
        // spread, no cap -> unchanged.
        assert_eq!(
            profiling_dispatch_delays_ms(&offs, false, None),
            vec![500.0, 50.0, 650.0]
        );
        // burst, no cap -> anchor at min(50) -> [450, 0, 600].
        assert_eq!(
            profiling_dispatch_delays_ms(&offs, true, None),
            vec![450.0, 0.0, 600.0]
        );
        // cap 30 -> shift = min(50)-30 = 20; spread -> [480, 30, 630].
        assert_eq!(leading_idle_shift_ms(&offs, Some(30.0)), 20.0);
        assert_eq!(
            profiling_dispatch_delays_ms(&offs, false, Some(30.0)),
            vec![480.0, 30.0, 630.0]
        );
        // cap 30 + burst -> t0 = 30 -> [450, 0, 600].
        assert_eq!(
            profiling_dispatch_delays_ms(&offs, true, Some(30.0)),
            vec![450.0, 0.0, 600.0]
        );
        // cap above the leading idle -> no shift.
        assert_eq!(leading_idle_shift_ms(&offs, Some(100.0)), 0.0);
    }

    #[test]
    fn replay_schedule_matches_python_golden() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/agentx/schedule_golden.json");
        let raw = match std::fs::read(&path) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("skip: schedule golden absent");
                return;
            }
        };
        let scenarios: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        for sc in scenarios.as_array().unwrap() {
            let name = sc["name"].as_str().unwrap();
            let ts: Vec<Option<f64>> = sc["timestamps_ms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64())
                .collect();
            let t_star = sc["t_star_ms"].as_f64().unwrap();
            let sched = replay_schedule(&ts, t_star);
            let want = sc["per_turn"].as_array().unwrap();
            assert_eq!(sched.len(), want.len(), "{name} turn count");
            for (got, w) in sched.iter().zip(want) {
                let want_phase = match w["phase"].as_str().unwrap() {
                    "history" => ReplayPhase::History,
                    "warmup" => ReplayPhase::Warmup,
                    "profiling" => ReplayPhase::Profiling,
                    other => panic!("bad phase {other}"),
                };
                assert_eq!(got.phase, want_phase, "{name} k{} phase", got.k);
                assert_eq!(
                    got.offset_ms,
                    w["offset_ms"].as_f64(),
                    "{name} k{} offset",
                    got.k
                );
            }
        }
    }

    #[test]
    fn t_star_rng_pick_matches_numpy() {
        // seed = seed_for_trace_lane(42, "trace_x", 0).
        let seed = seed_for_trace_lane(42, "trace_x", 0);
        // np.random.default_rng(seed).choice([2..7]) == 4.
        assert_eq!(legacy_pick_start_turn(seed, &[2, 3, 4, 5, 6, 7]), Some(4));
        // np.random.default_rng(seed).uniform(1000, 2000) == 1430.3229154123997.
        let t = timestamped_t_star_ms(seed, 1000.0, 2000.0);
        assert!((t - 1430.3229154123997).abs() < 1e-9, "got {t}");
        // hi == lo short-circuits to lo (no draw).
        assert_eq!(timestamped_t_star_ms(seed, 500.0, 500.0), 500.0);
    }

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
        assert_eq!(
            legacy_start_turn_candidates(10, 0.25, 0.75),
            vec![2, 3, 4, 5, 6, 7]
        );
        assert_eq!(legacy_start_turn_candidates(2, 0.25, 0.75), vec![0]);
        assert!(legacy_start_turn_candidates(1, 0.25, 0.75).is_empty());
        // n=3: k_min=min(0,1)=0, k_max=min(2,1)=1 -> [0,1].
        assert_eq!(legacy_start_turn_candidates(3, 0.25, 0.75), vec![0, 1]);
    }

    #[test]
    fn warmup_lead_capping_and_global_alignment() {
        // Cap bounds the lead; no cap leaves it; under cap is unchanged.
        assert_eq!(capped_warmup_lead_ms(5000.0, Some(10000.0)), 5000.0);
        assert_eq!(capped_warmup_lead_ms(15000.0, Some(10000.0)), 10000.0);
        assert_eq!(capped_warmup_lead_ms(5000.0, None), 5000.0);
        // Global t*-alignment: largest lead fires at 0, others offset by
        // (max_lead - lead); a leadless stream (None) fires at 0.
        let offsets = warmup_dispatch_offsets_ms(&[Some(2000.0), Some(5000.0), None, Some(0.0)]);
        assert_eq!(offsets, vec![3000.0, 0.0, 0.0, 5000.0]);
        // All-leadless (e.g. every warmup turn missing a timestamp) -> all 0.
        assert_eq!(warmup_dispatch_offsets_ms(&[None, None]), vec![0.0, 0.0]);
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
