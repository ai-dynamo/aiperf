// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AgentX agentic-replay timing mode — a faithful port of Python's
//! `AgenticReplayStrategy` (`src/aiperf/timing/strategies/agentic_replay.py`).
//!
//! A scheduled-runtime [`Workload`]: it owns its own dispatch timing (per-lane
//! t\* sampling, warmup global-alignment, profiling offsets, cache-bust) and
//! reuses the shared [`ScheduledRuntime`] for transport/metrics/records/report.
//! It is selected per-phase by [`crate::engine::protocol::PhaseSpec::AgenticReplay`]
//! and runs as two instances — a WARMUP phase (dispatches the last pre-t\* turn of
//! each stream to prime server KV) and a PROFILING phase (resumes at t\* and
//! replays each stream at its recorded offsets).
//!
//! The pure timing kernel lives in [`crate::agentx::trajectory_source`] and the
//! byte-exact marker builder in [`crate::agentx::cache_bust`]; this module drives
//! them against the [`crate::multiturn::ConversationSource`] seam.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agentx::cache_bust::CacheBustTarget;
use crate::agentx::trajectory_source::{
    capped_warmup_lead_ms, next_turn_index_at_or_after, offset_ms, profiling_dispatch_delays_ms,
    seed_for_trace_lane, timestamped_t_star_ms, warmup_dispatch_offsets_ms,
};
use crate::multiturn::ConversationSource;
use crate::scheduled::{ScheduledRuntime, Workload};

/// Nanoseconds per millisecond.
const NS_PER_MS: f64 = 1_000_000.0;
/// Lead the real-clock scheduling pass by this much so `schedule_at_ns` targets
/// stay in the future while the O(n) pass runs (mirrors fixed_schedule).
const SCHEDULE_START_LEAD_NS: i64 = 25_000_000;

/// Per-lane dispatch decision computed from the trajectory's recorded timestamps.
struct LaneDispatch {
    /// The conversation/session template id.
    conversation_id: String,
    /// The lane's phase-start dispatch offset in ms (warmup lead-aligned or
    /// profiling t\*-relative), before cross-lane alignment.
    warm_lead_ms: Option<f64>,
    /// The first post-t\* turn's offset from t\* (profiling), or 0 when none.
    first_profiling_offset_ms: f64,
    /// Whether the lane has any post-t\* (profiling) turn.
    has_profiling: bool,
    /// Whether the lane has a warmup turn (a pre-t\* turn to prime).
    has_warmup: bool,
}

/// Which phase an [`AgenticReplayWorkload`] instance drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenticPhase {
    /// Dispatch the warmup turn (n-1) of each stream to prime the server KV.
    Warmup,
    /// Resume each stream at t\* and replay its post-t\* turns at their offsets.
    Profiling,
}

/// Configuration for one agentic-replay phase instance (from the authored
/// [`PhaseSpec::AgenticReplay`](crate::engine::protocol::PhaseSpec) + run identity).
#[derive(Debug, Clone)]
pub struct AgenticReplayConfig {
    /// Which phase this instance drives.
    pub phase: AgenticPhase,
    /// Trajectory-start window lower ratio.
    pub start_min_ratio: f64,
    /// Trajectory-start window upper ratio.
    pub start_max_ratio: f64,
    /// Idle-gap cap in ms for warmup-lead / leading-idle capping (`None` = uncapped).
    pub idle_gap_cap_ms: Option<f64>,
    /// Anchor phase-start bursts at the earliest post-t\* request instead of spread.
    pub burst_phase_starts: bool,
    /// Base random seed for per-lane t\* sampling.
    pub random_seed: u64,
    /// The run's benchmark id (cache-bust digest input).
    pub benchmark_id: String,
    /// Cache-bust placement (scenario-locked to first-turn-prefix for the MVP).
    pub cache_bust_target: CacheBustTarget,
}

/// The agentic-replay workload: drives one phase's dispatch over a
/// [`ConversationSource`] of reconstructed WEKA trajectories.
pub struct AgenticReplayWorkload {
    #[allow(dead_code)]
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    #[allow(dead_code)]
    config: AgenticReplayConfig,
}

impl AgenticReplayWorkload {
    /// Build the workload over a reconstructed-trajectory conversation source.
    pub fn new(
        source: Box<dyn ConversationSource>,
        config: AgenticReplayConfig,
    ) -> Result<Self> {
        Ok(Self {
            source: Rc::new(RefCell::new(source)),
            config,
        })
    }
}

#[async_trait(?Send)]
impl Workload for AgenticReplayWorkload {
    fn name(&self) -> &'static str {
        "agentic_replay"
    }

    /// Authored per-turn dispatch times (not credit-paced), like fixed_schedule.
    fn has_credit_timestamps(&self) -> bool {
        false
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        let cfg = &self.config;
        // 1) Per-lane t\* sampling + turn classification from recorded timestamps.
        let lanes: Vec<LaneDispatch> = {
            let source = self.source.borrow();
            source
                .conversations()
                .iter()
                .enumerate()
                .map(|(lane_index, meta)| {
                    let ts: Vec<Option<f64>> =
                        meta.turns.iter().map(|t| t.timestamp_ms).collect();
                    let seed =
                        seed_for_trace_lane(cfg.random_seed, &meta.conversation_id, lane_index as i64);
                    // t\* is uniform over [lo, hi) in wall-clock time (Python parity).
                    let known: Vec<f64> = ts.iter().filter_map(|x| *x).collect();
                    let (lo, hi) = if known.is_empty() {
                        (0.0, 0.0)
                    } else {
                        let mn = known.iter().copied().fold(f64::INFINITY, f64::min);
                        let mx = known.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let dur = mx - mn;
                        (mn + cfg.start_min_ratio * dur, mn + cfg.start_max_ratio * dur)
                    };
                    let t_star = timestamped_t_star_ms(seed, lo, hi);
                    let next_idx = next_turn_index_at_or_after(&ts, t_star);
                    let has_warmup = next_idx.is_some_and(|n| n >= 1) || (next_idx.is_none() && ts.len() >= 2);
                    let (has_profiling, first_profiling_offset_ms) = match next_idx {
                        Some(n) => (
                            true,
                            offset_ms(ts.get(n as usize).copied().flatten(), t_star),
                        ),
                        None => (false, 0.0),
                    };
                    // Warmup turn's lead = t\* - its recorded timestamp, capped.
                    let warm_lead_ms = next_idx.and_then(|n| {
                        (n >= 1)
                            .then(|| ts.get((n - 1) as usize).copied().flatten())
                            .flatten()
                            .map(|warm_ts| capped_warmup_lead_ms(t_star - warm_ts, cfg.idle_gap_cap_ms))
                    });
                    LaneDispatch {
                        conversation_id: meta.conversation_id.clone(),
                        warm_lead_ms,
                        first_profiling_offset_ms,
                        has_profiling,
                        has_warmup,
                    }
                })
                .collect()
        };

        // 2) Cross-lane phase-start offsets (ms) via the ported timing kernel.
        let offsets_ms: Vec<f64> = match cfg.phase {
            AgenticPhase::Warmup => {
                let leads: Vec<Option<f64>> =
                    lanes.iter().map(|l| l.warm_lead_ms).collect();
                warmup_dispatch_offsets_ms(&leads)
            }
            AgenticPhase::Profiling => {
                let raw: Vec<f64> =
                    lanes.iter().map(|l| l.first_profiling_offset_ms).collect();
                profiling_dispatch_delays_ms(&raw, cfg.burst_phase_starts, cfg.idle_gap_cap_ms)
            }
        };

        // 3) Anchor and schedule each lane's dispatch at its phase-start offset.
        let lead_ns = if runtime.clock().is_virtual() {
            0
        } else {
            SCHEDULE_START_LEAD_NS
        };
        let anchor_ns = runtime.now_ns().saturating_add(lead_ns);
        let source = Rc::clone(&self.source);
        for (lane, offset_ms_val) in lanes.iter().zip(offsets_ms.iter()) {
            let dispatch = match cfg.phase {
                AgenticPhase::Warmup if lane.has_warmup => true,
                AgenticPhase::Profiling if lane.has_profiling => true,
                _ => false,
            };
            if !dispatch {
                continue;
            }
            // The lane's first turn (the ConversationSource replays sequentially;
            // continuation chaining follows recorded delays).
            let session = match source
                .borrow()
                .session_for(&lane.conversation_id, lane.conversation_id.clone())
            {
                Ok(session) => session,
                Err(error) => {
                    tracing::warn!(error = %error, lane = %lane.conversation_id, "agentic lane session failed");
                    continue;
                }
            };
            let first = match session.build_first_turn(None) {
                Ok(turn) => turn,
                Err(error) => {
                    tracing::warn!(error = %error, lane = %lane.conversation_id, "agentic first turn failed");
                    continue;
                }
            };
            let target_ns = anchor_ns.saturating_add((offset_ms_val.max(0.0) * NS_PER_MS) as i64);
            // Profiling chains continuations on response; warmup fires one turn.
            let chain = cfg.phase == AgenticPhase::Profiling;
            schedule_agentic_turn(runtime.clone(), source.clone(), first, target_ns, chain);
        }
        Ok(())
    }
}

/// Schedule one lane turn at `target_ns`, then (when `chain`) recursively schedule
/// its continuation on completion at the recorded inter-turn `delay_ms` from the
/// prior turn's end — Python's per-stream sequential continuation.
fn schedule_agentic_turn(
    runtime: Rc<ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    turn: crate::multiturn::TurnToSend,
    target_ns: i64,
    chain: bool,
) {
    let scheduler = runtime.scheduler();
    scheduler.schedule_at_ns(
        target_ns,
        Box::pin(async move {
            let runtime_c = runtime.clone();
            let source_c = source.clone();
            runtime.issue_turn(
                turn,
                target_ns,
                None,
                Box::new(move |credit, outcome| {
                    Box::pin(async move {
                        if !chain || credit.is_final_turn() {
                            return;
                        }
                        let (delay_ms, next_turn) = {
                            let src = source_c.borrow();
                            let meta = match src.next_turn_metadata(&credit) {
                                Ok(meta) => meta,
                                Err(_) => return,
                            };
                            let next = match src.next_turn(&credit, outcome.to_turn_response()) {
                                Ok(Some(turn)) => turn,
                                _ => return,
                            };
                            (meta.delay_ms.unwrap_or(0.0), next)
                        };
                        let next_target = outcome
                            .end_ns
                            .saturating_add((delay_ms.max(0.0) * NS_PER_MS) as i64);
                        schedule_agentic_turn(runtime_c, source_c, next_turn, next_target, true);
                    })
                }),
            );
        }),
    );
}
