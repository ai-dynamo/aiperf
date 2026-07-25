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

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agentx::cache_bust::CacheBustTarget;
use crate::agentx::trajectory_source::{
    profiling_dispatch_delays_ms, warmup_dispatch_offsets_ms,
};
use crate::multiturn::ConversationSource;
use crate::scheduled::{ScheduledRuntime, Workload};
use crate::scheduler::LocalTaskScheduler;

/// Nanoseconds per millisecond.
const NS_PER_MS: f64 = 1_000_000.0;
/// Lead the real-clock scheduling pass by this much so `schedule_at_ns` targets
/// stay in the future while the O(n) pass runs (mirrors fixed_schedule).
const SCHEDULE_START_LEAD_NS: i64 = 25_000_000;

/// Round-robin recycle cursor over the profiling lanes: when a lane's stream
/// completes and the phase budget still permits, the next trajectory is redrawn
/// (Python `next_recycle_conversation_id` with the `sequential` sampler) and
/// dispatched immediately to sustain a duration run.
struct RecycleState {
    /// Profiling conversation ids (templates) in dataset order.
    ids: Vec<String>,
    /// Next index to draw.
    cursor: Cell<usize>,
    /// Per-base-trace recycle pass (Python `CacheBustLedger.recycle_pass`): the
    /// baked dataset marker is pass 0, so each recycle mints the next pass.
    recycle_pass: RefCell<std::collections::HashMap<String, i64>>,
    /// Benchmark id for the cache-bust digest.
    benchmark_id: String,
    /// Cache-bust placement.
    cache_bust_target: CacheBustTarget,
}

/// One drawn recycle instance: which template to replay, its fresh correlation
/// id, and the fresh cache-bust marker to swap into the first turn's body.
struct RecycleDraw {
    template: String,
    correlation: String,
    marker: Option<String>,
}

impl RecycleState {
    /// Draw the next recycle instance: round-robin template, a unique correlation
    /// (Python's double-recycle guard), and a freshly-minted cache-bust marker at
    /// the next `recycle_pass` for that base trace.
    fn next_draw(&self) -> Option<RecycleDraw> {
        if self.ids.is_empty() {
            return None;
        }
        let i = self.cursor.get();
        self.cursor.set(i + 1);
        let template = self.ids[i % self.ids.len()].clone();
        let correlation = format!("{template}#r{i}");
        let base = crate::agentx::cache_bust::base_trace_id(&template).to_string();
        let pass = {
            let mut passes = self.recycle_pass.borrow_mut();
            let entry = passes.entry(base.clone()).or_insert(0);
            *entry += 1; // baked dataset marker is pass 0
            *entry
        };
        let marker = crate::agentx::cache_bust::build_cache_bust_marker(
            &self.benchmark_id,
            pass,
            i as i64,
            &base,
            self.cache_bust_target,
        );
        Some(RecycleDraw {
            template,
            correlation,
            marker,
        })
    }
}

/// Rewrite the first message's content in a chat `request_body`, swapping any
/// existing `[rid:<hex>]\n\n` cache-bust prefix for `marker` (Python mints a
/// fresh marker per recycle at request-build time). No-op if the body cannot be
/// parsed or `marker` is `None`.
fn rewrite_first_turn_marker(turn: &mut crate::multiturn::TurnToSend, marker: &str) {
    let Some(body) = &turn.request_body else {
        return;
    };
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return;
    };
    let Some(content) = value
        .get_mut("messages")
        .and_then(|m| m.get_mut(0))
        .and_then(|m| m.get_mut("content"))
        .and_then(|c| c.as_str().map(str::to_string))
    else {
        return;
    };
    // Strip an existing `[rid:<12hex>]\n\n` prefix (the baked pass-0 marker).
    let stripped = strip_rid_prefix(&content);
    let new_content = format!("{marker}{stripped}");
    if let Some(msg) = value
        .get_mut("messages")
        .and_then(|m| m.get_mut(0))
        .and_then(|m| m.as_object_mut())
    {
        msg.insert("content".into(), serde_json::Value::String(new_content));
        if let Ok(bytes) = serde_json::to_vec(&value) {
            turn.request_body = Some(bytes.into());
        }
    }
}

/// Strip a leading `[rid:<12hex>]\n\n` cache-bust marker, if present.
fn strip_rid_prefix(content: &str) -> &str {
    if let Some(rest) = content.strip_prefix("[rid:")
        && let Some(close) = rest.find(']')
        && rest[..close].len() == 12
        && rest[..close].bytes().all(|b| b.is_ascii_hexdigit())
    {
        return rest[close + 1..].strip_prefix("\n\n").unwrap_or(&rest[close + 1..]);
    }
    content
}

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
        // 1) Read each lane's t\*-relative dispatch offsets. The snapshot slice
        // ([`crate::agentx::weka_dataset::slice_trajectories_at_tstar`]) already
        // sampled t\*, excluded history, and rebased `timestamp_ms` to the offset
        // from t\* — so the lane's first turn's timestamp is its phase-start offset.
        let want_warmup = cfg.phase == AgenticPhase::Warmup;
        let lanes: Vec<LaneDispatch> = {
            let source = self.source.borrow();
            source
                .conversations()
                .iter()
                // Warmup convs carry the `::warmup` suffix; each phase dispatches
                // only its own conversations (Python's separate warmup barrier).
                .filter(|meta| {
                    meta.conversation_id.ends_with(crate::agentx::weka_dataset::WARMUP_SUFFIX)
                        == want_warmup
                })
                .map(|meta| {
                    let first_offset = meta
                        .turns
                        .first()
                        .and_then(|t| t.timestamp_ms)
                        .unwrap_or(0.0);
                    LaneDispatch {
                        conversation_id: meta.conversation_id.clone(),
                        warm_lead_ms: None,
                        first_profiling_offset_ms: first_offset,
                        has_profiling: !meta.turns.is_empty(),
                        has_warmup: !meta.turns.is_empty(),
                    }
                })
                .collect()
        };

        // 2) Cross-lane phase-start alignment (ms) via the ported timing kernel:
        // profiling applies the leading-idle cap + burst/spread t0; warmup uses the
        // global t\*-alignment (largest lead at 0). The per-lane offsets are the
        // pre-baked t\*-relative first-turn offsets.
        let raw: Vec<f64> = lanes.iter().map(|l| l.first_profiling_offset_ms).collect();
        let offsets_ms: Vec<f64> = match cfg.phase {
            AgenticPhase::Warmup => {
                let leads: Vec<Option<f64>> =
                    lanes.iter().map(|l| Some(l.first_profiling_offset_ms)).collect();
                warmup_dispatch_offsets_ms(&leads)
            }
            AgenticPhase::Profiling => {
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
        // Profiling recycles exhausted trajectories to sustain a duration run;
        // warmup is a one-shot prime with no recycle.
        let recycle = (cfg.phase == AgenticPhase::Profiling).then(|| {
            Rc::new(RecycleState {
                ids: lanes.iter().map(|l| l.conversation_id.clone()).collect(),
                cursor: Cell::new(0),
                recycle_pass: RefCell::new(std::collections::HashMap::new()),
                benchmark_id: cfg.benchmark_id.clone(),
                cache_bust_target: cfg.cache_bust_target,
            })
        });
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
            schedule_agentic_turn(
                runtime.clone(),
                source.clone(),
                first,
                target_ns,
                chain,
                recycle.clone(),
            );
        }
        Ok(())
    }
}

/// Schedule one lane turn at `target_ns`, then (when `chain`) recursively schedule
/// its continuation on completion at the recorded inter-turn `delay_ms` from the
/// prior turn's end — Python's per-stream sequential continuation.
#[allow(clippy::too_many_arguments)]
fn schedule_agentic_turn(
    runtime: Rc<ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    turn: crate::multiturn::TurnToSend,
    target_ns: i64,
    chain: bool,
    recycle: Option<Rc<RecycleState>>,
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
                        if credit.is_final_turn() {
                            // The stream is done. Recycle a fresh trajectory to
                            // sustain the run while the phase budget permits.
                            if let Some(recycle) = &recycle
                                && runtime_c.can_issue(true)
                                && let Some(draw) = recycle.next_draw()
                            {
                                let session = source_c
                                    .borrow()
                                    .session_for(&draw.template, draw.correlation);
                                if let Ok(session) = session
                                    && let Ok(mut first) = session.build_first_turn(None)
                                {
                                    // Fresh cache-bust marker for the recycled tree.
                                    if let Some(marker) = &draw.marker {
                                        rewrite_first_turn_marker(&mut first, marker);
                                    }
                                    let now = runtime_c.now_ns();
                                    schedule_agentic_turn(
                                        runtime_c,
                                        source_c,
                                        first,
                                        now,
                                        true,
                                        Some(recycle.clone()),
                                    );
                                }
                            }
                            return;
                        }
                        if !chain {
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
                        schedule_agentic_turn(
                            runtime_c, source_c, next_turn, next_target, true, recycle,
                        );
                    })
                }),
            );
        }),
    );
}
