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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;

use crate::agentx::cache_bust::CacheBustTarget;
use crate::agentx::handoff::{
    AcceleratedObserver, FinalizeInputs, HandoffBaseDelayInputs, HandoffCredit, HandoffRecorder,
    LegacyWarmupHandoff, PrevLaneTrajectory, finalize, finish_accelerated,
};
use crate::agentx::replay_gate::ReplayGate;
use crate::agentx::session_tree::{PhaseKey, SessionTreeRegistry, SlotReleaser};
use crate::agentx::trajectory_source::{profiling_dispatch_delays_ms, warmup_dispatch_offsets_ms};
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
    let Ok(wire) = body.to_wire() else {
        return;
    };
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&wire) else {
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
            turn.request_body = Some(crate::body_plan::RequestBody::wire(bytes.into()));
        }
    }
}

/// Resolve the requested output-token count for an issued turn under the
/// accelerated cache-warmup pressure override. `Some(n)` forces `n` (Python's
/// `_WARMUP_MAX_TOKENS=1` on every credit), overriding whatever the recorded
/// turn carried; `None` leaves the recorded `recorded` value untouched.
fn effective_max_output_tokens(recorded: usize, max_tokens_override: Option<u32>) -> usize {
    max_tokens_override.map_or(recorded, |n| n as usize)
}

/// Apply [`AgenticReplayConfig::max_tokens_override`] to a freshly built turn:
/// when set, the turn's requested output tokens are forced to the override,
/// mirroring Python's per-credit `max_tokens=1` cache-pressure warmup.
fn apply_max_tokens_override(
    turn: &mut crate::multiturn::TurnToSend,
    max_tokens_override: Option<u32>,
) {
    turn.max_output_tokens =
        effective_max_output_tokens(turn.max_output_tokens, max_tokens_override);
}

/// Strip a leading `[rid:<12hex>]\n\n` cache-bust marker, if present.
fn strip_rid_prefix(content: &str) -> &str {
    if let Some(rest) = content.strip_prefix("[rid:")
        && let Some(close) = rest.find(']')
        && rest[..close].len() == 12
        && rest[..close].bytes().all(|b| b.is_ascii_hexdigit())
    {
        return rest[close + 1..]
            .strip_prefix("\n\n")
            .unwrap_or(&rest[close + 1..]);
    }
    content
}

/// Per-lane dispatch decision computed from the trajectory's recorded timestamps.
struct LaneDispatch {
    /// The conversation/session template id.
    conversation_id: String,
    /// The first post-t\* turn's offset from t\* (profiling), or 0 when none.
    first_profiling_offset_ms: f64,
    /// Whether the lane has any post-t\* (profiling) turn.
    has_profiling: bool,
    /// Whether the lane has a warmup turn (a pre-t\* turn to prime).
    has_warmup: bool,
}

/// Cross-phase carrier for the accelerated cache-warmup handoff — the legacy
/// analogue of the graph path's `Rc<RefCell<Option<GraphWarmupHandoff>>>`
/// ([`crate::engine::graph_phase_runtime`]). The WARMUP phase's `execute`
/// populates it at drain/finalize; the PROFILING phase's `execute` reads it to
/// resume each lane at its residual frontier. `Arc<Mutex<..>>` (not `Rc<RefCell>`)
/// so the carrier can ride the `Send + Sync` shared run resources; it is written
/// and read only at phase boundaries on the single global-hop coordinator, never
/// on a per-request/token path.
pub type WarmupHandoffCarrier = Arc<Mutex<Option<LegacyWarmupHandoff>>>;

/// Construct a fresh, empty typed accelerated-warmup carrier. The agentic run
/// assembly (`lower_legacy_agentic`) creates one and stores it type-erased on the
/// prepared dataset so both agentic phase instances share it.
pub fn new_warmup_handoff_carrier() -> WarmupHandoffCarrier {
    Arc::new(Mutex::new(None))
}

/// Downcast a type-erased [`WarmupHandoffCarrierAny`](crate::agentic_tree::WarmupHandoffCarrierAny)
/// to the typed carrier, or `None` for the empty non-agentic carrier.
pub fn downcast_warmup_handoff_carrier(
    any: &crate::agentic_tree::WarmupHandoffCarrierAny,
) -> Option<WarmupHandoffCarrier> {
    any.clone()
        .downcast::<Mutex<Option<LegacyWarmupHandoff>>>()
        .ok()
}

/// Per-lane recorded metadata the accelerated-warmup substage needs to project
/// the drained frontier into residual handoff delays.
struct LaneMeta {
    /// Template/conversation id of the live lane.
    conversation_id: String,
    /// Number of recorded turns available in the lane.
    num_turns: usize,
    /// Recorded relative `delay_ms` per turn index (`None` when absent).
    turn_delays_ms: Vec<Option<f64>>,
    /// Recorded absolute `timestamp_ms` per turn index (`None` when absent).
    turn_timestamps_ms: Vec<Option<f64>>,
}

/// Project a barrier-retained [`ReplayTurn`](crate::agentx::replay_gate::ReplayTurn)
/// into a [`PendingHandoffTurn`](crate::agentx::handoff::PendingHandoffTurn) for the
/// finalize projection, filling the linear-MVP DAG defaults (agent_depth 0, root =
/// self). `num_turns` is recovered from the lane's recorded turn count.
fn pending_handoff_turn(
    turn: &crate::agentx::replay_gate::ReplayTurn,
    lane_by_conv: &HashMap<String, usize>,
    lanes: &[LaneMeta],
) -> crate::agentx::handoff::PendingHandoffTurn {
    let num_turns = lane_by_conv
        .get(&turn.key.conversation_id)
        .and_then(|&i| lanes.get(i))
        .map_or(0, |l| l.num_turns) as i64;
    crate::agentx::handoff::PendingHandoffTurn {
        conversation_id: turn.key.conversation_id.clone(),
        x_correlation_id: turn.root_id.clone(),
        turn_index: turn.key.turn_index,
        num_turns,
        agent_depth: 0,
        parent_correlation_id: None,
        root_correlation_id: None,
        branch_mode: crate::agentx::handoff::BranchMode::default(),
    }
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
    /// Global system-idle cap in ms; shifts pending replay issuance without rewriting trace timing.
    pub system_idle_gap_cap_ms: Option<f64>,
    /// Anchor phase-start bursts at the earliest post-t\* request instead of spread.
    pub burst_phase_starts: bool,
    /// Base random seed for per-lane t\* sampling.
    pub random_seed: u64,
    /// The run's benchmark id (cache-bust digest input).
    pub benchmark_id: String,
    /// Cache-bust placement (scenario-locked to first-turn-prefix for the MVP).
    pub cache_bust_target: CacheBustTarget,
    /// Side-channel subagent join-gate specs built from the reconstruction. Empty
    /// for a run with no subagent trees — the workload's join gate then stays a
    /// pass-through (never defers, recycles as before).
    pub trees: Rc<Vec<TreeSpec>>,
    /// Accelerated cache-warmup duration in seconds (Python
    /// `--agentic-cache-warmup-duration`), threaded from the WARMUP phase's
    /// authored `agentic_cache_warmup_duration`. `None` (the default) leaves the
    /// warmup phase as the standard turn-(n-1) prime; a later stage consumes this
    /// to drive the accelerated substage. Absent on the PROFILING instance.
    pub cache_warmup_duration_s: Option<f64>,
    /// Optional per-turn `max_tokens` override for the accelerated cache-warmup
    /// substage (`None` = use each turn's recorded cap). Set to `Some(1)` on the
    /// WARMUP instance when `cache_warmup_duration_s` is present so every pressure
    /// credit forces single-token generation (Python `_WARMUP_MAX_TOKENS=1`).
    pub max_tokens_override: Option<u32>,
    /// Cross-phase accelerated-warmup handoff carrier. The WARMUP instance writes
    /// the drained [`LegacyWarmupHandoff`] here at finalize; the PROFILING instance
    /// reads it to resume each lane at its residual frontier. An empty carrier
    /// (`None` inside) on PROFILING means the non-accelerated path — profiling runs
    /// exactly as today. Default-constructed (empty) for every non-accelerated run.
    pub warmup_handoff: WarmupHandoffCarrier,
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
    pub fn new(source: Box<dyn ConversationSource>, config: AgenticReplayConfig) -> Result<Self> {
        Ok(Self {
            source: Rc::new(RefCell::new(source)),
            config,
        })
    }
}

/// Clone the accelerated-warmup handoff without extending the carrier lock into
/// profiling resume.
fn profiling_warmup_handoff(carrier: &WarmupHandoffCarrier) -> Option<LegacyWarmupHandoff> {
    carrier.lock().ok().and_then(|handoff| handoff.clone())
}

#[cfg(test)]
mod warmup_handoff_tests {
    use super::*;

    #[test]
    fn profiling_handoff_clone_releases_carrier_lock_before_resume() {
        let carrier = new_warmup_handoff_carrier();
        let handoff = LegacyWarmupHandoff::default();
        *carrier.lock().unwrap() = Some(handoff.clone());

        assert_eq!(profiling_warmup_handoff(&carrier), Some(handoff));
        assert!(carrier.try_lock().is_ok());
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
        // Accelerated cache-pressure warmup: instead of the static turn-(n-1)
        // prime, continue replaying the live profiling trajectories under
        // compressed traffic (zero idle, `max_tokens=1`) for the configured
        // wall-duration, then hand the drained per-lane frontier to PROFILING via
        // the carrier (Python `_start_accelerated_warmup` / `finalize_phase`).
        if cfg.phase == AgenticPhase::Warmup
            && cfg
                .cache_warmup_duration_s
                .is_some_and(|d| d.is_finite() && d > 0.0)
        {
            return self.execute_accelerated_warmup(runtime).await;
        }
        // PROFILING resume from a populated accelerated-warmup carrier (Python
        // `setup_phase` prefix-reseed + residual-offset dispatch). An empty carrier
        // leaves the profiling path EXACTLY as the non-accelerated run.
        if cfg.phase == AgenticPhase::Profiling
            && let Some(handoff) = profiling_warmup_handoff(&cfg.warmup_handoff)
        {
            return self.execute_profiling_resume(runtime, handoff).await;
        }
        self.execute_standard(runtime).await
    }
}

fn system_idle_continuation_delay_ms(
    delay_ms: f64,
    cap_ms: Option<f64>,
    scheduler_task_count: usize,
) -> f64 {
    match cap_ms {
        Some(cap_ms) if scheduler_task_count <= 1 => delay_ms.min(cap_ms),
        _ => delay_ms,
    }
}

fn cap_system_idle_offsets_ms(offsets_ms: &[f64], cap_ms: Option<f64>) -> Vec<f64> {
    let Some(cap_ms) = cap_ms else {
        return offsets_ms.to_vec();
    };
    if offsets_ms.is_empty() {
        return Vec::new();
    }

    let mut adjusted = offsets_ms.to_vec();
    let mut indices: Vec<usize> = (0..adjusted.len()).collect();
    indices.sort_by(|&left, &right| adjusted[left].total_cmp(&adjusted[right]));

    let mut previous = 0.0;
    let mut total_shift = 0.0;
    for idx in indices {
        let shifted = adjusted[idx] - total_shift;
        let gap = shifted - previous;
        if gap > cap_ms {
            total_shift += gap - cap_ms;
        }
        adjusted[idx] = (adjusted[idx] - total_shift).max(previous);
        previous = adjusted[idx];
    }
    adjusted
}

impl AgenticReplayWorkload {
    /// Standard (non-accelerated) warmup/profiling dispatch: warmup primes each
    /// `::warmup` turn-(n-1) once; profiling replays each post-t\* lane at its
    /// recorded offsets with recycle. Unchanged from the pre-accelerated port.
    async fn execute_standard(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
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
                    meta.conversation_id
                        .ends_with(crate::agentx::weka_dataset::WARMUP_SUFFIX)
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
                let leads: Vec<Option<f64>> = lanes
                    .iter()
                    .map(|l| Some(l.first_profiling_offset_ms))
                    .collect();
                warmup_dispatch_offsets_ms(&leads)
            }
            AgenticPhase::Profiling => {
                profiling_dispatch_delays_ms(&raw, cfg.burst_phase_starts, cfg.idle_gap_cap_ms)
            }
        };
        let offsets_ms = cap_system_idle_offsets_ms(&offsets_ms, cfg.system_idle_gap_cap_ms);

        // 3) Anchor and schedule each lane's dispatch at its phase-start offset.
        let lead_ns = if runtime.clock().is_virtual() {
            0
        } else {
            SCHEDULE_START_LEAD_NS
        };
        let anchor_ns = runtime.now_ns().saturating_add(lead_ns);
        let source = Rc::clone(&self.source);
        // Subagent-join gate. The tree specs are threaded from lowering
        // (`build_tree_specs` over the sliced profiling conversations). A run
        // with no subagent trees carries an empty `Vec`, so the gate is a
        // pass-through (`None`) that never defers and recycles as before.
        let tree_specs = &cfg.trees;
        let gate: Option<Rc<TreeGate>> = if tree_specs.is_empty() {
            None
        } else {
            Some(Rc::new(TreeGate::try_new(tree_specs)?))
        };
        // Per-run deferral queue for gated join turns (drained on child terminal).
        let defer_queue: Rc<RefCell<Vec<PendingJoin>>> = Rc::new(RefCell::new(Vec::new()));
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
            let mut first = match session.build_first_turn(None) {
                Ok(turn) => turn,
                Err(error) => {
                    tracing::warn!(error = %error, lane = %lane.conversation_id, "agentic first turn failed");
                    continue;
                }
            };
            // Accelerated cache-warmup forces `max_tokens` on every issued credit
            // (Python `_WARMUP_MAX_TOKENS=1`); a `None` override is a no-op.
            apply_max_tokens_override(&mut first, cfg.max_tokens_override);
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
                gate.clone(),
                defer_queue.clone(),
                // Standard warmup/profiling path: no accelerated observer.
                None,
                AccelCtx::default(),
                cfg.system_idle_gap_cap_ms,
            );
        }
        Ok(())
    }

    /// Accelerated cache-pressure warmup substage (Python `_start_accelerated_warmup`
    /// / `_dispatch_accelerated_trajectory` / `_finish_accelerated_warmup` /
    /// `finalize_phase`).
    ///
    /// Pressure-replays each live profiling lane (the non-`::warmup` conversations,
    /// which `slice_trajectories_at_tstar` already rebased to start at their post-t\*
    /// turn 0) from turn 0 at zero idle delay with `max_tokens=1`, chaining
    /// continuations under compression. A Clock-scheduled duration timer sets the
    /// drain latch (no new issuance), pauses the replay barrier gate, and cancels
    /// pending dispatches; already-issued requests drain via `wait_idle`. The drained
    /// DAG is then projected into a [`LegacyWarmupHandoff`] and published on the
    /// carrier for PROFILING to resume from.
    async fn execute_accelerated_warmup(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        let cfg = &self.config;
        let duration_s = cfg
            .cache_warmup_duration_s
            .expect("accelerated warmup requires a duration");

        // Live lanes = the profiling (non-`::warmup`) conversations, in dataset
        // order. Each is a warmup pressure lane AND its own linear-MVP tree root.
        let lanes: Vec<LaneMeta> = {
            let source = self.source.borrow();
            source
                .conversations()
                .iter()
                .filter(|meta| {
                    !meta
                        .conversation_id
                        .ends_with(crate::agentx::weka_dataset::WARMUP_SUFFIX)
                })
                .map(|meta| LaneMeta {
                    conversation_id: meta.conversation_id.clone(),
                    num_turns: meta.turns.len(),
                    // Recorded inter-turn delays (delay_ms per turn index), used for
                    // the residual base after compression.
                    turn_delays_ms: meta.turns.iter().map(|t| t.delay_ms).collect(),
                    turn_timestamps_ms: meta.turns.iter().map(|t| t.timestamp_ms).collect(),
                })
                .collect()
        };

        // Barrier gate + return-observation recorder. Linear MVP: no cross-stream
        // predecessors (each lane is an independent tree root), so the gate is a
        // pass-through until paused for drain; it still records completed prefixes
        // and retained pending turns for the handoff.
        let mut gate = ReplayGate::new(BTreeMap::new());
        gate.activate();
        let observer = Rc::new(AcceleratedObserver::new(gate, HandoffRecorder::new()));

        // Lane bookkeeping for the finalize projection: root/correlation -> lane.
        let mut root_to_lane: BTreeMap<String, usize> = BTreeMap::new();
        let mut correlation_to_lane: BTreeMap<String, usize> = BTreeMap::new();
        let mut prev_lanes: Vec<PrevLaneTrajectory> = Vec::new();

        let anchor_ns = runtime.now_ns();
        let accel = AccelCtx {
            draining: Rc::new(Cell::new(false)),
            zero_idle: true,
            max_tokens_override: cfg.max_tokens_override,
        };
        let defer_queue: Rc<RefCell<Vec<PendingJoin>>> = Rc::new(RefCell::new(Vec::new()));

        for (lane_idx, lane) in lanes.iter().enumerate() {
            if lane.num_turns == 0 {
                prev_lanes.push(PrevLaneTrajectory {
                    conversation_id: lane.conversation_id.clone(),
                    x_correlation_id: lane.conversation_id.clone(),
                });
                continue;
            }
            // Linear MVP: correlation id == conversation id (unique per lane), and
            // the lane is its own tree root.
            let corr = lane.conversation_id.clone();
            root_to_lane.insert(corr.clone(), lane_idx);
            correlation_to_lane.insert(corr.clone(), lane_idx);
            prev_lanes.push(PrevLaneTrajectory {
                conversation_id: lane.conversation_id.clone(),
                x_correlation_id: corr.clone(),
            });

            let session = match self
                .source
                .borrow()
                .session_for(&lane.conversation_id, corr)
            {
                Ok(session) => session,
                Err(error) => {
                    tracing::warn!(error = %error, lane = %lane.conversation_id, "accelerated warmup lane session failed");
                    continue;
                }
            };
            // `build_turn_at(start_turn_index + 1)`; the sliced profiling lane's
            // start turn is turn 0, so the pressure replay begins at turn 0.
            let mut first = match session.build_turn_at(0, None) {
                Ok(turn) => turn,
                Err(error) => {
                    tracing::warn!(error = %error, lane = %lane.conversation_id, "accelerated warmup first turn failed");
                    continue;
                }
            };
            apply_max_tokens_override(&mut first, cfg.max_tokens_override);
            schedule_agentic_turn(
                runtime.clone(),
                self.source.clone(),
                first,
                anchor_ns,
                true,
                None,
                None,
                defer_queue.clone(),
                Some(observer.clone()),
                accel.clone(),
                None,
            );
        }

        // Arm the Clock-driven drain timer: at +duration set the drain latch, pause
        // the barrier gate (Python `_finish_accelerated_warmup`), and cancel pending
        // (not-yet-issued) dispatches so no new pressure work starts.
        let duration_ns = (duration_s * 1_000_000_000.0) as i64;
        {
            let observer = observer.clone();
            let draining = accel.draining.clone();
            let runtime = runtime.clone();
            runtime.clone().scheduler().schedule_later(
                duration_ns,
                Box::pin(async move {
                    draining.set(true);
                    finish_accelerated(&observer);
                    runtime.scheduler().cancel_pending();
                }),
            );
        }
        // Await the drain timer + in-flight settle. The drain timer is a tracked
        // task, so `wait_idle` cannot resolve before +duration; after it fires and
        // cancels pending, the in-flight requests drain and this resolves.
        runtime.scheduler().wait_idle().await;

        // Finalize: project the drained DAG into the carrier (Python `finalize_phase`).
        let finalized_ns = runtime.now_ns();
        let lane_by_conv: HashMap<String, usize> = lanes
            .iter()
            .enumerate()
            .map(|(i, l)| (l.conversation_id.clone(), i))
            .collect();
        let base_delay_inputs = |credit: &HandoffCredit| -> HandoffBaseDelayInputs {
            let Some(&lane_idx) = lane_by_conv.get(&credit.conversation_id) else {
                return HandoffBaseDelayInputs::default();
            };
            let lane = &lanes[lane_idx];
            let next = credit.turn_index + 1;
            HandoffBaseDelayInputs {
                next_delay_ms: lane.turn_delays_ms.get(next).copied().flatten(),
                prev_timestamp_ms: lane
                    .turn_timestamps_ms
                    .get(credit.turn_index)
                    .copied()
                    .flatten(),
                next_timestamp_ms: lane.turn_timestamps_ms.get(next).copied().flatten(),
                prev_api_time_ms: None,
            }
        };
        let pending_by_root = observer
            .gate
            .borrow()
            .pending_turns_by_root()
            .into_iter()
            .map(|(root, turns)| {
                let turns = turns
                    .into_iter()
                    .map(|t| pending_handoff_turn(&t, &lane_by_conv, &lanes))
                    .collect::<Vec<_>>();
                (root, turns)
            })
            .collect::<BTreeMap<_, _>>();
        let completed_prefixes =
            |root: &str| -> Vec<crate::agentx::replay_dependencies::ReplayResumeBoundary> {
                observer
                    .gate
                    .borrow()
                    .completed_prefixes(root)
                    .unwrap_or_default()
            };
        // Empty lanes (a fully-drained tree) recycle a fresh root, drawing the lane's
        // own template with a fresh correlation id (Python `next_recycle_conversation_id`).
        let recycle_cursor = Cell::new(0usize);
        let recycle_ids: Vec<String> = lanes.iter().map(|l| l.conversation_id.clone()).collect();
        let recycle_draw = || -> Option<(String, String)> {
            if recycle_ids.is_empty() {
                return None;
            }
            let i = recycle_cursor.get();
            recycle_cursor.set(i + 1);
            let template = recycle_ids[i % recycle_ids.len()].clone();
            let corr = format!("{template}#w{i}");
            Some((template, corr))
        };

        let handoff = {
            let recorder = observer.recorder.borrow();
            finalize(FinalizeInputs {
                handoff_credits: recorder.handoff_credits(),
                return_wall_ns: recorder.return_wall_ns(),
                pending_by_root: &pending_by_root,
                root_to_lane: &root_to_lane,
                correlation_to_lane: &correlation_to_lane,
                num_lanes: lanes.len(),
                finalized_ns,
                cap_ms: cfg.idle_gap_cap_ms,
                base_delay_inputs,
                completed_prefixes,
                recycle_draw,
                prev_lanes: &prev_lanes,
            })
        };
        if let Ok(mut guard) = cfg.warmup_handoff.lock() {
            *guard = Some(handoff);
        }
        Ok(())
    }

    /// PROFILING resume from the accelerated-warmup carrier (Python `setup_phase`
    /// prefix-reseed + residual-offset dispatch). Each surviving lane state is
    /// resumed at its true `next_turn_index` via `build_turn_at` at its residual
    /// `next_dispatch_offset_ms`; recycled (empty-warmup-drained) lanes start a
    /// fresh root at turn 0. The barrier gate is re-seeded with each lane's
    /// completed prefixes and re-activated (inert for the linear MVP, kept for
    /// parity). Continuations chain at recorded cadence with recorded `max_tokens`.
    async fn execute_profiling_resume(
        &self,
        runtime: Rc<ScheduledRuntime>,
        handoff: LegacyWarmupHandoff,
    ) -> Result<()> {
        let cfg = &self.config;
        let lead_ns = if runtime.clock().is_virtual() {
            0
        } else {
            SCHEDULE_START_LEAD_NS
        };
        let anchor_ns = runtime.now_ns().saturating_add(lead_ns);

        // Re-seed the barrier gate with the merged completed prefixes and activate
        // it (Python `setup_phase`). Inert without submit-wiring in the linear MVP,
        // but preserves the parity surface.
        let mut gate = ReplayGate::new(BTreeMap::new());
        for lane in handoff.lanes.values() {
            if let Some(state) = lane.states.first() {
                let root = state.effective_root_correlation_id();
                let _ = gate.seed_completed_prefixes(root, &lane.boundaries);
            }
        }
        gate.activate();

        let tree_specs = &cfg.trees;
        let tree_gate: Option<Rc<TreeGate>> = if tree_specs.is_empty() {
            None
        } else {
            Some(Rc::new(TreeGate::try_new(tree_specs)?))
        };
        let defer_queue: Rc<RefCell<Vec<PendingJoin>>> = Rc::new(RefCell::new(Vec::new()));
        // Recycle over the resumed lane templates to sustain a duration run.
        let recycle_ids: Vec<String> = handoff
            .lanes
            .values()
            .filter_map(|l| l.states.first().map(|s| s.conversation_id.clone()))
            .collect();
        let recycle = Rc::new(RecycleState {
            ids: recycle_ids,
            cursor: Cell::new(0),
            recycle_pass: RefCell::new(HashMap::new()),
            benchmark_id: cfg.benchmark_id.clone(),
            cache_bust_target: cfg.cache_bust_target,
        });

        for lane in handoff.lanes.values() {
            for state in &lane.states {
                let session = match self
                    .source
                    .borrow()
                    .session_for(&state.conversation_id, state.x_correlation_id.clone())
                {
                    Ok(session) => session,
                    Err(error) => {
                        tracing::warn!(error = %error, lane = %state.conversation_id, "profiling resume session failed");
                        continue;
                    }
                };
                let start_index = state.next_turn_index.max(0) as usize;
                let first = match session.build_turn_at(start_index, None) {
                    Ok(turn) => turn,
                    Err(error) => {
                        tracing::warn!(error = %error, lane = %state.conversation_id, index = start_index, "profiling resume turn failed");
                        continue;
                    }
                };
                let target_ns = anchor_ns
                    .saturating_add((state.next_dispatch_offset_ms.max(0.0) * NS_PER_MS) as i64);
                schedule_agentic_turn(
                    runtime.clone(),
                    self.source.clone(),
                    first,
                    target_ns,
                    true,
                    Some(recycle.clone()),
                    tree_gate.clone(),
                    defer_queue.clone(),
                    None,
                    AccelCtx::default(),
                    cfg.system_idle_gap_cap_ms,
                );
            }
        }
        Ok(())
    }
}

/// A join turn deferred because its awaited children are not yet terminal.
///
/// Held in the per-run deferral queue (`Rc<RefCell<Vec<PendingJoin>>>`) until a
/// child's terminal callback clears the gate and re-dispatches it at `now_ns`.
struct PendingJoin {
    /// The gated parent turn to re-issue once its children have drained.
    turn: crate::multiturn::TurnToSend,
    /// Whether the re-dispatched turn chains its continuation on completion.
    chain: bool,
}

/// Drain the deferral `queue` of every entry whose gate join is now satisfied.
///
/// `key` projects a queued item to its `(conversation_id, turn_index)` join
/// coordinate; an item is returned (removed) exactly when
/// [`TreeGate::is_waiting`] no longer holds for that coordinate. Generic over
/// the item type so the decision logic is unit-testable without a full
/// [`TurnToSend`](crate::multiturn::TurnToSend).
///
/// Exposed (`pub`) so integration tests can drive the real deferral/release
/// decision under simulated live-child latency rather than reimplementing it.
pub fn take_ready<T>(
    queue: &RefCell<Vec<T>>,
    gate: &TreeGate,
    key: impl for<'a> Fn(&'a T) -> (&'a str, usize),
) -> Vec<T> {
    let mut q = queue.borrow_mut();
    let mut ready = Vec::new();
    let mut i = 0;
    while i < q.len() {
        let (conv, idx) = key(&q[i]);
        if gate.is_waiting(conv, idx) {
            i += 1;
        } else {
            ready.push(q.remove(i));
        }
    }
    ready
}

/// Schedule one lane turn at `target_ns`, then (when `chain`) recursively schedule
/// its continuation on completion at the recorded inter-turn `delay_ms` from the
/// prior turn's end — Python's per-stream sequential continuation.
///
/// `gate`/`defer_queue` implement subagent-join gating: a turn whose
/// `(conversation_id, turn_index)` is a waiting join point is parked in
/// `defer_queue` instead of issued, and re-dispatched from a child's terminal
/// callback once [`TreeGate::is_waiting`] clears. A `None` gate (the no-tree
/// degenerate case) never defers and recycles exactly as before.
///
/// `observer` is the accelerated cache-warmup return seam (Python
/// `observe_credit_return`, lines 698-717): when `Some`, every credit return
/// advances the replay barrier gate ([`ReplayGate::complete`]) and records the
/// warmup-to-profile handoff, routing the return wall through the injected
/// [`Clock`](crate::clock::Clock). `None` (every current caller — the standard
/// warmup/profiling path) changes no runtime behavior; the accelerated substage
/// (a later task) is the sole caller that arms it.
/// Accelerated-warmup dispatch context threaded alongside the tree gate: the
/// drain latch (set when the duration timer fires — no new continuation/recycle
/// issuance after it) and the zero-idle flag (accelerated pressure fires each
/// continuation immediately, ignoring the recorded inter-turn `delay_ms`).
#[derive(Clone, Default)]
struct AccelCtx {
    /// Set once the accelerated-warmup duration timer fires; consulted before any
    /// new continuation or recycle is scheduled so the DAG drains without new work.
    draining: Rc<Cell<bool>>,
    /// Accelerated pressure fires continuations at zero idle delay (Python
    /// compressed traffic), overriding the recorded inter-turn cadence.
    zero_idle: bool,
    /// Per-credit `max_tokens` override applied to EVERY pressure turn, including
    /// chained continuations (Python `_WARMUP_MAX_TOKENS=1` on every credit).
    max_tokens_override: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn schedule_agentic_turn(
    runtime: Rc<ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    turn: crate::multiturn::TurnToSend,
    target_ns: i64,
    chain: bool,
    recycle: Option<Rc<RecycleState>>,
    gate: Option<Rc<TreeGate>>,
    defer_queue: Rc<RefCell<Vec<PendingJoin>>>,
    observer: Option<Rc<AcceleratedObserver>>,
    accel: AccelCtx,
    system_idle_gap_cap_ms: Option<f64>,
) {
    // Defer a gated join turn until its awaited children terminate. The child
    // terminal callbacks (below) drain the queue and re-dispatch at `now_ns`.
    // The phase lifecycle cancels pending scheduled tasks at phase end, so a
    // never-satisfied join is bounded by the phase, not an ad-hoc timeout.
    if let Some(g) = &gate
        && g.is_waiting(&turn.conversation_id, turn.turn_index)
    {
        defer_queue.borrow_mut().push(PendingJoin { turn, chain });
        return;
    }

    // Captured for the terminal callback's gate accounting (the turn itself is
    // moved into `issue_turn`).
    let conv_id = turn.conversation_id.clone();
    let scheduler = runtime.scheduler();
    scheduler.schedule_at_ns(
        target_ns,
        Box::pin(async move {
            let runtime_c = runtime.clone();
            let source_c = source.clone();
            let gate_c = gate.clone();
            let defer_c = defer_queue.clone();
            let observer_c = observer.clone();
            let accel_c = accel.clone();
            runtime.issue_turn(
                turn,
                target_ns,
                None,
                Box::new(move |credit, outcome| {
                    Box::pin(async move {
                        // Accelerated cache-warmup return observation (Python
                        // `observe_credit_return`): advance the replay barrier
                        // gate, then record/pop the warmup-to-profile handoff.
                        // Runs on BOTH final and non-final returns, before the
                        // final/no-chain early returns below. `None` observer is
                        // the standard path and is a no-op here.
                        if let Some(obs) = &observer_c {
                            let projection =
                                crate::agentx::handoff::HandoffCredit::from_credit(&credit);
                            let root_id = credit.turn.x_correlation_id.clone();
                            let gate_key = crate::agentx::replay_dependencies::ReplayTurnKey {
                                conversation_id: credit.turn.conversation_id.clone(),
                                turn_index: credit.turn.turn_index as i64,
                            };
                            obs.gate.borrow_mut().complete(&root_id, gate_key);
                            // Return wall via the injected Clock — never Instant::now.
                            let wall_ns = runtime_c.now_ns();
                            obs.recorder.borrow_mut().observe(
                                projection,
                                credit.is_final_turn(),
                                wall_ns,
                            );
                        }
                        if credit.is_final_turn() {
                            // This conversation reached terminal. Release any
                            // parent join turn waiting on it (a child terminal
                            // FAILURE still counts as done — no success gating),
                            // then dispatch the now-unblocked joins at `now_ns`.
                            if let Some(g) = &gate_c {
                                g.on_child_terminal(&conv_id);
                                let ready = take_ready(&defer_c, g, |pj| {
                                    (pj.turn.conversation_id.as_str(), pj.turn.turn_index)
                                });
                                for pj in ready {
                                    schedule_agentic_turn(
                                        runtime_c.clone(),
                                        source_c.clone(),
                                        pj.turn,
                                        runtime_c.now_ns(),
                                        pj.chain,
                                        recycle.clone(),
                                        gate_c.clone(),
                                        defer_c.clone(),
                                        observer_c.clone(),
                                        accel_c.clone(),
                                        system_idle_gap_cap_ms,
                                    );
                                }
                            }
                            // Recycle only when the whole tree has drained. A
                            // no-tree gate reports drained immediately, so the
                            // recycle behaves exactly as before.
                            let drained =
                                gate_c.as_ref().is_none_or(|g| g.on_lane_terminal(&conv_id));
                            // No new recycle issuance once the accelerated-warmup
                            // drain latch is set (Python `mark_sending_complete`).
                            if drained
                                && !accel_c.draining.get()
                                && let Some(recycle) = &recycle
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
                                        gate_c,
                                        defer_c,
                                        observer_c,
                                        accel_c.clone(),
                                        system_idle_gap_cap_ms,
                                    );
                                }
                            }
                            return;
                        }
                        // Stop chaining new continuations once the drain latch is
                        // set: already-issued requests drain, no new ones start.
                        if !chain || accel_c.draining.get() {
                            return;
                        }
                        let (delay_ms, next_turn) = {
                            let src = source_c.borrow();
                            // A truncated chain is invisible downstream — the lane
                            // just stops producing turns — so a failure here must
                            // say so. `Ok(None)` is the ordinary end of a recorded
                            // conversation and stays silent.
                            let meta = match src.next_turn_metadata(&credit) {
                                Ok(meta) => meta,
                                Err(error) => {
                                    tracing::warn!(
                                        error = %error,
                                        conversation = %credit.turn.conversation_id,
                                        turn_index = credit.turn.turn_index,
                                        "agentic continuation metadata failed; chain truncated"
                                    );
                                    return;
                                }
                            };
                            let mut next = match src.next_turn(&credit, outcome.to_turn_response())
                            {
                                Ok(Some(turn)) => turn,
                                Ok(None) => return,
                                Err(error) => {
                                    tracing::warn!(
                                        error = %error,
                                        conversation = %credit.turn.conversation_id,
                                        turn_index = credit.turn.turn_index,
                                        "agentic continuation failed; chain truncated"
                                    );
                                    return;
                                }
                            };
                            // Force the pressure output cap on every chained credit.
                            apply_max_tokens_override(&mut next, accel_c.max_tokens_override);
                            (meta.delay_ms.unwrap_or(0.0), next)
                        };
                        // Accelerated pressure fires continuations at zero idle;
                        // the standard path honors the recorded inter-turn delay.
                        let effective_delay_ms = if accel_c.zero_idle {
                            0.0
                        } else {
                            system_idle_continuation_delay_ms(
                                delay_ms,
                                system_idle_gap_cap_ms,
                                runtime_c.scheduler().task_count(),
                            )
                        };
                        let next_target = outcome
                            .end_ns
                            .saturating_add((effective_delay_ms.max(0.0) * NS_PER_MS) as i64);
                        schedule_agentic_turn(
                            runtime_c,
                            source_c,
                            next_turn,
                            next_target,
                            true,
                            recycle,
                            gate_c,
                            defer_c,
                            observer_c,
                            accel_c,
                            system_idle_gap_cap_ms,
                        );
                    })
                }),
            );
        }),
    );
}

pub use crate::agentic_tree::TreeSpec;

/// Build the side `Vec<TreeSpec>` join-gate description from reconstructed
/// conversations, grouping each root with its subagent children and its join
/// turns.
///
/// A root is a conversation with `parent_conversation_id == None`; its children
/// are the **full transitive descendant set** — every conversation whose
/// `parent_conversation_id` chain walks up to that root, at any depth. A depth>1
/// subagent (whose immediate parent is another subagent rather than the root) is
/// flattened into its root's `children` so nested descendants gate the root join
/// and hold recycle, matching Python's recursive `session_tree` descendant
/// registration. WARMUP conversations (the `::warmup` turn-(n-1) primes, which
/// also carry a parent id) are excluded — they are dispatch primes, not tree
/// members. A root with no surviving descendants yields no spec (the gate stays a
/// pass-through).
///
/// Descendants are grouped by walking the explicit `parent_conversation_id`
/// chain (via an `id -> parent_id` map). If that chain is broken (a parent id
/// that names no known conversation), the walk falls back to
/// [`crate::agentx::cache_bust::base_trace_id`] — which strips `::`-suffixes to
/// the root base — to recover the root base id. A conversation whose chain
/// neither terminates at a `None`-parent root nor resolves via the base id is
/// treated as its own root (safe: it forms no spurious join membership) and, if
/// childless, yields no spec.
///
/// `join_turns` are read from each surviving root turn's `join_prerequisite`.
/// **Callers must pass the already-sliced profiling conversations**
/// ([`crate::agentx::weka_dataset::slice_trajectories_at_tstar`]) so the
/// enumerate index is the profiling (post-t\*, history-excluded) turn index the
/// workload gate consults; a join whose turn fell in the dropped history is
/// simply absent. Required child ids are intersected with the tree's surviving
/// children so a child that did not survive slicing cannot form a dangling join
/// (which would otherwise fail [`TreeGate::try_new`] closed).
pub fn build_tree_specs(
    convs: &[crate::agentx::loader::ReconstructedConversation],
) -> Vec<TreeSpec> {
    let is_warmup = |id: &str| id.ends_with(crate::agentx::weka_dataset::WARMUP_SUFFIX);

    // `id -> parent_id` over surviving (non-warmup) conversations, plus the set
    // of known session ids and the roots (parent_conversation_id == None), for
    // O(1) chain-walking.
    let mut parent_of: HashMap<&str, &str> = HashMap::new();
    let mut known: HashSet<&str> = HashSet::new();
    let mut is_root: HashSet<&str> = HashSet::new();
    for conv in convs {
        if is_warmup(&conv.session_id) {
            continue;
        }
        known.insert(conv.session_id.as_str());
        match &conv.parent_conversation_id {
            Some(parent) => {
                parent_of.insert(conv.session_id.as_str(), parent.as_str());
            }
            None => {
                is_root.insert(conv.session_id.as_str());
            }
        }
    }

    // Resolve the tree root of a conversation by walking the explicit
    // parent_conversation_id chain to the `None`-parent ancestor. If the chain
    // breaks (a parent id that names no known conversation), fall back to the
    // `::`-stripped base id. Guard cycles / missing parents with a visited set +
    // a hard step bound; an unresolvable conversation is treated as its own root
    // (safe — it forms no spurious membership under an unrelated tree).
    let resolve_root = |start: &str| -> String {
        let mut cur = start;
        let mut visited: HashSet<&str> = HashSet::new();
        loop {
            if is_root.contains(cur) {
                return cur.to_string();
            }
            if !visited.insert(cur) {
                break; // cycle
            }
            match parent_of.get(cur) {
                Some(&parent) if known.contains(parent) => cur = parent,
                _ => break, // broken chain (parent unknown or absent)
            }
        }
        // Fallback: `::`-stripped base id, if it names a known root.
        let base = crate::agentx::cache_bust::base_trace_id(cur);
        if is_root.contains(base) {
            return base.to_string();
        }
        cur.to_string() // treat as its own root
    };

    // Group the full transitive descendant set under each resolved tree root.
    let mut children_by_parent: HashMap<String, Vec<String>> = HashMap::new();
    for conv in convs {
        if is_warmup(&conv.session_id) || conv.parent_conversation_id.is_none() {
            continue; // roots contribute no membership to themselves
        }
        let root = resolve_root(&conv.session_id);
        if root == conv.session_id {
            continue; // unresolved / self-root descendant: no membership
        }
        children_by_parent
            .entry(root)
            .or_default()
            .push(conv.session_id.clone());
    }

    let mut specs = Vec::new();
    for conv in convs {
        if is_warmup(&conv.session_id) || conv.parent_conversation_id.is_some() {
            continue; // roots only
        }
        let Some(children) = children_by_parent.get(&conv.session_id).cloned() else {
            continue; // no subagent children → no tree/gate
        };
        if children.is_empty() {
            continue;
        }
        let child_set: HashSet<&str> = children.iter().map(String::as_str).collect();
        let join_turns: Vec<(usize, Vec<String>)> = conv
            .turns
            .iter()
            .enumerate()
            .filter_map(|(turn_index, turn)| {
                let join = turn.join_prerequisite.as_ref()?;
                let required: Vec<String> = join
                    .child_session_ids
                    .iter()
                    .filter(|c| child_set.contains(c.as_str()))
                    .cloned()
                    .collect();
                (!required.is_empty()).then_some((turn_index, required))
            })
            .collect();
        specs.push(TreeSpec {
            root: conv.session_id.clone(),
            children,
            join_turns,
        });
    }
    specs
}

/// No-op [`SlotReleaser`]: the [`TreeGate`] uses [`SessionTreeRegistry`] purely
/// for tree-drain accounting and does not itself release concurrency slots.
struct NoopReleaser;

impl SlotReleaser for NoopReleaser {
    fn release_session_slot(&mut self, _phase: &PhaseKey) {}
}

/// The phase key under which the gate opens every tree. The gate does not model
/// per-phase slot release, so a single opaque key suffices.
const GATE_PHASE: &str = "profiling";

/// Pure subagent join-gate + tree-drain accounting over
/// [`SessionTreeRegistry`], driven by the workload through `&self` (shared,
/// current-thread `!Send`) via interior mutability. No `Arc<Mutex>`.
///
/// Construction ([`TreeGate::new`] / [`TreeGate::try_new`]) opens one tree per
/// [`TreeSpec`] with `root_pending=true` and registers each spec's descendants.
/// [`TreeGate::is_waiting`] defers a root join turn until all its required
/// children are terminal; [`TreeGate::on_child_terminal`] records a child
/// terminal and accounts it against the owning tree; [`TreeGate::on_lane_terminal`]
/// accounts a root's or child's terminal turn and reports whether the whole tree
/// has drained.
pub struct TreeGate {
    /// Tree-drain accounting registry (interior-mutable for `&self` driving).
    registry: RefCell<SessionTreeRegistry<NoopReleaser>>,
    /// Per root: its join points as `turn_index -> required child ids`.
    joins: HashMap<String, HashMap<usize, Vec<String>>>,
    /// Every child id -> the root id of the tree that owns it.
    child_root: HashMap<String, String>,
    /// Children observed as terminated (idempotent guard + join satisfaction).
    terminated: RefCell<HashSet<String>>,
}

impl TreeGate {
    /// Build a gate from `specs`, panicking on an invalid spec.
    ///
    /// A join that references a child id absent from that tree's `children` is a
    /// construction-time (fail-closed) error; this constructor `.expect`s
    /// [`TreeGate::try_new`]. Prefer `try_new` where a `Result` is usable.
    pub fn new(specs: &[TreeSpec]) -> Self {
        Self::try_new(specs).expect("TreeGate::new: invalid TreeSpec (dangling join child)")
    }

    /// Build a gate from `specs`, failing closed on an invalid spec.
    ///
    /// Returns an error if any `join_turns` entry references a child id not
    /// present in the owning tree's `children`.
    pub fn try_new(specs: &[TreeSpec]) -> Result<Self> {
        let mut registry = SessionTreeRegistry::new(NoopReleaser);
        let mut joins: HashMap<String, HashMap<usize, Vec<String>>> = HashMap::new();
        let mut child_root: HashMap<String, String> = HashMap::new();

        for spec in specs {
            let child_set: HashSet<&str> = spec.children.iter().map(|c| c.as_str()).collect();
            for (turn_index, required) in &spec.join_turns {
                for child in required {
                    if !child_set.contains(child.as_str()) {
                        anyhow::bail!(
                            "TreeSpec for root {:?} join turn {} references child {:?} \
                             not in its children",
                            spec.root,
                            turn_index,
                            child
                        );
                    }
                }
            }

            registry.open_tree(&spec.root, GATE_PHASE.to_string(), true);
            registry.register_descendants(&spec.root, spec.children.len() as i64);

            for child in &spec.children {
                child_root.insert(child.clone(), spec.root.clone());
            }
            let entry = joins.entry(spec.root.clone()).or_default();
            for (turn_index, required) in &spec.join_turns {
                entry
                    .entry(*turn_index)
                    .or_default()
                    .extend(required.iter().cloned());
            }
        }

        Ok(Self {
            registry: RefCell::new(registry),
            joins,
            child_root,
            terminated: RefCell::new(HashSet::new()),
        })
    }

    /// True iff `conversation_id` is a root with a join at `turn_index` whose
    /// required children are not all terminated yet.
    pub fn is_waiting(&self, conversation_id: &str, turn_index: usize) -> bool {
        let Some(root_joins) = self.joins.get(conversation_id) else {
            return false;
        };
        let Some(required) = root_joins.get(&turn_index) else {
            return false;
        };
        let terminated = self.terminated.borrow();
        required.iter().any(|c| !terminated.contains(c))
    }

    /// Record `child_id` as terminal and account it against the owning tree
    /// (idempotent: a repeat is ignored).
    pub fn on_child_terminal(&self, child_id: &str) {
        if !self.terminated.borrow_mut().insert(child_id.to_string()) {
            return;
        }
        if let Some(root) = self.child_root.get(child_id) {
            self.registry.borrow_mut().on_descendant_done(root);
        }
    }

    /// Account a lane's terminal turn. For a root, clears root-pending; for a
    /// child, folds in as [`Self::on_child_terminal`]. Returns `true` only when
    /// the whole tree has drained (root terminal AND all descendants done).
    pub fn on_lane_terminal(&self, conversation_id: &str) -> bool {
        if self.joins.contains_key(conversation_id)
            || self.registry.borrow().has_tree(conversation_id)
        {
            // A root lane (either it declared joins or is a tracked tree).
            return self.registry.borrow_mut().on_root_terminal(conversation_id);
        }
        // Otherwise treat as a child terminal; report drain of the owning tree.
        let first_time = self
            .terminated
            .borrow_mut()
            .insert(conversation_id.to_string());
        if let Some(root) = self.child_root.get(conversation_id).cloned() {
            if first_time {
                return self.registry.borrow_mut().on_descendant_done(&root);
            }
            // A repeat child terminal (e.g. already folded by `on_child_terminal`)
            // is not itself a fresh drain event.
            return false;
        }
        // An unknown conversation — not a tracked root and not a registered
        // child — is a lone conversation and thus its own trivially-drained
        // tree, so recycle behaves as it does with no gate.
        true
    }
}

#[cfg(test)]
mod system_idle_gap_tests {
    use super::*;

    #[test]
    fn cap_system_idle_offsets_preserves_spacing_and_caps_next_delay() {
        let offsets = vec![0.0, 100_000.0, 101_000.0];
        let capped = cap_system_idle_offsets_ms(&offsets, Some(10_000.0));

        assert_eq!(capped, vec![0.0, 10_000.0, 11_000.0]);
    }

    #[test]
    fn cap_system_idle_offsets_noops_without_large_idle_gap() {
        let offsets = vec![0.0, 9_000.0, 9_500.0];
        let capped = cap_system_idle_offsets_ms(&offsets, Some(10_000.0));

        assert_eq!(capped, offsets);
    }

    #[test]
    fn system_idle_continuation_delay_caps_only_when_no_other_tasks_are_pending() {
        assert_eq!(
            system_idle_continuation_delay_ms(100_000.0, Some(10_000.0), 1),
            10_000.0
        );
        assert_eq!(
            system_idle_continuation_delay_ms(100_000.0, Some(10_000.0), 2),
            100_000.0
        );
        assert_eq!(
            system_idle_continuation_delay_ms(5_000.0, Some(10_000.0), 1),
            5_000.0
        );
        assert_eq!(
            system_idle_continuation_delay_ms(100_000.0, None, 1),
            100_000.0
        );
    }
}

#[cfg(test)]
mod tree_gate_tests {
    use super::*;

    #[test]
    fn tree_gate_defers_join_until_children_terminal() {
        let spec = TreeSpec {
            root: "t".into(),
            children: vec!["t::sa:a".into()],
            join_turns: vec![(2usize, vec!["t::sa:a".into()])],
        };
        let gate = TreeGate::new(&[spec]);
        assert!(gate.is_waiting("t", 2));
        gate.on_child_terminal("t::sa:a");
        assert!(!gate.is_waiting("t", 2));
        assert!(!gate.on_lane_terminal("t::sa:a"));
        assert!(gate.on_lane_terminal("t")); // whole tree drained
    }

    #[test]
    fn join_gating_defers_and_releases_parent() {
        // One tree: root "t" has a join at turn 1 waiting on child "t::sa:a".
        let spec = TreeSpec {
            root: "t".into(),
            children: vec!["t::sa:a".into()],
            join_turns: vec![(1usize, vec!["t::sa:a".into()])],
        };
        let gate = TreeGate::new(&[spec]);

        // The deferral queue is keyed by `(conversation_id, turn_index)`; a
        // `(String, usize)` stand-in exercises the real `take_ready` decision
        // without a full `TurnToSend`.
        let queue: RefCell<Vec<(String, usize)>> = RefCell::new(Vec::new());

        // The parent join turn is waiting: it is deferred (queued), not issued.
        assert!(gate.is_waiting("t", 1));
        queue.borrow_mut().push(("t".to_string(), 1));

        // While the child is live, `take_ready` leaves the join parked.
        assert!(take_ready(&queue, &gate, |x| (x.0.as_str(), x.1)).is_empty());
        assert_eq!(queue.borrow().len(), 1);

        // The child terminal clears the gate and drains the parent join.
        gate.on_child_terminal("t::sa:a");
        assert!(!gate.is_waiting("t", 1));
        let ready = take_ready(&queue, &gate, |x| (x.0.as_str(), x.1));
        assert_eq!(ready, vec![("t".to_string(), 1)]);
        assert!(queue.borrow().is_empty());
    }

    #[test]
    fn build_tree_specs_groups_root_and_children() {
        use crate::agentx::loader::{
            JoinPrerequisite, ReconstructedConversation, ReconstructedTurn,
        };

        // A bare reconstructed turn; `join` optionally hangs a join prerequisite
        // on it (the only field `build_tree_specs` reads besides ordering).
        let turn = |join: Option<Vec<String>>| ReconstructedTurn {
            timestamp_ms: Some(0.0),
            delay_ms: None,
            api_time_ms: None,
            source_trace_id: "t".into(),
            source_outer_idx: 0,
            source_kind: "weka_main".into(),
            model: "m".into(),
            max_tokens: 1,
            raw_messages: vec![],
            reset_context: false,
            theoretical_prefix_cache_hit_blocks: 0,
            theoretical_prefix_cache_total_blocks: 0,
            input_kind: None,
            spawn_branch: None,
            join_prerequisite: join.map(|child_session_ids| JoinPrerequisite {
                branch_id: "br:a".into(),
                child_session_ids,
            }),
        };
        // Root "t": turns 0,1 plain; turn 2 joins on child "t::sa:a".
        let root = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![turn(None), turn(None), turn(Some(vec!["t::sa:a".into()]))],
        };
        let child = ReconstructedConversation {
            session_id: "t::sa:a".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: Some("t".into()),
            turns: vec![turn(None)],
        };
        let specs = build_tree_specs(&[root, child]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].root, "t");
        assert_eq!(specs[0].children, vec!["t::sa:a".to_string()]);
        assert_eq!(
            specs[0].join_turns,
            vec![(2usize, vec!["t::sa:a".to_string()])]
        );
    }

    #[test]
    fn build_tree_specs_flattens_nested_subagents() {
        use crate::agentx::loader::{ReconstructedConversation, ReconstructedTurn};

        let turn = || ReconstructedTurn {
            timestamp_ms: Some(0.0),
            delay_ms: None,
            api_time_ms: None,
            source_trace_id: "t".into(),
            source_outer_idx: 0,
            source_kind: "weka_main".into(),
            model: "m".into(),
            max_tokens: 1,
            raw_messages: vec![],
            reset_context: false,
            theoretical_prefix_cache_hit_blocks: 0,
            theoretical_prefix_cache_total_blocks: 0,
            input_kind: None,
            spawn_branch: None,
            join_prerequisite: None,
        };
        // root "t" -> child "t::sa:a" (parent "t") -> grandchild
        // "t::sa:a:fa:0" (parent "t::sa:a", a depth-2 descendant).
        let root = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![turn()],
        };
        let child = ReconstructedConversation {
            session_id: "t::sa:a".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: Some("t".into()),
            turns: vec![turn()],
        };
        let grandchild = ReconstructedConversation {
            session_id: "t::sa:a:fa:0".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: Some("t::sa:a".into()),
            turns: vec![turn()],
        };
        let specs = build_tree_specs(&[root, child, grandchild]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].root, "t");
        let mut children = specs[0].children.clone();
        children.sort();
        assert_eq!(
            children,
            vec!["t::sa:a".to_string(), "t::sa:a:fa:0".to_string()]
        );
    }

    #[test]
    fn max_tokens_override_some_forces_value_over_recorded() {
        // A `Some(1)` override forces the built turn's requested output tokens to
        // 1 regardless of the recorded cap (Python's `_WARMUP_MAX_TOKENS=1`).
        assert_eq!(effective_max_output_tokens(128, Some(1)), 1);
        assert_eq!(effective_max_output_tokens(0, Some(7)), 7);
    }

    #[test]
    fn max_tokens_override_none_leaves_recorded_unchanged() {
        // The default (`None`) preserves each turn's recorded output-token cap.
        assert_eq!(effective_max_output_tokens(128, None), 128);
        assert_eq!(effective_max_output_tokens(0, None), 0);
    }

    #[test]
    fn tree_gate_fails_closed_on_dangling_join_child() {
        let spec = TreeSpec {
            root: "t".into(),
            children: vec!["t::sa:a".into()],
            join_turns: vec![(2usize, vec!["t::sa:missing".into()])],
        };
        assert!(TreeGate::try_new(&[spec]).is_err());
    }
}
