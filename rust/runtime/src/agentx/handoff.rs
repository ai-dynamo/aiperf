// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Warmup-to-profile handoff observation for the accelerated cache-warmup
//! substage — a port of the return-observation half of Python's
//! `AgenticReplayStrategy.observe_credit_return`
//! (`src/aiperf/timing/strategies/agentic_replay.py`, lines 698-717).
//!
//! During accelerated cache-pressure warmup Python runs `observe_credit_return`
//! on *every* credit return: it advances the replay barrier gate
//! (`replay_gate.complete`), then — for non-final returns — records the live
//! credit and its return wall-time keyed by the credit's runtime correlation id,
//! and — for final returns — pops both records. Task 5 consumes these
//! `handoff_credits` / `return_wall_ns` maps to drive the compressed-turn
//! handoff; this module supplies the pure, deterministically-testable recorder.
//!
//! The [`HandoffCredit`] projection captures the credit fields the
//! warmup-to-profile handoff needs. The core identity fields
//! (`conversation_id`, `x_correlation_id`, `turn_index`, `num_turns`) exist on
//! the Rust [`IssuedCredit`]/[`TurnToSend`] seam. The DAG fields
//! (`agent_depth` / `parent_correlation_id` / `root_correlation_id` /
//! `branch_mode`) that Python's `Credit` carries have **no** Rust `TurnToSend`
//! equivalent yet (a known loader-seam gap): [`HandoffCredit::from_credit`]
//! populates them with the linear-lane MVP defaults (`agent_depth = 0`,
//! `parent_correlation_id = None`, `root_correlation_id = x_correlation_id`,
//! `branch_mode = default`). Full subagent-depth population depends on the WEKA
//! loader emitting these onto the turn. The residual/rebuild goldens construct
//! [`HandoffCredit`] rows explicitly so they are byte-exact regardless of the
//! live seam.
//!
//! # Finalize (Task 5)
//!
//! At the profiling boundary the drained accelerated-warmup DAG is projected
//! into a [`LegacyWarmupHandoff`] — the analogue of the graph path's
//! `GraphWarmupHandoff` — via [`finalize`]. This ports Python
//! `AgenticReplayStrategy._build_handoff_states` /
//! `_add_returned_handoff_states` / `_returned_credit_handoff_state` /
//! `_add_pending_handoff_states` / `_pending_turn_handoff_state` /
//! `_handoff_lane_for_turn` / `_handoff_residual_delay_ms` /
//! `_handoff_base_delay_ms` / `_build_handoff_replay_boundaries` /
//! `_build_handoff_trajectories` (lines 749-1094). Join annotations are empty
//! for the linear MVP (Python `_handoff_annotations` returns `({}, {})` when
//! there is no branch orchestrator); the `TreeGate`-annotation hookup is left to
//! Task 6 wiring. All wall values flow through the injected
//! [`Clock`](crate::clock::Clock); byte-exactness holds only under `SimClock`.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use crate::agentx::replay_dependencies::ReplayResumeBoundary;
use crate::agentx::replay_gate::ReplayGate;
use crate::multiturn::IssuedCredit;

/// Projection of the returned credit fields the warmup-to-profile handoff needs.
///
/// A pure value type (no `Rc`/lifetimes) so the recorder logic is unit-testable
/// without constructing a full [`IssuedCredit`]. Keyed into the recorder maps by
/// [`HandoffCredit::x_correlation_id`], mirroring Python's `credit.x_correlation_id`
/// dictionary key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffCredit {
    /// Template/conversation identity of the returned turn
    /// (`credit.conversation_id`).
    pub conversation_id: String,
    /// Runtime session correlation id — the recorder map key
    /// (`credit.x_correlation_id`).
    pub x_correlation_id: String,
    /// Zero-based index of the returned turn (`credit.turn_index`).
    pub turn_index: usize,
    /// Total turns this runtime session will send (`credit.num_turns`).
    pub num_turns: usize,
    /// Static DAG nesting level (`credit.agent_depth`, `0` = tree root). Defaults
    /// to `0` off the live seam (linear-lane MVP).
    pub agent_depth: i64,
    /// Parent session correlation id for a DAG child (`credit.parent_correlation_id`);
    /// `None` for roots and off the live seam.
    pub parent_correlation_id: Option<String>,
    /// Correlation id of the depth-0 tree root (`credit.root_correlation_id`);
    /// `None` when this credit is itself the root. Off the live seam this defaults
    /// to `None` and [`HandoffCredit::effective_root_correlation_id`] falls back to
    /// `x_correlation_id`.
    pub root_correlation_id: Option<String>,
    /// Branch orchestration mode (`credit.branch_mode`); [`BranchMode::default`]
    /// off the live seam.
    pub branch_mode: BranchMode,
}

impl HandoffCredit {
    /// Whether the returned credit is its session's final turn (mirrors
    /// [`IssuedCredit::is_final_turn`] / Python `credit.is_final_turn`).
    pub fn is_final(&self) -> bool {
        self.turn_index + 1 >= self.num_turns
    }

    /// Tree-root correlation id (`credit.effective_root_correlation_id`):
    /// `root_correlation_id` when present, else `x_correlation_id`.
    pub fn effective_root_correlation_id(&self) -> &str {
        self.root_correlation_id
            .as_deref()
            .unwrap_or(&self.x_correlation_id)
    }

    /// Project the fields the handoff needs off a returned [`IssuedCredit`].
    ///
    /// The DAG fields have no `TurnToSend` equivalent yet (known loader-seam gap);
    /// they take the linear-lane MVP defaults documented on the struct fields.
    pub fn from_credit(credit: &IssuedCredit) -> Self {
        Self {
            conversation_id: credit.turn.conversation_id.clone(),
            x_correlation_id: credit.turn.x_correlation_id.clone(),
            turn_index: credit.turn.turn_index,
            num_turns: credit.turn.num_turns,
            agent_depth: 0,
            parent_correlation_id: None,
            root_correlation_id: None,
            branch_mode: BranchMode::default(),
        }
    }
}

/// Branch orchestration mode carried across the handoff (Python `credit.branch_mode`).
///
/// The linear-lane MVP only ever produces [`BranchMode::Default`]; the full
/// branch-mode taxonomy is a Task 6 concern gated on the branch orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchMode {
    /// The single mode the linear MVP emits (no branch orchestrator active).
    #[default]
    Default,
}

/// The warmup-to-profile handoff record: the last live (non-final) credit and
/// its return wall-time per correlation id.
///
/// Pure and deterministic — the caller injects the return wall through
/// [`HandoffRecorder::observe`] (routed via the runtime [`Clock`](crate::clock::Clock)
/// at the call site, never `Instant::now`). Ordered [`BTreeMap`]s make the map
/// contents and any snapshot deterministic.
#[derive(Debug, Default)]
pub struct HandoffRecorder {
    /// Live (non-final) returned credit per correlation id
    /// (Python `_handoff_credits`).
    handoff_credits: BTreeMap<String, HandoffCredit>,
    /// Return wall-clock nanoseconds per correlation id
    /// (Python `_handoff_returned_at_ns`).
    return_wall_ns: BTreeMap<String, i64>,
}

impl HandoffRecorder {
    /// Construct an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one credit return: pop both records on a final turn, otherwise
    /// record the live credit and its return wall (Python lines 710-717). The
    /// barrier-gate `complete` call is the caller's responsibility (it precedes
    /// this in `observe_credit_return`).
    pub fn observe(&mut self, credit: HandoffCredit, is_final: bool, wall_ns: i64) {
        let key = credit.x_correlation_id.clone();
        if is_final {
            self.handoff_credits.remove(&key);
            self.return_wall_ns.remove(&key);
        } else {
            self.return_wall_ns.insert(key.clone(), wall_ns);
            self.handoff_credits.insert(key, credit);
        }
    }

    /// The recorded live-credit map (Python `_handoff_credits`).
    pub fn handoff_credits(&self) -> &BTreeMap<String, HandoffCredit> {
        &self.handoff_credits
    }

    /// The recorded return-wall map (Python `_handoff_returned_at_ns`).
    pub fn return_wall_ns(&self) -> &BTreeMap<String, i64> {
        &self.return_wall_ns
    }
}

/// Bundle threaded into the accelerated-warmup return seam: the replay barrier
/// gate plus the handoff recorder, driven together on every credit return.
///
/// Held behind an `Option<Rc<AcceleratedObserver>>` at the scheduling call site;
/// `None` (every current caller) is the standard warmup/profiling path and
/// changes no runtime behavior. Interior [`RefCell`]s allow `&self` driving from
/// the shared, current-thread completion callback without `Arc<Mutex>`.
#[derive(Debug)]
pub struct AcceleratedObserver {
    /// Replay interval-barrier coordinator (Task 3).
    pub gate: RefCell<ReplayGate>,
    /// Warmup-to-profile handoff recorder.
    pub recorder: RefCell<HandoffRecorder>,
}

impl AcceleratedObserver {
    /// Bundle a barrier `gate` and handoff `recorder`.
    pub fn new(gate: ReplayGate, recorder: HandoffRecorder) -> Self {
        Self {
            gate: RefCell::new(gate),
            recorder: RefCell::new(recorder),
        }
    }
}

/// One live conversation stream in a warmup-to-profile handoff lane
/// (port of Python `ConversationState`, the subset the linear MVP produces).
#[derive(Debug, Clone, PartialEq)]
pub struct HandoffConversationState {
    /// Template conversation id of this live stream.
    pub conversation_id: String,
    /// Per-session correlation id (sticky-routing key).
    pub x_correlation_id: String,
    /// Index of the next turn this stream will dispatch in profiling.
    pub next_turn_index: i64,
    /// Normalized wall-clock offset (ms) at which the next turn dispatches — the
    /// residual carried forward from the recorded cadence.
    pub next_dispatch_offset_ms: f64,
    /// Static DAG nesting level (`0` = tree root).
    pub agent_depth: i64,
    /// Parent session correlation id for a DAG child; `None` for roots.
    pub parent_correlation_id: Option<String>,
    /// Correlation id of the depth-0 tree root; `None` when this is the root.
    pub root_correlation_id: Option<String>,
    /// True while this stream is blocked awaiting spawned child completion.
    pub waiting_on_children: bool,
    /// Turn index resumed once gated children join; `None` when ungated.
    pub join_target_turn_index: Option<i64>,
    /// Branch id this stream is gated under; `None` for the linear MVP.
    pub branch_id: Option<String>,
    /// Branch orchestration mode.
    pub branch_mode: BranchMode,
}

impl HandoffConversationState {
    /// Tree-root correlation id: `root_correlation_id` when present, else
    /// `x_correlation_id` (Python `ConversationState.effective_root_correlation_id`).
    pub fn effective_root_correlation_id(&self) -> &str {
        self.root_correlation_id
            .as_deref()
            .unwrap_or(&self.x_correlation_id)
    }
}

/// One lane's surviving states plus its merged replay-resume boundaries
/// (per-lane payload of [`LegacyWarmupHandoff`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaneHandoff {
    /// Surviving live streams, sorted `(agent_depth, x_correlation_id)`.
    pub states: Vec<HandoffConversationState>,
    /// Merged completed-prefix boundaries, sorted by `conversation_id`.
    pub boundaries: Vec<ReplayResumeBoundary>,
}

/// The warmup-to-profile carrier (analogue of `GraphWarmupHandoff`): per lane,
/// the surviving [`HandoffConversationState`]s and their replay-resume
/// boundaries. PROFILING reads this instead of the load-time `::warmup`-split
/// trajectory list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LegacyWarmupHandoff {
    /// Per-lane handoff payload, keyed by lane index for deterministic ordering.
    pub lanes: BTreeMap<usize, LaneHandoff>,
}

/// A barrier-retained pending turn projected for the handoff. The Rust
/// [`ReplayTurn`](crate::agentx::replay_gate::ReplayTurn) seam only carries
/// `(root_id, conversation_id, turn_index)`; the DAG fields here take the
/// linear-lane MVP defaults until the loader emits them (known seam gap). The
/// pending-state goldens construct these explicitly.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingHandoffTurn {
    /// Template conversation id of the pending turn.
    pub conversation_id: String,
    /// Per-session correlation id.
    pub x_correlation_id: String,
    /// Zero-based index of the pending turn.
    pub turn_index: i64,
    /// Total turns this runtime session will send.
    pub num_turns: i64,
    /// Static DAG nesting level.
    pub agent_depth: i64,
    /// Parent session correlation id for a DAG child.
    pub parent_correlation_id: Option<String>,
    /// Correlation id of the depth-0 tree root.
    pub root_correlation_id: Option<String>,
    /// Branch orchestration mode.
    pub branch_mode: BranchMode,
}

impl PendingHandoffTurn {
    /// Tree-root correlation id: `root_correlation_id` when present, else
    /// `x_correlation_id`.
    pub fn effective_root_correlation_id(&self) -> &str {
        self.root_correlation_id
            .as_deref()
            .unwrap_or(&self.x_correlation_id)
    }
}

/// Recorded next-turn/previous-turn metadata a returned credit needs to compute
/// its base delay (the inputs of Python `_handoff_base_delay_ms`, lines 978-1006).
///
/// All fields are already `_as_timestamp_ms`-coerced by the caller: a non-finite
/// or absent source value is `None`. When `next_delay_ms` is `Some`, it wins;
/// otherwise the timestamp fallback applies.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HandoffBaseDelayInputs {
    /// The next turn's recorded relative `delay_ms`, when present and finite.
    pub next_delay_ms: Option<f64>,
    /// The returned (previous) turn's absolute `timestamp_ms`.
    pub prev_timestamp_ms: Option<f64>,
    /// The next turn's absolute `timestamp_ms`.
    pub next_timestamp_ms: Option<f64>,
    /// The returned turn's recorded `api_time_ms` (server duration).
    pub prev_api_time_ms: Option<f64>,
}

/// Coerce a metadata timestamp to finite ms, treating non-finite as absent
/// (port of Python `_as_timestamp_ms`, `trajectory_source.py` lines 892-901).
fn as_timestamp_ms(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

/// Recorded delay from a returned warmup credit to its next turn (port of Python
/// `_handoff_base_delay_ms`, lines 978-1006).
///
/// The `delay_ms` path wins when present and finite (clamped `>= 0`); otherwise
/// the timestamp fallback `next_ts - prev_ts - max(0, prev_api)` applies (clamped
/// `>= 0`), and `0.0` when either timestamp is absent.
pub fn base_delay_ms(inputs: &HandoffBaseDelayInputs) -> f64 {
    if let Some(delay) = as_timestamp_ms(inputs.next_delay_ms) {
        return delay.max(0.0);
    }
    match (
        as_timestamp_ms(inputs.prev_timestamp_ms),
        as_timestamp_ms(inputs.next_timestamp_ms),
    ) {
        (Some(prev_ts), Some(next_ts)) => {
            let prev_duration = as_timestamp_ms(inputs.prev_api_time_ms)
                .unwrap_or(0.0)
                .max(0.0);
            (next_ts - prev_ts - prev_duration).max(0.0)
        }
        _ => 0.0,
    }
}

/// Residual profiling dispatch delay for a drained warmup stream (port of Python
/// `_handoff_residual_delay_ms`, lines 958-976).
///
/// Carries the next turn's recorded `base_ms` forward, minus any wall-clock time
/// already spent waiting for the drain/finalize (`returned_ns` present), floored
/// at `0`, then clamped to the trace idle-gap cap when configured. `returned_ns`
/// and `finalized_ns` are Clock-derived nanoseconds (never `Instant::now`);
/// byte-exact only under `SimClock`.
pub fn residual_delay_ms(
    base_ms: f64,
    returned_ns: Option<i64>,
    finalized_ns: i64,
    cap_ms: Option<f64>,
) -> f64 {
    let mut delay_ms = base_ms;
    if let Some(returned_ns) = returned_ns {
        let elapsed_ms = ((finalized_ns - returned_ns) as f64 / 1_000_000.0).max(0.0);
        delay_ms = (delay_ms - elapsed_ms).max(0.0);
    }
    if let Some(cap_ms) = cap_ms {
        delay_ms = delay_ms.min(cap_ms);
    }
    delay_ms
}

/// A fresh recycle draw for an empty lane: `(conversation_id, x_correlation_id)`.
///
/// Python draws `next_recycle_conversation_id()` for the conversation and
/// `uuid.uuid4()` for the correlation; both are injected from a seeded source so
/// the rebuild is deterministic on both sides.
pub type RecycleDraw = (String, String);

/// The previous trajectory identity of a lane, used as the fallback root
/// identity when a rebuilt lane has no depth-0 root state (Python `previous`).
#[derive(Debug, Clone, PartialEq)]
pub struct PrevLaneTrajectory {
    /// Previous lane conversation id.
    pub conversation_id: String,
    /// Previous lane correlation id.
    pub x_correlation_id: String,
}

/// Everything [`finalize`] needs to project the drained accelerated-warmup DAG
/// into a [`LegacyWarmupHandoff`]. Keeps the port pure and deterministically
/// unit-testable: the caller supplies the recorder maps, gate-derived pending
/// turns and completed prefixes, the lane maps, and recorded-turn metadata.
pub struct FinalizeInputs<'a, BaseFn, PrefixFn, RecycleFn>
where
    BaseFn: Fn(&HandoffCredit) -> HandoffBaseDelayInputs,
    PrefixFn: Fn(&str) -> Vec<ReplayResumeBoundary>,
    RecycleFn: FnMut() -> Option<RecycleDraw>,
{
    /// Live (non-final) returned credits per correlation id (recorder map).
    pub handoff_credits: &'a BTreeMap<String, HandoffCredit>,
    /// Return wall-clock nanoseconds per correlation id (recorder map).
    pub return_wall_ns: &'a BTreeMap<String, i64>,
    /// Barrier-retained pending turns grouped by runtime root id.
    pub pending_by_root: &'a BTreeMap<String, Vec<PendingHandoffTurn>>,
    /// Root correlation id -> lane index (populated during dispatch).
    pub root_to_lane: &'a BTreeMap<String, usize>,
    /// Any correlation id -> lane index (populated during dispatch).
    pub correlation_to_lane: &'a BTreeMap<String, usize>,
    /// Number of lanes (trajectories) — states are built for every lane `0..n`.
    pub num_lanes: usize,
    /// Finalize wall (Clock ns) used for the residual elapsed subtraction.
    pub finalized_ns: i64,
    /// Trace idle-gap cap in ms, when configured (`_phase_offset_cap_ms`).
    pub cap_ms: Option<f64>,
    /// Base-delay metadata for a returned credit (`_handoff_base_delay_ms` inputs).
    pub base_delay_inputs: BaseFn,
    /// Completed-prefix boundaries for a tree-root id (`gate.completed_prefixes`).
    pub completed_prefixes: PrefixFn,
    /// Fresh recycle draws for empty lanes (`next_recycle_conversation_id` + uuid).
    pub recycle_draw: RecycleFn,
    /// Previous trajectory identity per lane, for the no-root fallback.
    pub prev_lanes: &'a [PrevLaneTrajectory],
}

/// Build a returned credit's handoff state, or `None` when it has no lane or is
/// already on its final turn (port of Python `_returned_credit_handoff_state`,
/// lines 836-865). Join annotations are empty for the linear MVP.
fn returned_credit_handoff_state(
    credit: &HandoffCredit,
    root_to_lane: &BTreeMap<String, usize>,
    returned_wall: Option<i64>,
    base_ms: f64,
    finalized_ns: i64,
    cap_ms: Option<f64>,
) -> Option<(usize, HandoffConversationState)> {
    let lane = *root_to_lane.get(credit.effective_root_correlation_id())?;
    if credit.turn_index + 1 >= credit.num_turns {
        return None;
    }
    let state = HandoffConversationState {
        conversation_id: credit.conversation_id.clone(),
        x_correlation_id: credit.x_correlation_id.clone(),
        next_turn_index: credit.turn_index as i64 + 1,
        next_dispatch_offset_ms: residual_delay_ms(base_ms, returned_wall, finalized_ns, cap_ms),
        agent_depth: credit.agent_depth,
        parent_correlation_id: credit.parent_correlation_id.clone(),
        root_correlation_id: credit.root_correlation_id.clone(),
        // No branch orchestrator in the linear MVP => never blocked / gated.
        waiting_on_children: false,
        join_target_turn_index: None,
        branch_id: None,
        branch_mode: credit.branch_mode,
    };
    Some((lane, state))
}

/// Resolve a pending turn's lane (port of Python `_handoff_lane_for_turn`,
/// lines 943-956): root id, then effective-root id, then parent correlation,
/// then the turn's own correlation.
fn handoff_lane_for_turn(
    root_correlation_id: &str,
    turn: &PendingHandoffTurn,
    root_to_lane: &BTreeMap<String, usize>,
    correlation_to_lane: &BTreeMap<String, usize>,
) -> Option<usize> {
    if let Some(lane) = root_to_lane.get(root_correlation_id) {
        return Some(*lane);
    }
    if let Some(lane) = root_to_lane.get(turn.effective_root_correlation_id()) {
        return Some(*lane);
    }
    if let Some(parent) = &turn.parent_correlation_id
        && let Some(lane) = correlation_to_lane.get(parent)
    {
        return Some(*lane);
    }
    correlation_to_lane.get(&turn.x_correlation_id).copied()
}

/// Build a barrier-pending turn's handoff state at offset `0.0`, or `None` when
/// it has no lane, is a duplicate of an already-seen state, or is past its final
/// turn (port of Python `_pending_turn_handoff_state`, lines 907-941).
fn pending_turn_handoff_state(
    root_correlation_id: &str,
    turn: &PendingHandoffTurn,
    seen: &BTreeSet<(String, String, i64)>,
    root_to_lane: &BTreeMap<String, usize>,
    correlation_to_lane: &BTreeMap<String, usize>,
) -> Option<(usize, (String, String, i64), HandoffConversationState)> {
    let lane = handoff_lane_for_turn(root_correlation_id, turn, root_to_lane, correlation_to_lane)?;
    let key = (
        turn.conversation_id.clone(),
        turn.x_correlation_id.clone(),
        turn.turn_index,
    );
    if seen.contains(&key) || turn.turn_index >= turn.num_turns {
        return None;
    }
    let state = HandoffConversationState {
        conversation_id: turn.conversation_id.clone(),
        x_correlation_id: turn.x_correlation_id.clone(),
        next_turn_index: turn.turn_index,
        next_dispatch_offset_ms: 0.0,
        agent_depth: turn.agent_depth,
        parent_correlation_id: turn.parent_correlation_id.clone(),
        root_correlation_id: turn.root_correlation_id.clone(),
        waiting_on_children: false,
        join_target_turn_index: None,
        branch_id: None,
        branch_mode: turn.branch_mode,
    };
    Some((lane, key, state))
}

/// Tree-root id shared by every stream of a lane's states (port of Python
/// `_lane_root_corr`, lines 287-324): the first non-`None` `root_correlation_id`,
/// else the first depth-0 `x_correlation_id`, else the first non-`None`
/// `parent_correlation_id`, else the first `x_correlation_id`.
fn lane_root_corr(states: &[HandoffConversationState]) -> Option<String> {
    states
        .iter()
        .find_map(|s| s.root_correlation_id.clone())
        .or_else(|| {
            states
                .iter()
                .find(|s| s.agent_depth == 0)
                .map(|s| s.x_correlation_id.clone())
        })
        .or_else(|| states.iter().find_map(|s| s.parent_correlation_id.clone()))
        .or_else(|| states.first().map(|s| s.x_correlation_id.clone()))
}

/// Merge live stream positions with terminal warmup history into sorted
/// replay-resume boundaries (port of Python `_build_handoff_replay_boundaries`,
/// lines 1008-1034).
fn build_handoff_replay_boundaries(
    states: &[HandoffConversationState],
    completed_prefixes: impl Fn(&str) -> Vec<ReplayResumeBoundary>,
) -> Vec<ReplayResumeBoundary> {
    let mut next_turn_by_conversation: BTreeMap<String, i64> = BTreeMap::new();
    for state in states {
        if state.next_turn_index > 0 {
            next_turn_by_conversation.insert(state.conversation_id.clone(), state.next_turn_index);
        }
    }
    if !states.is_empty()
        && let Some(root) = lane_root_corr(states)
    {
        for boundary in completed_prefixes(&root) {
            let entry = next_turn_by_conversation
                .entry(boundary.conversation_id.clone())
                .or_insert(0);
            *entry = (*entry).max(boundary.next_turn_index);
        }
    }
    next_turn_by_conversation
        .into_iter()
        .map(|(conversation_id, next_turn_index)| ReplayResumeBoundary {
            conversation_id,
            next_turn_index,
        })
        .collect()
}

/// Project the drained accelerated-warmup DAG into a [`LegacyWarmupHandoff`].
///
/// Ports Python `finalize_phase` / `_build_handoff_states` /
/// `_add_returned_handoff_states` / `_add_pending_handoff_states` /
/// `_build_handoff_replay_boundaries` / `_build_handoff_trajectories`
/// (lines 749-1094) for the linear-lane MVP: join annotations are empty, so no
/// state is `waiting_on_children` or branch-gated. Returned credits are projected
/// first (deduped by `(conversation_id, x_correlation_id, next_turn_index)`),
/// then barrier-pending turns at offset `0.0`; each lane's states are sorted
/// `(agent_depth, x_correlation_id)`, an empty lane draws a fresh recycle root,
/// and boundaries are merged and sorted by `conversation_id`.
pub fn finalize<BaseFn, PrefixFn, RecycleFn>(
    mut inputs: FinalizeInputs<'_, BaseFn, PrefixFn, RecycleFn>,
) -> LegacyWarmupHandoff
where
    BaseFn: Fn(&HandoffCredit) -> HandoffBaseDelayInputs,
    PrefixFn: Fn(&str) -> Vec<ReplayResumeBoundary>,
    RecycleFn: FnMut() -> Option<RecycleDraw>,
{
    let mut states_by_lane: Vec<Vec<HandoffConversationState>> = vec![Vec::new(); inputs.num_lanes];
    let mut seen: BTreeSet<(String, String, i64)> = BTreeSet::new();

    // Returned mid-flight credits (recorder map is sorted by correlation id).
    for credit in inputs.handoff_credits.values() {
        let base = base_delay_ms(&(inputs.base_delay_inputs)(credit));
        let returned = inputs.return_wall_ns.get(&credit.x_correlation_id).copied();
        if let Some((lane, state)) = returned_credit_handoff_state(
            credit,
            inputs.root_to_lane,
            returned,
            base,
            inputs.finalized_ns,
            inputs.cap_ms,
        ) {
            seen.insert((
                state.conversation_id.clone(),
                state.x_correlation_id.clone(),
                state.next_turn_index,
            ));
            if lane < states_by_lane.len() {
                states_by_lane[lane].push(state);
            }
        }
    }

    // Barrier-pending turns (pending_by_root is sorted by root id).
    for (root_correlation_id, turns) in inputs.pending_by_root {
        for turn in turns {
            if let Some((lane, key, state)) = pending_turn_handoff_state(
                root_correlation_id,
                turn,
                &seen,
                inputs.root_to_lane,
                inputs.correlation_to_lane,
            ) {
                seen.insert(key);
                if lane < states_by_lane.len() {
                    states_by_lane[lane].push(state);
                }
            }
        }
    }

    // Rebuild each lane: sort states, merge boundaries, recycle empty lanes.
    let mut lanes = BTreeMap::new();
    for (lane, mut states) in states_by_lane.into_iter().enumerate() {
        let mut boundaries = if states.is_empty() {
            Vec::new()
        } else {
            build_handoff_replay_boundaries(&states, &inputs.completed_prefixes)
        };
        if states.is_empty() {
            if let Some((conversation_id, x_correlation_id)) = (inputs.recycle_draw)() {
                states.push(HandoffConversationState {
                    conversation_id,
                    x_correlation_id,
                    next_turn_index: 0,
                    next_dispatch_offset_ms: 0.0,
                    agent_depth: 0,
                    parent_correlation_id: None,
                    root_correlation_id: None,
                    waiting_on_children: false,
                    join_target_turn_index: None,
                    branch_id: None,
                    branch_mode: BranchMode::default(),
                });
                boundaries = Vec::new();
            } else {
                // No recycle available: Python keeps the previous trajectory as-is.
                // The carrier records an empty lane; PROFILING falls back to the
                // prior lane identity via `prev_lanes`.
                let _ = inputs.prev_lanes.get(lane);
            }
        }
        // Sort states (agent_depth, x_correlation_id) for the profiling snapshot.
        states.sort_by(|a, b| {
            a.agent_depth
                .cmp(&b.agent_depth)
                .then_with(|| a.x_correlation_id.cmp(&b.x_correlation_id))
        });
        lanes.insert(lane, LaneHandoff { states, boundaries });
    }

    LegacyWarmupHandoff { lanes }
}

/// Signal an accelerated-warmup drain (port of Python `_finish_accelerated_warmup`,
/// lines 630-634): pause the replay barrier gate so no newly ready turn is issued,
/// letting the already-issued requests drain.
///
/// The `mark_sending_complete` half (stop new issuance) is the caller's execute-loop
/// flag; the Task 6 wiring arms this via `scheduler.schedule_later(duration_ns, ..)`.
pub fn finish_accelerated(observer: &AcceleratedObserver) {
    observer.gate.borrow_mut().pause_releases();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn credit(conv: &str, corr: &str, turn_index: usize, num_turns: usize) -> HandoffCredit {
        HandoffCredit {
            conversation_id: conv.into(),
            x_correlation_id: corr.into(),
            turn_index,
            num_turns,
            agent_depth: 0,
            parent_correlation_id: None,
            root_correlation_id: None,
            branch_mode: BranchMode::default(),
        }
    }

    #[test]
    fn observe_records_non_final_and_pops_on_final() {
        let mut rec = HandoffRecorder::new();

        // Two lanes each return a non-final (turn 0 of 3) credit at a distinct
        // virtual wall — both are recorded under their correlation id.
        let a0 = credit("conv-a", "x-a", 0, 3);
        let b0 = credit("conv-b", "x-b", 0, 2);
        rec.observe(a0.clone(), a0.is_final(), 1_000);
        rec.observe(b0.clone(), b0.is_final(), 2_000);

        assert_eq!(rec.handoff_credits().get("x-a"), Some(&a0));
        assert_eq!(rec.handoff_credits().get("x-b"), Some(&b0));
        assert_eq!(rec.return_wall_ns().get("x-a"), Some(&1_000));
        assert_eq!(rec.return_wall_ns().get("x-b"), Some(&2_000));

        // A later non-final return on lane A overwrites the live credit + wall.
        let a1 = credit("conv-a", "x-a", 1, 3);
        rec.observe(a1.clone(), a1.is_final(), 3_500);
        assert_eq!(rec.handoff_credits().get("x-a"), Some(&a1));
        assert_eq!(rec.return_wall_ns().get("x-a"), Some(&3_500));

        // Lane A's final turn (2 of 3) pops BOTH maps for its correlation id;
        // lane B is untouched.
        let a2 = credit("conv-a", "x-a", 2, 3);
        assert!(a2.is_final());
        rec.observe(a2.clone(), a2.is_final(), 9_999);
        assert!(!rec.handoff_credits().contains_key("x-a"));
        assert!(!rec.return_wall_ns().contains_key("x-a"));
        assert_eq!(rec.handoff_credits().get("x-b"), Some(&b0));
        assert_eq!(rec.return_wall_ns().get("x-b"), Some(&2_000));
    }

    #[test]
    fn finish_accelerated_pauses_the_gate() {
        use crate::agentx::replay_gate::{ReplayGate, ReplayTurn};

        let mut gate = ReplayGate::new(BTreeMap::new());
        gate.activate();
        let obs = AcceleratedObserver::new(gate, HandoffRecorder::new());

        // Drain: pause releases so a now-ready turn is retained, not issued.
        finish_accelerated(&obs);
        obs.gate
            .borrow_mut()
            .submit(ReplayTurn::new("R", "A", 0))
            .unwrap();
        assert!(obs.gate.borrow().released().is_empty());
        assert_eq!(obs.gate.borrow().pending_turns("R").len(), 1);
    }

    #[test]
    fn finalize_returned_state_recycles_empty_lane_and_sorts() {
        // Lane 0 has a returned mid-flight credit (turn 0 of 3 -> resume at 1);
        // lane 1 drained empty and must draw a recycle root.
        let mut handoff_credits = BTreeMap::new();
        let c = credit("conv-a", "x-a", 0, 3);
        handoff_credits.insert(c.x_correlation_id.clone(), c);
        let mut return_wall_ns = BTreeMap::new();
        return_wall_ns.insert("x-a".to_string(), 1_000_000); // 1ms
        let mut root_to_lane = BTreeMap::new();
        root_to_lane.insert("x-a".to_string(), 0usize);
        let correlation_to_lane = BTreeMap::new();
        let pending_by_root = BTreeMap::new();
        let prev_lanes = vec![
            PrevLaneTrajectory {
                conversation_id: "conv-a".into(),
                x_correlation_id: "x-a".into(),
            },
            PrevLaneTrajectory {
                conversation_id: "conv-b".into(),
                x_correlation_id: "x-b".into(),
            },
        ];
        let mut recycles = vec![("recycle-conv".to_string(), "recycle-corr".to_string())];

        let handoff = finalize(FinalizeInputs {
            handoff_credits: &handoff_credits,
            return_wall_ns: &return_wall_ns,
            pending_by_root: &pending_by_root,
            root_to_lane: &root_to_lane,
            correlation_to_lane: &correlation_to_lane,
            num_lanes: 2,
            finalized_ns: 3_000_000, // 3ms
            cap_ms: None,
            base_delay_inputs: |_c: &HandoffCredit| HandoffBaseDelayInputs {
                next_delay_ms: Some(10.0),
                ..Default::default()
            },
            completed_prefixes: |_root: &str| Vec::new(),
            recycle_draw: move || recycles.pop(),
            prev_lanes: &prev_lanes,
        });

        // Lane 0: one live state, resume index 1, residual = 10 - (3-1)ms = 8.
        let lane0 = &handoff.lanes[&0];
        assert_eq!(lane0.states.len(), 1);
        assert_eq!(lane0.states[0].next_turn_index, 1);
        assert_eq!(lane0.states[0].next_dispatch_offset_ms, 8.0);
        assert_eq!(
            lane0.boundaries,
            vec![ReplayResumeBoundary {
                conversation_id: "conv-a".into(),
                next_turn_index: 1,
            }]
        );

        // Lane 1: drew a recycle root at turn 0, no boundaries.
        let lane1 = &handoff.lanes[&1];
        assert_eq!(lane1.states.len(), 1);
        assert_eq!(lane1.states[0].conversation_id, "recycle-conv");
        assert_eq!(lane1.states[0].x_correlation_id, "recycle-corr");
        assert_eq!(lane1.states[0].next_turn_index, 0);
        assert!(lane1.boundaries.is_empty());
    }

    #[test]
    fn final_return_for_unrecorded_correlation_is_a_noop() {
        // A single-turn session (turn 0 of 1) is final on its first return: it
        // records nothing and the pop is a harmless no-op.
        let mut rec = HandoffRecorder::new();
        let only = credit("conv-c", "x-c", 0, 1);
        assert!(only.is_final());
        rec.observe(only.clone(), only.is_final(), 4_242);
        assert!(rec.handoff_credits().is_empty());
        assert!(rec.return_wall_ns().is_empty());
    }
}
