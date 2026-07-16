// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral Graph-IR phase orchestration.
//!
//! Authored root selection, arrival, admission, lifecycle, ramps, adaptive
//! control, exact node/trace terminal accounting, and reporting records live
//! here once. Online HTTP and in-process offline simulation inject only a
//! phase-local whole-trace execution backend.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use crate::adaptive::{AdaptiveControlVariable, AdaptiveRunConfig, build_adaptive_scale};
use crate::adaptive_core::{
    AdaptiveError, ControlActuator, ControlSnapshot, RequestRateActuator,
    SessionConcurrencyActuator, SharedWindowSampler, TumblingWindowSampler,
};
use crate::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use crate::cellular::{CellPartition, ModuloCellPartition};
use crate::clock::Clock;
use crate::failure::OnFailure;
use crate::graph::errors::TraceError;
use crate::graph::execution::TracePlacement;
use crate::graph::input::GraphInputBundle;
use crate::graph::model::{GraphTracePlan, ParsedGraph, TraceRecord};
use crate::graph::policy::{ContinueRunFailurePolicy, FailFastRunFailurePolicy, RunFailurePolicy};
use crate::graph::snapshot::{chop_trie_at_frontier, chop_trie_at_tstar, rewrite_for_warmup};
use crate::graph::tstar::{PermutationDraw, TStarSampler, WindowTStarSampler, trace_duration_us};
use crate::graph::warmup_handoff::{GraphWarmupHandoff, LaneHandoff};
use crate::graph::workload::{
    CyclingGraphTraceSource, GraphArrivalPolicy, GraphTraceInstanceSequence, GraphTraceRunResult,
    GraphTraceSource, GraphWorkload, GraphWorkloadError, GraphWorkloadObserver,
    GraphWorkloadReport, ImmediateGraphArrival, IntervalGraphArrival, PartitionedGraphTraceSource,
    SlotPoolTraceAdmission, TraceAdmissionInfo,
};
use crate::metrics_core::Phase as MetricsPhase;
use crate::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhaseSidecar,
};
use crate::rng::{RngRoot, namespace};
use crate::timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, LocalPhaseFuture, NoopPhaseObserver,
    PhaseConfig, PhaseContext, PhaseExecution, PhaseExecutionError, PhaseExecutionFactory,
    PhaseObserver, PhaseReturn, PhaseSend, PhaseStats, RampDriver, SlotPool, drive_phases,
    make_interval_generator,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use tokio::sync::{Notify, mpsc};
use uuid::Uuid;

use crate::engine::execute::{
    AdaptiveScheduledPhaseController, RampActuatorRngRoots, adaptive_run_config,
    integer_adaptive_bound, metrics_phase, phase_config, phase_seamless_to_next, ramp_strategy,
    seconds_to_u64_ns,
};
use crate::engine::graph_execution::{
    ChannelRunnerGraphExecutionEventSink, GraphCancellationConfig, GraphExecutionEvent,
    GraphExecutionEventSink, ObservedRunnerGraphPlacement,
};
use crate::engine::graph_input::TStarWindow;
use crate::engine::protocol::{AdaptiveControlVariableSpec, PhaseCommonSpec, PhaseSpec};
use crate::engine::protocol_v2::FailureStageV2;
use crate::engine::records::CapturedRecord;
use crate::engine::registry::PreparedRunFailure;

/// Backend-owned inputs for one already lowered Graph-IR phase.
pub(crate) struct GraphPhaseBackendConfig {
    pub(crate) metrics_phase: MetricsPhase,
    pub(crate) prefill_concurrency: Option<usize>,
    pub(crate) cancellation: Option<GraphCancellationConfig>,
    pub(crate) events: Arc<dyn GraphExecutionEventSink>,
}

/// One phase-local whole-trace backend returned by an injected implementation.
pub(crate) struct PreparedGraphPhaseBackend {
    pub(crate) placement: Rc<dyn TracePlacement>,
    pub(crate) requires_node_records: bool,
}

/// Backend construction seam beneath the one shared graph phase driver.
pub(crate) trait GraphPhaseBackendFactory {
    fn prepare_backend(&self, config: GraphPhaseBackendConfig)
    -> Result<PreparedGraphPhaseBackend>;
}

/// Complete result of one shared Graph-IR phase-orchestrator run.
pub(crate) struct GraphPhaseRunOutput {
    pub(crate) captured: Vec<CapturedRecord>,
    pub(crate) phases: Vec<PhaseStats>,
    pub(crate) workload: GraphWorkloadReport,
}

/// Validate transport-neutral authored Graph-IR phase policy.
pub(crate) fn validate_graph_phases(phases: &[PhaseSpec]) -> Result<()> {
    ensure!(
        !phases.is_empty(),
        "graph execution requires at least one phase"
    );
    for (phase_index, phase) in phases.iter().enumerate() {
        ensure!(
            matches!(
                phase,
                PhaseSpec::Concurrency { .. }
                    | PhaseSpec::Poisson { .. }
                    | PhaseSpec::Gamma { .. }
                    | PhaseSpec::Constant { .. }
            ),
            "graph phase {phase_index} must use concurrency, poisson, gamma, or constant scheduling"
        );
        let common = phase.common();
        // A cache-pressure warmup (agentic_cache_warmup_duration) whose FOLLOWING
        // profiling phase is authored `seamless` races the handoff: profiling's
        // create() pops the handoff slot before the warmup's background finalize()
        // stashes it, so profiling silently falls back to plain t* (loses the
        // pressure-resume, refires executed nodes). Reject the combination up
        // front rather than degrade at runtime.
        if common.name == "warmup"
            && common.agentic_cache_warmup_duration.is_some()
            && phase_seamless_to_next(phases, phase_index)
        {
            bail!(
                "graph phase {phase_index}: cache-pressure warmup (agentic_cache_warmup_duration) \
                 cannot be seamless into profiling: the warmup->profiling handoff requires the \
                 warmup drain to complete before profiling prepares"
            );
        }
        ensure!(
            common.requests != Some(0) && common.sessions != Some(0),
            "graph phase {phase_index} request/session bounds must be positive when configured"
        );
        ensure!(
            phase.concurrency() != Some(0),
            "graph phase {phase_index} concurrency must be positive when configured"
        );
        ensure!(
            common.prefill_concurrency != Some(0),
            "graph phase {phase_index} prefill_concurrency must be positive when configured"
        );
        if let (Some(prefill), Some(concurrency)) =
            (common.prefill_concurrency, phase.concurrency())
        {
            ensure!(
                prefill <= concurrency,
                "graph phase {phase_index} prefill_concurrency must be <= concurrency"
            );
        }
        if let Some(duration) = common.duration {
            seconds_to_u64_ns(duration)
                .with_context(|| format!("validating graph phase {phase_index} duration"))?;
            ensure!(duration > 0.0, "graph phase duration must be positive");
        }
        match phase {
            PhaseSpec::Poisson { rate, .. } | PhaseSpec::Constant { rate, .. } => ensure!(
                rate.is_finite() && *rate > 0.0,
                "graph phase {phase_index} rate must be finite and positive"
            ),
            PhaseSpec::Gamma {
                rate, smoothness, ..
            } => {
                ensure!(
                    rate.is_finite() && *rate > 0.0,
                    "graph phase {phase_index} rate must be finite and positive"
                );
                ensure!(
                    smoothness.is_none_or(|value| value.is_finite() && value > 0.0),
                    "graph phase {phase_index} gamma smoothness must be finite and positive"
                );
            }
            PhaseSpec::Concurrency { .. } => {}
            PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => unreachable!(),
        }
        if let Some(cancellation) = common.cancellation {
            ensure!(
                cancellation.rate.is_finite() && (0.0..=100.0).contains(&cancellation.rate),
                "graph phase {phase_index} cancellation.rate must be finite and in 0..=100"
            );
            ensure!(
                cancellation.delay.is_finite() && cancellation.delay >= 0.0,
                "graph phase {phase_index} cancellation.delay must be finite and non-negative"
            );
        }
        if let Some(ramp) = &common.concurrency_ramp {
            ensure!(
                phase.concurrency().is_some(),
                "graph phase {phase_index} concurrency_ramp requires a concurrency target"
            );
            validate_graph_ramp(phase_index, "concurrency_ramp", ramp.duration)?;
        }
        if let Some(ramp) = &common.prefill_ramp {
            ensure!(
                common.prefill_concurrency.is_some(),
                "graph phase {phase_index} prefill_ramp requires prefill_concurrency"
            );
            validate_graph_ramp(phase_index, "prefill_ramp", ramp.duration)?;
        }
        if let Some(ramp) = &common.rate_ramp {
            ensure!(
                phase
                    .request_arrival()
                    .and_then(|(_, rate, _)| rate)
                    .is_some(),
                "graph phase {phase_index} rate_ramp requires a rate-controlled phase"
            );
            validate_graph_ramp(phase_index, "rate_ramp", ramp.duration)?;
        }
        let _ = graph_adaptive_config(phase, "validation", Path::new("."))?;
    }
    Ok(())
}

fn validate_graph_ramp(phase_index: usize, name: &str, duration: f64) -> Result<()> {
    ensure!(
        duration.is_finite() && duration > 0.0,
        "graph phase {phase_index} {name}.duration must be finite and positive"
    );
    let _ = seconds_to_u64_ns(duration)?;
    Ok(())
}

/// Upper bound in seconds on the extended (cache-pressure) warmup phase's
/// drain grace period.
///
/// Port of `Environment.GRAPH.PRESSURE_DRAIN_GRACE_CAP` (default `300.0`) in
/// `src/aiperf/common/environment.py:657`, consumed by
/// `timing/config.py::_graph_pressure_grace_sec`.
pub const PRESSURE_DRAIN_GRACE_CAP_SEC: f64 = 300.0;

/// Upper bound in seconds on any single extended-warmup handoff residual delay
/// (the recorded inter-turn gap minus drain time a resumed profiling frontier
/// waits before firing; see [`chop_trie_at_frontier`]).
///
/// Port of `Environment.GRAPH.HANDOFF_RESIDUAL_CAP` (default `60.0`) in
/// `src/aiperf/common/environment.py:641` (branch `ajc/aiperf-graph-ir`),
/// consumed by `graph_ir_replay.py::_graph_at_handoff` as
/// `cap_s * MICROS_PER_SECOND`. The 60s default matches the recorded idle-gap
/// cap so a handoff lane cannot park for minutes at the profiling start.
pub const HANDOFF_RESIDUAL_CAP_SEC: f64 = 60.0;

/// Microseconds per second, the [`Clock`]-ledger scale the warmup handoff walls
/// (`clock.now_ns() / 1000`) and `chop_trie_at_frontier` residuals share.
const MICROS_PER_SECOND: f64 = 1_000_000.0;

/// Derive the drain grace (seconds) for a cache-pressure-mode graph warmup.
///
/// Port of `timing/config.py::_graph_pressure_grace_sec`
/// (`src/aiperf/timing/config.py:771-791`). An EXPLICIT `user_grace`
/// (`Some`) is honored verbatim -- the operator's escape hatch when a healthy
/// drain outlives the pressure duration (e.g. a 45s prefill in flight at a
/// 30s deadline). Otherwise the drain is bounded by
/// `min(cache_pressure, PRESSURE_DRAIN_GRACE_CAP_SEC)` so a wedged or lost
/// return cannot hang the run. AgentX's benchmark-grace floor branch is
/// intentionally not ported. Callers gate on a set pressure duration first.
pub fn graph_pressure_grace_sec(user_grace: Option<f64>, cache_pressure: f64) -> f64 {
    match user_grace {
        Some(grace) => grace,
        None => cache_pressure.min(PRESSURE_DRAIN_GRACE_CAP_SEC),
    }
}

struct PreparedGraphPhase {
    workload: GraphWorkload,
    placement: Rc<dyn TracePlacement>,
    events: mpsc::UnboundedReceiver<GraphExecutionEvent>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_initial: Option<usize>,
    controller: Rc<dyn ScheduledPhaseController>,
    failures: Rc<GraphPhaseFailures>,
    adaptive: Option<AdaptiveRunConfig>,
    /// Whether this is the WARMUP phase (gates warmup-failure accounting).
    is_warmup: bool,
    /// Warmup cache-pressure recycle inputs, present only when the warmup phase
    /// carried `agentic_cache_warmup_duration`. Drives [`GraphPressureRecycle`]
    /// in place of the single-pass workload; `None` keeps the warmup unchanged.
    pressure: Option<PreparedPressureRecycle>,
    /// Deferred frontier-resume ingredients for a non-warmup (profiling) phase.
    /// `Some` for every non-warmup phase; consumed only by the first profiling
    /// phase that pops a stashed warmup handoff, to rebuild its trace source with
    /// `chop_trie_at_frontier` plans. `None` for the warmup phase.
    resume: Option<ProfilingResume>,
}

struct GraphTracePhaseProgress {
    expected_nodes: usize,
    returned_nodes: usize,
    first_token_uuids: HashSet<Uuid>,
    returned_uuids: HashSet<Uuid>,
}

/// One lane's warmup-drain resume identity — the join key E3c's `LaneHandoff`
/// assembly consumes to reconstruct each profiling lane's resume graph.
///
/// Port of the per-live-lane tuple
/// `_pressure_live[lane] = (template_id, instance_id, t_star_us, pressure_pass)`
/// recorded in `src/aiperf/timing/strategies/graph_ir_replay.py:2267` (branch
/// `ajc/aiperf-graph-ir`). The `pressure_pass` recycle-pass member is
/// intentionally omitted here: E3a is observability only, and the recycle loop
/// (E3b) owns the pass concept.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLaneIdentity {
    /// Authored root template id (the `instance_id` prefix before `"::"`).
    pub(crate) template_trace_id: String,
    /// Unique execution-instance id stamped as `x_correlation_id` on every node
    /// record dispatched for this lane.
    pub(crate) instance_id: String,
    /// This lane's trajectory-start `t*` in microseconds.
    pub(crate) t_star_us: f64,
}

/// Per-lane executed-node and drain-clock return-wall observability for the
/// extended-warmup cache-pressure handoff (consumed by E3c).
///
/// Ports the three ledgers `graph_ir_replay.py` accumulates over a warmup phase
/// so the profiling phase can resume each lane at the exact drain frontier:
///   * `_return_walls: {instance_id: {node_id: wall_us}}`
///     (`graph_ir_replay.py:499`), written per return by `_record_return_wall`
///     (`graph_ir_replay.py:884`);
///   * `executed_node_ids = frozenset(live_walls)`, derived from the return-wall
///     keys at handoff assembly (`graph_ir_replay.py:2288`); tracked eagerly
///     here as a set;
///   * `_pressure_live: {lane: (template, instance, t*, pass)}`, the lane
///     identity map (`graph_ir_replay.py:2267`).
///
/// All three are keyed by lane index (`u64`) to match E1's
/// `BTreeMap<u64, LaneHandoff>`; an internal `instance_id -> lane` reverse index
/// attributes each terminal node return — which arrives on the wire keyed only
/// by its instance id (`x_correlation_id`) — to the owning lane. Every wall is
/// taken from the injected [`Clock`] at return-handling time
/// (`clock.now_ns() / 1000`), never `Instant::now()`, mirroring Python's
/// monotonic `_wall_us` (`perf_counter_ns() / 1_000`, `graph_ir_replay.py:881`).
///
/// Lane registration ([`register_lane`](Self::register_lane)) is left for E3b's
/// recycle loop to drive as it assigns instances to lanes; until a lane is
/// registered, [`observe_return`](Self::observe_return) is a graceful no-op, so
/// the ledger is inert on the default (no-scenario) graph path.
#[derive(Default)]
pub(crate) struct GraphLaneLedger {
    identities: RefCell<BTreeMap<u64, GraphLaneIdentity>>,
    instance_to_lane: RefCell<HashMap<String, u64>>,
    executed_node_ids: RefCell<BTreeMap<u64, BTreeSet<String>>>,
    return_wall_us: RefCell<BTreeMap<u64, BTreeMap<String, f64>>>,
    /// The shared-sampler corpus draw cursor advanced by E3b's pressure recycle:
    /// the next undrawn corpus position, one past the last template any lane has
    /// claimed (pass-0 assignment + every recycle draw). Ports the strategy's
    /// `_pressure_next_index` (`graph_ir_replay.py:504`, advanced in
    /// `_run_pressure_lanes` at `:1570`), which the teardown handoff stashes so
    /// profiling's bounded recycle resumes where pressure left off
    /// (`graph_ir_replay.py:2294` `corpus_cursor=self._pressure_next_index`).
    /// Inert (`0`) on the no-pressure path; E3c reads it at warmup teardown.
    corpus_cursor: Cell<u64>,
}

impl GraphLaneLedger {
    /// Associate a lane index with the instance it is executing.
    ///
    /// Records the lane -> identity map and the reverse `instance_id -> lane`
    /// index used by [`observe_return`](Self::observe_return). E3b calls this as
    /// it launches each lane's instance; re-registering a lane overwrites its
    /// identity (a recycle draw reusing the lane) while leaving already-ledgered
    /// returns for the prior instance intact under that lane key.
    #[allow(dead_code)] // Driven by E3b's recycle loop as it assigns instances to lanes.
    pub(crate) fn register_lane(&self, lane: u64, identity: GraphLaneIdentity) {
        self.instance_to_lane
            .borrow_mut()
            .insert(identity.instance_id.clone(), lane);
        self.identities.borrow_mut().insert(lane, identity);
    }

    /// Ledger one terminal node return against its lane's executed set and
    /// return-wall map.
    ///
    /// Resolves the lane from `instance_id` via the reverse index; an
    /// unregistered instance (or a return that predates E3b lane assignment) is
    /// a graceful no-op — the analogue of Python's unknown-`instance_id`
    /// early-return in `_record_return_wall` (`graph_ir_replay.py:895`). The
    /// wall is the caller's [`Clock`]-derived microsecond instant at
    /// return-handling time. The set insert is idempotent; the wall assignment
    /// is last-write-wins, matching Python's
    /// `self._return_walls.setdefault(instance_id, {})[node_id] = wall`.
    pub(crate) fn observe_return(&self, instance_id: &str, node_id: &str, return_wall_us: f64) {
        let Some(lane) = self.instance_to_lane.borrow().get(instance_id).copied() else {
            return;
        };
        self.executed_node_ids
            .borrow_mut()
            .entry(lane)
            .or_default()
            .insert(node_id.to_owned());
        self.return_wall_us
            .borrow_mut()
            .entry(lane)
            .or_default()
            .insert(node_id.to_owned(), return_wall_us);
    }

    /// This lane's registered resume identity, if any.
    #[allow(dead_code)] // Consumed by E3c's `LaneHandoff` assembly at warmup teardown.
    pub(crate) fn lane_identity(&self, lane: u64) -> Option<GraphLaneIdentity> {
        self.identities.borrow().get(&lane).cloned()
    }

    /// The set of node ids this lane executed (returned a non-cancelled record).
    #[allow(dead_code)] // Consumed by E3c's `LaneHandoff.executed_node_ids`.
    pub(crate) fn executed_node_ids(&self, lane: u64) -> BTreeSet<String> {
        self.executed_node_ids
            .borrow()
            .get(&lane)
            .cloned()
            .unwrap_or_default()
    }

    /// This lane's `node_id -> return_wall_us` ledger on the Clock-derived wall.
    #[allow(dead_code)] // Consumed by E3c's `LaneHandoff.return_wall_us`.
    pub(crate) fn return_wall_us(&self, lane: u64) -> BTreeMap<String, f64> {
        self.return_wall_us
            .borrow()
            .get(&lane)
            .cloned()
            .unwrap_or_default()
    }

    /// Every lane index that has a registered identity, ascending.
    #[allow(dead_code)] // Consumed by E3c to iterate live lanes at handoff assembly.
    pub(crate) fn registered_lanes(&self) -> Vec<u64> {
        self.identities.borrow().keys().copied().collect()
    }

    /// Record the shared-sampler corpus draw cursor after a lane draw.
    ///
    /// Driven by [`GraphPressureRecycle`] as it assigns pass-0 lanes and every
    /// recycle draw, mirroring Python advancing `_pressure_next_index`. The
    /// last write wins; a single-threaded lane fan-out on the current-thread
    /// runtime keeps this race-free (each draw increments then stores).
    pub(crate) fn set_corpus_cursor(&self, cursor: u64) {
        self.corpus_cursor.set(cursor);
    }

    /// The next undrawn corpus position (E3c stashes it as `corpus_cursor`).
    #[allow(dead_code)] // Consumed by E3c's `LaneHandoff.corpus_cursor` stash.
    pub(crate) fn corpus_cursor(&self) -> u64 {
        self.corpus_cursor.get()
    }
}

/// One recycle-corpus entry: a dispatchable warmup plan plus the identity
/// members E3a's [`GraphLaneLedger::register_lane`] records for the handoff.
///
/// The `plan` is the warmup-rewritten graph (`apply_tstar_split` on the WARMUP
/// phase), `template_id` is the authored root id (the `instance_id` prefix
/// before `"::"`), `t_star_us` is this template's lane-0 `t*` — byte-equal to
/// the value `apply_tstar_split` sampled when it produced `plan` (see
/// [`sample_plan_tstar`]) — and `duration_us` is the ORIGINAL (pre-warmup-rewrite)
/// replayable span the per-lane resample feeds back into
/// [`WindowTStarSampler::sample_t_star`] so a recycle lane's salted `t*` uses the
/// same duration lane 0 did (`graph_ir_replay.py:_plan_for_lane`, lines 743-772).
#[derive(Clone)]
struct PressureTemplate {
    plan: GraphTracePlan,
    template_id: String,
    t_star_us: f64,
    duration_us: f64,
    /// The ORIGINAL (pre-warmup-rewrite) full plan, retained so a higher recycle
    /// lane can re-prime the graph at ITS OWN lane-salted `t*`
    /// (`rewrite_for_warmup(parsed, lane_t*)`) instead of dispatching lane 0's
    /// prebuilt warmup graph. Lane 0 reuses `plan` verbatim, so the original is
    /// consulted only on an active (non-default) window
    /// (`graph_ir_replay.py:_plan_for_lane`/`_lane_source`, lines 743-791).
    original_plan: GraphTracePlan,
}

/// Prepared warmup cache-pressure recycle inputs, carried on the warmup
/// [`PreparedGraphPhase`] until the phase execution binds a placement + progress.
///
/// Present only when the warmup phase carried `agentic_cache_warmup_duration`
/// (`Some`); `None` leaves the warmup on the unchanged single-pass workload path.
struct PreparedPressureRecycle {
    templates: Rc<Vec<PressureTemplate>>,
    /// Clock-driven pressure budget in nanoseconds (`cache_pressure_duration`).
    duration_ns: i64,
    /// Requested concurrent lane count (the CONCURRENCY_BURST warmup width).
    lane_target: usize,
    /// `--num-conversations` lane clamp, if any (absent for the auto warmup).
    session_limit: Option<u64>,
    /// Whether an explicit phase stop condition exists (governs the corpus
    /// lane clamp in [`pressure_resolve_lane_count`]); false for the auto warmup.
    recycle_bounded: bool,
    /// The trajectory-start `t*` window, threaded onto [`GraphPressureRecycle`]
    /// so each recycle lane resamples its OWN lane-salted `t*` at dispatch time
    /// (`graph_ir_replay.py:_plan_for_lane`/`_lane_source`, lines 743-791).
    t_star: TStarWindow,
}

/// Strategy-aware corpus-index draw shared by every graph recycle draw site.
///
/// Reproduces legacy agentx
/// `SequentialSampler`/`ShuffleSampler`/`RandomSampler`
/// (`dataset/dataset_samplers.py`) BYTE-EXACT: the single choke point every
/// cross-trace draw in the pressure/profiling lane fan-out + recycle routes
/// through, so `--dataset-sampling-strategy` governs WHICH template a freed lane
/// serves without changing the draw COUNTERS (only the counter -> index remap
/// changes). Thin wrapper over the shared [`PermutationDraw`] the window resolves
/// ([`TStarWindow::recycle_draw`]): `Sequential` returns `x % total` unchanged;
/// `Shuffle` returns the persistent-epoch shuffle draw; `Random` returns the
/// with-replacement CPython MT19937 draw — each seeded once from the RUN root.
struct PressureDraw {
    /// The shared resolved draw (identical semantics to the profiling recycle's
    /// `CyclingGraphTraceSource` draw), unifying both draw sites.
    inner: PermutationDraw,
}

impl PressureDraw {
    /// Build the draw from a resolved `t*` window: the strategy governs the remap
    /// and the run root (`run_random_seed`) salts the per-strategy child seed.
    fn from_window(t_star: TStarWindow) -> Self {
        Self {
            inner: t_star.recycle_draw(),
        }
    }

    /// Remap draw counter `x` to a corpus index in `[0, total)`; delegates to the
    /// resolved [`PermutationDraw`] (`Sequential` is its byte-unchanged
    /// `x % total`).
    fn index(&self, x: u64, total: usize) -> usize {
        self.inner.index(x, total)
    }
}

/// Resolve how many concurrent recycle lanes the pressure stage fans out.
///
/// Port of `graph_ir_replay.py:_resolve_lane_count` (`:585`): the lane count is
/// the phase concurrency, independent of corpus size when recycle is bounded (a
/// stop condition exists for lanes to recycle toward), clamped by an explicit
/// `--num-conversations` session cap and — when recycle is UNbounded — to the
/// corpus size (each lane then covers one corpus position). At least one lane.
fn pressure_resolve_lane_count(
    concurrency: usize,
    total: usize,
    session_limit: Option<u64>,
    recycle_bounded: bool,
) -> usize {
    let mut lanes = concurrency;
    if let Some(sessions) = session_limit
        && sessions > 0
    {
        lanes = lanes.min(usize::try_from(sessions).unwrap_or(usize::MAX));
    }
    if !recycle_bounded {
        lanes = lanes.min(total);
    }
    lanes.max(1)
}

/// Resolve the pass-0 lane -> corpus-index map and the resume cursor.
///
/// Port of `graph_ir_replay.py:_resolve_pass0_lanes` (`:1577`): walk the corpus
/// in draw order assigning lane `i` to the next SPAWNABLE template, skipping
/// unspawnable ones, and return the per-lane corpus indices (length `<= lanes`)
/// plus the cursor one past the last consumed position so recycle resumes there.
///
/// "Spawnable" at the current native fidelity is "the warmup-rewritten plan has
/// at least one dispatchable node". Python tests spawnability at the candidate's
/// TARGET RANK's lane-salted `t*` (`_is_spawnable(trace, rank)`, rank advances
/// only on a hit), and per-lane salting IS built (DF1b/DF6:
/// [`sample_lane_tstar`] / [`pressure_plan_for_lane`]), so a candidate is judged
/// against the lane-`rank`-salted warmup graph it will ACTUALLY dispatch — NOT
/// lane 0's prebuilt plan. This closes DF6's spawnability divergence: a template
/// non-empty at lane 0's `t*` but EMPTY at a higher lane's `t*` is no longer
/// "consumed" by that lane (which would then dispatch empty and fail `admit` with
/// "contains no dispatchable nodes"). The rewritten graph is cached by
/// `(template_index, rank)` (shared with the dispatch loop's `_lane_plans`), so
/// the spawnability check and the eventual dispatch re-rewrite once. At the
/// default `[0, 0]` window every lane collapses to `t*=0` and the warmup graph is
/// the prebuilt lane-0 plan, so lane `i` takes corpus index `i` (sequential)
/// byte-for-byte with the prior assignment.
fn pressure_resolve_pass0_lanes(
    templates: &[PressureTemplate],
    lanes: usize,
    draw: &PressureDraw,
    t_star: TStarWindow,
    cache: &RefCell<HashMap<(usize, u64), GraphTracePlan>>,
) -> (Vec<usize>, u64) {
    let n = templates.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let mut pass0 = Vec::new();
    let mut cursor: u64 = 0;
    // Bound the skip walk to one full corpus pass past the wrap of `lanes` so an
    // all-unspawnable corpus can't spin (agentx caps at `_target_size + pool`);
    // we stop at `lanes` hits regardless.
    let max_cursor = lanes as u64 + n as u64;
    while pass0.len() < lanes && cursor < max_cursor {
        let idx = draw.index(cursor, n);
        cursor = cursor.saturating_add(1);
        // Judge spawnability at the candidate's TARGET rank (== the lane it would
        // dispatch on), so the salted `t*` here is the one the lane dispatches at
        // (Python `_is_spawnable(trace, rank)` with `rank = len(pass0)`).
        let rank = u64::try_from(pass0.len()).unwrap_or(u64::MAX);
        let lane_t_star_us = sample_lane_tstar(&templates[idx], rank, t_star);
        let plan = pressure_plan_for_lane(templates, idx, rank, lane_t_star_us, t_star, cache);
        if !plan.graph.nodes.is_empty() {
            pass0.push(idx);
        }
    }
    (pass0, cursor)
}

/// Compute one plan's lane-0 trajectory-start `t*` in microseconds.
///
/// Mirrors the per-plan `t*` sample in [`apply_tstar_split`] exactly (same
/// single-trace [`ParsedGraph`] view, [`trace_duration_us`], and
/// [`WindowTStarSampler`] at lane `0`), so the value stored in a lane's
/// [`GraphLaneIdentity`] equals the `t*` that produced the dispatched warmup
/// plan. The default `[0, 0]` window short-circuits to `0.0` (no RNG draw), the
/// only currently product-reachable pressure case.
fn sample_plan_tstar(original_plan: &GraphTracePlan, t_star: TStarWindow) -> f64 {
    if t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0 {
        return 0.0;
    }
    let sampler = WindowTStarSampler {
        start_min_ratio: t_star.start_min_ratio,
        start_max_ratio: t_star.start_max_ratio,
        random_seed: t_star.random_seed,
    };
    let duration_us = plan_trace_duration_us(original_plan);
    sampler.sample_t_star(&original_plan.trace.id, 0, duration_us)
}

/// Resolve one recycle lane's trajectory-start `t*` in microseconds.
///
/// Direct port of `graph_ir_replay.py:_plan_for_lane`/`_lane_source`
/// (lines 743-791, branch `ajc/aiperf-graph-ir`): lane `0` reuses the prebuilt
/// lane-0 `t*` (`template.t_star_us`, byte-identical to the single-pass plan),
/// so the first pass of every template is unchanged. A higher lane draws a
/// DISTINCT lane-salted `t*` (`sha256(seed:trace_id:lane)` via
/// [`seed_for_trace_lane`]) so the SAME template recurring across recycle lanes
/// resumes at a different snapshot instant, anchored to the ORIGINAL plan's
/// replayable span (`template.duration_us`) exactly as lane 0 was. The default
/// `[0, 0]` window collapses every lane to `t*=0` (identity) — both via this
/// early return and via `sample_t_star`'s `hi <= lo` short-circuit — so lane
/// fan-out adds no `t*` divergence on the working default-window profiling path.
fn sample_lane_tstar(template: &PressureTemplate, lane: u64, t_star: TStarWindow) -> f64 {
    lane_salted_tstar(
        &template.template_id,
        template.t_star_us,
        template.duration_us,
        lane,
        t_star,
    )
}

/// The lane-salted `t*` draw shared by the WARMUP ([`sample_lane_tstar`]) and
/// PROFILING ([`profiling_sample_lane_tstar`]) per-lane planners.
///
/// Direct port of `graph_ir_replay.py:_plan_for_lane`/`_lane_source` +
/// `graph_ir_source.py:_sample_t_star` (branch `ajc/aiperf-graph-ir`): lane `0`
/// (and the default `[0, 0]` window, whose every lane collapses to `t*=0`) reuses
/// the prebuilt lane-0 `t*` (`lane_0_tstar_us`, byte-identical to the single-pass
/// plan); a higher lane draws a DISTINCT `sha256(seed:template_id:lane)`-salted
/// `t*` anchored to the ORIGINAL plan's replayable span (`duration_us`). Factored
/// so the warmup boundary rewrite and the profiling frontier chop of the SAME
/// `(template, lane)` share ONE `t*` — the identity a lane records and the graph
/// it dispatches can never diverge.
fn lane_salted_tstar(
    template_id: &str,
    lane_0_tstar_us: f64,
    duration_us: f64,
    lane: u64,
    t_star: TStarWindow,
) -> f64 {
    if lane == 0 || (t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0) {
        return lane_0_tstar_us;
    }
    let sampler = WindowTStarSampler {
        start_min_ratio: t_star.start_min_ratio,
        start_max_ratio: t_star.start_max_ratio,
        random_seed: t_star.random_seed,
    };
    sampler.sample_t_star(template_id, lane, duration_us)
}

/// Resolve one PROFILING lane's trajectory-start `t*` in microseconds.
///
/// The [`GraphTracePlan`] analogue of [`sample_lane_tstar`]
/// (`graph_ir_replay.py:_plan_for_lane`, lines 743-791): the profiling side holds
/// the corpus as `original_plans` rather than [`PressureTemplate`]s, so the salt
/// inputs are read off the plan directly — `original_plan.trace.id` is the
/// template id (identical to the warmup template's `template_id`, which is the
/// warmup rewrite's `trace.id` == the original trace id) and
/// [`plan_trace_duration_us`] is the same span [`sample_lane_tstar`] anchors to.
/// The returned value is therefore BYTE-IDENTICAL to the warmup lane `t*` for the
/// same `(template, lane)`, so warmup priming and profiling resume of one lane
/// agree on the snapshot instant. Lane 0 / default `[0, 0]` window reuse the
/// lane-0 `t*` ([`sample_plan_tstar`]).
fn profiling_sample_lane_tstar(
    original_plan: &GraphTracePlan,
    lane: u64,
    t_star: TStarWindow,
) -> f64 {
    lane_salted_tstar(
        &original_plan.trace.id,
        sample_plan_tstar(original_plan, t_star),
        plan_trace_duration_us(original_plan),
        lane,
        t_star,
    )
}

/// Resolve the per-lane WARMUP GRAPH a recycle dispatch primes on `lane`.
///
/// Direct port of `graph_ir_replay.py:_plan_for_lane`/`_lane_source`
/// (lines 743-791, branch `ajc/aiperf-graph-ir`): lane `0` (and the default
/// `[0, 0]` window, whose every lane collapses to `t*=0`) reuses the prebuilt
/// lane-0 warmup graph (`template.plan`), byte-identical to the single-pass
/// path. A higher lane on an ACTIVE window re-primes the ORIGINAL graph at ITS
/// OWN lane-salted `t*` (`rewrite_for_warmup(parsed, lane_t*)`) so the executed
/// nodes match the `t*` that lane's [`GraphLaneIdentity`] records — the two are
/// the SAME `lane_t_star_us` value, so identity and dispatched graph can never
/// diverge. The rewritten graph is cached by `(template_index, lane)` (Python's
/// `_lane_plans`) so repeated recycle passes onto the same lane re-rewrite once.
fn pressure_plan_for_lane(
    templates: &[PressureTemplate],
    template_index: usize,
    lane: u64,
    lane_t_star_us: f64,
    t_star: TStarWindow,
    cache: &RefCell<HashMap<(usize, u64), GraphTracePlan>>,
) -> GraphTracePlan {
    let template = &templates[template_index];
    // Lane 0, or the default window (every lane `t*=0`): dispatch the prebuilt
    // lane-0 warmup graph verbatim — no per-lane divergence, no cache entry.
    if lane == 0 || (t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0) {
        return template.plan.clone();
    }
    let key = (template_index, lane);
    if let Some(cached) = cache.borrow().get(&key) {
        return cached.clone();
    }
    // Re-prime the ORIGINAL (pre-warmup) graph at the lane's own `t*`, the same
    // `rewrite_for_warmup` [`apply_tstar_split`] applied for lane 0 — just at the
    // lane-salted instant instead of lane 0's.
    let parsed = single_trace_parsed(&template.original_plan);
    let rewritten = rewrite_for_warmup(&parsed, lane_t_star_us);
    let plan = GraphTracePlan {
        graph: rewritten.graph,
        trace: template.original_plan.trace.clone(),
        arrival_offset_ns: template.original_plan.arrival_offset_ns,
    };
    cache.borrow_mut().insert(key, plan.clone());
    plan
}

/// Resolve the per-lane PROFILING PLAN a pass-0 profiling lane dispatches.
///
/// The [`GraphTracePlan`] frontier-chop analogue of [`pressure_plan_for_lane`]
/// (`graph_ir_replay.py:_run_instance`, lines 1837-1840: `plan =
/// _plan_for_lane(trace, lane_index)` for a pass-0, no-handoff, non-fresh-start
/// lane): lane `0` (and the default `[0, 0]` window) reuse the prebuilt lane-0
/// profiling split (`split[template_index]`, byte-identical to the pre-DF6
/// assignment). A higher lane on an ACTIVE window re-chops the ORIGINAL graph at
/// ITS OWN lane-salted `t*` ([`chop_trie_at_tstar`] at
/// [`profiling_sample_lane_tstar`]) — the SAME instant its warmup lane primed and
/// its [`GraphLaneIdentity`] records — instead of dispatching lane 0's post-`t*`
/// frontier. Cached by `(template_index, lane)` so the pass-0 spawnability scan
/// ([`profiling_resolve_pass0_lanes`]) and the eventual dispatch re-chop once.
fn profiling_plan_for_lane(
    original_plans: &[GraphTracePlan],
    split: &[GraphTracePlan],
    template_index: usize,
    lane: u64,
    t_star: TStarWindow,
    cache: &RefCell<HashMap<(usize, u64), GraphTracePlan>>,
) -> GraphTracePlan {
    // Lane 0, or the default window (every lane `t*=0`): dispatch the prebuilt
    // lane-0 profiling split verbatim — no per-lane divergence, no cache entry.
    if lane == 0 || (t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0) {
        return split[template_index].clone();
    }
    let key = (template_index, lane);
    if let Some(cached) = cache.borrow().get(&key) {
        return cached.clone();
    }
    // Re-chop the ORIGINAL (pre-split) graph at the lane's own `t*`, the same
    // `chop_trie_at_tstar` [`apply_tstar_split`] applied for lane 0 — just at the
    // lane-salted instant instead of lane 0's.
    let original = &original_plans[template_index];
    let lane_t_star_us = profiling_sample_lane_tstar(original, lane, t_star);
    let parsed = single_trace_parsed(original);
    let rewritten = chop_trie_at_tstar(&parsed, lane_t_star_us);
    let plan = GraphTracePlan {
        graph: rewritten.graph,
        trace: original.trace.clone(),
        arrival_offset_ns: original.arrival_offset_ns,
    };
    cache.borrow_mut().insert(key, plan.clone());
    plan
}

/// Measure one root plan's intrinsic replayable span in microseconds.
///
/// The single-trace [`ParsedGraph`] view + [`trace_duration_us`] shared by
/// [`sample_plan_tstar`], [`apply_tstar_split`], and the pressure recycle's
/// per-lane resample, so every lane's `t*` draw is anchored to the SAME duration
/// (the value `apply_tstar_split` used for lane 0), matching Python planning the
/// trace once and reusing the duration across lanes
/// (`graph_ir_replay.py:_plan_for_lane`, lines 743-772).
fn plan_trace_duration_us(plan: &GraphTracePlan) -> f64 {
    let view_trace = TraceRecord {
        id: plan.trace.id.clone(),
        graph_ref: None,
        initial_state: plan.trace.initial_state.clone(),
    };
    let parsed = ParsedGraph {
        graph: plan.graph.clone(),
        graphs: std::collections::BTreeMap::new(),
        traces: vec![view_trace.clone()],
    };
    trace_duration_us(&parsed, &view_trace)
}

/// In-runtime, duration-bounded cache-pressure warmup recycle controller.
///
/// The dataflow-native port of the Graph-IR strategy's pressure stage
/// (`graph_ir_replay.py:accelerated_warmup` `:1440` + `_run_pressure_lanes`
/// `:1488`): a per-lane recycle loop that, for `duration_ns` on the injected
/// [`Clock`], keeps `concurrency` lanes replaying the warmup corpus — lane `i`
/// starts on its pass-0 template and, when its instance drains, draws the next
/// wrap template from the shared corpus cursor and re-dispatches onto the freed
/// slot. It REGISTERS each dispatched instance with E3a's
/// [`GraphLaneLedger::register_lane`] so the executed-node/return-wall ledgers
/// (populated on the shared drain path via `observe_lane_return`) attribute each
/// terminal return to its lane, and advances the shared corpus cursor
/// ([`GraphLaneLedger::set_corpus_cursor`]) for E3c's resume.
///
/// It is engaged (constructed) ONLY for the warmup phase carrying
/// `agentic_cache_warmup_duration`; the no-pressure warmup keeps the unchanged
/// single-pass [`GraphWorkload`] path. Only the pressure DURATION check and a
/// consecutive-error backoff gate each lane loop — the `_can_recycle` session
/// gate is intentionally NOT consulted here (it lives in the profiling
/// `_run_lanes`, not the pressure loop, `graph_ir_replay.py:1488`). Handoff
/// assembly and profiling resume are E3c's; this stops at "lanes registered,
/// ledgers populated, cursor advanced".
struct GraphPressureRecycle {
    clock: Rc<dyn Clock>,
    placement: Rc<dyn TracePlacement>,
    progress: Rc<GraphPhaseProgress>,
    templates: Rc<Vec<PressureTemplate>>,
    duration_ns: i64,
    lane_target: usize,
    session_limit: Option<u64>,
    recycle_bounded: bool,
    /// The trajectory-start `t*` window. Each recycle lane resamples its OWN
    /// lane-salted `t*` from this window at dispatch time (port of
    /// `graph_ir_replay.py:_plan_for_lane`/`_lane_source`, lines 743-791): lane 0
    /// reuses the prebuilt lane-0 `t*` (byte-identical to the single-pass plan),
    /// higher lanes draw a distinct `sha256(seed:trace_id:lane)`-salted instant.
    /// The default `[0, 0]` window collapses every lane to `t*=0` (identity).
    t_star: TStarWindow,
    cancelled: Rc<Cell<bool>>,
    /// Number of pass-0 lanes this stage launched (`len(pass0_traces)`), stamped
    /// once [`run`](Self::run) resolves the lane fan-out. E3c's warmup teardown
    /// reads it as the handoff `pressure_lane_count`. Port of Python
    /// `self._pressure_lane_count = len(pass0_traces)`
    /// (`graph_ir_replay.py:1509`). `0` until the stage runs.
    pressure_lane_count: Cell<u64>,
}

impl GraphPressureRecycle {
    /// Latch external cancellation (the phase `stop_issuing` / drain teardown).
    ///
    /// Lanes check this each iteration and after each dispatch so the stage
    /// halts promptly, the analogue of Python's `_past_duration_deadline` +
    /// `wait_for` cancel cooperative twin (`graph_ir_replay.py:1228`).
    fn cancel(&self) {
        self.cancelled.set(true);
    }

    /// Drive the pressure recycle to its duration budget, then return so the
    /// phase lifecycle drains the last per-lane instances for the handoff.
    async fn run(&self) {
        let n = self.templates.len();
        if n == 0 {
            return;
        }
        let lanes = pressure_resolve_lane_count(
            self.lane_target,
            n,
            self.session_limit,
            self.recycle_bounded,
        );
        // Strategy-aware corpus-index draw shared by the pass-0 resolve and the
        // per-lane recycle below (`_draw_index`); one instance so `Shuffle`/
        // `Random` permutations are cached across every draw of this stage.
        let draw = Rc::new(PressureDraw::from_window(self.t_star));
        // Per-lane warmup-graph cache shared across the pass-0 spawnability scan
        // AND every lane task, the native `_lane_plans`
        // (`graph_ir_replay.py:_plan_for_lane`, lines 743-772): a lane's warmup
        // graph re-primed at its salted `t*` is computed once per
        // `(template_index, lane)` and reused, so judging spawnability at a lane's
        // `t*` and later dispatching that lane share ONE rewrite. A plain
        // `Rc<RefCell<_>>` has the single-loop atomicity Python's instance attr had.
        let lane_plans: Rc<RefCell<HashMap<(usize, u64), GraphTracePlan>>> =
            Rc::new(RefCell::new(HashMap::new()));
        let (pass0, cursor) =
            pressure_resolve_pass0_lanes(&self.templates, lanes, &draw, self.t_star, &lane_plans);
        self.progress.lanes.set_corpus_cursor(cursor);
        // Record the pass-0 lane fan-out for the E3c warmup handoff before any
        // lane runs (Python `_pressure_lane_count = len(pass0_traces)`).
        self.pressure_lane_count
            .set(u64::try_from(pass0.len()).unwrap_or(u64::MAX));
        if pass0.is_empty() {
            return;
        }
        // Clock-driven deadline: never `Instant::now`, so the budget holds under
        // SimClock (each lane checks it after a dispatch, exactly as Python's
        // `_run_pressure_lanes` checks `_past_duration_deadline`).
        let deadline_ns = self.clock.now_ns().saturating_add(self.duration_ns);
        // Shared recycle cursor: one event loop mutates it, so a plain `Cell`
        // has the atomicity Python's single-loop instance attr had.
        let next_index = Rc::new(Cell::new(cursor));
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<()>();
        for (lane_index, &start_template) in pass0.iter().enumerate() {
            let clock = self.clock.clone();
            let placement = self.placement.clone();
            let progress = self.progress.clone();
            let templates = self.templates.clone();
            let cancelled = self.cancelled.clone();
            let next_index = next_index.clone();
            let lane_plans = lane_plans.clone();
            let draw = draw.clone();
            let done_tx = done_tx.clone();
            let t_star = self.t_star;
            let lane = lane_index as u64;
            tokio::task::spawn_local(async move {
                let mut template_index = start_template;
                let mut consecutive_errors: i32 = 0;
                loop {
                    if cancelled.get() {
                        break;
                    }
                    let template = &templates[template_index];
                    // `{template}::{nonce}`: instance identity is the authored
                    // template plus a fresh nonce (Python `_run_pressure_lanes`
                    // uses `f"{trace.id}::{uuid4().hex}"`, `:1546`), so every
                    // recycle is a distinct correlation id.
                    let instance_id =
                        format!("{}::{}", template.template_id, Uuid::new_v4().simple());
                    // Per-lane `t*` salt: lane 0 reuses the prebuilt lane-0 `t*`
                    // (byte-identical to the single-pass plan); a higher recycle
                    // lane draws its OWN `sha256(seed:trace_id:lane)`-salted `t*`
                    // so the same template recurring across lanes resumes at a
                    // distinct snapshot instant (`graph_ir_replay.py:_plan_for_lane`,
                    // lines 743-791). Default `[0, 0]` window => every lane `t*=0`.
                    let lane_t_star_us = sample_lane_tstar(template, lane, t_star);
                    // Per-lane WARMUP GRAPH: lane 0 / default window dispatch the
                    // prebuilt lane-0 warmup graph; a higher lane on an active
                    // window primes the ORIGINAL graph re-rewritten at THIS lane's
                    // `t*` (the very `lane_t_star_us` registered below), so the
                    // executed-node set and the identity `t*` stay consistent
                    // (`graph_ir_replay.py:_plan_for_lane`, lines 743-791).
                    let mut plan = pressure_plan_for_lane(
                        &templates,
                        template_index,
                        lane,
                        lane_t_star_us,
                        t_star,
                        &lane_plans,
                    );
                    plan.trace.id = instance_id.clone();
                    // Register BEFORE dispatch so the instance's terminal node
                    // returns (drain path -> `observe_lane_return`) attribute to
                    // this lane; re-registering the lane on recycle overwrites its
                    // identity while leaving prior instances' ledger rows intact.
                    progress.lanes.register_lane(
                        lane,
                        GraphLaneIdentity {
                            template_trace_id: template.template_id.clone(),
                            instance_id: instance_id.clone(),
                            t_star_us: lane_t_star_us,
                        },
                    );
                    progress.admit(&TraceAdmissionInfo {
                        trace_id: instance_id,
                        node_count: plan.graph.nodes.len(),
                        arrival_ns: clock.now_ns(),
                    });
                    let result = placement.execute_trace(plan).await;
                    // A drain-cancel return ends the lane (a fresh instance would
                    // be rejected instantly); consecutive non-cancel errors back
                    // off exponentially so a deterministically failing server
                    // cannot hot-spin adapter builds for the whole duration.
                    let cancelled_return = matches!(result, Err(TraceError::Cancelled(_)));
                    match &result {
                        Ok(()) => consecutive_errors = 0,
                        Err(TraceError::Cancelled(_)) => {}
                        Err(_) => consecutive_errors = consecutive_errors.saturating_add(1),
                    }
                    if cancelled.get() || cancelled_return || clock.now_ns() >= deadline_ns {
                        break;
                    }
                    if consecutive_errors > 0 {
                        let backoff_s = (0.25 * 2f64.powi(consecutive_errors)).min(5.0);
                        let backoff_ns = (backoff_s * 1_000_000_000.0) as i64;
                        clock.clone().sleep(backoff_ns).await;
                        if cancelled.get() || clock.now_ns() >= deadline_ns {
                            break;
                        }
                    }
                    let drawn = next_index.get();
                    next_index.set(drawn.saturating_add(1));
                    progress.lanes.set_corpus_cursor(next_index.get());
                    template_index = draw.index(drawn, n);
                }
                let _ = done_tx.send(());
            });
        }
        drop(done_tx);
        while done_rx.recv().await.is_some() {}
    }
}

trait GraphPhaseProgressSink {
    fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String>;
    fn record_first_token(&self);
    fn record_returned(&self, returned: PhaseReturn);
    fn mark_all_sent(&self);
}

impl GraphPhaseProgressSink for PhaseContext {
    fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String> {
        PhaseContext::record_sent_batch(self, sent)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn record_returned(&self, returned: PhaseReturn) {
        PhaseContext::record_returned(self, returned);
    }

    fn record_first_token(&self) {
        PhaseContext::record_first_token(self);
    }

    fn mark_all_sent(&self) {
        PhaseContext::mark_all_sent(self);
    }
}

struct GraphPhaseProgress {
    sink: Rc<dyn GraphPhaseProgressSink>,
    failures: Rc<GraphPhaseFailures>,
    traces: RefCell<HashMap<String, GraphTracePhaseProgress>>,
    outcome: Rc<RefCell<GraphWorkloadReport>>,
    /// Injected clock: the sole source of the per-node return wall stamped into
    /// the lane ledger (`clock.now_ns() / 1000`); never `Instant::now()`.
    clock: Rc<dyn Clock>,
    /// Per-lane executed-node/return-wall observability for the E3c handoff.
    lanes: Rc<GraphLaneLedger>,
    /// Whether this phase is the WARMUP phase, gating warmup-failure accounting.
    is_warmup: bool,
    /// Run-scoped ledger of WARMUP-phase trace ids that produced a terminal,
    /// non-cancelled failure. Shared across every phase's progress so the
    /// sequencer can abort before PROFILING. Mirrors agentx
    /// `graph_ir_replay.py:_record_warmup_failure` (a WARMUP return carrying a
    /// non-None error that is NOT cancelled).
    warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
}

impl GraphPhaseProgress {
    #[allow(clippy::too_many_arguments)]
    fn new(
        sink: Rc<dyn GraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
        outcome: Rc<RefCell<GraphWorkloadReport>>,
        clock: Rc<dyn Clock>,
        lanes: Rc<GraphLaneLedger>,
        is_warmup: bool,
        warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
    ) -> Self {
        Self {
            sink,
            failures,
            traces: RefCell::new(HashMap::new()),
            outcome,
            clock,
            lanes,
            is_warmup,
            warmup_failed_trace_ids,
        }
    }

    /// Ledger one non-cancelled terminal node return into its lane's
    /// executed-node/return-wall observability.
    ///
    /// The wall is taken from the injected clock at return-handling time, the
    /// analogue of Python routing every `trace_id`-bearing return through
    /// `_record_return_wall` from `_on_graph_return` (`graph_ir_replay.py:964`).
    /// Cancelled returns are excluded by the caller: a drain-cancel is
    /// self-inflicted teardown and the server may never have executed the node,
    /// so keeping it out of `executed_node_ids` lets profiling refire it — the
    /// exact rationale in Python's `if self._pressure_enabled and not cancelled`
    /// guard (`graph_ir_replay.py:968`).
    fn observe_lane_return(&self, instance_id: &str, node_id: &str) {
        let wall_us = self.clock.now_ns() as f64 / 1_000.0;
        self.lanes.observe_return(instance_id, node_id, wall_us);
    }

    /// Ledger one terminal (non-cancelled) WARMUP-phase failure for `trace_id`,
    /// deduplicated so each failed trace is reported once. No-op outside the
    /// WARMUP phase. Port of `graph_ir_replay.py:908` (`_record_warmup_failure`);
    /// the run sequencer consumes the ledger to abort before PROFILING
    /// (`phase/runner.py:578` `report_warmup_failures`).
    fn note_warmup_failure(&self, trace_id: &str) {
        if !self.is_warmup {
            return;
        }
        let mut ledger = self.warmup_failed_trace_ids.borrow_mut();
        if !ledger.iter().any(|existing| existing == trace_id) {
            ledger.push(trace_id.to_owned());
        }
    }

    fn admit(&self, info: &TraceAdmissionInfo) {
        if info.node_count == 0 {
            self.failures.record(format!(
                "graph trace {:?} contains no dispatchable nodes",
                info.trace_id
            ));
            return;
        }
        if self
            .traces
            .borrow_mut()
            .insert(
                info.trace_id.clone(),
                GraphTracePhaseProgress {
                    expected_nodes: info.node_count,
                    returned_nodes: 0,
                    first_token_uuids: HashSet::new(),
                    returned_uuids: HashSet::new(),
                },
            )
            .is_some()
        {
            self.failures.record(format!(
                "graph trace {:?} was admitted more than once",
                info.trace_id
            ));
            return;
        }
        let mut sent = Vec::with_capacity(info.node_count);
        sent.push(PhaseSend::single_turn_session());
        sent.extend((1..info.node_count).map(|_| PhaseSend::dag_child()));
        if let Err(error) = self.sink.record_sent_batch(&sent) {
            self.failures.record(format!(
                "recording graph trace {:?} admitted send batch: {error}",
                info.trace_id
            ));
        }
        let mut outcome = self.outcome.borrow_mut();
        outcome.admitted = outcome.admitted.saturating_add(1);
    }

    fn record(&self, record: &CapturedRecord) {
        let (completes_session, released_at_first_token) = {
            let mut traces = self.traces.borrow_mut();
            let Some(trace) = traces.get_mut(&record.x_correlation_id) else {
                self.failures.record(format!(
                    "graph trace {:?} emitted a node record before admission or after completion",
                    record.x_correlation_id
                ));
                return;
            };
            if trace.returned_uuids.contains(&record.uuid) {
                self.failures.record(format!(
                    "graph trace {:?} emitted duplicate terminal record for request {}",
                    record.x_correlation_id, record.uuid
                ));
                return;
            }
            if trace.returned_nodes >= trace.expected_nodes {
                self.failures.record(format!(
                    "graph trace {:?} emitted more than {} node records",
                    record.x_correlation_id, trace.expected_nodes
                ));
                return;
            }
            let released_at_first_token = trace.first_token_uuids.contains(&record.uuid);
            let metadata_has_first_token = record.ingest.first_token_ns.is_some();
            if released_at_first_token != metadata_has_first_token {
                self.failures.record(format!(
                    "graph trace {:?} request {} first-token event/record mismatch: event={} record={}",
                    record.x_correlation_id,
                    record.uuid,
                    released_at_first_token,
                    metadata_has_first_token
                ));
            }
            trace.returned_uuids.insert(record.uuid);
            trace.returned_nodes += 1;
            (
                trace.returned_nodes == trace.expected_nodes,
                released_at_first_token,
            )
        };
        // AgentX warmup gate: a WARMUP node return carrying a non-cancelled
        // error is a terminal warmup failure. The graph node-failure policy for
        // the warmup priming turns is resilient (a failed reply does not abort
        // the boundary trace), so the failure surfaces here as an errored
        // per-node record rather than a trace-level `complete()` error — the
        // exact analogue of Python `_on_graph_return`'s per-return dispatch to
        // `_record_warmup_failure` (`graph_ir_replay.py:964`).
        if record.ingest.errored && !record.ingest.canceled {
            self.note_warmup_failure(&record.x_correlation_id);
        }
        self.sink.record_returned(PhaseReturn {
            completes_session,
            cancelled: record.ingest.canceled,
            errored: record.ingest.errored,
            releases_prefill: !released_at_first_token,
        });
    }

    fn first_token(&self, trace_id: &str, uuid: Uuid) {
        let mut traces = self.traces.borrow_mut();
        let Some(trace) = traces.get_mut(trace_id) else {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted a first token before admission or after completion"
            ));
            return;
        };
        if trace.returned_uuids.contains(&uuid) {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted first-token event after terminal record for request {uuid}"
            ));
            return;
        }
        if trace.first_token_uuids.contains(&uuid) {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted duplicate first-token event for request {uuid}"
            ));
            return;
        }
        if trace.first_token_uuids.len() >= trace.expected_nodes {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted more than {} first-token events",
                trace.expected_nodes
            ));
            return;
        }
        trace.first_token_uuids.insert(uuid);
        self.sink.record_first_token();
    }

    fn complete(
        &self,
        trace_id: &str,
        node_count: usize,
        requires_node_records: bool,
        result: &Result<(), TraceError>,
    ) {
        let Some(trace) = self.traces.borrow_mut().remove(trace_id) else {
            self.failures.record(format!(
                "graph trace {trace_id:?} completed before admission or more than once"
            ));
            return;
        };
        if trace.expected_nodes != node_count {
            self.failures.record(format!(
                "graph trace {trace_id:?} completed with {node_count} nodes after reserving {}",
                trace.expected_nodes
            ));
        }
        let missing = trace.expected_nodes.saturating_sub(trace.returned_nodes);
        let orphan_first_tokens = trace
            .first_token_uuids
            .difference(&trace.returned_uuids)
            .count();
        if orphan_first_tokens > missing {
            self.failures.record(format!(
                "graph trace {trace_id:?} has {orphan_first_tokens} first-token events without terminals but only {missing} missing terminal records"
            ));
        }
        let first_tokens_without_terminals = orphan_first_tokens.min(missing);
        let (cancelled, mut errored) = match result {
            Ok(()) => (false, false),
            Err(TraceError::Cancelled(_)) => (true, false),
            Err(_) => (false, true),
        };
        if result.is_ok() && missing > 0 && requires_node_records {
            errored = true;
            self.failures.record(format!(
                "graph trace {trace_id:?} completed successfully without {missing} reserved node records"
            ));
        }
        for index in 0..missing {
            self.sink.record_returned(PhaseReturn {
                completes_session: index + 1 == missing,
                cancelled,
                errored,
                releases_prefill: index >= first_tokens_without_terminals,
            });
        }
        if let Err(error) = result
            && !cancelled
        {
            self.failures
                .record(format!("graph trace {trace_id:?} failed: {error}"));
            // A WARMUP trace that aborts (fail-fast node policy) rather than
            // completing with an errored record still counts as a terminal
            // warmup failure for the pre-profiling abort gate.
            self.note_warmup_failure(trace_id);
        }
        let mut outcome = self.outcome.borrow_mut();
        match result {
            Ok(()) => outcome.completed = outcome.completed.saturating_add(1),
            Err(TraceError::Cancelled(_)) => {
                outcome.cancelled = outcome.cancelled.saturating_add(1);
            }
            Err(_) => outcome.failed = outcome.failed.saturating_add(1),
        }
        outcome.traces.push(GraphTraceRunResult {
            trace_id: trace_id.to_owned(),
            result: result.clone(),
        });
    }
}

#[derive(Default)]
struct GraphPhaseFailures {
    messages: RefCell<Vec<String>>,
    notify: Notify,
}

impl GraphPhaseFailures {
    fn record(&self, message: impl Into<String>) {
        self.messages.borrow_mut().push(message.into());
        self.notify.notify_waiters();
    }

    fn first(&self) -> Option<String> {
        self.messages.borrow().first().cloned()
    }

    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.first().is_some() {
                return;
            }
            notified.await;
        }
    }
}

struct GraphPhaseWorkloadObserver {
    progress: Rc<GraphPhaseProgress>,
}

impl GraphWorkloadObserver for GraphPhaseWorkloadObserver {
    fn on_trace_admit(&self, info: &TraceAdmissionInfo, _admit_ns: i64) {
        self.progress.admit(info);
    }

    fn on_sending_complete(&self, _at_ns: i64) {
        self.progress.sink.mark_all_sent();
    }
}

#[derive(Default)]
struct GraphRecordDrainStop {
    stopped: Cell<bool>,
    notify: Notify,
}

impl GraphRecordDrainStop {
    fn stop(&self) {
        if !self.stopped.replace(true) {
            self.notify.notify_waiters();
        }
    }
}

struct GraphPhaseExecution {
    clock: Rc<dyn Clock>,
    context: PhaseContext,
    workload: Rc<GraphWorkload>,
    placement: Rc<dyn TracePlacement>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_initial: Option<usize>,
    adaptive_control_variable: Option<AdaptiveControlVariable>,
    controller: Rc<dyn ScheduledPhaseController>,
    failures: Rc<GraphPhaseFailures>,
    events: RefCell<Option<mpsc::UnboundedReceiver<GraphExecutionEvent>>>,
    captured: Rc<RefCell<Vec<CapturedRecord>>>,
    progress: Rc<GraphPhaseProgress>,
    adaptive_sampler: Option<SharedWindowSampler>,
    sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    drain_stop: Rc<GraphRecordDrainStop>,
    drain_task: RefCell<Option<tokio::task::JoinHandle<()>>>,
    setup_error: Option<String>,
    /// Warmup cache-pressure recycle, `Some` only for a warmup phase carrying
    /// `agentic_cache_warmup_duration`. When present, [`execute`](PhaseExecution::execute)
    /// drives it in place of the single-pass workload and `stop_issuing` latches
    /// its cancel; `None` leaves the byte-unchanged workload path.
    pressure_recycle: Option<Rc<GraphPressureRecycle>>,
    /// Consume-once WARMUP -> PROFILING handoff slot, shared with the factory and
    /// every phase. A warmup phase with a pressure recycle STASHES a
    /// [`GraphWarmupHandoff`] here at the end of [`finalize`](PhaseExecution::finalize)
    /// (after its drain completes); non-warmup phases never write it.
    warmup_handoff: Rc<RefCell<Option<GraphWarmupHandoff>>>,
}

impl GraphPhaseExecution {
    fn start_record_drain(&self) -> Result<()> {
        ensure!(
            self.drain_task.borrow().is_none(),
            "graph record drain was already started"
        );
        let mut events = self
            .events
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("graph record receiver was already consumed"))?;
        let captured = self.captured.clone();
        let sampler = self.adaptive_sampler.clone();
        let progress = self.progress.clone();
        let stop = self.drain_stop.clone();
        *self.drain_task.borrow_mut() = Some(tokio::task::spawn_local(async move {
            loop {
                while let Ok(event) = events.try_recv() {
                    ingest_graph_execution_event(&captured, sampler.as_ref(), &progress, event);
                }
                if stop.stopped.get() {
                    return;
                }
                let stopped = stop.notify.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if stop.stopped.get() {
                    continue;
                }
                tokio::select! {
                    biased;
                    event = events.recv() => match event {
                        Some(event) => ingest_graph_execution_event(
                            &captured,
                            sampler.as_ref(),
                            &progress,
                            event,
                        ),
                        None => return,
                    },
                    () = &mut stopped => {}
                }
            }
        }));
        Ok(())
    }
}

fn ingest_graph_execution_event(
    captured: &Rc<RefCell<Vec<CapturedRecord>>>,
    sampler: Option<&SharedWindowSampler>,
    progress: &GraphPhaseProgress,
    event: GraphExecutionEvent,
) {
    match event {
        GraphExecutionEvent::FirstToken { trace_id, uuid } => {
            progress.first_token(&trace_id, uuid);
        }
        GraphExecutionEvent::Record { record, node_id } => {
            if let Some(sampler) = sampler {
                sampler.borrow_mut().on_record(&record.ingest);
            }
            progress.record(&record);
            // Attribute a non-cancelled terminal node return to its lane's
            // executed-node/return-wall ledgers. Cancelled returns are excluded
            // (see `observe_lane_return`); backends without a node id (offline
            // dynosim) carry `None` and never feed the warmup handoff.
            if let Some(node_id) = node_id.as_deref()
                && !record.ingest.canceled
            {
                progress.observe_lane_return(&record.x_correlation_id, node_id);
            }
            captured.borrow_mut().push(*record);
        }
        GraphExecutionEvent::TraceComplete {
            trace_id,
            node_count,
            requires_node_records,
            result,
        } => {
            progress.complete(&trace_id, node_count, requires_node_records, &result);
        }
    }
}

impl PhaseExecution for GraphPhaseExecution {
    fn configure(&self, config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        if let Some(error) = &self.setup_error {
            return Err(PhaseExecutionError::new(error.clone()));
        }
        if self.adaptive_control_variable != Some(AdaptiveControlVariable::Concurrency)
            && let (Some(limit), Some(slots)) = (config.concurrency, &self.session_slots)
        {
            slots.set_limit(limit);
        }
        if self.adaptive_control_variable != Some(AdaptiveControlVariable::PrefillConcurrency)
            && let Some(limit) = self.prefill_initial
        {
            self.placement
                .set_prefill_limit(limit)
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        }
        Ok(())
    }

    fn setup(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        Box::pin(async move {
            for sidecar in &sidecars {
                sidecar.start().await.map_err(|error| {
                    PhaseExecutionError::new(format!("starting graph phase sidecar: {error:#}"))
                })?;
            }
            let phase_start_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_start(phase_start_ns);
            }
            Ok(())
        })
    }

    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        self.start_record_drain()
            .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        self.controller
            .start()
            .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        if let Some(error) = self.failures.first() {
            return Err(PhaseExecutionError::new(error));
        }
        Ok(())
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        // Warmup cache-pressure recycle path: the in-runtime lane-recycle loop
        // replaces the single-pass workload for a warmup phase carrying
        // `agentic_cache_warmup_duration`. The loop is Clock-duration-bounded and
        // registers each lane's instance into the E3a ledger; a phase failure
        // still short-circuits. The no-pressure warmup never enters here.
        if let Some(recycle) = self.pressure_recycle.clone() {
            let failures = self.failures.clone();
            return Box::pin(async move {
                let run = recycle.run();
                let failed = failures.wait();
                tokio::pin!(run);
                tokio::pin!(failed);
                tokio::select! {
                    biased;
                    () = &mut failed => Err(PhaseExecutionError::new(
                        failures.first().unwrap_or_else(|| "graph warmup cache pressure failed".into())
                    )),
                    () = &mut run => match failures.first() {
                        Some(error) => Err(PhaseExecutionError::new(error)),
                        None => Ok(()),
                    },
                }
            });
        }
        let workload = self.workload.clone();
        let context = self.context.clone();
        let controller = self.controller.clone();
        let failures = self.failures.clone();
        Box::pin(async move {
            let execute = workload.execute();
            let adaptive_stop = controller.wait_until_stop();
            let failed = failures.wait();
            tokio::pin!(execute);
            tokio::pin!(adaptive_stop);
            tokio::pin!(failed);
            tokio::select! {
                biased;
                () = &mut failed => Err(PhaseExecutionError::new(
                    failures.first().unwrap_or_else(|| "graph phase failed".into())
                )),
                () = &mut adaptive_stop => {
                    workload.cancel();
                    context.mark_all_sent();
                    Ok(())
                }
                result = &mut execute => result
                    .map(|_| ())
                    .map_err(|error| PhaseExecutionError::new(error.to_string())),
            }
        })
    }

    fn stop_issuing(&self) {
        if let Some(recycle) = &self.pressure_recycle {
            recycle.cancel();
        }
        self.workload.cancel();
    }

    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let result = self
            .placement
            .cancel_inflight()
            .map_err(|error| PhaseExecutionError::new(error.to_string()));
        Box::pin(async move { result })
    }

    fn stop_ramps(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let controller = self.controller.clone();
        let failures = self.failures.clone();
        Box::pin(async move {
            controller
                .stop()
                .await
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
            match failures.first() {
                Some(error) => Err(PhaseExecutionError::new(error)),
                None => Ok(()),
            }
        })
    }

    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        self.drain_stop.stop();
        let drain = self.drain_task.borrow_mut().take();
        let failures = self.failures.clone();
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        let progress = self.progress.clone();
        let pressure_recycle = self.pressure_recycle.clone();
        let warmup_handoff = self.warmup_handoff.clone();
        Box::pin(async move {
            if let Some(drain) = drain {
                drain.await.map_err(|error| {
                    PhaseExecutionError::new(format!("graph record drain failed: {error}"))
                })?;
            }
            let phase_end_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_end(phase_end_ns);
            }
            for sidecar in &sidecars {
                sidecar.finish().await.map_err(|error| {
                    PhaseExecutionError::new(format!("finishing graph phase sidecar: {error:#}"))
                })?;
            }
            if let Some(error) = failures.first() {
                return Err(PhaseExecutionError::new(error));
            }
            // The record drain has now processed every WARMUP return, so the
            // shared warmup-failure ledger is complete. A WARMUP phase that
            // recorded a terminal (non-cancelled) trace failure fails HERE so
            // the orchestrator never advances to PROFILING — the resilient
            // warmup node policy lets the boundary trace complete, so this
            // finalize-time gate is the point at which the run must stop.
            // Mirrors agentx `report_warmup_failures` raising after warmup
            // returning-complete (`phase/runner.py:578`); `run_graph_phases`
            // then renders the structured `trajectory_warmup_failed` envelope
            // from the same ledger.
            if progress.is_warmup && !progress.warmup_failed_trace_ids.borrow().is_empty() {
                return Err(PhaseExecutionError::new(
                    "warmup phase recorded terminal trace failures; aborting before profiling",
                ));
            }
            // Extended-warmup handoff stash: the drain above has processed every
            // pressure return, so the per-lane ledger is complete and `now_ns`
            // here credits the whole drain wait toward each recorded gap
            // (agentx `finalize_phase`'s `finalized_at_ns` parity). Only a warmup
            // phase that RAN the pressure recycle stashes. Completeness gate
            // (Python `_stash_pressure_handoff`): if any admitted pressure trace
            // never returned (a cancelled run or finite grace) OR a failure was
            // recorded, a handoff built from the incomplete ledger would mark
            // server-executed nodes as not-executed and profiling would REFIRE
            // them — so skip the stash and let profiling start from the plain
            // `t*` plans. `progress.traces` empties as each admission completes,
            // so an empty map is the native `graph_all_returned()` analogue.
            if progress.is_warmup
                && let Some(recycle) = &pressure_recycle
                && progress.traces.borrow().is_empty()
                && failures.first().is_none()
            {
                // Nanoseconds -> microseconds, the same ledger scale
                // `observe_lane_return` stamps returns on (`now_ns / 1000`).
                let drain_end_wall_us = clock.now_ns() as f64 / 1_000.0;
                let handoff = build_warmup_handoff(
                    &progress.lanes,
                    recycle.pressure_lane_count.get(),
                    drain_end_wall_us,
                );
                *warmup_handoff.borrow_mut() = Some(handoff);
            }
            Ok(())
        })
    }
}

struct GraphPhaseExecutionFactory {
    phases: RefCell<HashMap<String, PreparedGraphPhase>>,
    sidecars: RefCell<HashMap<String, Vec<Rc<dyn ScheduledPhaseSidecar>>>>,
    placements: Vec<Rc<dyn TracePlacement>>,
    captured: Rc<RefCell<Vec<CapturedRecord>>>,
    outcome: Rc<RefCell<GraphWorkloadReport>>,
    /// Run-scoped ledger of terminal WARMUP-phase trace failures, consumed by
    /// [`run_graph_phases`] to abort before PROFILING.
    warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
    /// Captures the WARMUP phase's per-lane executed-node/return-wall ledger so
    /// E3c can read the drain frontier at warmup teardown. Populated in
    /// [`create`](Self::create) when the warmup phase's progress is built;
    /// remains `None` for a run with no warmup phase.
    warmup_lane_ledger: Rc<RefCell<Option<Rc<GraphLaneLedger>>>>,
    /// Consume-once WARMUP -> PROFILING handoff slot (the native
    /// `GraphPhaseChannel.warmup_handoff`). The warmup phase's `finalize` stashes
    /// a [`GraphWarmupHandoff`] here after its drain completes; the first
    /// profiling phase's [`create`](Self::create) pops it (and clears it).
    warmup_handoff: Rc<RefCell<Option<GraphWarmupHandoff>>>,
}

impl PhaseExecutionFactory for GraphPhaseExecutionFactory {
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        let Some(mut prepared) = self.phases.borrow_mut().remove(&config.id) else {
            return Rc::new(FailedGraphPhaseExecution {
                error: format!("graph phase {:?} has no prepared execution plan", config.id),
            });
        };
        let pressure_prepared = prepared.pressure.take();
        let lanes = Rc::new(GraphLaneLedger::default());
        if prepared.is_warmup {
            *self.warmup_lane_ledger.borrow_mut() = Some(lanes.clone());
        }
        let progress = Rc::new(GraphPhaseProgress::new(
            Rc::new(context.clone()),
            prepared.failures.clone(),
            self.outcome.clone(),
            context.clock(),
            lanes,
            prepared.is_warmup,
            self.warmup_failed_trace_ids.clone(),
        ));
        let observer = Rc::new(GraphPhaseWorkloadObserver {
            progress: progress.clone(),
        });
        let mut setup_error = None;
        // Consume-once warmup handoff: EVERY create pops the shared slot (the
        // warmup phase's create runs before any stash, so it pops `None`; the
        // first profiling create after an extended warmup pops the stashed
        // handoff; a later phase sees `None` and never a stale re-cut). Port of
        // `phase/runner.py:416-424` (`warmup_handoff = channel.warmup_handoff;
        // channel.warmup_handoff = None`). A non-warmup phase with resume
        // ingredients rebuilds its trace source at the drain frontier; every
        // other case keeps the byte-unchanged prepared (`chop_trie_at_tstar`)
        // workload, so the no-pressure path is identical to today.
        let popped_handoff = self.warmup_handoff.borrow_mut().take();
        let base_workload = match (prepared.resume.take(), popped_handoff) {
            (Some(resume), Some(handoff)) => {
                match rebuild_resume_workload(&resume, prepared.placement.clone(), &handoff) {
                    Ok(workload) => workload,
                    Err(error) => {
                        setup_error = Some(format!(
                            "building warmup handoff resume workload: {error:#}"
                        ));
                        prepared.workload
                    }
                }
            }
            _ => prepared.workload,
        };
        let workload = Rc::new(base_workload.with_observer(observer));
        let mut controller = prepared.controller;
        let adaptive_control_variable = prepared
            .adaptive
            .as_ref()
            .map(|adaptive| adaptive.control_variable);
        let adaptive_sampler = prepared.adaptive.map(|adaptive| {
            let sampler: SharedWindowSampler = Rc::new(RefCell::new(Box::new(
                TumblingWindowSampler::new(context.clock().now_ns()),
            )));
            match graph_adaptive_actuator(
                &adaptive,
                prepared.session_slots.clone(),
                prepared.intervals.clone(),
                prepared.placement.clone(),
            )
            .and_then(|actuator| {
                build_adaptive_scale(adaptive, context.clock(), actuator, sampler.clone())
            }) {
                Ok(scale) => {
                    controller = Rc::new(AdaptiveScheduledPhaseController::new(
                        scale,
                        controller.clone(),
                    ));
                }
                Err(error) => setup_error = Some(error.to_string()),
            }
            sampler
        });
        let sidecars = self
            .sidecars
            .borrow_mut()
            .remove(&config.id)
            .unwrap_or_default();
        // Bind the prepared pressure recycle to this phase's placement + progress
        // (built above). Present only for a warmup phase with a cache-pressure
        // duration; the sole driver of E3a `register_lane` on the product path.
        let pressure_recycle = pressure_prepared.map(|prepared_pressure| {
            Rc::new(GraphPressureRecycle {
                clock: context.clock(),
                placement: prepared.placement.clone(),
                progress: progress.clone(),
                templates: prepared_pressure.templates,
                duration_ns: prepared_pressure.duration_ns,
                lane_target: prepared_pressure.lane_target,
                session_limit: prepared_pressure.session_limit,
                recycle_bounded: prepared_pressure.recycle_bounded,
                t_star: prepared_pressure.t_star,
                cancelled: Rc::new(Cell::new(false)),
                pressure_lane_count: Cell::new(0),
            })
        });
        Rc::new(GraphPhaseExecution {
            clock: context.clock(),
            context,
            workload,
            placement: prepared.placement,
            session_slots: prepared.session_slots,
            prefill_initial: prepared.prefill_initial,
            adaptive_control_variable,
            controller,
            failures: prepared.failures,
            events: RefCell::new(Some(prepared.events)),
            captured: self.captured.clone(),
            progress,
            adaptive_sampler,
            sidecars,
            drain_stop: Rc::new(GraphRecordDrainStop::default()),
            drain_task: RefCell::new(None),
            setup_error,
            pressure_recycle,
            warmup_handoff: self.warmup_handoff.clone(),
        })
    }

    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let errors = self
            .placements
            .iter()
            .enumerate()
            .filter_map(|(index, placement)| {
                placement
                    .cancel_inflight()
                    .err()
                    .map(|error| format!("placement {index}: {error}"))
            })
            .collect::<Vec<_>>();
        let result = if errors.is_empty() {
            Ok(())
        } else {
            Err(PhaseExecutionError::new(format!(
                "cancelling graph placements: {}",
                errors.join("; ")
            )))
        };
        Box::pin(async move { result })
    }
}

struct FailedGraphPhaseExecution {
    error: String,
}

impl PhaseExecution for FailedGraphPhaseExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        Err(PhaseExecutionError::new(self.error.clone()))
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let error = PhaseExecutionError::new(self.error.clone());
        Box::pin(async move { Err(error) })
    }
}

fn graph_adaptive_actuator(
    config: &AdaptiveRunConfig,
    session_slots: Option<Rc<SlotPool>>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    placement: Rc<dyn TracePlacement>,
) -> Result<Rc<dyn ControlActuator>> {
    Ok(match config.control_variable {
        AdaptiveControlVariable::Concurrency => Rc::new(SessionConcurrencyActuator::new(
            session_slots
                .ok_or_else(|| anyhow!("adaptive graph concurrency requires session admission"))?,
            integer_adaptive_bound(config.minimum, "concurrency minimum")?,
            integer_adaptive_bound(config.maximum, "concurrency maximum")?,
        )?),
        AdaptiveControlVariable::PrefillConcurrency => Rc::new(GraphPrefillActuator::new(
            placement,
            integer_adaptive_bound(config.minimum, "prefill minimum")?,
            integer_adaptive_bound(config.maximum, "prefill maximum")?,
        )?),
        AdaptiveControlVariable::RequestRate => Rc::new(RequestRateActuator::new(
            intervals,
            config.minimum,
            config.maximum,
        )?),
        AdaptiveControlVariable::Users => {
            bail!("adaptive users is not defined for Graph-IR phases")
        }
    })
}

struct GraphPrefillActuator {
    placement: Rc<dyn TracePlacement>,
    minimum: usize,
    maximum: usize,
    current: Cell<usize>,
}

impl GraphPrefillActuator {
    fn new(placement: Rc<dyn TracePlacement>, minimum: usize, maximum: usize) -> Result<Self> {
        ensure!(
            minimum > 0,
            "adaptive graph prefill minimum must be positive"
        );
        ensure!(
            maximum > minimum,
            "adaptive graph prefill maximum must be greater than minimum"
        );
        Ok(Self {
            placement,
            minimum,
            maximum,
            current: Cell::new(minimum),
        })
    }
}

impl ControlActuator for GraphPrefillActuator {
    fn variable(&self) -> &'static str {
        "prefill_concurrency"
    }

    fn minimum(&self) -> f64 {
        self.minimum as f64
    }

    fn maximum(&self) -> f64 {
        self.maximum as f64
    }

    fn current(&self) -> f64 {
        self.current.get() as f64
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        if !value.is_finite() {
            return Err(AdaptiveError::Actuator(format!(
                "graph prefill control value must be finite, got {value}"
            )));
        }
        let value = value
            .clamp(self.minimum as f64, self.maximum as f64)
            .trunc() as usize;
        self.placement
            .set_prefill_limit(value)
            .map_err(|error| AdaptiveError::Actuator(error.to_string()))?;
        self.current.set(value);
        Ok(value as f64)
    }

    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot {
            target_value: self.current(),
            actual_value: self.current(),
            active_users: None,
            retiring_users: None,
            cancelled: None,
        }
    }
}

/// Lower and execute every authored Graph-IR phase through one lifecycle path.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_graph_phases(
    phases: &[PhaseSpec],
    benchmark_id: &str,
    artifact_dir: &Path,
    input: &GraphInputBundle,
    clock: Rc<dyn Clock>,
    rng_root: RngRoot,
    allow_dataset_wrap: bool,
    t_star: TStarWindow,
    phase_sidecars: Vec<Vec<Rc<dyn ScheduledPhaseSidecar>>>,
    backends: &dyn GraphPhaseBackendFactory,
    on_failure: OnFailure,
) -> Result<GraphPhaseRunOutput> {
    validate_dataset_wrap_policy(phases, input, allow_dataset_wrap)?;
    ensure!(
        phase_sidecars.len() == phases.len(),
        "graph phase sidecars must be provided one list per phase"
    );
    let trace_instances = GraphTraceInstanceSequence::default();
    let session_slots = phases
        .iter()
        .any(graph_phase_uses_session_admission)
        .then(|| Rc::new(SlotPool::new(1)));
    let mut prepared = Vec::with_capacity(phases.len());
    for (phase_index, phase) in phases.iter().enumerate() {
        prepared.push(prepare_graph_phase(
            phase_index,
            phase,
            benchmark_id,
            artifact_dir,
            input,
            clock.clone(),
            rng_root,
            t_star,
            trace_instances.clone(),
            session_slots.clone(),
            backends,
            on_failure,
        )?);
    }

    let captured = Rc::new(RefCell::new(Vec::new()));
    let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
    let placements = prepared
        .iter()
        .map(|phase| phase.placement.clone())
        .collect();
    let phase_configs = phases
        .iter()
        .enumerate()
        .map(|(index, spec)| phase_config(spec, phase_seamless_to_next(phases, index)))
        .collect::<Result<Vec<_>>>()?;
    let prepared = prepared
        .into_iter()
        .zip(&phase_configs)
        .map(|(phase, config)| (config.id.clone(), phase))
        .collect::<HashMap<_, _>>();
    // Side-channel telemetry producers are barrier-synchronized, not per-token,
    // so the graph runtime drives them through the shared phase-sidecar seam the
    // scheduled runtime uses: start before issuance, finish after drain.
    let sidecars = phase_configs
        .iter()
        .zip(phase_sidecars)
        .map(|(config, sidecars)| (config.id.clone(), sidecars))
        .collect::<HashMap<_, _>>();
    let warmup_failed_trace_ids = Rc::new(RefCell::new(Vec::new()));
    // Captures the warmup phase's per-lane executed-node/return-wall ledger for
    // the E3c cache-pressure handoff; the recycle loop (E3b) registers lanes,
    // and the handoff assembly (E3c) reads it here after the warmup phase drains.
    let warmup_lane_ledger: Rc<RefCell<Option<Rc<GraphLaneLedger>>>> = Rc::new(RefCell::new(None));
    // Consume-once WARMUP -> PROFILING handoff slot (the native
    // `GraphPhaseChannel.warmup_handoff`): the warmup phase stashes at teardown,
    // the first profiling phase pops-and-clears at create. `None` on every
    // no-pressure run.
    let warmup_handoff: Rc<RefCell<Option<GraphWarmupHandoff>>> = Rc::new(RefCell::new(None));
    let execution_factory: Rc<dyn PhaseExecutionFactory> = Rc::new(GraphPhaseExecutionFactory {
        phases: RefCell::new(prepared),
        sidecars: RefCell::new(sidecars),
        placements,
        captured: captured.clone(),
        outcome: outcome.clone(),
        warmup_failed_trace_ids: warmup_failed_trace_ids.clone(),
        warmup_lane_ledger: warmup_lane_ledger.clone(),
        warmup_handoff: warmup_handoff.clone(),
    });
    let phase_observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
    // The virtual (offline) clock has no signal driver, so the SIGINT/SIGTERM
    // listener is armed only under a wall clock; capture the axis before the
    // clock is moved into the runner factory.
    let clock_is_virtual = clock.is_virtual();
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock,
        phase_observer.clone(),
        execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(phase_configs, runner_factory, phase_observer)?;
    // A terminal (non-cancelled) failure during the WARMUP phase already fails
    // that phase, so `run_all` returns before PROFILING starts. Before
    // propagating any error we consult the warmup ledger: if the WARMUP phase
    // recorded terminal trace failures, the run aborts with the structured
    // `trajectory_warmup_failed` envelope (the `TrajectoryWarmupFailedError`
    // analogue) so benchmark numbers are never taken from a pool the warmup
    // could not faithfully prime. Mirrors agentx `phase/runner.py:578`
    // (`report_warmup_failures`) raising before PROFILING. Ctrl-C during the
    // graph run drains through the same cancellation latch as the scheduled path.
    let run_result = drive_phases(orchestrator, clock_is_virtual).await;
    let warmup_failures = std::mem::take(&mut *warmup_failed_trace_ids.borrow_mut());
    if !warmup_failures.is_empty() {
        return Err(trajectory_warmup_failed_error(&warmup_failures));
    }
    let phase_stats = run_result?;

    let mut captured = std::mem::take(&mut *captured.borrow_mut());
    captured.sort_by(|left, right| {
        left.ingest
            .start_ns
            .cmp(&right.ingest.start_ns)
            .then_with(|| left.uuid.cmp(&right.uuid))
    });
    for (request_index, record) in captured.iter_mut().enumerate() {
        record.ingest.request_index = Some(request_index);
    }
    let workload = std::mem::take(&mut *outcome.borrow_mut());
    Ok(GraphPhaseRunOutput {
        captured,
        phases: phase_stats,
        workload,
    })
}

/// Build the structured `trajectory_warmup_failed` protocol-v2 execution
/// failure for a WARMUP phase that recorded terminal trace failures.
///
/// The wire envelope is a stable lowercase-snake-case `code`
/// (`trajectory_warmup_failed`, the `kind`) plus a message listing the failed
/// trace ids — the `{ kind, failed_trace_ids }` analogue in the runner's
/// `code` + `message` diagnostic shape. Returned as a downcastable
/// [`PreparedRunFailure`] so the process coordinator emits an
/// execution-stage failure envelope rather than the generic `execution_failed`
/// fallback. Mirrors Python `TrajectoryWarmupFailedError`
/// (`src/aiperf/common/scenario/base.py:182`), whose message likewise embeds the
/// failed trace ids, raised by `report_warmup_failures`
/// (`graph_ir_replay.py:908-931`, `phase/runner.py:578`).
fn trajectory_warmup_failed_error(failed_trace_ids: &[String]) -> anyhow::Error {
    let message = format!(
        "Trajectory warmup failed for {} trace(s): {}. Run aborted to preserve metrics integrity.",
        failed_trace_ids.len(),
        failed_trace_ids.join(", ")
    );
    match PreparedRunFailure::new(
        FailureStageV2::Execution,
        "trajectory_warmup_failed",
        message,
    ) {
        Ok(failure) => anyhow::Error::new(failure),
        Err(error) => error,
    }
}

fn validate_dataset_wrap_policy(
    phases: &[PhaseSpec],
    input: &GraphInputBundle,
    allow_dataset_wrap: bool,
) -> Result<()> {
    if allow_dataset_wrap {
        return Ok(());
    }
    let distinct = u64::try_from(input.plans.len()).context("graph root count exceeds u64")?;
    for phase in phases {
        let common = phase.common();
        let one_pass =
            common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
        if one_pass || common.sessions.is_some_and(|sessions| sessions <= distinct) {
            continue;
        }
        let concurrency = phase.concurrency().unwrap_or(1);
        if u64::try_from(concurrency).context("graph concurrency exceeds u64")? > distinct {
            bail!(
                "graph phase {:?} concurrency {} exceeds the {} distinct loaded traces while dataset wrapping is disabled; reduce concurrency to at most {}, bound sessions within the loaded corpus, or set dataset.synthesis.allow_dataset_wrap=true",
                common.name,
                concurrency,
                distinct,
                distinct
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_graph_phase(
    phase_index: usize,
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
    input: &GraphInputBundle,
    clock: Rc<dyn Clock>,
    rng_root: RngRoot,
    t_star: TStarWindow,
    trace_instances: GraphTraceInstanceSequence,
    session_slots: Option<Rc<SlotPool>>,
    backends: &dyn GraphPhaseBackendFactory,
    on_failure: OnFailure,
) -> Result<PreparedGraphPhase> {
    let phase_index = u64::try_from(phase_index).context("graph phase index exceeds u64")?;
    let phase_rng = rng_root.derive_indexed_root(namespace::GRAPH_PHASE, phase_index);
    let common = phase.common();
    let one_pass =
        common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
    let session_limit = if one_pass {
        Some(u64::try_from(input.plans.len()).context("graph root count exceeds u64")?)
    } else {
        common.sessions
    };
    // A multi-cell process owns only its interleaved slice of the global session
    // ordinals, so it swaps the single-cell cycler for the partitioned source. The
    // `common.requests.is_none()` guard keeps a configured static-node request budget
    // on the cycler because `PartitionedGraphTraceSource` does not yet slice that
    // budget across cells; a later controller step must own the split before a
    // per-cell run can honor a static request_limit.
    // Per-phase trajectory-start snapshot split: the warmup phase primes each
    // trace's boundary turns (`rewrite_for_warmup`), the profiling (and any other
    // non-warmup) phase replays only the post-`t*` frontier (`chop_trie_at_tstar`).
    // At the default `[0, 0]` window `t*` is `0` for every trace, so profiling is
    // the unchanged full graph and warmup is empty. Both phases sample the SAME
    // deterministic `t*` per trace, so warmup primes exactly what profiling resumes.
    let phase_plans = apply_tstar_split(&input.plans, phase, t_star);
    // Warmup cache-pressure recycle: engaged ONLY for the warmup phase carrying
    // `agentic_cache_warmup_duration` (Python `_init_accelerated_warmup`,
    // `graph_ir_replay.py:468`). It runs the warmup-rewritten corpus in a
    // duration-bounded per-lane recycle instead of the single-pass workload; the
    // no-pressure warmup leaves `pressure` `None` and is byte-unchanged. Built
    // here (before `phase_plans` moves into the source) from the same warmup
    // plans the source would cycle, paired with each template's lane-0 `t*`.
    let pressure = build_pressure_recycle(&phase_plans, &input.plans, phase, common, t_star)
        .with_context(|| {
            format!("preparing warmup cache-pressure recycle for phase {phase_index}")
        })?;
    let source = build_graph_trace_source(
        phase_plans,
        session_limit,
        common.requests,
        trace_instances.clone(),
        // Up-front build: no warmup handoff yet, so the corpus draw starts at 0.
        // A profiling phase that later pops a handoff rebuilds its source with the
        // resume cursor in `rebuild_resume_workload`.
        0,
        t_star,
    )?;
    let seed = phase_rng.derive_seed_or_entropy(namespace::GRAPH_ARRIVAL);
    let intervals = Rc::new(RefCell::new(match phase.request_arrival() {
        Some((pattern, rate, smoothness)) => {
            make_interval_generator(pattern, rate, smoothness, seed)
        }
        None => make_interval_generator(
            crate::timing::ArrivalPattern::ConcurrencyBurst,
            None,
            None,
            seed,
        ),
    }));
    let arrival: Rc<dyn GraphArrivalPolicy> = match phase {
        PhaseSpec::Concurrency { .. } => Rc::new(ImmediateGraphArrival),
        PhaseSpec::Poisson { .. } | PhaseSpec::Gamma { .. } | PhaseSpec::Constant { .. } => {
            Rc::new(IntervalGraphArrival::new(intervals.clone()))
        }
        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => {
            unreachable!("unsupported graph phase rejected before input acquisition")
        }
    };
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let event_sink: Arc<dyn GraphExecutionEventSink> =
        Arc::new(ChannelRunnerGraphExecutionEventSink::new(events_tx));
    let adaptive = graph_adaptive_config(phase, benchmark_id, artifact_dir)?;
    let prefill_initial = match (common.prefill_concurrency, adaptive.as_ref()) {
        (Some(limit), _) => Some(limit),
        (None, Some(config))
            if config.control_variable == AdaptiveControlVariable::PrefillConcurrency =>
        {
            Some(integer_adaptive_bound(config.minimum, "prefill minimum")?)
        }
        (None, _) => None,
    };
    let cancellation = common
        .cancellation
        .map(|cancellation| GraphCancellationConfig {
            rate: cancellation.rate,
            delay_seconds: cancellation.delay,
            rng_root: phase_rng.derive_root(namespace::GRAPH_NODE_CANCELLATION),
            phase: if common.name == "warmup" {
                crate::timing::Phase::Warmup
            } else {
                crate::timing::Phase::Profiling
            },
        });
    let backend = backends.prepare_backend(GraphPhaseBackendConfig {
        metrics_phase: metrics_phase(phase)?,
        prefill_concurrency: prefill_initial,
        cancellation,
        events: event_sink.clone(),
    })?;
    let placement: Rc<dyn TracePlacement> = Rc::new(ObservedRunnerGraphPlacement::new(
        backend.placement,
        event_sink,
        backend.requires_node_records,
    ));
    let failures = Rc::new(GraphPhaseFailures::default());
    let controller = graph_ramp_controller(
        phase,
        clock.clone(),
        intervals.clone(),
        session_slots.clone(),
        placement.clone(),
        phase_rng,
        failures.clone(),
    )?;
    // Run-level failure discipline is config-selected (default fail-fast for the
    // graph path). `Abort` latches the whole run on the first non-cancellation
    // trace failure; `Continue` keeps admitting unrelated roots and records the
    // failed traces (the coordinator relaxes its `failed == 0` assertion to
    // match). See `specs/2026-07-13-scheduled-graph-convergence-implementation.md`.
    let run_failure: Rc<dyn RunFailurePolicy> = match on_failure {
        OnFailure::Abort => Rc::new(FailFastRunFailurePolicy::default()),
        OnFailure::Continue => Rc::new(ContinueRunFailurePolicy),
    };
    let uses_admission = graph_phase_uses_session_admission(phase);
    let is_warmup = common.name == "warmup";
    // The frontier-resume ingredients are shared (`Rc` clones) with the workload
    // built here; only a non-warmup phase that later pops a warmup handoff
    // rebuilds its source, so the arrival/run-failure objects are never used by
    // two live workloads at once (the discarded prepared workload is dropped).
    let resume = if is_warmup {
        None
    } else {
        Some(ProfilingResume {
            original_plans: Rc::new(input.plans.clone()),
            phase: phase.clone(),
            t_star,
            session_limit,
            request_limit: common.requests,
            trace_instances,
            arrival: arrival.clone(),
            run_failure: run_failure.clone(),
            uses_admission,
            session_slots: session_slots.clone(),
            clock: clock.clone(),
        })
    };
    let workload = assemble_graph_workload(
        clock,
        source,
        placement.clone(),
        arrival,
        run_failure,
        uses_admission,
        session_slots.clone(),
    )?;
    Ok(PreparedGraphPhase {
        workload,
        placement,
        events: events_rx,
        intervals,
        session_slots,
        prefill_initial,
        controller,
        failures,
        adaptive,
        is_warmup,
        pressure,
        resume,
    })
}

/// Build the warmup cache-pressure recycle inputs, or `None` when this phase is
/// not a warmup carrying `agentic_cache_warmup_duration`.
///
/// Pairs each warmup-rewritten plan (`warmup_plans`, already `apply_tstar_split`
/// output for this phase) with its authored template id and lane-0 `t*` sampled
/// from the ORIGINAL plan (`original_plans`, same order), so a registered lane's
/// identity carries the exact `t*` that produced the dispatched warmup graph.
/// The pressure duration comes from the wire `agentic_cache_warmup_duration`
/// (the strategy's `_cache_pressure_duration_s`), the lane target from the phase
/// concurrency (the CONCURRENCY_BURST width), and `recycle_bounded` from any
/// explicit phase stop condition (none on the auto warmup).
fn build_pressure_recycle(
    warmup_plans: &[GraphTracePlan],
    original_plans: &[GraphTracePlan],
    phase: &PhaseSpec,
    common: &PhaseCommonSpec,
    t_star: TStarWindow,
) -> Result<Option<PreparedPressureRecycle>> {
    if common.name != "warmup" {
        return Ok(None);
    }
    let Some(duration_s) = common.agentic_cache_warmup_duration else {
        return Ok(None);
    };
    ensure!(
        duration_s.is_finite() && duration_s > 0.0,
        "warmup agentic_cache_warmup_duration must be finite and positive"
    );
    let duration_ns = i64::try_from(seconds_to_u64_ns(duration_s)?)
        .context("warmup cache-pressure duration in nanoseconds exceeds i64")?;
    let templates = warmup_plans
        .iter()
        .zip(original_plans.iter())
        .map(|(warmup_plan, original_plan)| PressureTemplate {
            plan: warmup_plan.clone(),
            template_id: warmup_plan.trace.id.clone(),
            t_star_us: sample_plan_tstar(original_plan, t_star),
            duration_us: plan_trace_duration_us(original_plan),
            original_plan: original_plan.clone(),
        })
        .collect::<Vec<_>>();
    let lane_target = phase.concurrency().unwrap_or(1).max(1);
    // Only an explicit phase stop condition makes recycle "bounded" (governs the
    // corpus lane clamp). The auto warmup carries none — the pressure duration
    // deadline is separate from a phase stop condition, exactly as Python's
    // `_recycle_has_stop_condition` excludes the pressure `_duration_deadline`.
    let recycle_bounded =
        common.duration.is_some() || common.requests.is_some() || common.sessions.is_some();
    Ok(Some(PreparedPressureRecycle {
        templates: Rc::new(templates),
        duration_ns,
        lane_target,
        session_limit: common.sessions,
        recycle_bounded,
        t_star,
    }))
}

/// Apply the per-trace trajectory-start (`t*`) snapshot split for one phase.
///
/// For each root plan the intrinsic replayable span is measured
/// ([`trace_duration_us`]), a per-trace `t*` is drawn from the window
/// ([`WindowTStarSampler`]), and the plan graph is rewritten: the WARMUP phase
/// keeps only the boundary priming turns ([`rewrite_for_warmup`]); every other
/// phase (profiling) keeps only the live post-`t*` frontier
/// ([`chop_trie_at_tstar`]).
///
/// The default `[0, 0]` window is a TRUE NO-OP: BOTH warmup and profiling return
/// the plans UNCHANGED. This restores pre-`t*`-split behavior — a plain
/// (`dag_jsonl` / no-scenario) run has a normal warmup that dispatches the full
/// graph. The t* warmup/profiling split only engages when a snapshot window is
/// actually configured (non-default). In the Python source the warmup graph is
/// routed through [`rewrite_for_warmup`] only under the GRAPH_IR replay strategy
/// that a t* scenario activates; without a scenario, `rewrite_for_warmup` at
/// `t* = 0` would empty the warmup graph and every trace would report "contains
/// no dispatchable nodes". The default-window early return skips both the
/// per-trace sample and the rewrite (also avoiding a needless per-plan clone).
///
/// Warmup and profiling are independent phases that each clone `plans`, but they
/// sample the identical deterministic `t*` per trace (same window, seed, trace
/// id, and lane), so warmup primes exactly the prefix profiling resumes from.
///
/// Lane index: this seam samples each root template's LANE-0 `t*` — the base
/// single-pass product selection dispatches each root once, matching Python's
/// first/only lane (`0`). Per-lane `t*` decorrelation IS built (DF1b/DF6): the
/// recycle/resume paths re-sample and re-rewrite a higher lane at its own salted
/// `t*` via [`pressure_plan_for_lane`] (warmup) and [`profiling_plan_for_lane`]
/// (profiling), so this lane-0 split is the shared anchor those per-lane planners
/// reuse for lane 0 and the default window. Mirrors
/// `graph_ir_source.py:_plan_trace` selecting the chop vs the warmup rewrite per
/// phase.
fn apply_tstar_split(
    plans: &[GraphTracePlan],
    phase: &PhaseSpec,
    t_star: TStarWindow,
) -> Vec<GraphTracePlan> {
    // Default `[0, 0]` window: the split is inactive, so leave both warmup and
    // profiling graphs untouched (pre-split behavior). Only a configured
    // (non-default) window engages the per-trace warmup/profiling rewrite.
    if t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0 {
        return plans.to_vec();
    }
    let is_warmup = phase.common().name == "warmup";
    let sampler = WindowTStarSampler {
        start_min_ratio: t_star.start_min_ratio,
        start_max_ratio: t_star.start_max_ratio,
        random_seed: t_star.random_seed,
    };
    plans
        .iter()
        .map(|plan| {
            // A single-trace `ParsedGraph` view over the already-resolved plan
            // graph. The view trace clears `graph_ref` so `resolve_trace_graph`
            // returns this graph directly (recorded plans carry the resolved
            // graph on the plan, not a named `graph_ref`).
            let view_trace = TraceRecord {
                id: plan.trace.id.clone(),
                graph_ref: None,
                initial_state: plan.trace.initial_state.clone(),
            };
            let parsed = ParsedGraph {
                graph: plan.graph.clone(),
                graphs: std::collections::BTreeMap::new(),
                traces: vec![view_trace.clone()],
            };
            let duration_us = trace_duration_us(&parsed, &view_trace);
            let t = sampler.sample_t_star(&plan.trace.id, 0, duration_us);
            let rewritten = if is_warmup {
                rewrite_for_warmup(&parsed, t)
            } else {
                chop_trie_at_tstar(&parsed, t)
            };
            GraphTracePlan {
                graph: rewritten.graph,
                trace: plan.trace.clone(),
                arrival_offset_ns: plan.arrival_offset_ns,
            }
        })
        .collect()
}

/// A single-trace [`ParsedGraph`] view over one already-resolved plan graph.
///
/// The view trace clears `graph_ref` so `resolve_trace_graph` returns this
/// graph directly (recorded plans carry the resolved graph on the plan, not a
/// named `graph_ref`) — the same construction [`apply_tstar_split`] and
/// [`sample_plan_tstar`] build inline; factored so the frontier resume shares it.
fn single_trace_parsed(plan: &GraphTracePlan) -> ParsedGraph {
    let view_trace = TraceRecord {
        id: plan.trace.id.clone(),
        graph_ref: None,
        initial_state: plan.trace.initial_state.clone(),
    };
    ParsedGraph {
        graph: plan.graph.clone(),
        graphs: std::collections::BTreeMap::new(),
        traces: vec![view_trace],
    }
}

/// Assemble the WARMUP -> PROFILING [`GraphWarmupHandoff`] from E3a's per-lane
/// ledgers plus the pass-0 lane count and the [`Clock`]-derived drain-end wall.
///
/// Port of `graph_ir_replay.py::_stash_pressure_handoff`
/// (`src/aiperf/timing/strategies/graph_ir_replay.py:2231-2299`, branch
/// `ajc/aiperf-graph-ir`): each live lane's [`LaneHandoff`] is built
/// field-for-field from that lane's recorded identity (`template_trace_id`,
/// `instance_id`, `t_star_us`), executed-node set, and return-wall ledger, and
/// the top-level handoff carries `drain_end_wall_us`, `corpus_cursor`
/// (`self._pressure_next_index`, `:2294`), and `pressure_lane_count`
/// (`:2295`). Python derives `executed_node_ids = frozenset(live_walls)` from
/// the live instance's return-wall keys; the native ledger populates the
/// executed set and the return-wall map together in `observe_return`, so their
/// keys agree by construction. Lanes are keyed by index to match E1's
/// `BTreeMap<u64, LaneHandoff>`; the last identity a lane registered (its live
/// instance at drain) is the one carried, the `_pressure_live[lane]` analogue.
///
/// The pressure-pass-0 boundary-priming wall merge Python performs
/// (`_priming_instance_ids` under the pass-0 anchor set, `:2280-2283`) is
/// subsumed here: the native ledger already keys executed nodes and walls by
/// LANE (not per-instance), so a lane's boundary-priming returns and its
/// pass-0 pressure returns land in one lane-keyed map — no separate merge.
fn build_warmup_handoff(
    lanes: &GraphLaneLedger,
    pressure_lane_count: u64,
    drain_end_wall_us: f64,
) -> GraphWarmupHandoff {
    let mut lane_map: BTreeMap<u64, LaneHandoff> = BTreeMap::new();
    for lane in lanes.registered_lanes() {
        let Some(identity) = lanes.lane_identity(lane) else {
            continue;
        };
        lane_map.insert(
            lane,
            LaneHandoff {
                template_trace_id: identity.template_trace_id,
                instance_id: identity.instance_id,
                t_star_us: identity.t_star_us,
                executed_node_ids: lanes.executed_node_ids(lane),
                return_wall_us: lanes.return_wall_us(lane),
            },
        );
    }
    GraphWarmupHandoff {
        lanes: lane_map,
        drain_end_wall_us,
        corpus_cursor: lanes.corpus_cursor(),
        pressure_lane_count,
    }
}

/// One profiling resume prefix instance plus its per-lane resume identity.
///
/// The `plan` carries the already-cut graph (frontier chop / fresh-start full
/// replay / normal pass-0 t*), and `lane` is the load-bearing lane index used to
/// mint a per-lane-unique instance id at dispatch. Two lanes that resumed the
/// SAME template (rare over-fan-out) are TWO distinct [`LaneResumePlan`]s with
/// different `lane` values and their OWN frontier chops — no union, no collision.
struct LaneResumePlan {
    lane: u64,
    plan: GraphTracePlan,
    /// The exact instance id to stamp on the dispatched trace when this lane
    /// RESUMES a handoff entry whose template was found in the corpus: the live
    /// pressure instance's id (`entry.instance_id`). Reusing it verbatim keeps
    /// the per-instance cache-bust marker (a digest of `trace.id`; see the
    /// continuity contract on [`crate::graph::warmup_handoff::LaneHandoff::instance_id`],
    /// `warmup_handoff.rs:39-44`) continuous across warmup->profiling, so the KV
    /// the pressure stage built at that id transfers instead of cold-prefilling
    /// behind a fresh marker. `None` for fresh-start lanes, normal pass-0 lanes,
    /// and the not-in-corpus resume fallback -- those were never primed at a
    /// pressure id, so they keep the normal per-lane-unique native id and mint a
    /// fresh marker (Python `_run_instance`, `graph_ir_replay.py:1809-1828`:
    /// `instance_id = handoff_entry.instance_id` only when the entry survives the
    /// `trace.id == handoff_entry.template_trace_id` guard, else a fresh nonce).
    resume_instance_id: Option<String>,
}

/// Build the PROFILING phase's PER-LANE resume instances from a warmup handoff.
///
/// Faithful port of `graph_ir_replay.py::_run_lanes`'s profiling loop
/// (`src/aiperf/timing/strategies/graph_ir_replay.py:1276-1375`, branch
/// `ajc/aiperf-graph-ir`): one instance per lane in `0..lanes`, each carrying its
/// OWN resume-chopped / fresh-start / pass-0 plan at its lane t* — NOT one plan
/// per corpus template. Same-template lanes are therefore two SEPARATE instances
/// keyed by lane (no union, no last-write-wins) and an empty pressure lane
/// fresh-starts at `t*=0` from the shared cursor, exactly as Python does.
///
/// Returns the ordered prefix instances (in lane order) plus the number of
/// fresh-start draws consumed from the cursor, so [`rebuild_resume_workload`] can
/// advance the following corpus recycle past the fresh-start templates (Python's
/// shared `next_index` threading fresh-starts and recycles through one cursor).
///
/// Per-lane semantics (`graph_ir_replay.py:1276-1375`):
/// - `lanes = len(pass0_traces)`, bumped to `max(lanes, pressure_lane_count)` so
///   every drained pressure lane is honored (`:1295`).
/// - `next_index = handoff.corpus_cursor` when `recycle_is_bounded` else the
///   pass-0 cursor (`:1297-1300`) — DF3 keeps the bounded-cursor part consistent.
/// - base `trace = pass0_traces[lane]` if `lane < len(pass0_traces)` else a
///   wrap-around fallback `traces[draw_index(lane, len)]` (`:1330-1334`).
/// - `entry = handoff.lanes.get(lane)`: present AND its `template_trace_id` in the
///   corpus → RESUME that template via [`chop_trie_at_frontier`] with THAT lane's
///   `executed_node_ids` / `return_wall_us` / `t_star_us` (`:1336-1350`).
/// - elif `recycle_is_bounded` AND `lane < pressure_lane_count` → FRESH-START:
///   draw `traces[draw_index(next_index, len)]`, `next_index += 1`, full `t*=0`
///   replay (`:1352-1364`).
/// - else → normal pass-0 assignment at the lane's t* (`chop_trie_at_tstar`, here
///   via [`apply_tstar_split`]'s lane-0 window — the current native t* fidelity).
///
/// A handoff template absent from the corpus falls back to the lane's pass-0 t*
/// plan (Python `resumed is None` warning path, `:1345-1349`). The residual
/// re-root delays inside the frontier chop are clamped at
/// [`HANDOFF_RESIDUAL_CAP_SEC`].
fn build_profiling_resume_lane_plans(
    original_plans: &[GraphTracePlan],
    phase: &PhaseSpec,
    t_star: TStarWindow,
    handoff: &GraphWarmupHandoff,
    session_limit: Option<u64>,
    recycle_bounded: bool,
) -> (Vec<LaneResumePlan>, u64) {
    let residual_cap_us = Some(HANDOFF_RESIDUAL_CAP_SEC * MICROS_PER_SECOND);
    let n = original_plans.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Normal pass-0 profiling plans (the byte-unchanged no-handoff assignment):
    // lane `i`'s "else" branch and its wrap-around fallback both draw from here,
    // and the corpus recycle after the prefix runs FULL `t*=0` replay from
    // `original_plans` directly (Python recycles run fresh t*=0 templates).
    let split = apply_tstar_split(original_plans, phase, t_star);

    // Strategy-aware corpus-index draw shared by pass-0 resolve, the index-safe
    // fallback, and the fresh-start recycle draw below (`_draw_index`).
    let draw = PressureDraw::from_window(t_star);

    // Per-lane frontier-chop cache shared across the pass-0 spawnability scan AND
    // the per-lane assignment loop (the profiling `_lane_plans`,
    // `graph_ir_replay.py:_plan_for_lane`): a lane's plan re-chopped at its salted
    // `t*` is computed once per `(template_index, lane)`, so judging spawnability
    // at a lane's `t*` and later dispatching that lane share ONE chop.
    let chop_cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());

    // Lane count: `_resolve_lane_count` (phase concurrency clamped) resolves the
    // pass-0 lanes, then `max(len(pass0), pressure_lane_count)` honors every
    // drained pressure lane (`graph_ir_replay.py:1287-1295`).
    let concurrency = phase.concurrency().unwrap_or(n);
    let lane_target = pressure_resolve_lane_count(concurrency, n, session_limit, recycle_bounded);
    let (pass0, pass0_cursor) = profiling_resolve_pass0_lanes(
        original_plans,
        &split,
        lane_target,
        &draw,
        t_star,
        &chop_cache,
    );
    let lanes = pass0
        .len()
        .max(usize::try_from(handoff.pressure_lane_count).unwrap_or(usize::MAX));

    // Shared fresh-start / recycle draw cursor (Python `next_index`): the bounded
    // resume continues the wrap from where the pressure warmup drained; the
    // unbounded single-pass keeps its own pass-0 cursor.
    let mut next_index = if recycle_bounded {
        handoff.corpus_cursor
    } else {
        pass0_cursor
    };
    let fresh_start_base = next_index;

    let mut lane_plans = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let lane_u64 = u64::try_from(lane).unwrap_or(u64::MAX);
        // Index-safe pass-0 base template: a handoff can bump `lanes` past
        // `len(pass0)`, so lanes beyond the resolved set take a wrap-around
        // fallback (Python `traces[self._draw_index(lane, len(traces))]`, `:1332`).
        let base_index = pass0
            .get(lane)
            .copied()
            .unwrap_or_else(|| draw.index(lane_u64, n));

        // `resume_instance_id` is set ONLY when the lane resumes a handoff entry
        // whose template resolved in the corpus (the frontier-chop branch): the
        // dispatched trace then carries the pressure instance's exact id so the
        // cache-bust marker stays continuous (Python `_run_instance` reuses
        // `handoff_entry.instance_id` only after the `trace.id ==
        // handoff_entry.template_trace_id` guard survives, `:1809-1828`).
        let (plan, resume_instance_id) = if let Some(entry) = handoff.lanes.get(&lane_u64) {
            // RESUME entry: frontier-chop the resumed template with THIS lane's
            // executed set / return walls / t*. Absent from the corpus → the
            // lane's normal pass-0 assignment (Python `resumed is None`).
            match original_plans
                .iter()
                .find(|plan| plan.trace.id == entry.template_trace_id)
            {
                Some(original) => {
                    let parsed = single_trace_parsed(original);
                    let executed: HashSet<String> =
                        entry.executed_node_ids.iter().cloned().collect();
                    let return_wall_us: HashMap<String, f64> = entry
                        .return_wall_us
                        .iter()
                        .map(|(k, v)| (k.clone(), *v))
                        .collect();
                    let rewritten = chop_trie_at_frontier(
                        &parsed,
                        entry.t_star_us,
                        &executed,
                        &return_wall_us,
                        handoff.drain_end_wall_us,
                        residual_cap_us,
                    );
                    let plan = GraphTracePlan {
                        graph: rewritten.graph,
                        trace: original.trace.clone(),
                        arrival_offset_ns: original.arrival_offset_ns,
                    };
                    (plan, Some(entry.instance_id.clone()))
                }
                // Handoff template not in this corpus: fall back to the lane's
                // normal pass-0 plan (chopped at THIS lane's salted `t*`, DF6) and
                // its normal id (Python `_run_instance` drops the entry via the
                // `trace.id != handoff_entry.template_trace_id` guard, then plans
                // `_plan_for_lane(trace, lane)` — the lane-salted pass-0 plan, not
                // lane 0's; no marker continuity to preserve).
                None => (
                    profiling_plan_for_lane(
                        original_plans,
                        &split,
                        base_index,
                        lane_u64,
                        t_star,
                        &chop_cache,
                    ),
                    None,
                ),
            }
        } else if recycle_bounded && lane_u64 < handoff.pressure_lane_count {
            // Empty pressure lane completed at drain: fresh-start it from the
            // shared cursor at full `t*=0` replay (Python `:1352-1364`) instead
            // of re-running a t* resume the pressure stage already warmed. It was
            // never primed, so it keeps its normal id / fresh marker.
            let draw_index = draw.index(next_index, n);
            next_index = next_index.saturating_add(1);
            (original_plans[draw_index].clone(), None)
        } else {
            // Normal pass-0 assignment at THIS lane's salted `t*` (DF6): each
            // pass-0 profiling lane chops the ORIGINAL graph at its own lane `t*`
            // (Python `_run_instance` -> `_plan_for_lane(trace, lane_index)` for a
            // pass-0, no-handoff, non-fresh-start lane, `graph_ir_replay.py:1837`),
            // not lane 0's split — consistent with the warmup/handoff `t*` for
            // this lane. Normal id.
            (
                profiling_plan_for_lane(
                    original_plans,
                    &split,
                    base_index,
                    lane_u64,
                    t_star,
                    &chop_cache,
                ),
                None,
            )
        };
        lane_plans.push(LaneResumePlan {
            lane: lane_u64,
            plan,
            resume_instance_id,
        });
    }

    (lane_plans, next_index - fresh_start_base)
}

/// Resolve the profiling pass-0 lane -> corpus-index map and the resume cursor.
///
/// The [`GraphTracePlan`] analogue of [`pressure_resolve_pass0_lanes`]: walk the
/// corpus in draw order (`graph_ir_replay.py:_resolve_pass0_lanes`, `:1577`)
/// assigning lane `i` to the next SPAWNABLE template (a plan whose lane-`i`
/// frontier chop has at least one dispatchable node), skipping empties, and return
/// the per-lane corpus indices (length `<= lanes`) plus the cursor one past the
/// last consumed position. Python judges spawnability at the candidate's TARGET
/// RANK's lane-salted `t*` (`_is_spawnable(trace, rank)`); per-lane salting is
/// built (DF1b/DF6: [`profiling_plan_for_lane`]), so the check re-chops each
/// candidate at the lane-`rank` `t*` it will DISPATCH at rather than lane 0's
/// split — a template non-empty at lane 0 but empty at a higher lane is not
/// consumed there. The chop is cached by `(template_index, rank)` (shared with the
/// dispatch loop) so the scan and dispatch re-chop once. At the default `[0, 0]`
/// window every lane collapses to `t*=0` and reuses `split[idx]`, so lane `i`
/// takes corpus index `i` sequentially byte-for-byte with the prior assignment.
fn profiling_resolve_pass0_lanes(
    original_plans: &[GraphTracePlan],
    split: &[GraphTracePlan],
    lanes: usize,
    draw: &PressureDraw,
    t_star: TStarWindow,
    cache: &RefCell<HashMap<(usize, u64), GraphTracePlan>>,
) -> (Vec<usize>, u64) {
    let n = split.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let mut pass0 = Vec::new();
    let mut cursor: u64 = 0;
    let max_cursor = lanes as u64 + n as u64;
    while pass0.len() < lanes && cursor < max_cursor {
        let idx = draw.index(cursor, n);
        cursor = cursor.saturating_add(1);
        // Judge spawnability at the candidate's TARGET rank (== the lane it would
        // dispatch on), the same `t*` the lane dispatches at (Python
        // `_is_spawnable(trace, rank)` with `rank = len(pass0)`).
        let rank = u64::try_from(pass0.len()).unwrap_or(u64::MAX);
        let plan = profiling_plan_for_lane(original_plans, split, idx, rank, t_star, cache);
        if !plan.graph.nodes.is_empty() {
            pass0.push(idx);
        }
    }
    (pass0, cursor)
}

/// PROFILING trace source: dispatch the per-lane resume PREFIX first (one
/// instance per drained/pressure lane, in lane order), then delegate to the
/// cursor-continued corpus recycle.
///
/// The faithful native shape of `graph_ir_replay.py::_run_lanes` (`:1276-1375`):
/// the prefix is the "every drained lane dispatched at the handoff" set (resume /
/// fresh-start / pass-0 instances, NOT gated by the session cap — they are
/// continuations), and the delegate is the bounded recycle that Python's shared
/// `next_index` cursor drives after them.
///
/// Instance-id stamping (DF4b, the KV-continuity mechanism):
/// - A lane RESUMING a handoff entry (template found in corpus) carries the
///   pressure instance's EXACT id (`entry.instance_id`, in
///   [`LaneResumePlan::resume_instance_id`]). The per-instance cache-bust marker
///   is a digest of `trace.id`, so reusing the id keeps the marker continuous
///   and the KV the pressure stage built transfers instead of cold-prefilling
///   behind a fresh marker (contract:
///   [`crate::graph::warmup_handoff::LaneHandoff::instance_id`],
///   `warmup_handoff.rs:39-44`; Python `graph_ir_replay.py:1809-1820`).
/// - Fresh-start lanes, normal pass-0 lanes, and the not-in-corpus resume
///   fallback were NOT primed at a pressure id, so they keep the per-lane-unique
///   `{template}::resume-lane-{lane}` id (two same-template lanes stay distinct)
///   and mint a fresh marker.
///
/// The recycle keeps its own `::instance-{n}` identities from the shared
/// [`GraphTraceInstanceSequence`], a disjoint id space. Pressure instance ids are
/// distinct per lane (each lane minted its own `t-N#L.pK`), so reusing them
/// cannot collide two prefix instances.
struct LaneResumeGraphTraceSource {
    prefix: Vec<LaneResumePlan>,
    next_prefix: Cell<usize>,
    recycle: Rc<dyn GraphTraceSource>,
}

impl GraphTraceSource for LaneResumeGraphTraceSource {
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError> {
        let idx = self.next_prefix.get();
        if idx < self.prefix.len() {
            self.next_prefix.set(idx + 1);
            let entry = &self.prefix[idx];
            let mut plan = entry.plan.clone();
            // Resume lanes reuse the pressure instance's id verbatim for marker
            // continuity; every other lane keeps the per-lane-unique native id.
            plan.trace.id = match &entry.resume_instance_id {
                Some(instance_id) => instance_id.clone(),
                None => format!("{}::resume-lane-{}", plan.trace.id, entry.lane),
            };
            return Ok(Some(plan));
        }
        self.recycle.next_trace()
    }
}

/// Build the cell-aware graph trace source for one phase's plans.
///
/// Factored from [`prepare_graph_phase`] so the profiling phase can rebuild its
/// source at [`create`](PhaseExecutionFactory::create) time (after warmup) with
/// frontier-resumed plans while sharing the identical cell-partition selection
/// and the run-scoped [`GraphTraceInstanceSequence`] — instance ordinals are
/// minted at `next_trace` (dispatch) time, so a source built later still draws
/// the same run-unique ids.
fn build_graph_trace_source(
    plans: Vec<GraphTracePlan>,
    session_limit: Option<u64>,
    request_limit: Option<u64>,
    trace_instances: GraphTraceInstanceSequence,
    start_ordinal: u64,
    t_star: TStarWindow,
) -> Result<Rc<dyn GraphTraceSource>> {
    // `--dataset-sampling-strategy` governs WHICH template a freed recycle lane
    // serves via the SAME resolved draw the pressure stage uses (both route
    // through the shared `PermutationDraw` the window resolves). `Sequential` (the
    // default) keeps the byte-unchanged modulo pick; `Shuffle`/`Random` derive a
    // child generator off the RUN root (`run_random_seed`, not
    // `t_star_random_seed`), so a freed profiling lane continues the SAME order
    // the pressure stage replayed under.
    Ok(match ModuloCellPartition::from_env() {
        Some(partition) if partition.cell_count() > 1 && request_limit.is_none() => {
            tracing::debug!(
                cell_id = partition.cell_id(),
                cell_count = partition.cell_count(),
                "graph phase using partitioned trace source for cell"
            );
            // The bounded profiling recycle resume (`start_ordinal`) is a
            // single-cell Cycling concern: the partitioned source only serves the
            // request-unbounded cellular case (`request_limit.is_none()`), where
            // Python's `recycle_is_bounded` corpus-cursor continuation does not
            // engage, so a nonzero cursor cannot reach this arm on the product path.
            Rc::new(
                PartitionedGraphTraceSource::new(
                    plans,
                    session_limit,
                    partition.cell_id(),
                    partition.cell_count(),
                )?
                .with_sampling(t_star.recycle_draw()),
            )
        }
        _ => Rc::new(
            CyclingGraphTraceSource::with_budgets_and_sequence(
                plans,
                session_limit,
                request_limit,
                trace_instances,
            )?
            .starting_at(start_ordinal)
            .with_sampling(t_star.recycle_draw()),
        ),
    })
}

/// Assemble one graph phase's [`GraphWorkload`] from its resolved parts.
///
/// Factored from [`prepare_graph_phase`] so the deferred profiling resume build
/// shares the exact arrival / run-failure / admission wiring.
fn assemble_graph_workload(
    clock: Rc<dyn Clock>,
    source: Rc<dyn GraphTraceSource>,
    placement: Rc<dyn TracePlacement>,
    arrival: Rc<dyn GraphArrivalPolicy>,
    run_failure: Rc<dyn RunFailurePolicy>,
    uses_admission: bool,
    session_slots: Option<Rc<SlotPool>>,
) -> Result<GraphWorkload> {
    let mut workload = GraphWorkload::new(clock, source, placement)
        .with_arrival(arrival)
        .with_run_failure(run_failure);
    if uses_admission {
        workload = workload.with_admission(Rc::new(SlotPoolTraceAdmission::new(
            session_slots
                .ok_or_else(|| anyhow!("graph phase requires shared session admission"))?,
        )));
    }
    Ok(workload)
}

/// Deferred profiling-phase resume ingredients, carried on the profiling
/// [`PreparedGraphPhase`] so its trace source can be REBUILT with
/// frontier-resumed plans at [`create`](PhaseExecutionFactory::create) time when
/// a warmup handoff is present.
///
/// Present only for non-warmup phases; absent a stashed handoff the phase uses
/// its already-prepared (`apply_tstar_split`) workload verbatim — the
/// no-pressure path is byte-unchanged. Everything here is a cheap `Rc` clone of
/// a value [`prepare_graph_phase`] already built, except `original_plans` (the
/// pre-`t*` corpus the frontier chop re-cuts, cloned once per non-warmup phase).
struct ProfilingResume {
    original_plans: Rc<Vec<GraphTracePlan>>,
    phase: PhaseSpec,
    t_star: TStarWindow,
    session_limit: Option<u64>,
    request_limit: Option<u64>,
    trace_instances: GraphTraceInstanceSequence,
    arrival: Rc<dyn GraphArrivalPolicy>,
    run_failure: Rc<dyn RunFailurePolicy>,
    uses_admission: bool,
    session_slots: Option<Rc<SlotPool>>,
    clock: Rc<dyn Clock>,
}

/// Rebuild the profiling workload with the warmup handoff's PER-LANE resume
/// instances (consume-once path). Dispatches one prefix instance per drained /
/// pressure lane ([`build_profiling_resume_lane_plans`]), then delegates to the
/// cursor-continued corpus recycle — the faithful native shape of
/// `graph_ir_replay.py::_run_lanes` (`:1276-1375`). Shares
/// [`build_graph_trace_source`] / [`assemble_graph_workload`] with the up-front
/// prepare so the only difference from the no-pressure path is the per-lane
/// prefix and the resumed cursor.
fn rebuild_resume_workload(
    resume: &ProfilingResume,
    placement: Rc<dyn TracePlacement>,
    handoff: &GraphWarmupHandoff,
) -> Result<GraphWorkload> {
    // Gated on a stop condition (`session_limit`/`request_limit` present) exactly
    // as Python's `if self._warmup_handoff is not None and recycle_is_bounded`
    // branch (`graph_ir_replay.py:1297-1300`).
    let recycle_bounded = resume.session_limit.is_some() || resume.request_limit.is_some();

    // Per-lane resume PREFIX (resume / fresh-start / pass-0 instances, one per
    // drained/pressure lane) plus the number of cursor positions the fresh-start
    // draws consumed, so the following corpus recycle continues past them.
    let (prefix, fresh_start_draws) = build_profiling_resume_lane_plans(
        &resume.original_plans,
        &resume.phase,
        resume.t_star,
        handoff,
        resume.session_limit,
        recycle_bounded,
    );

    // Continue the bounded recycle from where the pressure warmup drained AND the
    // fresh-start prefix advanced the shared cursor (Python threads fresh-starts
    // and recycles through one `next_index`, `:1352-1373`); the unbounded
    // single-pass profiling keeps its own cursor (`0`) so a carried cursor cannot
    // hole the cover-the-corpus-once contract. The session budget stays anchored
    // to the raw recycle ordinal inside the source, so the cursor governs only
    // template selection. The recycle runs FULL `t*=0` replay over the original
    // corpus (Python recycles run fresh t*=0 templates).
    let start_ordinal = if recycle_bounded {
        handoff
            .corpus_cursor
            .checked_add(fresh_start_draws)
            .ok_or_else(|| anyhow!("graph resume recycle cursor exceeds u64"))?
    } else {
        0
    };
    let recycle = build_graph_trace_source(
        resume.original_plans.as_ref().clone(),
        resume.session_limit,
        resume.request_limit,
        resume.trace_instances.clone(),
        start_ordinal,
        resume.t_star,
    )?;
    let source: Rc<dyn GraphTraceSource> = Rc::new(LaneResumeGraphTraceSource {
        prefix,
        next_prefix: Cell::new(0),
        recycle,
    });
    assemble_graph_workload(
        resume.clock.clone(),
        source,
        placement,
        resume.arrival.clone(),
        resume.run_failure.clone(),
        resume.uses_admission,
        resume.session_slots.clone(),
    )
}

fn graph_phase_uses_session_admission(phase: &PhaseSpec) -> bool {
    phase.concurrency().is_some()
        || phase
            .common()
            .adaptive_scale
            .as_ref()
            .is_some_and(|adaptive| {
                matches!(
                    adaptive.control_variable,
                    AdaptiveControlVariableSpec::Concurrency
                )
            })
}

#[allow(clippy::too_many_arguments)]
fn graph_ramp_controller(
    spec: &PhaseSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    placement: Rc<dyn TracePlacement>,
    rng_root: RngRoot,
    failures: Rc<GraphPhaseFailures>,
) -> Result<Rc<dyn ScheduledPhaseController>> {
    let common = spec.common();
    let rng_roots = RampActuatorRngRoots::from_phase_root(rng_root);
    let target_rate = spec
        .request_arrival()
        .and_then(|(_, target_rate, _)| target_rate);
    let mut drivers = Vec::new();
    if let Some(ramp) = &common.concurrency_ramp {
        let target = spec
            .concurrency()
            .ok_or_else(|| anyhow!("concurrency_ramp requires a concurrency target"))?;
        let slots = session_slots
            .clone()
            .ok_or_else(|| anyhow!("concurrency_ramp requires graph session admission"))?;
        let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_roots.concurrency())?;
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            slots.set_limit(value.round() as usize)
        }));
    }
    if let Some(ramp) = &common.prefill_ramp {
        let target = common
            .prefill_concurrency
            .ok_or_else(|| anyhow!("prefill_ramp requires prefill_concurrency"))?;
        let strategy = ramp_strategy(
            ramp,
            1.0,
            target as f64,
            false,
            rng_roots.prefill_concurrency(),
        )?;
        let placement = placement.clone();
        let failures = failures.clone();
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            if let Err(error) = placement.set_prefill_limit(value.round() as usize) {
                failures.record(format!("applying graph prefill ramp: {error}"));
            }
        }));
    }
    if let Some(ramp) = &common.rate_ramp {
        let target = target_rate.ok_or_else(|| anyhow!("rate_ramp requires a rate phase"))?;
        let duration_ns = seconds_to_u64_ns(ramp.duration)?;
        let start = target * RATE_RAMP_UPDATE_INTERVAL_NS as f64 / duration_ns as f64;
        let strategy = ramp_strategy(ramp, start, target, true, rng_roots.request_rate())?;
        drivers.push(RampDriver::new(clock, strategy, move |value| {
            intervals.borrow_mut().set_rate(value)
        }));
    }
    if drivers.is_empty() {
        Ok(Rc::new(crate::phase_runtime::NoopScheduledPhaseController))
    } else {
        Ok(Rc::new(RampScheduledPhaseController::new(drivers)))
    }
}

fn graph_adaptive_config(
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
) -> Result<Option<AdaptiveRunConfig>> {
    let Some(config) = adaptive_run_config(phase, benchmark_id, artifact_dir)? else {
        return Ok(None);
    };
    match config.control_variable {
        AdaptiveControlVariable::Concurrency => {
            ensure!(
                phase.common().concurrency_ramp.is_none(),
                "adaptive graph concurrency cannot be combined with concurrency_ramp"
            );
        }
        AdaptiveControlVariable::PrefillConcurrency => {
            ensure!(
                phase.common().prefill_ramp.is_none(),
                "adaptive graph prefill_concurrency cannot be combined with prefill_ramp"
            );
            let session_target = phase.concurrency().ok_or_else(|| {
                anyhow!("adaptive graph prefill_concurrency requires a session concurrency cap")
            })?;
            ensure!(
                config.maximum <= session_target as f64,
                "adaptive graph prefill_concurrency maximum must be <= concurrency"
            );
        }
        AdaptiveControlVariable::RequestRate => {
            ensure!(
                phase.request_arrival().is_some(),
                "adaptive graph request_rate requires a rate-controlled phase"
            );
            ensure!(
                phase.common().rate_ramp.is_none(),
                "adaptive graph request_rate cannot be combined with rate_ramp"
            );
        }
        AdaptiveControlVariable::Users => {
            bail!("adaptive users is not defined for Graph-IR phases")
        }
    }
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use crate::adaptive_core::{SharedWindowSampler, TumblingWindowSampler};
    use crate::dataset::SegmentPool;
    use crate::engine::graph_input::GraphSamplingStrategy;
    use crate::graph::errors::TraceError;
    use crate::graph::model::{GraphRecord, GraphTracePlan, TraceRecord};
    use crate::graph::tstar::{legacy_random_seed, legacy_shuffle_seed};
    use crate::graph::workload::{GraphWorkloadReport, TraceAdmissionInfo};
    use crate::timing::{PhaseReturn, PhaseSend};
    use uuid::Uuid;

    use super::*;
    use crate::engine::records::CapturedModelOutput;

    fn wrap_policy_input(root_count: usize) -> GraphInputBundle {
        let plans = (0..root_count)
            .map(|index| GraphTracePlan {
                graph: GraphRecord::default(),
                trace: TraceRecord {
                    id: format!("root-{index}"),
                    graph_ref: None,
                    initial_state: Default::default(),
                },
                arrival_offset_ns: None,
            })
            .collect();
        GraphInputBundle {
            plans,
            segments: Arc::new(SegmentPool::new().freeze()),
            metadata: crate::graph::input::GraphInputMetadata {
                format: "weka_trace".into(),
                root_count,
                node_count: 0,
            },
        }
    }

    fn concurrency_phase(concurrency: usize, sessions: Option<u64>) -> PhaseSpec {
        let mut value = serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "concurrency": concurrency,
        });
        if let Some(sessions) = sessions {
            value["sessions"] = serde_json::json!(sessions);
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn graph_pressure_grace_matches_python_derived_values() {
        // _graph_pressure_grace_sec: explicit user_grace honored verbatim.
        assert_eq!(graph_pressure_grace_sec(Some(30.0), 45.0), 30.0);
        assert_eq!(graph_pressure_grace_sec(Some(45.0), 30.0), 45.0);
        // Explicit grace of 0.0 is still honored (not treated as absent).
        assert_eq!(graph_pressure_grace_sec(Some(0.0), 100.0), 0.0);
        // None -> min(cache_pressure, cap); below cap returns the duration.
        assert_eq!(graph_pressure_grace_sec(None, 10.0), 10.0);
        // At the cap boundary (300.0) returns the cap exactly.
        assert_eq!(
            graph_pressure_grace_sec(None, PRESSURE_DRAIN_GRACE_CAP_SEC),
            300.0
        );
        // Above the cap clamps to the cap.
        assert_eq!(graph_pressure_grace_sec(None, 10_000.0), 300.0);
    }

    fn cache_pressure_warmup(seamless_profiling: bool) -> Vec<PhaseSpec> {
        let warmup: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "warmup",
            "exclude_from_results": true,
            "concurrency": 2,
            "agentic_cache_warmup_duration": 12.5,
        }))
        .unwrap();
        let profiling: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "concurrency": 2,
            "seamless": seamless_profiling,
        }))
        .unwrap();
        vec![warmup, profiling]
    }

    #[test]
    fn cache_pressure_warmup_seamless_into_profiling_is_rejected() {
        let error = validate_graph_phases(&cache_pressure_warmup(true)).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("cache-pressure warmup") && message.contains("seamless"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn cache_pressure_warmup_non_seamless_profiling_is_accepted() {
        validate_graph_phases(&cache_pressure_warmup(false)).unwrap();
    }

    #[test]
    fn recorded_graph_wrap_policy_rejects_unintentional_lane_cloning() {
        let input = wrap_policy_input(2);
        let phases = [concurrency_phase(3, Some(3))];
        let error = validate_dataset_wrap_policy(&phases, &input, false).unwrap_err();
        assert!(format!("{error:#}").contains("dataset wrapping is disabled"));
        validate_dataset_wrap_policy(&phases, &input, true).unwrap();
    }

    #[test]
    fn recorded_graph_wrap_policy_allows_bounded_or_one_pass_corpora() {
        let input = wrap_policy_input(2);
        validate_dataset_wrap_policy(&[concurrency_phase(3, Some(2))], &input, false).unwrap();
        validate_dataset_wrap_policy(&[concurrency_phase(3, None)], &input, false).unwrap();
    }

    fn named_concurrency_phase(name: &str) -> PhaseSpec {
        serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": name,
            "exclude_from_results": false,
            "concurrency": 1,
        }))
        .unwrap()
    }

    /// One linear-chain recorded plan `n_0 -> n_1 -> n_2` at arrivals 0/1e6/2e6
    /// us, so the `n` chain straddles a mid-chain `t*` and the split is
    /// observable in the surviving node ids.
    fn tstar_chain_plan() -> Vec<GraphTracePlan> {
        use crate::graph::model::{ChannelRequirement, LlmNode, StaticEdge};
        use serde_json::json;
        use std::collections::BTreeMap;

        let node = |arrival: u64, inputs: &[&str]| {
            let mut metadata = BTreeMap::new();
            metadata.insert("arrival_offset_us".to_owned(), json!(arrival));
            // All `n_*` turns belong to one recorded session chain. Real
            // recorded-graph nodes (weka/dynamo/aiperf_trace) always carry
            // `metadata["conversation_id"]` (graph/recorded/trie/mod.rs:170),
            // which `warmup_boundary_nodes` groups chains by; mirror that here
            // so the `n` chain straddles t* rather than fragmenting into
            // per-full-id singletons.
            metadata.insert("conversation_id".to_owned(), json!("n"));
            LlmNode {
                output: "out".to_owned(),
                streaming: true,
                inputs: inputs
                    .iter()
                    .map(|c| ChannelRequirement {
                        channel: (*c).to_owned(),
                        count: Default::default(),
                    })
                    .collect(),
                min_start_delay_us: None,
                max_tokens: None,
                items: Vec::new(),
                metadata,
            }
        };
        let edge = |source: &str, target: &str| StaticEdge {
            source: source.to_owned(),
            target: target.to_owned(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        };
        let mut nodes = BTreeMap::new();
        nodes.insert("n_0".to_owned(), node(0, &[]));
        nodes.insert("n_1".to_owned(), node(1_000_000, &["n_0_out"]));
        nodes.insert("n_2".to_owned(), node(2_000_000, &["n_1_out"]));
        let graph = GraphRecord {
            nodes,
            edges: vec![
                edge("START", "n_0"),
                edge("n_0", "n_1"),
                edge("n_1", "n_2"),
                edge("n_2", "END"),
            ],
            ..Default::default()
        };
        vec![GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: "t".to_owned(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        }]
    }

    fn node_ids(plans: &[GraphTracePlan]) -> Vec<String> {
        plans[0].graph.nodes.keys().cloned().collect()
    }

    fn plan_node_ids(plan: &GraphTracePlan) -> Vec<String> {
        plan.graph.nodes.keys().cloned().collect()
    }

    #[test]
    fn tstar_split_profiling_keeps_only_post_tstar_frontier() {
        // Collapsed window min==max==0.5 draws no RNG: t* = 0.5 * dur(2e6) = 1e6.
        // Survivors are the nodes arriving at/after t*: n_1, n_2 (n_0 dropped).
        let window = TStarWindow {
            start_min_ratio: 0.5,
            start_max_ratio: 0.5,
            random_seed: 0,
            ..Default::default()
        };
        let split = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
        );
        assert_eq!(node_ids(&split), vec!["n_1".to_owned(), "n_2".to_owned()]);
    }

    #[test]
    fn tstar_split_warmup_primes_boundary_turns() {
        // Same t* = 1e6. The `n` chain's last pre-t* turn is n_0; it primes the
        // chain prefix while n_1/n_2 (post-t*) are profiled, not warmed.
        let window = TStarWindow {
            start_min_ratio: 0.5,
            start_max_ratio: 0.5,
            random_seed: 0,
            ..Default::default()
        };
        let split = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("warmup"),
            window,
        );
        assert_eq!(node_ids(&split), vec!["n_0".to_owned()]);
        // Boundary priming node is flattened: fan-in cleared, re-rooted at START.
        assert!(split[0].graph.nodes["n_0"].inputs.is_empty());
        assert_eq!(split[0].graph.edges.len(), 1);
        assert_eq!(split[0].graph.edges[0].source, "START");
        assert_eq!(split[0].graph.edges[0].target, "n_0");
    }

    #[test]
    fn tstar_default_window_is_unchanged_full_replay() {
        // Regression guard (fixA): the default [0, 0] window is a TRUE no-op, so
        // BOTH profiling and warmup return the full graph unchanged. Emptying the
        // warmup graph here (via rewrite_for_warmup at t* = 0) regressed plain
        // dag_jsonl runs with "contains no dispatchable nodes".
        let window = TStarWindow::default();
        let original = tstar_chain_plan();
        let profiling = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
        );
        assert_eq!(
            node_ids(&profiling),
            vec!["n_0".to_owned(), "n_1".to_owned(), "n_2".to_owned()]
        );
        // Edges carried through verbatim (default window returns plans as-is).
        assert_eq!(
            profiling[0].graph.edges.len(),
            original[0].graph.edges.len()
        );

        // The warmup phase must ALSO be unchanged at the default window — a plain
        // (no-scenario) warmup runs the full graph, not an empty priming rewrite.
        let warmup = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("warmup"),
            window,
        );
        assert_eq!(
            node_ids(&warmup),
            vec!["n_0".to_owned(), "n_1".to_owned(), "n_2".to_owned()]
        );
        assert_eq!(warmup[0].graph.edges.len(), original[0].graph.edges.len());
    }

    #[derive(Default)]
    struct RecordingGraphPhaseProgressSink {
        sent: RefCell<Vec<PhaseSend>>,
        returned: RefCell<Vec<PhaseReturn>>,
        first_tokens: Cell<u64>,
        all_sent: Cell<bool>,
    }

    impl GraphPhaseProgressSink for RecordingGraphPhaseProgressSink {
        fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String> {
            self.sent.borrow_mut().extend_from_slice(sent);
            Ok(())
        }

        fn record_first_token(&self) {
            self.first_tokens
                .set(self.first_tokens.get().saturating_add(1));
        }

        fn record_returned(&self, returned: PhaseReturn) {
            self.returned.borrow_mut().push(returned);
        }

        fn mark_all_sent(&self) {
            self.all_sent.set(true);
        }
    }

    fn progress(
        sink: Rc<RecordingGraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
    ) -> (GraphPhaseProgress, Rc<RefCell<GraphWorkloadReport>>) {
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        (
            GraphPhaseProgress::new(
                sink,
                failures,
                outcome.clone(),
                Rc::new(crate::clock::SimClock::new()),
                Rc::new(GraphLaneLedger::default()),
                false,
                Rc::new(RefCell::new(Vec::new())),
            ),
            outcome,
        )
    }

    /// Build a WARMUP-phase progress plus the shared warmup-failure ledger it
    /// records into, so a test can assert which trace ids the pre-profiling
    /// abort gate would report.
    fn warmup_progress(
        sink: Rc<RecordingGraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
    ) -> (GraphPhaseProgress, Rc<RefCell<Vec<String>>>) {
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let ledger = Rc::new(RefCell::new(Vec::new()));
        (
            GraphPhaseProgress::new(
                sink,
                failures,
                outcome,
                Rc::new(crate::clock::SimClock::new()),
                Rc::new(GraphLaneLedger::default()),
                true,
                ledger.clone(),
            ),
            ledger,
        )
    }

    fn graph_phase_record(trace_id: &str, errored: bool, canceled: bool) -> CapturedRecord {
        let mut ingest =
            crate::metrics_core::RecordIngest::minimal(0, 1, crate::metrics_core::Phase::Profiling);
        ingest.errored = errored;
        ingest.canceled = canceled;
        CapturedRecord {
            uuid: Uuid::nil(),
            x_correlation_id: trace_id.into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        }
    }

    #[test]
    fn graph_phase_progress_preserves_completed_nodes_before_trace_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-cancelled".into(),
            node_count: 3,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("trace-cancelled", false, false));
        progress.complete(
            "trace-cancelled",
            3,
            true,
            &Err(TraceError::Cancelled("phase grace expired".into())),
        );

        assert_eq!(
            *sink.sent.borrow(),
            vec![
                PhaseSend::single_turn_session(),
                PhaseSend::dag_child(),
                PhaseSend::dag_child(),
            ]
        );
        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert_eq!(outcome.borrow().admitted, 1);
        assert_eq!(outcome.borrow().cancelled, 1);
        assert!(failures.first().is_none());
    }

    #[test]
    fn graph_phase_progress_never_masks_backend_failure_as_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-failed".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("trace-failed", false, false));
        progress.complete(
            "trace-failed",
            2,
            true,
            &Err(TraceError::Other("backend failed".into())),
        );

        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    errored: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert_eq!(outcome.borrow().failed, 1);
        assert_eq!(
            failures.first().as_deref(),
            Some("graph trace \"trace-failed\" failed: backend failed")
        );
    }

    #[test]
    fn graph_first_token_reaches_progress_before_adaptive_terminal_record() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-adaptive".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let mut record = graph_phase_record("trace-adaptive", false, false);
        record.ingest.first_token_ns = Some(5);
        record.ingest.token_arrival_ns.push(5);
        record.ingest.usage.completion_tokens = Some(1);
        let sampler: SharedWindowSampler =
            Rc::new(RefCell::new(Box::new(TumblingWindowSampler::new(0))));
        let captured = Rc::new(RefCell::new(Vec::new()));

        ingest_graph_execution_event(
            &captured,
            Some(&sampler),
            &progress,
            GraphExecutionEvent::FirstToken {
                trace_id: "trace-adaptive".into(),
                uuid: Uuid::nil(),
            },
        );
        ingest_graph_execution_event(
            &captured,
            Some(&sampler),
            &progress,
            GraphExecutionEvent::Record {
                record: Box::new(record),
                node_id: None,
            },
        );

        let window = sampler.borrow_mut().take(10);
        assert_eq!(window.completed(), 1);
        assert_eq!(captured.borrow().len(), 1);
        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            *sink.returned.borrow(),
            vec![PhaseReturn {
                completes_session: true,
                releases_prefill: false,
                ..PhaseReturn::default()
            }]
        );
    }

    #[test]
    fn graph_phase_rejects_duplicate_first_token_for_one_request() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-duplicate-token".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        let uuid = Uuid::from_u128(7);
        progress.first_token("trace-duplicate-token", uuid);
        progress.first_token("trace-duplicate-token", uuid);

        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-duplicate-token\" emitted duplicate first-token event for request 00000000-0000-0000-0000-000000000007"
            )
        );
    }

    #[test]
    fn graph_phase_rejects_duplicate_terminal_record_without_double_return() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-duplicate-terminal".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        let record = graph_phase_record("trace-duplicate-terminal", false, false);
        progress.record(&record);
        progress.record(&record);

        assert_eq!(sink.returned.borrow().len(), 1);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-duplicate-terminal\" emitted duplicate terminal record for request 00000000-0000-0000-0000-000000000000"
            )
        );
    }

    #[test]
    fn graph_phase_uses_live_first_token_membership_when_record_metadata_disagrees() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-token-mismatch".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let mut record = graph_phase_record("trace-token-mismatch", false, false);
        record.ingest.first_token_ns = Some(5);
        progress.record(&record);

        assert_eq!(
            *sink.returned.borrow(),
            vec![PhaseReturn {
                completes_session: true,
                releases_prefill: true,
                ..PhaseReturn::default()
            }]
        );
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-token-mismatch\" request 00000000-0000-0000-0000-000000000000 first-token event/record mismatch: event=false record=true"
            )
        );
    }

    #[test]
    fn graph_phase_rejects_first_token_after_terminal_record() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-late-token".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let record = graph_phase_record("trace-late-token", false, false);
        progress.record(&record);
        progress.first_token("trace-late-token", record.uuid);

        assert_eq!(sink.first_tokens.get(), 0);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-late-token\" emitted first-token event after terminal record for request 00000000-0000-0000-0000-000000000000"
            )
        );
    }

    #[test]
    fn graph_phase_does_not_double_release_orphan_first_token_on_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-orphan-token".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        progress.first_token("trace-orphan-token", Uuid::from_u128(11));
        progress.complete(
            "trace-orphan-token",
            2,
            true,
            &Err(TraceError::Cancelled("phase cancelled".into())),
        );

        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    cancelled: true,
                    releases_prefill: false,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert!(failures.first().is_none());
    }

    #[test]
    fn warmup_terminal_node_failure_is_ledgered_for_abort_gate() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-fail".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Resilient warmup priming: a failed reply surfaces as an errored,
        // non-cancelled per-node record (Python `_on_graph_return` -> error).
        progress.record(&graph_phase_record("warmup-fail", true, false));
        assert_eq!(*ledger.borrow(), vec!["warmup-fail".to_owned()]);
    }

    #[test]
    fn warmup_trace_abort_is_ledgered_for_abort_gate() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-abort".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Fail-fast node policy: the trace itself aborts with a non-cancelled
        // error and never emits a terminal record.
        progress.complete(
            "warmup-abort",
            1,
            true,
            &Err(TraceError::Other("boom".into())),
        );
        assert_eq!(*ledger.borrow(), vec!["warmup-abort".to_owned()]);
        assert!(failures.first().is_some());
    }

    #[test]
    fn warmup_cancelled_return_is_not_a_terminal_failure() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-cancelled".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Cancelled returns are excluded even when they carry error text
        // (agentx parity: a drain-cancel is self-inflicted teardown).
        progress.record(&graph_phase_record("warmup-cancelled", true, true));
        assert!(ledger.borrow().is_empty());
        // And a trace cancelled at completion is likewise not ledgered.
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-drain".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        progress.complete(
            "warmup-drain",
            1,
            true,
            &Err(TraceError::Cancelled("phase grace expired".into())),
        );
        assert!(ledger.borrow().is_empty());
    }

    #[test]
    fn profiling_phase_never_ledgers_warmup_failures() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let ledger = Rc::new(RefCell::new(Vec::new()));
        // is_warmup = false: the same terminal failure is a normal profiling
        // failure, never a pre-profiling abort trigger.
        let progress = GraphPhaseProgress::new(
            sink,
            failures,
            outcome,
            Rc::new(crate::clock::SimClock::new()),
            Rc::new(GraphLaneLedger::default()),
            false,
            ledger.clone(),
        );
        progress.admit(&TraceAdmissionInfo {
            trace_id: "profiling-fail".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("profiling-fail", true, false));
        assert!(ledger.borrow().is_empty());
    }

    fn lane_identity(template: &str, instance: &str, t_star_us: f64) -> GraphLaneIdentity {
        GraphLaneIdentity {
            template_trace_id: template.into(),
            instance_id: instance.into(),
            t_star_us,
        }
    }

    #[test]
    fn lane_ledger_records_executed_and_return_walls_per_registered_lane() {
        // The ledger keys everything by lane index; a return arriving keyed by
        // its instance id is attributed to the lane that registered it.
        let ledger = GraphLaneLedger::default();
        ledger.register_lane(0, lane_identity("tmpl-a", "tmpl-a::inst-a", 1_000.0));
        ledger.register_lane(1, lane_identity("tmpl-b", "tmpl-b::inst-b", 2_000.0));

        ledger.observe_return("tmpl-a::inst-a", "n_0", 5.0);
        ledger.observe_return("tmpl-a::inst-a", "n_1", 9.0);
        ledger.observe_return("tmpl-b::inst-b", "n_0", 12.0);
        // A return for an unregistered instance is a graceful no-op.
        ledger.observe_return("tmpl-c::inst-c", "n_0", 99.0);

        assert_eq!(
            ledger.executed_node_ids(0),
            BTreeSet::from(["n_0".to_owned(), "n_1".to_owned()])
        );
        assert_eq!(
            ledger.return_wall_us(0),
            BTreeMap::from([("n_0".to_owned(), 5.0), ("n_1".to_owned(), 9.0)])
        );
        assert_eq!(
            ledger.executed_node_ids(1),
            BTreeSet::from(["n_0".to_owned()])
        );
        assert_eq!(
            ledger.return_wall_us(1),
            BTreeMap::from([("n_0".to_owned(), 12.0)])
        );
        assert_eq!(
            ledger.lane_identity(0),
            Some(lane_identity("tmpl-a", "tmpl-a::inst-a", 1_000.0))
        );
        assert_eq!(ledger.registered_lanes(), vec![0, 1]);
        // The unregistered instance never created a lane entry.
        assert_eq!(ledger.registered_lanes().len(), 2);
    }

    #[test]
    fn lane_return_wall_is_taken_from_the_injected_clock_and_excludes_cancellations() {
        // Drive terminal node returns through the drain-ingest path with a
        // SimClock so the recorded wall is exactly `clock.now_ns() / 1000` at
        // return-handling time, and a cancelled return is kept out of the
        // executed set (agentx `not cancelled` parity).
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let clock = Rc::new(crate::clock::SimClock::new());
        let lanes = Rc::new(GraphLaneLedger::default());
        lanes.register_lane(0, lane_identity("tmpl-a", "tmpl-a::inst-a", 1_000.0));
        let progress = GraphPhaseProgress::new(
            sink,
            failures.clone(),
            outcome,
            clock.clone(),
            lanes.clone(),
            false,
            Rc::new(RefCell::new(Vec::new())),
        );
        // Instance a reserves three nodes: two real returns plus a cancelled one.
        progress.admit(&TraceAdmissionInfo {
            trace_id: "tmpl-a::inst-a".into(),
            node_count: 3,
            arrival_ns: 0,
        });
        let captured = Rc::new(RefCell::new(Vec::new()));

        let feed = |node_id: &str, uuid: u128, canceled: bool, at_ns: i64| {
            clock.advance_to(at_ns);
            let mut record = graph_phase_record("tmpl-a::inst-a", canceled, canceled);
            record.uuid = Uuid::from_u128(uuid);
            ingest_graph_execution_event(
                &captured,
                None,
                &progress,
                GraphExecutionEvent::Record {
                    record: Box::new(record),
                    node_id: Some(node_id.to_owned()),
                },
            );
        };
        feed("n_0", 1, false, 5_000);
        feed("n_1", 2, false, 9_000);
        // A cancelled return at a later instant is NOT ledgered.
        feed("n_2", 3, true, 20_000);

        assert!(failures.first().is_none());
        assert_eq!(
            lanes.executed_node_ids(0),
            BTreeSet::from(["n_0".to_owned(), "n_1".to_owned()])
        );
        assert_eq!(
            lanes.return_wall_us(0),
            BTreeMap::from([("n_0".to_owned(), 5.0), ("n_1".to_owned(), 9.0)])
        );
    }

    #[test]
    fn lane_return_without_a_node_id_is_not_ledgered() {
        // The offline dynosim adapter carries no node id; such a record must
        // never touch the lane ledger even for a registered instance.
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let clock = Rc::new(crate::clock::SimClock::new());
        let lanes = Rc::new(GraphLaneLedger::default());
        lanes.register_lane(0, lane_identity("tmpl-a", "tmpl-a::inst-a", 0.0));
        let progress = GraphPhaseProgress::new(
            sink,
            failures,
            outcome,
            clock,
            lanes.clone(),
            false,
            Rc::new(RefCell::new(Vec::new())),
        );
        progress.admit(&TraceAdmissionInfo {
            trace_id: "tmpl-a::inst-a".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let captured = Rc::new(RefCell::new(Vec::new()));
        ingest_graph_execution_event(
            &captured,
            None,
            &progress,
            GraphExecutionEvent::Record {
                record: Box::new(graph_phase_record("tmpl-a::inst-a", false, false)),
                node_id: None,
            },
        );
        assert!(lanes.executed_node_ids(0).is_empty());
        assert!(lanes.return_wall_us(0).is_empty());
    }

    #[test]
    fn trajectory_warmup_failed_error_downcasts_to_execution_stage_failure() {
        let error = trajectory_warmup_failed_error(&["trace-a".to_owned(), "trace-b".to_owned()]);
        let failure = error
            .downcast_ref::<PreparedRunFailure>()
            .expect("structured warmup failure");
        assert_eq!(failure.stage, FailureStageV2::Execution);
        assert_eq!(failure.code, "trajectory_warmup_failed");
        assert!(failure.message.contains("trace-a"));
        assert!(failure.message.contains("trace-b"));
        assert!(failure.message.contains("2 trace(s)"));
    }

    #[test]
    fn pressure_lane_count_matches_python_resolution() {
        // recycle_bounded: lane count is the concurrency, independent of corpus.
        assert_eq!(pressure_resolve_lane_count(8, 2, None, true), 8);
        // --num-conversations clamps total distinct roots.
        assert_eq!(pressure_resolve_lane_count(8, 2, Some(3), true), 3);
        // Unbounded (no stop condition): clamp to the corpus (single pass width).
        assert_eq!(pressure_resolve_lane_count(8, 2, None, false), 2);
        // Always at least one lane.
        assert_eq!(pressure_resolve_lane_count(0, 0, None, true), 1);
    }

    #[test]
    fn pressure_draw_index_is_sequential_wrap() {
        // The default/sequential draw is the byte-unchanged `x % total` cursor.
        let draw = PermutationDraw::sequential();
        assert_eq!(draw.index(0, 3), 0);
        assert_eq!(draw.index(2, 3), 2);
        assert_eq!(draw.index(3, 3), 0);
        assert_eq!(draw.index(7, 3), 1);
        // Degenerate empty corpus never panics.
        assert_eq!(draw.index(5, 0), 0);
    }

    fn pressure_one_node_plan(id: &str) -> GraphTracePlan {
        use crate::graph::model::{GraphRecord, LlmNode, StaticEdge};
        use std::collections::BTreeMap;
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "n_0".to_owned(),
            LlmNode {
                output: "out".to_owned(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: Vec::new(),
                metadata: BTreeMap::new(),
            },
        );
        GraphTracePlan {
            graph: GraphRecord {
                nodes,
                edges: vec![StaticEdge {
                    source: "START".to_owned(),
                    target: "n_0".to_owned(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                }],
                ..Default::default()
            },
            trace: TraceRecord {
                id: id.to_owned(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        }
    }

    #[test]
    fn pressure_pass0_lanes_skip_empty_and_report_cursor() {
        // A 3-template corpus with the middle template rewritten empty
        // (unspawnable): lane assignment skips it and the cursor advances past
        // the skip (agentx `_resolve_pass0_lanes` parity).
        let mut empty = pressure_one_node_plan("b");
        empty.graph.nodes.clear();
        let templates = vec![
            PressureTemplate {
                plan: pressure_one_node_plan("a"),
                template_id: "a".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("a"),
            },
            PressureTemplate {
                plan: empty,
                template_id: "b".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("b"),
            },
            PressureTemplate {
                plan: pressure_one_node_plan("c"),
                template_id: "c".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("c"),
            },
        ];
        let window = TStarWindow::default();
        let draw = PressureDraw::from_window(window);
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());
        let (pass0, cursor) = pressure_resolve_pass0_lanes(&templates, 2, &draw, window, &cache);
        // Lanes 0 and 1 take templates a (index 0) and c (index 2); b skipped.
        assert_eq!(pass0, vec![0, 2]);
        // Cursor is one past the last consumed corpus position (a, b, c walked).
        assert_eq!(cursor, 3);
    }

    #[test]
    fn pressure_draw_sequential_default_is_byte_unchanged() {
        // Default / Sequential window: `PressureDraw::index` == `x % total` at
        // every counter (byte-for-byte the historical cursor-with-wrap draw).
        let draw = PressureDraw::from_window(TStarWindow::default());
        for x in 0u64..20 {
            for total in 1usize..7 {
                assert_eq!(draw.index(x, total), (x % total as u64) as usize);
            }
        }
        // Degenerate empty corpus never panics.
        assert_eq!(draw.index(5, 0), 0);
    }

    #[test]
    fn pressure_draw_shuffle_matches_legacy_shuffle_and_covers_each_pass() {
        // Shuffle window: `PressureDraw` routes through the shared persistent-epoch
        // `PermutationDraw`, so its draws equal a reference draw on the same legacy
        // `ShuffleSampler` child seed, and every full pass covers each index once.
        // run root 0 -> legacy_shuffle_seed(0) == 5203359018791016587.
        let window = TStarWindow {
            start_min_ratio: 0.0,
            start_max_ratio: 0.0,
            random_seed: 0,
            run_random_seed: 0,
            sampling_strategy: GraphSamplingStrategy::Shuffle,
        };
        let draw = PressureDraw::from_window(window);
        let reference = PermutationDraw::shuffle(legacy_shuffle_seed(0));
        let total = 8usize;
        for pass in 0u64..3 {
            let mut seen = Vec::new();
            for offset in 0..total as u64 {
                let x = pass * total as u64 + offset;
                let idx = draw.index(x, total);
                assert_eq!(idx, reference.index(x, total));
                seen.push(idx);
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..total).collect::<Vec<_>>(), "pass {pass} coverage");
        }
    }

    #[test]
    fn pressure_draw_random_is_with_replacement_and_distinct_from_shuffle() {
        // `Random` now reproduces legacy `RandomSampler` (WITH replacement) via the
        // CPython MT19937 `choice` stream — a DISTINCT draw from the shuffle-epoch
        // path, and it salts the SAME run root with a different label.
        let root = 42u64;
        let shuffle_window = TStarWindow {
            run_random_seed: root,
            sampling_strategy: GraphSamplingStrategy::Shuffle,
            ..Default::default()
        };
        let random_window = TStarWindow {
            run_random_seed: root,
            sampling_strategy: GraphSamplingStrategy::Random,
            ..Default::default()
        };
        let shuffle_draw = PressureDraw::from_window(shuffle_window);
        let random_draw = PressureDraw::from_window(random_window);
        // The random draw equals the reference stream on the legacy child seed.
        let reference = PermutationDraw::random(legacy_random_seed(root));
        let total = 8usize;
        let mut diverged = false;
        for x in 0u64..20 {
            assert_eq!(random_draw.index(x, total), reference.index(x, total));
            if random_draw.index(x, total) != shuffle_draw.index(x, total) {
                diverged = true;
            }
        }
        assert!(
            diverged,
            "random must diverge from shuffle (with replacement)"
        );
    }

    #[test]
    fn sample_lane_tstar_salts_higher_lanes_and_keeps_lane0_and_default_identity() {
        // Port parity for `graph_ir_replay.py:_plan_for_lane`/`_lane_source`
        // (lines 743-791): lane 0 reuses the prebuilt lane-0 t*, higher lanes
        // draw a DISTINCT lane-salted t*, and the default `[0, 0]` window keeps
        // every lane at t*=0 (identity). Expected values come straight from the
        // already-tested `WindowTStarSampler`/`seed_for_trace_lane` substrate.
        let window = TStarWindow {
            start_min_ratio: 0.2,
            start_max_ratio: 0.8,
            random_seed: 0,
            ..Default::default()
        };
        let sampler = WindowTStarSampler {
            start_min_ratio: window.start_min_ratio,
            start_max_ratio: window.start_max_ratio,
            random_seed: window.random_seed,
        };
        let duration_us = 1_000.0;
        // The prebuilt lane-0 t* the recycle template carries (what
        // `apply_tstar_split`/`sample_plan_tstar` produced for lane 0).
        let lane0_expected = sampler.sample_t_star("t", 0, duration_us);
        let template = PressureTemplate {
            plan: pressure_one_node_plan("t"),
            template_id: "t".into(),
            t_star_us: lane0_expected,
            duration_us,
            original_plan: pressure_one_node_plan("t"),
        };

        // Lane 0 is byte-identical to the prebuilt single-pass lane-0 value.
        assert_eq!(sample_lane_tstar(&template, 0, window), lane0_expected);
        // A higher lane draws its OWN salted t*, matching `sample_t_star` on that
        // lane index, and is DISTINCT from lane 0 (lanes decorrelate).
        let lane1 = sample_lane_tstar(&template, 1, window);
        let lane2 = sample_lane_tstar(&template, 2, window);
        assert_eq!(lane1, sampler.sample_t_star("t", 1, duration_us));
        assert_eq!(lane2, sampler.sample_t_star("t", 2, duration_us));
        assert_ne!(lane1, lane0_expected, "lane 1 must not reuse lane 0's t*");
        assert_ne!(lane2, lane1, "distinct lanes must draw distinct t*");

        // HARD guard: the default `[0, 0]` window collapses EVERY lane to t*=0,
        // so lane fan-out adds no divergence on the working profiling path.
        let default_window = TStarWindow::default();
        let default_template = PressureTemplate {
            plan: pressure_one_node_plan("t"),
            template_id: "t".into(),
            t_star_us: 0.0,
            duration_us,
            original_plan: pressure_one_node_plan("t"),
        };
        for lane in 0..3u64 {
            assert_eq!(
                sample_lane_tstar(&default_template, lane, default_window),
                0.0,
                "default window lane {lane} must stay at t*=0"
            );
        }
    }

    /// Build the recycle template for the 3-turn chain fixture exactly as
    /// [`build_pressure_recycle`] does for one root: the prebuilt `plan` is the
    /// lane-0 warmup rewrite, `original_plan` is the full pre-warmup graph, and
    /// the lane-0 `t*`/`duration_us` come from the original.
    fn tstar_chain_pressure_template(window: TStarWindow) -> PressureTemplate {
        let original = tstar_chain_plan().remove(0);
        let lane0_t_star = sample_plan_tstar(&original, window);
        let parsed = single_trace_parsed(&original);
        let warmup = rewrite_for_warmup(&parsed, lane0_t_star);
        PressureTemplate {
            plan: GraphTracePlan {
                graph: warmup.graph,
                trace: original.trace.clone(),
                arrival_offset_ns: original.arrival_offset_ns,
            },
            template_id: original.trace.id.clone(),
            t_star_us: lane0_t_star,
            duration_us: plan_trace_duration_us(&original),
            original_plan: original,
        }
    }

    #[test]
    fn pressure_plan_for_lane_reprimes_higher_lanes_at_their_own_tstar() {
        // DF1b: a higher recycle lane must dispatch a warmup graph re-primed at
        // ITS OWN lane-salted t* (the same value its `GraphLaneIdentity` records),
        // not lane 0's prebuilt graph. `graph_ir_replay.py:_plan_for_lane` 743-791.
        let window = TStarWindow {
            start_min_ratio: 0.2,
            start_max_ratio: 0.8,
            random_seed: 0,
            ..Default::default()
        };
        let template = tstar_chain_pressure_template(window);
        let templates = vec![template.clone()];
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());

        // HARD guard: lane 0 dispatches the prebuilt lane-0 warmup graph, and
        // never populates the per-lane cache.
        let lane0 = pressure_plan_for_lane(&templates, 0, 0, template.t_star_us, window, &cache);
        assert_eq!(plan_node_ids(&lane0), plan_node_ids(&template.plan));
        assert_eq!(lane0.graph.edges.len(), template.plan.graph.edges.len());
        assert!(cache.borrow().is_empty(), "lane 0 must not cache a rewrite");

        // A higher lane's dispatched graph equals `rewrite_for_warmup` at the SAME
        // t* the lane's identity carries (`sample_lane_tstar`), proving the
        // executed-node set and the identity t* can never diverge.
        let lane1_t_star = sample_lane_tstar(&template, 1, window);
        let expected = GraphTracePlan {
            graph: rewrite_for_warmup(&single_trace_parsed(&template.original_plan), lane1_t_star)
                .graph,
            trace: template.original_plan.trace.clone(),
            arrival_offset_ns: template.original_plan.arrival_offset_ns,
        };
        let lane1 = pressure_plan_for_lane(&templates, 0, 1, lane1_t_star, window, &cache);
        assert_eq!(plan_node_ids(&lane1), plan_node_ids(&expected));
        assert_eq!(lane1.graph.edges.len(), expected.graph.edges.len());
        // Cached by `(template_index, lane)`; a repeat pass returns the same graph.
        assert_eq!(cache.borrow().len(), 1);
        let lane1_again = pressure_plan_for_lane(&templates, 0, 1, lane1_t_star, window, &cache);
        assert_eq!(plan_node_ids(&lane1_again), plan_node_ids(&lane1));
        assert_eq!(cache.borrow().len(), 1, "repeat pass must reuse the cache");

        // Real divergence: over the [0.2, 0.8]*2e6 window (straddling the mid-chain
        // 1e6 arrival) SOME lane primes a DIFFERENT boundary-node set than lane 0.
        let lane0_nodes = plan_node_ids(&lane0);
        let diverged = (1..32u64).any(|lane| {
            let t = sample_lane_tstar(&template, lane, window);
            let plan = pressure_plan_for_lane(&templates, 0, lane, t, window, &cache);
            plan_node_ids(&plan) != lane0_nodes
        });
        assert!(
            diverged,
            "no higher lane primed a boundary set distinct from lane 0"
        );
    }

    #[test]
    fn pressure_plan_for_lane_default_window_is_lane0_for_every_lane() {
        // HARD guard: the default `[0, 0]` window collapses every lane to t*=0, so
        // each lane dispatches the prebuilt lane-0 warmup graph with NO per-lane
        // divergence and NO cache population (the product/default path).
        let window = TStarWindow::default();
        let template = tstar_chain_pressure_template(window);
        let templates = vec![template.clone()];
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());
        for lane in 0..5u64 {
            let t = sample_lane_tstar(&template, lane, window);
            let plan = pressure_plan_for_lane(&templates, 0, lane, t, window, &cache);
            assert_eq!(
                plan_node_ids(&plan),
                plan_node_ids(&template.plan),
                "default window lane {lane} must reuse the prebuilt lane-0 graph"
            );
            assert_eq!(plan.graph.edges.len(), template.plan.graph.edges.len());
        }
        assert!(
            cache.borrow().is_empty(),
            "default window must never cache a per-lane rewrite"
        );
    }

    /// A single 2-turn recorded chain whose two turns arrive at `1e6` and `2e6`
    /// us. `warmup_boundary_nodes` primes a chain ONLY when `t*` straddles its
    /// arrivals (some turn `< t*` AND some turn `>= t*`), so this chain's warmup
    /// rewrite is NON-EMPTY exactly for `t*` in `(1e6, 2e6]` and EMPTY below `1e6`
    /// — the straddle-sensitive fixture DF6's per-lane spawnability needs.
    fn two_turn_chain_plan() -> GraphTracePlan {
        use crate::graph::model::{ChannelRequirement, GraphRecord, LlmNode, StaticEdge};
        use serde_json::json;
        use std::collections::BTreeMap;

        let node = |arrival: u64, inputs: &[&str]| {
            let mut metadata = BTreeMap::new();
            metadata.insert("arrival_offset_us".to_owned(), json!(arrival));
            metadata.insert("conversation_id".to_owned(), json!("n"));
            LlmNode {
                output: "out".to_owned(),
                streaming: true,
                inputs: inputs
                    .iter()
                    .map(|c| ChannelRequirement {
                        channel: (*c).to_owned(),
                        count: Default::default(),
                    })
                    .collect(),
                min_start_delay_us: None,
                max_tokens: None,
                items: Vec::new(),
                metadata,
            }
        };
        let edge = |source: &str, target: &str| StaticEdge {
            source: source.to_owned(),
            target: target.to_owned(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        };
        let mut nodes = BTreeMap::new();
        nodes.insert("n_0".to_owned(), node(1_000_000, &[]));
        nodes.insert("n_1".to_owned(), node(2_000_000, &["n_0_out"]));
        let graph = GraphRecord {
            nodes,
            edges: vec![edge("START", "n_0"), edge("n_0", "n_1"), edge("n_1", "END")],
            ..Default::default()
        };
        GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: "t".to_owned(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        }
    }

    /// Build the warmup [`PressureTemplate`] for one root at `window`, exactly as
    /// [`build_pressure_recycle`] does (lane-0 warmup rewrite as `plan`, full
    /// pre-warmup graph as `original_plan`, lane-0 `t*`/duration from the original).
    fn pressure_template_for(original: GraphTracePlan, window: TStarWindow) -> PressureTemplate {
        let lane0_t_star = sample_plan_tstar(&original, window);
        let parsed = single_trace_parsed(&original);
        let warmup = rewrite_for_warmup(&parsed, lane0_t_star);
        PressureTemplate {
            plan: GraphTracePlan {
                graph: warmup.graph,
                trace: original.trace.clone(),
                arrival_offset_ns: original.arrival_offset_ns,
            },
            template_id: original.trace.id.clone(),
            t_star_us: lane0_t_star,
            duration_us: plan_trace_duration_us(&original),
            original_plan: original,
        }
    }

    #[test]
    fn pressure_resolve_pass0_judges_spawnability_at_each_lane_tstar() {
        // DF6: `_resolve_pass0_lanes` judges each candidate at its TARGET RANK's
        // lane-salted t* (Python `_is_spawnable(trace, rank)`), NOT lane 0's
        // prebuilt plan. Over an active window the straddle-sensitive 2-turn chain
        // is non-empty at some lane t* and EMPTY at others, so a template
        // dispatchable at lane 0 can be empty at a higher lane. Judging at the
        // lane's own t* means that lane never "consumes" (and then dispatches
        // empty in `admit`) such a template. Deterministically pick a base seed
        // where the single template is spawnable at lane 0 but EMPTY at lane 1.
        let base = |seed| TStarWindow {
            start_min_ratio: 0.1,
            start_max_ratio: 0.9,
            random_seed: seed,
            ..Default::default()
        };
        let empty_at = |template: &PressureTemplate, lane: u64, window: TStarWindow| -> bool {
            let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> =
                RefCell::new(HashMap::new());
            let templates = std::slice::from_ref(template);
            let t = sample_lane_tstar(template, lane, window);
            pressure_plan_for_lane(templates, 0, lane, t, window, &cache)
                .graph
                .nodes
                .is_empty()
        };
        let seed = (0u64..10_000)
            .find(|&s| {
                let w = base(s);
                let template = pressure_template_for(two_turn_chain_plan(), w);
                !empty_at(&template, 0, w) && empty_at(&template, 1, w)
            })
            .expect("a base seed with lane 0 spawnable but lane 1 empty must exist");
        let window = base(seed);
        let templates = vec![pressure_template_for(two_turn_chain_plan(), window)];
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());
        let draw = PressureDraw::from_window(window);
        let (pass0, cursor) = pressure_resolve_pass0_lanes(&templates, 2, &draw, window, &cache);

        // Lane 1 would dispatch EMPTY at its own t*, so the resolve does NOT
        // consume the template for it (the pre-DF6 lane-0 judge WOULD have
        // returned `[0, 0]` and dispatched an empty lane 1 -> spurious "contains
        // no dispatchable nodes" in `admit`). Only lane 0 survives.
        assert_eq!(
            pass0,
            vec![0],
            "empty-at-lane-1 template must not fill lane 1"
        );
        // The skip walk exhausts its bounded budget (`lanes + n` = 3).
        assert_eq!(cursor, 3);
        // Invariant: every assigned lane is non-empty at ITS OWN t*.
        for (lane, &idx) in pass0.iter().enumerate() {
            let l = lane as u64;
            let t = sample_lane_tstar(&templates[idx], l, window);
            assert!(
                !pressure_plan_for_lane(&templates, idx, l, t, window, &cache)
                    .graph
                    .nodes
                    .is_empty(),
                "lane {l} assigned a template it dispatches empty at"
            );
        }
    }

    #[test]
    fn pressure_resolve_pass0_default_window_is_byte_unchanged() {
        // HARD guard: at the default `[0, 0]` window spawnability collapses to
        // "the prebuilt lane-0 warmup plan is non-empty" for every lane, so the
        // resolve is byte-identical to the pre-DF6 lane-0 judge. A 3-template
        // corpus with an empty middle template skips it and reports the same
        // cursor as `pressure_pass0_lanes_skip_empty_and_report_cursor`.
        let mut empty = pressure_one_node_plan("b");
        empty.graph.nodes.clear();
        let templates = vec![
            PressureTemplate {
                plan: pressure_one_node_plan("a"),
                template_id: "a".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("a"),
            },
            PressureTemplate {
                plan: empty,
                template_id: "b".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("b"),
            },
            PressureTemplate {
                plan: pressure_one_node_plan("c"),
                template_id: "c".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("c"),
            },
        ];
        let window = TStarWindow::default();
        let draw = PressureDraw::from_window(window);
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());
        let (pass0, cursor) = pressure_resolve_pass0_lanes(&templates, 2, &draw, window, &cache);
        assert_eq!(pass0, vec![0, 2]);
        assert_eq!(cursor, 3);
        // Default window judges via the prebuilt lane-0 plan, so no per-lane
        // rewrite is cached.
        assert!(cache.borrow().is_empty(), "default window must not cache");
    }

    #[test]
    fn profiling_plan_for_lane_chops_higher_lanes_at_their_own_tstar() {
        // DF6: a pass-0 PROFILING lane chops the ORIGINAL graph at ITS OWN
        // lane-salted t* (the same instant its warmup lane primed / its identity
        // records), not lane 0's split. `graph_ir_replay.py:_run_instance` 1837:
        // `plan = _plan_for_lane(trace, lane_index)`.
        let window = TStarWindow {
            start_min_ratio: 0.2,
            start_max_ratio: 0.8,
            random_seed: 0,
            ..Default::default()
        };
        let original = tstar_chain_plan().remove(0);
        let originals = vec![original.clone()];
        // Profiling `split` = lane-0 frontier chop (what `apply_tstar_split`
        // produces for the profiling phase).
        let lane0_t = sample_plan_tstar(&original, window);
        let split = vec![GraphTracePlan {
            graph: chop_trie_at_tstar(&single_trace_parsed(&original), lane0_t).graph,
            trace: original.trace.clone(),
            arrival_offset_ns: original.arrival_offset_ns,
        }];
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());

        // Lane 0 reuses the lane-0 split verbatim; never caches.
        let lane0 = profiling_plan_for_lane(&originals, &split, 0, 0, window, &cache);
        assert_eq!(plan_node_ids(&lane0), plan_node_ids(&split[0]));
        assert!(cache.borrow().is_empty(), "lane 0 must not cache a chop");

        // A higher lane's plan equals `chop_trie_at_tstar` at the SAME t* the lane
        // records (`profiling_sample_lane_tstar`), and is BYTE-IDENTICAL to the
        // warmup lane t* for the same template+lane (shared salt).
        let lane1_t = profiling_sample_lane_tstar(&original, 1, window);
        assert_eq!(
            lane1_t,
            sample_lane_tstar(&pressure_template_for(original.clone(), window), 1, window),
            "profiling and warmup lane t* must agree for the same (template, lane)"
        );
        let expected = chop_trie_at_tstar(&single_trace_parsed(&original), lane1_t).graph;
        let lane1 = profiling_plan_for_lane(&originals, &split, 0, 1, window, &cache);
        assert_eq!(
            plan_node_ids(&lane1),
            expected.nodes.keys().cloned().collect::<Vec<_>>()
        );
        assert_eq!(cache.borrow().len(), 1);
        // Repeat pass reuses the cache.
        let lane1_again = profiling_plan_for_lane(&originals, &split, 0, 1, window, &cache);
        assert_eq!(plan_node_ids(&lane1_again), plan_node_ids(&lane1));
        assert_eq!(cache.borrow().len(), 1, "repeat pass must reuse the cache");

        // Real divergence: over the window SOME lane chops a DIFFERENT frontier
        // than lane 0.
        let lane0_nodes = plan_node_ids(&lane0);
        let diverged = (1..32u64).any(|lane| {
            let plan = profiling_plan_for_lane(&originals, &split, 0, lane, window, &cache);
            plan_node_ids(&plan) != lane0_nodes
        });
        assert!(
            diverged,
            "no higher profiling lane chopped a distinct frontier"
        );
    }

    #[test]
    fn profiling_plan_for_lane_default_window_is_lane0_split_for_every_lane() {
        // HARD guard: the default `[0, 0]` window collapses every lane to t*=0, so
        // each lane reuses the lane-0 split with NO cache population (the
        // product/default path is byte-unchanged).
        let window = TStarWindow::default();
        let original = tstar_chain_plan().remove(0);
        let originals = vec![original.clone()];
        let split = apply_tstar_split(&originals, &named_concurrency_phase("profiling"), window);
        let cache: RefCell<HashMap<(usize, u64), GraphTracePlan>> = RefCell::new(HashMap::new());
        for lane in 0..5u64 {
            let plan = profiling_plan_for_lane(&originals, &split, 0, lane, window, &cache);
            assert_eq!(
                plan_node_ids(&plan),
                plan_node_ids(&split[0]),
                "default window lane {lane} must reuse the lane-0 split"
            );
        }
        assert!(
            cache.borrow().is_empty(),
            "default window must never cache a per-lane chop"
        );
    }

    /// Placement stub that advances the injected clock a fixed amount per
    /// dispatch (so the pressure deadline is eventually reached under SimClock)
    /// and records every dispatched instance id.
    struct RecycleClockPlacement {
        clock: Rc<dyn crate::clock::Clock>,
        per_trace_ns: i64,
        dispatched: Rc<RefCell<Vec<String>>>,
    }

    #[async_trait::async_trait(?Send)]
    impl crate::graph::execution::TracePlacement for RecycleClockPlacement {
        async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
            self.dispatched.borrow_mut().push(plan.trace.id.clone());
            self.clock.clone().sleep(self.per_trace_ns).await;
            Ok(())
        }
    }

    #[test]
    fn pressure_recycle_recycles_beyond_one_corpus_pass_and_registers_lanes() {
        // A 2-template corpus at concurrency 2 with a duration that fits several
        // recycles: each lane must draw past the corpus (cursor > corpus size)
        // and every dispatched instance must register its lane in the ledger.
        let sim = Rc::new(crate::clock::SimClock::new());
        let clock: Rc<dyn crate::clock::Clock> = sim.clone();
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let placement: Rc<dyn crate::graph::execution::TracePlacement> =
            Rc::new(RecycleClockPlacement {
                clock: clock.clone(),
                per_trace_ns: 40,
                dispatched: dispatched.clone(),
            });
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let ledger = Rc::new(GraphLaneLedger::default());
        let progress = Rc::new(GraphPhaseProgress::new(
            sink,
            failures,
            Rc::new(RefCell::new(GraphWorkloadReport::default())),
            clock.clone(),
            ledger.clone(),
            true,
            Rc::new(RefCell::new(Vec::new())),
        ));
        let templates = Rc::new(vec![
            PressureTemplate {
                plan: pressure_one_node_plan("a"),
                template_id: "a".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("a"),
            },
            PressureTemplate {
                plan: pressure_one_node_plan("b"),
                template_id: "b".into(),
                t_star_us: 0.0,
                duration_us: 0.0,
                original_plan: pressure_one_node_plan("b"),
            },
        ]);
        let recycle = Rc::new(GraphPressureRecycle {
            clock: clock.clone(),
            placement,
            progress,
            templates,
            duration_ns: 100,
            lane_target: 2,
            session_limit: None,
            recycle_bounded: false,
            t_star: TStarWindow::default(),
            cancelled: Rc::new(Cell::new(false)),
            pressure_lane_count: Cell::new(0),
        });
        let run_recycle = recycle.clone();
        let outcome = crate::graph::runtime::drive_sim(sim, move |_handle| async move {
            run_recycle.run().await;
        });
        assert!(!outcome.deadlocked);

        // Recycle drew past the two-template corpus: the shared cursor advanced
        // well beyond the corpus size and more than `corpus` instances ran.
        assert!(
            ledger.corpus_cursor() > 2,
            "corpus cursor {} did not advance past the 2-template corpus",
            ledger.corpus_cursor()
        );
        assert!(
            dispatched.borrow().len() > 2,
            "only {} instances dispatched; expected recycle beyond one pass",
            dispatched.borrow().len()
        );
        // Both lanes registered their (recycled) identities in the E3a ledger.
        assert_eq!(ledger.registered_lanes(), vec![0, 1]);
        // Every dispatched instance carries the `{template}::{nonce}` identity.
        assert!(
            dispatched
                .borrow()
                .iter()
                .all(|id| id.starts_with("a::") || id.starts_with("b::"))
        );
    }

    fn warmup_pressure_phase(concurrency: usize, duration: Option<f64>) -> PhaseSpec {
        let mut value = serde_json::json!({
            "type": "concurrency",
            "name": "warmup",
            "exclude_from_results": true,
            "concurrency": concurrency,
        });
        if let Some(duration) = duration {
            value["agentic_cache_warmup_duration"] = serde_json::json!(duration);
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn build_pressure_recycle_engages_only_for_warmup_with_a_duration() {
        let plans = vec![pressure_one_node_plan("a"), pressure_one_node_plan("b")];
        let window = TStarWindow::default();

        // Warmup + duration: recycle engaged with concurrency lanes.
        let phase = warmup_pressure_phase(4, Some(2.0));
        let prepared = build_pressure_recycle(&plans, &plans, &phase, phase.common(), window)
            .unwrap()
            .expect("warmup with cache-pressure duration engages the recycle");
        assert_eq!(prepared.lane_target, 4);
        assert_eq!(prepared.templates.len(), 2);
        assert_eq!(prepared.duration_ns, 2_000_000_000);
        // Auto warmup carries no explicit phase stop condition.
        assert!(!prepared.recycle_bounded);
        // At the default window every lane-0 t* is 0.
        assert!(prepared.templates.iter().all(|t| t.t_star_us == 0.0));

        // Warmup WITHOUT a duration: no recycle (byte-unchanged single pass).
        let phase = warmup_pressure_phase(4, None);
        assert!(
            build_pressure_recycle(&plans, &plans, &phase, phase.common(), window)
                .unwrap()
                .is_none()
        );

        // Non-warmup phase never engages the recycle even with the field set.
        let mut profiling = warmup_pressure_phase(4, Some(2.0));
        if let PhaseSpec::Concurrency { common, .. } = &mut profiling {
            common.name = "profiling".into();
        }
        assert!(
            build_pressure_recycle(&plans, &plans, &profiling, profiling.common(), window)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn warmup_handoff_carries_ledger_executed_sets_cursor_and_lane_count() {
        // E3c handoff assembly: build_warmup_handoff reads each registered lane's
        // identity + executed set + return-wall ledger, plus the corpus cursor and
        // the pass-0 lane count, and stamps the Clock-derived drain-end wall.
        let ledger = GraphLaneLedger::default();
        ledger.register_lane(0, lane_identity("t", "t::inst-0", 1_000.0));
        ledger.register_lane(1, lane_identity("u", "u::inst-1", 0.0));
        ledger.observe_return("t::inst-0", "n_0", 5.0);
        ledger.observe_return("t::inst-0", "n_1", 9.0);
        ledger.observe_return("u::inst-1", "n_0", 12.0);
        ledger.set_corpus_cursor(7);

        let handoff = build_warmup_handoff(&ledger, 2, 100.0);
        assert_eq!(handoff.corpus_cursor, 7);
        assert_eq!(handoff.pressure_lane_count, 2);
        assert_eq!(handoff.drain_end_wall_us, 100.0);
        assert_eq!(handoff.lanes.len(), 2);

        let lane0 = &handoff.lanes[&0];
        assert_eq!(lane0.template_trace_id, "t");
        assert_eq!(lane0.instance_id, "t::inst-0");
        assert_eq!(lane0.t_star_us, 1_000.0);
        assert_eq!(
            lane0.executed_node_ids,
            BTreeSet::from(["n_0".to_owned(), "n_1".to_owned()])
        );
        assert_eq!(
            lane0.return_wall_us,
            BTreeMap::from([("n_0".to_owned(), 5.0), ("n_1".to_owned(), 9.0)])
        );
        // Executed keys and return-wall keys agree by construction (Python derives
        // `executed_node_ids = frozenset(live_walls)`).
        assert_eq!(
            handoff.lanes[&1].executed_node_ids,
            BTreeSet::from(["n_0".to_owned()])
        );
    }

    #[test]
    fn profiling_resume_frontier_drops_only_executed_nodes() {
        // At the default `[0, 0]` window the base profiling split is the full
        // `n_0 -> n_1 -> n_2` chain; a handoff marking n_0 executed re-cuts lane
        // 0's INSTANCE via chop_trie_at_frontier so it resumes ONLY not-yet-
        // executed nodes (n_1, n_2) — the frontier-continuity contract.
        let window = TStarWindow::default();
        let handoff = GraphWarmupHandoff {
            lanes: BTreeMap::from([(
                0u64,
                LaneHandoff {
                    template_trace_id: "t".to_owned(),
                    instance_id: "t::inst-0".to_owned(),
                    t_star_us: 0.0,
                    executed_node_ids: BTreeSet::from(["n_0".to_owned()]),
                    return_wall_us: BTreeMap::from([("n_0".to_owned(), 5.0)]),
                },
            )]),
            drain_end_wall_us: 5.0,
            corpus_cursor: 1,
            pressure_lane_count: 1,
        };
        let (prefix, fresh) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &handoff,
            Some(1),
            true,
        );
        assert_eq!(prefix.len(), 1);
        assert_eq!(prefix[0].lane, 0);
        assert_eq!(
            plan_node_ids(&prefix[0].plan),
            vec!["n_1".to_owned(), "n_2".to_owned()]
        );
        // Lane 0 resumes its OWN template — no fresh-start draw consumed.
        assert_eq!(fresh, 0);
    }

    #[test]
    fn profiling_resume_frontier_combines_t_star_and_executed() {
        // t*=1e6 drops the pre-t* n_0 AND the executed set drops n_1, so lane 0's
        // instance leaves only n_2 — the frontier chop honors both the lane's t*
        // and its executed ledger together.
        let window = TStarWindow::default();
        let handoff = GraphWarmupHandoff {
            lanes: BTreeMap::from([(
                0u64,
                LaneHandoff {
                    template_trace_id: "t".to_owned(),
                    instance_id: "t::inst-0".to_owned(),
                    t_star_us: 1_000_000.0,
                    executed_node_ids: BTreeSet::from(["n_1".to_owned()]),
                    return_wall_us: BTreeMap::from([("n_1".to_owned(), 9.0)]),
                },
            )]),
            drain_end_wall_us: 9.0,
            corpus_cursor: 1,
            pressure_lane_count: 1,
        };
        let (prefix, _) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &handoff,
            Some(1),
            true,
        );
        assert_eq!(prefix.len(), 1);
        assert_eq!(plan_node_ids(&prefix[0].plan), vec!["n_2".to_owned()]);
    }

    #[test]
    fn profiling_resume_two_lanes_same_template_are_two_instances() {
        // DF4 HARD guard: over-fan-out (pressure_lane_count > distinct traces) puts
        // TWO lanes on the SAME template. The faithful per-lane model dispatches
        // TWO distinct instances — each frontier-chopped with its OWN lane's
        // executed set — NOT one merged/overwritten plan. Lane 0 warmed n_0 (so its
        // instance resumes n_1, n_2); lane 1 warmed n_1 (so its instance resumes
        // n_0, n_2). The two survivor sets DIFFER, proving no union / no collision.
        let window = TStarWindow::default();
        let handoff = GraphWarmupHandoff {
            lanes: BTreeMap::from([
                (
                    0u64,
                    LaneHandoff {
                        template_trace_id: "t".to_owned(),
                        instance_id: "t::inst-0".to_owned(),
                        t_star_us: 0.0,
                        executed_node_ids: BTreeSet::from(["n_0".to_owned()]),
                        return_wall_us: BTreeMap::from([("n_0".to_owned(), 5.0)]),
                    },
                ),
                (
                    1u64,
                    LaneHandoff {
                        template_trace_id: "t".to_owned(),
                        instance_id: "t::inst-1".to_owned(),
                        t_star_us: 0.0,
                        executed_node_ids: BTreeSet::from(["n_1".to_owned()]),
                        return_wall_us: BTreeMap::from([("n_1".to_owned(), 8.0)]),
                    },
                ),
            ]),
            drain_end_wall_us: 10.0,
            corpus_cursor: 2,
            pressure_lane_count: 2,
        };
        let (prefix, _) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &handoff,
            Some(2),
            true,
        );
        // TWO distinct instances, one per lane, in lane order.
        assert_eq!(prefix.len(), 2);
        assert_eq!(prefix[0].lane, 0);
        assert_eq!(prefix[1].lane, 1);
        // Lane 0 dropped n_0 -> survivors n_1, n_2; lane 1 dropped n_1 -> survivors
        // n_0, n_2. Distinct survivor sets: not a single merged/union chop.
        assert_eq!(
            plan_node_ids(&prefix[0].plan),
            vec!["n_1".to_owned(), "n_2".to_owned()]
        );
        assert_eq!(
            plan_node_ids(&prefix[1].plan),
            vec!["n_0".to_owned(), "n_2".to_owned()]
        );
        assert_ne!(
            plan_node_ids(&prefix[0].plan),
            plan_node_ids(&prefix[1].plan)
        );
        // DF4b: the two lanes carry the two DIFFERENT pressure instance ids, so
        // each stays KV-continuous with its own pressure instance and the two
        // resumed instances remain distinct.
        assert_eq!(prefix[0].resume_instance_id.as_deref(), Some("t::inst-0"));
        assert_eq!(prefix[1].resume_instance_id.as_deref(), Some("t::inst-1"));
    }

    #[test]
    fn profiling_resume_empty_lane_fresh_starts_from_cursor() {
        // DF4 HARD guard: a pressure lane below `pressure_lane_count` with NO
        // handoff entry (completed exactly at drain) fresh-starts at full `t*=0`
        // replay from the shared cursor, NOT a re-run of an already-warm t* resume.
        // Corpus [a, b, c]; lane 0 resumes template "a"; lane 1 (< pressure_lane_count
        // 2, no entry, bounded) fresh-starts drawing cursor template 1 = "b".
        let window = TStarWindow::default();
        let corpus = vec![
            pressure_one_node_plan("a"),
            pressure_one_node_plan("b"),
            pressure_one_node_plan("c"),
        ];
        let handoff = GraphWarmupHandoff {
            lanes: BTreeMap::from([(
                0u64,
                LaneHandoff {
                    template_trace_id: "a".to_owned(),
                    instance_id: "a::inst-0".to_owned(),
                    t_star_us: 0.0,
                    executed_node_ids: BTreeSet::new(),
                    return_wall_us: BTreeMap::new(),
                },
            )]),
            drain_end_wall_us: 5.0,
            corpus_cursor: 1,
            pressure_lane_count: 2,
        };
        let (prefix, fresh) = build_profiling_resume_lane_plans(
            &corpus,
            &named_concurrency_phase("profiling"),
            window,
            &handoff,
            Some(3),
            true,
        );
        assert_eq!(prefix.len(), 2);
        // Lane 0 resumes its own template "a".
        assert_eq!(prefix[0].lane, 0);
        assert_eq!(prefix[0].plan.trace.id, "a");
        // Lane 1 fresh-starts template "b" (cursor 1 % 3) at the full one-node graph.
        assert_eq!(prefix[1].lane, 1);
        assert_eq!(prefix[1].plan.trace.id, "b");
        assert_eq!(plan_node_ids(&prefix[1].plan), vec!["n_0".to_owned()]);
        // One fresh-start draw consumed one cursor position.
        assert_eq!(fresh, 1);
        // DF4b: lane 0 resumed -> carries the pressure id; lane 1 fresh-started
        // (never primed) -> None, so it keeps its normal per-lane native id.
        assert_eq!(prefix[0].resume_instance_id.as_deref(), Some("a::inst-0"));
        assert_eq!(prefix[1].resume_instance_id, None);
    }

    #[test]
    fn profiling_resume_without_matching_lane_equals_tstar_split() {
        // HARD no-pressure regression guard at the plan layer: an EMPTY handoff
        // yields exactly one pass-0 lane instance byte-identical to the plain
        // `apply_tstar_split` profiling plan, and a handoff whose template is not
        // in the loaded corpus leaves its lane on the normal pass-0 t* assignment.
        let window = TStarWindow::default();
        let base = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
        );

        let empty = GraphWarmupHandoff::default();
        let (prefix_empty, fresh_empty) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &empty,
            Some(1),
            true,
        );
        assert_eq!(prefix_empty.len(), 1);
        assert_eq!(plan_node_ids(&prefix_empty[0].plan), node_ids(&base));
        assert_eq!(fresh_empty, 0);

        let foreign = GraphWarmupHandoff {
            lanes: BTreeMap::from([(
                0u64,
                LaneHandoff {
                    template_trace_id: "not-in-corpus".to_owned(),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        let (prefix_foreign, _) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &foreign,
            Some(1),
            true,
        );
        assert_eq!(prefix_foreign.len(), 1);
        assert_eq!(plan_node_ids(&prefix_foreign[0].plan), node_ids(&base));
        // DF4b: a not-in-corpus resume falls back to the normal pass-0 plan and
        // its normal id (Python drops the entry on the template-id guard), so no
        // pressure id is reused — there is no matching KV to preserve.
        assert_eq!(prefix_foreign[0].resume_instance_id, None);
    }

    #[test]
    fn lane_resume_source_dispatches_prefix_then_recycle() {
        // The profiling source yields the per-lane resume PREFIX first (one
        // instance per lane, per-lane-unique id) THEN delegates to the corpus
        // recycle. Two same-template lanes stay distinct; the recycle keeps its own
        // `::instance-{n}` id space, so no id collides across the seam.
        let prefix = vec![
            LaneResumePlan {
                lane: 0,
                plan: pressure_one_node_plan("t"),
                resume_instance_id: None,
            },
            LaneResumePlan {
                lane: 1,
                plan: pressure_one_node_plan("t"),
                resume_instance_id: None,
            },
        ];
        let recycle = build_graph_trace_source(
            vec![pressure_one_node_plan("t")],
            Some(2),
            None,
            GraphTraceInstanceSequence::default(),
            0,
            TStarWindow::default(),
        )
        .unwrap();
        let source = LaneResumeGraphTraceSource {
            prefix,
            next_prefix: Cell::new(0),
            recycle,
        };
        let drawn: Vec<String> =
            std::iter::from_fn(|| source.next_trace().unwrap().map(|plan| plan.trace.id)).collect();
        assert_eq!(
            drawn,
            vec![
                "t::resume-lane-0".to_owned(),
                "t::resume-lane-1".to_owned(),
                "t::instance-0".to_owned(),
                "t::instance-1".to_owned(),
            ]
        );
    }

    #[test]
    fn lane_resume_source_reuses_pressure_instance_id_for_marker_continuity() {
        // DF4b HARD guard: a lane resuming a handoff entry dispatches under the
        // EXACT pressure instance id (`entry.instance_id`), NOT a fresh
        // `::resume-lane-{lane}` id, so the cache-bust marker (digest of
        // trace.id) is continuous and the pressure-built KV transfers. Two lanes
        // on the same template carry the two DIFFERENT pressure instance ids —
        // still distinct, still KV-continuous each.
        let prefix = vec![
            LaneResumePlan {
                lane: 0,
                plan: pressure_one_node_plan("t"),
                resume_instance_id: Some("t::inst-0".to_owned()),
            },
            LaneResumePlan {
                lane: 1,
                plan: pressure_one_node_plan("t"),
                resume_instance_id: Some("t::inst-1".to_owned()),
            },
        ];
        let recycle = build_graph_trace_source(
            vec![pressure_one_node_plan("t")],
            Some(2),
            None,
            GraphTraceInstanceSequence::default(),
            0,
            TStarWindow::default(),
        )
        .unwrap();
        let source = LaneResumeGraphTraceSource {
            prefix,
            next_prefix: Cell::new(0),
            recycle,
        };
        let drawn: Vec<String> =
            std::iter::from_fn(|| source.next_trace().unwrap().map(|plan| plan.trace.id)).collect();
        assert_eq!(
            drawn,
            vec![
                "t::inst-0".to_owned(),
                "t::inst-1".to_owned(),
                "t::instance-0".to_owned(),
                "t::instance-1".to_owned(),
            ]
        );
    }

    #[test]
    fn build_profiling_resume_lane_plans_carries_pressure_instance_id() {
        // DF4b end-to-end: the resume-plan builder stamps the resumed lane's
        // `resume_instance_id` with the handoff entry's `instance_id` so the
        // dispatch source can reuse it. Fresh-start / pass-0 lanes stay `None`.
        let window = TStarWindow::default();
        let handoff = GraphWarmupHandoff {
            lanes: BTreeMap::from([(
                0u64,
                LaneHandoff {
                    template_trace_id: "t".to_owned(),
                    instance_id: "t::inst-0".to_owned(),
                    t_star_us: 0.0,
                    executed_node_ids: BTreeSet::from(["n_0".to_owned()]),
                    return_wall_us: BTreeMap::from([("n_0".to_owned(), 5.0)]),
                },
            )]),
            drain_end_wall_us: 5.0,
            corpus_cursor: 1,
            pressure_lane_count: 1,
        };
        let (prefix, _) = build_profiling_resume_lane_plans(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
            &handoff,
            Some(1),
            true,
        );
        assert_eq!(prefix.len(), 1);
        assert_eq!(prefix[0].resume_instance_id.as_deref(), Some("t::inst-0"));
    }

    #[test]
    fn build_graph_trace_source_resumes_from_corpus_cursor() {
        // DF3: with a handoff carrying corpus_cursor = k, the rebuilt profiling
        // source's first draw serves the k-th template (mod corpus), not the 0th,
        // so profiling continues where the pressure warmup left off.
        let plans = vec![
            pressure_one_node_plan("a"),
            pressure_one_node_plan("b"),
            pressure_one_node_plan("c"),
        ];
        let source = build_graph_trace_source(
            plans,
            Some(3),
            None,
            GraphTraceInstanceSequence::default(),
            // corpus_cursor = 4 over a 3-template corpus: first draw is 4 % 3 = 1.
            4,
            TStarWindow::default(),
        )
        .unwrap();
        let drawn: Vec<String> = std::iter::from_fn(|| {
            source
                .next_trace()
                .unwrap()
                .map(|plan| plan.trace.id.split_once("::").unwrap().0.to_owned())
        })
        .collect();
        // Continues b -> c -> a; the session budget (3) still counts from 0.
        assert_eq!(drawn, vec!["b", "c", "a"]);
    }

    #[test]
    fn build_graph_trace_source_zero_cursor_is_unchanged() {
        // HARD no-handoff guard: start_ordinal = 0 draws the 0th template first,
        // byte-identical to the pre-DF3 build.
        let plans = vec![pressure_one_node_plan("a"), pressure_one_node_plan("b")];
        let source = build_graph_trace_source(
            plans,
            Some(2),
            None,
            GraphTraceInstanceSequence::default(),
            0,
            TStarWindow::default(),
        )
        .unwrap();
        let drawn: Vec<String> = std::iter::from_fn(|| {
            source
                .next_trace()
                .unwrap()
                .map(|plan| plan.trace.id.split_once("::").unwrap().0.to_owned())
        })
        .collect();
        assert_eq!(drawn, vec!["a", "b"]);
    }
}
