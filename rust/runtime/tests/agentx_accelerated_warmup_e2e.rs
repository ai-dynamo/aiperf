// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic SimClock e2e for the `agentic_replay` accelerated cache-pressure
//! warmup + warmup→profiling residual handoff.
//!
//! Drives the REAL [`AgenticReplayWorkload::execute`] through the shared scheduled
//! runtime (`run_scheduled_workload`) against a virtual clock and a deterministic
//! streaming dispatcher, then asserts from the raw per-request dispatch stream that:
//!
//! (a) WARMUP records carry `max_tokens == 1` and fire at ZERO idle delay
//!     (compressed cache-pressure traffic);
//! (b) PROFILING resumes each live lane at its TRUE next turn index (NOT turn 0)
//!     at the residual dispatch offset carried across the handoff;
//! (c) a lane whose tree fully drained during warmup recycles a FRESH root at turn 0;
//! (d) aggregate metrics are present and the run completes cleanly (no slot panic).
//!
//! The two agentic phase instances share one accelerated-warmup carrier exactly as
//! the engine threads it (`WARMUP` writes it at finalize, `PROFILING` reads it), so
//! the residual frontier is produced by the production code path, not the test.


use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use aiperf_runtime::agentic_replay::{
    AgenticPhase, AgenticReplayConfig, AgenticReplayWorkload, new_warmup_handoff_carrier,
};
use aiperf_runtime::agentx::cache_bust::CacheBustTarget;
use aiperf_runtime::clock::Clock;
use aiperf_runtime::clock::sim_clock::SimClock;
use aiperf_runtime::dataset::{
    ComposeConfig, Dataset, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::endpoints::{
    EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
};
use aiperf_runtime::graph::runtime::drive_sim;
use aiperf_runtime::multiturn::{
    ConversationSource, NativeDatasetConversationSource, PreparedEndpointReference, TurnToSend,
};
use aiperf_runtime::phase_runtime::{ScheduledPhasePlan, run_scheduled_phases};
use aiperf_runtime::rng::RngRoot;
use aiperf_runtime::scheduled::{
    ModelResponseMetadata, ScheduledAncillaryPolicies as PhaseAncillary, ScheduledRunReport,
    TurnDispatchOutcome, TurnDispatcher, Workload, run_scheduled_workload,
};
use aiperf_runtime::timing::{
    GracePeriod, PhaseBranchStats, PhaseConfig, PhaseKind, PhaseObserver, PhaseStats, StopConfig,
};
use async_trait::async_trait;

/// No-op phase observer for the full phase-runner harness.
struct NoopPhaseObserver;
impl PhaseObserver for NoopPhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, _stats: PhaseStats) {}
    fn on_progress(&self, _stats: PhaseStats) {}
    fn on_sending_complete(&self, _stats: PhaseStats) {}
    fn on_phase_complete(&self, _stats: PhaseStats, _branch: Option<PhaseBranchStats>) {}
}

/// One observed dispatch: the identity + forced output cap + virtual dispatch instant.
#[derive(Clone, Debug)]
struct Seen {
    conversation_id: String,
    x_correlation_id: String,
    turn_index: usize,
    max_output_tokens: usize,
    dispatch_ms: f64,
}

/// A streaming dispatcher over the virtual clock: each token costs `ttft`/`itl`
/// virtual ns, so `max_output_tokens == 1` turns take exactly one TTFT step.
struct SimDispatcher {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    ttft_ns: i64,
    itl_ns: i64,
    seen: RefCell<Vec<Seen>>,
}

#[async_trait(?Send)]
impl TurnDispatcher for SimDispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        let start_ns = self.clock.now_ns();
        self.seen.borrow_mut().push(Seen {
            conversation_id: turn.conversation_id.clone(),
            x_correlation_id: turn.x_correlation_id.clone(),
            turn_index: turn.turn_index,
            max_output_tokens: turn.max_output_tokens,
            dispatch_ms: (start_ns - self.origin_ns) as f64 / 1_000_000.0,
        });
        observer.on_admit(
            turn.uuid,
            (start_ns - self.origin_ns) as f64 / 1_000_000.0,
            0,
        );
        self.clock.clone().sleep(self.ttft_ns).await;
        on_first_token(self.ttft_ns);
        observer.on_token(
            turn.uuid,
            (self.clock.now_ns() - self.origin_ns) as f64 / 1_000_000.0,
        );
        for _ in 1..turn.max_output_tokens {
            self.clock.clone().sleep(self.itl_ns).await;
            observer.on_token(
                turn.uuid,
                (self.clock.now_ns() - self.origin_ns) as f64 / 1_000_000.0,
            );
        }
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        Ok(TurnDispatchOutcome {
            start_ns,
            end_ns: self.clock.now_ns(),
            terminal: ReplayTerminalStatus::Completed,
            response_text: format!("reply-{}", turn.turn_index),
            model_response: ModelResponseMetadata::default(),
            prompt_tokens: None,
            completion_tokens: None,
            http: aiperf_runtime::metrics_core::RequestTrace::default(),
        })
    }
}

/// Build a self-contained (MessageArrayWithResponses) multi-turn source from the
/// `single_turn` loader: rows sharing a `session_id` compose into one conversation
/// whose per-turn context is reproducible up front — the property `build_turn_at(k)`
/// jump-resume requires. `delay` on a row becomes that turn's recorded `delay_ms`.
async fn multiturn_source(rows: serde_json::Value, model: &str) -> Box<dyn ConversationSource> {
    let dataset: Dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("single_turn"),
            &LoadConfig::new(DatasetSource::Inline(rows)),
            &ComposeConfig::new(model, RngRoot::new(Some(1))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let endpoint = EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new("chat").unwrap(),
            RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = PreparedEndpointTable::new();
    let key = table.push(endpoint).unwrap();
    Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            model,
            4,
            Rc::new(table),
            PreparedEndpointReference {
                key,
                endpoint_id: EndpointId::new("chat").unwrap(),
            },
        )
        .unwrap(),
    )
}

fn block_on_source(rows: serde_json::Value) -> Box<dyn ConversationSource> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(multiturn_source(rows, "test-model"))
}

fn config(
    phase: AgenticPhase,
    carrier: aiperf_runtime::agentic_replay::WarmupHandoffCarrier,
    cache_warmup_duration_s: Option<f64>,
    max_tokens_override: Option<u32>,
) -> AgenticReplayConfig {
    AgenticReplayConfig {
        phase,
        start_min_ratio: 0.0,
        start_max_ratio: 1.0,
        idle_gap_cap_ms: None,
        burst_phase_starts: false,
        random_seed: 1,
        benchmark_id: "bench".into(),
        cache_bust_target: CacheBustTarget::FirstTurnPrefix,
        trees: Rc::new(Vec::new()),
        cache_warmup_duration_s,
        max_tokens_override,
        warmup_handoff: carrier,
    }
}

fn run(
    clock: Rc<SimClock>,
    workload: Rc<dyn Workload>,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
    enforce_stop: bool,
) -> ScheduledRunReport {
    let report = Rc::new(RefCell::new(None));
    let out = report.clone();
    let clock_for_body = clock.clone();
    let outcome = drive_sim(clock, move |_handle| async move {
        let clock_dyn: Rc<dyn Clock> = clock_for_body;
        let r = run_scheduled_workload(workload, clock_dyn, 0, dispatcher, stop, enforce_stop)
            .await
            .unwrap();
        *out.borrow_mut() = Some(r);
    });
    assert!(!outcome.deadlocked, "workload must drain");
    Rc::try_unwrap(report).ok().unwrap().into_inner().unwrap()
}

/// Two five-turn live lanes ("a","b") and one single-turn lane ("c"). Rows 1.. carry
/// a recorded 100 ms inter-turn delay so the handoff residual is non-zero.
fn dataset_rows() -> serde_json::Value {
    let lane = |id: &str, turns: usize| -> Vec<serde_json::Value> {
        (0..turns)
            .map(|i| {
                let mut row = serde_json::json!({
                    "session_id": id,
                    "text": format!("{id} q{i} alpha beta gamma"),
                    "output_length": 4,
                });
                if i > 0 {
                    row["delay"] = serde_json::json!(100.0);
                }
                row
            })
            .collect()
    };
    let mut rows = lane("a", 5);
    rows.extend(lane("b", 5));
    rows.extend(lane("c", 1));
    serde_json::Value::Array(rows)
}

#[test]
fn accelerated_warmup_pressure_then_profiling_resumes_at_residual_frontier() {
    // Shared cross-phase carrier: WARMUP populates it, PROFILING consumes it.
    let carrier: aiperf_runtime::agentic_replay::WarmupHandoffCarrier =
        new_warmup_handoff_carrier();

    // ---- WARMUP: accelerated cache-pressure substage (max_tokens=1, zero idle). ----
    let warmup_seen = {
        let source = block_on_source(dataset_rows());
        let cfg = config(AgenticPhase::Warmup, carrier.clone(), Some(0.015), Some(1));
        let workload: Rc<dyn Workload> = Rc::new(AgenticReplayWorkload::new(source, cfg).unwrap());
        let clock = Rc::new(SimClock::new());
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        let dispatcher = Rc::new(SimDispatcher {
            clock: clock_dyn,
            origin_ns: clock.now_ns(),
            ttft_ns: 10_000_000, // 10 ms / token
            itl_ns: 5_000_000,
            seen: RefCell::new(Vec::new()),
        });
        let seen_handle = dispatcher.clone();
        let report = run(clock, workload, dispatcher, StopConfig::default(), false);
        // (d) aggregate metrics present for the warmup phase.
        assert!(!report.turns.is_empty(), "warmup produced timing records");
        seen_handle.seen.borrow().clone()
    };

    // (a) Every warmup dispatch forces max_tokens == 1; each lane's first turn fires
    //     at zero idle (t == 0). Lanes a,b advance under compression; c is single-turn.
    assert!(!warmup_seen.is_empty(), "warmup dispatched pressure turns");
    for s in &warmup_seen {
        assert_eq!(
            s.max_output_tokens, 1,
            "warmup pressure forces max_tokens=1: {s:?}"
        );
    }
    for id in ["a", "b", "c"] {
        let first = warmup_seen
            .iter()
            .find(|s| s.conversation_id == id)
            .unwrap();
        assert_eq!(first.turn_index, 0, "lane {id} pressure starts at turn 0");
        assert!(
            first.dispatch_ms.abs() < 1e-6,
            "lane {id} first pressure turn at zero idle"
        );
    }
    // Lanes a,b advanced past turn 0 under compression (proves live-trajectory replay).
    assert!(
        warmup_seen
            .iter()
            .any(|s| s.conversation_id == "a" && s.turn_index >= 1)
    );
    assert!(
        warmup_seen
            .iter()
            .any(|s| s.conversation_id == "b" && s.turn_index >= 1)
    );

    // The carrier now holds the drained frontier the WARMUP finalize produced. These
    // assertions read the production `finalize` output directly (the load-bearing
    // handoff): resume indices, residual offsets, and the recycled empty lane.
    let handoff = carrier
        .lock()
        .unwrap()
        .clone()
        .expect("warmup populated the carrier");
    assert_eq!(
        handoff.lanes.len(),
        3,
        "one handoff lane per live trajectory"
    );
    // (b, carrier) Lanes a,b survived mid-flight and resume at their TRUE next turn
    //     index (> 0) carrying a strictly-positive residual dispatch offset.
    let mut surviving = 0usize;
    let mut max_offset_ms = 0.0f64;
    for lane in handoff.lanes.values() {
        for st in &lane.states {
            if st.x_correlation_id == "a" || st.x_correlation_id == "b" {
                assert!(
                    st.next_turn_index >= 2,
                    "surviving lane resumes past turn 0 (drained frontier)"
                );
                assert!(
                    st.next_dispatch_offset_ms > 0.0,
                    "surviving lane carries a residual offset"
                );
                surviving += 1;
            }
            max_offset_ms = max_offset_ms.max(st.next_dispatch_offset_ms);
        }
    }
    assert_eq!(surviving, 2, "lanes a and b both survived the warmup drain");
    // (c, carrier) The drained single-turn lane recycled a FRESH root (turn 0, a
    //     `#w` recycle correlation, zero offset) rather than a surviving resume.
    let recycled_state = handoff
        .lanes
        .values()
        .flat_map(|l| &l.states)
        .find(|s| s.x_correlation_id.contains("#w"))
        .expect("drained lane recycled a fresh root");
    assert_eq!(
        recycled_state.next_turn_index, 0,
        "recycled root starts at turn 0"
    );
    assert_eq!(
        recycled_state.next_dispatch_offset_ms, 0.0,
        "recycled root has zero residual offset"
    );

    // ---- PROFILING: drive the REAL resume dispatch off the carrier. ----
    // Turns are made effectively non-terminating within the observation window (10 s
    // virtual each) and the duration budget is set just past the largest residual
    // offset, so exactly the initial resume/recycle dispatches fire — cleanly, with
    // no lane completing into a recycle the bare harness could not cancel.
    let deadline_ns = ((max_offset_ms + 20.0) * 1_000_000.0) as i64;
    let profiling_seen = {
        let source = block_on_source(dataset_rows());
        let cfg = config(AgenticPhase::Profiling, carrier.clone(), None, None);
        let workload: Rc<dyn Workload> = Rc::new(AgenticReplayWorkload::new(source, cfg).unwrap());
        let clock = Rc::new(SimClock::new());
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        let dispatcher = Rc::new(SimDispatcher {
            clock: clock_dyn,
            origin_ns: clock.now_ns(),
            ttft_ns: 10_000_000_000, // 10 s TTFT: no turn completes inside the window
            itl_ns: 1_000_000_000,
            seen: RefCell::new(Vec::new()),
        });
        let seen_handle = dispatcher.clone();
        // Drive through the FULL phase runner so the phase's duration deadline cancels
        // pending issuance (the production agentic termination path), rather than the
        // bare workload driver which cannot cancel scheduled recycles at a time bound.
        let phase = PhaseConfig::new(
            "profiling",
            PhaseKind::Profiling,
            StopConfig {
                expected_duration_ns: Some(deadline_ns),
                ..StopConfig::default()
            },
        )
        .with_grace_period(GracePeriod::Disabled)
        .with_runtime_intervals(1_000_000, 1_000_000);
        // Agentic phases own their dispatch timing (enforce_stop=false, exactly as
        // `lower_legacy_agentic` lowers them); the phase's duration+grace lifecycle
        // cancels pending issuance at the deadline.
        let plan = ScheduledPhasePlan::new(phase, workload, PhaseAncillary::default())
            .with_enforce_stop(false);
        let dispatcher_dyn: Rc<dyn TurnDispatcher> = dispatcher;
        let clock_dyn2: Rc<dyn Clock> = clock.clone();
        let observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
        let phase_report = Rc::new(RefCell::new(None));
        let out = phase_report.clone();
        let outcome = drive_sim(clock, move |_h| async move {
            let r = run_scheduled_phases(vec![plan], clock_dyn2, dispatcher_dyn, observer)
                .await
                .unwrap();
            *out.borrow_mut() = Some(r);
        });
        assert!(!outcome.deadlocked, "profiling phase must drain");
        // (d) aggregate metrics present and clean completion (no slot panic reached here).
        let report = Rc::try_unwrap(phase_report)
            .ok()
            .unwrap()
            .into_inner()
            .unwrap();
        assert_eq!(report.reports.len(), 1, "one profiling phase report");
        seen_handle.seen.borrow().clone()
    };

    // (b) Real resume dispatch: lanes a,b fire at turn index 2 (NOT 0), at a residual
    //     offset, under their original correlation id, with the recorded output cap.
    for id in ["a", "b"] {
        let resume = profiling_seen
            .iter()
            .find(|s| s.conversation_id == id && s.x_correlation_id == id)
            .unwrap_or_else(|| panic!("lane {id} resumed in profiling"));
        assert_eq!(
            resume.turn_index, 2,
            "lane {id} resumes at recorded next turn index 2, not 0"
        );
        assert!(
            resume.dispatch_ms > 1.0,
            "lane {id} resumes at residual offset, got {}",
            resume.dispatch_ms
        );
        assert_eq!(
            resume.max_output_tokens, 4,
            "profiling restores the recorded output cap"
        );
    }
    // (c) Real recycle dispatch: the drained lane's fresh root fires at turn 0, ~zero offset.
    let recycled = profiling_seen
        .iter()
        .find(|s| s.x_correlation_id.contains("#w"))
        .expect("drained lane recycled a fresh root in profiling");
    assert_eq!(
        recycled.turn_index, 0,
        "recycled drained lane starts a fresh root at turn 0"
    );
    assert!(
        recycled.dispatch_ms.abs() < 1.0,
        "recycled fresh root fires at ~zero offset"
    );
}

/// Regression guard: with NO accelerated duration and an empty carrier, the profiling
/// path is byte-for-byte the non-accelerated behavior — every lane starts at turn 0.
#[test]
fn non_accelerated_profiling_is_unchanged_starts_at_turn_zero() {
    let empty_carrier: aiperf_runtime::agentic_replay::WarmupHandoffCarrier =
        Arc::new(Mutex::new(None));
    let source = block_on_source(dataset_rows());
    let cfg = config(AgenticPhase::Profiling, empty_carrier, None, None);
    let workload: Rc<dyn Workload> = Rc::new(AgenticReplayWorkload::new(source, cfg).unwrap());
    let clock = Rc::new(SimClock::new());
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let dispatcher = Rc::new(SimDispatcher {
        clock: clock_dyn,
        origin_ns: clock.now_ns(),
        ttft_ns: 10_000_000,
        itl_ns: 5_000_000,
        seen: RefCell::new(Vec::new()),
    });
    let seen_handle = dispatcher.clone();
    let stop = StopConfig {
        expected_duration_ns: Some(50_000_000),
        ..StopConfig::default()
    };
    let _ = run(clock, workload, dispatcher, stop, true);
    let seen = seen_handle.seen.borrow();
    // Every lane's FIRST dispatch is turn 0 (no carrier resume).
    for id in ["a", "b", "c"] {
        let first = seen.iter().find(|s| s.conversation_id == id);
        if let Some(first) = first {
            assert_eq!(
                first.turn_index, 0,
                "non-accelerated lane {id} starts at turn 0"
            );
        }
    }
    let _keep: Cell<usize> = Cell::new(0);
}
