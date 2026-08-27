// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundedness, backpressure, and identity contracts for the streaming
//! terminal record-processing lane.
#![cfg(feature = "streaming")]

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::rc::Rc;

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::metrics_core::RequestTrace;
use aiperf_runtime::multiturn::{ConversationSource, IssuedCredit, TurnToSend};
use aiperf_runtime::scheduled::{
    ModelResponseMetadata, ScheduledRuntime, ScheduledSessionIdentity, TurnDispatchOutcome,
    TurnDispatcher, TurnRecordProcessor,
};
use aiperf_runtime::scheduler::LocalTaskScheduler;
use aiperf_runtime::streaming::checkpoint::{
    CheckpointEpoch, CheckpointGeneration, StreamRunIdentity,
};
use aiperf_runtime::streaming::identity::{ContentDigest, LogicalReplayRunId};
use aiperf_runtime::streaming::reliability::{
    OrdinaryStreamingIssue, StreamingIssueComponentId, StreamingIssueReportError,
    StreamingIssueReportStatus, StreamingIssueReporterEndpoint, StreamingIssueReporterHandle,
    StreamingIssueScope, StreamingTerminalInvariant,
};
use aiperf_runtime::streaming::terminal_lane::{
    BoundedTerminalProcessorLane, TerminalBoundError, TerminalLaneError, TerminalLaneIssueScope,
    TerminalLaneLimits, TerminalRecordSizeBound, TerminalRecordSizeInputs, terminal_record_bytes,
};
use aiperf_runtime::timing::StopConfig;
use async_trait::async_trait;

mod common;

/// Lane byte capacity large enough for every declared bound used here.
const TEST_LANE_BYTES: usize = 1 << 20;

fn local<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("current-thread runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, future)
}

fn bound(bytes: usize) -> TerminalRecordSizeBound {
    TerminalRecordSizeBound::new(NonZeroUsize::new(bytes).expect("non-zero test bound"))
}

/// The declared, transport-enforced inputs every record bound here is proven from.
fn declared_inputs() -> TerminalRecordSizeInputs {
    TerminalRecordSizeInputs {
        max_response_body_bytes: Some(8_192),
        max_output_tokens: 0,
        max_bytes_per_output_token: 0,
        usage_metric_envelope_bytes: 512,
        max_request_retained_bytes: 8_192,
        captures_raw_payload: false,
    }
}

fn proven_bound() -> TerminalRecordSizeBound {
    declared_inputs().prove().expect("declared bound is finite")
}

/// Processor that fails a fixed number of times and counts every invocation.
struct CountingProcessor {
    seen: Cell<u64>,
    fail_first: u64,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for CountingProcessor {
    async fn process(
        &self,
        _credit: &IssuedCredit,
        _outcome: &TurnDispatchOutcome,
    ) -> anyhow::Result<()> {
        let seen = self.seen.get().saturating_add(1);
        self.seen.set(seen);
        if seen <= self.fail_first {
            anyhow::bail!("synthetic terminal processor fault");
        }
        Ok(())
    }
}

#[derive(Default)]
struct ExportIssueCounters {
    accepted_export: Cell<u64>,
}

struct RecordingIssueReporter {
    counters: Rc<ExportIssueCounters>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for RecordingIssueReporter {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if matches!(issue.scope(), StreamingIssueScope::Export { .. }) {
            self.counters
                .accepted_export
                .set(self.counters.accepted_export.get() + 1);
        }
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

/// Host-side recorder standing in for the Task 1D-R reliability owner.
struct ExportIssueRecorder {
    counters: Rc<ExportIssueCounters>,
}

impl ExportIssueRecorder {
    fn new() -> Self {
        Self {
            counters: Rc::new(ExportIssueCounters::default()),
        }
    }

    fn scope(&self) -> TerminalLaneIssueScope {
        TerminalLaneIssueScope::new(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([7; 32])),
            StreamingIssueComponentId::new("terminal_lane_test_exporter")
                .expect("checked component id"),
            CheckpointGeneration::new(CheckpointEpoch::new(1), ContentDigest::from_bytes([2; 32])),
            ContentDigest::from_bytes([3; 32]),
            StreamingIssueReporterHandle::new(RecordingIssueReporter {
                counters: Rc::clone(&self.counters),
            }),
        )
    }

    fn accepted_export_issues(&self) -> u64 {
        self.counters.accepted_export.get()
    }
}

/// Terminal-only dispatcher: no transport, no clock advance, no failure modes.
struct ImmediateDispatcher;

#[async_trait(?Send)]
impl TurnDispatcher for ImmediateDispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        _on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        Ok(completed_outcome(String::new()))
    }
}

fn completed_outcome(response_text: String) -> TurnDispatchOutcome {
    TurnDispatchOutcome {
        start_ns: 0,
        end_ns: 0,
        terminal: ReplayTerminalStatus::Completed,
        response_text,
        model_response: ModelResponseMetadata::default(),
        prompt_tokens: None,
        completion_tokens: None,
        http: RequestTrace::default(),
    }
}

/// One proven-bounded terminal record built from a real materialized turn.
struct TerminalRecord {
    credit: IssuedCredit,
    outcome: TurnDispatchOutcome,
    bound: TerminalRecordSizeBound,
}

async fn one_turn_source(turns: usize) -> Box<dyn ConversationSource> {
    let turn_objs = (0..turns.max(1))
        .map(|index| {
            serde_json::json!({
                "text": format!("terminal lane turn {index}"),
                "input_length": 4,
                "output_length": 1,
            })
        })
        .collect::<Vec<_>>();
    common::prepared_source_from_conversations(
        serde_json::json!([{"session_id": "terminal-lane", "turns": turn_objs}]),
        "terminal-lane-model",
        1,
    )
    .await
}

/// Build `count` terminal records whose retained bytes span a wide range.
async fn terminal_records(count: usize, response_sizes: &[usize]) -> Vec<TerminalRecord> {
    let mut source = one_turn_source(1).await;
    let bound = proven_bound();
    let mut records = Vec::with_capacity(count);
    for index in 0..count {
        let session = source
            .next(Some(format!("record-{index}")))
            .expect("sampled session");
        let turn = session.build_first_turn(None).expect("materialized turn");
        let credit = IssuedCredit::from_turn(index as u64, 0, &turn);
        let size = response_sizes[index % response_sizes.len()];
        records.push(TerminalRecord {
            credit,
            outcome: completed_outcome("r".repeat(size)),
            bound,
        });
    }
    records
}

/// Fixed-size observation of one scheduled-runtime harness run.
struct SessionObservation {
    completed_sessions: usize,
    active_session_entries: usize,
    active_url_index_entries: usize,
    next_session_number: u64,
    last_session_number: Option<u64>,
}

fn scheduled_harness() -> Rc<ScheduledRuntime> {
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ImmediateDispatcher);
    ScheduledRuntime::new(clock, start_ns, dispatcher, StopConfig::default(), false)
}

fn noop_completion() -> aiperf_runtime::scheduled::CompletionHandler {
    Box::new(|_credit, _outcome| Box::pin(async {}))
}

// ---------------------------------------------------------------------------
// Bounded lane
// ---------------------------------------------------------------------------

#[test]
fn record_count_does_not_increase_drain_task_count() {
    local(async {
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 4,
            max_bytes: 4096,
        })
        .expect("lane");
        let control = lane.control();
        lane.start_local_drain().expect("one drain owner");
        for index in 0..100_000_u64 {
            lane.submit_test_terminal(index, 1)
                .await
                .expect("bounded submit");
        }
        control.close();
        control.drain().await.expect("drain");
        let snapshot = control.snapshot();
        assert_eq!(snapshot.drain_tasks_started, 1);
        assert_eq!(snapshot.queued_items, 0);
        assert!(snapshot.high_water_items <= 4);
        assert_eq!(snapshot.processed_records, 100_000);
    });
}

#[test]
fn full_lane_reservation_waits_for_a_settled_record() {
    local(async {
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 1,
            max_bytes: 64,
        })
        .expect("lane");
        let control = lane.control();
        let held = control.reserve(bound(64)).await.expect("first reservation");
        let second = control.reserve(bound(64));
        tokio::pin!(second);
        // Both dimensions are exhausted, so the second reservation cannot
        // resolve while the first permit is alive.
        assert!(
            futures::poll!(second.as_mut()).is_pending(),
            "a full lane must not admit a second reservation"
        );
        drop(held);
        let recovered = second.await.expect("reservation after release");
        assert_eq!(recovered.bound().get(), 64);
        assert_eq!(control.snapshot().reserved_items, 1);
    });
}

#[test]
fn checked_invariant_latches_once_and_wakes_the_phase_owner() {
    local(async {
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 2,
            max_bytes: 256,
        })
        .expect("lane");
        let control = lane.control();
        lane.start_local_drain().expect("one drain owner");
        assert_eq!(control.checked_invariant(), None);
        let waiter = {
            let control = control.clone();
            tokio::task::spawn_local(async move { control.wait_for_invariant().await })
        };
        // A settlement larger than its own validated reservation is an
        // accounting contradiction, not an export fault.
        let permit = control.reserve(bound(8)).await.expect("reservation");
        let error = permit
            .settle_measured_for_test(9)
            .expect_err("oversized settlement must be refused");
        assert!(matches!(
            error,
            TerminalLaneError::ActualExceedsReservedBound {
                reserved_bytes: 8,
                actual_bytes: 9
            }
        ));
        assert_eq!(
            waiter.await.expect("waiter"),
            StreamingTerminalInvariant::AccountingCorruption
        );
        assert_eq!(
            control.checked_invariant(),
            Some(StreamingTerminalInvariant::AccountingCorruption)
        );
        // The refused permit returned its whole charge.
        assert_eq!(control.snapshot().reserved_bytes, 0);
        control.close();
        control.drain().await.expect("drain");
    });
}

#[test]
fn dropping_a_permit_returns_its_exact_charge() {
    local(async {
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 2,
            max_bytes: 128,
        })
        .expect("lane");
        let control = lane.control();
        let permit = control.reserve(bound(96)).await.expect("reservation");
        let charged = control.snapshot();
        assert_eq!(charged.reserved_items, 1);
        assert_eq!(charged.reserved_bytes, 96);
        drop(permit);
        let released = control.snapshot();
        assert_eq!(released.reserved_items, 0);
        assert_eq!(released.reserved_bytes, 0);
        assert_eq!(released.submitted_records, 0);
        // The full bound is immediately reusable.
        let reused = control.reserve(bound(128)).await.expect("reuse");
        assert_eq!(reused.bound().get(), 128);
    });
}

#[test]
fn ordinary_terminal_processor_error_reports_export_issue_and_drain_continues() {
    local(async {
        let reporter = ExportIssueRecorder::new();
        let lane = BoundedTerminalProcessorLane::new(
            TerminalLaneLimits {
                max_items: 4,
                max_bytes: TEST_LANE_BYTES,
            },
            reporter.scope(),
        )
        .expect("lane");
        let processor = Rc::new(CountingProcessor {
            seen: Cell::new(0),
            fail_first: 2,
        });
        lane.add_processor(processor.clone());
        let control = lane.control();
        lane.start_local_drain().expect("one drain owner");
        for record in terminal_records(5, &[0, 16, 64]).await {
            let permit = control.reserve(record.bound).await.expect("reservation");
            permit
                .settle(record.credit, record.outcome)
                .expect("settlement");
        }
        control.close();
        control.drain().await.expect("drain");
        let snapshot = control.snapshot();
        // Two faults, and every record still ran.
        assert_eq!(snapshot.ordinary_processor_failures, 2);
        assert_eq!(snapshot.processed_records, 5);
        assert_eq!(processor.seen.get(), 5);
        // Both faults reached the scoped reporter as export-scoped ordinary facts.
        assert_eq!(reporter.accepted_export_issues(), 2);
        // And NONE of them latched the invariant.
        assert_eq!(snapshot.checked_invariant, None);
        assert_eq!(snapshot.queued_items, 0);
        assert_eq!(snapshot.reserved_bytes, 0);
    });
}

#[test]
fn unbounded_terminal_payload_is_refused_before_dispatch() {
    let unbounded = TerminalRecordSizeInputs {
        max_response_body_bytes: None,
        max_output_tokens: 128,
        max_bytes_per_output_token: 4,
        usage_metric_envelope_bytes: 512,
        max_request_retained_bytes: 4_096,
        captures_raw_payload: false,
    };
    assert_eq!(
        unbounded.prove().expect_err("no enforced response limit"),
        TerminalBoundError::UnprovenResponseLimit
    );
    // A declared transport cap is sufficient, and the token product lowers it.
    let bounded = TerminalRecordSizeInputs {
        max_response_body_bytes: Some(1_000_000),
        ..unbounded
    };
    let proven = bounded.prove().expect("finite bound");
    assert!(proven.get() < 1_000_000);
    // Raw capture doubles the response term.
    let with_raw = TerminalRecordSizeInputs {
        captures_raw_payload: true,
        ..bounded
    };
    assert!(with_raw.prove().expect("finite bound").get() > proven.get());
}

#[test]
fn actual_terminal_bytes_never_exceed_reserved_bound() {
    local(async {
        const RECORDS: usize = 6;
        // No drain owner runs yet, so every settled record stays charged. That
        // is what makes the shrink observable: a lane that only released at
        // drain would still hold `RECORDS * bound`.
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: RECORDS,
            max_bytes: TEST_LANE_BYTES,
        })
        .expect("lane");
        let control = lane.control();
        let mut settled_bytes = 0usize;
        let mut reserved_if_unshrunk = 0usize;
        for record in terminal_records(RECORDS, &[0, 1, 512, 2_048]).await {
            let measured = terminal_record_bytes(&record.credit, &record.outcome);
            assert!(
                measured <= record.bound.get(),
                "measured {measured} exceeded proven bound {}",
                record.bound.get()
            );
            let permit = control.reserve(record.bound).await.expect("reservation");
            permit
                .settle(record.credit, record.outcome)
                .expect("settlement within bound");
            settled_bytes += measured;
            reserved_if_unshrunk += record.bound.get();
            // The lease shrank to the actual size at settlement, not at drain.
            let snapshot = control.snapshot();
            assert_eq!(snapshot.reserved_bytes, settled_bytes);
            assert!(settled_bytes < reserved_if_unshrunk);
        }
        lane.start_local_drain().expect("one drain owner");
        control.close();
        control.drain().await.expect("drain");
        assert_eq!(control.snapshot().checked_invariant, None);
    });
}

#[test]
fn terminal_lane_accounting_corruption_wakes_failed_run() {
    local(async {
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 2,
            max_bytes: 64,
        })
        .expect("lane");
        let control = lane.control();
        lane.start_local_drain().expect("one drain owner");
        let permit = control.reserve(bound(4)).await.expect("reservation");
        assert!(permit.settle_measured_for_test(5).is_err());
        control.close();
        control.drain().await.expect("drain");
        // The invariant survives the drain and is what the phase owner reads
        // BEFORE any report is constructed.
        assert_eq!(
            control.checked_invariant(),
            Some(StreamingTerminalInvariant::AccountingCorruption)
        );
        // Ordinary counters stayed clean: this was never an export fault.
        assert_eq!(control.snapshot().ordinary_processor_failures, 0);
    });
}

// ---------------------------------------------------------------------------
// Bounded session identity
// ---------------------------------------------------------------------------

#[test]
fn one_turn_sessions_leave_no_active_session_entries() {
    let report = local(async {
        let mut source = one_turn_source(1).await;
        let runtime = scheduled_harness();
        let start_ns = runtime.start_ns();
        let completed = Rc::new(Cell::new(0usize));
        for index in 0..100_000_usize {
            let session = source
                .next(Some(format!("session-{index}")))
                .expect("sampled session");
            let turn = session.build_first_turn(None).expect("materialized turn");
            let completed = Rc::clone(&completed);
            let admitted = runtime.issue_turn(
                turn,
                start_ns,
                None,
                Box::new(move |_credit, _outcome| {
                    completed.set(completed.get() + 1);
                    Box::pin(async {})
                }),
            );
            assert!(admitted, "issuance {index} was refused");
            if index % 1_024 == 1_023 {
                runtime.scheduler().wait_idle().await;
            }
        }
        runtime.scheduler().wait_idle().await;
        SessionObservation {
            completed_sessions: completed.get(),
            active_session_entries: runtime.active_session_count(),
            active_url_index_entries: runtime.active_url_index_count(),
            next_session_number: runtime.next_session_number(),
            last_session_number: runtime.last_session_number(),
        }
    });
    assert_eq!(report.completed_sessions, 100_000);
    // The lifetime map is gone; only in-flight sessions are retained.
    assert_eq!(report.active_session_entries, 0);
    assert_eq!(report.active_url_index_entries, 0);
    // The allocator still counted every session.
    assert_eq!(report.next_session_number, 100_000);
    assert_eq!(report.last_session_number, Some(99_999));
}

#[test]
fn streaming_identity_supplies_the_external_ordinal_without_a_map_entry() {
    let observed = local(async {
        let mut source = one_turn_source(1).await;
        let runtime = scheduled_harness();
        let start_ns = runtime.start_ns();
        let lane = BoundedTerminalProcessorLane::new_for_test(TerminalLaneLimits {
            max_items: 4,
            max_bytes: TEST_LANE_BYTES,
        })
        .expect("lane");
        runtime.install_terminal_lane(lane.control());
        lane.start_local_drain().expect("one drain owner");
        let session = source
            .next(Some("stream-session".to_string()))
            .expect("sampled session");
        let turn = session.build_first_turn(None).expect("materialized turn");
        let permit = runtime
            .reserve_terminal_processing(proven_bound())
            .await
            .expect("terminal reservation");
        let admitted = runtime.issue_turn_with_streaming_identity(
            turn,
            start_ns,
            None,
            ScheduledSessionIdentity {
                stable_ordinal: 4_242,
            },
            permit,
            Box::new(|_ttft_ns| {}),
            noop_completion(),
            None,
        );
        assert!(admitted);
        runtime.scheduler().wait_idle().await;
        let control = lane.control();
        control.close();
        control.drain().await.expect("drain");
        assert_eq!(control.snapshot().processed_records, 1);
        assert_eq!(control.snapshot().checked_invariant, None);
        SessionObservation {
            completed_sessions: 1,
            active_session_entries: runtime.active_session_count(),
            active_url_index_entries: runtime.active_url_index_count(),
            next_session_number: runtime.next_session_number(),
            last_session_number: runtime.last_session_number(),
        }
    });
    assert_eq!(observed.last_session_number, Some(4_242));
    assert_eq!(observed.active_session_entries, 0);
    // The finite allocator did not advance for a streaming issuance.
    assert_eq!(observed.next_session_number, 0);
}

#[test]
fn finite_session_numbers_are_unchanged_by_the_bounded_allocator() {
    // Two sessions, three turns each, interleaved: ordinals must be assigned in
    // first-sight order starting at zero, exactly as `sessions.len()` did.
    let (ordinals_a, ordinals_b, active_entries) = local(async {
        let mut source = one_turn_source(3).await;
        let runtime = scheduled_harness();
        let start_ns = runtime.start_ns();
        let sessions = ["a", "b"].map(|name| {
            source
                .next(Some(name.to_string()))
                .expect("sampled session")
        });
        let mut ordinals_a = Vec::new();
        let mut ordinals_b = Vec::new();
        for turn_index in 0..3 {
            for (slot, session) in sessions.iter().enumerate() {
                // Deferred turns carry the scheduling identity without
                // materializing a body. The prepared chat endpoint splices live
                // captured replies into context, so a jump-resume materialization
                // past turn 0 is refused by design; identity is all this pins.
                let turn = session
                    .build_deferred_turn(turn_index, None)
                    .expect("deferred turn identity");
                let admitted =
                    runtime.issue_turn(turn, start_ns, None, noop_completion());
                assert!(admitted);
                let ordinal = runtime
                    .last_session_number()
                    .expect("an admitted turn always reports an ordinal");
                if slot == 0 {
                    ordinals_a.push(ordinal);
                } else {
                    ordinals_b.push(ordinal);
                }
            }
            runtime.scheduler().wait_idle().await;
        }
        (ordinals_a, ordinals_b, runtime.active_session_count())
    });
    assert_eq!(ordinals_a, vec![0, 0, 0]);
    assert_eq!(ordinals_b, vec![1, 1, 1]);
    assert_eq!(active_entries, 0);
}
