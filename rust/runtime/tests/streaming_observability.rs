// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "streaming")]

//! Streaming-plane observability: bounded stage-boundary recording, an
//! associative cross-shard merge, and a feature-independent report projection.
//!
//! Timing assertions are exact rather than tolerance-bounded, which is only
//! possible because every observation is taken from a [`SimClock`] reading.

use std::{collections::BTreeMap, rc::Rc};

use aiperf_runtime::{
    clock::{Clock, SimClock},
    metrics_core::{NativeReport, NativeReportInput, NativeReporter, Reporter, RunOutcome},
    streaming::{
        budget::BudgetSnapshot,
        checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, DecodeHorizon, DiscoveryHorizon,
            EventTimeWatermark, OrderedActionHorizon, TerminalActionHorizon,
        },
        failure::StreamingFailureStage,
        identity::{ContentDigest, GlobalSequence, SessionCausalFrontier},
        observability::{
            CheckpointHorizonSnapshot, STREAMING_DROP_REASONS, STREAMING_STAGES,
            ScheduledActionHorizon, StreamingDropReason, StreamingPlaneMetrics,
            StreamingPlaneObserver, StreamingStage, stage_for_failure,
        },
        reliability::{
            StreamingIssueClass, StreamingIssueDisposition, StreamingIssueScopeKind,
            StreamingIssueSummary,
        },
        unit::SourcePosition,
    },
};

fn observer(clock: &Rc<SimClock>) -> StreamingPlaneObserver {
    StreamingPlaneObserver::new(Rc::clone(clock) as Rc<dyn Clock>)
}

/// Advance virtual time by an exact number of nanoseconds.
fn advance(clock: &Rc<SimClock>, ns: i64) {
    clock.advance_to(clock.now_ns() + ns);
}

fn cut_through(terminal: u64) -> CheckpointCut {
    let position = SourcePosition::new(terminal);
    CheckpointCut {
        discovered: DiscoveryHorizon::new(position),
        acquired: AcquisitionHorizon::new(position),
        decoded: DecodeHorizon::new(position),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(terminal)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(terminal)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(terminal)),
        event_watermark: EventTimeWatermark::Unknown,
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(terminal),
            event_time: None,
            digest: ContentDigest::from_bytes([0; 32]),
        },
        handled_issues: aiperf_runtime::streaming::reliability::HandledIssueCut::empty(),
    }
}

fn horizons(terminal: u64, scheduled: u64) -> CheckpointHorizonSnapshot {
    CheckpointHorizonSnapshot {
        cut: cut_through(terminal),
        scheduled: ScheduledActionHorizon::new(GlobalSequence::new(scheduled)),
    }
}

/// One action's worth of stage transitions, each with a pinned virtual duration.
fn record_one_action(
    recorder: &mut StreamingPlaneObserver,
    clock: &Rc<SimClock>,
    endpoint_ns: i64,
    slip_ns: i64,
) {
    let span = recorder.open_span(StreamingStage::Action);
    advance(clock, slip_ns);
    recorder.close_span(span);

    let span = recorder.open_span(StreamingStage::Terminal);
    advance(clock, endpoint_ns);
    recorder.close_span(span);
}

#[test]
fn stage_metrics_separate_lag_wait_slip_and_endpoint_time() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    // Each stage gets a distinct pinned duration, so a metric folded into the
    // wrong distribution is visible as a wrong value rather than a wrong count.
    for (stage, duration_ns) in [
        (StreamingStage::Source, 11),
        (StreamingStage::Acquire, 22),
        (StreamingStage::Decode, 33),
        (StreamingStage::Session, 44),
        (StreamingStage::Placement, 55),
    ] {
        let span = recorder.open_span(stage);
        advance(&clock, duration_ns);
        recorder.close_span(span);
    }
    record_one_action(&mut recorder, &clock, 900, 7);

    recorder.observe_queue(
        StreamingStage::Acquire,
        BudgetSnapshot {
            used_items: 1,
            used_bytes: 64,
            high_water_items: 2,
            high_water_bytes: 128,
        },
        4,
        256,
    );
    recorder.refresh_boundary(StreamingIssueSummary::empty(), horizons(0, 1));

    let metrics = recorder.snapshot();
    assert_eq!(metrics.schedule_slip_ns.totals.count, 1);
    assert_eq!(metrics.schedule_slip_ns.totals.sum_ns, 7);
    assert_eq!(metrics.endpoint_ns.totals.count, 1);
    assert_eq!(metrics.endpoint_ns.totals.sum_ns, 900);
    assert_eq!(metrics.publication_lag_ns.totals.sum_ns, 11);
    assert_eq!(metrics.acquisition_duration_ns.totals.sum_ns, 22);
    assert_eq!(metrics.decode_duration_ns.totals.sum_ns, 33);
    assert_eq!(metrics.causal_wait_ns.totals.sum_ns, 44);
    assert_eq!(metrics.admission_wait_ns.totals.sum_ns, 55);

    for queue in metrics.queues.values() {
        assert!(
            queue.is_within_limits(),
            "high water {queue:?} exceeded its authored limit"
        );
    }
    let cut = &metrics
        .checkpoint_horizons
        .as_ref()
        .expect("boundary installs the horizons")
        .cut;
    assert_eq!(
        cut.terminal,
        TerminalActionHorizon::new(GlobalSequence::new(0))
    );
}

#[test]
fn observability_separates_failed_action_from_failed_run() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    // One endpoint failure that terminalizes exactly one action: truthful
    // terminal membership for that action, and no run-level disposition.
    record_one_action(&mut recorder, &clock, 500, 0);
    recorder.observe_failed_terminal_action();

    let mut issues = StreamingIssueSummary::empty();
    issues.total = 1;
    issues
        .by_disposition
        .insert(StreamingIssueDisposition::TerminalActionReceipt, 1);
    issues
        .by_scope
        .insert(StreamingIssueScopeKind::Action, 1);
    issues.by_class.insert(StreamingIssueClass::Permanent, 1);
    recorder.refresh_boundary(issues, horizons(1, 1));

    let metrics = recorder.snapshot();
    assert_eq!(metrics.failed_terminal_actions, 1);
    assert_eq!(
        metrics
            .issues
            .by_disposition
            .get(&StreamingIssueDisposition::TerminalActionReceipt),
        Some(&1)
    );
    assert_eq!(
        metrics
            .issues
            .by_disposition
            .get(&StreamingIssueDisposition::FailRun),
        None,
        "a failed action must not be reported as a failed run"
    );
    assert!(!metrics.issues.is_admission_fenced);
    // The run's own progress is sealed through the failed action's sequence.
    assert_eq!(
        metrics
            .checkpoint_horizons
            .expect("horizons installed")
            .cut
            .terminal,
        TerminalActionHorizon::new(GlobalSequence::new(1))
    );
}

#[test]
fn distributions_are_exactly_correct_under_sim_clock() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    for endpoint_ns in [100, 250, 175, 1_000] {
        record_one_action(&mut recorder, &clock, endpoint_ns, 1);
    }

    let endpoint = recorder.snapshot().endpoint_ns;
    assert_eq!(endpoint.totals.count, 4);
    assert_eq!(endpoint.totals.sum_ns, 100 + 250 + 175 + 1_000);
    assert_eq!(endpoint.totals.max_ns, 1_000);
    assert_eq!(endpoint.sketch.count(), 4);
}

/// Build three deliberately dissimilar shards so an order-dependent merge shows
/// up as a difference rather than as a coincidence.
fn shards() -> [StreamingPlaneMetrics; 3] {
    let clock = Rc::new(SimClock::new());

    let mut first = observer(&clock);
    record_one_action(&mut first, &clock, 100, 3);
    first.observe_drop(StreamingDropReason::Late);
    first.observe_retry(1);
    first.refresh_boundary(StreamingIssueSummary::empty(), horizons(2, 4));

    let mut second = observer(&clock);
    record_one_action(&mut second, &clock, 700, 5);
    second.observe_drop(StreamingDropReason::Duplicate);
    second.observe_gap();
    second.observe_failed_terminal_action();
    let mut fenced = StreamingIssueSummary::empty();
    fenced.total = 2;
    fenced.is_admission_fenced = true;
    fenced.by_class.insert(StreamingIssueClass::Capacity, 2);
    second.refresh_boundary(fenced, horizons(9, 9));

    let mut third = observer(&clock);
    record_one_action(&mut third, &clock, 40, 11);
    third.observe_drop(StreamingDropReason::Overload);
    third.observe_retry(2);
    third.observe_incomplete_derived_sink("parquet_records");
    third.refresh_boundary(StreamingIssueSummary::empty(), horizons(5, 6));

    [first.snapshot(), second.snapshot(), third.snapshot()]
}

fn merged(order: [usize; 3], shards: &[StreamingPlaneMetrics; 3]) -> StreamingPlaneMetrics {
    let mut folded = StreamingPlaneMetrics::default();
    for index in order {
        folded.merge(&shards[index]);
    }
    folded
}

#[test]
fn merge_is_associative_and_commutative_across_shards() {
    let shards = shards();
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let reference = merged(orders[0], &shards);
    assert_eq!(reference.endpoint_ns.totals.count, 3);
    assert_eq!(reference.endpoint_ns.totals.sum_ns, 100 + 700 + 40);
    assert_eq!(reference.duplicate_count, 1);
    assert_eq!(reference.gap_count, 1);
    assert_eq!(reference.failed_terminal_actions, 1);
    assert_eq!(reference.retry_ordinals, BTreeMap::from([(1, 1), (2, 1)]));

    for order in orders {
        let folded = merged(order, &shards);
        assert_eq!(
            folded.endpoint_ns.totals, reference.endpoint_ns.totals,
            "endpoint totals depend on reduce order {order:?}"
        );
        assert_eq!(folded.drops_by_reason, reference.drops_by_reason);
        assert_eq!(folded.retry_ordinals, reference.retry_ordinals);
        assert_eq!(folded.issues, reference.issues);
        assert_eq!(
            folded.incomplete_derived_sinks,
            reference.incomplete_derived_sinks
        );
        // The greatest committed cut wins in every order.
        assert_eq!(
            folded
                .checkpoint_horizons
                .as_ref()
                .expect("a shard installed horizons")
                .cut
                .terminal,
            TerminalActionHorizon::new(GlobalSequence::new(9)),
            "horizon reduce order {order:?} did not take the greatest cut"
        );
    }
}

#[test]
fn queue_high_water_never_exceeds_the_authored_limit() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    // A saturated stage: the peak equals the limit exactly, which is what proves
    // the recorder observes real saturation rather than always reporting slack.
    recorder.observe_queue(
        StreamingStage::Session,
        BudgetSnapshot {
            used_items: 8,
            used_bytes: 4_096,
            high_water_items: 8,
            high_water_bytes: 4_096,
        },
        8,
        4_096,
    );
    // A later, smaller sample must not walk the peak backwards.
    recorder.observe_queue(
        StreamingStage::Session,
        BudgetSnapshot {
            used_items: 1,
            used_bytes: 16,
            high_water_items: 1,
            high_water_bytes: 16,
        },
        8,
        4_096,
    );
    recorder.observe_queue(
        StreamingStage::Acquire,
        BudgetSnapshot {
            used_items: 2,
            used_bytes: 32,
            high_water_items: 3,
            high_water_bytes: 64,
        },
        16,
        1_024,
    );

    let metrics = recorder.snapshot();
    assert!(metrics.queues.values().all(|queue| queue.is_within_limits()));
    let session = metrics.queues[&StreamingStage::Session];
    assert_eq!(session.items, session.item_limit);
    assert_eq!(session.bytes, session.byte_limit);
}

#[test]
fn drop_reasons_are_distinct_and_exhaustive() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    for reason in STREAMING_DROP_REASONS {
        recorder.observe_drop(reason);
    }

    let metrics = recorder.snapshot();
    assert_eq!(metrics.drops_by_reason.len(), STREAMING_DROP_REASONS.len());
    assert!(metrics.drops_by_reason.values().all(|count| *count == 1));
    assert_eq!(
        metrics.duplicate_count,
        metrics.drops_by_reason[&StreamingDropReason::Duplicate],
        "duplicate_count and its drop bucket must agree"
    );
}

#[test]
fn retry_ordinals_are_counted_at_the_disposition_boundary() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    for ordinal in 1..=3 {
        recorder.observe_retry(ordinal);
    }
    assert_eq!(
        recorder.snapshot().failed_terminal_actions,
        0,
        "a retry is telemetry, not terminal membership"
    );

    recorder.observe_failed_terminal_action();
    let metrics = recorder.snapshot();
    assert_eq!(
        metrics.retry_ordinals,
        BTreeMap::from([(1, 1), (2, 1), (3, 1)])
    );
    assert_eq!(metrics.failed_terminal_actions, 1);
}

#[test]
fn admission_fence_state_is_reported_and_survives_merge() {
    let clock = Rc::new(SimClock::new());

    let mut open = observer(&clock);
    open.refresh_issues(StreamingIssueSummary::empty());
    let open = open.snapshot();

    let mut fenced_summary = StreamingIssueSummary::empty();
    fenced_summary.is_admission_fenced = true;
    let mut fenced = observer(&clock);
    fenced.refresh_issues(fenced_summary);
    let fenced = fenced.snapshot();

    let mut forward = open.clone();
    forward.merge(&fenced);
    let mut backward = fenced.clone();
    backward.merge(&open);

    assert!(forward.issues.is_admission_fenced);
    assert!(backward.issues.is_admission_fenced);
}

#[test]
fn failure_stages_map_onto_observable_stages() {
    // Both taxonomies are retained deliberately; the mapping is what lets a
    // failure count and a latency distribution be read side by side.
    assert_eq!(
        stage_for_failure(StreamingFailureStage::Acquisition),
        StreamingStage::Acquire
    );
    assert_eq!(
        stage_for_failure(StreamingFailureStage::StateBudget),
        StreamingStage::Session
    );
    assert_eq!(
        stage_for_failure(StreamingFailureStage::Dispatch),
        StreamingStage::Action
    );
    assert_eq!(
        stage_for_failure(StreamingFailureStage::Checkpoint),
        StreamingStage::Result
    );
}

#[test]
fn report_shape_is_identical_with_and_without_the_streaming_section() {
    let summary = aiperf_runtime::metrics_core::AccumulatorSummary::default();
    let plain = NativeReporter.report(NativeReportInput {
        metrics: &summary,
        outcome: &RunOutcome::default(),
    });
    let plain_json = serde_json::to_string(&plain).expect("report serializes");
    assert!(
        !plain_json.contains("streaming"),
        "an absent streaming section must not appear in the serialized report"
    );

    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);
    record_one_action(&mut recorder, &clock, 321, 4);
    recorder.observe_drop(StreamingDropReason::Late);
    recorder.refresh_boundary(StreamingIssueSummary::empty(), horizons(1, 2));

    let outcome = RunOutcome {
        streaming: Some(recorder.snapshot().to_report()),
        ..RunOutcome::default()
    };
    let with_streaming: NativeReport = NativeReporter.report(NativeReportInput {
        metrics: &summary,
        outcome: &outcome,
    });
    let section = with_streaming
        .streaming
        .as_ref()
        .expect("the section is present");
    assert_eq!(section.distributions["endpoint_ns"].sum_ns, 321);
    assert_eq!(section.distributions["schedule_slip_ns"].count, 1);
    assert_eq!(section.drops_by_reason["late"], 1);
    assert_eq!(
        section.horizons.expect("horizons project").terminal,
        1,
        "the committed terminal horizon must survive the projection"
    );
    // The only difference between the two reports is the added section.
    assert_eq!(
        NativeReport {
            streaming: None,
            ..with_streaming
        },
        plain
    );
}

#[test]
fn snapshot_size_is_bounded_by_closed_enums() {
    let clock = Rc::new(SimClock::new());
    let mut recorder = observer(&clock);

    for index in 0..100_000u64 {
        recorder.observe_stage_ns(
            STREAMING_STAGES[(index % STREAMING_STAGES.len() as u64) as usize],
            index % 997,
        );
        recorder.observe_drop(
            STREAMING_DROP_REASONS[(index % STREAMING_DROP_REASONS.len() as u64) as usize],
        );
        recorder.observe_queue(
            STREAMING_STAGES[(index % STREAMING_STAGES.len() as u64) as usize],
            BudgetSnapshot {
                used_items: 1,
                used_bytes: 1,
                high_water_items: 1,
                high_water_bytes: 1,
            },
            8,
            8,
        );
    }

    let metrics = recorder.snapshot();
    assert_eq!(metrics.queues.len(), STREAMING_STAGES.len());
    assert_eq!(metrics.drops_by_reason.len(), STREAMING_DROP_REASONS.len());
    // Every observation landed, and the digest stayed bounded rather than
    // retaining one centroid per value.
    let observed: u64 = STREAMING_STAGES
        .iter()
        .map(|stage| metrics.distribution(*stage).count())
        .sum();
    assert_eq!(observed, 100_000);
    let report = metrics.to_report();
    assert!(report.distributions.len() <= STREAMING_STAGES.len());
    assert!(report.queues.len() <= STREAMING_STAGES.len());
    let encoded = serde_json::to_vec(&metrics).expect("snapshot serializes");
    assert!(
        encoded.len() < 200_000,
        "snapshot grew to {} bytes, which is not a compile-time bound",
        encoded.len()
    );
}
