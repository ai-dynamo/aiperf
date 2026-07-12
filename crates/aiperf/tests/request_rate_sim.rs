// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic request-rate multi-turn policy proofs over `SimClock`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use aiperf::multiturn::{
    ConversationDataset, ConversationSource, DatasetConversationSource,
    SyntheticConversationSource, TurnToSend,
};
use aiperf::request_rate::{RequestRateConfig, RequestRateWorkload};
use aiperf::scheduled::{
    ScheduledRunReport, TurnDispatchOutcome, TurnDispatcher, Workload, run_scheduled_workload,
};
use aiperf::workload::SkeletonWorkload;
use aiperf_clock::Clock;
use aiperf_clock::sim_clock::SimClock;
use aiperf_graph::runtime::drive_sim;
use aiperf_timing::{ArrivalPattern, StopConfig};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;

#[derive(Clone, Debug, PartialEq, Eq)]
struct SeenTurn {
    conversation_id: String,
    turn_index: usize,
    roles: Vec<String>,
    issued_ns: i64,
}

struct SimDispatcher {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    ttft_ns: Option<i64>,
    decode_ns: i64,
    active: Cell<usize>,
    max_active: Cell<usize>,
    seen: RefCell<Vec<SeenTurn>>,
}

struct FailingDispatcher {
    clock: Rc<dyn Clock>,
    delay_ns: i64,
}

impl SimDispatcher {
    fn new(clock: Rc<dyn Clock>, ttft_ns: Option<i64>, decode_ns: i64) -> Rc<Self> {
        Rc::new(Self {
            origin_ns: clock.now_ns(),
            clock,
            ttft_ns,
            decode_ns,
            active: Cell::new(0),
            max_active: Cell::new(0),
            seen: RefCell::new(Vec::new()),
        })
    }
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
        let active = self.active.get() + 1;
        self.active.set(active);
        self.max_active.set(self.max_active.get().max(active));
        self.seen.borrow_mut().push(SeenTurn {
            conversation_id: turn.conversation_id.clone(),
            turn_index: turn.turn_index,
            roles: turn
                .messages
                .iter()
                .map(|message| message.role.clone())
                .collect(),
            issued_ns: start_ns,
        });
        observer.on_admit(
            turn.uuid,
            (start_ns - self.origin_ns) as f64 / 1_000_000.0,
            0,
        );
        if let Some(ttft_ns) = self.ttft_ns {
            self.clock.clone().sleep(ttft_ns).await;
            on_first_token(ttft_ns);
            observer.on_token(
                turn.uuid,
                (self.clock.now_ns() - self.origin_ns) as f64 / 1_000_000.0,
            );
        }
        self.clock.clone().sleep(self.decode_ns).await;
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        let end_ns = self.clock.now_ns();
        self.active.set(self.active.get() - 1);
        Ok(TurnDispatchOutcome {
            start_ns,
            end_ns,
            terminal: ReplayTerminalStatus::Completed,
            response_text: format!("reply-{}-{}", turn.conversation_id, turn.turn_index),
            model_response: aiperf::scheduled::ModelResponseMetadata::default(),
            prompt_tokens: None,
            completion_tokens: Some(1),
            http: aiperf_metrics::HttpTrace::default(),
        })
    }
}

#[async_trait(?Send)]
impl TurnDispatcher for FailingDispatcher {
    async fn dispatch_turn(
        &self,
        _turn: TurnToSend,
        _observer: &dyn RequestObserver,
        _on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        self.clock.clone().sleep(self.delay_ns).await;
        anyhow::bail!("injected pre-token dispatch failure")
    }
}

fn rate_config(
    session_concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
) -> RequestRateConfig {
    RequestRateConfig {
        arrival_pattern: ArrivalPattern::Constant,
        request_rate: Some(10.0),
        arrival_smoothness: None,
        session_concurrency,
        prefill_concurrency,
        seed: 7,
    }
}

fn synthetic(turns: usize, think_time_ms: Option<u64>) -> Box<dyn ConversationSource> {
    Box::new(
        SyntheticConversationSource::new(SkeletonWorkload {
            num_requests: 0,
            input_tokens: 2,
            output_tokens: 1,
            turns,
            think_time_ms,
        })
        .unwrap(),
    )
}

fn run_sim(
    clock: Rc<SimClock>,
    workload: Rc<RequestRateWorkload>,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
) -> ScheduledRunReport {
    let report = Rc::new(RefCell::new(None));
    let output = report.clone();
    let clock_for_body = clock.clone();
    let workload: Rc<dyn Workload> = workload;
    let outcome = drive_sim(clock, move |_handle| async move {
        let clock: Rc<dyn Clock> = clock_for_body;
        let result = run_scheduled_workload(workload, clock, 0, dispatcher, stop, true)
            .await
            .unwrap();
        *output.borrow_mut() = Some(result);
    });
    assert!(!outcome.deadlocked, "request-rate workload must drain");
    Rc::try_unwrap(report).ok().unwrap().into_inner().unwrap()
}

#[test]
fn ready_continuations_win_each_rate_tick_and_materialize_live_replies() {
    let dataset = ConversationDataset::from_json_or_jsonl(
        r#"{
          "conversations": [
            {"conversation_id":"a","turns":[
              {"prompt_text":"a0","input_length":1,"max_output_tokens":1},
              {"prompt_text":"a1","input_length":1,"max_output_tokens":1},
              {"prompt_text":"a2","input_length":1,"max_output_tokens":1}
            ]},
            {"conversation_id":"b","turns":[
              {"prompt_text":"b0","input_length":1,"max_output_tokens":1},
              {"prompt_text":"b1","input_length":1,"max_output_tokens":1},
              {"prompt_text":"b2","input_length":1,"max_output_tokens":1}
            ]},
            {"conversation_id":"c","turns":[
              {"prompt_text":"c0","input_length":1,"max_output_tokens":1},
              {"prompt_text":"c1","input_length":1,"max_output_tokens":1},
              {"prompt_text":"c2","input_length":1,"max_output_tokens":1}
            ]}
          ]
        }"#,
        1,
        1,
    )
    .unwrap();
    let source: Box<dyn ConversationSource> = Box::new(DatasetConversationSource::new(dataset));
    let workload = Rc::new(RequestRateWorkload::new(rate_config(Some(3), None), source).unwrap());
    let slots = workload.session_slots().unwrap();
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), Some(150_000_000), 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher.clone(),
        StopConfig {
            total_expected_requests: Some(7),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| (turn.conversation_id.as_str(), turn.turn_index))
            .collect::<Vec<_>>(),
        vec![
            ("a", 0),
            ("b", 0),
            ("a", 1),
            ("b", 1),
            ("a", 2),
            ("b", 2),
            ("c", 0),
        ]
    );
    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 200, 300, 400, 500, 600, 700]
    );
    let a2 = dispatcher
        .seen
        .borrow()
        .iter()
        .find(|turn| turn.conversation_id == "a" && turn.turn_index == 2)
        .cloned()
        .unwrap();
    assert_eq!(
        a2.roles,
        vec!["user", "assistant", "user", "assistant", "user"]
    );
    assert_eq!(slots.stats().acquire_count, 3);
    assert_eq!(slots.stats().release_count, 3);
}

#[test]
fn prefill_guard_releases_at_ttft_while_decode_remains_in_flight() {
    let workload = Rc::new(
        RequestRateWorkload::new(rate_config(Some(2), Some(1)), synthetic(1, None)).unwrap(),
    );
    let prefill = workload.prefill_slots().unwrap();
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), Some(50_000_000), 300_000_000);
    let report = run_sim(
        clock,
        workload,
        dispatcher.clone(),
        StopConfig {
            total_expected_requests: Some(2),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 200]
    );
    assert_eq!(dispatcher.max_active.get(), 2);
    assert_eq!(prefill.stats().acquire_count, 2);
    assert_eq!(prefill.stats().release_count, 2);
}

#[test]
fn terminal_without_a_token_releases_prefill_and_prevents_deadlock() {
    let workload = Rc::new(
        RequestRateWorkload::new(rate_config(Some(2), Some(1)), synthetic(1, None)).unwrap(),
    );
    let prefill = workload.prefill_slots().unwrap();
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), None, 250_000_000);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        StopConfig {
            total_expected_requests: Some(2),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 400]
    );
    assert_eq!(prefill.stats().acquire_count, 2);
    assert_eq!(prefill.stats().release_count, 2);
}

#[test]
fn dispatch_error_before_ttft_uses_the_same_terminal_prefill_fallback() {
    let workload = Rc::new(
        RequestRateWorkload::new(rate_config(Some(2), Some(1)), synthetic(1, None)).unwrap(),
    );
    let prefill = workload.prefill_slots().unwrap();
    let clock = Rc::new(SimClock::new());
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(FailingDispatcher {
        clock: clock.clone(),
        delay_ns: 250_000_000,
    });
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        StopConfig {
            total_expected_requests: Some(2),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 400]
    );
    assert_eq!(report.performance.request_counts.completed_requests, 0);
    assert_eq!(prefill.stats().acquire_count, 2);
    assert_eq!(prefill.stats().release_count, 2);
}

#[test]
fn think_time_defers_queue_insertion_and_session_stop_drains_the_chain() {
    let workload = Rc::new(
        RequestRateWorkload::new(rate_config(Some(1), None), synthetic(2, Some(250))).unwrap(),
    );
    let sessions = workload.session_slots().unwrap();
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), Some(10_000_000), 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        StopConfig {
            expected_num_sessions: Some(1),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 400]
    );
    assert_eq!(report.performance.request_counts.num_requests, 2);
    assert_eq!(sessions.stats().acquire_count, 1);
    assert_eq!(sessions.stats().release_count, 1);
}

#[test]
fn skipped_session_ticks_retry_the_cached_sampler_draw() {
    let dataset = ConversationDataset::from_json_or_jsonl(
        r#"{
          "conversations": [
            {"conversation_id":"a","turns":[{"input_length":1,"max_output_tokens":1}]},
            {"conversation_id":"b","turns":[{"input_length":1,"max_output_tokens":1}]},
            {"conversation_id":"c","turns":[{"input_length":1,"max_output_tokens":1}]}
          ]
        }"#,
        1,
        1,
    )
    .unwrap();
    let workload = Rc::new(
        RequestRateWorkload::new(
            rate_config(Some(1), None),
            Box::new(DatasetConversationSource::new(dataset)),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), Some(250_000_000), 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        StopConfig {
            total_expected_requests: Some(2),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| (
                turn.conversation_id.as_str(),
                turn.issued_offset_ns / 1_000_000
            ))
            .collect::<Vec<_>>(),
        vec![("a", 100), ("b", 400)]
    );
}

#[test]
fn duration_bound_stops_at_the_clock_boundary_and_drains_inflight_turns() {
    let workload =
        Rc::new(RequestRateWorkload::new(rate_config(None, None), synthetic(1, None)).unwrap());
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), Some(75_000_000), 100_000_000);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        StopConfig {
            expected_duration_ns: Some(250_000_000),
            ..StopConfig::default()
        },
    );

    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![100, 200]
    );
    assert_eq!(report.performance.request_counts.completed_requests, 2);
    assert_eq!(report.performance.throughput.wall_time_ms, 375.0);
}
