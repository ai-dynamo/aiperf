// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic end-to-end schedule-policy tests over the real workload runtime.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use aiperf::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use aiperf::multiturn::{
    ConversationDataset, ConversationSource, DatasetConversationSource,
    SyntheticConversationSource, TurnResponse, TurnToSend,
};
use aiperf::scheduled::{
    ScheduledAncillaryPolicies, ScheduledRunReport, TurnDispatchOutcome, TurnDispatcher, Workload,
    run_scheduled_workload, run_scheduled_workload_with_ancillary,
};
use aiperf::user_centric::UserTargetController;
use aiperf::user_centric::{UserCentricConfig, UserCentricWorkload};
use aiperf::workload::SkeletonWorkload;
use aiperf_clock::Clock;
use aiperf_clock::sim_clock::SimClock;
use aiperf_graph::runtime::drive_sim;
use aiperf_timing::{BernoulliFixedDelay, Phase, RoundRobinUrlSelector, StopConfig};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;

#[derive(Clone, Debug)]
struct SeenTurn {
    conversation_id: String,
    turn_index: usize,
    roles: Vec<String>,
    cancel_after_ns: Option<i64>,
    url_index: Option<u32>,
}

struct SimDispatcher {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    ttft_ns: i64,
    itl_ns: i64,
    active: Cell<usize>,
    max_active: Cell<usize>,
    seen: RefCell<Vec<SeenTurn>>,
}

type SeenCreditUrls = Rc<RefCell<Vec<(usize, Option<u32>, Option<u32>)>>>;

struct CreditInspectWorkload {
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    first: TurnToSend,
    seen: SeenCreditUrls,
}

#[async_trait(?Send)]
impl Workload for CreditInspectWorkload {
    fn name(&self) -> &'static str {
        "credit_inspection"
    }

    async fn execute(
        &self,
        runtime: Rc<aiperf::scheduled::ScheduledRuntime>,
    ) -> anyhow::Result<()> {
        issue_inspection_turn(
            runtime,
            self.source.clone(),
            self.first.clone(),
            self.seen.clone(),
        );
        Ok(())
    }
}

fn issue_inspection_turn(
    runtime: Rc<aiperf::scheduled::ScheduledRuntime>,
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    turn: TurnToSend,
    seen: SeenCreditUrls,
) {
    let scheduled_ns = runtime.now_ns();
    let runtime_for_completion = runtime.clone();
    runtime.issue_turn(
        turn,
        scheduled_ns,
        None,
        Box::new(move |credit, outcome| {
            Box::pin(async move {
                seen.borrow_mut().push((
                    credit.turn.turn_index,
                    credit.url_index,
                    credit.turn.url_index,
                ));
                let next = source
                    .borrow()
                    .next_turn(
                        &credit,
                        TurnResponse {
                            text: outcome.response_text,
                            completion_tokens: outcome.completion_tokens,
                            terminal: outcome.terminal,
                        },
                    )
                    .unwrap();
                if let Some(next) = next {
                    issue_inspection_turn(runtime_for_completion, source, next, seen);
                }
            })
        }),
    );
}

impl SimDispatcher {
    fn new(clock: Rc<dyn Clock>, ttft_ns: i64, itl_ns: i64) -> Rc<Self> {
        Rc::new(Self {
            origin_ns: clock.now_ns(),
            clock,
            ttft_ns,
            itl_ns,
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
            cancel_after_ns: turn.cancel_after_ns,
            url_index: turn.url_index,
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
        let end_ns = self.clock.now_ns();
        self.active.set(self.active.get() - 1);
        Ok(TurnDispatchOutcome {
            start_ns,
            end_ns,
            terminal: ReplayTerminalStatus::Completed,
            response_text: format!("reply-{}", turn.turn_index),
            model_response: aiperf::scheduled::ModelResponseMetadata::default(),
            prompt_tokens: None,
            completion_tokens: None,
            http: aiperf_metrics::HttpTrace::default(),
        })
    }
}

fn run_sim(
    clock: Rc<SimClock>,
    workload: Rc<dyn Workload>,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: aiperf_timing::StopConfig,
    enforce_stop: bool,
) -> ScheduledRunReport {
    let report = Rc::new(RefCell::new(None));
    let output = report.clone();
    let clock_for_body = clock.clone();
    let outcome = drive_sim(clock, move |_handle| async move {
        let clock_dyn: Rc<dyn Clock> = clock_for_body;
        let result = run_scheduled_workload(workload, clock_dyn, 0, dispatcher, stop, enforce_stop)
            .await
            .unwrap();
        *output.borrow_mut() = Some(result);
    });
    assert!(!outcome.deadlocked, "scheduled workload must drain");
    Rc::try_unwrap(report).ok().unwrap().into_inner().unwrap()
}

fn run_sim_with_policies(
    clock: Rc<SimClock>,
    workload: Rc<dyn Workload>,
    dispatcher: Rc<dyn TurnDispatcher>,
    stop: StopConfig,
    enforce_stop: bool,
    policies: ScheduledAncillaryPolicies,
) -> ScheduledRunReport {
    let report = Rc::new(RefCell::new(None));
    let output = report.clone();
    let clock_for_body = clock.clone();
    let outcome = drive_sim(clock, move |_handle| async move {
        let clock_dyn: Rc<dyn Clock> = clock_for_body;
        let result = run_scheduled_workload_with_ancillary(
            workload,
            clock_dyn,
            0,
            dispatcher,
            stop,
            enforce_stop,
            policies,
        )
        .await
        .unwrap();
        *output.borrow_mut() = Some(result);
    });
    assert!(!outcome.deadlocked, "scheduled workload must drain");
    Rc::try_unwrap(report).ok().unwrap().into_inner().unwrap()
}

#[test]
fn fixed_schedule_replays_absolute_relative_and_immediate_turns_exactly() {
    let dataset = ConversationDataset::from_json_or_jsonl(
        r#"{
          "conversations": [
            {"conversation_id":"a","turns":[
              {"timestamp_ms":1000,"prompt_text":"a0","input_length":1,"max_output_tokens":2},
              {"timestamp_ms":1120,"prompt_text":"a1","input_length":1,"max_output_tokens":2},
              {"delay_ms":40,"prompt_text":"a2","input_length":1,"max_output_tokens":2}
            ]},
            {"conversation_id":"b","turns":[
              {"timestamp_ms":1050,"prompt_text":"b0","input_length":1,"max_output_tokens":2},
              {"delay_ms":25,"prompt_text":"b1","input_length":1,"max_output_tokens":2},
              {"prompt_text":"b2","input_length":1,"max_output_tokens":2}
            ]}
          ]
        }"#,
        1,
        2,
    )
    .unwrap();
    let source: Box<dyn ConversationSource> = Box::new(DatasetConversationSource::new(dataset));
    let schedule_source = Rc::new(
        DatasetFixedScheduleSource::new(FixedScheduleConfig {
            auto_offset_timestamps: true,
            start_offset_ms: None,
        })
        .unwrap(),
    );
    let workload: Rc<dyn Workload> =
        Rc::new(FixedScheduleWorkload::new(source, schedule_source).unwrap());
    let clock = Rc::new(SimClock::new());
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let dispatcher = SimDispatcher::new(clock_dyn, 10_000_000, 5_000_000);
    let report = run_sim(
        clock,
        workload,
        dispatcher.clone(),
        aiperf_timing::StopConfig::default(),
        false,
    );

    assert_eq!(report.performance.request_counts.num_requests, 6);
    assert_eq!(report.performance.request_counts.completed_requests, 6);
    assert_eq!(report.performance.request_counts.total_output_tokens, 12);
    assert_eq!(
        report
            .native_metrics
            .finite_value(aiperf_metrics::MetricTag::RequestCount),
        Some(6.0)
    );
    assert!(
        report
            .native_metrics
            .result(aiperf_metrics::MetricTag::EffectiveConcurrency)
            .and_then(aiperf_metrics::MetricResult::distribution)
            .is_some()
    );
    assert!(
        report
            .native_metrics
            .result(aiperf_metrics::MetricTag::CreditDropLatency)
            .is_none(),
        "fixed schedules have no inherited credit-drop latency"
    );
    assert!(
        report
            .native_metrics
            .result(aiperf_metrics::MetricTag::CreditToStartLatency)
            .is_none(),
        "fixed schedules have no policy-credit timestamp"
    );
    assert!(
        report
            .native_metrics
            .result(aiperf_metrics::MetricTag::EffectiveLatency)
            .is_none(),
        "fixed schedules must omit credit-relative effective latency"
    );
    assert_eq!(report.schedule_timing.early_turns, 0);
    assert_eq!(report.schedule_timing.max_issue_lateness_ms, 0.0);
    assert_eq!(report.schedule_timing.mean_ttft_ms, Some(10.0));

    let offsets = |conversation_id: &str| {
        report
            .turns
            .iter()
            .filter(|record| record.conversation_id == conversation_id)
            .map(|record| record.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>()
    };
    assert_eq!(offsets("a"), vec![0, 120, 175]);
    assert_eq!(offsets("b"), vec![50, 90, 105]);

    let seen = dispatcher.seen.borrow();
    let a2 = seen
        .iter()
        .find(|turn| turn.conversation_id == "a" && turn.turn_index == 2)
        .unwrap();
    assert_eq!(
        a2.roles,
        vec!["user", "assistant", "user", "assistant", "user"]
    );
}

#[test]
fn ancillary_policies_cancel_each_selected_turn_and_pin_urls_per_session() {
    let dataset = ConversationDataset::from_json_or_jsonl(
        r#"{
          "conversations": [
            {"conversation_id":"a","turns":[
              {"timestamp_ms":0,"input_length":1,"max_output_tokens":1},
              {"delay_ms":1,"input_length":1,"max_output_tokens":1},
              {"delay_ms":1,"input_length":1,"max_output_tokens":1}
            ]},
            {"conversation_id":"b","turns":[
              {"timestamp_ms":0,"input_length":1,"max_output_tokens":1},
              {"delay_ms":1,"input_length":1,"max_output_tokens":1},
              {"delay_ms":1,"input_length":1,"max_output_tokens":1}
            ]}
          ]
        }"#,
        1,
        1,
    )
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        FixedScheduleWorkload::new(
            Box::new(DatasetConversationSource::new(dataset)),
            Rc::new(
                DatasetFixedScheduleSource::new(FixedScheduleConfig {
                    auto_offset_timestamps: true,
                    start_offset_ms: None,
                })
                .unwrap(),
            ),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 1, 0);
    let cancellation = BernoulliFixedDelay::from_delay_ns_seed(Some(100.0), 123, Some(9)).unwrap();
    let selector = RoundRobinUrlSelector::new(vec!["http://a".into(), "http://b".into()]).unwrap();

    let report = run_sim_with_policies(
        clock,
        workload,
        dispatcher.clone(),
        StopConfig::default(),
        false,
        ScheduledAncillaryPolicies {
            cancellation_policy: Some(Box::new(cancellation)),
            url_selector: Some(Box::new(selector)),
            phase: Phase::Profiling,
        },
    );
    assert_eq!(report.turns.len(), 6);
    let seen = dispatcher.seen.borrow();
    assert!(seen.iter().all(|turn| turn.cancel_after_ns == Some(123)));
    for (conversation, expected_index) in [("a", 0), ("b", 1)] {
        let turns = seen
            .iter()
            .filter(|turn| turn.conversation_id == conversation)
            .collect::<Vec<_>>();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].turn_index, 0);
        assert!(
            turns
                .iter()
                .all(|turn| turn.url_index == Some(expected_index)),
            "all turns in session {conversation} must stay on endpoint {expected_index}"
        );
    }
}

#[test]
fn issued_credit_keeps_selector_output_on_turn_zero_only() {
    let mut source: Box<dyn ConversationSource> = Box::new(
        SyntheticConversationSource::new(SkeletonWorkload {
            num_requests: 0,
            input_tokens: 1,
            output_tokens: 1,
            turns: 3,
            think_time_ms: None,
        })
        .unwrap(),
    );
    let first = source
        .next(Some("session".into()))
        .unwrap()
        .build_first_turn(None)
        .unwrap();
    let seen = Rc::new(RefCell::new(Vec::new()));
    let workload: Rc<dyn Workload> = Rc::new(CreditInspectWorkload {
        source: Rc::new(RefCell::new(source)),
        first,
        seen: seen.clone(),
    });
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 1, 0);
    let selector = RoundRobinUrlSelector::new(vec!["http://a".into(), "http://b".into()]).unwrap();
    let report = run_sim_with_policies(
        clock,
        workload,
        dispatcher,
        StopConfig::default(),
        false,
        ScheduledAncillaryPolicies {
            url_selector: Some(Box::new(selector)),
            ..ScheduledAncillaryPolicies::default()
        },
    );

    assert_eq!(report.turns.len(), 3);
    assert_eq!(
        *seen.borrow(),
        vec![
            (0, Some(0), Some(0)),
            (1, None, Some(0)),
            (2, None, Some(0)),
        ]
    );
}

#[test]
fn fixed_absolute_timestamp_in_the_past_fires_on_response_return() {
    let dataset = ConversationDataset::from_json_or_jsonl(
        r#"{"conversation_id":"late","turns":[
          {"timestamp_ms":0,"input_length":1,"max_output_tokens":1},
          {"timestamp_ms":20,"input_length":1,"max_output_tokens":1}
        ]}"#,
        1,
        1,
    )
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        FixedScheduleWorkload::new(
            Box::new(DatasetConversationSource::new(dataset)),
            Rc::new(
                DatasetFixedScheduleSource::new(FixedScheduleConfig {
                    auto_offset_timestamps: true,
                    start_offset_ms: None,
                })
                .unwrap(),
            ),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 50_000_000, 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        aiperf_timing::StopConfig::default(),
        false,
    );
    assert_eq!(report.turns[1].scheduled_offset_ns, 20_000_000);
    assert_eq!(report.turns[1].issued_offset_ns, 50_000_000);
    assert_eq!(report.schedule_timing.max_issue_lateness_ms, 30.0);
}

#[test]
fn user_centric_seed_churn_and_per_user_pacing_match_the_contract() {
    let source = SyntheticConversationSource::new(SkeletonWorkload {
        num_requests: 0,
        input_tokens: 2,
        output_tokens: 1,
        turns: 3,
        think_time_ms: None,
    })
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 2,
                request_rate: 20.0,
                concurrency: None,
            },
            Box::new(source),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 10_000_000, 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        aiperf_timing::StopConfig {
            total_expected_requests: Some(8),
            expected_num_sessions: None,
            expected_duration_ns: None,
        },
        true,
    );

    assert_eq!(report.performance.request_counts.num_requests, 8);
    assert_eq!(report.schedule_timing.issued_turns, 8);
    assert_eq!(report.schedule_timing.early_turns, 0);
    assert_eq!(report.schedule_timing.max_issue_lateness_ms, 0.0);
    let first_turns = report
        .turns
        .iter()
        .filter(|record| record.turn_index == 0)
        .map(|record| record.issued_offset_ns / 1_000_000)
        .collect::<Vec<_>>();
    assert_eq!(first_turns, vec![0, 50, 150, 300]);

    for correlation_id in report
        .turns
        .iter()
        .map(|record| record.x_correlation_id.clone())
        .collect::<std::collections::HashSet<_>>()
    {
        let times = report
            .turns
            .iter()
            .filter(|record| record.x_correlation_id == correlation_id)
            .map(|record| record.issued_offset_ns)
            .collect::<Vec<_>>();
        for gap in times.windows(2) {
            assert!(gap[1] - gap[0] >= 100_000_000);
        }
    }
}

#[test]
fn user_centric_optional_concurrency_caps_live_sessions() {
    let source = SyntheticConversationSource::new(SkeletonWorkload {
        num_requests: 0,
        input_tokens: 2,
        output_tokens: 1,
        turns: 2,
        think_time_ms: None,
    })
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 4,
                request_rate: 100.0,
                concurrency: Some(2),
            },
            Box::new(source),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 50_000_000, 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher.clone(),
        aiperf_timing::StopConfig {
            total_expected_requests: Some(10),
            expected_num_sessions: None,
            expected_duration_ns: None,
        },
        true,
    );
    assert_eq!(report.performance.request_counts.num_requests, 10);
    assert!(dispatcher.max_active.get() <= 2);
}

#[test]
fn user_centric_session_bound_starts_exact_sessions_then_drains_turns() {
    let source = SyntheticConversationSource::new(SkeletonWorkload {
        num_requests: 0,
        input_tokens: 2,
        output_tokens: 1,
        turns: 3,
        think_time_ms: None,
    })
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 2,
                request_rate: 20.0,
                concurrency: None,
            },
            Box::new(source),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 10_000_000, 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        aiperf_timing::StopConfig {
            total_expected_requests: None,
            expected_num_sessions: Some(4),
            expected_duration_ns: None,
        },
        true,
    );
    assert_eq!(
        report
            .turns
            .iter()
            .filter(|turn| turn.turn_index == 0)
            .count(),
        4
    );
    assert_eq!(report.performance.request_counts.num_requests, 10);
}

#[test]
fn user_centric_duration_cancels_future_schedule_but_drains_inflight() {
    let source = SyntheticConversationSource::new(SkeletonWorkload {
        num_requests: 0,
        input_tokens: 2,
        output_tokens: 1,
        turns: 3,
        think_time_ms: None,
    })
    .unwrap();
    let workload: Rc<dyn Workload> = Rc::new(
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 2,
                request_rate: 20.0,
                concurrency: None,
            },
            Box::new(source),
        )
        .unwrap(),
    );
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 10_000_000, 0);
    let report = run_sim(
        clock,
        workload,
        dispatcher,
        aiperf_timing::StopConfig {
            total_expected_requests: None,
            expected_num_sessions: None,
            expected_duration_ns: Some(175_000_000),
        },
        true,
    );
    assert_eq!(
        report
            .turns
            .iter()
            .map(|turn| turn.issued_offset_ns / 1_000_000)
            .collect::<Vec<_>>(),
        vec![0, 50, 100, 150]
    );
    assert_eq!(report.performance.request_counts.completed_requests, 4);
    assert_eq!(report.performance.throughput.wall_time_ms, 175.0);
}

#[test]
fn adaptive_scale_up_interrupts_spawn_sleep_and_uses_new_turn_gap() {
    let source = SyntheticConversationSource::new(SkeletonWorkload {
        num_requests: 0,
        input_tokens: 2,
        output_tokens: 1,
        turns: 3,
        think_time_ms: None,
    })
    .unwrap();
    let workload = Rc::new(
        UserCentricWorkload::new(
            UserCentricConfig {
                num_users: 2,
                request_rate: 20.0,
                concurrency: None,
            },
            Box::new(source),
        )
        .unwrap(),
    );
    let control = workload.control();
    let workload_trait: Rc<dyn Workload> = workload;
    let clock = Rc::new(SimClock::new());
    let dispatcher = SimDispatcher::new(clock.clone(), 10_000_000, 0);
    let report_slot = Rc::new(RefCell::new(None));
    let output = report_slot.clone();
    let clock_for_body = clock.clone();
    let outcome = drive_sim(clock, move |_handle| async move {
        let control_clock = clock_for_body.clone();
        tokio::task::spawn_local(async move {
            control_clock.clone().sleep(125_000_000).await;
            control.set_target_users(3, control_clock.now_ns()).unwrap();
        });
        let clock_dyn: Rc<dyn Clock> = clock_for_body;
        let report = run_scheduled_workload(
            workload_trait,
            clock_dyn,
            0,
            dispatcher,
            aiperf_timing::StopConfig {
                total_expected_requests: Some(10),
                expected_num_sessions: None,
                expected_duration_ns: None,
            },
            true,
        )
        .await
        .unwrap();
        *output.borrow_mut() = Some(report);
    });
    assert!(!outcome.deadlocked);
    let report = Rc::try_unwrap(report_slot)
        .ok()
        .unwrap()
        .into_inner()
        .unwrap();
    assert_eq!(report.user_control.unwrap().target_value, 3);
    assert!(
        report
            .turns
            .iter()
            .any(|turn| turn.turn_index == 0 && turn.scheduled_offset_ns == 125_000_000),
        "scale-up spawn must preempt the prior 150ms heap sleep"
    );
}
