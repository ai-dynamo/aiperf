// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Linux real-clock characterization for high-rate request scheduling.
#![cfg(all(feature = "engine", target_os = "linux"))]

use std::future::Future;
use std::rc::Rc;

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::multiturn::TurnToSend;
use aiperf_runtime::request_rate::{RequestRateConfig, RequestRateWorkload};
use aiperf_runtime::scheduled::{
    ModelResponseMetadata, TurnDispatchOutcome, TurnDispatcher, Workload, run_scheduled_workload,
};
use aiperf_runtime::timing::{ArrivalPattern, StopConfig};
use async_trait::async_trait;

mod common;

const REQUESTS: u64 = 5_000;
const REQUEST_RATE: f64 = 5_000.0;
const MIN_ACHIEVED_RATE: f64 = 2_000.0;
const MAX_ACHIEVED_RATE: f64 = 8_000.0;

struct ImmediateDispatcher {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
}

#[async_trait(?Send)]
impl TurnDispatcher for ImmediateDispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        let now_ns = self.clock.now_ns();
        let offset_ms = now_ns.saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        observer.on_admit(turn.uuid, offset_ms, 0);
        on_first_token(0);
        observer.on_token(turn.uuid, offset_ms);
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        Ok(TurnDispatchOutcome {
            start_ns: now_ns,
            end_ns: now_ns,
            terminal: ReplayTerminalStatus::Completed,
            response_text: String::new(),
            model_response: ModelResponseMetadata::default(),
            prompt_tokens: None,
            completion_tokens: Some(1),
            http: Default::default(),
        })
    }
}

fn run_local<F: Future>(future: F) -> F::Output {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

#[test]
fn real_clock_delivers_exact_high_rate_request_count() {
    let (report, elapsed_ns) = run_local(async {
        let source = common::synthetic_prepared_source(1, 8, 1, None, "model").await;
        let workload: Rc<dyn Workload> = Rc::new(
            RequestRateWorkload::new(
                RequestRateConfig {
                    arrival_pattern: ArrivalPattern::Constant,
                    request_rate: Some(REQUEST_RATE),
                    arrival_smoothness: None,
                    session_concurrency: None,
                    prefill_concurrency: None,
                    seed: 7,
                },
                source,
            )
            .unwrap(),
        );
        let clock = RealClock::new();
        let start_ns = clock.now_ns();
        let runtime_clock: Rc<dyn Clock> = clock.clone();
        let report = run_scheduled_workload(
            workload,
            runtime_clock.clone(),
            start_ns,
            Rc::new(ImmediateDispatcher {
                clock: runtime_clock,
                origin_ns: start_ns,
            }),
            StopConfig {
                total_expected_requests: Some(REQUESTS),
                ..StopConfig::default()
            },
            true,
        )
        .await
        .unwrap();
        let elapsed_ns = clock.now_ns().saturating_sub(start_ns);
        (report, elapsed_ns)
    });

    assert_eq!(
        report.performance.request_counts.num_requests,
        REQUESTS as usize
    );
    assert_eq!(
        report.performance.request_counts.completed_requests,
        REQUESTS as usize
    );
    assert_eq!(report.turns.len(), REQUESTS as usize);

    let achieved_rate = REQUESTS as f64 * 1_000_000_000.0 / elapsed_ns as f64;
    eprintln!(
        "request_rate_real_receipt exact_count={REQUESTS} elapsed_ns={elapsed_ns} achieved_rate={achieved_rate:.3}"
    );
    assert!(
        (MIN_ACHIEVED_RATE..=MAX_ACHIEVED_RATE).contains(&achieved_rate),
        "target {REQUEST_RATE:.0} req/s delivered {achieved_rate:.3} req/s over {elapsed_ns}ns"
    );
}
