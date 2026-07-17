// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP phased scheduling over the normal transport path.

use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::phase_runtime::{ScheduledPhasePlan, run_scheduled_phases};
use aiperf_runtime::scheduled::{
    ScheduledAncillaryPolicies, SingleTurnDatasetWorkload, TurnDispatcher, Workload,
};
use aiperf_runtime::timing::{
    GracePeriod, NoopPhaseObserver, PhaseConfig, PhaseKind, PhaseObserver, StopConfig,
};
use aiperf_runtime::transport::http::TransportSink;
use axum::{Router, http::header, response::IntoResponse, routing::post};

mod common;

const SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn seamless_phases_overlap_over_the_real_http_dispatcher() {
    let calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let calls = calls.clone();
            move || {
                let calls = calls.clone();
                async move {
                    let delay_ms = if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        30
                    } else {
                        5
                    };
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    ([(header::CONTENT_TYPE, "text/event-stream")], SSE).into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let local = tokio::task::LocalSet::new();
    let report = local
        .run_until(async move {
            let clock: Rc<dyn Clock> = RealClock::new();
            let start_ns = clock.now_ns();
            let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
                TransportSink::new_multi(
                    clock.clone(),
                    start_ns,
                    &[format!("http://{address}")],
                    "model",
                    false,
                )
                .unwrap()
                .with_prepared_endpoints(common::chat_dispatch_table()),
            );
            let observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
            run_scheduled_phases(
                vec![
                    ScheduledPhasePlan::new(
                        phase("warmup", PhaseKind::Warmup, true),
                        one_request_workload().await,
                        ScheduledAncillaryPolicies::default(),
                    )
                    .with_start_ns(start_ns),
                    ScheduledPhasePlan::new(
                        phase("profiling", PhaseKind::Profiling, false),
                        one_request_workload().await,
                        ScheduledAncillaryPolicies::default(),
                    )
                    .with_start_ns(start_ns),
                ],
                clock,
                dispatcher,
                observer,
            )
            .await
        })
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(report.phases.len(), 2);
    assert_eq!(report.reports.len(), 2);
    assert_eq!(report.phases[0].final_requests_completed, Some(1));
    assert_eq!(report.phases[1].final_requests_completed, Some(1));
    assert!(
        report.phases[1].start_ns.unwrap() < report.phases[0].requests_end_ns.unwrap(),
        "profiling must start while seamless warmup still waits for its HTTP return"
    );
}

fn phase(id: &str, kind: PhaseKind, seamless: bool) -> PhaseConfig {
    PhaseConfig::new(
        id,
        kind,
        StopConfig {
            total_expected_requests: Some(1),
            ..StopConfig::default()
        },
    )
    .with_grace_period(GracePeriod::Infinite)
    .with_seamless(seamless)
}

async fn one_request_workload() -> Rc<dyn Workload> {
    let source = common::synthetic_prepared_source(1, 4, 1, None, "model").await;
    Rc::new(SingleTurnDatasetWorkload::new(source, 1).unwrap())
}
