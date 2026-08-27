// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static-versus-erased cost at the thread-per-core request boundary.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use uuid::Uuid;

use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::{RequestObserver, TurnResponseObserver};
use aiperf_runtime::endpoints::{EndpointId, EndpointKey};
use aiperf_runtime::engine::turn_execution::WorkerSink;
use aiperf_runtime::metrics::RequestMetricMetadata;
use aiperf_runtime::metrics_core::InferenceDimensions;
use aiperf_runtime::multiturn::PreparedEndpointReference;
use aiperf_runtime::transport::core::{
    DispatchResult, MeasuredContext, MeasuredOutcome, PreparedEndpointBinding, PreparedTurn,
    Request, RequestExecutor, TurnDataPolicy,
};
use aiperf_runtime::transport::http::{Clock, RealClock};

struct NoopObserver;

impl RequestObserver for NoopObserver {
    fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}

    fn on_admit(&self, _: Uuid, _: f64, _: usize) {}

    fn on_token(&self, _: Uuid, _: f64) {}

    fn on_terminal(&self, _: Uuid, _: ReplayTerminalStatus) {}
}

struct NoopSink {
    clock: Rc<dyn Clock>,
}

#[async_trait(?Send)]
impl WorkerSink for NoopSink {
    fn set_run_origin(&self, _: i64) {}

    fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    fn inference_dimensions(&self, _: &aiperf_runtime::multiturn::TurnToSend) -> InferenceDimensions {
        InferenceDimensions::default()
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    async fn dispatch_measured(
        &self,
        _: &dyn RequestObserver,
        _: PreparedTurn,
        _: &MeasuredContext,
        _: &dyn Fn(i64),
        _: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        unreachable!("criterion drops the constructed dispatch future without polling it")
    }
}

#[async_trait(?Send)]
impl RequestExecutor for NoopSink {
    fn set_run_origin(&self, _: i64) -> Result<()> {
        Ok(())
    }

    fn inference_dimensions(&self, _: &aiperf_runtime::multiturn::TurnToSend) -> InferenceDimensions {
        InferenceDimensions::default()
    }

    async fn execute_measured(
        &self,
        _: PreparedTurn,
        _: MeasuredContext,
        _: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        unreachable!("criterion drops the constructed execution future without polling it")
    }
}

fn prepared_turn() -> PreparedTurn {
    PreparedTurn {
        runtime_session_id: "criterion-session".to_owned(),
        request: Request {
            uuid: Uuid::nil(),
            input_length: 1,
            max_output_tokens: 1,
            prompt_text: None,
            body: None,
            headers: BTreeMap::new(),
            parameters: BTreeMap::new(),
            endpoint_path: None,
            streaming: false,
            x_correlation_id: Some("criterion-session".to_owned()),
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
            image_count: None,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        },
        model: "criterion-model".to_owned(),
        endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
            key: EndpointKey::from_index(0),
            endpoint_id: EndpointId::new("chat").expect("valid endpoint id"),
        }),
        endpoint_aware: true,
        data_policy: TurnDataPolicy::ordinary(),
        deferred: None,
    }
}

fn measured_context() -> MeasuredContext {
    MeasuredContext {
        arrival_ms: 0.0,
        input_length: 1,
        requested_output_length: 1,
        metadata: RequestMetricMetadata::default(),
        wants_live_record: false,
        wants_http_exchange: false,
        consume_record: false,
    }
}

fn bench_dispatch_boundary(criterion: &mut Criterion) {
    let sink = NoopSink {
        clock: RealClock::new(),
    };
    let erased_worker: &dyn WorkerSink = black_box(&sink);
    let erased_executor: &dyn RequestExecutor = black_box(&sink);
    let observer = NoopObserver;
    let on_first_token = |_: i64| {};

    let mut worker = criterion.benchmark_group("worker_sink_dispatch_future");
    worker.bench_function("static", |bencher| {
        bencher.iter_batched(
            || (prepared_turn(), measured_context()),
            |(turn, context)| {
                black_box(WorkerSink::dispatch_measured(
                    black_box(&sink),
                    black_box(&observer),
                    turn,
                    black_box(&context),
                    black_box(&on_first_token),
                    None,
                ));
            },
            BatchSize::SmallInput,
        );
    });
    worker.bench_function("erased", |bencher| {
        bencher.iter_batched(
            || (prepared_turn(), measured_context()),
            |(turn, context)| {
                black_box(WorkerSink::dispatch_measured(
                    erased_worker,
                    black_box(&observer),
                    turn,
                    black_box(&context),
                    black_box(&on_first_token),
                    None,
                ));
            },
            BatchSize::SmallInput,
        );
    });
    worker.finish();

    let mut executor = criterion.benchmark_group("request_executor_future");
    executor.bench_function("static", |bencher| {
        bencher.iter_batched(
            || (prepared_turn(), measured_context()),
            |(turn, context)| {
                black_box(RequestExecutor::execute_measured(
                    black_box(&sink),
                    turn,
                    context,
                    black_box(&on_first_token),
                ));
            },
            BatchSize::SmallInput,
        );
    });
    executor.bench_function("erased", |bencher| {
        bencher.iter_batched(
            || (prepared_turn(), measured_context()),
            |(turn, context)| {
                black_box(RequestExecutor::execute_measured(
                    erased_executor,
                    turn,
                    context,
                    black_box(&on_first_token),
                ));
            },
            BatchSize::SmallInput,
        );
    });
    executor.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(200)
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(15));
    targets = bench_dispatch_boundary
}
criterion_main!(benches);
