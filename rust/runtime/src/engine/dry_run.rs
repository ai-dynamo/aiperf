// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight `dry_run` transport: a fake execution leaf that fabricates every
//! request outcome from a small analytic latency model, with zero network.
//!
//! # What this is
//!
//! `dry_run` is a first-class registered transport (`transport.type: dry_run`,
//! `--dry-run` on the CLI). It is classified as a *native* transport
//! ([`crate::engine::online_execution::classify_native_transport`]) so it reuses
//! the entire native scheduled/graph runtime — pacing, [`crate::scheduled`]
//! admission, phase orchestration, the metrics accumulator, and the whole export
//! plane — unchanged. Only the leaf that would open a socket is swapped: instead
//! of [`crate::http::TransportSink`], the run drives [`FakeRequestExecutor`],
//! which synthesizes each request's timing analytically and drives the same
//! [`NativeMetricsObserver`] the real HTTP path drives.
//!
//! The value: a user can run `aiperf profile --dry-run` and exercise the full
//! pipeline (does my config produce valid artifacts? how fast can the loadgen
//! itself dispatch?) without an inference server. See the design record at
//! `~/.aiperf/docs/superpowers/specs/2026-07-16-dry-run-transport-design.md`.
//!
//! # Fabrication contract
//!
//! For one request with input length `ISL` and requested output length `OSL`,
//! given analytic `ttft_ms`/`itl_ms`, the fake dispatch begins at the request's
//! scheduled arrival and emits `OSL` synthetic output tokens spaced by `itl_ms`
//! after an initial `ttft_ms`. The exact observer event sequence mirrors
//! [`crate::http::TransportSink::dispatch_measured`]: `register_metadata` →
//! `on_arrival` → `OSL`× `on_token` → `on_usage` → `on_terminal` →
//! `record_response`. Downstream, TTFT = `first_token − start = ttft_ms`, ITL =
//! `itl_ms`, and `request_latency = ttft_ms + (OSL−1)·itl_ms`, all exact under a
//! zero-jitter model — which is what the end-to-end test asserts.
//!
//! # Extension points left open
//!
//! - **Clock:** the fabrication computes analytic timestamps directly (no
//!   sleeping), so it is correct under `RealClock` (loadgen self-benchmark) and,
//!   once the scheduled loop is driven by `drive_sim`, under `SimClock`
//!   (deterministic virtual-time CI). The design phases the sim-clock inline
//!   executor as a follow-up; this module already fabricates clock-agnostically.
//! - **Jitter:** `jitter_cv` is threaded through [`DryRunParams`] for a future
//!   seeded per-request jitter draw off [`crate::rng`]; the current default of
//!   `0.0` keeps the run byte-deterministic.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;

use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedUsage, RequestObserver};
use uuid::Uuid;

use crate::clock::Clock;
use crate::engine::graph_execution::GraphTransportKind;
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{
    NativeTransportExecution, RunnerClockKind, RunnerRunContext, RunnerTransportDescriptor,
    RunnerTransportFactory, ValidatedTransportConfig, WorkloadRequirements, strict_decode,
};
use crate::engine::turn_execution::{ExecutionBackendConfig, RequestExecutorFactory};
use crate::extensions::AIPerfRegistry;
use crate::http::{
    DispatchResult, MeasuredContext, MeasuredOutcome, PreparedTurn, RequestExecutor,
};
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata};
use crate::metrics_core::{HttpTrace, InferenceDimensions, MetricsConfig, RecordIngest};
use crate::multiturn::TurnToSend;
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use crate::transport_http::models::{RequestRecord, Response, TextResponse};

/// Default synthetic time-to-first-token (milliseconds).
const fn default_ttft_ms() -> f64 {
    10.0
}

/// Default synthetic inter-token latency (milliseconds).
const fn default_itl_ms() -> f64 {
    2.0
}

/// Built-in `dry_run` transport descriptor (real clock, no network).
pub static DRY_RUN_TRANSPORT_DESCRIPTOR: RunnerTransportDescriptor = RunnerTransportDescriptor {
    id: "dry_run",
    description: "Fake execution leaf: analytic-latency synthetic responses, zero network",
    clock: RunnerClockKind::Real,
    features: &["dry_run"],
};

/// Strict validated config owned by the `dry_run` transport.
///
/// Every field has a serde default so `{"type":"dry_run"}` alone is valid; the
/// CLI projects concrete `ttft_ms`/`itl_ms` so a run is fully specified and its
/// metrics are exactly predictable.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunTransportConfigV2 {
    /// Synthetic time-to-first-token in milliseconds (>= 0).
    #[serde(default = "default_ttft_ms")]
    pub ttft_ms: f64,
    /// Synthetic inter-token latency in milliseconds (>= 0).
    #[serde(default = "default_itl_ms")]
    pub itl_ms: f64,
    /// Coefficient of variation for a future seeded per-request jitter draw.
    /// `0.0` (default) keeps the fabricated timing byte-deterministic.
    #[serde(default)]
    pub jitter_cv: f64,
    /// Root seed for the future jitter draw. Unused while `jitter_cv == 0.0`.
    #[serde(default)]
    pub seed: u64,
}

impl DryRunTransportConfigV2 {
    /// Project the validated config into the `Copy` params carried through
    /// [`crate::engine::online_execution::classify_native_transport`].
    pub fn params(&self) -> DryRunParams {
        DryRunParams {
            ttft_ms: self.ttft_ms,
            itl_ms: self.itl_ms,
            jitter_cv: self.jitter_cv,
            seed: self.seed,
        }
    }
}

/// Analytic latency parameters carried by value into the execution leaf.
///
/// `Copy` so the [`DryRunNativeExecution`] binding can hold them and hand a fresh
/// [`FakeRequestExecutorFactory`] to each run without shared state.
#[derive(Debug, Clone, Copy)]
pub struct DryRunParams {
    /// Synthetic time-to-first-token in milliseconds.
    pub ttft_ms: f64,
    /// Synthetic inter-token latency in milliseconds.
    pub itl_ms: f64,
    /// Reserved coefficient of variation for a future seeded jitter draw.
    pub jitter_cv: f64,
    /// Reserved root seed for the future jitter draw.
    pub seed: u64,
}

/// Registered strict decoder for the always-built `dry_run` transport.
#[derive(Debug, Clone, Copy, Default)]
pub struct DryRunTransportFactoryV2;

impl RunnerTransportFactory for DryRunTransportFactoryV2 {
    fn descriptor(&self) -> &'static RunnerTransportDescriptor {
        &DRY_RUN_TRANSPORT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn crate::engine::registry::ValidatedTransportConfig>> {
        let config =
            strict_decode::<DryRunTransportConfigV2>(authored, "dry_run transport config")?;
        ensure!(
            config.ttft_ms >= 0.0 && config.ttft_ms.is_finite(),
            "dry_run ttft_ms must be a finite non-negative value, got {}",
            config.ttft_ms
        );
        ensure!(
            config.itl_ms >= 0.0 && config.itl_ms.is_finite(),
            "dry_run itl_ms must be a finite non-negative value, got {}",
            config.itl_ms
        );
        ensure!(
            config.jitter_cv >= 0.0 && config.jitter_cv.is_finite(),
            "dry_run jitter_cv must be a finite non-negative value, got {}",
            config.jitter_cv
        );
        Ok(Box::new(config))
    }

    fn native_execution(
        &self,
        config: &dyn ValidatedTransportConfig,
        _context: &RunnerRunContext,
    ) -> Result<Option<Arc<dyn NativeTransportExecution>>> {
        let config = ValidatedTransportConfig::as_any(config)
            .downcast_ref::<DryRunTransportConfigV2>()
            .ok_or_else(|| anyhow::anyhow!("dry_run transport received a non-dry_run config"))?;
        Ok(Some(Arc::new(DryRunNativeExecution {
            params: config.params(),
        })))
    }
}

/// Native execution binding for the built-in `dry_run` transport.
///
/// The fake leaf carries its analytic latency params by value and builds its own
/// [`FakeRequestExecutorFactory`], so `dry_run` needs no process-global
/// execution factory and no per-transport branch in the workloads — it is a
/// transport like any other. Readiness is skipped (no server) and the graph
/// workload is not yet supported (a fake whole-trace `Dispatcher` is a follow-up).
#[derive(Debug)]
pub struct DryRunNativeExecution {
    params: DryRunParams,
}

impl NativeTransportExecution for DryRunNativeExecution {
    fn executor_factory(&self) -> Arc<dyn RequestExecutorFactory> {
        Arc::new(FakeRequestExecutorFactory::new(self.params))
    }

    fn readiness_enabled(&self) -> bool {
        false
    }

    fn graph_transport_kind(&self) -> Result<GraphTransportKind> {
        anyhow::bail!(
            "the dry_run transport does not yet support the graph workload; use it with a scheduled (synthetic/file) dataset"
        )
    }

    fn validate_run(&self, _run: &AuthoredRunSpecV2, _context: &RunnerRunContext) -> Result<()> {
        // A dry run touches no server, so URL-scheme and readiness validation are
        // skipped; the fake leaf fabricates every outcome from the analytic model.
        Ok(())
    }

    fn provenance(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("transport".to_owned(), "dry_run".to_owned())])
    }
}

/// Register the always-built `dry_run` transport into a mutable runner registry.
pub fn register_dry_run_transport(registry: &mut AIPerfRegistry) -> Result<()> {
    registry.register_transport(Arc::new(DryRunTransportFactoryV2))
}

/// Execution-placement factory for the fake leaf.
///
/// Built inside `prepare_native_operation` from the run's [`DryRunParams`]; it
/// carries no process-global state, so a `dry_run` run needs no change to
/// [`crate::engine::execution_factories::RunnerExecutionFactories`].
#[derive(Debug, Clone, Copy)]
pub struct FakeRequestExecutorFactory {
    params: DryRunParams,
}

impl FakeRequestExecutorFactory {
    /// Build a factory that fabricates outcomes from `params`.
    pub fn new(params: DryRunParams) -> Self {
        Self { params }
    }
}

impl RequestExecutorFactory for FakeRequestExecutorFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        ensure!(
            config.workers > 0,
            "dry_run execution workers must be positive"
        );
        // Fabrication is CPU-only and single-observer: worker count is ignored,
        // and everything accumulates into one coordinator-reactor observer, so a
        // dry run has no thread-per-core setup cost.
        Ok(Rc::new(FakeRequestExecutor {
            clock: config.coordinator_clock,
            model: config.model,
            params: self.params,
            origin_ns: Cell::new(0),
            observer: RefCell::new(None),
        }))
    }
}

/// Fake [`RequestExecutor`]: synthesizes each request's timing analytically and
/// drives the same [`NativeMetricsObserver`] the real HTTP path drives.
struct FakeRequestExecutor {
    clock: Rc<dyn Clock>,
    model: String,
    params: DryRunParams,
    origin_ns: Cell<i64>,
    observer: RefCell<Option<Rc<NativeMetricsObserver>>>,
}

impl FakeRequestExecutor {
    fn observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.observer.borrow().clone().ok_or_else(|| {
            anyhow::anyhow!("dry_run measurement was not configured before dispatch")
        })
    }

    /// Absolute-ns → relative-to-origin milliseconds, the unit `on_arrival` /
    /// `on_token` expect.
    fn rel_ms(&self, absolute_ns: i64) -> f64 {
        (absolute_ns - self.origin_ns.get()) as f64 / 1_000_000.0
    }
}

#[async_trait(?Send)]
impl RequestExecutor for FakeRequestExecutor {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        self.origin_ns.set(start_ns);
        Ok(())
    }

    fn supports_response_streaming(&self) -> bool {
        // The scheduled dispatcher only calls `execute_measured` (non-streaming);
        // fabricated per-token timing lives in the observer regardless of the
        // wire streaming flag, so live-frame forwarding is unnecessary.
        false
    }

    fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: None,
            model: Some(self.model.clone()),
        }
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        self.origin_ns.set(origin_ns);
        *self.observer.borrow_mut() = Some(Rc::new(NativeMetricsObserver::new(
            self.clock.clone(),
            origin_ns,
            config,
        )));
        Ok(())
    }

    async fn execute_measured(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        let observer = self.observer()?;
        let uuid = turn.request.uuid;
        let isl = context.input_length as u64;
        let osl = context.requested_output_length;

        let ttft_ns = (self.params.ttft_ms * 1_000_000.0).round() as i64;
        let itl_ns = (self.params.itl_ms * 1_000_000.0).round() as i64;
        // Dispatch begins at the scheduled arrival: a dry run adds no queueing
        // delay, so start == arrival and request_latency == ttft + (osl-1)*itl.
        let start_abs = self.origin_ns.get() + (context.arrival_ms * 1_000_000.0).round() as i64;
        let recv_start_abs = start_abs + ttft_ns;
        let token_abs = |index: usize| start_abs + ttft_ns + (index as i64) * itl_ns;
        let end_abs = if osl > 0 {
            token_abs(osl - 1)
        } else {
            recv_start_abs
        };

        // Mirror the real observer event sequence (dispatch_measured).
        observer.register_metadata(uuid, context.metadata.clone());
        observer.on_arrival(uuid, context.arrival_ms, isl as usize, osl);
        for index in 0..osl {
            let at_ms = self.rel_ms(token_abs(index));
            if index == 0 {
                on_first_token(ttft_ns);
            }
            observer.on_token(uuid, at_ms);
        }
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(isl as usize),
                completion_tokens: Some(osl),
                total_tokens: Some(isl as usize + osl),
                ..ObservedUsage::default()
            },
        );
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);
        observer.record_response(
            uuid,
            NativeResponseMetadata {
                start_ns: Some(start_abs),
                end_ns: Some(end_abs),
                prompt_tokens: Some(isl),
                completion_tokens: Some(osl as u64),
                http: HttpTrace::default(),
            },
        );

        // Build the raw HTTP exchange record consumed by profile_export.jsonl /
        // raw exporters: one synthetic text chunk per generated token, stamped at
        // the same absolute perf_ns the observer saw.
        let response_text = synthetic_text(osl);
        let request_payload = turn
            .request
            .request_body_bytes
            .clone()
            .unwrap_or_else(Bytes::new);
        let responses = (0..osl)
            .map(|index| {
                Response::Text(TextResponse {
                    perf_ns: token_abs(index),
                    text: SYNTHETIC_TOKEN.to_string(),
                    body: Bytes::new(),
                    content_type: Some("text/event-stream".to_string()),
                })
            })
            .collect::<Vec<_>>();
        let record = RequestRecord {
            start_ns: start_abs,
            request_body: request_payload.clone(),
            end_ns: Some(end_abs),
            recv_start_ns: Some(recv_start_abs),
            status: Some(200),
            responses,
            ..RequestRecord::started(start_abs)
        };

        let live_record = context
            .wants_live_record
            .then(|| {
                if context.consume_record {
                    observer.drain_terminal_record(uuid, 0)
                } else {
                    observer.snapshot_record(uuid, 0)
                }
            })
            .flatten();

        Ok(MeasuredOutcome {
            result: DispatchResult {
                outcome: TurnDispatchOutcome {
                    start_ns: start_abs,
                    end_ns: end_abs,
                    terminal: ReplayTerminalStatus::Completed,
                    response_text: response_text.clone(),
                    model_response: ModelResponseMetadata {
                        content: Some(response_text),
                        finish_reason: Some("stop".to_string()),
                        ..ModelResponseMetadata::default()
                    },
                    prompt_tokens: Some(isl),
                    completion_tokens: Some(osl as u64),
                    http: HttpTrace::default(),
                },
                request_payload,
                record,
            },
            live_record,
        })
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        match self.observer.borrow_mut().take() {
            Some(observer) => Ok(observer
                .take_finalizer_at(end_ns)
                .finish_with_records()
                .records),
            None => Ok(Vec::new()),
        }
    }
}

/// One synthetic output-token surface form. Cosmetic: output-length metrics come
/// from the fabricated observer usage, not from parsing this text.
const SYNTHETIC_TOKEN: &str = "x";

/// Build the joined synthetic assistant text for `osl` tokens.
fn synthetic_text(osl: usize) -> String {
    SYNTHETIC_TOKEN.repeat(osl)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::clock::{RealClock, RealClockAnchor};
    use crate::endpoints::{EndpointId, EndpointKey};
    use crate::http::{HttpRequest, PreparedHttpEndpoint, TransportSinkConfig};
    use crate::multiturn::{PreparedEndpointReference, TurnDataPolicy};

    fn build_executor(params: DryRunParams) -> Rc<dyn RequestExecutor> {
        let anchor = RealClockAnchor::now();
        let clock: Rc<dyn Clock> = RealClock::from_anchor(anchor);
        let executor = FakeRequestExecutorFactory::new(params)
            .build(ExecutionBackendConfig {
                workers: 1,
                coordinator_clock: clock.clone(),
                real_clock_anchor: anchor,
                base_urls: vec!["http://dry-run.invalid".to_string()],
                model: "fixture-model".to_string(),
                transport: TransportSinkConfig::default(),
                prepared_endpoints: None,
            })
            .expect("build fake executor");
        let origin_ns = clock.now_ns();
        executor.set_run_origin(origin_ns).unwrap();
        executor
            .configure_measurement(MetricsConfig::default(), origin_ns)
            .unwrap();
        executor
    }

    fn fixture_turn(uuid: Uuid, isl: usize, osl: usize) -> PreparedTurn {
        PreparedTurn {
            request: HttpRequest {
                uuid,
                input_length: isl,
                max_output_tokens: osl,
                prompt_text: None,
                request_body: None,
                request_body_bytes: None,
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: None,
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
            },
            model: "fixture-model".to_string(),
            endpoint: PreparedHttpEndpoint::Prepared(PreparedEndpointReference {
                key: EndpointKey::from_index(0),
                endpoint_id: EndpointId::new("chat").unwrap(),
            }),
            endpoint_aware: false,
            data_policy: TurnDataPolicy::ordinary(),
        }
    }

    fn context(isl: usize, osl: usize) -> MeasuredContext {
        MeasuredContext {
            arrival_ms: 5.0,
            input_length: isl,
            requested_output_length: osl,
            metadata: crate::metrics::RequestMetricMetadata::default(),
            wants_live_record: false,
            consume_record: false,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fabricates_exact_analytic_timing() {
        const ISL: usize = 20;
        const OSL: usize = 12;
        let params = DryRunParams {
            ttft_ms: 10.0,
            itl_ms: 2.0,
            jitter_cv: 0.0,
            seed: 0,
        };
        let ttft_ns = 10_000_000_i64;
        let itl_ns = 2_000_000_i64;
        let executor = build_executor(params);
        let uuid = Uuid::new_v4();

        let local = tokio::task::LocalSet::new();
        let outcome = local
            .run_until(executor.execute_measured(
                fixture_turn(uuid, ISL, OSL),
                context(ISL, OSL),
                &|_| {},
            ))
            .await
            .expect("fabricated dispatch");

        // Backend-neutral outcome: exact analytic timing and usage.
        let o = &outcome.result.outcome;
        assert_eq!(o.end_ns - o.start_ns, ttft_ns + (OSL as i64 - 1) * itl_ns);
        assert_eq!(o.prompt_tokens, Some(ISL as u64));
        assert_eq!(o.completion_tokens, Some(OSL as u64));
        assert_eq!(o.terminal, ReplayTerminalStatus::Completed);

        // Raw record: one synthetic chunk per token, first at start + ttft.
        let record = &outcome.result.record;
        assert_eq!(record.responses.len(), OSL);
        assert_eq!(record.responses[0].perf_ns() - record.start_ns, ttft_ns);
        assert_eq!(
            record.responses[OSL - 1].perf_ns() - record.start_ns,
            ttft_ns + (OSL as i64 - 1) * itl_ns
        );
        assert_eq!(record.status, Some(200));

        // Drained RecordIngest (the metrics/summary source) carries OSL tokens.
        let drained = executor.drain_records(0).expect("drain");
        assert_eq!(drained.len(), 1);
        let (drained_uuid, ingest) = &drained[0];
        assert_eq!(*drained_uuid, uuid);
        assert_eq!(ingest.tokens.output, Some(OSL as u64));
        assert_eq!(ingest.token_arrival_ns.len(), OSL);
    }

    #[test]
    fn validate_rejects_negative_latency() {
        let factory = DryRunTransportFactoryV2;
        let bad = serde_json::value::to_raw_value(&serde_json::json!({"itl_ms": -1.0}))
            .expect("raw value");
        assert!(
            factory
                .validate(&bad, &WorkloadRequirements::default())
                .is_err(),
            "negative itl_ms must be rejected"
        );
        let good = serde_json::value::to_raw_value(&serde_json::json!({})).expect("raw value");
        assert!(
            factory
                .validate(&good, &WorkloadRequirements::default())
                .is_ok(),
            "empty dry_run config must validate with defaults"
        );
    }
}
