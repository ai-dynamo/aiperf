// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-owned `dry_run` transport execution leaf.
//!
//! This module is the implementation leaf for the built-in `dry_run` transport.
//! It contains the analytic latency model, the fake fabrication core, the
//! scheduled-path [`FakeRequestExecutor`], and the graph-path [`FakeDispatcher`].
//! The host adapter in `engine::dry_run` owns the validated config, the
//! [`crate::engine::registry::TransportFactory`] implementation, and the
//! [`crate::engine::registry::NativeTransportExecution`] binding that wires these
//! together at bootstrap.
//!
//! See [`crate::engine::dry_run`] for the fabrication contract and clock-mode
//! documentation.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{ObservedUsage, RequestObserver};
use uuid::Uuid;

use crate::clock::Clock;
use crate::engine::protocol::HopRouting;
use crate::engine::turn_execution::{ExecutionBackendConfig, RequestExecutorFactory, pick_worker};
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest, RequestTrace};
use crate::multiturn::TurnToSend;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator};
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use crate::transport::core::{DispatchResult, ErrorDetails, MeasuredContext, MeasuredOutcome};
use crate::transport::core::{PreparedTurn, RequestExecutor};
use crate::transport::core::{RequestRecord, Response, TextResponse};

/// Which clock drives the dry run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DryRunClock {
    /// Real wall clock for pacing, schedules, and duration bounds.
    Real,
    /// Virtual `SimClock` for deterministic single-reactor execution.
    #[default]
    Sim,
}

/// Which analytic latency curve the fake leaf uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DryRunLatencyModel {
    /// Linear model: `base + token terms + concurrency terms`.
    #[default]
    Linear,
    /// Dynamo `PerfModel::Polynomial`: TTFT from prefill tokens and ITL from
    /// KV-cache utilization.
    AiconfiguratorPolynomial,
    /// Reproduce the trace's pre-known recorded `api_time` per request as the
    /// total response latency. Falls back to the linear analytic model for any
    /// request lacking a recorded api_time.
    Recorded,
}

/// In-flight count used by analytic contention terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VirtualContentionScope {
    /// Count all requests executing on every virtual worker.
    #[default]
    Global,
    /// Count only requests assigned to the selected virtual worker.
    WorkerLocal,
}

/// Optional latency profile for one logical worker.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunVirtualWorkerProfileV2 {
    /// Zero-based worker index.
    pub worker: usize,
    /// Multiplier applied to analytic TTFT.
    #[serde(default = "one")]
    pub ttft_multiplier: f64,
    /// Multiplier applied to analytic ITL.
    #[serde(default = "one")]
    pub itl_multiplier: f64,
}

/// Strict single-reactor virtual worker configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunVirtualWorkersConfigV2 {
    /// Enable logical worker placement.
    #[serde(default)]
    pub enabled: bool,
    /// Logical width; defaults to authored `runtime.workers`.
    #[serde(default)]
    pub width: Option<usize>,
    /// Scope of the contention input supplied to the analytic model.
    #[serde(default)]
    pub contention_scope: VirtualContentionScope,
    /// Optional per-worker latency multipliers.
    #[serde(default)]
    pub profiles: Vec<DryRunVirtualWorkerProfileV2>,
}

fn one() -> f64 {
    1.0
}

/// Analytic latency parameters carried by value into the execution leaf.
///
/// `Copy` so the [`crate::engine::dry_run::DryRunNativeExecution`] binding can
/// hold them and hand a fresh [`FakeRequestExecutorFactory`] to each run without
/// shared state.
#[derive(Debug, Clone, Copy)]
pub struct DryRunParams {
    /// Base time-to-first-token in milliseconds.
    pub ttft_ms: f64,
    /// Base inter-token latency in milliseconds.
    pub itl_ms: f64,
    /// Prefill cost per input token (ms).
    pub ttft_per_isl_token_ms: f64,
    /// Super-linear prefill contention coefficient (ms per inflight²).
    pub ttft_concurrency_quad_ms: f64,
    /// Decode cost per output token (ms).
    pub itl_per_osl_token_ms: f64,
    /// Linear decode contention coefficient (ms per inflight).
    pub itl_concurrency_lin_ms: f64,
    /// Lognormal TTFT jitter coefficient of variation.
    pub ttft_jitter_cv: f64,
    /// Lognormal ITL jitter coefficient of variation.
    pub itl_jitter_cv: f64,
    /// Root seed for the per-request jitter draw.
    pub seed: u64,
    /// Selected analytic latency curve.
    pub latency_model: DryRunLatencyModel,
    /// KV-cache utilization for the polynomial decode curve.
    pub kv_utilization: f64,
    /// Which clock drives the run.
    pub clock: DryRunClock,
}

impl DryRunParams {
    /// Compute the effective `(ttft_ns, itl_ns)` for one request from the
    /// analytic model.
    pub(crate) fn effective_latencies_ns(
        &self,
        isl: usize,
        osl: usize,
        active_inflight: usize,
        ordinal: u64,
    ) -> (i64, i64) {
        let active = active_inflight as f64;
        let (base_ttft_ms, base_itl_ms) = match self.latency_model {
            DryRunLatencyModel::Linear | DryRunLatencyModel::Recorded => (
                self.ttft_ms
                    + self.ttft_per_isl_token_ms * isl as f64
                    + self.ttft_concurrency_quad_ms * active * active,
                self.itl_ms
                    + self.itl_per_osl_token_ms * osl as f64
                    + self.itl_concurrency_lin_ms * active,
            ),
            DryRunLatencyModel::AiconfiguratorPolynomial => (
                aic_polynomial_prefill_ms(isl as f64),
                aic_polynomial_decode_ms(self.kv_utilization),
            ),
        };
        let ttft_ms = base_ttft_ms
            * lognormal_jitter(
                &mut seeded_rng(self.seed, ordinal ^ 0x11),
                self.ttft_jitter_cv,
            );
        let itl_ms = base_itl_ms
            * lognormal_jitter(
                &mut seeded_rng(self.seed, ordinal ^ 0x22),
                self.itl_jitter_cv,
            );
        (ms_to_ns(ttft_ms), ms_to_ns(itl_ns))
    }
}

/// Per-request recorded timing carried from a recorded trace into the fake leaf.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RecordedLatency {
    /// Recorded total response latency (api_time) in nanoseconds.
    pub(crate) api_time_ns: Option<i64>,
    /// Recorded time-to-first-token in nanoseconds, when the trace supplies it.
    pub(crate) ttft_ns: Option<i64>,
}

/// Split a recorded total `api_time` (nanoseconds) into `(ttft_ns, itl_ns)` for
/// a stream of `osl` output tokens so the fabricated request ends exactly at
/// `total_ns`.
pub(crate) fn recorded_latencies_ns(
    total_ns: i64,
    ttft_ns: Option<i64>,
    osl: usize,
) -> (i64, i64) {
    let total = total_ns.max(0);
    if osl <= 1 {
        return (total, 0);
    }
    let steps = osl as i64 - 1;
    match ttft_ns {
        Some(ttft) => {
            let ttft = ttft.clamp(0, total);
            (ttft, (total - ttft) / steps)
        }
        None => {
            let per = total / osl as i64;
            (per, per)
        }
    }
}

/// `PerfModel::Polynomial` prefill curve in milliseconds.
pub(crate) fn aic_polynomial_prefill_ms(prefill_tokens: f64) -> f64 {
    if prefill_tokens <= 0.0 {
        return 0.0;
    }
    (4.209_989e-7 * prefill_tokens * prefill_tokens + 1.518_344e-2 * prefill_tokens + 1.650_142e1)
        .max(0.0)
}

/// `PerfModel::Polynomial` decode curve in milliseconds with a 1 ms floor.
pub(crate) fn aic_polynomial_decode_ms(active_perc: f64) -> f64 {
    (-25.74 * active_perc * active_perc + 54.01 * active_perc + 5.74).max(1.0)
}

/// Convert milliseconds to rounded, non-negative nanoseconds.
pub(crate) fn ms_to_ns(ms: f64) -> i64 {
    (ms * 1_000_000.0).max(0.0).round() as i64
}

/// Deterministic per-request RNG derived from a root seed and salt.
fn seeded_rng(seed: u64, salt: u64) -> ConfiguredRandomGenerator {
    ConfiguredRandomGenerator::from_seed_or_entropy(Some(
        seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15),
    ))
}

/// Mean-preserving lognormal jitter; non-positive `cv` yields `1.0`.
fn lognormal_jitter(rng: &mut ConfiguredRandomGenerator, cv: f64) -> f64 {
    if cv <= 0.0 {
        return 1.0;
    }
    let sigma = (1.0 + cv * cv).ln().sqrt();
    let u1 = rng.random().max(1e-12);
    let u2 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (sigma * z - 0.5 * sigma * sigma).exp()
}

/// One synthetic output-token surface form.
pub(crate) const SYNTHETIC_TOKEN: &str = "x";

/// Build the joined synthetic assistant text for `osl` tokens.
pub(crate) fn synthetic_text(osl: usize) -> String {
    SYNTHETIC_TOKEN.repeat(osl)
}

/// Execution-placement factory for the fake leaf.
///
/// Built inside `prepare_native_operation` from the run's [`DryRunParams`]; it
/// carries no process-global state, so a `dry_run` run needs no change to
/// [`crate::engine::execution_factories::ExecutionFactories`].
#[derive(Debug, Clone)]
pub struct FakeRequestExecutorFactory {
    pub(crate) params: DryRunParams,
    pub(crate) virtual_workers: DryRunVirtualWorkersConfigV2,
}

impl FakeRequestExecutorFactory {
    /// Build a factory that fabricates outcomes from `params`.
    pub fn new(params: DryRunParams) -> Self {
        Self {
            params,
            virtual_workers: DryRunVirtualWorkersConfigV2::default(),
        }
    }

    pub(crate) fn with_virtual_workers(
        params: DryRunParams,
        virtual_workers: DryRunVirtualWorkersConfigV2,
    ) -> Self {
        Self {
            params,
            virtual_workers,
        }
    }
}

impl RequestExecutorFactory for FakeRequestExecutorFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        ensure!(
            config.workers > 0,
            "dry_run execution workers must be positive"
        );
        let placement = config
            .virtual_worker_width
            .map(|width| {
                ensure!(width > 0, "dry_run virtual worker width must be positive");
                for profile in &self.virtual_workers.profiles {
                    ensure!(
                        profile.worker < width,
                        "dry_run virtual worker profile {} is outside width {width}",
                        profile.worker
                    );
                }
                Ok(VirtualPlacement::new(
                    width,
                    config.hop_routing,
                    self.virtual_workers.contention_scope,
                    &self.virtual_workers.profiles,
                ))
            })
            .transpose()?;
        Ok(Rc::new(FakeRequestExecutor {
            core: FakeFabricator::new(config.coordinator_clock, config.model, self.params, 0),
            observer: RefCell::new(None),
            placement,
        }))
    }
}

struct VirtualPlacement {
    width: usize,
    routing: HopRouting,
    contention_scope: VirtualContentionScope,
    next_worker: Cell<usize>,
    send_seq: Cell<u64>,
    assignment_index: Cell<u64>,
    inflight: Vec<crate::engine::turn_execution::WorkerLoad>,
    sticky: RefCell<HashMap<String, usize>>,
    profiles: Vec<(f64, f64)>,
}

impl VirtualPlacement {
    fn new(
        width: usize,
        routing: HopRouting,
        contention_scope: VirtualContentionScope,
        profiles: &[DryRunVirtualWorkerProfileV2],
    ) -> Self {
        let mut multipliers = vec![(1.0, 1.0); width];
        for profile in profiles {
            multipliers[profile.worker] = (profile.ttft_multiplier, profile.itl_multiplier);
        }
        Self {
            width,
            routing,
            contention_scope,
            next_worker: Cell::new(0),
            send_seq: Cell::new(0),
            assignment_index: Cell::new(0),
            inflight: (0..width)
                .map(|_| crate::engine::turn_execution::WorkerLoad::default())
                .collect(),
            sticky: RefCell::new(HashMap::new()),
            profiles: multipliers,
        }
    }

    fn assign(&self, correlation_id: Option<&str>, is_final_turn: bool) -> (usize, u64) {
        let mut cursor = self.next_worker.get();
        let worker = pick_worker(
            self.routing,
            self.width,
            correlation_id,
            is_final_turn,
            &self.inflight,
            &mut self.sticky.borrow_mut(),
            &mut cursor,
        );
        self.next_worker.set(cursor);
        let chosen = &self.inflight[worker];
        chosen.sent.set(chosen.sent.get() + 1);
        self.send_seq.set(self.send_seq.get() + 1);
        chosen.last_sent.set(self.send_seq.get());
        let index = self.assignment_index.get();
        self.assignment_index.set(index.wrapping_add(1));
        (worker, index)
    }
}

struct VirtualInflightGuard<'a> {
    slot: &'a Cell<usize>,
}

impl<'a> VirtualInflightGuard<'a> {
    fn new(slot: &'a Cell<usize>) -> Self {
        slot.set(slot.get().saturating_add(1));
        Self { slot }
    }
}

impl Drop for VirtualInflightGuard<'_> {
    fn drop(&mut self) {
        self.slot.set(self.slot.get().saturating_sub(1));
    }
}

/// Shared analytic-fabrication core used by both the scheduled
/// [`FakeRequestExecutor`] and the graph [`FakeDispatcher`].
struct FakeFabricator {
    clock: Rc<dyn Clock>,
    model: String,
    params: DryRunParams,
    origin_ns: Cell<i64>,
    inflight: Cell<usize>,
    ordinal: Cell<u64>,
}

struct FabricationRequest<'a> {
    observer: &'a dyn RequestObserver,
    uuid: Uuid,
    isl: u64,
    osl: usize,
    start_abs: i64,
    cancel_after_ns: Option<i64>,
    request_payload: Bytes,
    recorded: RecordedLatency,
    contention_override: Option<usize>,
    latency_multipliers: (f64, f64),
    on_first_token: &'a dyn Fn(i64),
}

impl FakeFabricator {
    fn new(clock: Rc<dyn Clock>, model: String, params: DryRunParams, origin_ns: i64) -> Self {
        Self {
            clock,
            model,
            params,
            origin_ns: Cell::new(origin_ns),
            inflight: Cell::new(0),
            ordinal: Cell::new(0),
        }
    }

    fn set_origin(&self, origin_ns: i64) {
        self.origin_ns.set(origin_ns);
    }

    fn rel_ms(&self, absolute_ns: i64) -> f64 {
        (absolute_ns - self.origin_ns.get()) as f64 / 1_000_000.0
    }

    async fn fabricate(&self, request: FabricationRequest<'_>) -> DispatchResult {
        let FabricationRequest {
            observer,
            uuid,
            isl,
            osl,
            start_abs,
            cancel_after_ns,
            request_payload,
            recorded,
            contention_override,
            latency_multipliers,
            on_first_token,
        } = request;
        let _global_inflight_guard = VirtualInflightGuard::new(&self.inflight);
        let active_inflight = self.inflight.get();
        let ordinal = self.ordinal.get();
        self.ordinal.set(ordinal + 1);
        let recorded_total = (self.params.latency_model == DryRunLatencyModel::Recorded)
            .then_some(recorded.api_time_ns)
            .flatten();
        let (ttft_ns, itl_ns) = match recorded_total {
            Some(total) => recorded_latencies_ns(total, recorded.ttft_ns, osl),
            None => self.params.effective_latencies_ns(
                isl as usize,
                osl,
                contention_override.unwrap_or(active_inflight),
                ordinal,
            ),
        };
        let ttft_ns = (ttft_ns as f64 * latency_multipliers.0).round() as i64;
        let itl_ns = (itl_ns as f64 * latency_multipliers.1).round() as i64;
        let recv_start_abs = start_abs + ttft_ns;
        let token_abs = |index: usize| start_abs + ttft_ns + (index as i64) * itl_ns;
        let end_abs = if let Some(total) = recorded_total {
            start_abs + total.max(0)
        } else if osl > 0 {
            token_abs(osl - 1)
        } else {
            recv_start_abs
        };
        let cancellation_abs = cancel_after_ns
            .map(|delay| start_abs.saturating_add(delay.max(0)))
            .filter(|deadline| *deadline < end_abs);
        let virtual_time = self.clock.is_virtual();
        for index in 0..osl {
            if cancellation_abs.is_some_and(|deadline| token_abs(index) > deadline) {
                break;
            }
            if virtual_time {
                let wait = token_abs(index) - self.clock.now_ns();
                if wait > 0 {
                    self.clock.clone().sleep(wait).await;
                }
            }
            if index == 0 {
                on_first_token(ttft_ns);
            }
            observer.on_token(uuid, self.rel_ms(token_abs(index)));
        }
        if let Some(cancellation_abs) = cancellation_abs {
            if virtual_time {
                let wait = cancellation_abs - self.clock.now_ns();
                if wait > 0 {
                    self.clock.clone().sleep(wait).await;
                }
            }
            let cancel_after_ns = cancel_after_ns.unwrap_or_default().max(0);
            let error = ErrorDetails::cancelled(format!(
                "RequestCancellationError: request cancelled {cancel_after_ns}ns after being sent"
            ));
            observer.on_usage(uuid, ObservedUsage::default());
            observer.on_terminal(uuid, ReplayTerminalStatus::Canceled);
            let record = RequestRecord {
                start_ns: start_abs,
                end_ns: Some(cancellation_abs),
                error: Some(error),
                cancellation_ns: Some(cancellation_abs),
                ..RequestRecord::started(start_abs)
            };
            return DispatchResult {
                outcome: TurnDispatchOutcome {
                    start_ns: start_abs,
                    end_ns: cancellation_abs,
                    terminal: ReplayTerminalStatus::Canceled,
                    response_text: String::new(),
                    model_response: ModelResponseMetadata {
                        error_kind: Some("RequestCancellationError".to_string()),
                        error_message: Some(format!(
                            "RequestCancellationError: request cancelled {cancel_after_ns}ns after being sent"
                        )),
                        ..ModelResponseMetadata::default()
                    },
                    prompt_tokens: None,
                    completion_tokens: None,
                    http: RequestTrace::default(),
                },
                request_payload,
                record,
            };
        }
        if virtual_time {
            let wait = end_abs - self.clock.now_ns();
            if wait > 0 {
                self.clock.clone().sleep(wait).await;
            }
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
        let response_text = synthetic_text(osl);
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
            end_ns: Some(end_abs),
            recv_start_ns: Some(recv_start_abs),
            status: Some(200),
            responses,
            ..RequestRecord::started(start_abs)
        };
        DispatchResult {
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
                http: RequestTrace::default(),
            },
            request_payload,
            record,
        }
    }
}

/// Fake [`RequestExecutor`] (scheduled path).
struct FakeRequestExecutor {
    core: FakeFabricator,
    observer: RefCell<Option<Rc<NativeMetricsObserver>>>,
    placement: Option<VirtualPlacement>,
}

impl FakeRequestExecutor {
    fn observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.observer.borrow().clone().ok_or_else(|| {
            anyhow::anyhow!("dry_run measurement was not configured before dispatch")
        })
    }
}

#[async_trait(?Send)]
impl RequestExecutor for FakeRequestExecutor {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        self.core.set_origin(start_ns);
        Ok(())
    }

    fn supports_response_streaming(&self) -> bool {
        false
    }

    fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: None,
            model: Some(self.core.model.clone()),
        }
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        self.core.set_origin(origin_ns);
        *self.observer.borrow_mut() = Some(Rc::new(NativeMetricsObserver::new(
            self.core.clock.clone(),
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
        let start_abs =
            self.core.origin_ns.get() + (context.arrival_ms * 1_000_000.0).round() as i64;
        let request_payload = match &turn.request.body {
            Some(body) => body.to_wire()?,
            None => Bytes::new(),
        };
        let mut metadata = context.metadata.clone();
        let (_worker_guard, contention_override, multipliers) = if let Some(placement) =
            &self.placement
        {
            let (worker, assignment_index) = placement.assign(
                metadata.correlation_id.as_deref(),
                turn.request.is_final_turn,
            );
            metadata.worker_id = Some(format!("dry-run-{worker}").into());
            metadata.worker_assignment_index = Some(assignment_index);
            let guard = VirtualInflightGuard::new(&placement.inflight[worker].inflight);
            let contention = (placement.contention_scope == VirtualContentionScope::WorkerLocal)
                .then(|| placement.inflight[worker].inflight.get());
            (Some(guard), contention, placement.profiles[worker])
        } else {
            (None, None, (1.0, 1.0))
        };
        observer.register_metadata(uuid, metadata);
        observer.on_arrival(uuid, context.arrival_ms, isl as usize, osl);
        let recorded = RecordedLatency {
            api_time_ns: turn.request.recorded_api_time_ns,
            ttft_ns: turn.request.recorded_ttft_ns,
        };
        let result = self
            .core
            .fabricate(FabricationRequest {
                observer: &*observer,
                uuid,
                isl,
                osl,
                start_abs,
                cancel_after_ns: turn.request.cancel_after_ns,
                request_payload,
                recorded,
                contention_override,
                latency_multipliers: multipliers,
                on_first_token,
            })
            .await;
        observer.record_response(
            uuid,
            NativeResponseMetadata {
                start_ns: Some(result.outcome.start_ns),
                end_ns: Some(result.outcome.end_ns),
                prompt_tokens: Some(isl),
                completion_tokens: Some(osl as u64),
                http: RequestTrace::default(),
            },
        );
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
            result,
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

/// Fake [`Dispatcher`] (graph path).
pub(crate) struct FakeDispatcher {
    core: FakeFabricator,
}

impl FakeDispatcher {
    pub(crate) fn new(core: FakeFabricator) -> Self {
        Self { core }
    }

    pub(crate) fn from_params(
        clock: Rc<dyn Clock>,
        model: String,
        params: DryRunParams,
        run_origin_ns: i64,
    ) -> Self {
        Self {
            core: FakeFabricator::new(clock, model, params, run_origin_ns),
        }
    }
}

#[async_trait(?Send)]
impl crate::transport::core::Dispatcher for FakeDispatcher {
    async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        let uuid = turn.request.uuid;
        let isl = turn.request.input_length as u64;
        let osl = turn.request.max_output_tokens;
        let start_abs = self.core.clock.now_ns();
        let request_payload = match &turn.request.body {
            Some(body) => body.to_wire()?,
            None => Bytes::new(),
        };
        let recorded = RecordedLatency {
            api_time_ns: turn.request.recorded_api_time_ns,
            ttft_ns: turn.request.recorded_ttft_ns,
        };
        Ok(self
            .core
            .fabricate(FabricationRequest {
                observer,
                uuid,
                isl,
                osl,
                start_abs,
                cancel_after_ns: turn.request.cancel_after_ns,
                request_payload,
                recorded,
                contention_override: None,
                latency_multipliers: (1.0, 1.0),
                on_first_token,
            })
            .await)
    }

    fn inference_dimensions(
        &self,
        _request: &crate::transport::core::Request,
    ) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: Some("dry_run://sim".to_string()),
            model: Some(self.core.model.clone()),
        }
    }

    fn supports_response_streaming(&self) -> bool {
        false
    }
}
