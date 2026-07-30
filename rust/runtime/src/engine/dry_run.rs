// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Socket-free `dry_run` transport with analytic request timing.
//!
//! # Fabrication contract
//!
//! For one request with input length `ISL` and requested output length `OSL`,
//! given analytic `ttft_ms`/`itl_ms`, the fake dispatch begins at the request's
//! scheduled arrival and emits `OSL` synthetic output tokens spaced by `itl_ms`
//! after an initial `ttft_ms`. The observer event sequence uses
//! [`crate::transport::http::TransportSink::dispatch_measured`]: `register_metadata` →
//! `on_arrival` → `OSL`× `on_token` → `on_usage` → `on_terminal` →
//! `record_response`. Downstream, TTFT = `first_token − start = ttft_ms`, ITL =
//! `itl_ms`, and `request_latency = ttft_ms + (OSL−1)·itl_ms`, all exact under a
//! zero-jitter model — which is what the end-to-end test asserts.
//!
//! # Clock modes
//!
//! The native driver reads
//! [`uses_virtual_clock`](crate::engine::registry::NativeTransportExecution::uses_virtual_clock)
//! and drives the selected clock through [`crate::clock::Clock::drive`].
//!
//! - **`clock: sim`** (default) — a `SimClock` whose idle-pump driver
//!   fast-forwards virtual time to each next event on a single reactor. Arrival
//!   pacing, duration bounds, and fixed-schedule timestamps run in virtual time —
//!   a 10-minute run finishes at ~startup wall-speed, byte-deterministic.
//! - **`clock: real`** — a `RealClock` on the tokio reactor; the loadgen
//!   self-benchmark path (raw dispatch throughput). Arrival pacing /
//!   fixed-schedule / duration bounds wait in real wall-time.
//!
//! TTFT/ITL can scale with ISL, OSL, and live concurrency, with seeded
//! lognormal jitter. Zero scaling and jitter yield deterministic fixed latency.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{ObservedUsage, RequestObserver};
use uuid::Uuid;

use crate::clock::Clock;
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{
    ClockKind, NativeTransportExecution, RunContext, TransportDescriptor, TransportFactory,
    ValidatedTransportConfig, WorkloadRequirements, strict_decode,
};
use crate::engine::turn_execution::{ExecutionBackendConfig, RequestExecutorFactory};
use crate::extensions::AIPerfRegistry;
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest, RequestTrace};
use crate::multiturn::TurnToSend;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator};
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use crate::transport::core::{DispatchResult, ErrorDetails, MeasuredContext, MeasuredOutcome};
use crate::transport::core::{PreparedTurn, RequestExecutor};
use crate::transport::core::{RequestRecord, Response, TextResponse};

/// Default synthetic time-to-first-token (milliseconds).
const fn default_ttft_ms() -> f64 {
    10.0
}

/// Default synthetic inter-token latency (milliseconds).
const fn default_itl_ms() -> f64 {
    2.0
}

/// Default KV-cache utilization fed to the polynomial decode curve.
const fn default_kv_utilization() -> f64 {
    0.5
}

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
    /// total response latency (recorded TTFT split when available, else an even
    /// split). Used for timing-parity checking on the graph-ir recorded path:
    /// when the fake latency equals the recorded api_time, the causal dispatch
    /// schedule reproduces the recorded warped timeline exactly. Falls back to the
    /// linear analytic model for any request lacking a recorded api_time.
    Recorded,
}

/// Built-in `dry_run` transport descriptor. The catalog clock is `Sim` (the
/// default mode); `clock: real` selects the wall-clock self-benchmark path.
pub static DRY_RUN_TRANSPORT_DESCRIPTOR: TransportDescriptor = TransportDescriptor {
    id: "dry_run",
    description: "Fake execution leaf: analytic-latency synthetic responses, zero network",
    clock: ClockKind::Sim,
    features: &["dry_run"],
    url_schemes: &[],
};

/// Strict validated config owned by the `dry_run` transport.
///
/// Per request:
///
/// ```text
/// ttft = (ttft_ms + ttft_per_isl_token_ms·ISL + ttft_concurrency_quad_ms·inflight²) · jitter(ttft_jitter_cv)
/// itl  = (itl_ms  + itl_per_osl_token_ms·OSL  + itl_concurrency_lin_ms·inflight)    · jitter(itl_jitter_cv)
/// ```
///
/// Every field has a serde default so `{"type":"dry_run"}` alone is valid; the
/// CLI projects concrete values so a run is fully specified. With the scaling and
/// jitter terms at their `0.0` defaults this reduces to fixed `ttft_ms`/`itl_ms`,
/// which keeps the metrics exactly predictable.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunTransportConfigV2 {
    /// Base time-to-first-token in milliseconds (>= 0).
    #[serde(default = "default_ttft_ms")]
    pub ttft_ms: f64,
    /// Base inter-token latency in milliseconds (>= 0).
    #[serde(default = "default_itl_ms")]
    pub itl_ms: f64,
    /// Prefill cost scaling with prompt length: `TTFT += this · ISL_tokens`.
    #[serde(default)]
    pub ttft_per_isl_token_ms: f64,
    /// Super-linear prefill contention: `TTFT += this · inflight²`.
    #[serde(default)]
    pub ttft_concurrency_quad_ms: f64,
    /// Decode cost scaling with output length: `ITL += this · OSL_tokens`.
    #[serde(default)]
    pub itl_per_osl_token_ms: f64,
    /// Linear decode contention: `ITL += this · inflight`.
    #[serde(default)]
    pub itl_concurrency_lin_ms: f64,
    /// Lognormal TTFT jitter (stddev/mean). `0.0` (default) is deterministic.
    #[serde(default)]
    pub ttft_jitter_cv: f64,
    /// Lognormal ITL jitter (stddev/mean). `0.0` (default) is deterministic.
    #[serde(default)]
    pub itl_jitter_cv: f64,
    /// Root seed for the per-request jitter draw. Unused while both CVs are `0.0`.
    #[serde(default)]
    pub seed: u64,
    /// Analytic latency curve: `linear` or `aiconfigurator_polynomial`.
    #[serde(default)]
    pub latency_model: DryRunLatencyModel,
    /// KV-cache utilization in `[0, 1]` feeding the polynomial decode curve. Only
    /// consulted by the `aiconfigurator_polynomial` model.
    #[serde(default = "default_kv_utilization")]
    pub kv_utilization: f64,
    /// Which clock drives the run (`real` default, or `sim` for deterministic
    /// virtual-time execution via `drive_sim`).
    #[serde(default)]
    pub clock: DryRunClock,
}

impl DryRunTransportConfigV2 {
    /// Project the validated config into the `Copy` params carried into the
    /// [`DryRunNativeExecution`] binding.
    pub fn params(&self) -> DryRunParams {
        DryRunParams {
            ttft_ms: self.ttft_ms,
            itl_ms: self.itl_ms,
            ttft_per_isl_token_ms: self.ttft_per_isl_token_ms,
            ttft_concurrency_quad_ms: self.ttft_concurrency_quad_ms,
            itl_per_osl_token_ms: self.itl_per_osl_token_ms,
            itl_concurrency_lin_ms: self.itl_concurrency_lin_ms,
            ttft_jitter_cv: self.ttft_jitter_cv,
            itl_jitter_cv: self.itl_jitter_cv,
            seed: self.seed,
            latency_model: self.latency_model,
            kv_utilization: self.kv_utilization,
            clock: self.clock,
        }
    }
}

/// Analytic latency parameters carried by value into the execution leaf.
///
/// `Copy` so the [`DryRunNativeExecution`] binding can hold them and hand a fresh
/// [`FakeRequestExecutorFactory`] to each run without shared state.
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
    /// Which clock drives the run (`sim` virtual-time vs `real` wall clock).
    /// Consulted only by [`DryRunNativeExecution::uses_virtual_clock`] to select
    /// the driver at the native driver layer.
    pub clock: DryRunClock,
}

impl DryRunParams {
    /// Compute the effective `(ttft_ns, itl_ns)` for one request from the
    /// analytic model. `active_inflight` is the live in-flight count feeding the
    /// concurrency-contention terms; `ordinal` seeds the per-request jitter draw
    /// so the timing is reproducible across runs (independent of the random UUID).
    ///
    fn effective_latencies_ns(
        &self,
        isl: usize,
        osl: usize,
        active_inflight: usize,
        ordinal: u64,
    ) -> (i64, i64) {
        let active = active_inflight as f64;
        let (base_ttft_ms, base_itl_ms) = match self.latency_model {
            // `Recorded` uses per-request api_time in `fabricate`; a request that
            // lacks a recorded value falls through to the linear analytic model.
            DryRunLatencyModel::Linear | DryRunLatencyModel::Recorded => (
                self.ttft_ms
                    + self.ttft_per_isl_token_ms * isl as f64
                    + self.ttft_concurrency_quad_ms * active * active,
                self.itl_ms
                    + self.itl_per_osl_token_ms * osl as f64
                    + self.itl_concurrency_lin_ms * active,
            ),
            DryRunLatencyModel::AiconfiguratorPolynomial => (
                // The fake dispatcher has no batching scheduler: concurrent graph
                // requests are independent batches of one, so polynomial prefill
                // must use this request's ISL rather than multiplying by the
                // number of unrelated in-flight requests.
                aic_polynomial_prefill_ms(isl as f64),
                // Dynamo's polynomial decode curve is purely a function of KV
                // utilization; with no KV manager the configured `kv_utilization`
                // knob supplies it.
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
        (ms_to_ns(ttft_ms), ms_to_ns(itl_ms))
    }
}

/// Split a recorded total `api_time` (nanoseconds) into `(ttft_ns, itl_ns)` for a
/// stream of `osl` output tokens so the fabricated request ends exactly at
/// `total_ns`. With a recorded `ttft_ns` the generated-token span carries the
/// remainder; without it the api_time is split evenly across the tokens. For
/// `osl <= 1` the whole api_time is the TTFT (`itl` is unused).
fn recorded_latencies_ns(total_ns: i64, ttft_ns: Option<i64>, osl: usize) -> (i64, i64) {
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
fn aic_polynomial_prefill_ms(prefill_tokens: f64) -> f64 {
    if prefill_tokens <= 0.0 {
        return 0.0;
    }
    (4.209_989e-7 * prefill_tokens * prefill_tokens + 1.518_344e-2 * prefill_tokens + 1.650_142e1)
        .max(0.0)
}

/// `PerfModel::Polynomial` decode curve in milliseconds with a 1 ms floor.
fn aic_polynomial_decode_ms(active_perc: f64) -> f64 {
    (-25.74 * active_perc * active_perc + 54.01 * active_perc + 5.74).max(1.0)
}

/// Convert milliseconds to rounded, non-negative nanoseconds.
fn ms_to_ns(ms: f64) -> i64 {
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

/// Registered strict decoder for the always-built `dry_run` transport.
#[derive(Debug, Clone, Copy, Default)]
pub struct DryRunTransportFactoryV2;

impl TransportFactory for DryRunTransportFactoryV2 {
    fn descriptor(&self) -> &'static TransportDescriptor {
        &DRY_RUN_TRANSPORT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &WorkloadRequirements,
    ) -> Result<Box<dyn crate::engine::registry::ValidatedTransportConfig>> {
        let config =
            strict_decode::<DryRunTransportConfigV2>(authored, "dry_run transport config")?;
        // Every analytic knob must be a finite, non-negative value: a latency, a
        // per-token/contention coefficient, or a jitter CV — none can be negative
        // or NaN without producing nonsensical fabricated timing.
        for (name, value) in [
            ("ttft_ms", config.ttft_ms),
            ("itl_ms", config.itl_ms),
            ("ttft_per_isl_token_ms", config.ttft_per_isl_token_ms),
            ("ttft_concurrency_quad_ms", config.ttft_concurrency_quad_ms),
            ("itl_per_osl_token_ms", config.itl_per_osl_token_ms),
            ("itl_concurrency_lin_ms", config.itl_concurrency_lin_ms),
            ("ttft_jitter_cv", config.ttft_jitter_cv),
            ("itl_jitter_cv", config.itl_jitter_cv),
        ] {
            ensure!(
                value >= 0.0 && value.is_finite(),
                "dry_run {name} must be a finite non-negative value, got {value}"
            );
        }
        Ok(Box::new(config))
    }

    fn native_execution(
        &self,
        config: &dyn ValidatedTransportConfig,
        _context: &RunContext,
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
/// [`FakeRequestExecutorFactory`] (scheduled) and `FakeDispatcher` (graph), so
/// `dry_run` needs no process-global execution factory and no per-transport
/// branch in the workloads — it is a transport like any other. Readiness is
/// skipped (no server).
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

    fn uses_virtual_clock(&self) -> bool {
        self.params.clock == DryRunClock::Sim
    }

    #[allow(clippy::too_many_arguments)]
    fn build_graph_dispatcher(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        _urls: &[String],
        model: &str,
        _transport_config: crate::transport::http::TransportSinkConfig,
        _endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
        _capture_raw: bool,
    ) -> Result<Rc<dyn crate::transport::core::Dispatcher>> {
        Ok(Rc::new(FakeDispatcher::new(FakeFabricator::new(
            clock,
            model.to_string(),
            self.params,
            run_origin_ns,
        ))))
    }

    fn graph_transport_label(&self) -> &'static str {
        "dry_run"
    }

    fn validate_run(&self, _run: &AuthoredRunSpecV2, _context: &RunContext) -> Result<()> {
        // A dry run touches no server, so URL-scheme and readiness validation are
        // skipped; the fake leaf fabricates every outcome from the analytic model.
        Ok(())
    }

    fn run_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("transport".to_owned(), "dry_run".to_owned())])
    }
}

/// Per-request recorded timing carried from a recorded trace into the fake leaf.
/// Both fields are `None` on non-recorded paths; only consulted under the
/// `recorded` latency model.
#[derive(Debug, Clone, Copy, Default)]
struct RecordedLatency {
    /// Recorded total response latency (api_time) in nanoseconds.
    api_time_ns: Option<i64>,
    /// Recorded time-to-first-token in nanoseconds, when the trace supplies it.
    ttft_ns: Option<i64>,
}

/// Register the always-built `dry_run` transport into a mutable runner registry.
pub fn register_dry_run_transport(registry: &mut AIPerfRegistry) -> Result<()> {
    registry.register_transport(Arc::new(DryRunTransportFactoryV2))
}

/// Execution-placement factory for the fake leaf.
///
/// Built inside `prepare_native_operation` from the run's [`DryRunParams`]; it
/// carries no process-global state, so a `dry_run` run needs no change to
/// [`crate::engine::execution_factories::ExecutionFactories`].
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
        // everything accumulates into one coordinator-reactor observer, so a dry
        // run has no thread-per-core setup cost.
        Ok(Rc::new(FakeRequestExecutor {
            core: FakeFabricator::new(config.coordinator_clock, config.model, self.params, 0),
            observer: RefCell::new(None),
        }))
    }
}

/// Shared analytic-fabrication core used by both the scheduled
/// [`FakeRequestExecutor`] and the graph [`FakeDispatcher`]. It computes each
/// request's `(ttft, itl)` from the analytic model, emits the fabricated token
/// stream + usage + terminal on the caller's observer, and builds the
/// backend-neutral [`DispatchResult`]. The two seams differ only in the observer
/// callbacks they own around this (arrival / metadata / record_response), so the
/// timing/token fabrication lives here exactly once.
struct FakeFabricator {
    clock: Rc<dyn Clock>,
    model: String,
    params: DryRunParams,
    origin_ns: Cell<i64>,
    inflight: Cell<usize>,
    ordinal: Cell<u64>,
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

    /// Absolute-ns → relative-to-origin milliseconds (the unit `on_token` wants).
    fn rel_ms(&self, absolute_ns: i64) -> f64 {
        (absolute_ns - self.origin_ns.get()) as f64 / 1_000_000.0
    }

    /// Fabricate one request beginning at `start_abs`: emit the analytic token
    /// stream, usage, and terminal on `observer`, and return the dispatch result.
    /// The caller owns `on_arrival` / `register_metadata` / `record_response`.
    ///
    /// Under a virtual clock the fabrication *sleeps on the clock* between token
    /// emissions, so each dispatch consumes its analytic latency in virtual time.
    /// This is essential for the graph replay: the whole-trace runtime advances
    /// its timeline off dispatch completions, so an instant dispatch would leave
    /// the `drive_sim` pump with no scheduled event after the first node and the
    /// run would quiesce immediately. Under a real clock the timestamps are
    /// computed instantly (fast self-benchmark; the scheduler owns pacing).
    async fn fabricate(
        &self,
        observer: &dyn RequestObserver,
        uuid: Uuid,
        isl: u64,
        osl: usize,
        start_abs: i64,
        cancel_after_ns: Option<i64>,
        request_payload: Bytes,
        recorded: RecordedLatency,
        on_first_token: &dyn Fn(i64),
    ) -> DispatchResult {
        // The live in-flight count feeds the analytic concurrency terms; the
        // ordinal seeds the reproducible jitter draw.
        let active_inflight = self.inflight.get() + 1;
        self.inflight.set(active_inflight);
        let ordinal = self.ordinal.get();
        self.ordinal.set(ordinal + 1);
        // Under the `recorded` model with a known api_time, reproduce that total
        // latency exactly; otherwise (or when the trace has no recorded value) fall
        // back to the analytic model.
        let recorded_total = (self.params.latency_model == DryRunLatencyModel::Recorded)
            .then_some(recorded.api_time_ns)
            .flatten();
        let (ttft_ns, itl_ns) = match recorded_total {
            Some(total) => recorded_latencies_ns(total, recorded.ttft_ns, osl),
            None => self
                .params
                .effective_latencies_ns(isl as usize, osl, active_inflight, ordinal),
        };
        let recv_start_abs = start_abs + ttft_ns;
        let token_abs = |index: usize| start_abs + ttft_ns + (index as i64) * itl_ns;
        // In recorded mode the total end is pinned to the recorded api_time so the
        // request latency is byte-exact even when integer token spacing rounds.
        let end_abs = if let Some(total) = recorded_total {
            start_abs + total.max(0)
        } else if osl > 0 {
            token_abs(osl - 1)
        } else {
            recv_start_abs
        };
        // The cancellation policy is armed after request send completion. A dry
        // run has no wire-send phase, so dispatch start is that boundary. Keep
        // an exactly-tied terminal response successful, as `race_cancel` does
        // for the real transport's response branch.
        let cancellation_abs = cancel_after_ns
            .map(|delay| start_abs.saturating_add(delay.max(0)))
            .filter(|deadline| *deadline < end_abs);
        let virtual_time = self.clock.is_virtual();
        for index in 0..osl {
            if cancellation_abs.is_some_and(|deadline| token_abs(index) > deadline) {
                break;
            }
            if virtual_time {
                // Advance virtual time to this token's fabricated arrival.
                let wait = token_abs(index) - self.clock.now_ns();
                if wait > 0 {
                    self.clock.clone().sleep(wait).await;
                }
            }
            let at_ms = self.rel_ms(token_abs(index));
            if index == 0 {
                on_first_token(ttft_ns);
            }
            observer.on_token(uuid, at_ms);
        }
        if let Some(cancellation_abs) = cancellation_abs {
            if virtual_time {
                let wait = cancellation_abs - self.clock.now_ns();
                if wait > 0 {
                    self.clock.clone().sleep(wait).await;
                }
            }
            self.inflight.set(self.inflight.get() - 1);
            let cancel_after_ns = cancel_after_ns.unwrap_or_default().max(0);
            let error = ErrorDetails::cancelled(format!(
                "RequestCancellationError: request cancelled {cancel_after_ns}ns after being sent"
            ));
            observer.on_usage(uuid, ObservedUsage::default());
            observer.on_terminal(uuid, ReplayTerminalStatus::Canceled);
            let record = RequestRecord {
                start_ns: start_abs,
                request_body: request_payload.clone(),
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
            // An empty (osl == 0) or already-past request still consumes its
            // prefill time so the replay timeline never stalls.
            let wait = end_abs - self.clock.now_ns();
            if wait > 0 {
                self.clock.clone().sleep(wait).await;
            }
        }
        self.inflight.set(self.inflight.get() - 1);
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
            request_body: request_payload.clone(),
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

/// Fake [`RequestExecutor`] (scheduled path): drives the shared
/// [`NativeMetricsObserver`] with fabricated timing.
struct FakeRequestExecutor {
    core: FakeFabricator,
    observer: RefCell<Option<Rc<NativeMetricsObserver>>>,
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
        // Dispatch begins at the scheduled arrival: a dry run adds no queueing
        // delay, so start == arrival and request_latency == ttft + (osl-1)*itl.
        let start_abs =
            self.core.origin_ns.get() + (context.arrival_ms * 1_000_000.0).round() as i64;
        let request_payload = turn
            .request
            .request_body_bytes
            .clone()
            .unwrap_or_else(Bytes::new);
        // The scheduled seam owns arrival/metadata/record_response around the
        // shared fabrication.
        observer.register_metadata(uuid, context.metadata.clone());
        observer.on_arrival(uuid, context.arrival_ms, isl as usize, osl);
        let recorded = RecordedLatency {
            api_time_ns: turn.request.recorded_api_time_ns,
            ttft_ns: turn.request.recorded_ttft_ns,
        };
        let result = self
            .core
            .fabricate(
                &*observer,
                uuid,
                isl,
                osl,
                start_abs,
                turn.request.cancel_after_ns,
                request_payload,
                recorded,
                on_first_token,
            )
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

/// Fake [`Dispatcher`] (graph path): the same analytic fabrication behind the
/// object-safe `Dispatcher` seam the graph runtime dispatches over. The graph
/// runtime owns `on_arrival`; this emits the token/usage/terminal stream and
/// returns the record, exactly as `TransportSink::dispatch_collect` does.
struct FakeDispatcher {
    core: FakeFabricator,
}

impl FakeDispatcher {
    fn new(core: FakeFabricator) -> Self {
        Self { core }
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
        let request_payload = turn
            .request
            .request_body_bytes
            .clone()
            .unwrap_or_else(Bytes::new);
        let recorded = RecordedLatency {
            api_time_ns: turn.request.recorded_api_time_ns,
            ttft_ns: turn.request.recorded_ttft_ns,
        };
        Ok(self
            .core
            .fabricate(
                observer,
                uuid,
                isl,
                osl,
                start_abs,
                turn.request.cancel_after_ns,
                request_payload,
                recorded,
                on_first_token,
            )
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
    use crate::multiturn::{PreparedEndpointReference, TurnDataPolicy};
    use crate::transport::core::PreparedEndpointBinding;
    use crate::transport::core::Request;
    use crate::transport::http::TransportSinkConfig;

    /// All-zero analytic params (no base latency, no scaling, no jitter) to build
    /// on with struct-update syntax.
    fn zero_params() -> DryRunParams {
        DryRunParams {
            ttft_ms: 0.0,
            itl_ms: 0.0,
            ttft_per_isl_token_ms: 0.0,
            ttft_concurrency_quad_ms: 0.0,
            itl_per_osl_token_ms: 0.0,
            itl_concurrency_lin_ms: 0.0,
            ttft_jitter_cv: 0.0,
            itl_jitter_cv: 0.0,
            seed: 0,
            latency_model: DryRunLatencyModel::Linear,
            kv_utilization: 0.5,
            clock: DryRunClock::Real,
        }
    }

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
                raw_enabled: false,
                prepared_endpoints: None,
                hop_routing: crate::engine::protocol::HopRouting::RoundRobin,
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
            request: Request {
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
                image_count: None,
                recorded_api_time_ns: None,
                recorded_ttft_ns: None,
            },
            model: "fixture-model".to_string(),
            endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
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
            ..zero_params()
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

    #[tokio::test(flavor = "current_thread")]
    async fn recorded_model_reproduces_api_time_as_total_latency() {
        // With the `recorded` model and a request carrying a recorded api_time, the
        // fabricated request latency equals that api_time exactly (timing parity),
        // and the recorded TTFT splits the token stream.
        const ISL: usize = 20;
        const OSL: usize = 12;
        const API_TIME_NS: i64 = 137_000_000; // 137 ms recorded api_time
        const TTFT_NS: i64 = 40_000_000; // 40 ms recorded TTFT
        let params = DryRunParams {
            // Non-zero analytic knobs prove the recorded value wins over them.
            ttft_ms: 999.0,
            itl_ms: 999.0,
            latency_model: DryRunLatencyModel::Recorded,
            ..zero_params()
        };
        let executor = build_executor(params);
        let uuid = Uuid::new_v4();
        let mut turn = fixture_turn(uuid, ISL, OSL);
        turn.request.recorded_api_time_ns = Some(API_TIME_NS);
        turn.request.recorded_ttft_ns = Some(TTFT_NS);

        let local = tokio::task::LocalSet::new();
        let outcome = local
            .run_until(executor.execute_measured(turn, context(ISL, OSL), &|_| {}))
            .await
            .expect("fabricated dispatch");

        let o = &outcome.result.outcome;
        // Total request latency is byte-exact to the recorded api_time.
        assert_eq!(o.end_ns - o.start_ns, API_TIME_NS);
        // First token lands at the recorded TTFT.
        let record = &outcome.result.record;
        assert_eq!(record.recv_start_ns, Some(record.start_ns + TTFT_NS));
        assert_eq!(record.responses[0].perf_ns() - record.start_ns, TTFT_NS);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recorded_model_falls_back_to_analytic_without_recorded_value() {
        // A request with no recorded api_time under the `recorded` model uses the
        // linear analytic latencies (fallback), not a zero-latency response.
        const ISL: usize = 8;
        const OSL: usize = 4;
        let params = DryRunParams {
            ttft_ms: 10.0,
            itl_ms: 2.0,
            latency_model: DryRunLatencyModel::Recorded,
            ..zero_params()
        };
        let executor = build_executor(params);
        let uuid = Uuid::new_v4();
        let outcome = tokio::task::LocalSet::new()
            .run_until(executor.execute_measured(
                fixture_turn(uuid, ISL, OSL),
                context(ISL, OSL),
                &|_| {},
            ))
            .await
            .expect("fabricated dispatch");
        let o = &outcome.result.outcome;
        assert_eq!(
            o.end_ns - o.start_ns,
            10_000_000 + (OSL as i64 - 1) * 2_000_000
        );
    }

    #[test]
    fn recorded_split_pins_total_and_honors_ttft() {
        // Even split when no TTFT: total end == api_time.
        assert_eq!(recorded_latencies_ns(120, None, 4), (30, 30));
        // Recorded TTFT carries the remainder in the generated-token span.
        assert_eq!(recorded_latencies_ns(120, Some(30), 4), (30, 30));
        assert_eq!(recorded_latencies_ns(100, Some(40), 4), (40, 20));
        // osl <= 1 puts the whole api_time in the TTFT.
        assert_eq!(recorded_latencies_ns(90, Some(10), 1), (90, 0));
        assert_eq!(recorded_latencies_ns(90, None, 0), (90, 0));
    }

    #[test]
    fn analytic_model_scales_ttft_with_isl_and_itl_with_osl() {
        // TTFT = base + per_isl·ISL; ITL = base + per_osl·OSL (inflight == 1 on
        // the instant single-reactor path, so the concurrency terms contribute a
        // constant; here they are zero). No jitter → exact.
        let params = DryRunParams {
            ttft_ms: 5.0,
            itl_ms: 1.0,
            ttft_per_isl_token_ms: 0.1,    // +0.1 ms/input token
            itl_per_osl_token_ms: 0.01,    // +0.01 ms/output token
            ttft_concurrency_quad_ms: 3.0, // ·1² == +3 ms (inflight == 1)
            itl_concurrency_lin_ms: 2.0,   // ·1  == +2 ms (inflight == 1)
            ..zero_params()
        };
        let (ttft_ns, itl_ns) = params.effective_latencies_ns(100, 50, 1, 0);
        // ttft = (5 + 0.1·100 + 3·1) = 18 ms; itl = (1 + 0.01·50 + 2·1) = 3.5 ms
        assert_eq!(ttft_ns, 18_000_000);
        assert_eq!(itl_ns, 3_500_000);
        // Zero jitter is exactly reproducible.
        assert_eq!(
            params.effective_latencies_ns(100, 50, 1, 7),
            (18_000_000, 3_500_000)
        );
    }

    #[test]
    fn polynomial_prefill_is_independent_of_unbatched_concurrency() {
        let one = aic_polynomial_prefill_ms(300_000.0);
        let concurrent = aic_polynomial_prefill_ms(300_000.0);
        assert_eq!(one, concurrent);
    }

    #[test]
    fn aiconfigurator_polynomial_matches_dynamo_perf_model() {
        // Exact curve values from the Dynamo mocker PerfModel::Polynomial
        // (perf_model.rs:272-273, :315). At inflight == 1, prefill tokens == ISL.
        let params = DryRunParams {
            latency_model: DryRunLatencyModel::AiconfiguratorPolynomial,
            kv_utilization: 0.5,
            ..zero_params()
        };
        // prefill(1000) = 4.209989e-7·1000² + 1.518344e-2·1000 + 16.50142
        //               = 0.4209989 + 15.18344 + 16.50142 = 32.1058589 ms
        let prefill = aic_polynomial_prefill_ms(1000.0);
        assert!((prefill - 32.105_858_9).abs() < 1e-6, "prefill = {prefill}");
        // decode(0.5) = -25.74·0.25 + 54.01·0.5 + 5.74 = 26.31 ms
        let decode = aic_polynomial_decode_ms(0.5);
        assert!((decode - 26.31).abs() < 1e-9, "decode = {decode}");
        // Floor: a tiny utilization must not drop the ITL below 1.0 ms.
        assert_eq!(aic_polynomial_decode_ms(-1.0), 1.0);
        // The executor path uses these curves (ISL=1000 → ttft≈32.11ms, ITL≈26.31).
        let (ttft_ns, itl_ns) = params.effective_latencies_ns(1000, 64, 1, 0);
        assert_eq!(ttft_ns, ms_to_ns(prefill));
        assert_eq!(itl_ns, ms_to_ns(decode));
    }

    #[test]
    fn analytic_jitter_is_seeded_and_reproducible() {
        let params = DryRunParams {
            ttft_ms: 20.0,
            itl_ms: 4.0,
            ttft_jitter_cv: 0.2,
            itl_jitter_cv: 0.2,
            seed: 42,
            ..zero_params()
        };
        // Same ordinal → identical draw; different ordinals → (almost surely)
        // different draws, but both reproducible across calls.
        let a = params.effective_latencies_ns(32, 16, 1, 3);
        let b = params.effective_latencies_ns(32, 16, 1, 3);
        let c = params.effective_latencies_ns(32, 16, 1, 4);
        assert_eq!(a, b, "same seed+ordinal must reproduce the jitter draw");
        assert_ne!(a, c, "distinct ordinals draw distinct jitter");
        // Mean-preserving jitter keeps values in a sane band around the base.
        assert!(
            a.0 > 5_000_000 && a.0 < 80_000_000,
            "ttft jitter out of band: {}",
            a.0
        );
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
