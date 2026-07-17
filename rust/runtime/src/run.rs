// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The online run loop: a Clock-driven arrival pacer, gated by `StopChecker`
//! (request-count / duration) and `SlotPool` (concurrency), dispatching a synthetic
//! workload through [`TransportSink`] and measuring it with the shared
//! `TraceCollector`.
//!
//! One loop serves both modes via the timing-plane seam:
//! - **request-rate** — Poisson/Gamma/Constant inter-arrivals
//!   ([`IntervalGenerator`](crate::timing::IntervalGenerator)),
//! - **concurrency** — the degenerate `ConcurrencyBurst` (zero interval) bounded by a
//!   session [`SlotPool`].
//! - **ancillary policy** — Clock-driven session/prefill/rate ramps, per-request
//!   post-send cancellation, and ordered endpoint selection from shared traits.
//!
//! Stopping is condition-driven ([`StopChecker`]): the
//! loop pulls requests on demand until the request-count and/or duration bound fires,
//! not until a fixed list is exhausted. Arrival timing uses only `clock.now_ns()` +
//! `clock.sleep()`, so the identical loop runs on `RealClock` (online) or `SimClock`
//! (offline) — backend-agnostic by construction.
//!
//! The transport is `!Send` (`Rc<dyn Clock>`), so the loop runs on a single
//! `LocalSet` with `spawn_local`; a shared clock is the one time authority.
//!
//! Backend-neutral helpers are enabled for simulation and direct runtime tests.
#![cfg_attr(not(test), allow(unused_imports))]
#![cfg_attr(not(any(test, feature = "dynosim")), allow(dead_code))]

use std::cell::RefCell;
use std::rc::Rc;

use crate::clock::{Clock, RealClock};
use crate::metrics_core::{AccumulatorSummary, MetricsConfig};
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::observer::CollectorObserver;
use loadgen_core::sink::RequestObserver;

use crate::timing::{
    ArrivalPattern, LinearRamp, Phase, RampDriver, RampHandle, RamperConfig, RunState, SlotPool,
    StopChecker, StopConfig, make_interval_generator,
};

use crate::adaptive::{AdaptiveControlVariable, AdaptiveRunConfig, build_adaptive};
use crate::ancillary::{AncillaryTimingConfig, parse_base_urls, url_selector};
use crate::metrics::{
    NativeMetricsObserver, NativeResponseMetadata, ObserverTee, RequestMetricMetadata,
};
use crate::multiturn::ConversationSource;
use crate::request_rate::{RequestRateConfig, RequestRateWorkload};
use crate::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, ScheduledRunReport, ScheduledRuntime, TurnDispatcher,
    Workload, run_scheduled_workload_with_ancillary,
};
use crate::scheduler::LocalTaskScheduler;
use crate::transport::http::{HttpRequestDispatcher, TransportSink};
use crate::user_centric::{UserCentricConfig, UserCentricWorkload};
use crate::workload::SkeletonWorkload;

/// Online result carrying the compatibility summary and native metric engine output.
pub struct OnlineRunReport {
    /// Existing flat compatibility report.
    pub performance: TraceSimulationReport,
    /// Native typed metric distributions, sweeps, and timeslices.
    pub metrics: AccumulatorSummary,
}

pub(crate) fn validate_ramp_actuators(
    ancillary: &AncillaryTimingConfig,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
) -> anyhow::Result<()> {
    if ancillary.concurrency_ramp_duration_ns.is_some() {
        anyhow::ensure!(
            concurrency.is_some_and(|value| value > 0),
            "a concurrency ramp requires a positive concurrency limit"
        );
    }
    if ancillary.prefill_concurrency_ramp_duration_ns.is_some() {
        anyhow::ensure!(
            prefill_concurrency.is_some_and(|value| value > 0),
            "a prefill concurrency ramp requires --prefill-concurrency"
        );
    }
    if let Some(ramp_duration_ns) = ancillary.request_rate_ramp_duration_ns {
        anyhow::ensure!(
            pattern != ArrivalPattern::ConcurrencyBurst
                && rate.is_some_and(|value| value.is_finite() && value > 0.0),
            "a request-rate ramp requires --request-rate"
        );
        let target = rate.expect("positive finite request rate checked above");
        let start =
            target * ancillary.rate_ramp_update_interval_ns as f64 / ramp_duration_ns as f64;
        anyhow::ensure!(
            start.is_finite() && start > 0.0,
            "request-rate ramp proportional start must be positive and finite"
        );
    }
    Ok(())
}

pub(crate) fn validate_adaptive_ramp_ownership(
    ancillary: &AncillaryTimingConfig,
    adaptive: Option<&AdaptiveRunConfig>,
) -> anyhow::Result<()> {
    let Some(adaptive) = adaptive else {
        return Ok(());
    };
    let conflict = match adaptive.control_variable {
        AdaptiveControlVariable::Concurrency => ancillary.concurrency_ramp_duration_ns.is_some(),
        AdaptiveControlVariable::PrefillConcurrency => {
            ancillary.prefill_concurrency_ramp_duration_ns.is_some()
        }
        AdaptiveControlVariable::RequestRate => ancillary.request_rate_ramp_duration_ns.is_some(),
        AdaptiveControlVariable::Users => false,
    };
    anyhow::ensure!(
        !conflict,
        "adaptive control and a phase ramp cannot own the same actuator"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_ramps(
    ancillary: &AncillaryTimingConfig,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    rate: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
) -> anyhow::Result<Vec<RampHandle>> {
    let mut handles = Vec::new();
    if let Some(duration_ns) = ancillary.concurrency_ramp_duration_ns {
        let target = concurrency.expect("validated concurrency ramp target");
        let slots = session_slots.expect("validated session slot actuator");
        let config = RamperConfig::new(1.0, target as f64, duration_ns)?;
        handles.push(
            RampDriver::new(
                clock.clone(),
                Box::new(LinearRamp::new(config)),
                move |value| slots.set_limit(value as usize),
            )
            .spawn_local(),
        );
    }
    if let Some(duration_ns) = ancillary.prefill_concurrency_ramp_duration_ns {
        let target = prefill_concurrency.expect("validated prefill ramp target");
        let slots = prefill_slots.expect("validated prefill slot actuator");
        let config = RamperConfig::new(1.0, target as f64, duration_ns)?;
        handles.push(
            RampDriver::new(
                clock.clone(),
                Box::new(LinearRamp::new(config)),
                move |value| slots.set_limit(value as usize),
            )
            .spawn_local(),
        );
    }
    if let Some(duration_ns) = ancillary.request_rate_ramp_duration_ns {
        let target = rate.expect("validated request-rate ramp target");
        let update_interval_ns = ancillary.rate_ramp_update_interval_ns;
        // Start at one proportional update increment so the target is reached
        // exactly after the configured number of updates.
        let start = target * update_interval_ns as f64 / duration_ns as f64;
        let config = RamperConfig::new(start, target, duration_ns)?
            .with_update_interval_ns(update_interval_ns)?;
        handles.push(
            RampDriver::new(clock, Box::new(LinearRamp::new(config)), move |value| {
                intervals.borrow_mut().set_rate(value)
            })
            .spawn_local(),
        );
    }
    Ok(handles)
}

pub(crate) async fn stop_ramps(handles: Vec<RampHandle>) -> anyhow::Result<()> {
    for handle in handles {
        if handle.is_running() {
            handle.stop();
        }
        if let Err(error) = handle.wait().await
            && !error.is_cancelled()
        {
            anyhow::bail!("ramp task failed: {error}");
        }
    }
    Ok(())
}

fn scheduled_policies(
    ancillary: &AncillaryTimingConfig,
    base_urls: &[String],
    seed: u64,
) -> anyhow::Result<ScheduledAncillaryPolicies> {
    Ok(ScheduledAncillaryPolicies {
        cancellation_policy: ancillary.cancellation_policy(seed)?,
        url_selector: url_selector(base_urls)?,
        phase: Phase::Profiling,
    })
}

pub(crate) fn validate_user_centric_ramps(
    ancillary: &AncillaryTimingConfig,
    config: UserCentricConfig,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        ancillary.request_rate_ramp_duration_ns.is_none(),
        "user-centric cadence is schedule-authored and does not accept a request-rate ramp"
    );
    anyhow::ensure!(
        ancillary.prefill_concurrency_ramp_duration_ns.is_none(),
        "user-centric runtime has no prefill SlotPool actuator"
    );
    if ancillary.concurrency_ramp_duration_ns.is_some() {
        anyhow::ensure!(
            config.concurrency.is_some_and(|limit| limit > 0),
            "a user-centric concurrency ramp requires --concurrency"
        );
    }
    Ok(())
}

/// Backend-neutral paced issuer shared by real HTTP and the optional in-process
/// Dynamo engine. The caller supplies one clock, one dispatcher, and diagnostic
/// endpoint names; all scheduling, admission, adaptive control, observations,
/// and report construction below are identical across backends.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_paced_with_backend(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    sink: Rc<dyn HttpRequestDispatcher>,
    base_urls: Vec<String>,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
) -> anyhow::Result<OnlineRunReport> {
    ancillary.validate()?;
    validate_ramp_actuators(&ancillary, pattern, rate, concurrency, prefill_concurrency)?;
    validate_adaptive_ramp_ownership(&ancillary, adaptive.as_ref())?;
    let ms = |ns: i64| (ns - start_ns) as f64 / 1_000_000.0;

    let collector = Rc::new(CollectorObserver::new(false));
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        start_ns,
        MetricsConfig::default(),
    ));
    let delegates: Vec<Rc<dyn RequestObserver>> = vec![collector.clone(), native_metrics.clone()];
    let base_observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
    let adaptive_integer_minimum = match adaptive.as_ref() {
        Some(config)
            if matches!(
                config.control_variable,
                AdaptiveControlVariable::Concurrency
                    | AdaptiveControlVariable::PrefillConcurrency
                    | AdaptiveControlVariable::Users
            ) =>
        {
            anyhow::ensure!(
                config.minimum.is_finite()
                    && config.minimum >= 1.0
                    && config.minimum.fract() == 0.0
                    && config.minimum < usize::MAX as f64,
                "adaptive integer control minimum must be an integer >= 1"
            );
            Some(config.minimum as usize)
        }
        _ => None,
    };
    let session_initial = match adaptive.as_ref().map(|config| config.control_variable) {
        Some(AdaptiveControlVariable::Concurrency) => adaptive_integer_minimum,
        _ => concurrency,
    };
    let prefill_initial = match adaptive.as_ref().map(|config| config.control_variable) {
        Some(AdaptiveControlVariable::PrefillConcurrency) => adaptive_integer_minimum,
        _ => prefill_concurrency,
    };
    let session_slots = session_initial.map(|limit| Rc::new(SlotPool::new(limit)));
    let prefill_slots = prefill_initial.map(|limit| Rc::new(SlotPool::new(limit)));
    let intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>> = Rc::new(RefCell::new(
        make_interval_generator(pattern, rate, smoothness, seed),
    ));
    let ramp_handles = start_ramps(
        &ancillary,
        clock.clone(),
        intervals.clone(),
        session_slots.clone(),
        prefill_slots.clone(),
        rate,
        concurrency,
        prefill_concurrency,
    )?;
    let mut cancellation_policy = ancillary.cancellation_policy(seed)?;
    let mut endpoint_selector = url_selector(&base_urls)?;

    let (adaptive_scale, obs): (
        Option<Rc<crate::adaptive_core::AdaptiveScale>>,
        Rc<dyn RequestObserver>,
    ) = match adaptive {
        Some(config) => {
            let built = build_adaptive(
                config,
                clock.clone(),
                start_ns,
                base_observer.clone(),
                intervals.clone(),
                session_slots.clone(),
                prefill_slots.clone(),
                None,
            )?;
            built.scale.start()?;
            (Some(built.scale), built.observer)
        }
        None => (None, base_observer.clone()),
    };
    let assessment_task = adaptive_scale.as_ref().map(|scale| {
        let scale = scale.clone();
        tokio::task::spawn_local(scale.assessment_loop())
    });

    let checker = StopChecker::new(&stop);
    let mut state = RunState {
        started_at_ns: start_ns,
        ..Default::default()
    };

    // Absolute schedule: the next arrival's target time on the clock's timeline.
    // This is the (AfterInterval, Reanchor) policy named in `crate::timing::arrival`.
    // It keeps its own arithmetic rather than calling `next_arrival_target` because
    // the next interval is drawn at the tail of each iteration (line below), and this
    // loop shares its generator with live ramp actuators — moving that draw into the
    // shared helper's start-of-iteration position could shift it across a concurrent
    // rate change. The graph loop, which has no such ramp coupling, uses the helper.
    let mut next_target_ns = start_ns + intervals.borrow_mut().next_interval_ns();

    let mut handles = Vec::new();
    loop {
        if adaptive_scale
            .as_ref()
            .is_some_and(|scale| scale.should_stop_sending())
        {
            state.sending_complete = true;
        }
        if !checker.can_send_any(&state, clock.now_ns()) {
            break;
        }
        // Pace to the next arrival target. Falling behind re-anchors to `now` rather
        // than firing a catch-up salvo.
        let now = clock.now_ns();
        if next_target_ns < now {
            next_target_ns = now;
        }
        let wait_ns = next_target_ns - now;
        if wait_ns > 0 {
            if let Some(scale) = &adaptive_scale {
                tokio::select! {
                    _ = clock.clone().sleep(wait_ns) => {}
                    _ = scale.wait_until_stop_sending() => {
                        break;
                    }
                }
            } else {
                clock.clone().sleep(wait_ns).await;
            }
        }
        // Draw the next interval BEFORE dispatch so issue latency doesn't skew it.
        next_target_ns += intervals.borrow_mut().next_interval_ns();

        // Duration may have elapsed during the sleep — re-check before dispatching.
        if !checker.can_send_any(&state, clock.now_ns()) {
            break;
        }

        // Session slot (if capped): acquire before dispatch; the guard releases the
        // slot when the dispatch task completes. Open-loop rate passes `None`.
        let session_guard = match &session_slots {
            Some(pool) => Some(pool.acquire().await),
            None => None,
        };
        let prefill_guard = match &prefill_slots {
            Some(pool) => Some(pool.acquire().await),
            None => None,
        };

        if adaptive_scale
            .as_ref()
            .is_some_and(|scale| scale.should_stop_sending())
            || !checker.can_send_any(&state, clock.now_ns())
        {
            drop(prefill_guard);
            drop(session_guard);
            break;
        }

        let mut req = workload.make_request();
        req.cancel_after_ns = cancellation_policy
            .as_mut()
            .and_then(|policy| policy.next_cancel_delay_ns(Phase::Profiling));
        req.url_index = endpoint_selector.as_mut().map(|selector| {
            u32::try_from(selector.next_index())
                .expect("validated endpoint selector index must fit u32")
        });
        let dimensions = sink.inference_dimensions(&req);
        native_metrics.register_metadata(
            req.uuid,
            RequestMetricMetadata {
                session_num: Some(state.sent_sessions),
                correlation_id: Some(req.uuid.to_string()),
                dimensions,
                ..RequestMetricMetadata::default()
            },
        );
        obs.on_arrival(
            req.uuid,
            ms(clock.now_ns()),
            req.input_length,
            req.max_output_tokens,
        );
        // Single-turn synthetic: each request is its own session (turn 0 = final).
        state.requests_sent += 1;
        state.root_requests_sent += 1;
        state.sent_sessions += 1;

        let obs2 = obs.clone();
        let sink2 = sink.clone();
        let native_metrics2 = native_metrics.clone();
        handles.push(tokio::task::spawn_local(async move {
            let _session_guard = session_guard;
            let prefill_guard = Rc::new(RefCell::new(prefill_guard));
            let prefill_for_first_token = prefill_guard.clone();
            let uuid = req.uuid;
            let release_prefill = move |_ttft_ns| {
                prefill_for_first_token.borrow_mut().take();
            };
            match sink2
                .dispatch_collect(req, obs2.as_ref(), &release_prefill)
                .await
            {
                Ok(response) => native_metrics2.record_response(
                    uuid,
                    NativeResponseMetadata {
                        start_ns: Some(response.start_ns),
                        end_ns: Some(response.end_ns),
                        prompt_tokens: response.prompt_tokens.map(u64::from),
                        completion_tokens: response.completion_tokens.map(u64::from),
                        http: response.http,
                    },
                ),
                Err(e) => {
                    obs2.on_terminal(uuid, ReplayTerminalStatus::Failed);
                    tracing::warn!(%uuid, error = %e, "request dispatch failed");
                }
            }
            // A request that failed or completed without a first token releases
            // prefill at terminal instead of leaking the slot.
            prefill_guard.borrow_mut().take();
        }));

        // Concurrency is bounded by the slot pools, but the handle set is not:
        // open-loop rate has no session cap, so without pruning the Vec grows
        // O(total requests). Reap completed tasks so it stays O(in-flight).
        const HANDLE_REAP_THRESHOLD: usize = 1024;
        if handles.len() >= HANDLE_REAP_THRESHOLD {
            handles.retain(|handle| !handle.is_finished());
        }
    }
    for h in handles {
        if let Err(e) = h.await {
            tracing::warn!(error = %e, "request task join failed");
        }
    }

    stop_ramps(ramp_handles).await?;

    if let Some(scale) = &adaptive_scale {
        scale.deactivate();
        if let Some(task) = assessment_task {
            task.abort();
            let _ = task.await;
        }
        scale.complete_phase()?;
        if let Some(error) = scale.last_error() {
            anyhow::bail!("adaptive assessment failed: {error}");
        }
    }

    let wall_ms = ms(clock.now_ns());
    Ok(OnlineRunReport {
        performance: collector.finish(wall_ms),
        metrics: native_metrics.finish(),
    })
}

/// Backend-neutral request-rate runtime used by both real HTTP and the
/// in-process Dynamo engine. All scheduling, ramping, cancellation, adaptive
/// control, observer, and metric code executes above the injected dispatcher.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "dynosim"), allow(dead_code))]
pub(crate) async fn run_request_rate_with_backend(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    endpoint_names: Vec<String>,
    conversations: Box<dyn ConversationSource>,
    mut config: RequestRateConfig,
    stop: StopConfig,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
) -> anyhow::Result<ScheduledRunReport> {
    ancillary.validate()?;
    validate_ramp_actuators(
        &ancillary,
        config.arrival_pattern,
        config.request_rate,
        config.session_concurrency,
        config.prefill_concurrency,
    )?;
    validate_adaptive_ramp_ownership(&ancillary, adaptive.as_ref())?;
    anyhow::ensure!(
        stop.total_expected_requests.is_some()
            || stop.expected_num_sessions.is_some()
            || stop.expected_duration_ns.is_some(),
        "request-rate workload requires a request, session, or duration stop bound"
    );

    // An adaptive integer actuator owns a live pool even when the corresponding
    // steady-state CLI cap was omitted. Its controller constructor immediately
    // applies the minimum before workload execution begins.
    if let Some(adaptive) = &adaptive {
        match adaptive.control_variable {
            AdaptiveControlVariable::Concurrency if config.session_concurrency.is_none() => {
                config.session_concurrency = Some(adaptive.minimum as usize);
            }
            AdaptiveControlVariable::PrefillConcurrency if config.prefill_concurrency.is_none() => {
                config.prefill_concurrency = Some(adaptive.minimum as usize);
            }
            _ => {}
        }
    }

    let workload = Rc::new(RequestRateWorkload::new(config, conversations)?);
    let intervals = workload.intervals();
    let session_slots = workload.session_slots();
    let prefill_slots = workload.prefill_slots();
    let ramp_handles = start_ramps(
        &ancillary,
        clock.clone(),
        intervals.clone(),
        session_slots.clone(),
        prefill_slots.clone(),
        config.request_rate,
        config.session_concurrency,
        config.prefill_concurrency,
    )?;
    let policies = scheduled_policies(&ancillary, &endpoint_names, config.seed)?;

    let Some(adaptive) = adaptive else {
        let workload: Rc<dyn Workload> = workload;
        let result = run_scheduled_workload_with_ancillary(
            workload, clock, start_ns, dispatcher, stop, true, policies,
        )
        .await;
        stop_ramps(ramp_handles).await?;
        return result;
    };

    let collector = Rc::new(CollectorObserver::new(true));
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        start_ns,
        MetricsConfig::default(),
    ));
    let delegates: Vec<Rc<dyn RequestObserver>> = vec![collector.clone(), native_metrics.clone()];
    let base_observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
    let built = build_adaptive(
        adaptive,
        clock.clone(),
        start_ns,
        base_observer,
        intervals,
        session_slots,
        prefill_slots,
        None,
    )?;
    built.scale.start()?;
    let gate: Rc<dyn IssuanceGate> = built.scale.clone();
    let runtime = ScheduledRuntime::new_with_observer(
        clock,
        start_ns,
        dispatcher,
        stop,
        true,
        collector,
        native_metrics,
        built.observer,
        Some(gate),
    );
    runtime.configure_ancillary(
        policies.cancellation_policy,
        policies.url_selector,
        policies.phase,
    );

    let assessment_scale = built.scale.clone();
    let assessment = assessment_scale.assessment_loop();
    let execution = workload.execute(runtime.clone());
    tokio::pin!(assessment);
    tokio::pin!(execution);
    let execution_result = tokio::select! {
        result = &mut execution => {
            built.scale.deactivate();
            built.scale.complete_phase()?;
            result
        }
        _ = &mut assessment => Ok(()),
    };
    runtime.scheduler().cancel_pending();
    runtime.scheduler().wait_idle().await;
    stop_ramps(ramp_handles).await?;
    execution_result?;
    built.scale.complete_phase()?;
    if let Some(error) = built.scale.last_error() {
        anyhow::bail!("adaptive assessment failed: {error}");
    }
    Ok(runtime.finish(workload.name(), None))
}

/// Backend-neutral user-centric runtime with Clock-native ramps,
/// cancellation, and endpoint selection.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "dynosim"), allow(dead_code))]
pub(crate) async fn run_user_centric_with_backend(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    endpoint_names: Vec<String>,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> anyhow::Result<ScheduledRunReport> {
    ancillary.validate()?;
    validate_user_centric_ramps(&ancillary, config)?;
    let workload = Rc::new(UserCentricWorkload::new(config, conversations)?);
    let intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>> = Rc::new(RefCell::new(
        make_interval_generator(ArrivalPattern::ConcurrencyBurst, None, None, seed),
    ));
    let ramp_handles = start_ramps(
        &ancillary,
        clock.clone(),
        intervals,
        workload.session_slots(),
        None,
        None,
        config.concurrency,
        None,
    )?;
    let result = run_scheduled_workload_with_ancillary(
        workload,
        clock,
        start_ns,
        dispatcher,
        stop,
        true,
        scheduled_policies(&ancillary, &endpoint_names, seed)?,
    )
    .await;
    stop_ramps(ramp_handles).await?;
    result
}

impl IssuanceGate for crate::adaptive_core::AdaptiveScale {
    fn can_issue(&self) -> bool {
        !self.should_stop_sending()
    }
}

/// Backend-neutral adaptive user-centric runtime. The injected clock and
/// dispatcher are the only online/offline differences.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_user_centric_adaptive_with_backend(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    endpoint_names: Vec<String>,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    adaptive: AdaptiveRunConfig,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> anyhow::Result<ScheduledRunReport> {
    ancillary.validate()?;
    validate_user_centric_ramps(&ancillary, config)?;
    validate_adaptive_ramp_ownership(&ancillary, Some(&adaptive))?;
    anyhow::ensure!(
        adaptive.control_variable == AdaptiveControlVariable::Users,
        "user-centric adaptive scale requires control_variable=users"
    );
    let workload = Rc::new(UserCentricWorkload::new(config, conversations)?);
    let user_target: Rc<dyn crate::adaptive_core::UserTarget> = Rc::new(workload.control());
    let collector = Rc::new(CollectorObserver::new(true));
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        start_ns,
        MetricsConfig::default(),
    ));
    let delegates: Vec<Rc<dyn RequestObserver>> = vec![collector.clone(), native_metrics.clone()];
    let base_observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
    let intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>> = Rc::new(RefCell::new(
        make_interval_generator(ArrivalPattern::ConcurrencyBurst, None, None, 0),
    ));
    let ramp_handles = start_ramps(
        &ancillary,
        clock.clone(),
        intervals.clone(),
        workload.session_slots(),
        None,
        None,
        config.concurrency,
        None,
    )?;
    let built = build_adaptive(
        adaptive,
        clock.clone(),
        start_ns,
        base_observer,
        intervals,
        None,
        None,
        Some(user_target),
    )?;
    built.scale.start()?;
    let gate: Rc<dyn IssuanceGate> = built.scale.clone();
    let runtime = ScheduledRuntime::new_with_observer(
        clock,
        start_ns,
        dispatcher,
        stop,
        true,
        collector,
        native_metrics,
        built.observer,
        Some(gate),
    );
    let policies = scheduled_policies(&ancillary, &endpoint_names, seed)?;
    runtime.configure_ancillary(
        policies.cancellation_policy,
        policies.url_selector,
        policies.phase,
    );

    let assessment_scale = built.scale.clone();
    let assessment = assessment_scale.assessment_loop();
    let execution = workload.execute(runtime.clone());
    tokio::pin!(assessment);
    tokio::pin!(execution);
    let execution_result = tokio::select! {
        result = &mut execution => {
            built.scale.deactivate();
            built.scale.complete_phase()?;
            result
        }
        _ = &mut assessment => Ok(()),
    };
    runtime.scheduler().cancel_pending();
    runtime.scheduler().wait_idle().await;
    stop_ramps(ramp_handles).await?;
    execution_result?;
    built.scale.complete_phase()?;
    if let Some(error) = built.scale.last_error() {
        anyhow::bail!("adaptive assessment failed: {error}");
    }
    Ok(runtime.finish(workload.name(), workload.user_control_snapshot()))
}

// Test-only HTTP wrappers over the backend-neutral runtimes.
#[cfg(test)]
async fn run(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    concurrency: usize,
) -> anyhow::Result<TraceSimulationReport> {
    let stop = StopConfig {
        total_expected_requests: Some(workload.num_requests as u64),
        ..Default::default()
    };
    run_paced(
        base_url,
        model,
        workload,
        ArrivalPattern::ConcurrencyBurst,
        None,
        None,
        Some(concurrency),
        stop,
        0,
    )
    .await
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_paced(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
) -> anyhow::Result<TraceSimulationReport> {
    run_paced_adaptive(
        base_url,
        model,
        workload,
        pattern,
        rate,
        smoothness,
        concurrency,
        None,
        stop,
        seed,
        None,
    )
    .await
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_paced_adaptive(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
    adaptive: Option<AdaptiveRunConfig>,
) -> anyhow::Result<TraceSimulationReport> {
    Ok(run_paced_adaptive_with_metrics(
        base_url,
        model,
        workload,
        pattern,
        rate,
        smoothness,
        concurrency,
        prefill_concurrency,
        stop,
        seed,
        adaptive,
    )
    .await?
    .performance)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_paced_adaptive_with_metrics(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
    adaptive: Option<AdaptiveRunConfig>,
) -> anyhow::Result<OnlineRunReport> {
    let base_urls = parse_base_urls(&base_url)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let sink: Rc<dyn HttpRequestDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, false)?
            .with_wire_response_capture(false),
    );
    run_paced_with_backend(
        clock,
        start_ns,
        sink,
        base_urls,
        workload,
        pattern,
        rate,
        smoothness,
        concurrency,
        prefill_concurrency,
        stop,
        seed,
        adaptive,
        AncillaryTimingConfig::default(),
    )
    .await
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_user_centric_adaptive_online(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    http2: bool,
    adaptive: AdaptiveRunConfig,
) -> anyhow::Result<ScheduledRunReport> {
    let base_urls = parse_base_urls(&base_url)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, http2)?
            .with_wire_response_capture(false)
            .with_prepared_endpoints(crate::test_util::chat_dispatch_table()),
    );
    run_user_centric_adaptive_with_backend(
        clock,
        start_ns,
        dispatcher,
        base_urls,
        conversations,
        config,
        stop,
        adaptive,
        AncillaryTimingConfig::default(),
        0,
    )
    .await
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_single_turn_dataset_online(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    concurrency: usize,
    http2: bool,
    record_processors: Vec<Rc<dyn crate::scheduled::TurnRecordProcessor>>,
    prepared_endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
) -> anyhow::Result<ScheduledRunReport> {
    use crate::scheduled::{SingleTurnDatasetWorkload, run_scheduled_workload_with_processors};
    let base_urls = parse_base_urls(&base_url)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, http2)?
            .with_wire_response_capture(false)
            .with_prepared_endpoints(prepared_endpoints),
    );
    let workload: Rc<dyn Workload> =
        Rc::new(SingleTurnDatasetWorkload::new(conversations, concurrency)?);
    run_scheduled_workload_with_processors(
        workload,
        clock,
        start_ns,
        dispatcher,
        StopConfig::default(),
        false,
        ScheduledAncillaryPolicies::default(),
        record_processors,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::{AdaptiveRunConfig, AdaptiveStepConfig};
    use crate::adaptive_core::{CorrelationContext, SlaFilter, SlaOp, SlaStat};
    use crate::clock::SimClock;
    use crate::graph::runtime::drive_sim;
    use crate::test_util::synthetic_prepared_source;
    use crate::workload::SkeletonWorkload;

    #[test]
    fn phase_ramps_drive_the_live_slot_and_interval_actuators() {
        let clock = Rc::new(SimClock::new());
        let intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>> =
            Rc::new(RefCell::new(make_interval_generator(
                ArrivalPattern::Constant,
                Some(40.0),
                None,
                0,
            )));
        let session_slots = Rc::new(SlotPool::new(5));
        let prefill_slots = Rc::new(SlotPool::new(3));
        let ancillary = AncillaryTimingConfig {
            concurrency_ramp_duration_ns: Some(400),
            prefill_concurrency_ramp_duration_ns: Some(400),
            request_rate_ramp_duration_ns: Some(400),
            rate_ramp_update_interval_ns: 100,
            ..AncillaryTimingConfig::default()
        };
        let intervals_for_run = intervals.clone();
        let session_for_run = session_slots.clone();
        let prefill_for_run = prefill_slots.clone();
        let clock_for_run: Rc<dyn Clock> = clock.clone();
        let outcome = drive_sim(clock, move |_handle| async move {
            let handles = start_ramps(
                &ancillary,
                clock_for_run,
                intervals_for_run.clone(),
                Some(session_for_run.clone()),
                Some(prefill_for_run.clone()),
                Some(40.0),
                Some(5),
                Some(3),
            )
            .unwrap();

            // Initial application is synchronous, so issuance cannot observe
            // the steady-state targets before the driver is first polled.
            assert_eq!(session_for_run.current_limit(), 1);
            assert_eq!(prefill_for_run.current_limit(), 1);
            assert_eq!(intervals_for_run.borrow().rate(), 10.0);
            for handle in handles {
                handle.wait().await.unwrap();
            }
            assert_eq!(session_for_run.current_limit(), 5);
            assert_eq!(prefill_for_run.current_limit(), 3);
            assert_eq!(intervals_for_run.borrow().rate(), 40.0);
        });
        assert!(
            !outcome.deadlocked,
            "all phase ramps must reach their targets"
        );
        assert_eq!(session_slots.current_limit(), 5);
        assert_eq!(prefill_slots.current_limit(), 3);
        assert_eq!(intervals.borrow().rate(), 40.0);
    }

    #[tokio::test]
    async fn e2e_reports_finite_metrics() {
        // The transport sink is `!Send`, so drive the run on a LocalSet.
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let wl = SkeletonWorkload {
                    num_requests: 4,
                    input_tokens: 8,
                    output_tokens: 2,
                    turns: 1,
                    think_time_ms: None,
                };
                let report = run(base, "m".into(), wl, 2).await.unwrap();
                assert_eq!(report.request_counts.num_requests, 4);
                // 4 requests * 2 content chunks each.
                assert_eq!(report.request_counts.total_output_tokens, 8);
                assert!(report.latency.ttft.mean_ms.is_finite());
                assert!(report.throughput.output_throughput_tok_s.is_finite());
            })
            .await;
    }

    #[tokio::test]
    async fn e2e_native_metrics_include_distributions_sweeps_and_v2_report() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let workload = SkeletonWorkload {
                    num_requests: 4,
                    input_tokens: 8,
                    output_tokens: 2,
                    turns: 1,
                    think_time_ms: None,
                };
                let report = run_paced_adaptive_with_metrics(
                    base,
                    "m".into(),
                    workload,
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(2),
                    None,
                    StopConfig {
                        total_expected_requests: Some(4),
                        ..StopConfig::default()
                    },
                    0,
                    None,
                )
                .await
                .unwrap();

                assert_eq!(
                    report
                        .metrics
                        .finite_value(crate::metrics_core::MetricTag::RequestCount),
                    Some(4.0)
                );
                assert!(
                    report
                        .metrics
                        .result(crate::metrics_core::MetricTag::RequestLatency)
                        .and_then(crate::metrics_core::MetricResult::distribution)
                        .is_some()
                );
                assert!(
                    report
                        .metrics
                        .result(crate::metrics_core::MetricTag::EffectiveConcurrency)
                        .and_then(crate::metrics_core::MetricResult::distribution)
                        .is_some()
                );
                let native = crate::metrics_core::NativeReport::new(&report.metrics, None);
                let json = serde_json::to_value(native).unwrap();
                assert_eq!(json["schema_version"], "2.0");
                assert_eq!(json["metrics"]["request_count"]["type"], "counter");
                assert_eq!(
                    json["metrics"]["effective_concurrency"]["type"],
                    "distribution"
                );
            })
            .await;
    }

    #[tokio::test]
    async fn request_rate_paces_arrivals_by_the_clock() {
        // Constant 1000 req/s over a fast mock: the pacer sleeps ~1ms between the N
        // arrivals, so wall time >= (N-1)ms even though the mock replies instantly.
        // Open-loop (no concurrency cap); bounded by request count.
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let n = 20usize;
                let rate = 1000.0; // 1ms interval
                let wl = SkeletonWorkload {
                    num_requests: n,
                    input_tokens: 8,
                    output_tokens: 1,
                    turns: 1,
                    think_time_ms: None,
                };
                let stop = StopConfig {
                    total_expected_requests: Some(n as u64),
                    ..Default::default()
                };
                let report = run_paced(
                    base,
                    "m".into(),
                    wl,
                    ArrivalPattern::Constant,
                    Some(rate),
                    None,
                    None,
                    stop,
                    0,
                )
                .await
                .unwrap();
                assert_eq!(report.request_counts.num_requests, n);
                let floor_ms = (n as f64 - 1.0) / rate * 1000.0 * 0.75;
                assert!(
                    report.throughput.wall_time_ms >= floor_ms,
                    "wall {:.2}ms should reflect pacing floor {:.2}ms",
                    report.throughput.wall_time_ms,
                    floor_ms
                );
                assert!(report.latency.ttft.mean_ms.is_finite());
            })
            .await;
    }

    #[tokio::test]
    async fn duration_bound_stops_the_run() {
        // No request-count cap: the run is bounded purely by duration. Burst arrivals
        // + concurrency 4 against a fast mock; a 60ms duration must stop it (and admit
        // more than the handful a count-bound test would).
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let wl = SkeletonWorkload {
                    num_requests: 0, // unused: count bound is None below
                    input_tokens: 4,
                    output_tokens: 1,
                    turns: 1,
                    think_time_ms: None,
                };
                let stop = StopConfig {
                    total_expected_requests: None,
                    expected_num_sessions: None,
                    expected_duration_ns: Some(60_000_000), // 60ms
                };
                let report = run_paced(
                    base,
                    "m".into(),
                    wl,
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(4),
                    stop,
                    0,
                )
                .await
                .unwrap();
                assert!(
                    report.request_counts.num_requests > 0,
                    "duration run should admit at least one request"
                );
                assert!(report.throughput.output_throughput_tok_s.is_finite());
            })
            .await;
    }

    #[tokio::test]
    async fn adaptive_online_failure_stops_early_and_writes_schema_v2_artifacts() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let artifact_dir = std::env::temp_dir()
                    .join(format!("aiperf-adaptive-test-{}", uuid::Uuid::new_v4()));
                let workload = SkeletonWorkload {
                    num_requests: 0,
                    input_tokens: 4,
                    output_tokens: 1,
                    turns: 1,
                    think_time_ms: None,
                };
                let stop = StopConfig {
                    expected_duration_ns: Some(3_000_000_000),
                    ..Default::default()
                };
                let report = run_paced_adaptive(
                    base,
                    "m".into(),
                    workload,
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(2),
                    Some(2),
                    stop,
                    0,
                    Some(AdaptiveRunConfig {
                        control_variable: AdaptiveControlVariable::Concurrency,
                        minimum: 1.0,
                        maximum: 2.0,
                        assessment_period_ns: 1_000_000_000,
                        sustain_duration_ns: 1_000_000_000,
                        min_completed_requests: 1,
                        sla_filters: vec![
                            SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 0.0)
                                .unwrap(),
                        ],
                        step: AdaptiveStepConfig::SlaMargin {
                            base_step: 1,
                            max_step_multiplier: 1,
                        },
                        artifact_dir: artifact_dir.clone(),
                        correlation: CorrelationContext {
                            phase_id: "profiling".to_string(),
                            ..Default::default()
                        },
                    }),
                )
                .await
                .unwrap();

                assert!(report.request_counts.num_requests > 0);
                assert!(
                    report.throughput.wall_time_ms < 2_500.0,
                    "controller should stop before the 3s duration cap"
                );
                let events =
                    std::fs::read_to_string(artifact_dir.join("adaptive_scale_events.jsonl"))
                        .unwrap();
                assert!(events.contains("\"adaptive_phase_started\""));
                assert!(events.contains("\"adaptive_window\""));
                assert!(events.contains("\"adaptive_failed\""));
                let summary: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(artifact_dir.join("adaptive_scale_summary.json")).unwrap(),
                )
                .unwrap();
                assert_eq!(summary["schema_version"], 2);
                assert_eq!(summary["status"], "failed");
                assert_eq!(
                    summary["completed_reason"],
                    "no_sustainable_concurrency_found"
                );
                std::fs::remove_dir_all(artifact_dir).unwrap();
            })
            .await;
    }

    #[tokio::test]
    async fn adaptive_prefill_and_request_rate_actuators_drive_live_issuer_state() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                for (variable, pattern, rate, concurrency, prefill, minimum, maximum, name) in [
                    (
                        AdaptiveControlVariable::PrefillConcurrency,
                        ArrivalPattern::ConcurrencyBurst,
                        None,
                        Some(3),
                        Some(3),
                        1.0,
                        2.0,
                        "prefill_concurrency",
                    ),
                    (
                        AdaptiveControlVariable::RequestRate,
                        ArrivalPattern::Constant,
                        Some(20.0),
                        Some(4),
                        None,
                        5.0,
                        20.0,
                        "request_rate",
                    ),
                ] {
                    let base = crate::test_util::spawn_mock().await;
                    let artifact_dir = std::env::temp_dir()
                        .join(format!("aiperf-adaptive-{name}-{}", uuid::Uuid::new_v4()));
                    let report = run_paced_adaptive(
                        base,
                        "m".into(),
                        SkeletonWorkload {
                            num_requests: 0,
                            input_tokens: 4,
                            output_tokens: 1,
                            turns: 1,
                            think_time_ms: None,
                        },
                        pattern,
                        rate,
                        None,
                        concurrency,
                        prefill,
                        StopConfig {
                            expected_duration_ns: Some(3_000_000_000),
                            ..Default::default()
                        },
                        0,
                        Some(AdaptiveRunConfig {
                            control_variable: variable,
                            minimum,
                            maximum,
                            assessment_period_ns: 1_000_000_000,
                            sustain_duration_ns: 1_000_000_000,
                            min_completed_requests: 1,
                            sla_filters: vec![
                                SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 0.0)
                                    .unwrap(),
                            ],
                            step: AdaptiveStepConfig::SlaMargin {
                                base_step: 1,
                                max_step_multiplier: 1,
                            },
                            artifact_dir: artifact_dir.clone(),
                            correlation: CorrelationContext {
                                phase_id: "profiling".to_string(),
                                ..Default::default()
                            },
                        }),
                    )
                    .await
                    .unwrap();
                    assert!(report.request_counts.num_requests > 0);
                    let summary: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(artifact_dir.join("adaptive_scale_summary.json")).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(summary["control_variable"], name);
                    assert_eq!(summary["status"], "failed");
                    std::fs::remove_dir_all(artifact_dir).unwrap();
                }
            })
            .await;
    }

    #[tokio::test]
    async fn adaptive_users_controls_the_live_user_centric_pool() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let source = synthetic_prepared_source(2, 4, 1, None, "m").await;
                let artifact_dir = std::env::temp_dir()
                    .join(format!("aiperf-adaptive-users-{}", uuid::Uuid::new_v4()));
                let report = run_user_centric_adaptive_online(
                    base,
                    "m".into(),
                    source,
                    UserCentricConfig {
                        num_users: 1,
                        request_rate: 20.0,
                        concurrency: Some(2),
                    },
                    StopConfig {
                        expected_duration_ns: Some(3_000_000_000),
                        ..Default::default()
                    },
                    false,
                    AdaptiveRunConfig {
                        control_variable: AdaptiveControlVariable::Users,
                        minimum: 1.0,
                        maximum: 2.0,
                        assessment_period_ns: 1_000_000_000,
                        sustain_duration_ns: 1_000_000_000,
                        min_completed_requests: 1,
                        sla_filters: vec![
                            SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 0.0)
                                .unwrap(),
                        ],
                        step: AdaptiveStepConfig::SlaMargin {
                            base_step: 1,
                            max_step_multiplier: 1,
                        },
                        artifact_dir: artifact_dir.clone(),
                        correlation: CorrelationContext {
                            phase_id: "profiling".to_string(),
                            ..Default::default()
                        },
                    },
                )
                .await
                .unwrap();
                assert!(report.performance.request_counts.num_requests > 0);
                let control = report.user_control.expect("user-control snapshot");
                assert_eq!(control.target_value, 1);
                let summary: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(artifact_dir.join("adaptive_scale_summary.json")).unwrap(),
                )
                .unwrap();
                assert_eq!(summary["control_variable"], "users");
                assert_eq!(summary["status"], "failed");
                std::fs::remove_dir_all(artifact_dir).unwrap();
            })
            .await;
    }
}
