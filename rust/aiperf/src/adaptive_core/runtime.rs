// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-paced adaptive assessment runtime.
//!
//! The assessment task sleeps for one
//! assessment period on the injected clock, takes a tumbling window, applies
//! controller decisions, and asks the issuer to stop once the controller is
//! terminal.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::task::Poll;

use crate::clock::Clock;

use crate::adaptive_core::artifacts::{
    AdaptiveArtifactSink, AdaptiveCandidate, AdaptiveEvent, AdaptiveSummary, CorrelationContext,
};
use crate::adaptive_core::controller::{
    AssessmentOutcome, Controller, ControllerEvent, ControllerEventKind, ControllerPhase,
};
use crate::adaptive_core::error::AdaptiveError;
use crate::adaptive_core::sla::SlaFilter;
use crate::adaptive_core::window::WindowSampler;

/// Minimum supported assessment period: one second.
pub const MIN_ASSESSMENT_PERIOD_NS: i64 = 1_000_000_000;

/// Shared single-loop window sampler handle.
pub type SharedWindowSampler = Rc<RefCell<Box<dyn WindowSampler>>>;

/// Shared single-loop artifact sink handle.
pub type SharedArtifactSink = Rc<RefCell<Box<dyn AdaptiveArtifactSink>>>;

/// Runtime settings above the pure controller.
#[derive(Clone, Debug)]
pub struct AdaptiveScaleOptions {
    /// Assessment period in nanoseconds; must be at least one second.
    pub assessment_period_ns: i64,
    /// Sustain duration in nanoseconds, repeated into the terminal summary.
    pub sustain_duration_ns: i64,
    /// Run/phase correlation fields.
    pub correlation: CorrelationContext,
}

impl AdaptiveScaleOptions {
    /// Validate assessment and sustain timing.
    pub fn validate(self) -> Result<Self, AdaptiveError> {
        if self.assessment_period_ns < MIN_ASSESSMENT_PERIOD_NS {
            return Err(AdaptiveError::InvalidConfig(format!(
                "adaptive assessment period must be >= {MIN_ASSESSMENT_PERIOD_NS} ns"
            )));
        }
        if self.sustain_duration_ns <= 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive sustain duration must be > 0".to_string(),
            ));
        }
        Ok(self)
    }
}

/// Owns the controller, sampler, and artifact sink for one adaptive phase.
pub struct AdaptiveScale {
    clock: Rc<dyn Clock>,
    sampler: SharedWindowSampler,
    controller: RefCell<Box<dyn Controller>>,
    artifacts: SharedArtifactSink,
    options: AdaptiveScaleOptions,
    primary_sla: SlaFilter,
    active: Cell<bool>,
    stop_sending: Cell<bool>,
    stop_sending_waker: RefCell<Option<std::task::Waker>>,
    summary_written: Cell<bool>,
    candidates: RefCell<Vec<AdaptiveCandidate>>,
    last_error: RefCell<Option<String>>,
}

impl AdaptiveScale {
    /// Build a phase runtime. Call [`start`](Self::start) before dispatching.
    pub fn new(
        clock: Rc<dyn Clock>,
        sampler: SharedWindowSampler,
        controller: Box<dyn Controller>,
        artifacts: SharedArtifactSink,
        options: AdaptiveScaleOptions,
    ) -> Result<Self, AdaptiveError> {
        let options = options.validate()?;
        let primary_sla = controller.primary_sla().clone();
        Ok(Self {
            clock,
            sampler,
            controller: RefCell::new(controller),
            artifacts,
            options,
            primary_sla,
            active: Cell::new(false),
            stop_sending: Cell::new(false),
            stop_sending_waker: RefCell::new(None),
            summary_written: Cell::new(false),
            candidates: RefCell::new(Vec::new()),
            last_error: RefCell::new(None),
        })
    }

    /// Emit `adaptive_phase_started` and enable the assessment loop.
    pub fn start(&self) -> Result<(), AdaptiveError> {
        let now_ns = self.clock.now_ns();
        let event = {
            let mut controller = self.controller.borrow_mut();
            controller.start(now_ns)?
        };
        if let Some(event) = event {
            self.emit_controller_event(&event, now_ns, 0)?;
        }
        self.active.set(true);
        Ok(())
    }

    /// Run clock-paced assessments until the phase deactivates or the
    /// controller reaches `complete`.
    pub async fn assessment_loop(self: Rc<Self>) {
        while self.active.get() && !self.is_complete() {
            self.clock
                .clone()
                .sleep(self.options.assessment_period_ns)
                .await;
            if !self.active.get() || self.is_complete() {
                break;
            }
            if let Err(error) = self.assess_once() {
                *self.last_error.borrow_mut() = Some(error.to_string());
                let _ = self.fail_assessment(&error.to_string());
                break;
            }
        }
    }

    /// Take and assess the window ending at the clock's current timestamp.
    pub fn assess_once(&self) -> Result<(), AdaptiveError> {
        let now_ns = self.clock.now_ns();
        let stats = self.sampler.borrow_mut().take(now_ns);
        let outcome = {
            let mut controller = self.controller.borrow_mut();
            controller.assess(stats, now_ns)?
        };
        self.process_outcome(outcome, now_ns)
    }

    /// Finalize a phase that stopped because its external stop condition fired.
    pub fn complete_phase(&self) -> Result<(), AdaptiveError> {
        self.active.set(false);
        let now_ns = self.clock.now_ns();
        let outcome = {
            let mut controller = self.controller.borrow_mut();
            controller.complete_phase(now_ns)?
        };
        self.process_outcome(outcome, now_ns)
    }

    /// Convert an assessment failure into an `adaptive_failed` terminal record.
    pub fn fail_assessment(&self, message: &str) -> Result<(), AdaptiveError> {
        self.active.set(false);
        let now_ns = self.clock.now_ns();
        let outcome = {
            let mut controller = self.controller.borrow_mut();
            controller.fail_assessment(message, now_ns)?
        };
        self.process_outcome(outcome, now_ns)
    }

    /// Stop the background loop without changing an already-terminal controller.
    pub fn deactivate(&self) {
        self.active.set(false);
    }

    /// Whether the issuer must stop starting new requests.
    pub fn should_stop_sending(&self) -> bool {
        self.stop_sending.get()
    }

    /// Wait until a terminal controller decision asks the issuer to stop.
    ///
    /// This local future lets an issuer select the stop signal against a long
    /// arrival sleep instead of overshooting the terminal decision. It stores
    /// only the current single-loop waker and has no cross-thread synchronization.
    pub async fn wait_until_stop_sending(&self) {
        std::future::poll_fn(|context| {
            if self.stop_sending.get() {
                return Poll::Ready(());
            }
            let mut waiter = self.stop_sending_waker.borrow_mut();
            if waiter
                .as_ref()
                .is_none_or(|waker| !waker.will_wake(context.waker()))
            {
                *waiter = Some(context.waker().clone());
            }
            Poll::Pending
        })
        .await
    }

    /// Whether the controller reached its terminal phase.
    pub fn is_complete(&self) -> bool {
        self.controller.borrow().snapshot().phase == ControllerPhase::Complete
    }

    /// Last assessment or artifact error observed by the background loop.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.borrow().clone()
    }

    /// Shared sampler used by the request observer.
    pub fn sampler(&self) -> &SharedWindowSampler {
        &self.sampler
    }

    fn process_outcome(
        &self,
        outcome: AssessmentOutcome,
        timestamp_ns: i64,
    ) -> Result<(), AdaptiveError> {
        if let Some(candidate) = outcome.candidate {
            self.candidates
                .borrow_mut()
                .push(AdaptiveCandidate::from_observation(candidate));
        }
        if outcome.stop_sending {
            self.stop_sending.set(true);
            if let Some(waker) = self.stop_sending_waker.borrow_mut().take() {
                waker.wake();
            }
            self.active.set(false);
        }

        let terminal = outcome
            .events
            .iter()
            .find(|event| is_terminal(event.kind))
            .cloned();
        for event in &outcome.events {
            self.emit_controller_event(event, timestamp_ns, outcome.adaptive_iteration)?;
        }
        if let Some(terminal) = terminal {
            self.write_summary(&terminal)?;
        }
        Ok(())
    }

    fn emit_controller_event(
        &self,
        event: &ControllerEvent,
        timestamp_ns: i64,
        adaptive_iteration: u64,
    ) -> Result<(), AdaptiveError> {
        let snapshot = self.controller.borrow().snapshot();
        let artifact = AdaptiveEvent::from_controller(
            event,
            timestamp_ns,
            snapshot.control_variable,
            &self.primary_sla,
            &snapshot.step_policy,
            &self.options.correlation,
            adaptive_iteration,
        );
        self.artifacts.borrow_mut().emit_event(&artifact)
    }

    fn write_summary(&self, terminal: &ControllerEvent) -> Result<(), AdaptiveError> {
        if self.summary_written.replace(true) {
            return Ok(());
        }
        let summary = AdaptiveSummary::from_terminal(
            self.controller.borrow().snapshot(),
            &self.primary_sla,
            self.options.sustain_duration_ns,
            terminal,
            self.candidates.borrow().clone(),
        )?;
        self.artifacts.borrow_mut().write_summary(&summary)
    }
}

fn is_terminal(kind: ControllerEventKind) -> bool {
    matches!(
        kind,
        ControllerEventKind::AdaptiveComplete
            | ControllerEventKind::AdaptiveFailed
            | ControllerEventKind::AdaptiveIncomplete
    )
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::clock::SimClock;

    use super::*;
    use crate::adaptive_core::actuator::{ControlActuator, ControlSnapshot};
    use crate::adaptive_core::artifacts::{AdaptiveEvent, AdaptiveSummary};
    use crate::adaptive_core::controller::{RampUntilFailController, RampUntilFailOptions};
    use crate::adaptive_core::sla::{DefaultSlaEvaluator, SlaFilter, SlaOp, SlaStat};
    use crate::adaptive_core::step::SlaMarginStep;
    use crate::adaptive_core::window::TumblingWindowSampler;

    #[derive(Default)]
    struct RecordedArtifacts {
        events: Vec<ControllerEventKind>,
        summaries: usize,
    }

    struct RecordingSink(Rc<RefCell<RecordedArtifacts>>);

    impl AdaptiveArtifactSink for RecordingSink {
        fn emit_event(&mut self, event: &AdaptiveEvent) -> Result<(), AdaptiveError> {
            self.0.borrow_mut().events.push(event.event);
            Ok(())
        }

        fn write_summary(&mut self, _summary: &AdaptiveSummary) -> Result<(), AdaptiveError> {
            self.0.borrow_mut().summaries += 1;
            Ok(())
        }
    }

    struct TestActuator(Cell<f64>);

    impl ControlActuator for TestActuator {
        fn variable(&self) -> &'static str {
            "concurrency"
        }
        fn minimum(&self) -> f64 {
            1.0
        }
        fn maximum(&self) -> f64 {
            3.0
        }
        fn current(&self) -> f64 {
            self.0.get()
        }
        fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
            let value = value.clamp(1.0, 3.0);
            self.0.set(value);
            Ok(value)
        }
        fn snapshot(&self) -> ControlSnapshot {
            ControlSnapshot {
                target_value: self.current(),
                actual_value: self.current(),
                active_users: None,
                retiring_users: None,
                cancelled: None,
            }
        }
    }

    fn runtime(clock: Rc<SimClock>) -> (Rc<AdaptiveScale>, Rc<RefCell<RecordedArtifacts>>) {
        let filter = SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 100.0).unwrap();
        let controller = RampUntilFailController::new(
            Rc::new(TestActuator(Cell::new(1.0))),
            Box::new(DefaultSlaEvaluator),
            Box::new(SlaMarginStep::new(1, 1).unwrap()),
            vec![filter],
            RampUntilFailOptions {
                min_completed_requests: 1,
                sustain_duration_ns: 1_000_000_000,
            },
        )
        .unwrap();
        let sampler: SharedWindowSampler = Rc::new(RefCell::new(Box::new(
            TumblingWindowSampler::new(clock.now_ns()),
        )));
        let recorded = Rc::new(RefCell::new(RecordedArtifacts::default()));
        let sink: SharedArtifactSink =
            Rc::new(RefCell::new(Box::new(RecordingSink(recorded.clone()))));
        let runtime = Rc::new(
            AdaptiveScale::new(
                clock,
                sampler,
                Box::new(controller),
                sink,
                AdaptiveScaleOptions {
                    assessment_period_ns: 1_000_000_000,
                    sustain_duration_ns: 1_000_000_000,
                    correlation: CorrelationContext {
                        phase_id: "profiling".to_string(),
                        ..Default::default()
                    },
                },
            )
            .unwrap(),
        );
        (runtime, recorded)
    }

    #[tokio::test]
    async fn assessment_sleep_is_driven_by_sim_clock() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock = Rc::new(SimClock::new());
                let (runtime, recorded) = runtime(clock.clone());
                runtime.start().unwrap();
                runtime
                    .sampler
                    .borrow_mut()
                    .on_arrival(uuid::Uuid::nil(), 0);
                runtime.sampler.borrow_mut().on_terminal(
                    uuid::Uuid::nil(),
                    loadgen_core::collector::ReplayTerminalStatus::Completed,
                    20_000_000,
                );
                let task = tokio::task::spawn_local(runtime.clone().assessment_loop());
                tokio::task::yield_now().await;
                assert_eq!(clock.next_event_time(), Some(1_000_000_000));
                clock.advance_to(1_000_000_000);
                tokio::task::yield_now().await;
                assert!(
                    recorded
                        .borrow()
                        .events
                        .contains(&ControllerEventKind::AdaptiveWindow)
                );
                runtime.deactivate();
                task.abort();
            })
            .await;
    }

    #[tokio::test]
    async fn terminal_outcome_wakes_a_sleeping_issuer() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock = Rc::new(SimClock::new());
                let (runtime, _) = runtime(clock);
                runtime.start().unwrap();
                let woke = Rc::new(Cell::new(false));
                let waiter_runtime = runtime.clone();
                let waiter_woke = woke.clone();
                let waiter = tokio::task::spawn_local(async move {
                    waiter_runtime.wait_until_stop_sending().await;
                    waiter_woke.set(true);
                });
                tokio::task::yield_now().await;

                runtime.complete_phase().unwrap();
                waiter.await.unwrap();
                assert!(woke.get());
            })
            .await;
    }

    #[test]
    fn terminal_summary_is_idempotent() {
        let clock = Rc::new(SimClock::new());
        let (runtime, recorded) = runtime(clock);
        runtime.start().unwrap();
        runtime.complete_phase().unwrap();
        runtime.complete_phase().unwrap();
        assert_eq!(recorded.borrow().summaries, 1);
    }

    #[test]
    fn options_enforce_one_second_floor() {
        let options = AdaptiveScaleOptions {
            assessment_period_ns: 999_999_999,
            sustain_duration_ns: 1,
            correlation: CorrelationContext::default(),
        };
        assert!(options.validate().is_err());
    }
}
