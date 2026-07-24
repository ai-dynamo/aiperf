// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `discover -> sustain -> complete` adaptive controller.
//!
//! The discovered boundary
//! is the last passing value, sustain allows one recovery until a subsequent
//! passing window resets it, and an inconclusive window changes no control
//! state.

use std::rc::Rc;

use serde::Serialize;

use crate::adaptive_core::actuator::{ControlActuator, ControlSnapshot};
use crate::adaptive_core::error::AdaptiveError;
use crate::adaptive_core::sla::{
    SlaEvaluator, SlaFilter, SlaValues, can_evaluate_without_successes,
};
use crate::adaptive_core::step::{StepPolicy, StepPolicySnapshot};
use crate::adaptive_core::window::WindowStats;

/// Adaptive controller lifecycle phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerPhase {
    /// Monotone upward boundary discovery.
    Discover,
    /// Hold at the last passing boundary.
    Sustain,
    /// Terminal state.
    Complete,
}

/// Terminal adaptive result classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveStatus {
    /// Boundary discovery and sustain completed.
    Completed,
    /// Maximum configured control passed without finding saturation.
    Incomplete,
    /// No sustainable value or sustain recovery was found.
    Failed,
}

/// Adaptive event names written to the JSONL artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerEventKind {
    /// Adaptive phase initialization.
    AdaptivePhaseStarted,
    /// One tumbling assessment window.
    AdaptiveWindow,
    /// One control-value change or sustain pass.
    AdaptiveDecision,
    /// Transition into sustain.
    SustainStarted,
    /// Conservative last-passing boundary discovery.
    BoundaryDiscovered,
    /// Successful terminal state.
    AdaptiveComplete,
    /// Failed terminal state.
    AdaptiveFailed,
    /// Maximum reached without saturation.
    AdaptiveIncomplete,
}

/// One controller-produced event before run correlation is attached.
#[derive(Clone, Debug)]
pub struct ControllerEvent {
    /// Event type.
    pub kind: ControllerEventKind,
    /// Controller phase at event emission.
    pub phase: ControllerPhase,
    /// Human-readable decision or terminal reason.
    pub reason: String,
    /// Primary SLA value, when evaluated.
    pub sla_value: Option<f64>,
    /// Every evaluated SLA value.
    pub sla_values: SlaValues,
    /// Tightest SLA key, when margins were available.
    pub binding_sla: Option<String>,
    /// Completed-request throughput for the window.
    pub throughput: f64,
    /// Successful request count.
    pub sample_count: usize,
    /// Failed request count.
    pub error_count: usize,
    /// Cancelled request count.
    pub cancelled_count: usize,
    /// Control value before a decision.
    pub control_before: f64,
    /// Control value after a decision.
    pub control_after: f64,
    /// Effective control snapshot after the event.
    pub control_snapshot: ControlSnapshot,
    /// Boundary known at event emission.
    pub boundary_value: Option<f64>,
    /// Last passing value known at event emission.
    pub last_passing_value: Option<f64>,
    /// First failing value known at event emission.
    pub first_failing_value: Option<f64>,
    /// Pass/fail/inconclusive tri-state for a window.
    pub passed: Option<bool>,
    /// Absolute control delta for a decision.
    pub step_size: Option<f64>,
}

/// Candidate record paired with one assessment window.
#[derive(Clone, Debug)]
pub struct CandidateObservation {
    /// Adaptive iteration before the iteration counter advances.
    pub adaptive_iteration: u64,
    /// Control value assessed by the window.
    pub candidate_value: f64,
    /// Completed window data.
    pub stats: WindowStats,
    /// Whether every configured SLA filter passed.
    pub accepted: bool,
    /// Rejection category for a non-accepted candidate.
    pub rejection_reason: Option<&'static str>,
}

/// Result of assessing one tumbling window.
#[derive(Clone, Debug)]
pub struct AssessmentOutcome {
    /// Adaptive iteration assigned to the window.
    pub adaptive_iteration: u64,
    /// Events emitted in source order.
    pub events: Vec<ControllerEvent>,
    /// Candidate summary input for the window.
    pub candidate: Option<CandidateObservation>,
    /// Whether the issuer must stop starting work and drain in-flight requests.
    pub stop_sending: bool,
}

impl AssessmentOutcome {
    fn empty(iteration: u64) -> Self {
        Self {
            adaptive_iteration: iteration,
            events: Vec::new(),
            candidate: None,
            stop_sending: false,
        }
    }
}

/// Serializable state needed to construct a terminal summary.
#[derive(Clone, Debug)]
pub struct ControllerSnapshot {
    /// Current lifecycle phase.
    pub phase: ControllerPhase,
    /// Stable control-variable name.
    pub control_variable: &'static str,
    /// Current control value.
    pub control_value: f64,
    /// Current target/actual control snapshot.
    pub control: ControlSnapshot,
    /// Inclusive minimum control value.
    pub minimum: f64,
    /// Inclusive maximum control value.
    pub maximum: f64,
    /// Conservative discovered boundary.
    pub boundary_value: Option<f64>,
    /// Most recent passing control value.
    pub last_passing_value: Option<f64>,
    /// First failing discovery value.
    pub first_failing_value: Option<f64>,
    /// Clock timestamp at sustain entry.
    pub sustain_started_at_ns: Option<i64>,
    /// Number of conclusive sustain windows.
    pub sustain_windows: usize,
    /// Number of passing sustain windows.
    pub sustain_passed_windows: usize,
    /// Idempotent terminal reason.
    pub completed_reason: Option<String>,
    /// Terminal status classification.
    pub status: Option<AdaptiveStatus>,
    /// Next adaptive iteration number.
    pub adaptive_iteration: u64,
    /// Selected step policy and parameters.
    pub step_policy: StepPolicySnapshot,
}

/// Options specific to `ramp_until_fail`.
#[derive(Clone, Copy, Debug)]
pub struct RampUntilFailOptions {
    /// Minimum successful completions needed for a conclusive window.
    pub min_completed_requests: usize,
    /// Required sustain hold duration in nanoseconds.
    pub sustain_duration_ns: i64,
}

impl RampUntilFailOptions {
    /// Validate positive sample and duration settings.
    pub fn validate(self) -> Result<Self, AdaptiveError> {
        if self.min_completed_requests == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive minimum completed requests must be >= 1".to_string(),
            ));
        }
        if self.sustain_duration_ns <= 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive sustain duration must be > 0".to_string(),
            ));
        }
        Ok(self)
    }
}

/// Object-safe adaptive controller state machine.
pub trait Controller {
    /// Emit phase-start state once.
    fn start(&mut self, now_ns: i64) -> Result<Option<ControllerEvent>, AdaptiveError>;
    /// Assess one completed tumbling window.
    fn assess(
        &mut self,
        stats: WindowStats,
        now_ns: i64,
    ) -> Result<AssessmentOutcome, AdaptiveError>;
    /// Complete a naturally stopped phase if the controller is not terminal.
    fn complete_phase(&mut self, now_ns: i64) -> Result<AssessmentOutcome, AdaptiveError>;
    /// Convert an assessment-task failure into an idempotent terminal failure.
    fn fail_assessment(
        &mut self,
        message: &str,
        now_ns: i64,
    ) -> Result<AssessmentOutcome, AdaptiveError>;
    /// Current controller state.
    fn snapshot(&self) -> ControllerSnapshot;
    /// Primary SLA filter used by compact artifacts.
    fn primary_sla(&self) -> &SlaFilter;
}

/// Built-in monotone ramp, step-back, and sustain controller.
pub struct RampUntilFailController {
    actuator: Rc<dyn ControlActuator>,
    evaluator: Box<dyn SlaEvaluator>,
    step_policy: Box<dyn StepPolicy>,
    filters: Vec<SlaFilter>,
    options: RampUntilFailOptions,
    phase: ControllerPhase,
    boundary_value: Option<f64>,
    last_passing_value: Option<f64>,
    first_failing_value: Option<f64>,
    sustain_started_at_ns: Option<i64>,
    sustain_recovery_used: bool,
    sustain_windows: usize,
    sustain_passed_windows: usize,
    completed_reason: Option<String>,
    status: Option<AdaptiveStatus>,
    adaptive_iteration: u64,
    started: bool,
}

impl RampUntilFailController {
    /// Build and validate a `ramp_until_fail` controller, setting the actuator to
    /// its minimum before the phase begins.
    pub fn new(
        actuator: Rc<dyn ControlActuator>,
        evaluator: Box<dyn SlaEvaluator>,
        step_policy: Box<dyn StepPolicy>,
        filters: Vec<SlaFilter>,
        options: RampUntilFailOptions,
    ) -> Result<Self, AdaptiveError> {
        let options = options.validate()?;
        evaluator.validate_filters(&filters)?;
        if actuator.maximum() <= actuator.minimum() {
            return Err(AdaptiveError::InvalidConfig(format!(
                "adaptive {} maximum must be > minimum",
                actuator.variable()
            )));
        }
        actuator.set(actuator.minimum())?;
        Ok(Self {
            actuator,
            evaluator,
            step_policy,
            filters,
            options,
            phase: ControllerPhase::Discover,
            boundary_value: None,
            last_passing_value: None,
            first_failing_value: None,
            sustain_started_at_ns: None,
            sustain_recovery_used: false,
            sustain_windows: 0,
            sustain_passed_windows: 0,
            completed_reason: None,
            status: None,
            adaptive_iteration: 0,
            started: false,
        })
    }

    fn assess_inner(
        &mut self,
        stats: WindowStats,
        now_ns: i64,
        iteration: u64,
    ) -> Result<AssessmentOutcome, AdaptiveError> {
        if self.phase == ControllerPhase::Complete {
            return Ok(AssessmentOutcome::empty(iteration));
        }
        let candidate_value = self.actuator.current();
        let mut outcome = AssessmentOutcome::empty(iteration);

        // A window with zero successful requests but a saturated failure mode
        // (all errors or all cancellations) is normally discarded as
        // inconclusive. When every SLA filter targets an error_rate/
        // cancellation_rate metric, however, that window is exactly what the
        // SLA is measuring — evaluate it instead of early-returning, so an
        // error/cancellation-rate-only config can still converge.
        let can_evaluate_zero_success = stats.completed() == 0
            && (stats.errors > 0 || stats.cancelled > 0)
            && can_evaluate_without_successes(&self.filters, &stats);

        if stats.completed() == 0
            && (stats.errors > 0 || stats.cancelled > 0)
            && !can_evaluate_zero_success
        {
            outcome.events.push(self.event(
                ControllerEventKind::AdaptiveWindow,
                "no successful requests in assessment window".to_string(),
                None,
                SlaValues::new(),
                None,
                &stats,
                candidate_value,
                Some(false),
                None,
            ));
            outcome.candidate = Some(CandidateObservation {
                adaptive_iteration: iteration,
                candidate_value,
                stats: stats.clone(),
                accepted: false,
                rejection_reason: Some("error_threshold"),
            });
            self.assess_failed_window(&stats, now_ns, &mut outcome)?;
            return Ok(outcome);
        }

        if stats.completed() < self.options.min_completed_requests && !can_evaluate_zero_success {
            outcome.events.push(self.event(
                ControllerEventKind::AdaptiveWindow,
                "inconclusive: completed request count below minimum".to_string(),
                None,
                SlaValues::new(),
                None,
                &stats,
                candidate_value,
                None,
                None,
            ));
            outcome.candidate = Some(CandidateObservation {
                adaptive_iteration: iteration,
                candidate_value,
                stats,
                accepted: false,
                rejection_reason: Some("insufficient_samples"),
            });
            return Ok(outcome);
        }

        let sla_values = self.evaluator.values(&self.filters, &stats)?;
        let primary_key = self.evaluator.key(&self.filters[0]);
        let primary_value = *sla_values.get(&primary_key).ok_or_else(|| {
            AdaptiveError::Evaluation(format!("primary SLA value {primary_key} was not produced"))
        })?;
        let passing = self.evaluator.passes(&self.filters, &sla_values)?;
        let binding = self.evaluator.binding_key(&self.filters, &sla_values);
        outcome.events.push(self.event(
            ControllerEventKind::AdaptiveWindow,
            "SLA window evaluated".to_string(),
            Some(primary_value),
            sla_values.clone(),
            binding,
            &stats,
            candidate_value,
            Some(passing),
            None,
        ));
        outcome.candidate = Some(CandidateObservation {
            adaptive_iteration: iteration,
            candidate_value,
            stats: stats.clone(),
            accepted: passing,
            rejection_reason: (!passing).then_some("sla_miss"),
        });

        match self.phase {
            ControllerPhase::Discover => self.assess_discover(
                primary_value,
                passing,
                &stats,
                &sla_values,
                now_ns,
                &mut outcome,
            )?,
            ControllerPhase::Sustain => self.assess_sustain(
                Some(primary_value),
                passing,
                &stats,
                Some(&sla_values),
                None,
                now_ns,
                &mut outcome,
            )?,
            ControllerPhase::Complete => {}
        }
        Ok(outcome)
    }

    fn assess_failed_window(
        &mut self,
        stats: &WindowStats,
        now_ns: i64,
        outcome: &mut AssessmentOutcome,
    ) -> Result<(), AdaptiveError> {
        let reason = "all requests failed in assessment window";
        match self.phase {
            ControllerPhase::Discover if self.last_passing_value.is_none() => {
                self.first_failing_value = Some(self.actuator.current());
                self.finish(
                    ControllerEventKind::AdaptiveFailed,
                    "no_sustainable_concurrency_found",
                    AdaptiveStatus::Failed,
                    None,
                    SlaValues::new(),
                    stats,
                    outcome,
                );
            }
            ControllerPhase::Discover => {
                self.first_failing_value = Some(self.actuator.current());
                self.enter_sustain(None, stats, reason, now_ns, outcome)?;
            }
            ControllerPhase::Sustain => {
                self.assess_sustain(None, false, stats, None, Some(reason), now_ns, outcome)?
            }
            ControllerPhase::Complete => {}
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assess_discover(
        &mut self,
        sla_value: f64,
        passing: bool,
        stats: &WindowStats,
        sla_values: &SlaValues,
        now_ns: i64,
        outcome: &mut AssessmentOutcome,
    ) -> Result<(), AdaptiveError> {
        if passing {
            self.last_passing_value = Some(self.actuator.current());
            if self.actuator.current() >= self.actuator.maximum() {
                self.finish(
                    ControllerEventKind::AdaptiveIncomplete,
                    "max_control_value_reached_without_saturation",
                    AdaptiveStatus::Incomplete,
                    Some(sla_value),
                    sla_values.clone(),
                    stats,
                    outcome,
                );
                return Ok(());
            }
            let before = self.actuator.current();
            let step = self.step_policy.step_size(
                before,
                &self.filters,
                Some(sla_values),
                self.evaluator.as_ref(),
            );
            let next = (before + step).min(self.actuator.maximum());
            let after = self.actuator.set(next)?;
            outcome.events.push(self.event(
                ControllerEventKind::AdaptiveDecision,
                format!("SLA value {sla_value:.3} passes configured filters"),
                Some(sla_value),
                sla_values.clone(),
                self.evaluator.binding_key(&self.filters, sla_values),
                stats,
                before,
                None,
                Some((after - before).abs()),
            ));
            return Ok(());
        }

        self.first_failing_value = Some(self.actuator.current());
        if self.last_passing_value.is_none() {
            self.finish(
                ControllerEventKind::AdaptiveFailed,
                "no_sustainable_concurrency_found",
                AdaptiveStatus::Failed,
                Some(sla_value),
                sla_values.clone(),
                stats,
                outcome,
            );
        } else {
            self.enter_sustain(
                Some(sla_value),
                stats,
                &format!("SLA value {sla_value:.3} breaches configured filters"),
                now_ns,
                outcome,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assess_sustain(
        &mut self,
        sla_value: Option<f64>,
        passing: bool,
        stats: &WindowStats,
        sla_values: Option<&SlaValues>,
        reason: Option<&str>,
        now_ns: i64,
        outcome: &mut AssessmentOutcome,
    ) -> Result<(), AdaptiveError> {
        self.sustain_windows += 1;
        if passing {
            self.sustain_passed_windows += 1;
            self.last_passing_value = Some(self.actuator.current());
            self.sustain_recovery_used = false;
            let value = sla_value.expect("passing sustain has an SLA value");
            outcome.events.push(self.event(
                ControllerEventKind::AdaptiveDecision,
                format!("SLA value {value:.3} passes configured filters during sustain"),
                sla_value,
                sla_values.cloned().unwrap_or_default(),
                sla_values.and_then(|values| self.evaluator.binding_key(&self.filters, values)),
                stats,
                self.actuator.current(),
                None,
                None,
            ));
            if let Some(started) = self.sustain_started_at_ns
                && now_ns.saturating_sub(started) >= self.options.sustain_duration_ns
            {
                self.finish(
                    ControllerEventKind::AdaptiveComplete,
                    "sustain_duration_completed",
                    AdaptiveStatus::Completed,
                    sla_value,
                    sla_values.cloned().unwrap_or_default(),
                    stats,
                    outcome,
                );
            }
            return Ok(());
        }

        if self.sustain_recovery_used {
            self.finish(
                ControllerEventKind::AdaptiveFailed,
                "sustain_failed_after_recovery",
                AdaptiveStatus::Failed,
                sla_value,
                sla_values.cloned().unwrap_or_default(),
                stats,
                outcome,
            );
            return Ok(());
        }
        self.sustain_recovery_used = true;
        let before = self.actuator.current();
        let last_good_target = self
            .last_passing_value
            .unwrap_or(self.actuator.minimum())
            .max(self.actuator.minimum());
        let target = if last_good_target < before {
            last_good_target
        } else {
            let step = self.step_policy.step_size(
                before,
                &self.filters,
                sla_values,
                self.evaluator.as_ref(),
            );
            (before - step).max(self.actuator.minimum())
        };
        if target == before && before == self.actuator.minimum() {
            self.finish(
                ControllerEventKind::AdaptiveFailed,
                "sustain_failed_sla_unrecoverable",
                AdaptiveStatus::Failed,
                sla_value,
                sla_values.cloned().unwrap_or_default(),
                stats,
                outcome,
            );
            return Ok(());
        }
        let after = self.actuator.set(target)?;
        let reason = reason.map(str::to_string).unwrap_or_else(|| {
            format!(
                "SLA value {:.3} breaches configured filters during sustain",
                sla_value.unwrap_or(f64::INFINITY)
            )
        });
        outcome.events.push(self.event(
            ControllerEventKind::AdaptiveDecision,
            reason,
            sla_value,
            sla_values.cloned().unwrap_or_default(),
            sla_values.and_then(|values| self.evaluator.binding_key(&self.filters, values)),
            stats,
            before,
            None,
            Some((before - after).abs()),
        ));
        Ok(())
    }

    fn enter_sustain(
        &mut self,
        sla_value: Option<f64>,
        stats: &WindowStats,
        reason: &str,
        now_ns: i64,
        outcome: &mut AssessmentOutcome,
    ) -> Result<(), AdaptiveError> {
        let last_good = self.last_passing_value.ok_or_else(|| {
            AdaptiveError::Evaluation("cannot enter sustain without a passing boundary".to_string())
        })?;
        let boundary = last_good.max(self.actuator.minimum());
        let before = self.actuator.current();
        self.boundary_value = Some(boundary);
        self.actuator.set(boundary)?;
        self.phase = ControllerPhase::Sustain;
        self.sustain_started_at_ns = Some(now_ns);
        outcome.events.push(self.event(
            ControllerEventKind::SustainStarted,
            format!("holding boundary_value={boundary}"),
            sla_value,
            SlaValues::new(),
            None,
            stats,
            before,
            None,
            Some((before - boundary).abs()),
        ));
        outcome.events.push(self.event(
            ControllerEventKind::BoundaryDiscovered,
            reason.to_string(),
            sla_value,
            SlaValues::new(),
            None,
            stats,
            boundary,
            None,
            None,
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        kind: ControllerEventKind,
        reason: &str,
        status: AdaptiveStatus,
        sla_value: Option<f64>,
        sla_values: SlaValues,
        stats: &WindowStats,
        outcome: &mut AssessmentOutcome,
    ) {
        if self.completed_reason.is_some() {
            return;
        }
        self.phase = ControllerPhase::Complete;
        self.completed_reason = Some(reason.to_string());
        self.status = Some(status);
        outcome.events.push(self.event(
            kind,
            reason.to_string(),
            sla_value,
            sla_values.clone(),
            self.evaluator.binding_key(&self.filters, &sla_values),
            stats,
            self.actuator.current(),
            None,
            None,
        ));
        outcome.stop_sending = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn event(
        &self,
        kind: ControllerEventKind,
        reason: String,
        sla_value: Option<f64>,
        sla_values: SlaValues,
        binding_sla: Option<String>,
        stats: &WindowStats,
        control_before: f64,
        passed: Option<bool>,
        step_size: Option<f64>,
    ) -> ControllerEvent {
        ControllerEvent {
            kind,
            phase: self.phase,
            reason,
            sla_value,
            sla_values,
            binding_sla,
            throughput: stats.throughput(),
            sample_count: stats.completed(),
            error_count: stats.errors,
            cancelled_count: stats.cancelled,
            control_before,
            control_after: self.actuator.current(),
            control_snapshot: self.actuator.snapshot(),
            boundary_value: self.boundary_value,
            last_passing_value: self.last_passing_value,
            first_failing_value: self.first_failing_value,
            passed,
            step_size,
        }
    }

    fn terminal_without_window(
        &mut self,
        kind: ControllerEventKind,
        reason: String,
        status: AdaptiveStatus,
        now_ns: i64,
    ) -> AssessmentOutcome {
        let iteration = self.adaptive_iteration;
        if self.completed_reason.is_some() {
            return AssessmentOutcome::empty(iteration);
        }
        self.phase = ControllerPhase::Complete;
        self.completed_reason = Some(reason.clone());
        self.status = Some(status);
        let stats = WindowStats::empty(now_ns, now_ns);
        let event = self.event(
            kind,
            reason,
            None,
            SlaValues::new(),
            None,
            &stats,
            self.actuator.current(),
            None,
            None,
        );
        AssessmentOutcome {
            adaptive_iteration: iteration,
            events: vec![event],
            candidate: None,
            stop_sending: true,
        }
    }
}

impl Controller for RampUntilFailController {
    fn start(&mut self, now_ns: i64) -> Result<Option<ControllerEvent>, AdaptiveError> {
        if self.started {
            return Ok(None);
        }
        self.started = true;
        self.actuator.set(self.actuator.minimum())?;
        let stats = WindowStats::empty(now_ns, now_ns);
        Ok(Some(self.event(
            ControllerEventKind::AdaptivePhaseStarted,
            "adaptive scale discover phase started".to_string(),
            None,
            SlaValues::new(),
            None,
            &stats,
            self.actuator.current(),
            None,
            None,
        )))
    }

    fn assess(
        &mut self,
        stats: WindowStats,
        now_ns: i64,
    ) -> Result<AssessmentOutcome, AdaptiveError> {
        let iteration = self.adaptive_iteration;
        let result = self.assess_inner(stats, now_ns, iteration);
        self.adaptive_iteration += 1;
        result
    }

    fn complete_phase(&mut self, now_ns: i64) -> Result<AssessmentOutcome, AdaptiveError> {
        Ok(self.terminal_without_window(
            ControllerEventKind::AdaptiveComplete,
            "phase_stopped".to_string(),
            AdaptiveStatus::Completed,
            now_ns,
        ))
    }

    fn fail_assessment(
        &mut self,
        message: &str,
        now_ns: i64,
    ) -> Result<AssessmentOutcome, AdaptiveError> {
        Ok(self.terminal_without_window(
            ControllerEventKind::AdaptiveFailed,
            format!("assessment_failed: {message}"),
            AdaptiveStatus::Failed,
            now_ns,
        ))
    }

    fn snapshot(&self) -> ControllerSnapshot {
        ControllerSnapshot {
            phase: self.phase,
            control_variable: self.actuator.variable(),
            control_value: self.actuator.current(),
            control: self.actuator.snapshot(),
            minimum: self.actuator.minimum(),
            maximum: self.actuator.maximum(),
            boundary_value: self.boundary_value,
            last_passing_value: self.last_passing_value,
            first_failing_value: self.first_failing_value,
            sustain_started_at_ns: self.sustain_started_at_ns,
            sustain_windows: self.sustain_windows,
            sustain_passed_windows: self.sustain_passed_windows,
            completed_reason: self.completed_reason.clone(),
            status: self.status,
            adaptive_iteration: self.adaptive_iteration,
            step_policy: self.step_policy.snapshot(),
        }
    }

    fn primary_sla(&self) -> &SlaFilter {
        &self.filters[0]
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::adaptive_core::sla::{DefaultSlaEvaluator, SlaOp, SlaStat};
    use crate::adaptive_core::step::SlaMarginStep;
    use crate::adaptive_core::window::RequestSample;

    struct CellActuator {
        current: Cell<f64>,
        minimum: f64,
        maximum: f64,
    }

    impl ControlActuator for CellActuator {
        fn variable(&self) -> &'static str {
            "concurrency"
        }
        fn minimum(&self) -> f64 {
            self.minimum
        }
        fn maximum(&self) -> f64 {
            self.maximum
        }
        fn current(&self) -> f64 {
            self.current.get()
        }
        fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
            let value = value.clamp(self.minimum, self.maximum);
            self.current.set(value);
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

    fn controller(maximum: f64, sustain_ns: i64) -> RampUntilFailController {
        let actuator = Rc::new(CellActuator {
            current: Cell::new(2.0),
            minimum: 2.0,
            maximum,
        });
        let filter = SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 100.0).unwrap();
        RampUntilFailController::new(
            actuator,
            Box::new(DefaultSlaEvaluator),
            Box::new(SlaMarginStep::new(2, 1).unwrap()),
            vec![filter],
            RampUntilFailOptions {
                min_completed_requests: 1,
                sustain_duration_ns: sustain_ns,
            },
        )
        .unwrap()
    }

    fn stats(latency_ms: i64) -> WindowStats {
        WindowStats {
            successful_requests: vec![RequestSample {
                request_latency_ns: latency_ms * 1_000_000,
                ttft_ns: Some(10_000_000),
                inter_token_latency_ns: Some(1_000_000.0),
                output_sequence_length: Some(8),
            }],
            errors: 0,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 0,
            end_ns: 1_000_000_000,
        }
    }

    #[test]
    fn discover_ramps_then_steps_back_to_last_good_boundary() {
        let mut controller = controller(8.0, 2_000_000_000);
        controller.start(0).unwrap();
        let first = controller.assess(stats(20), 1_000_000_000).unwrap();
        assert!(first.events.iter().any(|event| {
            event.kind == ControllerEventKind::AdaptiveDecision && event.control_after == 4.0
        }));
        controller.assess(stats(20), 2_000_000_000).unwrap();
        let failing = controller.assess(stats(150), 3_000_000_000).unwrap();
        assert_eq!(controller.snapshot().phase, ControllerPhase::Sustain);
        assert_eq!(controller.snapshot().boundary_value, Some(4.0));
        assert_eq!(controller.snapshot().first_failing_value, Some(6.0));
        assert_eq!(controller.snapshot().control_value, 4.0);
        assert_eq!(
            failing
                .events
                .iter()
                .map(|event| event.kind)
                .collect::<Vec<_>>(),
            vec![
                ControllerEventKind::AdaptiveWindow,
                ControllerEventKind::SustainStarted,
                ControllerEventKind::BoundaryDiscovered,
            ]
        );
    }

    #[test]
    fn first_failing_minimum_is_terminal() {
        let mut controller = controller(8.0, 1);
        let outcome = controller.assess(stats(150), 1).unwrap();
        assert!(outcome.stop_sending);
        assert_eq!(controller.snapshot().status, Some(AdaptiveStatus::Failed));
        assert_eq!(
            controller.snapshot().completed_reason.as_deref(),
            Some("no_sustainable_concurrency_found")
        );
    }

    #[test]
    fn sparse_window_is_inconclusive_and_still_advances_iteration() {
        let mut controller = controller(8.0, 1);
        controller.options.min_completed_requests = 2;
        let outcome = controller.assess(stats(20), 1).unwrap();
        assert_eq!(outcome.events[0].passed, None);
        assert_eq!(controller.snapshot().control_value, 2.0);
        assert_eq!(controller.snapshot().adaptive_iteration, 1);
    }

    #[test]
    fn sustain_allows_one_recovery_then_fails() {
        let mut controller = controller(8.0, i64::MAX);
        controller.assess(stats(20), 1).unwrap();
        controller.assess(stats(20), 2).unwrap();
        controller.assess(stats(150), 3).unwrap();
        let recovery = controller.assess(stats(150), 4).unwrap();
        assert!(!recovery.stop_sending);
        let failed = controller.assess(stats(150), 5).unwrap();
        assert!(failed.stop_sending);
        assert_eq!(
            controller.snapshot().completed_reason.as_deref(),
            Some("sustain_failed_after_recovery")
        );
    }

    #[test]
    fn passing_sustain_window_completes_after_clock_duration() {
        let mut controller = controller(8.0, 10);
        controller.assess(stats(20), 0).unwrap();
        controller.assess(stats(150), 1).unwrap();
        let completed = controller.assess(stats(20), 11).unwrap();
        assert!(completed.stop_sending);
        assert_eq!(
            controller.snapshot().status,
            Some(AdaptiveStatus::Completed)
        );
    }

    #[test]
    fn passing_maximum_is_incomplete_without_a_boundary() {
        let mut controller = controller(4.0, 10);
        controller.assess(stats(20), 1).unwrap();
        let outcome = controller.assess(stats(20), 2).unwrap();
        assert!(outcome.stop_sending);
        assert_eq!(
            controller.snapshot().status,
            Some(AdaptiveStatus::Incomplete)
        );
        assert_eq!(controller.snapshot().last_passing_value, Some(4.0));
        assert_eq!(controller.snapshot().boundary_value, None);
        assert!(
            outcome
                .events
                .iter()
                .any(|event| event.kind == ControllerEventKind::AdaptiveIncomplete)
        );
    }

    #[test]
    fn all_failed_discover_window_uses_last_good_boundary() {
        let mut controller = controller(8.0, 10);
        controller.assess(stats(20), 1).unwrap();
        let failed = WindowStats {
            successful_requests: Vec::new(),
            errors: 3,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 1,
            end_ns: 2,
        };
        let outcome = controller.assess(failed, 2).unwrap();
        assert_eq!(controller.snapshot().phase, ControllerPhase::Sustain);
        assert_eq!(controller.snapshot().boundary_value, Some(2.0));
        assert_eq!(controller.snapshot().first_failing_value, Some(4.0));
        assert_eq!(
            outcome.candidate.unwrap().rejection_reason,
            Some("error_threshold")
        );
    }

    fn controller_with_filter(
        filter: SlaFilter,
        maximum: f64,
        sustain_ns: i64,
    ) -> RampUntilFailController {
        let actuator = Rc::new(CellActuator {
            current: Cell::new(2.0),
            minimum: 2.0,
            maximum,
        });
        RampUntilFailController::new(
            actuator,
            Box::new(DefaultSlaEvaluator),
            Box::new(SlaMarginStep::new(2, 1).unwrap()),
            vec![filter],
            RampUntilFailOptions {
                min_completed_requests: 1,
                sustain_duration_ns: sustain_ns,
            },
        )
        .unwrap()
    }

    #[test]
    fn zero_success_error_window_is_evaluated_under_error_rate_sla() {
        // With an error_rate SLA, a window of all errors and zero successes
        // must be EVALUATED as an SLA miss (error_rate=1.0 > 0.5), not
        // discarded as "no successful requests" — the fix that lets an
        // error-rate-only config converge.
        let filter = SlaFilter::new("error_rate", SlaStat::Avg, SlaOp::Le, 0.5).unwrap();
        let mut controller = controller_with_filter(filter, 8.0, 10);
        let failed = WindowStats {
            successful_requests: Vec::new(),
            errors: 4,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 1,
            end_ns: 2,
        };
        let outcome = controller.assess(failed, 1).unwrap();
        let candidate = outcome.candidate.expect("candidate produced");
        assert!(!candidate.accepted);
        // sla_miss (evaluated), NOT error_threshold (early-returned/discarded).
        assert_eq!(candidate.rejection_reason, Some("sla_miss"));
        assert!(
            outcome
                .events
                .iter()
                .any(|event| event.reason == "SLA window evaluated")
        );
    }

    #[test]
    fn passing_window_resets_recovery_for_one_more_step_down() {
        let mut controller = controller(10.0, i64::MAX);
        controller.assess(stats(20), 1).unwrap(); // 2 -> 4
        controller.assess(stats(20), 2).unwrap(); // 4 -> 6
        controller.assess(stats(20), 3).unwrap(); // 6 -> 8
        controller.assess(stats(150), 4).unwrap(); // boundary 6
        let first_recovery = controller.assess(stats(150), 5).unwrap(); // 6 -> 4
        assert!(!first_recovery.stop_sending);
        controller.assess(stats(20), 6).unwrap(); // pass resets recovery
        let second_recovery = controller.assess(stats(150), 7).unwrap(); // 4 -> 2
        assert!(!second_recovery.stop_sending);
        assert_eq!(controller.snapshot().control_value, 2.0);
    }

    #[test]
    fn phase_stop_is_idempotent_and_completed() {
        let mut controller = controller(8.0, 10);
        let first = controller.complete_phase(1).unwrap();
        let second = controller.complete_phase(2).unwrap();
        assert_eq!(first.events.len(), 1);
        assert!(second.events.is_empty());
        assert_eq!(
            controller.snapshot().status,
            Some(AdaptiveStatus::Completed)
        );
        assert_eq!(
            controller.snapshot().completed_reason.as_deref(),
            Some("phase_stopped")
        );
    }

    #[test]
    fn assessment_failure_has_failed_status_and_prefixed_reason() {
        let mut controller = controller(8.0, 10);
        let outcome = controller.fail_assessment("bad window", 1).unwrap();
        assert!(outcome.stop_sending);
        assert_eq!(controller.snapshot().status, Some(AdaptiveStatus::Failed));
        assert_eq!(
            controller.snapshot().completed_reason.as_deref(),
            Some("assessment_failed: bad window")
        );
        assert_eq!(outcome.events[0].kind, ControllerEventKind::AdaptiveFailed);
    }
}
