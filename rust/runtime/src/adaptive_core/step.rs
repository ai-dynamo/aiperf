// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive ramp step policies.
//!
//! The formulas are: fixed-percent uses
//! `max(1, ceil(current * percent / 100))`, while SLA-margin scaling is
//! `base_step * clamp(floor(margin * max_step_multiplier), 1,
//! max_step_multiplier)` over the tightest filter's normalized headroom.

use serde::Serialize;

use crate::adaptive_core::error::AdaptiveError;
use crate::adaptive_core::sla::{SlaEvaluator, SlaFilter, SlaValues};

/// Serializable configuration of a selected step policy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StepPolicySnapshot {
    /// Stable policy name.
    pub name: &'static str,
    /// Base step for SLA-margin scaling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_step: Option<usize>,
    /// Maximum base-step multiplier for SLA-margin scaling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_step_multiplier: Option<usize>,
    /// Percentage of current control for fixed-percent scaling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_percent: Option<f64>,
}

/// Object-safe policy for choosing the next monotone ramp increment.
pub trait StepPolicy {
    /// Compute a positive step from current load and evaluated SLA margins.
    fn step_size(
        &self,
        current: f64,
        filters: &[SlaFilter],
        observed: Option<&SlaValues>,
        evaluator: &dyn SlaEvaluator,
    ) -> f64;
    /// Snapshot policy name and parameters for artifacts.
    fn snapshot(&self) -> StepPolicySnapshot;
}

/// Tightest-SLA-margin-scaled step policy.
pub struct SlaMarginStep {
    base_step: usize,
    max_step_multiplier: usize,
}

impl SlaMarginStep {
    /// Build a policy with positive base and multiplier values.
    pub fn new(base_step: usize, max_step_multiplier: usize) -> Result<Self, AdaptiveError> {
        if base_step == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive SLA-margin base step must be >= 1".to_string(),
            ));
        }
        if max_step_multiplier == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive maximum step multiplier must be >= 1".to_string(),
            ));
        }
        if base_step.checked_mul(max_step_multiplier).is_none() {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive SLA-margin step range exceeds usize".to_string(),
            ));
        }
        Ok(Self {
            base_step,
            max_step_multiplier,
        })
    }
}

impl StepPolicy for SlaMarginStep {
    fn step_size(
        &self,
        _current: f64,
        filters: &[SlaFilter],
        observed: Option<&SlaValues>,
        evaluator: &dyn SlaEvaluator,
    ) -> f64 {
        let Some(observed) = observed else {
            return self.base_step as f64;
        };
        let margins: Vec<f64> = filters
            .iter()
            .filter_map(|filter| {
                let key = evaluator.key(filter);
                evaluator.margin(filter, observed.get(&key).copied())
            })
            .collect();
        let Some(tightest) = margins.into_iter().min_by(f64::total_cmp) else {
            return self.base_step as f64;
        };
        let effective_margin = tightest.max(0.0);
        let raw_multiplier = if effective_margin.is_finite() {
            (effective_margin * self.max_step_multiplier as f64).floor() as usize
        } else {
            self.max_step_multiplier
        };
        let multiplier = raw_multiplier.clamp(1, self.max_step_multiplier);
        (self.base_step * multiplier) as f64
    }

    fn snapshot(&self) -> StepPolicySnapshot {
        StepPolicySnapshot {
            name: "sla_margin",
            base_step: Some(self.base_step),
            max_step_multiplier: Some(self.max_step_multiplier),
            step_percent: None,
        }
    }
}

/// Fixed percentage of current-control step policy.
pub struct FixedPercentStep {
    percent: f64,
}

impl FixedPercentStep {
    /// Build a fixed-percent policy with a positive finite percentage.
    pub fn new(percent: f64) -> Result<Self, AdaptiveError> {
        if !percent.is_finite() || percent <= 0.0 {
            return Err(AdaptiveError::InvalidConfig(format!(
                "adaptive fixed step percent must be positive and finite, got {percent}"
            )));
        }
        Ok(Self { percent })
    }
}

impl StepPolicy for FixedPercentStep {
    fn step_size(
        &self,
        current: f64,
        _filters: &[SlaFilter],
        _observed: Option<&SlaValues>,
        _evaluator: &dyn SlaEvaluator,
    ) -> f64 {
        (current * self.percent / 100.0).ceil().max(1.0)
    }

    fn snapshot(&self) -> StepPolicySnapshot {
        StepPolicySnapshot {
            name: "fixed_percent_step",
            base_step: None,
            max_step_multiplier: None,
            step_percent: Some(self.percent),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_core::sla::{DefaultSlaEvaluator, SlaOp, SlaStat};

    #[test]
    fn margin_policy_uses_the_tightest_filter_and_clamps_multiplier() {
        let evaluator = DefaultSlaEvaluator;
        let latency = SlaFilter::new("request_latency", SlaStat::P95, SlaOp::Le, 100.0).unwrap();
        let throughput = SlaFilter::new("throughput", SlaStat::Avg, SlaOp::Ge, 1000.0).unwrap();
        let filters = vec![latency.clone(), throughput.clone()];
        let mut values = SlaValues::new();
        values.insert(evaluator.key(&latency), 10.0);
        values.insert(evaluator.key(&throughput), 1100.0);
        let policy = SlaMarginStep::new(10, 4).unwrap();
        assert_eq!(
            policy.step_size(100.0, &filters, Some(&values), &evaluator),
            10.0
        );

        values.insert(evaluator.key(&throughput), 5000.0);
        assert_eq!(
            policy.step_size(100.0, &filters, Some(&values), &evaluator),
            30.0,
            "latency margin 0.9 binds: floor(0.9 * 4) = 3"
        );
    }

    #[test]
    fn fixed_percent_uses_ceiling_and_a_one_unit_floor() {
        let policy = FixedPercentStep::new(25.0).unwrap();
        let evaluator = DefaultSlaEvaluator;
        assert_eq!(policy.step_size(2.0, &[], None, &evaluator), 1.0);
        assert_eq!(policy.step_size(10.0, &[], None, &evaluator), 3.0);
    }

    #[test]
    fn margin_policy_rejects_an_overflowing_step_range() {
        assert!(SlaMarginStep::new(usize::MAX, 2).is_err());
    }
}
