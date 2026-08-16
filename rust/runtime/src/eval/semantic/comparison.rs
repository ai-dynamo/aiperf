// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable baseline-locking contracts for paired evaluation comparisons.

/// One finite, independently attributable result from a trial variant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedMeasurements {
    quality: f64,
    cost: f64,
    latency_seconds: f64,
    critical_path_seconds: f64,
    tokens: u64,
    tool_calls: u64,
}

impl PairedMeasurements {
    /// Creates finite measurements for one trial variant.
    pub fn new(
        quality: f64,
        cost: f64,
        latency_seconds: f64,
        critical_path_seconds: f64,
        tokens: u64,
        tool_calls: u64,
    ) -> Result<Self, PairedComparisonError> {
        for (name, value) in [
            ("quality", quality),
            ("cost", cost),
            ("latency_seconds", latency_seconds),
            ("critical_path_seconds", critical_path_seconds),
        ] {
            if !value.is_finite() {
                return Err(PairedComparisonError::NonFiniteMeasurement(name));
            }
        }
        for (name, value) in [
            ("cost", cost),
            ("latency_seconds", latency_seconds),
            ("critical_path_seconds", critical_path_seconds),
        ] {
            if value < 0.0 {
                return Err(PairedComparisonError::NegativeMeasurement(name));
            }
        }
        Ok(Self {
            quality,
            cost,
            latency_seconds,
            critical_path_seconds,
            tokens,
            tool_calls,
        })
    }
}

/// Independent candidate-minus-baseline deltas for one paired experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedComparisonReport {
    quality_delta: f64,
    cost_delta: f64,
    latency_seconds_delta: f64,
    critical_path_seconds_delta: f64,
    token_delta: i128,
    tool_call_delta: i128,
}

impl PairedComparisonReport {
    /// Returns the candidate-minus-baseline quality delta.
    pub const fn quality_delta(&self) -> f64 {
        self.quality_delta
    }

    /// Returns the candidate-minus-baseline cost delta.
    pub const fn cost_delta(&self) -> f64 {
        self.cost_delta
    }

    /// Returns the candidate-minus-baseline latency delta in seconds.
    pub const fn latency_seconds_delta(&self) -> f64 {
        self.latency_seconds_delta
    }

    /// Returns the candidate-minus-baseline critical-path delta in seconds.
    pub const fn critical_path_seconds_delta(&self) -> f64 {
        self.critical_path_seconds_delta
    }

    /// Returns the candidate-minus-baseline generated-token delta.
    pub const fn token_delta(&self) -> i128 {
        self.token_delta
    }

    /// Returns the candidate-minus-baseline tool-call delta.
    pub const fn tool_call_delta(&self) -> i128 {
        self.tool_call_delta
    }
}

/// Dimensions that must stay fixed across a paired comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedComparisonSpec {
    task: String,
    model: String,
    seed: u64,
    policy: String,
    image: String,
    budget_seconds: u64,
}

impl PairedComparisonSpec {
    /// Creates a fully pinned paired-comparison baseline.
    pub fn new(
        task: impl Into<String>,
        model: impl Into<String>,
        seed: u64,
        policy: impl Into<String>,
        image: impl Into<String>,
        budget_seconds: u64,
    ) -> Result<Self, PairedComparisonError> {
        let task = task.into();
        let model = model.into();
        let policy = policy.into();
        let image = image.into();
        if task.trim().is_empty()
            || model.trim().is_empty()
            || policy.trim().is_empty()
            || image.trim().is_empty()
            || budget_seconds == 0
        {
            return Err(PairedComparisonError::InvalidBaseline);
        }
        Ok(Self {
            task,
            model,
            seed,
            policy,
            image,
            budget_seconds,
        })
    }

    /// Accepts only a comparison that preserves every baseline dimension.
    pub fn compare_to(&self, other: &Self) -> Result<(), PairedComparisonError> {
        if self == other {
            Ok(())
        } else {
            Err(PairedComparisonError::ChangedBaseline)
        }
    }

    /// Produces independent deltas only after every fixed baseline dimension matches.
    pub fn compare_measurements(
        &self,
        other: &Self,
        baseline: PairedMeasurements,
        candidate: PairedMeasurements,
    ) -> Result<PairedComparisonReport, PairedComparisonError> {
        self.compare_to(other)?;
        Ok(PairedComparisonReport {
            quality_delta: candidate.quality - baseline.quality,
            cost_delta: candidate.cost - baseline.cost,
            latency_seconds_delta: candidate.latency_seconds - baseline.latency_seconds,
            critical_path_seconds_delta: candidate.critical_path_seconds
                - baseline.critical_path_seconds,
            token_delta: i128::from(candidate.tokens) - i128::from(baseline.tokens),
            tool_call_delta: i128::from(candidate.tool_calls) - i128::from(baseline.tool_calls),
        })
    }
}

/// Invalid or incomparable paired-comparison specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairedComparisonError {
    /// A required fixed dimension was absent.
    InvalidBaseline,
    /// The candidate changed a supposedly fixed baseline dimension.
    ChangedBaseline,
    /// A metric was not finite at the serialization boundary.
    NonFiniteMeasurement(&'static str),
    /// A system resource metric was negative.
    NegativeMeasurement(&'static str),
}

impl std::fmt::Display for PairedComparisonError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBaseline => formatter.write_str("paired baseline must be complete"),
            Self::ChangedBaseline => {
                formatter.write_str("paired comparison changed a fixed baseline")
            }
            Self::NonFiniteMeasurement(name) => {
                write!(formatter, "paired measurement {name} must be finite")
            }
            Self::NegativeMeasurement(name) => {
                write!(formatter, "paired measurement {name} must not be negative")
            }
        }
    }
}

impl std::error::Error for PairedComparisonError {}
