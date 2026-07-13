// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-phase lifecycle configuration.
//!
//! Workload-specific arrival and dataset
//! knobs remain behind the injected execution strategy rather than becoming a
//! string/enum switch in this policy layer.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::timing::StopConfig;

/// Default interval between phase-owned progress observations.
pub const DEFAULT_PROGRESS_INTERVAL_NS: i64 = 2_000_000_000;

/// Sentinel `progress_interval_ns` that disables periodic progress observation
/// entirely: the phase emits only its opening and terminal snapshots and
/// schedules no intermediate progress clock event. Required for the offline
/// `execute_pass` single engine, which cannot stop at a finite clock deadline.
pub const DISABLED_PROGRESS_INTERVAL_NS: i64 = i64::MAX;

/// Default bounded wait for cancelled requests to return.
pub const DEFAULT_CANCEL_DRAIN_TIMEOUT_NS: i64 = 10_000_000_000;

/// The semantic role of a benchmark phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    /// Setup traffic excluded from profiling results and cancellation sampling.
    Warmup,
    /// Measured benchmark traffic.
    Profiling,
}

/// Return-wait policy after the sending deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "duration_ns")]
pub enum GracePeriod {
    /// Add no grace beyond the configured phase duration.
    Disabled,
    /// Add this many clock nanoseconds beyond the phase duration.
    Finite(i64),
    /// Wait without a deadline for every in-flight request to return.
    Infinite,
}

/// Validated policy for one bounded issuance phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseConfig {
    /// Stable phase identifier used by progress and report consumers.
    pub id: String,
    /// Warmup or profiling semantics.
    pub kind: PhaseKind,
    /// Request, session, and duration stop bounds.
    pub stop: StopConfig,
    /// Return-wait policy after sending completes.
    pub grace_period: GracePeriod,
    /// Whether a non-final phase hands off after sending instead of returns.
    pub seamless: bool,
    /// Optional shared session-concurrency target for this phase.
    pub concurrency: Option<usize>,
    /// Optional shared prefill-concurrency target for this phase.
    pub prefill_concurrency: Option<usize>,
    /// Clock nanoseconds between progress observations.
    pub progress_interval_ns: i64,
    /// Clock nanoseconds allowed for cancelled requests to drain.
    pub cancel_drain_timeout_ns: i64,
}

impl PhaseConfig {
    /// Create a phase with source-faithful kind defaults.
    ///
    /// Warmup waits indefinitely for returns; profiling has no extra grace.
    pub fn new(id: impl Into<String>, kind: PhaseKind, stop: StopConfig) -> Self {
        Self {
            id: id.into(),
            kind,
            stop,
            grace_period: match kind {
                PhaseKind::Warmup => GracePeriod::Infinite,
                PhaseKind::Profiling => GracePeriod::Disabled,
            },
            seamless: false,
            concurrency: None,
            prefill_concurrency: None,
            progress_interval_ns: DEFAULT_PROGRESS_INTERVAL_NS,
            cancel_drain_timeout_ns: DEFAULT_CANCEL_DRAIN_TIMEOUT_NS,
        }
    }

    /// Override the return-wait policy.
    pub fn with_grace_period(mut self, grace_period: GracePeriod) -> Self {
        self.grace_period = grace_period;
        self
    }

    /// Enable or disable seamless handoff for this phase.
    pub fn with_seamless(mut self, seamless: bool) -> Self {
        self.seamless = seamless;
        self
    }

    /// Set session and prefill concurrency targets.
    pub fn with_concurrency(
        mut self,
        concurrency: Option<usize>,
        prefill_concurrency: Option<usize>,
    ) -> Self {
        self.concurrency = concurrency;
        self.prefill_concurrency = prefill_concurrency;
        self
    }

    /// Override progress and cancellation-drain timing.
    pub fn with_runtime_intervals(
        mut self,
        progress_interval_ns: i64,
        cancel_drain_timeout_ns: i64,
    ) -> Self {
        self.progress_interval_ns = progress_interval_ns;
        self.cancel_drain_timeout_ns = cancel_drain_timeout_ns;
        self
    }

    /// Validate every local phase invariant before execution begins.
    pub fn validate(&self) -> Result<(), PhaseConfigError> {
        if self.id.trim().is_empty() {
            return Err(PhaseConfigError::EmptyId);
        }
        if self
            .stop
            .expected_duration_ns
            .is_some_and(|value| value <= 0)
        {
            return Err(PhaseConfigError::InvalidDuration(
                self.stop.expected_duration_ns.unwrap_or_default(),
            ));
        }
        if let GracePeriod::Finite(value) = self.grace_period
            && value < 0
        {
            return Err(PhaseConfigError::InvalidGrace(value));
        }
        if self.concurrency == Some(0) {
            return Err(PhaseConfigError::InvalidConcurrency);
        }
        if self.prefill_concurrency == Some(0) {
            return Err(PhaseConfigError::InvalidPrefillConcurrency);
        }
        if self.progress_interval_ns <= 0 {
            return Err(PhaseConfigError::InvalidProgressInterval(
                self.progress_interval_ns,
            ));
        }
        if self.cancel_drain_timeout_ns < 0 {
            return Err(PhaseConfigError::InvalidCancelDrainTimeout(
                self.cancel_drain_timeout_ns,
            ));
        }
        Ok(())
    }
}

/// Invalid phase configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseConfigError {
    /// The stable identifier was empty or whitespace-only.
    EmptyId,
    /// The configured duration was not positive.
    InvalidDuration(i64),
    /// A finite grace duration was negative.
    InvalidGrace(i64),
    /// Session concurrency was zero.
    InvalidConcurrency,
    /// Prefill concurrency was zero.
    InvalidPrefillConcurrency,
    /// The progress interval was not positive.
    InvalidProgressInterval(i64),
    /// The cancellation-drain timeout was negative.
    InvalidCancelDrainTimeout(i64),
}

impl Display for PhaseConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId => write!(f, "phase id cannot be empty"),
            Self::InvalidDuration(value) => {
                write!(f, "phase duration must be positive, got {value}ns")
            }
            Self::InvalidGrace(value) => {
                write!(f, "phase grace period cannot be negative, got {value}ns")
            }
            Self::InvalidConcurrency => write!(f, "phase concurrency must be positive"),
            Self::InvalidPrefillConcurrency => {
                write!(f, "phase prefill concurrency must be positive")
            }
            Self::InvalidProgressInterval(value) => {
                write!(f, "phase progress interval must be positive, got {value}ns")
            }
            Self::InvalidCancelDrainTimeout(value) => write!(
                f,
                "phase cancellation-drain timeout cannot be negative, got {value}ns"
            ),
        }
    }
}

impl Error for PhaseConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_defaults_preserve_warmup_drain_policy() {
        let warmup = PhaseConfig::new("warmup", PhaseKind::Warmup, StopConfig::default());
        let profiling = PhaseConfig::new("profiling", PhaseKind::Profiling, StopConfig::default());

        assert_eq!(warmup.grace_period, GracePeriod::Infinite);
        assert_eq!(profiling.grace_period, GracePeriod::Disabled);
    }

    #[test]
    fn invalid_time_and_concurrency_values_are_rejected() {
        let config = PhaseConfig::new(
            "profiling",
            PhaseKind::Profiling,
            StopConfig {
                expected_duration_ns: Some(0),
                ..StopConfig::default()
            },
        );
        assert_eq!(config.validate(), Err(PhaseConfigError::InvalidDuration(0)));

        let config = PhaseConfig::new("profiling", PhaseKind::Profiling, StopConfig::default())
            .with_concurrency(Some(0), None);
        assert_eq!(config.validate(), Err(PhaseConfigError::InvalidConcurrency));
    }
}
