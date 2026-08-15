// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable baseline-locking contracts for paired evaluation comparisons.

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
}

/// Invalid or incomparable paired-comparison specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairedComparisonError {
    /// A required fixed dimension was absent.
    InvalidBaseline,
    /// The candidate changed a supposedly fixed baseline dimension.
    ChangedBaseline,
}

impl std::fmt::Display for PairedComparisonError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBaseline => formatter.write_str("paired baseline must be complete"),
            Self::ChangedBaseline => {
                formatter.write_str("paired comparison changed a fixed baseline")
            }
        }
    }
}

impl std::error::Error for PairedComparisonError {}
