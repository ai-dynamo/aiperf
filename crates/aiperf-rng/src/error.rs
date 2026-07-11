// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the hash-derived RNG crate.

use std::fmt;

/// Result alias for `aiperf-rng` operations.
pub type Result<T> = std::result::Result<T, RngError>;

/// Validation and construction errors returned by RNG helpers.
#[derive(Clone, Debug, PartialEq)]
pub enum RngError {
    /// A range was empty or otherwise impossible to sample.
    EmptyRange { what: &'static str },
    /// A sequence argument was empty.
    EmptySequence { what: &'static str },
    /// A requested sample size is larger than the population.
    SampleTooLarge { k: usize, len: usize },
    /// A numeric argument violated a required bound.
    InvalidParameter { what: &'static str, value: f64 },
    /// Lower and upper bounds are contradictory.
    InvalidBounds { lower: f64, upper: f64 },
    /// Weights were missing, negative, non-finite, or summed to zero.
    InvalidWeights { reason: &'static str },
    /// Probabilities do not sum to the required total.
    InvalidProbabilitySum { total: f64 },
}

impl fmt::Display for RngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRange { what } => write!(f, "empty range for {what}"),
            Self::EmptySequence { what } => write!(f, "empty sequence for {what}"),
            Self::SampleTooLarge { k, len } => {
                write!(f, "sample size {k} exceeds population length {len}")
            }
            Self::InvalidParameter { what, value } => {
                write!(f, "invalid parameter {what}={value}")
            }
            Self::InvalidBounds { lower, upper } => {
                write!(f, "invalid bounds: lower ({lower}) > upper ({upper})")
            }
            Self::InvalidWeights { reason } => write!(f, "invalid weights: {reason}"),
            Self::InvalidProbabilitySum { total } => {
                write!(f, "probabilities must sum to 100.0, got {total}")
            }
        }
    }
}

impl std::error::Error for RngError {}
