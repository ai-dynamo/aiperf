// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive-scale error type.

use std::fmt::{Display, Formatter};

/// Error returned by adaptive configuration, evaluation, actuation, or artifact
/// emission.
#[derive(Debug)]
pub enum AdaptiveError {
    /// A user-supplied adaptive configuration is invalid.
    InvalidConfig(String),
    /// An SLA window could not be evaluated.
    Evaluation(String),
    /// A control actuator rejected an update.
    Actuator(String),
    /// An adaptive artifact could not be written.
    Artifact(String),
}

impl Display for AdaptiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid adaptive configuration: {message}"),
            Self::Evaluation(message) => write!(f, "adaptive SLA evaluation failed: {message}"),
            Self::Actuator(message) => write!(f, "adaptive actuator failed: {message}"),
            Self::Artifact(message) => write!(f, "adaptive artifact write failed: {message}"),
        }
    }
}

impl std::error::Error for AdaptiveError {}

impl From<std::io::Error> for AdaptiveError {
    fn from(error: std::io::Error) -> Self {
        Self::Artifact(error.to_string())
    }
}

impl From<serde_json::Error> for AdaptiveError {
    fn from(error: serde_json::Error) -> Self {
        Self::Artifact(error.to_string())
    }
}
