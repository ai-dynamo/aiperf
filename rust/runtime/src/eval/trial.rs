// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Resolved immutable evaluation trial identity.

use serde::{Deserialize, Serialize};

use super::{
    AgentVariantRef, ArtifactDigest, EvalTaskRef, ModelIdentity, PolicyIdentity, RuntimeIdentity,
};

/// Positive finite execution budgets resolved into a trial.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrialBudget {
    /// Maximum agent execution seconds.
    pub execution_seconds: f64,
    /// Maximum verifier execution seconds.
    pub verifier_seconds: f64,
}

impl TrialBudget {
    /// Creates a finite positive budget.
    pub fn new(execution_seconds: f64, verifier_seconds: f64) -> Result<Self, TrialIdentityError> {
        if !execution_seconds.is_finite() || execution_seconds <= 0.0 {
            return Err(TrialIdentityError::InvalidBudget("execution_seconds"));
        }
        if !verifier_seconds.is_finite() || verifier_seconds <= 0.0 {
            return Err(TrialIdentityError::InvalidBudget("verifier_seconds"));
        }
        Ok(Self {
            execution_seconds,
            verifier_seconds,
        })
    }
}

/// Fully resolved trial inputs that define reproducible evaluation identity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrialSpec {
    /// Content-addressed task.
    pub task: EvalTaskRef,
    /// Selected agent or graph variant.
    pub agent: AgentVariantRef,
    /// Provider and model.
    pub model: ModelIdentity,
    /// Deterministic trial seed.
    pub seed: u64,
    /// Policy snapshot.
    pub policy: PolicyIdentity,
    /// Execution and verifier limits.
    pub budget: TrialBudget,
    /// Environment recipe identity.
    pub environment: ArtifactDigest,
    /// Verifier identity.
    pub verifier: ArtifactDigest,
    /// Native runtime identity.
    pub runtime: RuntimeIdentity,
}

impl TrialSpec {
    /// Builds one resolved immutable trial.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: EvalTaskRef,
        agent: AgentVariantRef,
        model: ModelIdentity,
        seed: u64,
        policy: PolicyIdentity,
        budget: TrialBudget,
        environment: ArtifactDigest,
        verifier: ArtifactDigest,
        runtime: RuntimeIdentity,
    ) -> Result<Self, TrialIdentityError> {
        if !budget.execution_seconds.is_finite() || !budget.verifier_seconds.is_finite() {
            return Err(TrialIdentityError::InvalidBudget("budget"));
        }
        Ok(Self {
            task,
            agent,
            model,
            seed,
            policy,
            budget,
            environment,
            verifier,
            runtime,
        })
    }

    /// Computes the canonical BLAKE3 identity for this fully resolved trial.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let bytes = canonical_trial_bytes(self);
        ArtifactDigest::from_bytes(&bytes)
    }
}

fn canonical_trial_bytes(trial: &TrialSpec) -> Vec<u8> {
    format!(
        "task={}\u{1f}task_digest={}\u{1f}agent={:?}\u{1f}provider={}\u{1f}model={}\u{1f}seed={}\u{1f}policy={:?}\u{1f}execution_seconds={}\u{1f}verifier_seconds={}\u{1f}environment={}\u{1f}verifier={}\u{1f}runtime={:?}",
        trial.task.id.as_str(),
        trial.task.digest.as_str(),
        trial.agent.as_str(),
        trial.model.provider,
        trial.model.model,
        trial.seed,
        trial.policy.digest().as_str(),
        trial.budget.execution_seconds.to_bits(),
        trial.budget.verifier_seconds.to_bits(),
        trial.environment.as_str(),
        trial.verifier.as_str(),
        trial.runtime.as_str(),
    )
    .into_bytes()
}

/// Failed resolved-trial validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrialIdentityError {
    /// A budget value was non-finite or nonpositive.
    InvalidBudget(&'static str),
}

impl std::fmt::Display for TrialIdentityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget(field) => write!(formatter, "{field} must be finite and positive"),
        }
    }
}

impl std::error::Error for TrialIdentityError {}
