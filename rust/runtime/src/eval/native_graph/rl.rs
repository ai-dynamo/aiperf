// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable authoritative RL rollout facts.

use std::fmt;

use super::FrozenArtifact;

const DEFAULT_MAX_ENVIRONMENT_BYTES: usize = 4 * 1024;
const DEFAULT_MAX_HORIZON: u32 = 1_024;

/// Immutable resource limits selected before retaining one rollout trajectory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RlEvaluationLimits {
    max_environment_bytes: usize,
    max_horizon: u32,
}

impl RlEvaluationLimits {
    /// Creates positive bounds for one trusted environment policy and its trajectory.
    pub fn new(max_environment_bytes: usize, max_horizon: u32) -> Result<Self, RlRolloutError> {
        if max_environment_bytes == 0 || max_horizon == 0 {
            return Err(RlRolloutError::InvalidLimits);
        }
        Ok(Self {
            max_environment_bytes,
            max_horizon,
        })
    }

    /// Returns the maximum bytes admitted for the environment identity.
    pub const fn max_environment_bytes(&self) -> usize {
        self.max_environment_bytes
    }

    /// Returns the maximum retained rollout horizon.
    pub const fn max_horizon(&self) -> u32 {
        self.max_horizon
    }
}

impl Default for RlEvaluationLimits {
    fn default() -> Self {
        Self {
            max_environment_bytes: DEFAULT_MAX_ENVIRONMENT_BYTES,
            max_horizon: DEFAULT_MAX_HORIZON,
        }
    }
}

/// Immutable evaluation policy for one rollout.
#[derive(Clone, Debug, PartialEq)]
pub struct RlEvaluationPolicy {
    environment: String,
    horizon: u32,
    gamma: f64,
    limits: RlEvaluationLimits,
}
impl RlEvaluationPolicy {
    /// Validate one finite rollout policy.
    pub fn new(
        environment: impl AsRef<str>,
        horizon: u32,
        gamma: f64,
    ) -> Result<Self, RlRolloutError> {
        Self::new_with_limits(environment, horizon, gamma, RlEvaluationLimits::default())
    }

    /// Validates a policy against selected limits before allocating its environment identity.
    pub fn new_with_limits(
        environment: impl AsRef<str>,
        horizon: u32,
        gamma: f64,
        limits: RlEvaluationLimits,
    ) -> Result<Self, RlRolloutError> {
        let environment = environment.as_ref();
        if environment.len() > limits.max_environment_bytes {
            return Err(RlRolloutError::EnvironmentTooLong {
                actual: environment.len(),
                limit: limits.max_environment_bytes,
            });
        }
        if horizon == 0 {
            return Err(RlRolloutError::InvalidHorizon);
        }
        if horizon > limits.max_horizon {
            return Err(RlRolloutError::HorizonLimitExceeded {
                requested: horizon,
                limit: limits.max_horizon,
            });
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(RlRolloutError::InvalidGamma);
        }
        Ok(Self {
            environment: environment.to_owned(),
            horizon,
            gamma,
            limits,
        })
    }

    /// Returns the immutable maximum number of environment transitions.
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Borrows the trusted environment identity selected for the rollout.
    pub fn environment(&self) -> &str {
        &self.environment
    }

    /// Returns the finite discount factor used for derived returns.
    pub const fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Validate and derive one immutable trajectory.
    pub fn trajectory(
        &self,
        transitions: impl IntoIterator<Item = EnvironmentTransitionRecord>,
    ) -> Result<FrozenRolloutTrajectory, RlRolloutError> {
        let capacity =
            usize::try_from(self.horizon).map_err(|_| RlRolloutError::HorizonLimitExceeded {
                requested: self.horizon,
                limit: self.limits.max_horizon,
            })?;
        let mut retained = Vec::with_capacity(capacity);
        for transition in transitions {
            if retained.len() >= capacity {
                return Err(RlRolloutError::HorizonExceeded);
            }
            retained.push(transition);
        }
        let transitions = retained;
        if transitions
            .iter()
            .enumerate()
            .any(|(i, t)| t.step != i as u32)
        {
            return Err(RlRolloutError::InvalidStepOrder);
        }
        if transitions
            .iter()
            .enumerate()
            .any(|(i, t)| (t.terminated || t.truncated) && i + 1 != transitions.len())
        {
            return Err(RlRolloutError::PostTerminalStep);
        }
        match transitions.last() {
            Some(transition) if transition.terminated || transition.truncated => {}
            _ => return Err(RlRolloutError::MissingTerminal),
        }
        let mut undiscounted = 0.0;
        let mut discounted = 0.0;
        let mut discount_factor = 1.0;
        for transition in &transitions {
            undiscounted += transition.reward;
            let weighted_reward = discount_factor * transition.reward;
            discounted += weighted_reward;
            if !undiscounted.is_finite() || !weighted_reward.is_finite() || !discounted.is_finite()
            {
                return Err(RlRolloutError::NonFiniteReturn);
            }
            discount_factor *= self.gamma;
            if !discount_factor.is_finite() {
                return Err(RlRolloutError::NonFiniteReturn);
            }
        }
        Ok(FrozenRolloutTrajectory {
            policy: self.clone(),
            transitions,
            undiscounted,
            discounted,
        })
    }

    pub(crate) const fn limits(&self) -> RlEvaluationLimits {
        self.limits
    }
}
/// One authoritative environment transition.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentTransitionRecord {
    step: u32,
    observation: FrozenArtifact,
    reward: f64,
    terminated: bool,
    truncated: bool,
    info: FrozenArtifact,
    workspace_patch: FrozenArtifact,
}
impl EnvironmentTransitionRecord {
    /// Creates one environment-authoritative transition with immutable frozen evidence.
    pub fn new(
        step: u32,
        observation: FrozenArtifact,
        reward: f64,
        terminated: bool,
        truncated: bool,
        info: FrozenArtifact,
        workspace_patch: FrozenArtifact,
    ) -> Result<Self, RlRolloutError> {
        if !reward.is_finite() {
            return Err(RlRolloutError::NonFiniteReward);
        }
        if terminated && truncated {
            return Err(RlRolloutError::AmbiguousTerminal);
        }
        Ok(Self {
            step,
            observation,
            reward,
            terminated,
            truncated,
            info,
            workspace_patch,
        })
    }

    /// Returns whether the retained transition ended through truncation.
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Returns the zero-based environment transition index.
    pub const fn step(&self) -> u32 {
        self.step
    }

    /// Returns the finite environment-authoritative reward.
    pub const fn reward(&self) -> f64 {
        self.reward
    }

    /// Returns whether the retained transition reached an environment terminal state.
    pub const fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Borrows the frozen observation descriptor for this transition.
    pub fn observation(&self) -> &FrozenArtifact {
        &self.observation
    }

    /// Borrows the frozen diagnostic descriptor for this transition.
    pub fn info(&self) -> &FrozenArtifact {
        &self.info
    }

    /// Borrows the sealed workspace-patch archive descriptor for this transition.
    pub fn workspace_patch(&self) -> &FrozenArtifact {
        &self.workspace_patch
    }
}
/// Frozen derived return facts.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenRolloutTrajectory {
    policy: RlEvaluationPolicy,
    transitions: Vec<EnvironmentTransitionRecord>,
    undiscounted: f64,
    discounted: f64,
}
impl FrozenRolloutTrajectory {
    /// Borrows the immutable policy that Rust used to validate this trajectory.
    pub fn policy(&self) -> &RlEvaluationPolicy {
        &self.policy
    }

    /// Returns the selected limits that admitted this trajectory before retention.
    pub(crate) const fn limits(&self) -> RlEvaluationLimits {
        self.policy.limits()
    }

    /// Borrows the authoritative, validated transition stream in step order.
    pub fn transitions(&self) -> &[EnvironmentTransitionRecord] {
        &self.transitions
    }

    /// Returns the authoritative undiscounted return.
    pub fn undiscounted_return(&self) -> f64 {
        self.undiscounted
    }

    /// Returns the authoritative discounted return.
    pub fn discounted_return(&self) -> f64 {
        self.discounted
    }
}
/// Typed rollout validation refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlRolloutError {
    InvalidLimits,
    EnvironmentTooLong { actual: usize, limit: usize },
    HorizonLimitExceeded { requested: u32, limit: u32 },
    NonFiniteReward,
    AmbiguousTerminal,
    InvalidHorizon,
    InvalidGamma,
    HorizonExceeded,
    InvalidStepOrder,
    PostTerminalStep,
    MissingTerminal,
    NonFiniteReturn,
}
impl fmt::Display for RlRolloutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RlRolloutError {}
