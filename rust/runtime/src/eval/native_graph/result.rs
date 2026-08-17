// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Orthogonal, immutable outcome facts for one NativeGraph episode attempt.

use std::{borrow::Borrow, fmt};

use crate::eval::{ArtifactDigest, AttemptId};

/// Integrity classification for one episode attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeIntegrity {
    /// Rust established complete authoritative provider, artifact, and evidence facts.
    Valid,
    /// The provider did not supply a valid benchmark response.
    InvalidProvider,
    /// Native execution could not establish a trustworthy runtime result.
    InvalidRuntime,
    /// Required authoritative evidence was absent or inconsistent.
    InvalidEvidence,
}

/// Terminal execution state independent of episode integrity and score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeExecution {
    /// The graph reached its declared terminal output.
    Completed,
    /// The graph reached a terminal failure outcome.
    Failed,
    /// Native lifecycle policy stopped the graph at a declared truncation boundary.
    Truncated,
    /// Native lifecycle policy cancelled the graph before a terminal output.
    Cancelled,
}

/// Verifier score availability independent of execution state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpisodeScoreState {
    /// An independent evaluator produced a finite reward.
    Verified {
        /// Verifier-supplied reward for the episode.
        reward: f64,
    },
    /// No independent evaluator score is available for this episode.
    Unavailable,
}

/// Whether this result may participate in exact like-for-like comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeComparability {
    /// The episode belongs in the exact reward-comparison denominator.
    Scored,
    /// The episode is retained as evidence but excluded from reward aggregation.
    Unscored,
}

/// Immutable result facts emitted by one scheduled episode attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeResult {
    trial_digest: ArtifactDigest,
    attempt_id: AttemptId,
    integrity: EpisodeIntegrity,
    execution: EpisodeExecution,
    score: EpisodeScoreState,
    comparability: EpisodeComparability,
    evidence: Vec<ArtifactDigest>,
}

impl EpisodeResult {
    /// Creates one result without conflating integrity, execution, and scoring facts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trial_digest: ArtifactDigest,
        attempt_id: AttemptId,
        integrity: EpisodeIntegrity,
        execution: EpisodeExecution,
        score: EpisodeScoreState,
        comparability: EpisodeComparability,
        evidence: Vec<ArtifactDigest>,
    ) -> Result<Self, EpisodeResultError> {
        if let EpisodeScoreState::Verified { reward } = score
            && !reward.is_finite()
        {
            return Err(EpisodeResultError::NonFiniteReward);
        }
        Ok(Self {
            trial_digest,
            attempt_id,
            integrity,
            execution,
            score,
            comparability,
            evidence,
        })
    }

    /// Borrows the immutable resolved-trial identity.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        &self.trial_digest
    }

    /// Borrows the append-only attempt identity.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }

    /// Returns the independent integrity classification.
    pub const fn integrity(&self) -> EpisodeIntegrity {
        self.integrity
    }

    /// Returns the terminal execution classification.
    pub const fn execution(&self) -> EpisodeExecution {
        self.execution
    }

    /// Returns the verifier score state.
    pub const fn score(&self) -> EpisodeScoreState {
        self.score
    }

    /// Returns whether this result belongs in the reward-comparison denominator.
    pub const fn comparability(&self) -> EpisodeComparability {
        self.comparability
    }

    /// Borrows the immutable evidence identities carried by this result.
    pub fn evidence(&self) -> &[ArtifactDigest] {
        &self.evidence
    }

    /// Returns the verified reward, when one exists.
    pub const fn verified_reward(&self) -> Option<f64> {
        match self.score {
            EpisodeScoreState::Verified { reward } => Some(reward),
            EpisodeScoreState::Unavailable => None,
        }
    }
}

/// Bounded aggregate facts over independently retained episode results.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeAggregate {
    valid_attempts: usize,
    invalid_attempts: usize,
    scored_valid_attempts: usize,
    unscored_valid_attempts: usize,
    mean_reward: Option<f64>,
}

impl EpisodeAggregate {
    /// Returns attempts with valid authoritative episode facts.
    pub const fn valid_attempts(&self) -> usize {
        self.valid_attempts
    }

    /// Returns attempts excluded because their authoritative facts were invalid.
    pub const fn invalid_attempts(&self) -> usize {
        self.invalid_attempts
    }

    /// Returns valid attempts with an independent finite reward.
    pub const fn scored_valid_attempts(&self) -> usize {
        self.scored_valid_attempts
    }

    /// Returns valid attempts awaiting an independent reward.
    pub const fn unscored_valid_attempts(&self) -> usize {
        self.unscored_valid_attempts
    }

    /// Returns the mean over valid scored attempts, including failed zero-score episodes.
    pub const fn mean_reward(&self) -> Option<f64> {
        self.mean_reward
    }
}

/// Aggregates valid independently scored attempts without discarding execution failures.
pub fn aggregate_episode_results<I, R>(results: I) -> Result<EpisodeAggregate, EpisodeResultError>
where
    I: IntoIterator<Item = R>,
    R: Borrow<EpisodeResult>,
{
    let mut valid_attempts = 0usize;
    let mut invalid_attempts = 0usize;
    let mut scored_valid_attempts = 0usize;
    let mut unscored_valid_attempts = 0usize;
    let mut reward_total = 0.0f64;

    for result in results {
        let result = result.borrow();
        if result.integrity != EpisodeIntegrity::Valid {
            invalid_attempts += 1;
            continue;
        }
        valid_attempts += 1;
        match (result.comparability, result.verified_reward()) {
            (EpisodeComparability::Scored, Some(reward)) => {
                reward_total += reward;
                if !reward_total.is_finite() {
                    return Err(EpisodeResultError::RewardOverflow);
                }
                scored_valid_attempts += 1;
            }
            _ => unscored_valid_attempts += 1,
        }
    }

    let mean_reward = if scored_valid_attempts == 0 {
        None
    } else {
        Some(reward_total / scored_valid_attempts as f64)
    };
    Ok(EpisodeAggregate {
        valid_attempts,
        invalid_attempts,
        scored_valid_attempts,
        unscored_valid_attempts,
        mean_reward,
    })
}

/// Failed NativeGraph result construction or aggregation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeResultError {
    /// A verifier reward was non-finite.
    NonFiniteReward,
    /// Aggregating otherwise finite rewards overflowed.
    RewardOverflow,
}

impl fmt::Display for EpisodeResultError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteReward => formatter.write_str("episode reward must be finite"),
            Self::RewardOverflow => formatter.write_str("episode reward aggregate overflowed"),
        }
    }
}

impl std::error::Error for EpisodeResultError {}
