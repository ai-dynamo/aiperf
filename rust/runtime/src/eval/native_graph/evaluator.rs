// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Evaluator seams over frozen Harbor verifier facts.

use std::{
    fmt::{self, Display, Formatter},
    rc::Rc,
};

use async_trait::async_trait;

use crate::eval::{
    EpisodeComparability, EpisodeIntegrity, EpisodeResult, EpisodeResultError, FrozenAttemptBundle,
    FrozenAttemptError, ResolvedEpisodeTrial,
};

use super::NativeGraphCompletedAttempt;

/// Evaluates one frozen episode attempt through an independent scoring authority.
#[async_trait(?Send)]
pub trait EpisodeEvaluator {
    /// Maps one frozen attempt to orthogonal episode result facts.
    async fn evaluate(
        &self,
        attempt: FrozenAttemptBundle,
    ) -> Result<EpisodeResult, EpisodeEvaluationError>;

    /// Maps a sealed NativeGraph completion attempt to orthogonal episode result facts.
    ///
    /// Legacy implementations receive the already-frozen Harbor facts unchanged. The built-in
    /// evaluator overrides this seam to preserve rollout terminality.
    async fn evaluate_native_graph(
        &self,
        attempt: NativeGraphCompletedAttempt,
    ) -> Result<EpisodeResult, EpisodeEvaluationError> {
        if attempt.has_rollout() || attempt.has_compatibility() {
            return Err(EpisodeEvaluationError::RolloutAwareEvaluatorRequired);
        }
        self.evaluate(attempt.into_frozen_attempt()).await
    }
}

/// Creates a selected evaluator after package and runtime capabilities have frozen.
pub trait EpisodeEvaluatorFactory: Send + Sync {
    /// Creates an evaluator for one resolved immutable trial.
    fn create(
        &self,
        trial: &ResolvedEpisodeTrial,
    ) -> Result<Rc<dyn EpisodeEvaluator>, EpisodeEvaluationError>;
}

/// Built-in evaluator that maps existing Harbor verifier and regrade facts to episode axes.
#[derive(Clone, Copy, Debug, Default)]
pub struct HarborEpisodeEvaluator;

impl HarborEpisodeEvaluator {
    /// Creates the evaluator over Harbor's existing verifier and score authority.
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait(?Send)]
impl EpisodeEvaluator for HarborEpisodeEvaluator {
    async fn evaluate(
        &self,
        attempt: FrozenAttemptBundle,
    ) -> Result<EpisodeResult, EpisodeEvaluationError> {
        self.evaluate_native_graph(NativeGraphCompletedAttempt::from_frozen(attempt))
            .await
    }

    async fn evaluate_native_graph(
        &self,
        attempt: NativeGraphCompletedAttempt,
    ) -> Result<EpisodeResult, EpisodeEvaluationError> {
        let frozen = attempt.frozen_attempt();
        let score = frozen
            .selected_score()
            .ok_or(EpisodeEvaluationError::Frozen(
                FrozenAttemptError::EmptyScoreLineage,
            ))?;
        let result = EpisodeResult::new_with_fidelity(
            frozen.trial_digest().clone(),
            frozen.attempt().clone(),
            EpisodeIntegrity::Valid,
            attempt.execution(),
            crate::eval::EpisodeScoreState::Verified {
                reward: score.value,
            },
            EpisodeComparability::Scored,
            vec![frozen.identity_digest()],
            attempt.fidelity(),
        )?;
        Ok(result)
    }
}

/// Factory for the built-in evaluator over existing Harbor scoring facts.
#[derive(Clone, Copy, Debug, Default)]
pub struct HarborEpisodeEvaluatorFactory;

impl EpisodeEvaluatorFactory for HarborEpisodeEvaluatorFactory {
    fn create(
        &self,
        _: &ResolvedEpisodeTrial,
    ) -> Result<Rc<dyn EpisodeEvaluator>, EpisodeEvaluationError> {
        Ok(Rc::new(HarborEpisodeEvaluator))
    }
}

/// Failure while freezing or classifying an independently scored episode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeEvaluationError {
    /// The selected evaluator did not opt in to preserving sealed rollout terminality.
    RolloutAwareEvaluatorRequired,
    /// The incoming attempt did not preserve immutable evidence or score lineage.
    Frozen(FrozenAttemptError),
    /// The result contract rejected the verifier reward.
    Result(EpisodeResultError),
}

impl Display for EpisodeEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RolloutAwareEvaluatorRequired => {
                formatter.write_str("selected evaluator does not support sealed rollout evidence")
            }
            Self::Frozen(error) => error.fmt(formatter),
            Self::Result(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EpisodeEvaluationError {}

impl From<FrozenAttemptError> for EpisodeEvaluationError {
    fn from(error: FrozenAttemptError) -> Self {
        Self::Frozen(error)
    }
}

impl From<EpisodeResultError> for EpisodeEvaluationError {
    fn from(error: EpisodeResultError) -> Self {
        Self::Result(error)
    }
}
