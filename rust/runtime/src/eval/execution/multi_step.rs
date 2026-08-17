// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral execution sequencing for explicit benchmark steps.

use std::collections::BTreeMap;

use crate::eval::{ArtifactDigest, RewardDocument};

use super::{
    BenchmarkExecutionPlan, BenchmarkStepPlan, EvalExecutionError, MultiStepRewardStrategy,
};

/// Immutable output from one successfully verified benchmark step.
#[derive(Clone, Debug, PartialEq)]
pub struct StepExecutionResult {
    /// Authored name of the completed step.
    pub name: String,
    /// Immutable artifacts captured after the step agent completed.
    pub artifacts: Vec<(String, ArtifactDigest)>,
    /// Finite reward metrics emitted by the step verifier.
    pub reward: RewardDocument,
}

/// Immutable output from a successfully completed multi-step benchmark.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiStepExecutionResult {
    /// Verified step outputs in authored order.
    pub steps: Vec<StepExecutionResult>,
    /// Reward aggregated from the verified step rewards.
    pub reward: RewardDocument,
    /// Immutable identity of the verifier selected for the benchmark.
    pub verifier: ArtifactDigest,
}

/// Backend operations needed to execute one resolved benchmark step.
pub(crate) trait BenchmarkStepSession {
    /// Runs the externally supplied agent command for one step.
    fn run_agent(
        &mut self,
        step: &BenchmarkStepPlan,
        command: &[String],
    ) -> Result<(), EvalExecutionError>;

    /// Captures the step's effective artifact declarations.
    fn collect_artifacts(
        &mut self,
        step: &BenchmarkStepPlan,
    ) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError>;

    /// Runs the step verifier against its captured artifacts.
    fn run_verifier(
        &mut self,
        step: &BenchmarkStepPlan,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<RewardDocument, EvalExecutionError>;
}

/// Executes every resolved benchmark step and aggregates their verified rewards.
pub(crate) fn execute_benchmark_steps(
    plan: &BenchmarkExecutionPlan,
    command: &[String],
    verifier: ArtifactDigest,
    session: &mut dyn BenchmarkStepSession,
) -> Result<MultiStepExecutionResult, EvalExecutionError> {
    let mut steps = Vec::with_capacity(plan.steps().len());
    for step in plan.steps() {
        session.run_agent(step, command)?;
        let artifacts = session.collect_artifacts(step)?;
        let reward = session.run_verifier(step, &artifacts)?;
        steps.push(StepExecutionResult {
            name: step.name().to_owned(),
            artifacts,
            reward,
        });
    }

    let reward = aggregate_rewards(
        plan.multi_step_reward_strategy()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "multi-step reward strategy",
            ))?,
        &steps,
    )?;
    Ok(MultiStepExecutionResult {
        steps,
        reward,
        verifier,
    })
}

fn aggregate_rewards(
    strategy: MultiStepRewardStrategy,
    steps: &[StepExecutionResult],
) -> Result<RewardDocument, EvalExecutionError> {
    let last = steps
        .last()
        .ok_or(EvalExecutionError::InvalidRecipe("benchmark steps"))?;
    match strategy {
        MultiStepRewardStrategy::Final => Ok(last.reward.clone()),
        MultiStepRewardStrategy::Mean => {
            let mut metrics = BTreeMap::new();
            for (index, step) in steps.iter().enumerate() {
                let divisor = (index + 1) as f64;
                let retained_weight = index as f64 / divisor;
                for value in metrics.values_mut() {
                    *value *= retained_weight;
                }
                for (name, value) in &step.reward.metrics {
                    *metrics.entry(name.clone()).or_insert(0.0) += value / divisor;
                }
            }
            for (name, mean) in &mut metrics {
                let Some(value) = steps[0].reward.metrics.get(name) else {
                    continue;
                };
                if steps.iter().all(|step| {
                    step.reward
                        .metrics
                        .get(name)
                        .is_some_and(|candidate| candidate.to_bits() == value.to_bits())
                }) {
                    *mean = *value;
                }
            }
            RewardDocument::new(metrics)
                .map_err(|_| EvalExecutionError::InvalidRecipe("aggregated benchmark reward"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use crate::eval::{
        ArtifactDigest, EnvBinding, EnvironmentPlan, ImageSource, NetworkPolicy, PhasePlan,
        VerifierMode, VerifierPlan,
    };

    use super::*;

    #[test]
    fn runs_each_step_in_order_with_its_instruction() {
        let agent_command = vec!["agent".to_owned(), "--safe".to_owned()];
        let mut session = RecordingStepSession::with_rewards([
            reward([("score", 1.0)]),
            reward([("score", 3.0)]),
        ]);

        let result = execute_benchmark_steps(
            &plan(MultiStepRewardStrategy::Mean),
            &agent_command,
            ArtifactDigest::from_bytes(b"verifier"),
            &mut session,
        )
        .unwrap();

        assert_eq!(
            session.events,
            vec![
                "agent(one):First instruction.:agent --safe",
                "collect(one)",
                "verify(one)",
                "agent(two):Second instruction.:agent --safe",
                "collect(two)",
                "verify(two)",
            ]
        );
        assert_eq!(
            result
                .steps
                .iter()
                .map(|step| step.name.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "two"]
        );
        assert_eq!(result.reward, reward([("score", 2.0)]));
    }

    #[test]
    fn means_the_union_of_reward_metrics() {
        let mut session = RecordingStepSession::with_rewards([
            reward([("accuracy", 1.0), ("cost", 2.0)]),
            reward([("accuracy", 0.5), ("latency", 4.0)]),
        ]);

        let result = execute_benchmark_steps(
            &plan(MultiStepRewardStrategy::Mean),
            &["agent".to_owned()],
            ArtifactDigest::from_bytes(b"verifier"),
            &mut session,
        )
        .unwrap();

        assert_eq!(
            result.reward,
            reward([("accuracy", 0.75), ("cost", 1.0), ("latency", 2.0)])
        );
    }

    #[test]
    fn means_extreme_finite_reward_metrics_without_overflow() {
        let mut session = RecordingStepSession::with_rewards([
            reward([("score", f64::MAX)]),
            reward([("score", f64::MAX)]),
        ]);

        let result = execute_benchmark_steps(
            &plan(MultiStepRewardStrategy::Mean),
            &["agent".to_owned()],
            ArtifactDigest::from_bytes(b"verifier"),
            &mut session,
        );

        assert_eq!(result.unwrap().reward, reward([("score", f64::MAX)]));
    }

    #[test]
    fn preserves_three_equal_extreme_reward_metrics_exactly() {
        let mut session = RecordingStepSession::with_rewards([
            reward([("score", f64::MAX)]),
            reward([("score", f64::MAX)]),
            reward([("score", f64::MAX)]),
        ]);

        let result = execute_benchmark_steps(
            &plan_with_step_count(MultiStepRewardStrategy::Mean, 3),
            &["agent".to_owned()],
            ArtifactDigest::from_bytes(b"verifier"),
            &mut session,
        );

        assert_eq!(result.unwrap().reward, reward([("score", f64::MAX)]));
    }

    #[test]
    fn uses_the_last_reward_for_final_strategy() {
        let mut session = RecordingStepSession::with_rewards([
            reward([("accuracy", 1.0), ("cost", 2.0)]),
            reward([("latency", 4.0)]),
        ]);

        let result = execute_benchmark_steps(
            &plan(MultiStepRewardStrategy::Final),
            &["agent".to_owned()],
            ArtifactDigest::from_bytes(b"verifier"),
            &mut session,
        )
        .unwrap();

        assert_eq!(result.reward, reward([("latency", 4.0)]));
    }

    #[test]
    fn stops_after_the_first_phase_error() {
        for (phase, expected_events, error) in [
            (
                StepFailure::Agent,
                vec!["agent(one):First instruction.:agent"],
                EvalExecutionError::ProcessFailure("agent failed".to_owned()),
            ),
            (
                StepFailure::Collection,
                vec!["agent(one):First instruction.:agent", "collect(one)"],
                EvalExecutionError::ArtifactCollection("collect failed".to_owned()),
            ),
            (
                StepFailure::Verifier,
                vec![
                    "agent(one):First instruction.:agent",
                    "collect(one)",
                    "verify(one)",
                ],
                EvalExecutionError::ProcessFailure("verify failed".to_owned()),
            ),
        ] {
            let mut session = RecordingStepSession::failing(phase);

            assert_eq!(
                execute_benchmark_steps(
                    &plan(MultiStepRewardStrategy::Mean),
                    &["agent".to_owned()],
                    ArtifactDigest::from_bytes(b"verifier"),
                    &mut session,
                ),
                Err(error)
            );
            assert_eq!(session.events, expected_events);
        }
    }

    #[derive(Clone, Copy)]
    enum StepFailure {
        Agent,
        Collection,
        Verifier,
    }

    struct RecordingStepSession {
        events: Vec<String>,
        rewards: VecDeque<RewardDocument>,
        failure: Option<StepFailure>,
    }

    impl RecordingStepSession {
        fn with_rewards(rewards: impl IntoIterator<Item = RewardDocument>) -> Self {
            Self {
                events: Vec::new(),
                rewards: rewards.into_iter().collect(),
                failure: None,
            }
        }

        fn failing(failure: StepFailure) -> Self {
            Self {
                events: Vec::new(),
                rewards: VecDeque::new(),
                failure: Some(failure),
            }
        }
    }

    impl BenchmarkStepSession for RecordingStepSession {
        fn run_agent(
            &mut self,
            step: &BenchmarkStepPlan,
            command: &[String],
        ) -> Result<(), EvalExecutionError> {
            self.events.push(format!(
                "agent({}):{}:{}",
                step.name(),
                step.instruction().trim(),
                command.join(" ")
            ));
            if matches!(self.failure, Some(StepFailure::Agent)) {
                return Err(EvalExecutionError::ProcessFailure(
                    "agent failed".to_owned(),
                ));
            }
            Ok(())
        }

        fn collect_artifacts(
            &mut self,
            step: &BenchmarkStepPlan,
        ) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
            self.events.push(format!("collect({})", step.name()));
            if matches!(self.failure, Some(StepFailure::Collection)) {
                return Err(EvalExecutionError::ArtifactCollection(
                    "collect failed".to_owned(),
                ));
            }
            Ok(vec![(
                format!("{}.txt", step.name()),
                ArtifactDigest::from_bytes(step.name().as_bytes()),
            )])
        }

        fn run_verifier(
            &mut self,
            step: &BenchmarkStepPlan,
            _: &[(String, ArtifactDigest)],
        ) -> Result<RewardDocument, EvalExecutionError> {
            self.events.push(format!("verify({})", step.name()));
            if matches!(self.failure, Some(StepFailure::Verifier)) {
                return Err(EvalExecutionError::ProcessFailure(
                    "verify failed".to_owned(),
                ));
            }
            self.rewards
                .pop_front()
                .ok_or_else(|| EvalExecutionError::ProcessFailure("missing reward".to_owned()))
        }
    }

    fn plan(strategy: MultiStepRewardStrategy) -> BenchmarkExecutionPlan {
        plan_with_step_count(strategy, 2)
    }

    fn plan_with_step_count(
        strategy: MultiStepRewardStrategy,
        step_count: usize,
    ) -> BenchmarkExecutionPlan {
        let environment = environment();
        let agent = phase();
        let verifier = verifier(&environment);
        BenchmarkExecutionPlan {
            environment,
            agent: agent.clone(),
            verifier: verifier.clone(),
            artifacts: Vec::new(),
            compose: None,
            steps: [
                ("one", "First instruction."),
                ("two", "Second instruction."),
                ("three", "Third instruction."),
            ]
            .into_iter()
            .take(step_count)
            .map(|(name, instruction)| {
                BenchmarkStepPlan::new(
                    name.to_owned(),
                    instruction.to_owned(),
                    "tests".to_owned(),
                    agent.clone(),
                    verifier.clone(),
                    Vec::new(),
                )
            })
            .collect(),
            has_explicit_steps: true,
            multi_step_reward_strategy: Some(strategy),
        }
    }

    fn environment() -> EnvironmentPlan {
        EnvironmentPlan {
            image_source: ImageSource::task_dockerfile(ArtifactDigest::from_bytes(b"Dockerfile")),
            resources: None,
            workdir: None,
            user: None,
            env: BTreeMap::<String, EnvBinding>::new(),
            network: NetworkPolicy::public(),
            healthcheck: None,
        }
    }

    fn phase() -> PhasePlan {
        PhasePlan {
            user: None,
            env: BTreeMap::new(),
            network: NetworkPolicy::public(),
            timeout: None,
        }
    }

    fn verifier(environment: &EnvironmentPlan) -> VerifierPlan {
        VerifierPlan {
            phase: phase(),
            mode: VerifierMode::Shared,
            environment: environment.clone(),
        }
    }

    fn reward(entries: impl IntoIterator<Item = (&'static str, f64)>) -> RewardDocument {
        RewardDocument::new(
            entries
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
        )
        .unwrap()
    }
}
