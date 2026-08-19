// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Matrix episode execution over a Rust-owned NativeGraph callback and verifier facts.

use std::{
    cell::RefCell,
    fmt::{self, Display, Formatter},
    rc::Rc,
    sync::Arc,
};

use async_trait::async_trait;

use crate::{
    engine::{application::Application, record_lane::EvalNodeRecordArtifact},
    eval::{
        DockerProcessSandbox, DockerRuntime, HarborCompletedEvaluation,
        HarborEvaluationCoordinator, HarborLifecycleAgentContract, HarborLifecycleRequest,
        HarborSandboxRecipe, ImportedTask, NativeGraphEpisodeCallback,
        PreparedExternalDriverCapability, SecretProvider,
    },
    extensions::AIPerfRegistry,
};

use super::{
    EngineNativeGraphEpisodeCallback, ModelRuntimeConfig, NativeGraphAttemptAuthority,
    NativeGraphCompletedAttempt, ObservedNativeGraphTransportEvidence,
    bind_native_graph_environment_stepper,
};

use crate::eval::{EpisodeAssignment, EpisodeEvaluatorFactory, EpisodeRunner, MatrixError};

/// Executes one admitted NativeGraph episode and seals its completed Harbor and rollout facts.
///
/// Implementations own environment acquisition, invoke the Rust graph callback,
/// preserve the ordinary artifact/verifier lifecycle, and return only a sealed completion.
/// The matrix runner never invents a reward, verifier evidence, or execution terminality.
#[async_trait(?Send)]
pub trait NativeGraphEpisodeExecutor {
    /// Runs one admitted assignment through its model callback and verifier boundary.
    async fn execute(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError>;
}

/// Concrete Docker-backed executor for one imported, immutable NativeGraph package.
///
/// It owns the only path from the Rust model callback through the ordinary Docker
/// artifact/verifier transaction to the completed Harbor fact model. The matrix
/// runner receives only `HarborCompletedEvaluation::freeze()` output, never a
/// synthetic reward or verifier result.
pub struct DockerNativeGraphEpisodeExecutor {
    sandbox: DockerProcessSandbox,
    runtime: DockerExecutorRuntime,
    recipe: HarborSandboxRecipe,
    imported: ImportedTask,
    lifecycle: HarborLifecycleRequest,
    application: Rc<Application>,
    model_runtime: ModelRuntimeConfig,
    secrets: Rc<dyn SecretProvider>,
    record_artifact: Option<EvalNodeRecordArtifact>,
}

/// Concrete Docker-backed executor for one prepared externally driven episode.
///
/// The prepared Driver capability is consumed exactly once and remains opaque until the
/// Docker transaction binds it to its authorized spawn. The executor then freezes only the
/// bounded compatibility supplement beside the ordinary Harbor completion.
pub struct DockerExternallyDrivenEpisodeExecutor {
    sandbox: DockerProcessSandbox,
    runtime: DockerExecutorRuntime,
    recipe: HarborSandboxRecipe,
    imported: ImportedTask,
    lifecycle: HarborLifecycleRequest,
    prepared_driver: RefCell<Option<PreparedExternalDriverCapability>>,
    secrets: Rc<dyn SecretProvider>,
}

enum DockerExecutorRuntime {
    Host,
    Injected(Rc<dyn DockerRuntime>),
}

impl DockerExternallyDrivenEpisodeExecutor {
    /// Binds Docker's production runtime to one already-prepared external trial.
    pub fn new(
        sandbox: DockerProcessSandbox,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        prepared_driver: PreparedExternalDriverCapability,
        secrets: Rc<dyn SecretProvider>,
    ) -> Result<Self, EpisodeExecutionError> {
        Self::new_inner(
            sandbox,
            DockerExecutorRuntime::Host,
            recipe,
            imported,
            lifecycle,
            prepared_driver,
            secrets,
        )
    }

    /// Binds an injectable Docker provider to one already-prepared external trial.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime(
        sandbox: DockerProcessSandbox,
        runtime: Rc<dyn DockerRuntime>,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        prepared_driver: PreparedExternalDriverCapability,
        secrets: Rc<dyn SecretProvider>,
    ) -> Result<Self, EpisodeExecutionError> {
        Self::new_inner(
            sandbox,
            DockerExecutorRuntime::Injected(runtime),
            recipe,
            imported,
            lifecycle,
            prepared_driver,
            secrets,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        sandbox: DockerProcessSandbox,
        runtime: DockerExecutorRuntime,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        prepared_driver: PreparedExternalDriverCapability,
        secrets: Rc<dyn SecretProvider>,
    ) -> Result<Self, EpisodeExecutionError> {
        if !imported.package.native_graph().is_some_and(|package| {
            package.profile() == crate::eval::NativeGraphProfile::ExternallyDriven
        }) {
            return Err(EpisodeExecutionError::Configuration(
                "Docker external executor requires an imported externally_driven package"
                    .to_owned(),
            ));
        }
        if lifecycle.agent_contract != HarborLifecycleAgentContract::ExternallyDriven {
            return Err(EpisodeExecutionError::Configuration(
                "Docker external executor requires externally_driven lifecycle provenance"
                    .to_owned(),
            ));
        }
        HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        Ok(Self {
            sandbox,
            runtime,
            recipe,
            imported,
            lifecycle,
            prepared_driver: RefCell::new(Some(prepared_driver)),
            secrets,
        })
    }

    fn lifecycle_for_assignment(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<(HarborLifecycleRequest, crate::eval::TrialSpec), EpisodeExecutionError> {
        if assignment.package().identity_digest() != self.imported.package.identity_digest() {
            return Err(EpisodeExecutionError::Configuration(
                "external assignment package does not match the executor snapshot".to_owned(),
            ));
        }
        let mut lifecycle = self.lifecycle.clone();
        lifecycle.attempt = assignment.attempt_id().clone();
        let trial = HarborEvaluationCoordinator::resolve_trial(&self.imported, &lifecycle)
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        if &trial.identity_digest() != assignment.trial_digest() {
            return Err(EpisodeExecutionError::Configuration(
                "external assignment trial does not match the lifecycle request".to_owned(),
            ));
        }
        Ok((lifecycle, trial))
    }
}

impl DockerNativeGraphEpisodeExecutor {
    /// Binds Docker's production no-egress runtime after immutable inputs freeze.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sandbox: DockerProcessSandbox,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        application: Rc<Application>,
        model_runtime: ModelRuntimeConfig,
        secrets: Rc<dyn SecretProvider>,
        record_artifact: Option<EvalNodeRecordArtifact>,
    ) -> Result<Self, EpisodeExecutionError> {
        Self::new_inner(
            sandbox,
            DockerExecutorRuntime::Host,
            recipe,
            imported,
            lifecycle,
            application,
            model_runtime,
            secrets,
            record_artifact,
        )
    }

    /// Binds an injectable Docker provider after every immutable episode input has frozen.
    ///
    /// Production composition supplies its selected Docker provider here; tests use the
    /// same provider seam to inspect lifecycle ordering without introducing a second runner.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_runtime(
        sandbox: DockerProcessSandbox,
        runtime: Rc<dyn DockerRuntime>,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        application: Rc<Application>,
        model_runtime: ModelRuntimeConfig,
        secrets: Rc<dyn SecretProvider>,
        record_artifact: Option<EvalNodeRecordArtifact>,
    ) -> Result<Self, EpisodeExecutionError> {
        Self::new_inner(
            sandbox,
            DockerExecutorRuntime::Injected(runtime),
            recipe,
            imported,
            lifecycle,
            application,
            model_runtime,
            secrets,
            record_artifact,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        sandbox: DockerProcessSandbox,
        runtime: DockerExecutorRuntime,
        recipe: HarborSandboxRecipe,
        imported: ImportedTask,
        lifecycle: HarborLifecycleRequest,
        application: Rc<Application>,
        model_runtime: ModelRuntimeConfig,
        secrets: Rc<dyn SecretProvider>,
        record_artifact: Option<EvalNodeRecordArtifact>,
    ) -> Result<Self, EpisodeExecutionError> {
        if imported.package.native_graph().is_none() {
            return Err(EpisodeExecutionError::Configuration(
                "Docker NativeGraph executor requires an imported NativeGraph package".to_owned(),
            ));
        }
        if lifecycle.agent_contract != HarborLifecycleAgentContract::NativeGraph {
            return Err(EpisodeExecutionError::Configuration(
                "Docker NativeGraph executor requires native_graph lifecycle provenance".to_owned(),
            ));
        }
        HarborEvaluationCoordinator::resolve_trial(&imported, &lifecycle)
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        Ok(Self {
            sandbox,
            runtime,
            recipe,
            imported,
            lifecycle,
            application,
            model_runtime,
            secrets,
            record_artifact,
        })
    }

    fn lifecycle_for_assignment(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<(HarborLifecycleRequest, crate::eval::TrialSpec), EpisodeExecutionError> {
        if assignment.package().identity_digest() != self.imported.package.identity_digest() {
            return Err(EpisodeExecutionError::Configuration(
                "NativeGraph assignment package does not match the executor snapshot".to_owned(),
            ));
        }
        let mut lifecycle = self.lifecycle.clone();
        lifecycle.attempt = assignment.attempt_id().clone();
        let trial = HarborEvaluationCoordinator::resolve_trial(&self.imported, &lifecycle)
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        if &trial.identity_digest() != assignment.trial_digest() {
            return Err(EpisodeExecutionError::Configuration(
                "NativeGraph assignment trial does not match the lifecycle request".to_owned(),
            ));
        }
        Ok((lifecycle, trial))
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeExecutor for DockerExternallyDrivenEpisodeExecutor {
    async fn execute(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError> {
        let (lifecycle, trial) = self.lifecycle_for_assignment(assignment)?;
        let prepared_driver = self.prepared_driver.borrow_mut().take().ok_or_else(|| {
            EpisodeExecutionError::Configuration(
                "external Driver preparation was already consumed".to_owned(),
            )
        })?;
        let authority = NativeGraphAttemptAuthority::from_resolved_trial(assignment.trial());
        let (execution, supplement) = match &self.runtime {
            DockerExecutorRuntime::Host => {
                self.sandbox
                    .execute_externally_driven(
                        &self.recipe,
                        &self.imported.package,
                        assignment.trial(),
                        prepared_driver,
                        self.secrets.as_ref(),
                    )
                    .await
            }
            DockerExecutorRuntime::Injected(runtime) => {
                self.sandbox
                    .execute_externally_driven_with_runtime(
                        runtime.as_ref(),
                        &self.recipe,
                        &self.imported.package,
                        self.imported.package.execution_plan(),
                        assignment.trial(),
                        prepared_driver,
                        self.secrets.as_ref(),
                    )
                    .await
            }
        }
        .map_err(EpisodeExecutionError::Callback)?;
        let completed: HarborCompletedEvaluation = HarborEvaluationCoordinator::complete_attempt(
            self.imported.clone(),
            trial,
            &lifecycle.command,
            execution,
            &lifecycle,
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let frozen = completed
            .freeze()
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        NativeGraphCompletedAttempt::freeze_compatibility(&authority, frozen, supplement)
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))
    }
}

#[async_trait(?Send)]
impl NativeGraphEpisodeExecutor for DockerNativeGraphEpisodeExecutor {
    async fn execute(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<NativeGraphCompletedAttempt, EpisodeExecutionError> {
        let (lifecycle, trial) = self.lifecycle_for_assignment(assignment)?;
        let native = self.imported.package.native_graph().ok_or_else(|| {
            EpisodeExecutionError::Configuration(
                "NativeGraph package disappeared after immutable executor construction".to_owned(),
            )
        })?;
        let mut callback = EngineNativeGraphEpisodeCallback::new(
            self.application.as_ref(),
            native,
            &self.model_runtime,
            self.secrets.as_ref(),
            self.record_artifact.clone(),
        )
        .map_err(|error| {
            EpisodeExecutionError::Callback(crate::eval::EvalExecutionError::NativeGraphModel(
                error.to_string(),
            ))
        })?;
        if native.rollout().is_some() {
            let bound = bind_native_graph_environment_stepper(
                self.application.product_registry(),
                assignment.trial(),
            )
            .map_err(|error| EpisodeExecutionError::Configuration(error.to_string()))?;
            callback
                .bind_live_rollout(
                    bound,
                    NativeGraphAttemptAuthority::from_resolved_trial(assignment.trial()),
                    native
                        .rollout()
                        .ok_or_else(|| {
                            EpisodeExecutionError::Configuration(
                                "NativeGraph rollout disappeared after immutable executor construction"
                                    .to_owned(),
                            )
                        })?
                        .workspace_patch()
                        .clone(),
                )
                .map_err(|error| {
                    EpisodeExecutionError::Callback(
                        crate::eval::EvalExecutionError::NativeGraphModel(error.to_string()),
                    )
                })?;
        }
        let execution = match &self.runtime {
            DockerExecutorRuntime::Host => {
                self.sandbox
                    .execute_native_graph(
                        &self.recipe,
                        &self.imported.package,
                        self.model_runtime.secrets.clone(),
                        self.secrets.as_ref(),
                        &mut callback,
                    )
                    .await
            }
            DockerExecutorRuntime::Injected(runtime) => {
                self.sandbox
                    .execute_native_graph_with_runtime(
                        runtime.as_ref(),
                        &self.recipe,
                        &self.imported.package,
                        self.imported.package.execution_plan(),
                        self.secrets.as_ref(),
                        &mut callback,
                    )
                    .await
            }
        }
        .map_err(EpisodeExecutionError::Callback)?;
        let rollout = callback.take_rollout_evidence();
        let evidence = callback
            .transport_evidence()
            .ok_or(EpisodeExecutionError::MissingTransportEvidence)?;
        if !has_admitted_native_graph_transport_evidence(evidence) {
            return Err(EpisodeExecutionError::UnexpectedTransportEvidence {
                model_records: evidence.model_records(),
                completed_traces: evidence.completed_traces(),
                live_policy_calls: evidence.live_policy_calls(),
            });
        }
        let completed: HarborCompletedEvaluation = HarborEvaluationCoordinator::complete_attempt(
            self.imported.clone(),
            trial,
            &lifecycle.command,
            execution,
            &lifecycle,
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        let frozen = completed
            .freeze()
            .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))?;
        NativeGraphCompletedAttempt::freeze(
            &NativeGraphAttemptAuthority::from_resolved_trial(assignment.trial()),
            frozen,
            rollout,
        )
        .map_err(|error| EpisodeExecutionError::Facts(error.to_string()))
    }
}

fn has_admitted_native_graph_transport_evidence(
    evidence: ObservedNativeGraphTransportEvidence,
) -> bool {
    evidence.completed_traces() == 1
        && (evidence.model_records() > 0 || evidence.live_policy_calls() > 0)
}

/// Task-2 matrix runner that delegates scoring to the selected Task-3 evaluator.
pub struct NativeGraphEpisodeRunner {
    executor: Rc<dyn NativeGraphEpisodeExecutor>,
    evaluator_factory: Rc<dyn EpisodeEvaluatorFactory>,
}

struct RegisteredEpisodeEvaluatorFactory(Arc<dyn EpisodeEvaluatorFactory>);

impl EpisodeEvaluatorFactory for RegisteredEpisodeEvaluatorFactory {
    fn create(
        &self,
        trial: &crate::eval::ResolvedEpisodeTrial,
    ) -> Result<Rc<dyn crate::eval::EpisodeEvaluator>, crate::eval::EpisodeEvaluationError> {
        self.0.create(trial)
    }
}

impl NativeGraphEpisodeRunner {
    /// Binds one executor and evaluator after package/runtime selection has frozen.
    pub fn new(
        executor: Rc<dyn NativeGraphEpisodeExecutor>,
        evaluator_factory: Rc<dyn EpisodeEvaluatorFactory>,
    ) -> Self {
        Self {
            executor,
            evaluator_factory,
        }
    }

    /// Resolves the selected evaluator from the frozen application registry.
    pub fn with_registered_evaluator(
        executor: Rc<dyn NativeGraphEpisodeExecutor>,
        registry: &AIPerfRegistry,
        evaluator_name: &str,
    ) -> Result<Self, EpisodeExecutionError> {
        let evaluator_factory = registry
            .native_graph_evaluator(evaluator_name)
            .ok_or_else(|| {
                EpisodeExecutionError::Configuration(format!(
                    "no linked NativeGraph evaluator factory named {evaluator_name:?}"
                ))
            })?
            .clone();
        Ok(Self::new(
            executor,
            Rc::new(RegisteredEpisodeEvaluatorFactory(evaluator_factory)),
        ))
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for NativeGraphEpisodeRunner {
    async fn run(
        &self,
        assignment: EpisodeAssignment,
    ) -> Result<crate::eval::EpisodeResult, MatrixError> {
        let completed = self
            .executor
            .execute(&assignment)
            .await
            .map_err(|error| MatrixError::RunnerExecutionFailed(error.to_string()))?;
        let authority = NativeGraphAttemptAuthority::from_resolved_trial(assignment.trial());
        if authority.requires_rollout_evidence() != completed.has_rollout() {
            return Err(MatrixError::RunnerExecutionFailed(
                "native graph executor omitted or added sealed rollout evidence contrary to the imported assignment"
                    .to_owned(),
            ));
        }
        if authority.is_externally_driven() != completed.has_compatibility() {
            return Err(MatrixError::RunnerExecutionFailed(
                "native graph executor omitted or added sealed compatibility evidence contrary to the imported assignment"
                    .to_owned(),
            ));
        }
        if completed.frozen_attempt().trial_digest() != assignment.trial_digest()
            || completed.frozen_attempt().attempt() != assignment.attempt_id()
        {
            return Err(MatrixError::RunnerExecutionFailed(
                "native graph executor returned frozen facts for another assignment".to_owned(),
            ));
        }
        let evaluator = self
            .evaluator_factory
            .create(assignment.trial())
            .map_err(|error| MatrixError::RunnerExecutionFailed(error.to_string()))?;
        evaluator
            .evaluate_native_graph(completed)
            .await
            .map_err(|error| MatrixError::RunnerExecutionFailed(error.to_string()))
    }
}

/// Failure while obtaining immutable evaluator facts for an admitted episode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EpisodeExecutionError {
    /// Immutable executor inputs or an admitted assignment did not agree.
    Configuration(String),
    /// The Rust-owned graph callback failed before verifier collection.
    Callback(crate::eval::EvalExecutionError),
    /// The callback did not publish a completed transport-observer summary.
    MissingTransportEvidence,
    /// The completed observer summary disagreed with the static graph contract.
    UnexpectedTransportEvidence {
        /// Number of complete native model records produced by the graph callback.
        model_records: usize,
        /// Number of completed graph traces produced by the graph callback.
        completed_traces: usize,
        /// Number of non-raw selected-policy calls completed for a live rollout.
        live_policy_calls: u64,
    },
    /// Constructing immutable verifier or score facts failed.
    Facts(String),
}

impl Display for EpisodeExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(reason) => write!(
                formatter,
                "native graph episode configuration is invalid: {reason}"
            ),
            Self::Callback(error) => error.fmt(formatter),
            Self::MissingTransportEvidence => {
                formatter.write_str("native graph callback completed without transport evidence")
            }
            Self::UnexpectedTransportEvidence {
                model_records,
                completed_traces,
                live_policy_calls,
            } => write!(
                formatter,
                "native graph callback observed {model_records} model records, {live_policy_calls} live policy calls, and {completed_traces} completed traces"
            ),
            Self::Facts(reason) => write!(
                formatter,
                "native graph episode facts are invalid: {reason}"
            ),
        }
    }
}

impl std::error::Error for EpisodeExecutionError {}

#[cfg(test)]
mod tests {
    use crate::engine::graph_execution::NativeGraphTransportEvidence;

    use super::{
        ObservedNativeGraphTransportEvidence, has_admitted_native_graph_transport_evidence,
    };

    #[test]
    fn rollout_only_transport_evidence_requires_one_trace_and_one_live_policy_call() {
        assert!(has_admitted_native_graph_transport_evidence(
            ObservedNativeGraphTransportEvidence::from(NativeGraphTransportEvidence {
                model_records: 0,
                completed_traces: 1,
                live_policy_calls: 1,
            })
        ));
        assert!(!has_admitted_native_graph_transport_evidence(
            ObservedNativeGraphTransportEvidence::from(NativeGraphTransportEvidence {
                model_records: 0,
                completed_traces: 1,
                live_policy_calls: 0,
            })
        ));
    }
}
