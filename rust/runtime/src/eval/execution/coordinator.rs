// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native P0 ordering for import, sandbox, and verifier preparation.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, DeclaredArtifactTransfer, EvidenceEvent,
    EvidenceKind, FrozenAttemptBundle, FrozenAttemptError, HarborImportError, HarborImporter,
    HarborSource, ImportedTask, LocalExecutionResult, LocalProcessSandbox, ModelIdentity,
    PairedComparisonError, PairedComparisonReport, PairedComparisonSpec, PairedMeasurements,
    PolicyIdentity, RegradeError, RegradeRequest, RuntimeIdentity, ScoreVersion, SourceAcquirer,
    TrialBudget, TrialIdentityError, TrialSpec, VerifierExecutionError, VerifierMode,
    VerifierResult, VerifierSandboxFactory, prepare_verifier, regrade,
};

use super::{EvalExecutionError, EvalSandboxFactory, HarborAgentContract, HarborSandboxRecipe};

/// Native composition boundary for the P0 import and preparation lifecycle.
pub struct HarborEvaluationCoordinator<'a> {
    acquirer: &'a dyn SourceAcquirer,
    sandbox: &'a dyn EvalSandboxFactory,
    verifier: &'a dyn VerifierSandboxFactory,
}

/// Inputs that resolve and execute one local native Harbor-compatible evaluation.
pub struct HarborLocalEvaluationRequest {
    /// Source package acquired exactly once before environment provisioning.
    pub source: HarborSource,
    /// Immutable sandbox environment recipe.
    pub recipe: HarborSandboxRecipe,
    /// Selected installed or external agent contract.
    pub contract: HarborAgentContract,
    /// Immutable selected agent variant.
    pub agent_variant: AgentVariantRef,
    /// Immutable model selection.
    pub model: ModelIdentity,
    /// Deterministic trial seed.
    pub seed: u64,
    /// Immutable policy identity.
    pub policy: PolicyIdentity,
    /// Native runtime identity.
    pub runtime: RuntimeIdentity,
    /// Positive finite phase budgets.
    pub budget: TrialBudget,
    /// Append-only attempt identifier.
    pub attempt: AttemptId,
    /// Requested verifier topology.
    pub verifier_mode: VerifierMode,
    /// External agent argv, or `None` to select the package-installed command.
    pub agent_command: Option<Vec<String>>,
    /// Initial verifier metric.
    pub score_metric: String,
    /// Immutable initial score rationale.
    pub initial_rationale: ArtifactDigest,
    /// Verifier metric used for the append-only regrade.
    pub regrade_metric: String,
    /// Immutable regrade rationale.
    pub regrade_rationale: ArtifactDigest,
}

/// Versioned caller-authored identity inputs for one persisted evaluation lifecycle.
///
/// Package, environment, verifier, and backend identities are deliberately absent: the
/// executor derives those from its owned imported snapshot and selected backend.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HarborLifecycleRequest {
    /// Wire-contract version. Native execution currently supports only version one.
    pub version: u32,
    /// Immutable selected agent variant.
    pub agent_variant: AgentVariantRef,
    /// Explicit model identity.
    pub model: ModelIdentity,
    /// Deterministic attempt seed.
    pub seed: u64,
    /// Explicit immutable policy identity.
    pub policy: PolicyIdentity,
    /// Explicit native runtime identity.
    pub runtime: RuntimeIdentity,
    /// Append-only attempt identifier.
    pub attempt: AttemptId,
    /// Enforced agent and verifier phase budgets.
    pub budget: TrialBudget,
    /// Agent contract selected for the execution.
    pub agent_contract: HarborLifecycleAgentContract,
    /// Exact argv provenance selected by the caller.
    pub command: Vec<String>,
    /// Metric and rationale for the initial score.
    pub initial_score: HarborLifecycleScoreRequest,
    /// Metric and rationale for the append-only regrade.
    pub regrade: HarborLifecycleScoreRequest,
}

/// Agent contract recorded in a lifecycle request.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HarborLifecycleAgentContract {
    /// The package image provides the selected agent command.
    Installed,
    /// The caller supplies an external command.
    External,
    /// The runtime executes a native graph agent.
    NativeGraph,
    /// A package-declared compatibility driver owns the opaque episode loop.
    ExternallyDriven,
}

/// Immutable score selection supplied for one lifecycle revision.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HarborLifecycleScoreRequest {
    /// Exact finite metric emitted by the verifier.
    pub metric: String,
    /// Immutable rationale digest retained with the score.
    pub rationale: ArtifactDigest,
}

impl HarborLifecycleRequest {
    /// Validates the version and caller-owned fields before any environment provisioning.
    pub fn validate(&self) -> Result<(), HarborEvaluationError> {
        if self.version != 1 {
            return Err(HarborEvaluationError::InvalidRequest(format!(
                "unsupported lifecycle request version {}",
                self.version
            )));
        }
        TrialBudget::new(self.budget.execution_seconds, self.budget.verifier_seconds)
            .map_err(HarborEvaluationError::Trial)?;
        if self.command.is_empty() || self.command.iter().any(|value| value.trim().is_empty()) {
            return Err(HarborEvaluationError::InvalidRequest(
                "lifecycle command provenance must be a nonempty argv".to_owned(),
            ));
        }
        for metric in [&self.initial_score.metric, &self.regrade.metric] {
            if metric.trim().is_empty() {
                return Err(HarborEvaluationError::InvalidRequest(
                    "lifecycle score metrics must not be empty".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Immutable outputs from a completed local native evaluation attempt.
pub struct HarborCompletedEvaluation {
    /// Imported owned task snapshot.
    pub imported: ImportedTask,
    /// Resolved trial identity constructed before sandbox provisioning.
    pub trial: TrialSpec,
    /// Initial immutable score revision.
    pub initial_score: ScoreVersion,
    /// Append-only score regrade.
    pub regraded_score: ScoreVersion,
    /// Completed verifier result over declared artifacts.
    pub verifier_result: VerifierResult,
    /// Ordered immutable lifecycle evidence.
    pub evidence: Vec<EvidenceEvent>,
}

impl HarborCompletedEvaluation {
    /// Freezes completed Harbor facts for an evaluator without rerunning a verifier.
    pub fn freeze(&self) -> Result<FrozenAttemptBundle, FrozenAttemptError> {
        FrozenAttemptBundle::new(
            self.trial.identity_digest(),
            self.verifier_result.clone(),
            self.evidence.clone(),
            vec![self.initial_score.clone(), self.regraded_score.clone()],
        )
    }
}

impl<'a> HarborEvaluationCoordinator<'a> {
    /// Creates a coordinator over caller-owned native source and sandbox boundaries.
    pub fn new(
        acquirer: &'a dyn SourceAcquirer,
        sandbox: &'a dyn EvalSandboxFactory,
        verifier: &'a dyn VerifierSandboxFactory,
    ) -> Self {
        Self {
            acquirer,
            sandbox,
            verifier,
        }
    }

    /// Imports before provisioning, then preflights, opens, and prepares the verifier.
    pub fn prepare(
        &self,
        source: &HarborSource,
        recipe: &HarborSandboxRecipe,
        agent: &HarborAgentContract,
        verifier_mode: VerifierMode,
        transfer: &DeclaredArtifactTransfer,
    ) -> Result<ImportedTask, HarborEvaluationError> {
        let imported = HarborImporter::new(self.acquirer).import(source)?;
        self.sandbox.preflight(recipe, agent)?;
        self.sandbox.open(recipe)?;
        prepare_verifier(self.verifier, verifier_mode, transfer)?;
        Ok(imported)
    }

    /// Resolves the complete immutable trial before an executor provisions an environment.
    pub fn resolve_trial(
        imported: &ImportedTask,
        request: &HarborLifecycleRequest,
    ) -> Result<TrialSpec, HarborEvaluationError> {
        request.validate()?;
        let environment = ArtifactDigest::parse(imported.package.environment())
            .map_err(|error| HarborEvaluationError::InvalidRequest(error.to_string()))?;
        let verifier = ArtifactDigest::parse(imported.package.verifier())
            .map_err(|error| HarborEvaluationError::InvalidRequest(error.to_string()))?;
        TrialSpec::new(
            imported.task.clone(),
            request.agent_variant.clone(),
            request.model.clone(),
            request.seed,
            request.policy.clone(),
            request.budget.clone(),
            environment,
            verifier,
            request.runtime.clone(),
        )
        .map_err(HarborEvaluationError::Trial)
    }

    /// Completes immutable attempt records from executor-owned factual execution output.
    ///
    /// This method is executor-neutral: Docker and local paths supply the same completed
    /// execution result, while this coordinator owns score lineage and evidence ordering.
    pub fn complete_attempt(
        imported: ImportedTask,
        trial: TrialSpec,
        command: &[String],
        execution: LocalExecutionResult,
        request: &HarborLifecycleRequest,
    ) -> Result<HarborCompletedEvaluation, HarborEvaluationError> {
        request.validate()?;
        if command != request.command {
            return Err(HarborEvaluationError::InvalidRequest(
                "executed command disagrees with lifecycle command provenance".to_owned(),
            ));
        }
        let verifier_result = VerifierResult::new(
            request.attempt.clone(),
            execution.verifier.clone(),
            execution
                .artifacts
                .iter()
                .map(|(_, digest)| digest.clone())
                .collect(),
            execution.reward.clone(),
            request.regrade.rationale.clone(),
        )?;
        let initial_score = execution.initial_score(
            request.attempt.clone(),
            request.initial_score.metric.clone(),
            request.initial_score.rationale.clone(),
        )?;
        let regraded_score = regrade(RegradeRequest::new(
            initial_score.clone(),
            verifier_result.clone(),
            request.regrade.metric.clone(),
        )?)?;
        let evidence = completed_evidence(
            &imported,
            &request.attempt,
            command,
            &execution,
            &verifier_result,
        );
        Ok(HarborCompletedEvaluation {
            imported,
            trial,
            initial_score,
            regraded_score,
            verifier_result,
            evidence,
        })
    }

    /// Executes one local process attempt from import through evidence and score regrade.
    pub fn execute_local(
        &self,
        local: &LocalProcessSandbox,
        request: HarborLocalEvaluationRequest,
    ) -> Result<HarborCompletedEvaluation, HarborEvaluationError> {
        validate_local_request(&request)?;
        let imported = HarborImporter::new(self.acquirer).import(&request.source)?;
        let lifecycle = HarborLifecycleRequest {
            version: 1,
            agent_variant: request.agent_variant.clone(),
            model: request.model.clone(),
            seed: request.seed,
            policy: request.policy.clone(),
            runtime: request.runtime.clone(),
            attempt: request.attempt.clone(),
            budget: request.budget.clone(),
            agent_contract: match &request.contract {
                HarborAgentContract::Installed { .. } => HarborLifecycleAgentContract::Installed,
                HarborAgentContract::External { .. } => HarborLifecycleAgentContract::External,
                HarborAgentContract::NativeGraph { .. } => {
                    HarborLifecycleAgentContract::NativeGraph
                }
            },
            command: request
                .agent_command
                .clone()
                .unwrap_or_else(|| imported.package.agent_command().to_vec()),
            initial_score: HarborLifecycleScoreRequest {
                metric: request.score_metric.clone(),
                rationale: request.initial_rationale.clone(),
            },
            regrade: HarborLifecycleScoreRequest {
                metric: request.regrade_metric.clone(),
                rationale: request.regrade_rationale.clone(),
            },
        };
        let trial = Self::resolve_trial(&imported, &lifecycle)?;

        self.sandbox.preflight(&request.recipe, &request.contract)?;
        self.sandbox.open(&request.recipe)?;
        let execution = local
            .execute_with_agent_command(
                &request.recipe,
                &imported.package,
                &lifecycle.command,
                request.verifier_mode,
            )
            .map_err(HarborEvaluationError::Execution)?;
        Self::complete_attempt(imported, trial, &lifecycle.command, execution, &lifecycle)
    }

    /// Compares completed attempts only when their derived fixed baselines match `spec`.
    pub fn compare_completed(
        baseline: (&HarborCompletedEvaluation, PairedMeasurements),
        candidate: (&HarborCompletedEvaluation, PairedMeasurements),
        spec: &PairedComparisonSpec,
    ) -> Result<PairedComparisonReport, HarborEvaluationError> {
        let baseline_spec = paired_spec(&baseline.0.trial)?;
        let candidate_spec = paired_spec(&candidate.0.trial)?;
        if baseline.0.trial.budget != candidate.0.trial.budget {
            return Err(HarborEvaluationError::Paired(
                PairedComparisonError::ChangedBaseline,
            ));
        }
        spec.compare_to(&baseline_spec)?;
        baseline_spec
            .compare_measurements(&candidate_spec, baseline.1, candidate.1)
            .map_err(HarborEvaluationError::Paired)
    }
}

fn validate_local_request(
    request: &HarborLocalEvaluationRequest,
) -> Result<(), HarborEvaluationError> {
    if request.verifier_mode == VerifierMode::Separate {
        return Err(HarborEvaluationError::InvalidRequest(
            "local process execution cannot provide separate verifier isolation".to_owned(),
        ));
    }
    if request.score_metric.trim().is_empty() || request.regrade_metric.trim().is_empty() {
        return Err(HarborEvaluationError::InvalidRequest(
            "score metrics must not be empty".to_owned(),
        ));
    }
    match (&request.contract, &request.agent_command) {
        (HarborAgentContract::Installed { .. }, None) => Ok(()),
        (HarborAgentContract::External { .. }, Some(command))
            if !command.is_empty()
                && command.iter().all(|argument| !argument.trim().is_empty()) =>
        {
            Ok(())
        }
        (HarborAgentContract::External { .. }, None) => Err(HarborEvaluationError::InvalidRequest(
            "external agent contracts require an external command".to_owned(),
        )),
        (HarborAgentContract::Installed { .. }, Some(_)) => {
            Err(HarborEvaluationError::InvalidRequest(
                "installed agent contracts must use the package command".to_owned(),
            ))
        }
        (_, Some(_)) | (HarborAgentContract::NativeGraph { .. }, None) => {
            Err(HarborEvaluationError::InvalidRequest(
                "local process execution supports installed or external agents only".to_owned(),
            ))
        }
    }
}

fn completed_evidence(
    imported: &ImportedTask,
    attempt: &AttemptId,
    command: &[String],
    execution: &super::LocalExecutionResult,
    verifier_result: &VerifierResult,
) -> Vec<EvidenceEvent> {
    let mut evidence = Vec::with_capacity(execution.artifacts.len() + 3);
    let mut sequence = 0;
    evidence.push(EvidenceEvent::new(
        attempt.clone(),
        sequence,
        EvidenceKind::Sandbox,
        imported.report.source_digest.clone(),
        None,
    ));
    sequence += 1;
    evidence.push(EvidenceEvent::new(
        attempt.clone(),
        sequence,
        EvidenceKind::Agent,
        ArtifactDigest::from_bytes(command.join("\0").as_bytes()),
        None,
    ));
    for (_, digest) in &execution.artifacts {
        sequence += 1;
        evidence.push(EvidenceEvent::new(
            attempt.clone(),
            sequence,
            EvidenceKind::Artifact,
            digest.clone(),
            None,
        ));
    }
    sequence += 1;
    evidence.push(EvidenceEvent::new(
        attempt.clone(),
        sequence,
        EvidenceKind::Evaluator,
        ArtifactDigest::from_bytes(
            format!(
                "metrics={:?}\u{1f}rationale={}",
                execution.reward.metrics,
                verifier_result.rationale.as_str(),
            )
            .as_bytes(),
        ),
        None,
    ));
    evidence
}

fn paired_spec(trial: &TrialSpec) -> Result<PairedComparisonSpec, HarborEvaluationError> {
    let total_seconds = trial.budget.execution_seconds + trial.budget.verifier_seconds;
    if !total_seconds.is_finite() || total_seconds <= 0.0 || total_seconds.fract() != 0.0 {
        return Err(HarborEvaluationError::InvalidRequest(
            "paired local trials require integral finite total budgets".to_owned(),
        ));
    }
    PairedComparisonSpec::new(
        trial.task.digest.as_str(),
        format!("{}:{}", trial.model.provider, trial.model.model),
        trial.seed,
        trial.policy.digest().as_str(),
        trial.environment.as_str(),
        total_seconds as u64,
    )
    .map_err(HarborEvaluationError::Paired)
}

/// Failure during native P0 lifecycle preparation.
#[derive(Debug)]
pub enum HarborEvaluationError {
    /// Source import failed before environment provisioning.
    Import(HarborImportError),
    /// Agent sandbox validation or opening failed.
    Sandbox(EvalExecutionError),
    /// Verifier isolation preparation failed.
    Verifier(VerifierExecutionError),
    /// Trial identity resolution failed before sandbox provisioning.
    Trial(TrialIdentityError),
    /// Local process execution failed after the sandbox opened.
    Execution(EvalExecutionError),
    /// Immutable score construction or regrade failed.
    Score(RegradeError),
    /// Paired comparison baseline validation failed.
    Paired(PairedComparisonError),
    /// The local composition request selected an unsupported contract combination.
    InvalidRequest(String),
}

impl Display for HarborEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Import(error) => error.fmt(formatter),
            Self::Sandbox(error) => error.fmt(formatter),
            Self::Verifier(error) => error.fmt(formatter),
            Self::Trial(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
            Self::Score(error) => error.fmt(formatter),
            Self::Paired(error) => error.fmt(formatter),
            Self::InvalidRequest(error) => formatter.write_str(error),
        }
    }
}

impl std::error::Error for HarborEvaluationError {}

impl From<HarborImportError> for HarborEvaluationError {
    fn from(error: HarborImportError) -> Self {
        Self::Import(error)
    }
}

impl From<EvalExecutionError> for HarborEvaluationError {
    fn from(error: EvalExecutionError) -> Self {
        Self::Sandbox(error)
    }
}

impl From<VerifierExecutionError> for HarborEvaluationError {
    fn from(error: VerifierExecutionError) -> Self {
        Self::Verifier(error)
    }
}

impl From<TrialIdentityError> for HarborEvaluationError {
    fn from(error: TrialIdentityError) -> Self {
        Self::Trial(error)
    }
}

impl From<RegradeError> for HarborEvaluationError {
    fn from(error: RegradeError) -> Self {
        Self::Score(error)
    }
}

impl From<PairedComparisonError> for HarborEvaluationError {
    fn from(error: PairedComparisonError) -> Self {
        Self::Paired(error)
    }
}
