// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed execution for conventional native task directories.

use std::{
    cell::{Cell, RefCell},
    fs,
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::PermissionsExt,
    process::{Child, ChildStdout, Command, Stdio},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use tempfile::{NamedTempFile, TempDir};

use crate::{
    clock::{Clock, RealClock},
    eval::{ArtifactDigest, HarborTaskPackage, RewardDocument, VerifierMode},
};

use super::{
    BenchmarkExecutionPlan, BenchmarkStepPlan, ComposeProjectId, DockerBuildRequest,
    DockerComposeArchiveRequest, DockerComposeBuildRequest, DockerComposeConfigRequest,
    DockerComposeCopyRequest, DockerComposeDownRequest, DockerComposeExecRequest,
    DockerComposeRuntime, DockerComposeStopRequest, DockerComposeUpRequest, DockerCopyRequest,
    DockerCreateRequest, DockerExecRequest, DockerRemoveRequest, DockerRuntime, DockerStartRequest,
    EvalExecutionError, EvalExecutionPhase, HarborSandboxRecipe, LocalExecutionResult,
    MultiStepExecutionResult, NetworkPolicy, SecretProvider, collect_artifacts, preflight_docker,
    resolve_environment, resolve_phase_environment,
    shared_workdir_conflicts_reserved_verifier_path, transfer_artifacts,
    verifier_artifact_target_collision,
};

use super::docker_runtime::preflight_compose_configuration;
use super::multi_step::{BenchmarkStepSession, execute_benchmark_steps};
use super::{
    artifacts::{Deadline, collect_service_artifacts, collect_service_evidence},
    compose_project::ComposeProjectLease,
    task_environment::{ServiceArchiveRequest, ServiceExecRequest, TaskEnvironmentLease},
};

/// Executes a conventional task in a task-built Docker environment.
pub struct DockerProcessSandbox {
    clock: Rc<dyn Clock>,
}

impl std::fmt::Debug for DockerProcessSandbox {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DockerProcessSandbox")
            .finish_non_exhaustive()
    }
}

impl Default for DockerProcessSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl DockerProcessSandbox {
    /// Creates a Docker-backed task executor.
    pub fn new() -> Self {
        Self::with_clock(RealClock::new())
    }

    /// Creates a Docker-backed task executor using the supplied execution clock.
    pub fn with_clock(clock: Rc<dyn Clock>) -> Self {
        Self { clock }
    }

    /// Builds the task environment, executes an external agent, and runs a shared verifier.
    pub fn execute(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if package.execution_plan().is_multi_step() {
            return Err(EvalExecutionError::UnsupportedMultiStep);
        }
        if package.timeouts().is_some() && tokio::runtime::Handle::try_current().is_ok() {
            return Err(EvalExecutionError::RuntimeContext(
                "synchronous Docker execution",
            ));
        }
        if !package.is_standard_directory() {
            return Err(EvalExecutionError::Materialization(
                "Docker execution requires a standard task directory".to_owned(),
            ));
        }
        let _ = verifier_mode;
        let runtime = DockerCliRuntime {
            clock: self.clock.clone(),
        };
        self.execute_with_runtime(
            &runtime,
            recipe,
            package,
            package.execution_plan(),
            agent_command,
            &HostSecretProvider,
        )
    }

    /// Executes an explicit step layout in one persistent Docker agent environment.
    pub fn execute_multi_step(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
    ) -> Result<MultiStepExecutionResult, EvalExecutionError> {
        if !package.execution_plan().is_multi_step() {
            return Err(EvalExecutionError::InvalidRecipe(
                "multi-step execution plan",
            ));
        }
        if multi_step_uses_clock_drive(package.execution_plan())
            && tokio::runtime::Handle::try_current().is_ok()
        {
            return Err(EvalExecutionError::RuntimeContext(
                "synchronous Docker execution",
            ));
        }
        let runtime = DockerCliRuntime {
            clock: self.clock.clone(),
        };
        self.execute_multi_step_with_runtime(
            &runtime,
            recipe,
            package,
            package.execution_plan(),
            agent_command,
            &HostSecretProvider,
        )
    }

    /// Executes an explicit step layout through an injectable Docker provider.
    pub fn execute_multi_step_with_runtime(
        &self,
        runtime: &dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        agent_command: &[String],
        secrets: &dyn SecretProvider,
    ) -> Result<MultiStepExecutionResult, EvalExecutionError> {
        if !package.execution_plan().is_multi_step() || !plan.is_multi_step() {
            return Err(EvalExecutionError::InvalidRecipe(
                "multi-step execution plan",
            ));
        }
        if !package.is_standard_directory() {
            return Err(EvalExecutionError::Materialization(
                "Docker execution requires a standard task directory".to_owned(),
            ));
        }
        preflight_docker(runtime, plan)?;
        if plan.compose().is_some() {
            return self.execute_compose_multi_step_with_runtime(
                runtime,
                recipe,
                package,
                plan,
                agent_command,
                secrets,
            );
        }
        let environment = plan.environment();
        if !runtime.supports_phase_network_transitions()
            && plan.steps().iter().any(|step| {
                step.agent().network() != environment.network()
                    || step.verifier().phase().network() != step.verifier().environment().network()
            })
        {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "phase network transition",
            ));
        }
        let environment_workdir = recipe.resolve_workdir(environment.workdir());
        validate_shared_verifier_workdir(runtime, plan, None, environment_workdir)?;
        let materialized_source = package.materialize_source()?;
        let (source_root, environment_root) =
            standard_task_roots(package, materialized_source.root())?;

        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let (image, container) = docker_run_names(package);
        let mut containers = vec![container.clone()];
        let outcome = (|| {
            let baseline_network = network_lease(environment.network())?;
            let build_network = build_network_lease(environment.network())?;
            runtime.build(
                &DockerBuildRequest::new([
                    "build",
                    "--network",
                    build_network,
                    "--tag",
                    &image,
                    environment_root.to_string_lossy().as_ref(),
                ])
                .with_network_lease(build_network),
            )?;
            create_planned_container(
                runtime,
                &container,
                &image,
                ContainerWorkspace::at_workdir(workspace.path(), environment_workdir),
                environment,
                baseline_network,
                None,
            )?;
            runtime.start(&DockerStartRequest::new(&container))?;
            validate_shared_verifier_workdir(runtime, plan, Some(&container), environment_workdir)?;
            if let Some(healthcheck) = environment.healthcheck() {
                run_healthcheck(
                    self.clock.clone(),
                    runtime,
                    &container,
                    environment,
                    environment_workdir,
                    healthcheck,
                    baseline_network,
                    secrets,
                )?;
            }

            let mut session = DockerStepSession {
                clock: self.clock.clone(),
                runtime,
                recipe,
                source_root,
                environment,
                image: &image,
                agent_container: &container,
                secrets,
                containers: &mut containers,
                artifact_collection: None,
            };
            execute_benchmark_steps(plan, agent_command, package.source_digest(), &mut session)
        })();
        finish_with_cleanup(runtime, containers, outcome)
    }

    /// Executes a normalized standard task through an injectable Docker provider.
    ///
    /// This boundary keeps provider preflight and every plan-derived phase policy
    /// observable without exposing shell command construction to the importer.
    pub fn execute_with_runtime(
        &self,
        runtime: &dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        agent_command: &[String],
        secrets: &dyn SecretProvider,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if package.execution_plan().is_multi_step() || plan.is_multi_step() {
            return Err(EvalExecutionError::UnsupportedMultiStep);
        }
        if !package.is_standard_directory() {
            return Err(EvalExecutionError::Materialization(
                "Docker execution requires a standard task directory".to_owned(),
            ));
        }
        preflight_docker(runtime, plan)?;
        if plan.compose().is_some() {
            return self.execute_compose_with_runtime(
                runtime,
                recipe,
                package,
                plan,
                agent_command,
                secrets,
            );
        }
        let environment = plan.environment();
        let verifier = plan.verifier();
        if !runtime.supports_phase_network_transitions()
            && (plan.agent().network() != environment.network()
                || verifier.phase().network() != verifier.environment().network())
        {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "phase network transition",
            ));
        }
        let environment_workdir = recipe.resolve_workdir(environment.workdir());
        validate_shared_verifier_workdir(runtime, plan, None, environment_workdir)?;
        let materialized_source = package.materialize_source()?;
        let (source_root, environment_root) =
            standard_task_roots(package, materialized_source.root())?;

        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let safe_suffix = package
            .source_digest()
            .as_str()
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .take(32)
            .collect::<String>();
        let run_id = NEXT_DOCKER_RUN_ID.fetch_add(1, Ordering::Relaxed);
        let image = docker_image_name(&safe_suffix, std::process::id(), run_id);
        let container = docker_container_name(&safe_suffix, std::process::id(), run_id);
        let mut containers = vec![container.clone()];

        let outcome = (|| {
            let baseline_network = network_lease(environment.network())?;
            let build_network = build_network_lease(environment.network())?;
            runtime.build(
                &DockerBuildRequest::new([
                    "build",
                    "--network",
                    build_network,
                    "--tag",
                    &image,
                    environment_root.to_string_lossy().as_ref(),
                ])
                .with_network_lease(build_network),
            )?;
            create_planned_container(
                runtime,
                &container,
                &image,
                ContainerWorkspace::at_workdir(workspace.path(), environment_workdir),
                environment,
                baseline_network,
                Some(package.instruction()),
            )?;
            runtime.start(&DockerStartRequest::new(&container))?;
            validate_shared_verifier_workdir(runtime, plan, Some(&container), environment_workdir)?;

            if let Some(healthcheck) = environment.healthcheck() {
                run_healthcheck(
                    self.clock.clone(),
                    runtime,
                    &container,
                    environment,
                    environment_workdir,
                    healthcheck,
                    baseline_network,
                    secrets,
                )?;
            }

            prepare_workdir(
                runtime,
                &container,
                environment,
                plan.agent(),
                environment_workdir,
                baseline_network,
            )?;
            execute_planned_phase(
                runtime,
                &container,
                EvalExecutionPhase::Agent,
                agent_command,
                environment,
                plan.agent(),
                environment_workdir,
                secrets,
            )?;
            let artifact_collection = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            let artifacts = collect_artifacts(
                runtime,
                &container,
                plan.artifacts(),
                artifact_collection.path(),
            )?;

            let verifier_container = if verifier.mode() == VerifierMode::Separate {
                let verifier_workspace = tempfile::tempdir()
                    .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
                fs::set_permissions(verifier_workspace.path(), fs::Permissions::from_mode(0o755))
                    .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
                transfer_artifacts(
                    artifact_collection.path(),
                    verifier_workspace.path(),
                    &artifacts,
                )?;
                let name = format!("{container}-verifier");
                let verifier_network = network_lease(verifier.environment().network())?;
                let verifier_workdir = recipe.resolve_workdir(verifier.environment().workdir());
                if !plan.artifacts().is_empty()
                    && let Some(workdir) = verifier_workdir
                {
                    validate_verifier_artifact_staging(workdir, plan.artifacts())?;
                }
                create_planned_container(
                    runtime,
                    &name,
                    &image,
                    ContainerWorkspace::at_workdir(verifier_workspace.path(), None),
                    verifier.environment(),
                    verifier_network,
                    None,
                )?;
                containers.push(name.clone());
                runtime.start(&DockerStartRequest::new(&name))?;
                if !plan.artifacts().is_empty() {
                    let effective_verifier_workdir = match verifier_workdir {
                        Some(workdir) => workdir.to_owned(),
                        None => runtime.container_workdir(&name)?,
                    };
                    validate_verifier_artifact_staging(
                        &effective_verifier_workdir,
                        plan.artifacts(),
                    )?;
                    transfer_verifier_artifacts(
                        runtime,
                        &name,
                        verifier_workspace.path(),
                        Some(&effective_verifier_workdir),
                        verifier_network,
                    )?;
                }
                if let Some(healthcheck) = verifier.environment().healthcheck() {
                    run_healthcheck(
                        self.clock.clone(),
                        runtime,
                        &name,
                        verifier.environment(),
                        verifier_workdir,
                        healthcheck,
                        verifier_network,
                        secrets,
                    )?;
                }
                Some((name, verifier_workspace))
            } else {
                None
            };
            let verifier_name = verifier_container
                .as_ref()
                .map_or(container.as_str(), |(name, _)| name.as_str());
            let verifier_network = network_lease(verifier.phase().network())?;
            prepare_verifier_files(runtime, verifier_name, verifier_network)?;
            runtime.copy(&DockerCopyRequest::new([
                "cp".to_owned(),
                format!("{}/.", source_root.join("tests").display()),
                format!("{verifier_name}:/tests"),
            ]))?;
            let verifier_workdir = recipe.resolve_workdir(verifier.environment().workdir());
            prepare_workdir(
                runtime,
                verifier_name,
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                verifier_network,
            )?;
            execute_planned_phase(
                runtime,
                verifier_name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                secrets,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            let reward = read_reward_with_runtime(runtime, verifier_name, &reward_workspace)?;
            Ok(LocalExecutionResult {
                artifacts,
                reward,
                verifier: package.source_digest(),
            })
        })();
        let cleanup = containers
            .into_iter()
            .rev()
            .fold(None, |first_error, container| {
                let removal = runtime
                    .remove(&DockerRemoveRequest::new([
                        "rm",
                        "--force",
                        "--volumes",
                        &container,
                    ]))
                    .err();
                first_error.or(removal)
            });
        match (outcome, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Some(error)) => Err(error),
            (Ok(result), None) => Ok(result),
        }
    }

    fn execute_compose_with_runtime(
        &self,
        runtime: &dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        agent_command: &[String],
        secrets: &dyn SecretProvider,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        let mut prepared = self.prepare_compose_lease(runtime, recipe, package, plan, secrets)?;
        let outcome = (|| {
            let step = plan
                .steps()
                .first()
                .ok_or(EvalExecutionError::InvalidRecipe("Compose benchmark step"))?;
            let mut session = ComposeStepSession::new(
                self.clock.clone(),
                runtime,
                recipe,
                prepared.source_root,
                prepared.verifier_prefix.clone(),
                &prepared.environment,
                &mut prepared.lease,
                secrets,
            );
            session.run_agent(step, agent_command)?;
            let artifacts = session.collect_artifacts(step)?;
            let reward = session.run_verifier(step, &artifacts)?;
            Ok(LocalExecutionResult {
                artifacts,
                reward,
                verifier: package.source_digest(),
            })
        })();
        let cleanup = prepared.lease.teardown();
        match (outcome, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(result), Ok(())) => Ok(result),
        }
    }

    fn execute_compose_multi_step_with_runtime(
        &self,
        runtime: &dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        agent_command: &[String],
        secrets: &dyn SecretProvider,
    ) -> Result<MultiStepExecutionResult, EvalExecutionError> {
        let mut prepared = self.prepare_compose_lease(runtime, recipe, package, plan, secrets)?;
        let outcome = (|| {
            let mut session = ComposeStepSession::new(
                self.clock.clone(),
                runtime,
                recipe,
                prepared.source_root,
                prepared.verifier_prefix.clone(),
                &prepared.environment,
                &mut prepared.lease,
                secrets,
            );
            execute_benchmark_steps(plan, agent_command, package.source_digest(), &mut session)
        })();
        let cleanup = prepared.lease.teardown();
        match (outcome, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(result), Ok(())) => Ok(result),
        }
    }

    fn prepare_compose_lease<'a>(
        &self,
        runtime: &'a dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        secrets: &dyn SecretProvider,
    ) -> Result<PreparedComposeLease<'a>, EvalExecutionError> {
        let compose = plan
            .compose()
            .ok_or(EvalExecutionError::InvalidRecipe("Compose project plan"))?;
        let compose_runtime =
            runtime
                .compose_runtime()
                .ok_or(EvalExecutionError::UnsupportedEnforcement(
                    "Docker Compose runtime",
                ))?;
        let mut environment = plan.environment().clone();
        environment.workdir = recipe
            .resolve_workdir(environment.workdir())
            .map(ToOwned::to_owned);
        resolve_environment(&environment, secrets)?;
        for step in plan.steps() {
            resolve_phase_environment(&environment, step.agent(), secrets)?;
            resolve_environment(step.verifier().environment(), secrets)?;
            resolve_phase_environment(
                step.verifier().environment(),
                step.verifier().phase(),
                secrets,
            )?;
        }
        let materialized = package.materialize_source()?;
        let (source_root, environment_root) = standard_task_roots(package, materialized.root())?;
        let source_root = source_root.to_path_buf();
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        fs::set_permissions(workspace.path(), fs::Permissions::from_mode(0o755))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let (image, _) = docker_run_names(package);
        let mut lease = ComposeProjectLease::reserve(
            compose_runtime,
            compose,
            package.source_digest().as_str(),
            source_root.to_string_lossy(),
            image.clone(),
        )?;
        let labels = lease.project().ownership_labels();
        let overlay = fs::read(source_root.join(compose.definition_path()))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let generated = preflight_compose_configuration(
            runtime,
            plan,
            &environment,
            &environment_root,
            lease.project().clone(),
            &source_root,
            &image,
            &labels,
            workspace.path(),
            &overlay,
        )?;
        fs::write(
            source_root.join("aiperf.generated.compose.yaml"),
            generated.into_bytes(),
        )
        .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        runtime.build(
            &DockerBuildRequest::new([
                "build",
                "--network",
                "default",
                "--tag",
                &image,
                environment_root.to_string_lossy().as_ref(),
            ])
            .with_network_lease("default"),
        )?;
        lease.start()?;
        let main = lease.main_service().clone();
        prepare_lease_workdir_for_user(
            &mut lease,
            &main,
            environment.user(),
            EvalExecutionPhase::Healthcheck,
            environment.workdir(),
        )?;
        if let Some(healthcheck) = environment.healthcheck() {
            run_lease_healthcheck(
                self.clock.clone(),
                &mut lease,
                &environment,
                healthcheck,
                secrets,
            )?;
        }
        Ok(PreparedComposeLease {
            _materialized: materialized,
            _workspace: workspace,
            source_root,
            verifier_prefix: lease.project().as_str().to_owned(),
            environment,
            lease,
        })
    }
}

fn multi_step_uses_clock_drive(plan: &BenchmarkExecutionPlan) -> bool {
    let has_phase_timeout = plan.steps().iter().any(|step| {
        step.agent().timeout().is_some() || step.verifier().phase().timeout().is_some()
    });
    let has_timed_healthcheck = std::iter::once(plan.environment())
        .chain(
            plan.steps()
                .iter()
                .map(|step| step.verifier().environment()),
        )
        .filter_map(super::EnvironmentPlan::healthcheck)
        .any(|healthcheck| {
            healthcheck.start_period().is_some()
                || healthcheck.start_interval().is_some()
                || healthcheck.interval().is_some()
                || healthcheck.timeout().is_some()
        });
    has_phase_timeout || has_timed_healthcheck
}

struct PreparedComposeLease<'a> {
    _materialized: crate::eval::import::MaterializedSource,
    _workspace: TempDir,
    source_root: std::path::PathBuf,
    verifier_prefix: String,
    environment: super::EnvironmentPlan,
    lease: ComposeProjectLease<'a>,
}

struct ComposeStepSession<'a> {
    clock: Rc<dyn Clock>,
    runtime: &'a dyn DockerRuntime,
    recipe: &'a HarborSandboxRecipe,
    source_root: std::path::PathBuf,
    verifier_prefix: String,
    environment: &'a super::EnvironmentPlan,
    lease: &'a mut dyn TaskEnvironmentLease,
    secrets: &'a dyn SecretProvider,
    artifact_collection: Option<TempDir>,
}

impl<'a> ComposeStepSession<'a> {
    fn new(
        clock: Rc<dyn Clock>,
        runtime: &'a dyn DockerRuntime,
        recipe: &'a HarborSandboxRecipe,
        source_root: std::path::PathBuf,
        verifier_prefix: String,
        environment: &'a super::EnvironmentPlan,
        lease: &'a mut dyn TaskEnvironmentLease,
        secrets: &'a dyn SecretProvider,
    ) -> Self {
        Self {
            clock,
            runtime,
            recipe,
            source_root,
            verifier_prefix,
            environment,
            lease,
            secrets,
            artifact_collection: None,
        }
    }

    fn run_separate_verifier(
        &mut self,
        step: &BenchmarkStepPlan,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<RewardDocument, EvalExecutionError> {
        let verifier = step.verifier();
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        fs::set_permissions(workspace.path(), fs::Permissions::from_mode(0o755))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let collection =
            self.artifact_collection
                .as_ref()
                .ok_or(EvalExecutionError::InvalidRecipe(
                    "Compose artifact collection",
                ))?;
        transfer_artifacts(collection.path(), workspace.path(), artifacts)?;
        let image = self.lease.main_image_id()?.to_owned();
        let name = format!("{}-verifier-{}", self.verifier_prefix, step.name());
        let verifier_network = network_lease(verifier.environment().network())?;
        let verifier_workdir = self
            .recipe
            .resolve_workdir(verifier.environment().workdir());
        if let Some(workdir) = verifier_workdir {
            validate_verifier_artifact_staging(workdir, step.artifacts())?;
        }
        let outcome = (|| {
            create_planned_container(
                self.runtime,
                &name,
                &image,
                ContainerWorkspace::at_workdir(workspace.path(), None),
                verifier.environment(),
                verifier_network,
                None,
            )?;
            self.runtime.start(&DockerStartRequest::new(&name))?;
            let effective_workdir = match verifier_workdir {
                Some(workdir) => workdir.to_owned(),
                None => self.runtime.container_workdir(&name)?,
            };
            validate_verifier_artifact_staging(&effective_workdir, step.artifacts())?;
            transfer_verifier_artifacts(
                self.runtime,
                &name,
                workspace.path(),
                Some(&effective_workdir),
                verifier_network,
            )?;
            prepare_workdir(
                self.runtime,
                &name,
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                verifier_network,
            )?;
            if let Some(healthcheck) = verifier.environment().healthcheck() {
                run_healthcheck(
                    self.clock.clone(),
                    self.runtime,
                    &name,
                    verifier.environment(),
                    verifier_workdir,
                    healthcheck,
                    verifier_network,
                    self.secrets,
                )?;
            }
            prepare_verifier_files(self.runtime, &name, verifier_network)?;
            self.runtime.copy(&DockerCopyRequest::new([
                "cp".to_owned(),
                format!(
                    "{}/.",
                    self.source_root.join(step.verifier_test_root()).display()
                ),
                format!("{name}:/tests"),
            ]))?;
            execute_planned_phase(
                self.runtime,
                &name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                self.secrets,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            read_reward_with_runtime(self.runtime, &name, &reward_workspace)
        })();
        let cleanup = self.runtime.remove(&DockerRemoveRequest::new([
            "rm",
            "--force",
            "--volumes",
            &name,
        ]));
        match (outcome, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(reward), Ok(())) => Ok(reward),
        }
    }

    fn run_shared_verifier(
        &mut self,
        step: &BenchmarkStepPlan,
    ) -> Result<RewardDocument, EvalExecutionError> {
        let verifier = step.verifier();
        let main = self.lease.main_service().clone();
        self.lease.exec(ServiceExecRequest {
            service: &main,
            arguments: &[
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "rm -rf /tests /logs/verifier && mkdir -p /logs/verifier && chmod 0777 /logs/verifier"
                    .to_owned(),
            ],
            public_environment: Default::default(),
            secret_environment: Default::default(),
            phase: EvalExecutionPhase::Verifier,
            user: Some("root"),
            workdir: None,
            network_lease: "default",
            deadline: None,
        })?;
        self.lease.copy_into(
            &main,
            &format!(
                "{}/.",
                self.source_root.join(step.verifier_test_root()).display()
            ),
            "/tests",
        )?;
        let resolved =
            resolve_phase_environment(verifier.environment(), verifier.phase(), self.secrets)?;
        let verifier_workdir = self
            .recipe
            .resolve_workdir(verifier.environment().workdir());
        prepare_lease_workdir(
            self.lease,
            &main,
            verifier.environment(),
            verifier.phase(),
            EvalExecutionPhase::Verifier,
            verifier_workdir,
        )?;
        self.lease.exec(ServiceExecRequest {
            service: &main,
            arguments: &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
            public_environment: resolved.public().clone(),
            secret_environment: resolved.secrets().clone(),
            phase: EvalExecutionPhase::Verifier,
            user: verifier.phase().user().or(verifier.environment().user()),
            workdir: verifier_workdir,
            network_lease: "default",
            deadline: verifier.phase().timeout(),
        })?;
        read_reward_from_lease(self.lease, &main)
    }
}

impl BenchmarkStepSession for ComposeStepSession<'_> {
    fn run_agent(
        &mut self,
        step: &BenchmarkStepPlan,
        command: &[String],
    ) -> Result<(), EvalExecutionError> {
        if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
            return Err(EvalExecutionError::InvalidCommand);
        }
        let resolved = resolve_phase_environment(self.environment, step.agent(), self.secrets)?;
        let mut public_environment = resolved.public().clone();
        let mut secret_environment = resolved.secrets().clone();
        public_environment.insert(
            "AIPERF_EVAL_INSTRUCTION".to_owned(),
            step.instruction().to_owned(),
        );
        secret_environment.remove("AIPERF_EVAL_INSTRUCTION");
        let workdir = self.recipe.resolve_workdir(self.environment.workdir());
        let main = self.lease.main_service().clone();
        prepare_lease_workdir(
            self.lease,
            &main,
            self.environment,
            step.agent(),
            EvalExecutionPhase::Agent,
            workdir,
        )?;
        self.lease.exec(ServiceExecRequest {
            service: &main,
            arguments: command,
            public_environment,
            secret_environment,
            phase: EvalExecutionPhase::Agent,
            user: step.agent().user().or(self.environment.user()),
            workdir,
            network_lease: "default",
            deadline: step.agent().timeout(),
        })
    }

    fn collect_artifacts(
        &mut self,
        step: &BenchmarkStepPlan,
    ) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
        self.artifact_collection = None;
        let collection = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let deadline = Deadline::from_timeout(self.clock.clone(), step.collection_timeout());
        let has_terminal_evidence = !step.collect_hooks().is_empty()
            || step
                .artifacts()
                .iter()
                .any(|artifact| artifact.service() != "main");
        let artifacts = if has_terminal_evidence {
            collect_service_evidence(
                self.lease,
                step.collect_hooks(),
                step.artifacts(),
                collection.path(),
                deadline,
            )?
        } else {
            collect_service_artifacts(self.lease, step.artifacts(), collection.path(), deadline)?
        };
        self.artifact_collection = Some(collection);
        Ok(artifacts)
    }

    fn run_verifier(
        &mut self,
        step: &BenchmarkStepPlan,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<RewardDocument, EvalExecutionError> {
        if step.verifier().mode() == VerifierMode::Shared {
            let result = self.run_shared_verifier(step);
            self.artifact_collection = None;
            return result;
        }
        let has_non_main_evidence = step
            .artifacts()
            .iter()
            .any(|artifact| artifact.service() != "main")
            || step
                .collect_hooks()
                .iter()
                .any(|hook| hook.service().as_str() != "main");
        if has_non_main_evidence {
            self.lease.teardown()?;
        }
        let result = self.run_separate_verifier(step, artifacts);
        self.artifact_collection = None;
        result
    }
}

struct DockerStepSession<'a> {
    clock: Rc<dyn Clock>,
    runtime: &'a dyn DockerRuntime,
    recipe: &'a HarborSandboxRecipe,
    source_root: &'a std::path::Path,
    environment: &'a super::EnvironmentPlan,
    image: &'a str,
    agent_container: &'a str,
    secrets: &'a dyn SecretProvider,
    containers: &'a mut Vec<String>,
    artifact_collection: Option<TempDir>,
}

impl BenchmarkStepSession for DockerStepSession<'_> {
    fn run_agent(
        &mut self,
        step: &BenchmarkStepPlan,
        command: &[String],
    ) -> Result<(), EvalExecutionError> {
        let baseline_network = network_lease(self.environment.network())?;
        let workdir = self.recipe.resolve_workdir(self.environment.workdir());
        prepare_workdir(
            self.runtime,
            self.agent_container,
            self.environment,
            step.agent(),
            workdir,
            baseline_network,
        )?;
        if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
            return Err(EvalExecutionError::InvalidCommand);
        }
        let resolved = resolve_phase_environment(self.environment, step.agent(), self.secrets)?;
        let mut public_environment = resolved.public().clone();
        let mut secret_environment = resolved.secrets().clone();
        public_environment.insert(
            "AIPERF_EVAL_INSTRUCTION".to_owned(),
            step.instruction().to_owned(),
        );
        secret_environment.remove("AIPERF_EVAL_INSTRUCTION");
        let step_network = network_lease(step.agent().network())?;
        self.runtime.exec(
            &DockerExecRequest::new(
                self.agent_container,
                command.iter().cloned(),
                public_environment,
                secret_environment,
            )
            .with_phase(
                EvalExecutionPhase::Agent,
                step.agent().user().or(self.environment.user()),
                workdir,
                step_network,
                step.agent().timeout(),
            ),
        )
    }

    fn collect_artifacts(
        &mut self,
        step: &BenchmarkStepPlan,
    ) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
        self.artifact_collection = None;
        let collection = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let artifacts = collect_artifacts(
            self.runtime,
            self.agent_container,
            step.artifacts(),
            collection.path(),
        )?;
        self.artifact_collection = Some(collection);
        Ok(artifacts)
    }

    fn run_verifier(
        &mut self,
        step: &BenchmarkStepPlan,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<RewardDocument, EvalExecutionError> {
        let verifier = step.verifier();
        let verifier_workspace = if verifier.mode() == VerifierMode::Separate {
            let workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            fs::set_permissions(workspace.path(), fs::Permissions::from_mode(0o755))
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            let collection =
                self.artifact_collection
                    .as_ref()
                    .ok_or(EvalExecutionError::InvalidRecipe(
                        "multi-step artifact collection",
                    ))?;
            transfer_artifacts(collection.path(), workspace.path(), artifacts)?;
            Some(workspace)
        } else {
            None
        };
        let verifier_name = if let Some(workspace) = verifier_workspace.as_ref() {
            let name = format!("{}-verifier-{}", self.agent_container, step.name());
            let verifier_network = network_lease(verifier.environment().network())?;
            let verifier_workdir = self
                .recipe
                .resolve_workdir(verifier.environment().workdir());
            if let Some(workdir) = verifier_workdir {
                validate_verifier_artifact_staging(workdir, step.artifacts())?;
            }
            create_planned_container(
                self.runtime,
                &name,
                self.image,
                ContainerWorkspace::at_workdir(workspace.path(), None),
                verifier.environment(),
                verifier_network,
                None,
            )?;
            self.containers.push(name.clone());
            self.runtime.start(&DockerStartRequest::new(&name))?;
            let effective_verifier_workdir = match verifier_workdir {
                Some(workdir) => workdir.to_owned(),
                None => self.runtime.container_workdir(&name)?,
            };
            validate_verifier_artifact_staging(&effective_verifier_workdir, step.artifacts())?;
            transfer_verifier_artifacts(
                self.runtime,
                &name,
                workspace.path(),
                Some(&effective_verifier_workdir),
                verifier_network,
            )?;
            if let Some(healthcheck) = verifier.environment().healthcheck() {
                run_healthcheck(
                    self.clock.clone(),
                    self.runtime,
                    &name,
                    verifier.environment(),
                    verifier_workdir,
                    healthcheck,
                    verifier_network,
                    self.secrets,
                )?;
            }
            name
        } else {
            self.agent_container.to_owned()
        };
        let verifier_network = network_lease(verifier.phase().network())?;
        let outcome = (|| {
            prepare_verifier_files(self.runtime, &verifier_name, verifier_network)?;
            self.runtime.copy(&DockerCopyRequest::new([
                "cp".to_owned(),
                format!(
                    "{}/.",
                    self.source_root.join(step.verifier_test_root()).display()
                ),
                format!("{verifier_name}:/tests"),
            ]))?;
            let verifier_workdir = self
                .recipe
                .resolve_workdir(verifier.environment().workdir());
            prepare_workdir(
                self.runtime,
                &verifier_name,
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                verifier_network,
            )?;
            execute_planned_phase(
                self.runtime,
                &verifier_name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                self.secrets,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            read_reward_with_runtime(self.runtime, &verifier_name, &reward_workspace)
        })();
        let cleanup = if verifier.mode() == VerifierMode::Shared {
            clear_verifier_files(self.runtime, &verifier_name, verifier_network)
        } else {
            Ok(())
        };
        self.artifact_collection = None;
        match (outcome, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(reward), Ok(())) => Ok(reward),
        }
    }
}

fn validate_verifier_artifact_staging(
    workdir: &str,
    artifacts: &[super::ArtifactSpec],
) -> Result<(), EvalExecutionError> {
    let collision = verifier_artifact_target_collision(workdir, artifacts)
        .map_err(EvalExecutionError::InvalidWorkspace)?;
    if let Some(target) = collision {
        return Err(EvalExecutionError::InvalidWorkspace(format!(
            "artifact destination overlaps a reserved verifier path: {target:?}"
        )));
    }
    Ok(())
}

fn validate_shared_verifier_workdir(
    runtime: &dyn DockerRuntime,
    plan: &BenchmarkExecutionPlan,
    container: Option<&str>,
    explicit_workdir: Option<&str>,
) -> Result<(), EvalExecutionError> {
    if !plan.uses_shared_verifier() {
        return Ok(());
    }
    if let Some(workdir) = explicit_workdir {
        return validate_shared_verifier_workdir_path(workdir);
    }
    let Some(container) = container else {
        return Ok(());
    };
    let workdir = runtime.container_workdir(container)?;
    validate_shared_verifier_workdir_path(&workdir)
}

fn validate_shared_verifier_workdir_path(workdir: &str) -> Result<(), EvalExecutionError> {
    if shared_workdir_conflicts_reserved_verifier_path(workdir)
        .map_err(EvalExecutionError::InvalidWorkspace)?
    {
        return Err(EvalExecutionError::InvalidWorkspace(format!(
            "shared verifier workdir occupies a reserved verifier path: {workdir:?}"
        )));
    }
    Ok(())
}

fn standard_task_roots<'a>(
    package: &HarborTaskPackage,
    source_root: &'a std::path::Path,
) -> Result<(&'a std::path::Path, std::path::PathBuf), EvalExecutionError> {
    if !package.is_standard_directory() {
        return Err(EvalExecutionError::Materialization(
            "Docker execution requires a standard task directory".to_owned(),
        ));
    }
    let environment_root = source_root.join("environment");
    if !environment_root.join("Dockerfile").is_file() {
        return Err(EvalExecutionError::Materialization(
            "standard task is missing environment/Dockerfile".to_owned(),
        ));
    }
    Ok((source_root, environment_root))
}

fn docker_run_names(package: &HarborTaskPackage) -> (String, String) {
    let safe_suffix = package
        .source_digest()
        .as_str()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .take(32)
        .collect::<String>();
    let run_id = NEXT_DOCKER_RUN_ID.fetch_add(1, Ordering::Relaxed);
    (
        docker_image_name(&safe_suffix, std::process::id(), run_id),
        docker_container_name(&safe_suffix, std::process::id(), run_id),
    )
}

fn finish_with_cleanup<T>(
    runtime: &dyn DockerRuntime,
    containers: Vec<String>,
    outcome: Result<T, EvalExecutionError>,
) -> Result<T, EvalExecutionError> {
    let cleanup = containers
        .into_iter()
        .rev()
        .fold(None, |first_error, container| {
            let removal = runtime
                .remove(&DockerRemoveRequest::new([
                    "rm",
                    "--force",
                    "--volumes",
                    &container,
                ]))
                .err();
            first_error.or(removal)
        });
    match (outcome, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(_), Some(error)) => Err(error),
        (Ok(result), None) => Ok(result),
    }
}

static NEXT_DOCKER_RUN_ID: AtomicU64 = AtomicU64::new(1);

fn run_healthcheck(
    clock: Rc<dyn Clock>,
    runtime: &dyn DockerRuntime,
    container: &str,
    environment: &super::EnvironmentPlan,
    workdir: Option<&str>,
    healthcheck: &super::HealthcheckPlan,
    network_lease: &str,
    secrets: &dyn SecretProvider,
) -> Result<(), EvalExecutionError> {
    if let Some(start_period) = healthcheck.start_period() {
        sleep_with_clock(clock.clone(), start_period, container)?;
    }
    let retries = healthcheck.retries().unwrap_or(1);
    let health_environment = resolve_environment(environment, secrets)?;
    let mut last_error = None;
    for attempt in 0..retries {
        let request = DockerExecRequest::new(
            container,
            healthcheck.command().iter().cloned(),
            health_environment.public().clone(),
            health_environment.secrets().clone(),
        )
        .with_phase(
            EvalExecutionPhase::Healthcheck,
            environment.user(),
            workdir,
            network_lease,
            healthcheck.timeout(),
        );
        match runtime.exec(&request) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        if attempt + 1 < retries {
            let interval = if attempt == 0 {
                healthcheck.start_interval().or(healthcheck.interval())
            } else {
                healthcheck.interval().or(healthcheck.start_interval())
            };
            if let Some(interval) = interval {
                sleep_with_clock(clock.clone(), interval, container)?;
            }
        }
    }
    let reason = last_error.map_or_else(
        || "healthcheck exhausted without an execution result".to_owned(),
        |error| error.to_string(),
    );
    Err(EvalExecutionError::Unhealthy(reason))
}

fn run_lease_healthcheck(
    clock: Rc<dyn Clock>,
    lease: &mut dyn TaskEnvironmentLease,
    environment: &super::EnvironmentPlan,
    healthcheck: &super::HealthcheckPlan,
    secrets: &dyn SecretProvider,
) -> Result<(), EvalExecutionError> {
    let main = lease.main_service().clone();
    if let Some(start_period) = healthcheck.start_period() {
        sleep_with_clock(clock.clone(), start_period, main.as_str())?;
    }
    let retries = healthcheck.retries().unwrap_or(1);
    let health_environment = resolve_environment(environment, secrets)?;
    let mut last_error = None;
    for attempt in 0..retries {
        match lease.exec(ServiceExecRequest {
            service: &main,
            arguments: healthcheck.command(),
            public_environment: health_environment.public().clone(),
            secret_environment: health_environment.secrets().clone(),
            phase: EvalExecutionPhase::Healthcheck,
            user: environment.user(),
            workdir: environment.workdir(),
            network_lease: "default",
            deadline: healthcheck.timeout(),
        }) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        if attempt + 1 < retries {
            let interval = if attempt == 0 {
                healthcheck.start_interval().or(healthcheck.interval())
            } else {
                healthcheck.interval().or(healthcheck.start_interval())
            };
            if let Some(interval) = interval {
                sleep_with_clock(clock.clone(), interval, main.as_str())?;
            }
        }
    }
    Err(EvalExecutionError::Unhealthy(last_error.map_or_else(
        || "healthcheck exhausted without an execution result".to_owned(),
        |error| error.to_string(),
    )))
}

fn prepare_lease_workdir(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    environment: &super::EnvironmentPlan,
    phase: &super::PhasePlan,
    execution_phase: EvalExecutionPhase,
    workdir: Option<&str>,
) -> Result<(), EvalExecutionError> {
    prepare_lease_workdir_for_user(
        lease,
        service,
        phase.user().or(environment.user()),
        execution_phase,
        workdir,
    )
}

fn prepare_lease_workdir_for_user(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    user: Option<&str>,
    execution_phase: EvalExecutionPhase,
    workdir: Option<&str>,
) -> Result<(), EvalExecutionError> {
    let Some(workdir) = workdir else {
        return Ok(());
    };
    let Some(user) = user else {
        return Ok(());
    };
    if user == "root" {
        return Ok(());
    }
    lease.exec(ServiceExecRequest {
        service,
        arguments: &[
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            format!("mkdir -p {workdir} && chown {user} {workdir} && su -s /bin/sh {user} -c 'test -w {workdir}'"),
        ],
        public_environment: Default::default(),
        secret_environment: Default::default(),
        phase: execution_phase,
        user: Some("root"),
        workdir: Some(workdir),
        network_lease: "default",
        deadline: None,
    })
}

fn sleep_with_clock(
    clock: Rc<dyn Clock>,
    duration: Duration,
    container: &str,
) -> Result<(), EvalExecutionError> {
    let completed = Rc::new(Cell::new(false));
    let completion = completed.clone();
    let delay_ns = duration.as_nanos().min(i64::MAX as u128) as i64;
    let outcome = clock.clone().drive(Box::pin(async move {
        clock.sleep(delay_ns).await;
        completion.set(true);
    }));
    if outcome.deadlocked || !completed.get() {
        return Err(EvalExecutionError::TerminalUncertainty {
            phase: EvalExecutionPhase::Healthcheck,
            container: container.to_owned(),
            reason: "execution clock reached quiescence during healthcheck scheduling".to_owned(),
        });
    }
    Ok(())
}

struct HostSecretProvider;

impl SecretProvider for HostSecretProvider {
    fn resolve(&self, name: &super::EnvName) -> Result<super::SecretValue, EvalExecutionError> {
        std::env::var(name)
            .map(super::SecretValue::new)
            .map_err(|_| EvalExecutionError::MissingSecret(name.clone()))
    }
}

struct DockerCliRuntime {
    clock: Rc<dyn Clock>,
}

impl DockerRuntime for DockerCliRuntime {
    fn capabilities(&self) -> super::ProviderCapabilities {
        super::ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_resource_limits()
            .with_users()
            .with_phase_env()
            .with_workdir()
            .with_phase_timeouts()
            .with_separate_verifier()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
            .with_compose_project()
            .with_compose_config()
            .with_service_exec()
            .with_service_archive()
            .with_service_stop()
    }

    fn compose_runtime(&self) -> Option<&dyn DockerComposeRuntime> {
        Some(self)
    }

    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        docker(
            request.public_arguments().iter().map(String::as_str),
            "build task environment",
        )
        .map(|_| ())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        if request.network_lease() == Some(PUBLIC_NETWORK_LEASE) {
            ensure_public_network()?;
        }
        docker(
            request.public_arguments().iter().map(String::as_str),
            "create task container",
        )
        .map(|_| ())
    }

    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        docker(["start", request.container()], "start task container").map(|_| ())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        let mut arguments = vec!["exec".to_owned()];
        for (name, value) in request.public_environment() {
            arguments.extend(["--env".to_owned(), format!("{name}={value}")]);
        }
        for (name, value) in request.secret_environment() {
            arguments.extend(["--env".to_owned(), format!("{name}={}", value.exposed())]);
        }
        if let Some(user) = request.user() {
            arguments.extend(["--user".to_owned(), user.to_owned()]);
        }
        if let Some(workdir) = request.workdir() {
            arguments.extend(["--workdir".to_owned(), workdir.to_owned()]);
        }
        arguments.push(request.container().to_owned());
        arguments.extend(request.public_arguments().iter().cloned());
        if let Some(timeout) = request.deadline() {
            let references = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            return docker_exec_bounded(
                self.clock.clone(),
                request.container(),
                &references,
                "run planned Docker phase",
                request.phase(),
                timeout,
            );
        }
        let output = Command::new("docker")
            .args(&arguments)
            .output()
            .map_err(|_| {
                EvalExecutionError::ProcessSpawn("docker run planned Docker phase".to_owned())
            })?;
        if output.status.success() {
            Ok(())
        } else {
            Err(EvalExecutionError::ProcessFailure(format!(
                "docker run planned Docker phase: {}",
                redact_secret_values(
                    &String::from_utf8_lossy(&output.stderr),
                    request.secret_environment(),
                )
            )))
        }
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        docker(
            request.public_arguments().iter().map(String::as_str),
            "copy Docker files",
        )
        .map(|_| ())
    }

    fn container_workdir(&self, container: &str) -> Result<String, EvalExecutionError> {
        let output = docker(
            [
                "container",
                "inspect",
                "--format",
                "{{.Config.WorkingDir}}",
                container,
            ],
            "inspect task container workdir",
        )?;
        let workdir = std::str::from_utf8(&output)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?
            .trim();
        Ok(if workdir.is_empty() {
            "/".to_owned()
        } else {
            workdir.to_owned()
        })
    }

    fn copy_archive(
        &self,
        container: &str,
        source: &str,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.copy_archive_with_deadline(container, source, None)
    }

    fn copy_archive_bounded(
        &self,
        container: &str,
        source: &str,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.copy_archive_to_file_bounded(container, source, deadline)
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        let removal = if let Some(timeout) = request.deadline() {
            if tokio::runtime::Handle::try_current().is_ok() {
                return Err(EvalExecutionError::RuntimeContext(
                    "bounded Docker lease cleanup",
                ));
            }
            docker_remove_bounded(
                self.clock.clone(),
                request
                    .public_arguments()
                    .iter()
                    .map(String::as_str)
                    .collect(),
                timeout,
            )
        } else {
            docker(
                request.public_arguments().iter().map(String::as_str),
                "remove Docker lease",
            )
            .map(|_| ())
        };
        match removal {
            Ok(_) => Ok(()),
            Err(EvalExecutionError::ProcessFailure(error))
                if reports_absent_container(error.as_bytes()) =>
            {
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl DockerComposeRuntime for DockerCliRuntime {
    fn compose_config(
        &self,
        request: &DockerComposeConfigRequest,
    ) -> Result<Vec<u8>, EvalExecutionError> {
        let generated = request
            .project_directory()
            .join("aiperf.generated.compose.yaml");
        fs::write(&generated, request.generated_definition())
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let arguments = compose_arguments(
            request.project(),
            request.project_directory(),
            &generated,
            request.overlay_definition(),
        );
        compose_command(arguments.into_iter().chain([
            "config".to_owned(),
            "--format".to_owned(),
            "json".to_owned(),
            "--no-interpolate".to_owned(),
        ]))
    }

    fn compose_build(&self, request: &DockerComposeBuildRequest) -> Result<(), EvalExecutionError> {
        let arguments = compose_project_arguments(request.project(), request.project_directory())
            .into_iter()
            .chain(["build".to_owned()]);
        compose_command_bounded(
            self.clock.clone(),
            arguments,
            request.project().as_str(),
            request.deadline(),
        )
    }

    fn compose_up(&self, request: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
        let arguments = compose_project_arguments(request.project(), request.project_directory())
            .into_iter()
            .chain(["up".to_owned(), "--detach".to_owned(), "--wait".to_owned()]);
        compose_command_bounded(
            self.clock.clone(),
            arguments,
            request.project().as_str(),
            request.deadline(),
        )
    }

    fn compose_exec(&self, request: &DockerComposeExecRequest) -> Result<(), EvalExecutionError> {
        let container = compose_service_container(request.project(), request.service())?;
        self.exec(
            &DockerExecRequest::new(
                container,
                request.public_arguments().iter().cloned(),
                request.public_environment().clone(),
                request.secret_environment().clone(),
            )
            .with_phase(
                request.phase(),
                request.user(),
                request.workdir(),
                "default",
                request.deadline(),
            ),
        )
    }

    fn compose_copy_archive(
        &self,
        request: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let container = compose_service_container(request.project(), request.service())?;
        self.copy_archive(&container, request.source())
    }

    fn compose_copy_archive_bounded(
        &self,
        request: &DockerComposeArchiveRequest,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let container = compose_service_container(request.project(), request.service())?;
        self.copy_archive_to_file_bounded(&container, request.source(), deadline)
    }

    fn compose_copy_into(
        &self,
        request: &DockerComposeCopyRequest,
    ) -> Result<(), EvalExecutionError> {
        let container = compose_service_container(request.project(), request.service())?;
        self.copy(&DockerCopyRequest::new([
            "cp".to_owned(),
            request.source().to_owned(),
            format!("{container}:{}", request.destination()),
        ]))
    }

    fn compose_stop_service(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError> {
        self.compose_stop_service_bounded(request)
    }

    fn compose_stop_service_bounded(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError> {
        let container = compose_service_container(request.project(), request.service())?;
        let arguments = ["stop".to_owned(), container];
        docker_command_bounded(
            self.clock.clone(),
            arguments,
            request.project().as_str(),
            request.deadline(),
        )
    }

    fn compose_down(&self, request: &DockerComposeDownRequest) -> Result<(), EvalExecutionError> {
        let mut arguments =
            compose_project_arguments(request.project(), request.project_directory());
        arguments.extend([
            "down".to_owned(),
            "--volumes".to_owned(),
            "--remove-orphans".to_owned(),
            "--timeout".to_owned(),
            request.container_grace().as_secs().to_string(),
        ]);
        compose_command_bounded(
            self.clock.clone(),
            arguments.into_iter(),
            request.project().as_str(),
            request.deadline(),
        )
    }

    fn compose_owned_resources(
        &self,
        project: &ComposeProjectId,
    ) -> Result<super::OwnedComposeResources, EvalExecutionError> {
        let filters = compose_ownership_filters(project);
        let resources = |kind: &str, extra: &[&str]| -> Result<Vec<String>, EvalExecutionError> {
            let mut arguments = vec![kind.to_owned(), "ls".to_owned(), "-q".to_owned()];
            arguments.extend(extra.iter().map(|value| (*value).to_owned()));
            for filter in &filters {
                arguments.extend(["--filter".to_owned(), filter.clone()]);
            }
            let output = docker(
                arguments.iter().map(String::as_str),
                "list Compose resources",
            )?;
            Ok(String::from_utf8_lossy(&output)
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_owned)
                .collect())
        };
        Ok(super::OwnedComposeResources::new(
            resources("container", &["--all"])?,
            resources("network", &[])?,
            resources("volume", &[])?,
        ))
    }
}

fn compose_project_arguments(
    project: &ComposeProjectId,
    directory: &std::path::Path,
) -> Vec<String> {
    let generated = directory.join("aiperf.generated.compose.yaml");
    let overlay = directory.join("environment/docker-compose.yaml");
    compose_arguments(project, directory, &generated, &overlay)
}

fn compose_arguments(
    project: &ComposeProjectId,
    directory: &std::path::Path,
    generated: &std::path::Path,
    overlay: &std::path::Path,
) -> Vec<String> {
    vec![
        "compose".to_owned(),
        "--project-name".to_owned(),
        project.as_str().to_owned(),
        "--project-directory".to_owned(),
        directory.to_string_lossy().into_owned(),
        "--file".to_owned(),
        generated.to_string_lossy().into_owned(),
        "--file".to_owned(),
        overlay.to_string_lossy().into_owned(),
    ]
}

fn compose_command(
    arguments: impl IntoIterator<Item = String>,
) -> Result<Vec<u8>, EvalExecutionError> {
    let output = Command::new("docker")
        .env("COMPOSE_DISABLE_ENV_FILE", "1")
        .args(arguments)
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker compose command".to_owned()))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(EvalExecutionError::ProcessFailure(format!(
            "docker compose command: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn compose_command_bounded(
    clock: Rc<dyn Clock>,
    arguments: impl IntoIterator<Item = String>,
    target: &str,
    deadline: Option<Duration>,
) -> Result<(), EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    if let Some(deadline) = deadline {
        return docker_command_bounded(clock, arguments, target, Some(deadline));
    }
    compose_command(arguments).map(|_| ())
}

fn docker_command_bounded(
    clock: Rc<dyn Clock>,
    arguments: impl IntoIterator<Item = String>,
    target: &str,
    deadline: Option<Duration>,
) -> Result<(), EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let Some(deadline) = deadline else {
        return docker(arguments.iter().map(String::as_str), "run Docker command").map(|_| ());
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(EvalExecutionError::RuntimeContext(
            "bounded Docker Compose operation",
        ));
    }
    let child = Command::new("docker")
        .args(&arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker compose command".to_owned()))?;
    let mut process = DockerExecChild { child };
    let mut no_remove = |_target: &str| Ok(());
    drive_docker_exec(
        clock,
        &mut process,
        target,
        EvalExecutionPhase::CollectionHook,
        deadline,
        &mut no_remove,
    )
    .map_err(|error| match error {
        EvalExecutionError::ProcessFailure(reason) => EvalExecutionError::ProcessFailure(format!(
            "bounded docker {}: {reason}",
            arguments.join(" ")
        )),
        error => error,
    })
}

fn compose_service_container(
    project: &ComposeProjectId,
    service: &super::ComposeServiceName,
) -> Result<String, EvalExecutionError> {
    let mut arguments = vec![
        "container".to_owned(),
        "ls".to_owned(),
        "--all".to_owned(),
        "--quiet".to_owned(),
    ];
    for filter in compose_ownership_filters(project) {
        arguments.extend(["--filter".to_owned(), filter]);
    }
    arguments.extend([
        "--filter".to_owned(),
        format!("label=com.docker.compose.service={}", service.as_str()),
    ]);
    let output = docker(
        arguments.iter().map(String::as_str),
        "find Compose service container",
    )?;
    let output = String::from_utf8_lossy(&output);
    let mut containers = output.lines().filter(|line| !line.trim().is_empty());
    let container = containers
        .next()
        .ok_or_else(|| EvalExecutionError::ContainerTeardown {
            container: project.as_str().to_owned(),
            reason: format!("Compose service {:?} is absent", service.as_str()),
        })?;
    if containers.next().is_some() {
        return Err(EvalExecutionError::ContainerTeardown {
            container: project.as_str().to_owned(),
            reason: format!(
                "Compose service {:?} has ambiguous containers",
                service.as_str()
            ),
        });
    }
    Ok(container.to_owned())
}

fn compose_ownership_filters(project: &ComposeProjectId) -> Vec<String> {
    let mut filters = vec![format!(
        "label=com.docker.compose.project={}",
        project.as_str()
    )];
    for (name, value) in project.ownership_labels() {
        filters.push(format!("label={name}={value}"));
    }
    filters
}

impl DockerCliRuntime {
    fn copy_archive_with_deadline(
        &self,
        container: &str,
        source: &str,
        _: Option<Duration>,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let mut child = Command::new("docker")
            .args(["cp", &format!("{container}:{source}"), "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| {
                EvalExecutionError::ProcessSpawn("docker collect artifact archive".to_owned())
            })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            EvalExecutionError::ArtifactCollection(
                "docker collect artifact archive did not provide stdout".to_owned(),
            )
        })?;
        Ok(Box::new(DockerArchiveReader {
            child,
            stdout,
            is_complete: false,
        }))
    }

    fn copy_archive_to_file_bounded(
        &self,
        container: &str,
        source: &str,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let mut archive = NamedTempFile::new().map_err(|error| {
            EvalExecutionError::ArtifactCollection(format!(
                "could not allocate bounded artifact archive: {error}"
            ))
        })?;
        let output = archive.as_file().try_clone().map_err(|error| {
            EvalExecutionError::ArtifactCollection(format!(
                "could not retain bounded artifact archive: {error}"
            ))
        })?;
        let child = Command::new("docker")
            .args(["cp", &format!("{container}:{source}"), "-"])
            .stdout(Stdio::from(output))
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| {
                EvalExecutionError::ProcessSpawn("docker collect artifact archive".to_owned())
            })?;
        let mut process = DockerExecChild { child };
        let mut no_remove = |_target: &str| Ok(());
        drive_docker_exec(
            self.clock.clone(),
            &mut process,
            container,
            EvalExecutionPhase::CollectionHook,
            deadline,
            &mut no_remove,
        )
        .map_err(|error| {
            EvalExecutionError::ArtifactCollection(format!(
                "bounded archive {container}:{source}: {error}"
            ))
        })?;
        archive
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        Ok(Box::new(BoundedArchiveReader { archive }))
    }
}

fn docker_remove_bounded(
    clock: Rc<dyn Clock>,
    arguments: Vec<&str>,
    timeout: Duration,
) -> Result<(), EvalExecutionError> {
    let target = arguments.last().copied().unwrap_or("Docker lease");
    let stderr = tempfile::tempfile().map_err(|_| EvalExecutionError::ContainerTeardown {
        container: target.to_owned(),
        reason: "could not allocate bounded Docker removal diagnostics".to_owned(),
    })?;
    let mut stderr_reader =
        stderr
            .try_clone()
            .map_err(|_| EvalExecutionError::ContainerTeardown {
                container: target.to_owned(),
                reason: "could not retain bounded Docker removal diagnostics".to_owned(),
            })?;
    let child = Command::new("docker")
        .args(&arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "could not start bounded Docker removal".to_owned(),
        })?;
    let mut process = DockerExecChild { child };
    let mut no_remove = |_target: &str| Ok(());
    let outcome = drive_docker_exec(
        clock,
        &mut process,
        target,
        EvalExecutionPhase::CollectionHook,
        timeout,
        &mut no_remove,
    );
    let mut diagnostics = Vec::new();
    stderr_reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "could not read bounded Docker removal diagnostics".to_owned(),
        })?;
    stderr_reader
        .take(8192)
        .read_to_end(&mut diagnostics)
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "could not read bounded Docker removal diagnostics".to_owned(),
        })?;
    classify_bounded_remove_result(target, outcome, &diagnostics)
}

fn classify_bounded_remove_result(
    target: &str,
    outcome: Result<(), EvalExecutionError>,
    diagnostics: &[u8],
) -> Result<(), EvalExecutionError> {
    match outcome {
        Ok(()) => Ok(()),
        Err(EvalExecutionError::ProcessFailure(_)) if reports_absent_container(diagnostics) => {
            Ok(())
        }
        Err(_) => Err(EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "bounded Docker removal did not complete cleanly".to_owned(),
        }),
    }
}

struct DockerArchiveReader {
    child: Child,
    stdout: ChildStdout,
    is_complete: bool,
}

impl DockerArchiveReader {
    fn finish(&mut self) -> io::Result<()> {
        if self.is_complete {
            return Ok(());
        }
        let status = self.child.wait()?;
        self.is_complete = true;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other("docker collect artifact archive failed"))
        }
    }
}

impl Read for DockerArchiveReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.stdout.read(buffer)?;
        if count == 0 {
            self.finish()?;
        }
        Ok(count)
    }
}

impl Drop for DockerArchiveReader {
    fn drop(&mut self) {
        if !self.is_complete && self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct BoundedArchiveReader {
    archive: NamedTempFile,
}

impl Read for BoundedArchiveReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.archive.as_file_mut().read(buffer)
    }
}

fn network_lease(network: &NetworkPolicy) -> Result<&'static str, EvalExecutionError> {
    if network.is_no_network() {
        return Ok("none");
    }
    if network.is_public() {
        return Ok(PUBLIC_NETWORK_LEASE);
    }
    Err(EvalExecutionError::UnsupportedEnforcement(
        "allowlist_egress",
    ))
}

fn build_network_lease(network: &NetworkPolicy) -> Result<&'static str, EvalExecutionError> {
    if network.is_public() {
        return Ok("default");
    }
    network_lease(network)
}

const PUBLIC_NETWORK_LEASE: &str = "aiperf-eval-public";

fn ensure_public_network() -> Result<(), EvalExecutionError> {
    ensure_network_exists(public_network_exists, || {
        docker(
            ["network", "create", PUBLIC_NETWORK_LEASE],
            "create public network",
        )
        .map(|_| ())
    })
}

fn ensure_network_exists<I, C>(mut inspect: I, create: C) -> Result<(), EvalExecutionError>
where
    I: FnMut() -> Result<bool, EvalExecutionError>,
    C: FnOnce() -> Result<(), EvalExecutionError>,
{
    if inspect()? {
        return Ok(());
    }
    let create_error = create();
    if create_error.is_ok() || inspect()? {
        Ok(())
    } else {
        create_error
    }
}

fn public_network_exists() -> Result<bool, EvalExecutionError> {
    let inspect = Command::new("docker")
        .args(["network", "inspect", PUBLIC_NETWORK_LEASE])
        .output()
        .map_err(|_| {
            EvalExecutionError::ProcessSpawn("docker inspect public network".to_owned())
        })?;
    Ok(inspect.status.success())
}

fn docker_image_name(suffix: &str, process_id: u32, run_id: u64) -> String {
    format!("aiperf-eval:{suffix}-{process_id}-{run_id}")
}

fn docker_container_name(suffix: &str, process_id: u32, run_id: u64) -> String {
    format!("aiperf-eval-{suffix}-{process_id}-{run_id}")
}

fn redact_secret_values(
    diagnostic: &str,
    secrets: &std::collections::BTreeMap<super::EnvName, super::SecretValue>,
) -> String {
    let mut redacted = diagnostic.to_owned();
    for secret in secrets.values() {
        redacted = redacted.replace(secret.exposed(), "[REDACTED]");
    }
    redacted
}

struct ContainerWorkspace<'a> {
    path: &'a std::path::Path,
    target: Option<&'a str>,
}

impl<'a> ContainerWorkspace<'a> {
    fn at_workdir(path: &'a std::path::Path, workdir: Option<&'a str>) -> Self {
        Self {
            path,
            target: workdir,
        }
    }
}

fn create_planned_container(
    runtime: &dyn DockerRuntime,
    container: &str,
    image: &str,
    workspace: ContainerWorkspace<'_>,
    environment: &super::EnvironmentPlan,
    network_lease: &str,
    instruction: Option<&str>,
) -> Result<(), EvalExecutionError> {
    let mut arguments = vec![
        "create".to_owned(),
        "--name".to_owned(),
        container.to_owned(),
        "--network".to_owned(),
        network_lease.to_owned(),
    ];
    if let Some(target) = workspace.target {
        arguments.extend([
            "--volume".to_owned(),
            format!("{}:{target}", workspace.path.display()),
        ]);
    }
    if let Some(resources) = environment.resources() {
        arguments.extend([
            "--cpus".to_owned(),
            resources.cpus().to_string(),
            "--memory".to_owned(),
            format!("{}m", resources.memory_mb()),
        ]);
    }
    if let Some(instruction) = instruction {
        arguments.extend([
            "--env".to_owned(),
            format!("AIPERF_EVAL_INSTRUCTION={instruction}"),
        ]);
    }
    arguments.extend([image.to_owned(), "sleep".to_owned(), "infinity".to_owned()]);
    runtime.create(&DockerCreateRequest::new(arguments).with_network_lease(network_lease))
}

fn prepare_workdir(
    runtime: &dyn DockerRuntime,
    container: &str,
    environment: &super::EnvironmentPlan,
    phase: &super::PhasePlan,
    workdir: Option<&str>,
    network_lease: &str,
) -> Result<(), EvalExecutionError> {
    let Some(workdir) = workdir else {
        return Ok(());
    };
    let Some(user) = phase.user().or(environment.user()) else {
        return Ok(());
    };
    if user == "root" {
        return Ok(());
    }
    runtime.exec(
        &DockerExecRequest::new(
            container,
            [
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                format!("mkdir -p {workdir} && chown {user} {workdir} && su -s /bin/sh {user} -c 'test -w {workdir}'"),
            ],
            Default::default(),
            Default::default(),
        )
        .with_phase(
            EvalExecutionPhase::Agent,
            None,
            Some(workdir),
            network_lease,
            None,
        ),
    )
}

fn prepare_verifier_files(
    runtime: &dyn DockerRuntime,
    container: &str,
    network_lease: &str,
) -> Result<(), EvalExecutionError> {
    runtime.exec(
        &DockerExecRequest::new(
            container,
            [
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "rm -rf /tests /logs/verifier && mkdir -p /logs/verifier && chmod 0777 /logs/verifier"
                    .to_owned(),
            ],
            Default::default(),
            Default::default(),
        )
        .with_phase(
            EvalExecutionPhase::Verifier,
            Some("root"),
            None,
            network_lease,
            None,
        ),
    )
}

fn clear_verifier_files(
    runtime: &dyn DockerRuntime,
    container: &str,
    network_lease: &str,
) -> Result<(), EvalExecutionError> {
    runtime.exec(
        &DockerExecRequest::new(
            container,
            [
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "rm -rf /tests /logs/verifier".to_owned(),
            ],
            Default::default(),
            Default::default(),
        )
        .with_phase(
            EvalExecutionPhase::Verifier,
            Some("root"),
            None,
            network_lease,
            None,
        ),
    )
}

fn transfer_verifier_artifacts(
    runtime: &dyn DockerRuntime,
    container: &str,
    source: &std::path::Path,
    workdir: Option<&str>,
    network_lease: &str,
) -> Result<(), EvalExecutionError> {
    let target = match workdir {
        Some(workdir) => workdir.to_owned(),
        None => runtime.container_workdir(container)?,
    };
    if !target.starts_with('/') {
        return Err(EvalExecutionError::InvalidWorkspace(format!(
            "container workdir must be absolute: {target}"
        )));
    }
    runtime.exec(
        &DockerExecRequest::new(
            container,
            ["mkdir".to_owned(), "-p".to_owned(), target.clone()],
            Default::default(),
            Default::default(),
        )
        .with_phase(
            EvalExecutionPhase::Verifier,
            Some("root"),
            None,
            network_lease,
            None,
        ),
    )?;
    runtime.copy(&DockerCopyRequest::new([
        "cp".to_owned(),
        format!("{}/.", source.display()),
        format!("{container}:{target}"),
    ]))
}

fn execute_planned_phase(
    runtime: &dyn DockerRuntime,
    container: &str,
    execution_phase: EvalExecutionPhase,
    command: &[String],
    environment: &super::EnvironmentPlan,
    phase: &super::PhasePlan,
    workdir: Option<&str>,
    secrets: &dyn SecretProvider,
) -> Result<(), EvalExecutionError> {
    if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
        return Err(EvalExecutionError::InvalidCommand);
    }
    let resolved = resolve_phase_environment(environment, phase, secrets)?;
    let network_lease = network_lease(phase.network())?;
    runtime.exec(
        &DockerExecRequest::new(
            container,
            command.iter().cloned(),
            resolved.public().clone(),
            resolved.secrets().clone(),
        )
        .with_phase(
            execution_phase,
            phase.user().or(environment.user()),
            workdir,
            network_lease,
            phase.timeout(),
        ),
    )
}

fn read_reward_with_runtime(
    runtime: &dyn DockerRuntime,
    container: &str,
    workspace: &TempDir,
) -> Result<RewardDocument, EvalExecutionError> {
    let json = copy_optional_with_runtime(
        runtime,
        container,
        "/logs/verifier/reward.json",
        workspace,
        "reward.json",
    )?;
    let text = copy_optional_with_runtime(
        runtime,
        container,
        "/logs/verifier/reward.txt",
        workspace,
        "reward.txt",
    )?;
    RewardDocument::parse(json.as_deref(), text.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn read_reward_from_lease(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
) -> Result<RewardDocument, EvalExecutionError> {
    let json = read_optional_service_file(lease, service, "/logs/verifier/reward.json")?;
    let text = read_optional_service_file(lease, service, "/logs/verifier/reward.txt")?;
    RewardDocument::parse(json.as_deref(), text.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn read_optional_service_file(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    source: &str,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let archive = match lease.archive(ServiceArchiveRequest {
        service,
        source,
        deadline: Duration::from_secs(10),
    }) {
        Ok(archive) => archive,
        Err(EvalExecutionError::ProcessFailure(_))
        | Err(EvalExecutionError::ArtifactCollection(_)) => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let expected = std::path::Path::new(source)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(EvalExecutionError::InvalidRecipe("verifier reward path"))?;
    let mut archive = tar::Archive::new(archive);
    for entry in archive
        .entries()
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?
    {
        let mut entry =
            entry.map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let path = entry
            .path()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if path.file_name().and_then(|name| name.to_str()) == Some(expected)
            && entry.header().entry_type().is_file()
        {
            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            return Ok(Some(bytes));
        }
    }
    Ok(None)
}

fn copy_optional_with_runtime(
    runtime: &dyn DockerRuntime,
    container: &str,
    source: &str,
    workspace: &TempDir,
    destination: &str,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let destination_path = workspace.path().join(destination);
    match runtime.copy(&DockerCopyRequest::new([
        "cp".to_owned(),
        format!("{container}:{source}"),
        destination_path.to_string_lossy().into_owned(),
    ])) {
        Ok(()) => {}
        Err(EvalExecutionError::ProcessFailure(_)) => return Ok(None),
        Err(error) => return Err(error),
    }
    match fs::read(destination_path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(EvalExecutionError::Materialization(error.to_string())),
    }
}

fn docker<'a>(
    arguments: impl IntoIterator<Item = &'a str>,
    action: &str,
) -> Result<Vec<u8>, EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let output = Command::new("docker")
        .args(&arguments)
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn(format!("docker {action}")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(EvalExecutionError::ProcessFailure(format!(
            "docker {action}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn docker_exec_bounded(
    clock: Rc<dyn Clock>,
    container: &str,
    arguments: &[&str],
    action: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
) -> Result<(), EvalExecutionError> {
    let child = Command::new("docker")
        .args(arguments)
        // docker exec output is not part of the evaluation contract. Redirecting both streams
        // prevents an unconsumed pipe from blocking the child past its phase deadline.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn(format!("docker {action}")))?;
    let mut process = DockerExecChild { child };
    let mut remove = remove_timed_out_container;
    drive_docker_exec(clock, &mut process, container, phase, timeout, &mut remove).map_err(
        |error| match error {
            EvalExecutionError::ProcessFailure(reason) => {
                EvalExecutionError::ProcessFailure(format!("{action} ({phase}): {reason}"))
            }
            error => error,
        },
    )
}

const DOCKER_EXEC_POLL_NS: i64 = 10_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
enum DockerExecState {
    Running,
    Succeeded,
    Failed(String),
}

impl DockerExecState {
    fn is_terminal(&self) -> bool {
        !matches!(self, Self::Running)
    }
}

trait DockerExecProcess {
    fn try_wait(&mut self) -> Result<DockerExecState, String>;
    fn kill(&mut self) -> Result<(), String>;
    fn wait(&mut self) -> Result<(), String>;
}

struct DockerExecChild {
    child: Child,
}

impl DockerExecProcess for DockerExecChild {
    fn try_wait(&mut self) -> Result<DockerExecState, String> {
        self.child
            .try_wait()
            .map_err(|error| error.to_string())?
            .map_or_else(
                || Ok(DockerExecState::Running),
                |status| {
                    if status.success() {
                        Ok(DockerExecState::Succeeded)
                    } else {
                        Ok(DockerExecState::Failed(status.to_string()))
                    }
                },
            )
    }

    fn kill(&mut self) -> Result<(), String> {
        self.child.kill().map_err(|error| error.to_string())
    }

    fn wait(&mut self) -> Result<(), String> {
        self.child
            .wait()
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

fn drive_docker_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let result = Rc::new(RefCell::new(None));
    let result_slot = result.clone();
    let outcome = clock.clone().drive(Box::pin(async {
        *result_slot.borrow_mut() =
            Some(wait_for_docker_exec(clock, process, container, phase, timeout, remove).await);
    }));
    if outcome.deadlocked {
        return Err(EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: "execution clock reached quiescence before Docker exec terminated".to_owned(),
        });
    }
    result
        .borrow_mut()
        .take()
        .ok_or_else(|| EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: "execution clock ended before Docker exec produced a terminal result"
                .to_owned(),
        })?
}

async fn wait_for_docker_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let deadline = clock
        .now_ns()
        .saturating_add(timeout.as_nanos().min(i64::MAX as u128) as i64);
    loop {
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(process, container, phase, timeout, false, remove);
        }
        let state = process.try_wait().map_err(|reason| {
            EvalExecutionError::ProcessFailure(format!("docker exec process check: {reason}"))
        })?;
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(
                process,
                container,
                phase,
                timeout,
                state.is_terminal(),
                remove,
            );
        }
        match state {
            DockerExecState::Succeeded => {
                return Ok(());
            }
            DockerExecState::Failed(status) => {
                return Err(EvalExecutionError::ProcessFailure(format!(
                    "docker exec exited with {status}"
                )));
            }
            DockerExecState::Running => {
                let remaining_ns = deadline.saturating_sub(clock.now_ns());
                clock
                    .clone()
                    .sleep(remaining_ns.min(DOCKER_EXEC_POLL_NS))
                    .await;
            }
        }
    }
}

fn terminate_timed_out_exec<P, F>(
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    has_observed_terminal_client: bool,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let kill = (!has_observed_terminal_client).then(|| process.kill());
    let reap = process.wait();
    let removal = remove(container);
    if let Err(error) = removal {
        return Err(error);
    }
    let uncertainties = [
        kill.and_then(Result::err)
            .map(|reason| format!("could not kill docker exec client: {reason}")),
        reap.err()
            .map(|reason| format!("could not reap docker exec client: {reason}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if !uncertainties.is_empty() {
        return Err(EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: uncertainties.join("; "),
        });
    }
    Err(EvalExecutionError::Timeout { phase, timeout })
}

fn remove_timed_out_container(container: &str) -> Result<(), EvalExecutionError> {
    let removal = Command::new("docker")
        .args(["rm", "--force", "--volumes", container])
        .output()
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: "could not start docker rm --force --volumes".to_owned(),
        })?;
    if !removal.status.success() && !reports_absent_container(&removal.stderr) {
        return Err(EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: String::from_utf8_lossy(&removal.stderr).trim().to_owned(),
        });
    }
    let inspection = Command::new("docker")
        .args(["container", "inspect", container])
        .output()
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: "could not start docker container inspect".to_owned(),
        })?;
    if !inspection.status.success() && reports_absent_container(&inspection.stderr) {
        return Ok(());
    }
    let reason = if inspection.status.success() {
        "docker container inspect found the container after forced removal".to_owned()
    } else {
        String::from_utf8_lossy(&inspection.stderr)
            .trim()
            .to_owned()
    };
    Err(EvalExecutionError::ContainerTeardown {
        container: container.to_owned(),
        reason,
    })
}

fn reports_absent_container(stderr: &[u8]) -> bool {
    let diagnostic = String::from_utf8_lossy(stderr).to_ascii_lowercase();
    diagnostic.contains("no such container") || diagnostic.contains("no such object")
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::VecDeque, fs, rc::Rc, time::Duration};

    use super::{
        DockerCliRuntime, DockerExecProcess, DockerExecState, DockerProcessSandbox, DockerRuntime,
        EvalExecutionError, EvalExecutionPhase, classify_bounded_remove_result,
        compose_ownership_filters, docker_container_name, docker_image_name, drive_docker_exec,
        ensure_network_exists, redact_secret_values, reports_absent_container,
    };
    use crate::clock::SimClock;
    use crate::eval::{
        ComposeProjectId, DockerRemoveRequest, HarborImporter, HarborSandboxRecipe, HarborSource,
        NativeSourceAcquirer, VerifierMode,
    };

    #[tokio::test]
    async fn bounded_remove_refuses_nested_clock_drive_before_spawning_docker() {
        let runtime = DockerCliRuntime {
            clock: Rc::new(SimClock::new()),
        };
        let result = runtime.remove(
            &DockerRemoveRequest::new(["rm", "--force", "missing"])
                .with_deadline(Duration::from_secs(1)),
        );
        assert_eq!(
            result,
            Err(EvalExecutionError::RuntimeContext(
                "bounded Docker lease cleanup"
            ))
        );
    }

    #[test]
    fn compose_resource_discovery_requires_project_and_run_labels() {
        let project = ComposeProjectId::new("aiperf-fixture");
        let filters = compose_ownership_filters(&project);

        assert_eq!(
            filters[0],
            "label=com.docker.compose.project=aiperf-fixture"
        );
        assert!(filters.contains(&"label=aiperf.project=aiperf-fixture".to_owned()));
        assert!(filters.iter().any(|filter| {
            filter.starts_with("label=aiperf.run=") && filter != "label=aiperf.run=aiperf-fixture"
        }));
    }

    #[test]
    fn bounded_remove_treats_concurrently_absent_resource_as_success() {
        let result = classify_bounded_remove_result(
            "exact-id",
            Err(EvalExecutionError::ProcessFailure("status".to_owned())),
            b"Error response from daemon: No such object: exact-id",
        );
        assert_eq!(result, Ok(()));
        let failed = classify_bounded_remove_result(
            "exact-id",
            Err(EvalExecutionError::ProcessFailure("status".to_owned())),
            b"permission denied",
        );
        assert!(matches!(
            failed,
            Err(EvalExecutionError::ContainerTeardown { .. })
        ));
    }

    #[test]
    fn completed_command_observed_after_deadline_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(clock.clone(), [DockerExecState::Succeeded])
            .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(100),
            &mut remove,
        );

        assert_eq!(
            result,
            Err(EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::Agent,
                timeout: Duration::from_nanos(100),
            })
        );
        assert!(was_removed.get());
        assert!(!process.was_killed.get());
        assert!(process.was_reaped.get());
    }

    #[test]
    fn failed_command_observed_after_deadline_also_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(
            clock.clone(),
            [DockerExecState::Failed("exit status: 1".to_owned())],
        )
        .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(100),
            &mut remove,
        );

        assert_eq!(
            result,
            Err(EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::Agent,
                timeout: Duration::from_nanos(100),
            })
        );
        assert!(was_removed.get());
        assert!(!process.was_killed.get());
    }

    #[test]
    fn kill_failure_returns_typed_terminal_uncertainty_after_removal() {
        let clock = Rc::new(SimClock::new());
        let mut process =
            FakeDockerExec::new(clock.clone(), [DockerExecState::Running]).kill_fails();
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(10),
            &mut remove,
        );

        assert!(matches!(
            result,
            Err(EvalExecutionError::TerminalUncertainty {
                phase: EvalExecutionPhase::Agent,
                container,
                ..
            }) if container == "agent-container"
        ));
        assert!(was_removed.get());
        assert!(process.was_reaped.get());
    }

    #[test]
    fn reap_failure_returns_typed_terminal_uncertainty_after_removal() {
        let clock = Rc::new(SimClock::new());
        let mut process =
            FakeDockerExec::new(clock.clone(), [DockerExecState::Running]).wait_fails();
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(10),
            &mut remove,
        );

        assert!(matches!(
            result,
            Err(EvalExecutionError::TerminalUncertainty {
                phase: EvalExecutionPhase::Agent,
                container,
                ..
            }) if container == "agent-container"
        ));
        assert!(was_removed.get());
        assert!(process.was_killed.get());
        assert!(process.was_reaped.get());
    }

    #[tokio::test]
    async fn execute_within_tokio_runtime_returns_explicit_context_error() {
        let temporary = tempfile::tempdir().unwrap();
        fs::create_dir_all(temporary.path().join("environment")).unwrap();
        fs::create_dir_all(temporary.path().join("tests")).unwrap();
        fs::write(
            temporary.path().join("task.toml"),
            "schema_version = \"1.0\"\n[task]\nname = \"example/runtime-context\"\n[agent]\ntimeout_sec = 1\n[verifier]\ntimeout_sec = 1\n",
        )
        .unwrap();
        fs::write(
            temporary.path().join("instruction.md"),
            "Complete the task.\n",
        )
        .unwrap();
        fs::write(
            temporary.path().join("environment/Dockerfile"),
            "FROM scratch\n",
        )
        .unwrap();
        fs::write(temporary.path().join("tests/test.sh"), "exit 0\n").unwrap();
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(temporary.path().to_string_lossy()).unwrap())
            .unwrap();
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();

        let error = DockerProcessSandbox::new()
            .execute(
                &recipe,
                &imported.package,
                &["true".to_owned()],
                VerifierMode::Shared,
            )
            .expect_err("synchronous Docker execution must not nest a Tokio runtime");

        assert!(matches!(
            error,
            EvalExecutionError::RuntimeContext("synchronous Docker execution")
        ));
    }

    #[test]
    fn docker_exec_diagnostic_redacts_resolved_secret_values() {
        let secret = "actual-secret-value";
        let diagnostic = redact_secret_values(
            &format!("agent command wrote {secret} to standard error"),
            &std::collections::BTreeMap::from([(
                "TOKEN".to_owned(),
                super::super::SecretValue::new(secret),
            )]),
        );

        assert!(diagnostic.contains("[REDACTED]"));
        assert!(!diagnostic.contains(secret));
    }

    #[test]
    fn docker_run_names_include_process_and_run_identity() {
        assert_eq!(
            docker_image_name("digest", 41, 7),
            "aiperf-eval:digest-41-7"
        );
        assert_eq!(
            docker_container_name("digest", 42, 7),
            "aiperf-eval-digest-42-7"
        );
        assert_ne!(
            docker_container_name("digest", 41, 7),
            docker_container_name("digest", 42, 7)
        );
    }

    #[test]
    fn network_create_race_succeeds_when_reinspect_finds_the_network() {
        let inspections = Cell::new(0);
        let result = ensure_network_exists(
            || {
                inspections.set(inspections.get() + 1);
                Ok(inspections.get() == 2)
            },
            || {
                Err(EvalExecutionError::ProcessFailure(
                    "already exists".to_owned(),
                ))
            },
        );

        assert_eq!(result, Ok(()));
        assert_eq!(inspections.get(), 2);
    }

    #[test]
    fn absent_container_diagnostic_is_idempotent_cleanup() {
        assert!(reports_absent_container(
            b"Error response from daemon: No such container"
        ));
    }

    struct FakeDockerExec {
        clock: Rc<SimClock>,
        states: VecDeque<DockerExecState>,
        advance_terminal_to: Option<i64>,
        kill_fails: bool,
        wait_fails: bool,
        terminal_observed: bool,
        was_killed: Cell<bool>,
        was_reaped: Cell<bool>,
    }

    impl FakeDockerExec {
        fn new(clock: Rc<SimClock>, states: impl IntoIterator<Item = DockerExecState>) -> Self {
            Self {
                clock,
                states: states.into_iter().collect(),
                advance_terminal_to: None,
                kill_fails: false,
                wait_fails: false,
                terminal_observed: false,
                was_killed: Cell::new(false),
                was_reaped: Cell::new(false),
            }
        }

        fn advance_terminal_to(mut self, time_ns: i64) -> Self {
            self.advance_terminal_to = Some(time_ns);
            self
        }

        fn kill_fails(mut self) -> Self {
            self.kill_fails = true;
            self
        }

        fn wait_fails(mut self) -> Self {
            self.wait_fails = true;
            self
        }
    }

    impl DockerExecProcess for FakeDockerExec {
        fn try_wait(&mut self) -> Result<DockerExecState, String> {
            let state = self.states.pop_front().unwrap_or(DockerExecState::Running);
            if state != DockerExecState::Running {
                self.terminal_observed = true;
                if let Some(time_ns) = self.advance_terminal_to {
                    self.clock.advance_to(time_ns);
                }
            }
            Ok(state)
        }

        fn kill(&mut self) -> Result<(), String> {
            self.was_killed.set(true);
            if self.terminal_observed {
                Err("cannot kill an observed terminal child".to_owned())
            } else if self.kill_fails {
                Err("simulated kill failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn wait(&mut self) -> Result<(), String> {
            self.was_reaped.set(true);
            if self.wait_fails {
                Err("simulated reap failure".to_owned())
            } else {
                Ok(())
            }
        }
    }
}
