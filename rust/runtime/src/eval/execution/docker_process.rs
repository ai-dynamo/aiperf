// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed execution for conventional native task directories.

use std::{
    cell::{Cell, RefCell},
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::PermissionsExt,
    process::{Child, ChildStdout, Command, Stdio},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    thread,
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
    MultiStepExecutionResult, NetworkPolicy, SecretProvider, preflight_docker, resolve_environment,
    resolve_phase_environment, shared_workdir_conflicts_reserved_verifier_path,
    verifier_artifact_target_collision,
};

const MAX_DOCKER_ARCHIVE_BYTES: usize = 64 * 1024 * 1024;
const MAX_DOCKER_COMMAND_OUTPUT_BYTES: usize = 1024 * 1024;

use super::docker_runtime::preflight_compose_configuration;
use super::multi_step::{BenchmarkStepSession, execute_benchmark_steps};
use super::{
    artifacts::{
        Deadline, collect_artifacts_bounded, collect_service_artifacts, collect_service_evidence,
        transfer_artifacts_bounded,
    },
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
        if verifier_mode != package.execution_plan().verifier().mode() {
            return Err(EvalExecutionError::InvalidRecipe("verifier mode"));
        }
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
        finish_with_cleanup(self.clock.clone(), runtime, containers, outcome)
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
            return self.execute_legacy_with_runtime(
                runtime,
                recipe,
                package,
                plan,
                agent_command,
                secrets,
            );
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

            let agent_deadline = plan.agent().timeout().map(|timeout| {
                Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Agent, timeout)
            });
            let remaining = |deadline: &Option<Deadline>| {
                deadline.as_ref().map(Deadline::remaining).transpose()
            };
            prepare_workdir_with_deadline(
                runtime,
                &container,
                environment,
                plan.agent(),
                EvalExecutionPhase::Agent,
                environment_workdir,
                baseline_network,
                remaining(&agent_deadline)?,
            )?;
            execute_planned_phase_with_deadline(
                runtime,
                &container,
                EvalExecutionPhase::Agent,
                agent_command,
                environment,
                plan.agent(),
                environment_workdir,
                secrets,
                remaining(&agent_deadline)?,
            )?;
            let artifact_collection = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            let collection_timeout = plan
                .steps()
                .first()
                .ok_or(EvalExecutionError::InvalidRecipe("Docker benchmark step"))?
                .collection_timeout();
            let collection_deadline =
                Deadline::from_timeout(self.clock.clone(), collection_timeout);
            let artifacts = collect_artifacts_bounded(
                runtime,
                &container,
                plan.artifacts(),
                artifact_collection.path(),
                collection_deadline,
            )?;
            let verifier_deadline = verifier.phase().timeout().map(|timeout| {
                Deadline::from_phase_timeout(
                    self.clock.clone(),
                    EvalExecutionPhase::Verifier,
                    timeout,
                )
            });
            let remaining = |deadline: &Option<Deadline>| {
                deadline.as_ref().map(Deadline::remaining).transpose()
            };

            let verifier_container = if verifier.mode() == VerifierMode::Separate {
                let verifier_workspace = tempfile::tempdir()
                    .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
                fs::set_permissions(verifier_workspace.path(), fs::Permissions::from_mode(0o755))
                    .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
                if let Some(deadline) = verifier_deadline.as_ref() {
                    transfer_artifacts_bounded(
                        artifact_collection.path(),
                        verifier_workspace.path(),
                        &artifacts,
                        deadline,
                    )?;
                } else {
                    super::transfer_artifacts(
                        artifact_collection.path(),
                        verifier_workspace.path(),
                        &artifacts,
                    )?;
                }
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
                    remaining(&verifier_deadline)?,
                )?;
                containers.push(name.clone());
                let start = match remaining(&verifier_deadline)? {
                    Some(deadline) => DockerStartRequest::new(&name).with_deadline(deadline),
                    None => DockerStartRequest::new(&name),
                };
                runtime.start(&start)?;
                if !plan.artifacts().is_empty() {
                    let effective_verifier_workdir = match verifier_workdir {
                        Some(workdir) => workdir.to_owned(),
                        None => match remaining(&verifier_deadline)? {
                            Some(deadline) => runtime.container_workdir_bounded(&name, deadline)?,
                            None => runtime.container_workdir(&name)?,
                        },
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
                        verifier_deadline.as_ref(),
                    )?;
                }
                prepare_workdir_with_deadline(
                    runtime,
                    &name,
                    verifier.environment(),
                    verifier.phase(),
                    EvalExecutionPhase::Verifier,
                    verifier_workdir,
                    verifier_network,
                    remaining(&verifier_deadline)?,
                )?;
                if let Some(healthcheck) = verifier.environment().healthcheck() {
                    run_healthcheck_with_deadline(
                        self.clock.clone(),
                        runtime,
                        &name,
                        verifier.environment(),
                        verifier_workdir,
                        healthcheck,
                        verifier_network,
                        secrets,
                        verifier_deadline.as_ref(),
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
            prepare_verifier_files_with_deadline(
                runtime,
                verifier_name,
                verifier_network,
                remaining(&verifier_deadline)?,
            )?;
            let copy = DockerCopyRequest::new([
                "cp".to_owned(),
                format!("{}/.", source_root.join("tests").display()),
                format!("{verifier_name}:/tests"),
            ]);
            let copy = match remaining(&verifier_deadline)? {
                Some(deadline) => copy.with_deadline(deadline),
                None => copy,
            };
            runtime.copy(&copy)?;
            let verifier_workdir = recipe.resolve_workdir(verifier.environment().workdir());
            if verifier.mode() == VerifierMode::Shared {
                prepare_workdir_with_deadline(
                    runtime,
                    verifier_name,
                    verifier.environment(),
                    verifier.phase(),
                    EvalExecutionPhase::Verifier,
                    verifier_workdir,
                    verifier_network,
                    remaining(&verifier_deadline)?,
                )?;
            }
            execute_planned_phase_with_deadline(
                runtime,
                verifier_name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                secrets,
                remaining(&verifier_deadline)?,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            let reward = read_reward_with_runtime(
                runtime,
                verifier_name,
                &reward_workspace,
                verifier_deadline.as_ref(),
            )?;
            Ok(LocalExecutionResult {
                artifacts,
                reward,
                verifier: package.source_digest(),
            })
        })();
        let cleanup = remove_containers_with_deadline(self.clock.clone(), runtime, containers);
        combine_primary_and_cleanup(
            outcome,
            cleanup.map_or(Ok(()), Err),
            "Docker evaluation containers",
        )
    }

    fn execute_legacy_with_runtime(
        &self,
        runtime: &dyn DockerRuntime,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        plan: &BenchmarkExecutionPlan,
        agent_command: &[String],
        secrets: &dyn SecretProvider,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        preflight_docker(runtime, plan)?;
        let environment = plan.environment();
        let verifier = plan.verifier();
        let environment_workdir = recipe.resolve_workdir(environment.workdir());
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let (_, container) = docker_run_names(package);
        let mut containers = vec![container.clone()];
        let outcome = (|| {
            let agent_network = network_lease(environment.network())?;
            create_planned_container(
                runtime,
                &container,
                &recipe.image,
                ContainerWorkspace::at_workdir(workspace.path(), environment_workdir),
                environment,
                agent_network,
                Some(package.instruction()),
                None,
            )?;
            runtime.start(&DockerStartRequest::new(&container))?;
            execute_planned_phase_with_deadline(
                runtime,
                &container,
                EvalExecutionPhase::Agent,
                agent_command,
                environment,
                plan.agent(),
                environment_workdir,
                secrets,
                plan.agent().timeout(),
            )?;
            let artifact_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
            let collection_timeout = plan
                .steps()
                .first()
                .ok_or(EvalExecutionError::InvalidRecipe("Docker benchmark step"))?
                .collection_timeout();
            let artifacts = collect_artifacts_bounded(
                runtime,
                &container,
                plan.artifacts(),
                artifact_workspace.path(),
                Deadline::from_timeout(self.clock.clone(), collection_timeout),
            )?;
            let verifier_deadline = verifier.phase().timeout().map(|timeout| {
                Deadline::from_phase_timeout(
                    self.clock.clone(),
                    EvalExecutionPhase::Verifier,
                    timeout,
                )
            });
            let remaining = |deadline: &Option<Deadline>| {
                deadline.as_ref().map(Deadline::remaining).transpose()
            };
            let verifier_workdir = recipe.resolve_workdir(verifier.environment().workdir());
            let verifier_network = network_lease(verifier.phase().network())?;
            let verifier_container = if verifier.mode() == VerifierMode::Separate {
                let verifier_workspace = tempfile::tempdir()
                    .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
                fs::set_permissions(verifier_workspace.path(), fs::Permissions::from_mode(0o755))
                    .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
                if let Some(deadline) = verifier_deadline.as_ref() {
                    transfer_artifacts_bounded(
                        artifact_workspace.path(),
                        verifier_workspace.path(),
                        &artifacts,
                        deadline,
                    )?;
                } else {
                    super::transfer_artifacts(
                        artifact_workspace.path(),
                        verifier_workspace.path(),
                        &artifacts,
                    )?;
                }
                let name = format!("{container}-verifier");
                create_planned_container(
                    runtime,
                    &name,
                    &recipe.image,
                    ContainerWorkspace::at_workdir(verifier_workspace.path(), verifier_workdir),
                    verifier.environment(),
                    network_lease(verifier.environment().network())?,
                    None,
                    remaining(&verifier_deadline)?,
                )?;
                containers.push(name.clone());
                let start = match remaining(&verifier_deadline)? {
                    Some(deadline) => DockerStartRequest::new(&name).with_deadline(deadline),
                    None => DockerStartRequest::new(&name),
                };
                runtime.start(&start)?;
                if !plan.artifacts().is_empty() {
                    transfer_verifier_artifacts(
                        runtime,
                        &name,
                        verifier_workspace.path(),
                        verifier_workdir,
                        verifier_network,
                        verifier_deadline.as_ref(),
                    )?;
                }
                Some((name, verifier_workspace))
            } else {
                None
            };
            let verifier_name = verifier_container
                .as_ref()
                .map_or(container.as_str(), |(name, _)| name.as_str());
            prepare_verifier_files_with_deadline(
                runtime,
                verifier_name,
                verifier_network,
                remaining(&verifier_deadline)?,
            )?;
            execute_planned_phase_with_deadline(
                runtime,
                verifier_name,
                EvalExecutionPhase::Verifier,
                package.verifier_command(),
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                secrets,
                remaining(&verifier_deadline)?,
            )?;
            let reward = read_reward_archive_with_runtime(
                runtime,
                verifier_name,
                verifier_deadline.as_ref(),
            )?;
            let verifier = ArtifactDigest::parse(package.verifier())
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            Ok(LocalExecutionResult {
                artifacts,
                reward,
                verifier,
            })
        })();
        let cleanup = remove_containers_with_deadline(self.clock.clone(), runtime, containers);
        combine_primary_and_cleanup(
            outcome,
            cleanup.map_or(Ok(()), Err),
            "Docker evaluation containers",
        )
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
        let cleanup = if outcome.is_err() {
            prepared.lease.teardown_after_terminal_failure(
                super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE,
            )
        } else {
            prepared.lease.teardown()
        };
        combine_primary_and_cleanup(
            outcome,
            cleanup,
            prepared.lease.project().as_str().to_owned(),
        )
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
        let outcome = {
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
        };
        let cleanup = if outcome.is_err() {
            prepared.lease.teardown_after_terminal_failure(
                super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE,
            )
        } else {
            prepared.lease.teardown()
        };
        combine_primary_and_cleanup(
            outcome,
            cleanup,
            prepared.lease.project().as_str().to_owned(),
        )
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
        let materialized = package.materialize_source()?;
        let (source_root, environment_root) = standard_task_roots(package, materialized.root())?;
        let source_root = source_root.to_path_buf();
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        fs::set_permissions(workspace.path(), fs::Permissions::from_mode(0o755))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let (image, _) = docker_run_names(package);
        let mut lease = ComposeProjectLease::reserve_with_clock(
            compose_runtime,
            self.clock.clone(),
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
            compose.build_timeout().min(compose.startup_timeout()),
        )?;
        fs::write(
            source_root.join("aiperf.generated.compose.yaml"),
            generated.into_bytes(),
        )
        .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        if let Err(error) = runtime.build(
            &DockerBuildRequest::new([
                "build",
                "--network",
                "default",
                "--tag",
                &image,
                environment_root.to_string_lossy().as_ref(),
            ])
            .with_network_lease("default")
            .with_deadline(compose.build_timeout()),
        ) {
            let cleanup = lease.teardown_after_terminal_failure(
                super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE,
            );
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(EvalExecutionError::ContainerTeardown {
                    container: lease.project().as_str().to_owned(),
                    reason: format!("{error}; cleanup: {cleanup_error}"),
                }),
            };
        }
        lease.start()?;
        let main = lease.main_service().clone();
        prepare_lease_workdir_for_user(
            &mut lease,
            &main,
            environment.user(),
            EvalExecutionPhase::Healthcheck,
            environment.workdir(),
            environment
                .healthcheck()
                .and_then(|healthcheck| healthcheck.timeout()),
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
        deadline: Option<Deadline>,
    ) -> Result<RewardDocument, EvalExecutionError> {
        let verifier = step.verifier();
        let remaining =
            |deadline: &Option<Deadline>| deadline.as_ref().map(Deadline::remaining).transpose();
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
        if let Some(deadline) = deadline.as_ref() {
            transfer_artifacts_bounded(collection.path(), workspace.path(), artifacts, deadline)?;
        } else {
            super::transfer_artifacts(collection.path(), workspace.path(), artifacts)?;
        }
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
                remaining(&deadline)?,
            )?;
            let start = match remaining(&deadline)? {
                Some(deadline) => DockerStartRequest::new(&name).with_deadline(deadline),
                None => DockerStartRequest::new(&name),
            };
            self.runtime.start(&start)?;
            let effective_workdir = match verifier_workdir {
                Some(workdir) => workdir.to_owned(),
                None => match remaining(&deadline)? {
                    Some(deadline) => self.runtime.container_workdir_bounded(&name, deadline)?,
                    None => self.runtime.container_workdir(&name)?,
                },
            };
            validate_verifier_artifact_staging(&effective_workdir, step.artifacts())?;
            transfer_verifier_artifacts(
                self.runtime,
                &name,
                workspace.path(),
                Some(&effective_workdir),
                verifier_network,
                deadline.as_ref(),
            )?;
            prepare_workdir_with_deadline(
                self.runtime,
                &name,
                verifier.environment(),
                verifier.phase(),
                EvalExecutionPhase::Verifier,
                verifier_workdir,
                verifier_network,
                remaining(&deadline)?,
            )?;
            if let Some(healthcheck) = verifier.environment().healthcheck() {
                run_healthcheck_with_deadline(
                    self.clock.clone(),
                    self.runtime,
                    &name,
                    verifier.environment(),
                    verifier_workdir,
                    healthcheck,
                    verifier_network,
                    self.secrets,
                    deadline.as_ref(),
                )?;
            }
            prepare_verifier_files_with_deadline(
                self.runtime,
                &name,
                verifier_network,
                remaining(&deadline)?,
            )?;
            let copy = DockerCopyRequest::new([
                "cp".to_owned(),
                format!(
                    "{}/.",
                    self.source_root.join(step.verifier_test_root()).display()
                ),
                format!("{name}:/tests"),
            ]);
            let copy = match remaining(&deadline)? {
                Some(deadline) => copy.with_deadline(deadline),
                None => copy,
            };
            self.runtime.copy(&copy)?;
            execute_planned_phase_with_deadline(
                self.runtime,
                &name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                self.secrets,
                remaining(&deadline)?,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            read_reward_with_runtime(self.runtime, &name, &reward_workspace, deadline.as_ref())
        })();
        let cleanup = self.runtime.remove(
            &DockerRemoveRequest::new(["rm", "--force", "--volumes", &name])
                .with_deadline(verifier_cleanup_deadline(&deadline)),
        );
        match (outcome, cleanup) {
            (Err(error), Err(cleanup_error)) => Err(EvalExecutionError::ContainerTeardown {
                container: name,
                reason: format!("{error}; cleanup: {cleanup_error}"),
            }),
            (Err(error), Ok(())) => Err(error),
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
        let deadline = verifier.phase().timeout().map(|timeout| {
            Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Verifier, timeout)
        });
        let remaining =
            |deadline: &Option<Deadline>| deadline.as_ref().map(Deadline::remaining).transpose();
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
            deadline: remaining(&deadline)?,
        })?;
        let source = format!(
            "{}/.",
            self.source_root.join(step.verifier_test_root()).display()
        );
        if let Some(copy_deadline) = remaining(&deadline)? {
            self.lease
                .copy_into_bounded(&main, &source, "/tests", copy_deadline)?;
        } else {
            self.lease.copy_into(&main, &source, "/tests")?;
        }
        let resolved =
            resolve_phase_environment(verifier.environment(), verifier.phase(), self.secrets)?;
        let verifier_workdir = self
            .recipe
            .resolve_workdir(verifier.environment().workdir());
        prepare_lease_workdir_for_user(
            self.lease,
            &main,
            verifier.phase().user().or(verifier.environment().user()),
            EvalExecutionPhase::Verifier,
            verifier_workdir,
            remaining(&deadline)?,
        )?;
        self.lease.exec(ServiceExecRequest {
            service: &main,
            arguments: &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
            public_environment: resolved.public().clone(),
            secret_environment: resolved.secrets().clone(),
            phase: EvalExecutionPhase::Verifier,
            user: verifier.phase().user().or(verifier.environment().user()),
            workdir: verifier_workdir,
            deadline: remaining(&deadline)?,
        })?;
        read_reward_from_lease(self.lease, &main, deadline.as_ref())
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
        let deadline = step.agent().timeout().map(|timeout| {
            Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Agent, timeout)
        });
        let remaining =
            |deadline: &Option<Deadline>| deadline.as_ref().map(Deadline::remaining).transpose();
        prepare_lease_workdir_for_user(
            self.lease,
            &main,
            step.agent().user().or(self.environment.user()),
            EvalExecutionPhase::Agent,
            workdir,
            remaining(&deadline)?,
        )?;
        self.lease.exec(ServiceExecRequest {
            service: &main,
            arguments: command,
            public_environment,
            secret_environment,
            phase: EvalExecutionPhase::Agent,
            user: step.agent().user().or(self.environment.user()),
            workdir,
            deadline: remaining(&deadline)?,
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
        let deadline = step.verifier().phase().timeout().map(|timeout| {
            Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Verifier, timeout)
        });
        let has_non_main_evidence = step
            .artifacts()
            .iter()
            .any(|artifact| artifact.service() != "main")
            || step
                .collect_hooks()
                .iter()
                .any(|hook| hook.service().as_str() != "main");
        if has_non_main_evidence {
            let cleanup_timeout = deadline
                .as_ref()
                .map(Deadline::remaining)
                .transpose()?
                .unwrap_or(super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE);
            self.lease
                .teardown_after_terminal_failure(cleanup_timeout)?;
        }
        let result = self.run_separate_verifier(step, artifacts, deadline);
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
        let deadline = step.agent().timeout().map(|timeout| {
            Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Agent, timeout)
        });
        let remaining =
            |deadline: &Option<Deadline>| deadline.as_ref().map(Deadline::remaining).transpose();
        prepare_workdir_with_deadline(
            self.runtime,
            self.agent_container,
            self.environment,
            step.agent(),
            EvalExecutionPhase::Agent,
            workdir,
            baseline_network,
            remaining(&deadline)?,
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
                remaining(&deadline)?,
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
        let artifacts = collect_artifacts_bounded(
            self.runtime,
            self.agent_container,
            step.artifacts(),
            collection.path(),
            Deadline::from_timeout(self.clock.clone(), step.collection_timeout()),
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
        let deadline = verifier.phase().timeout().map(|timeout| {
            Deadline::from_phase_timeout(self.clock.clone(), EvalExecutionPhase::Verifier, timeout)
        });
        let remaining =
            |deadline: &Option<Deadline>| deadline.as_ref().map(Deadline::remaining).transpose();
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
            if let Some(deadline) = deadline.as_ref() {
                transfer_artifacts_bounded(
                    collection.path(),
                    workspace.path(),
                    artifacts,
                    deadline,
                )?;
            } else {
                super::transfer_artifacts(collection.path(), workspace.path(), artifacts)?;
            }
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
                remaining(&deadline)?,
            )?;
            self.containers.push(name.clone());
            let start = match remaining(&deadline)? {
                Some(deadline) => DockerStartRequest::new(&name).with_deadline(deadline),
                None => DockerStartRequest::new(&name),
            };
            self.runtime.start(&start)?;
            let effective_verifier_workdir = match verifier_workdir {
                Some(workdir) => workdir.to_owned(),
                None => match remaining(&deadline)? {
                    Some(deadline) => self.runtime.container_workdir_bounded(&name, deadline)?,
                    None => self.runtime.container_workdir(&name)?,
                },
            };
            validate_verifier_artifact_staging(&effective_verifier_workdir, step.artifacts())?;
            transfer_verifier_artifacts(
                self.runtime,
                &name,
                workspace.path(),
                Some(&effective_verifier_workdir),
                verifier_network,
                deadline.as_ref(),
            )?;
            prepare_workdir_with_deadline(
                self.runtime,
                &name,
                verifier.environment(),
                verifier.phase(),
                EvalExecutionPhase::Verifier,
                verifier_workdir,
                verifier_network,
                remaining(&deadline)?,
            )?;
            if let Some(healthcheck) = verifier.environment().healthcheck() {
                run_healthcheck_with_deadline(
                    self.clock.clone(),
                    self.runtime,
                    &name,
                    verifier.environment(),
                    verifier_workdir,
                    healthcheck,
                    verifier_network,
                    self.secrets,
                    deadline.as_ref(),
                )?;
            }
            name
        } else {
            self.agent_container.to_owned()
        };
        let verifier_network = network_lease(verifier.phase().network())?;
        let outcome = (|| {
            prepare_verifier_files_with_deadline(
                self.runtime,
                &verifier_name,
                verifier_network,
                remaining(&deadline)?,
            )?;
            let copy = DockerCopyRequest::new([
                "cp".to_owned(),
                format!(
                    "{}/.",
                    self.source_root.join(step.verifier_test_root()).display()
                ),
                format!("{verifier_name}:/tests"),
            ]);
            let copy = match remaining(&deadline)? {
                Some(deadline) => copy.with_deadline(deadline),
                None => copy,
            };
            self.runtime.copy(&copy)?;
            let verifier_workdir = self
                .recipe
                .resolve_workdir(verifier.environment().workdir());
            if verifier.mode() == VerifierMode::Shared {
                prepare_workdir_with_deadline(
                    self.runtime,
                    &verifier_name,
                    verifier.environment(),
                    verifier.phase(),
                    EvalExecutionPhase::Verifier,
                    verifier_workdir,
                    verifier_network,
                    remaining(&deadline)?,
                )?;
            }
            execute_planned_phase_with_deadline(
                self.runtime,
                &verifier_name,
                EvalExecutionPhase::Verifier,
                &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
                verifier.environment(),
                verifier.phase(),
                verifier_workdir,
                self.secrets,
                remaining(&deadline)?,
            )?;
            let reward_workspace = tempfile::tempdir()
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            read_reward_with_runtime(
                self.runtime,
                &verifier_name,
                &reward_workspace,
                deadline.as_ref(),
            )
        })();
        let cleanup = if verifier.mode() == VerifierMode::Shared {
            clear_verifier_files(
                self.runtime,
                &verifier_name,
                verifier_network,
                verifier_cleanup_deadline(&deadline),
            )
        } else {
            Ok(())
        };
        self.artifact_collection = None;
        combine_primary_and_cleanup(outcome, cleanup, verifier_name)
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
    clock: Rc<dyn Clock>,
    runtime: &dyn DockerRuntime,
    containers: Vec<String>,
    outcome: Result<T, EvalExecutionError>,
) -> Result<T, EvalExecutionError> {
    let cleanup = remove_containers_with_deadline(clock, runtime, containers);
    combine_primary_and_cleanup(
        outcome,
        cleanup.map_or(Ok(()), Err),
        "Docker evaluation containers",
    )
}

fn combine_primary_and_cleanup<T>(
    outcome: Result<T, EvalExecutionError>,
    cleanup: Result<(), EvalExecutionError>,
    container: impl Into<String>,
) -> Result<T, EvalExecutionError> {
    match (outcome, cleanup) {
        (Err(error), Err(cleanup_error)) => Err(EvalExecutionError::ContainerTeardown {
            container: container.into(),
            reason: format!("{error}; cleanup: {cleanup_error}"),
        }),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(result), Ok(())) => Ok(result),
    }
}

fn remove_containers_with_deadline(
    clock: Rc<dyn Clock>,
    runtime: &dyn DockerRuntime,
    containers: Vec<String>,
) -> Option<EvalExecutionError> {
    let deadline = Deadline::from_phase_timeout(
        clock,
        EvalExecutionPhase::Verifier,
        super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE,
    );
    containers
        .into_iter()
        .rev()
        .fold(None, |first_error, container| {
            let removal = deadline.remaining().and_then(|remaining| {
                runtime.remove(
                    &DockerRemoveRequest::new(["rm", "--force", "--volumes", &container])
                        .with_deadline(remaining),
                )
            });
            first_error.or(removal.err())
        })
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
    run_healthcheck_with_deadline(
        clock,
        runtime,
        container,
        environment,
        workdir,
        healthcheck,
        network_lease,
        secrets,
        None,
    )
}

fn run_healthcheck_with_deadline(
    clock: Rc<dyn Clock>,
    runtime: &dyn DockerRuntime,
    container: &str,
    environment: &super::EnvironmentPlan,
    workdir: Option<&str>,
    healthcheck: &super::HealthcheckPlan,
    network_lease: &str,
    secrets: &dyn SecretProvider,
    deadline: Option<&Deadline>,
) -> Result<(), EvalExecutionError> {
    if let Some(start_period) = healthcheck.start_period() {
        sleep_healthcheck_with_deadline(clock.clone(), start_period, container, deadline)?;
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
            match (
                healthcheck.timeout(),
                deadline.map(Deadline::remaining).transpose()?,
            ) {
                (Some(health), Some(remaining)) => Some(health.min(remaining)),
                (Some(health), None) => Some(health),
                (None, Some(remaining)) => Some(remaining),
                (None, None) => None,
            },
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
                sleep_healthcheck_with_deadline(clock.clone(), interval, container, deadline)?;
            }
        }
    }
    let reason = last_error.map_or_else(
        || "healthcheck exhausted without an execution result".to_owned(),
        |error| error.to_string(),
    );
    Err(EvalExecutionError::Unhealthy(reason))
}

fn sleep_healthcheck_with_deadline(
    clock: Rc<dyn Clock>,
    duration: Duration,
    container: &str,
    deadline: Option<&Deadline>,
) -> Result<(), EvalExecutionError> {
    let duration = match deadline.map(Deadline::remaining).transpose()? {
        Some(remaining) => duration.min(remaining),
        None => duration,
    };
    sleep_with_clock(clock, duration, container)?;
    if let Some(deadline) = deadline {
        deadline.remaining()?;
    }
    Ok(())
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

fn prepare_lease_workdir_for_user(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    user: Option<&str>,
    execution_phase: EvalExecutionPhase,
    workdir: Option<&str>,
    deadline: Option<Duration>,
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
        arguments: &prepare_workdir_arguments(workdir, user),
        public_environment: Default::default(),
        secret_environment: Default::default(),
        phase: execution_phase,
        user: Some("root"),
        workdir: Some(workdir),
        deadline,
    })
}

fn prepare_workdir_arguments(workdir: &str, user: &str) -> Vec<String> {
    vec![
        "/bin/sh".to_owned(),
        "-ec".to_owned(),
        "mkdir -p -- \"$1\"\nchown -- \"$2\" \"$1\"\nexec su -s /bin/sh -c 'test -w \"$0\"' -- \"$2\" \"$1\""
            .to_owned(),
        "--".to_owned(),
        workdir.to_owned(),
        user.to_owned(),
    ]
}

fn sleep_with_clock(
    clock: Rc<dyn Clock>,
    duration: Duration,
    container: &str,
) -> Result<(), EvalExecutionError> {
    if !clock.is_virtual() {
        // Evaluation is a synchronous provider boundary. A real-clock delay
        // therefore polls the clock directly instead of nesting Tokio's
        // `block_on` when the caller happens to be a Tokio task.
        std::thread::sleep(duration);
        return Ok(());
    }
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
        docker_command_bounded(
            self.clock.clone(),
            request.public_arguments().iter().cloned(),
            "task environment build",
            request.deadline(),
        )
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        let deadline_ns = request
            .deadline()
            .map(|deadline| provider_deadline_ns(&self.clock, deadline));
        if request.network_lease() == Some(PUBLIC_NETWORK_LEASE) {
            if let Some(deadline_ns) = deadline_ns {
                ensure_public_network_bounded(
                    self.clock.clone(),
                    remaining_provider_deadline(&self.clock, deadline_ns, PUBLIC_NETWORK_LEASE)?,
                )?;
            } else {
                ensure_public_network()?;
            }
        }
        docker_command_bounded(
            self.clock.clone(),
            request.public_arguments().iter().cloned(),
            "task container creation",
            match deadline_ns {
                Some(deadline_ns) => Some(remaining_provider_deadline(
                    &self.clock,
                    deadline_ns,
                    PUBLIC_NETWORK_LEASE,
                )?),
                None => None,
            },
        )
    }

    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        docker_command_bounded(
            self.clock.clone(),
            ["start".to_owned(), request.container().to_owned()],
            request.container(),
            request.deadline(),
        )
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
        let mut command = Command::new("docker");
        command.args(&arguments);
        run_docker_exec_without_deadline(&mut command, request.secret_environment())
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        docker_command_bounded(
            self.clock.clone(),
            request.public_arguments().iter().cloned(),
            "Docker file transfer",
            request.deadline(),
        )
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

    fn container_workdir_bounded(
        &self,
        container: &str,
        deadline: Duration,
    ) -> Result<String, EvalExecutionError> {
        let output = docker_output_bounded(
            self.clock.clone(),
            [
                "container".to_owned(),
                "inspect".to_owned(),
                "--format".to_owned(),
                "{{.Config.WorkingDir}}".to_owned(),
                container.to_owned(),
            ],
            container,
            Some(deadline),
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
        self.copy_archive_to_file_bounded(
            container,
            source,
            EvalExecutionPhase::CollectionHook,
            deadline,
        )
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        let removal = if let Some(timeout) = request.deadline() {
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
        compose_config_command_bounded(
            self.clock.clone(),
            arguments.into_iter().chain([
                "config".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
                "--no-interpolate".to_owned(),
            ]),
            request.project().as_str(),
            request.deadline(),
        )
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
        let deadline_ns = request
            .deadline()
            .map(|deadline| provider_deadline_ns(&self.clock, deadline));
        let container = compose_service_container(
            self.clock.clone(),
            request.project(),
            request.service(),
            remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?,
        )?;
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
                remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?,
            ),
        )
    }

    fn compose_copy_archive(
        &self,
        request: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let container = compose_service_container(
            self.clock.clone(),
            request.project(),
            request.service(),
            None,
        )?;
        self.copy_archive(&container, request.source())
    }

    fn compose_copy_archive_bounded(
        &self,
        request: &DockerComposeArchiveRequest,
        phase: EvalExecutionPhase,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let deadline_ns = provider_deadline_ns(&self.clock, deadline);
        let container = compose_service_container(
            self.clock.clone(),
            request.project(),
            request.service(),
            Some(remaining_provider_deadline(
                &self.clock,
                deadline_ns,
                request.project().as_str(),
            )?),
        )?;
        self.copy_archive_to_file_bounded(
            &container,
            request.source(),
            phase,
            remaining_provider_deadline(&self.clock, deadline_ns, request.project().as_str())?,
        )
    }

    fn compose_copy_into(
        &self,
        request: &DockerComposeCopyRequest,
    ) -> Result<(), EvalExecutionError> {
        let deadline_ns = request
            .deadline()
            .map(|deadline| provider_deadline_ns(&self.clock, deadline));
        let container = compose_service_container(
            self.clock.clone(),
            request.project(),
            request.service(),
            remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?,
        )?;
        let arguments = vec![
            "cp".to_owned(),
            request.source().to_owned(),
            format!("{container}:{}", request.destination()),
        ];
        docker_command_bounded(
            self.clock.clone(),
            arguments,
            request.project().as_str(),
            remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?,
        )
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
        let deadline_ns = request
            .deadline()
            .map(|deadline| provider_deadline_ns(&self.clock, deadline));
        let container = compose_service_container(
            self.clock.clone(),
            request.project(),
            request.service(),
            remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?,
        )?;
        let deadline =
            remaining_optional_provider_deadline(&self.clock, deadline_ns, request.project())?;
        let arguments = compose_stop_arguments(&container, deadline);
        docker_command_bounded(
            self.clock.clone(),
            arguments,
            request.project().as_str(),
            deadline,
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
            arguments,
            request.project().as_str(),
            request.deadline(),
        )
    }

    fn compose_owned_resources(
        &self,
        project: &ComposeProjectId,
        deadline: Duration,
    ) -> Result<super::OwnedComposeResources, EvalExecutionError> {
        let filters = compose_ownership_filters(project);
        let deadline_ns = self
            .clock
            .now_ns()
            .saturating_add(deadline.as_nanos().min(i64::MAX as u128) as i64);
        let resources = |kind: &str, extra: &[&str]| -> Result<Vec<String>, EvalExecutionError> {
            let mut arguments = vec![kind.to_owned(), "ls".to_owned(), "-q".to_owned()];
            arguments.extend(extra.iter().map(|value| (*value).to_owned()));
            for filter in &filters {
                arguments.extend(["--filter".to_owned(), filter.clone()]);
            }
            let output = docker_output_bounded(
                self.clock.clone(),
                arguments,
                project.as_str(),
                Some(remaining_provider_deadline(
                    &self.clock,
                    deadline_ns,
                    project.as_str(),
                )?),
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

fn remaining_provider_deadline(
    clock: &Rc<dyn Clock>,
    deadline_ns: i64,
    target: &str,
) -> Result<Duration, EvalExecutionError> {
    let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
    if remaining_ns <= 0 {
        return Err(EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "Docker provider deadline elapsed".to_owned(),
        });
    }
    Ok(Duration::from_nanos(remaining_ns as u64))
}

fn provider_deadline_ns(clock: &Rc<dyn Clock>, deadline: Duration) -> i64 {
    clock
        .now_ns()
        .saturating_add(deadline.as_nanos().min(i64::MAX as u128) as i64)
}

fn remaining_optional_provider_deadline(
    clock: &Rc<dyn Clock>,
    deadline_ns: Option<i64>,
    project: &ComposeProjectId,
) -> Result<Option<Duration>, EvalExecutionError> {
    deadline_ns
        .map(|deadline_ns| remaining_provider_deadline(clock, deadline_ns, project.as_str()))
        .transpose()
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

fn compose_config_command_bounded(
    clock: Rc<dyn Clock>,
    arguments: impl IntoIterator<Item = String>,
    target: &str,
    deadline: Option<Duration>,
) -> Result<Vec<u8>, EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let Some(deadline) = deadline else {
        return compose_command(arguments);
    };
    let output = tempfile::NamedTempFile::new()
        .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
    let output_writer = output
        .reopen()
        .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
    let mut child = Command::new("docker")
        .env("COMPOSE_DISABLE_ENV_FILE", "1")
        .args(&arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker compose config".to_owned()))?;
    let stdout = child.stdout.take().ok_or_else(|| {
        EvalExecutionError::ProcessFailure(
            "docker compose config did not provide stdout".to_owned(),
        )
    })?;
    // Drain stdout concurrently: a full pipe must not prevent deadline polling.
    let reader = std::thread::spawn(move || {
        copy_stream_bounded(
            stdout,
            output_writer,
            MAX_DOCKER_ARCHIVE_BYTES,
            "Docker Compose configuration",
        )
    });
    let mut process = DockerExecChild { child };
    let mut no_remove = |_target: &str, _: Duration| Ok(());
    let execution = drive_docker_exec(
        clock,
        &mut process,
        target,
        EvalExecutionPhase::CollectionHook,
        deadline,
        &mut no_remove,
    );
    let copied = reader.join().map_err(|_| {
        EvalExecutionError::ProcessFailure(
            "Docker Compose configuration reader panicked".to_owned(),
        )
    })?;
    copied?;
    execution?;
    read_file_bounded(
        output.path(),
        MAX_DOCKER_ARCHIVE_BYTES,
        "Docker Compose configuration",
    )
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

fn compose_stop_arguments(container: &str, deadline: Option<Duration>) -> Vec<String> {
    // `docker stop` otherwise waits for the daemon's ten-second default grace,
    // which can consume a collection window before its sidecar hooks start.
    let mut arguments = vec!["stop".to_owned()];
    if deadline.is_some() {
        // A CLI timeout only kills the client; the daemon keeps a graceful stop
        // running after that client is gone. Force the terminal state now.
        arguments.extend(["--time".to_owned(), "0".to_owned()]);
    }
    arguments.push(container.to_owned());
    arguments
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
    let child = Command::new("docker")
        .args(&arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker compose command".to_owned()))?;
    let mut process = DockerExecChild { child };
    let mut no_remove = |_target: &str, _: Duration| Ok(());
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

fn run_docker_exec_without_deadline(
    command: &mut Command,
    secrets: &std::collections::BTreeMap<super::EnvName, super::SecretValue>,
) -> Result<(), EvalExecutionError> {
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| {
            EvalExecutionError::ProcessSpawn("docker run planned Docker phase".to_owned())
        })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        EvalExecutionError::ProcessFailure(
            "docker run planned Docker phase did not provide stderr".to_owned(),
        )
    })?;
    let reader =
        thread::spawn(move || drain_output_bounded(stderr, MAX_DOCKER_COMMAND_OUTPUT_BYTES));
    let status = child
        .wait()
        .map_err(|error| EvalExecutionError::ProcessFailure(error.to_string()))?;
    let stderr = join_docker_output_reader(reader)?;
    if status.success() {
        Ok(())
    } else {
        Err(EvalExecutionError::ProcessFailure(format!(
            "docker run planned Docker phase: {}",
            redact_secret_values(&String::from_utf8_lossy(&stderr), secrets)
        )))
    }
}

fn docker_output_bounded(
    clock: Rc<dyn Clock>,
    arguments: impl IntoIterator<Item = String>,
    target: &str,
    deadline: Option<Duration>,
) -> Result<Vec<u8>, EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let mut child = Command::new("docker")
        .args(&arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker compose command".to_owned()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| EvalExecutionError::ProcessSpawn("docker resource output".to_owned()))?;
    let reader =
        thread::spawn(move || drain_output_bounded(stdout, MAX_DOCKER_COMMAND_OUTPUT_BYTES));
    let mut process = DockerExecChild { child };
    let Some(deadline) = deadline else {
        let status = process
            .child
            .wait()
            .map_err(|error| EvalExecutionError::ProcessFailure(error.to_string()))?;
        let output = join_docker_output_reader(reader)?;
        return if status.success() {
            Ok(output)
        } else {
            Err(EvalExecutionError::ProcessFailure(format!(
                "docker command exited with {status}"
            )))
        };
    };
    let mut no_remove = |_target: &str, _: Duration| Ok(());
    let result = drive_docker_exec(
        clock,
        &mut process,
        target,
        EvalExecutionPhase::CollectionHook,
        deadline,
        &mut no_remove,
    );
    match result {
        Ok(()) => join_docker_output_reader(reader),
        Err(error) => Err(error),
    }
}

fn join_docker_output_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
) -> Result<Vec<u8>, EvalExecutionError> {
    reader
        .join()
        .map_err(|_| {
            EvalExecutionError::ProcessFailure("docker output reader panicked".to_owned())
        })?
        .map_err(|error| EvalExecutionError::ProcessFailure(error.to_string()))
}

fn drain_output_bounded(mut source: impl Read, cap: usize) -> io::Result<Vec<u8>> {
    let mut captured = Vec::with_capacity(cap.min(8192));
    let mut buffer = [0_u8; 8192];
    let mut exceeded = false;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = cap.saturating_sub(captured.len());
        let retained = read.min(remaining);
        captured.extend_from_slice(&buffer[..retained]);
        exceeded |= retained < read;
    }
    if exceeded {
        return Err(io::Error::other(
            "docker command output exceeds the maximum size",
        ));
    }
    Ok(captured)
}

fn compose_service_container(
    clock: Rc<dyn Clock>,
    project: &ComposeProjectId,
    service: &super::ComposeServiceName,
    deadline: Option<Duration>,
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
    let output = docker_output_bounded(clock, arguments, project.as_str(), deadline)?;
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
        phase: EvalExecutionPhase,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        let mut archive = NamedTempFile::new().map_err(|error| {
            EvalExecutionError::ArtifactCollection(format!(
                "could not allocate bounded artifact archive: {error}"
            ))
        })?;
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
        let archive_writer = archive
            .reopen()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        // Drain stdout concurrently: a full pipe must not prevent deadline polling.
        let reader = std::thread::spawn(move || {
            copy_archive_stream_bounded(stdout, archive_writer, MAX_DOCKER_ARCHIVE_BYTES)
        });
        let mut process = DockerExecChild { child };
        let mut no_remove = |_target: &str, _: Duration| Ok(());
        let execution = drive_docker_exec(
            self.clock.clone(),
            &mut process,
            container,
            phase,
            deadline,
            &mut no_remove,
        );
        let copied = reader.join().map_err(|_| {
            EvalExecutionError::ArtifactCollection(
                "docker artifact archive reader panicked".to_owned(),
            )
        })?;
        copied?;
        execution?;
        archive
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        Ok(Box::new(BoundedArchiveReader { archive }))
    }
}

fn copy_archive_stream_bounded(
    source: impl Read,
    destination: impl Write,
    maximum_bytes: usize,
) -> Result<(), EvalExecutionError> {
    copy_stream_bounded(
        source,
        destination,
        maximum_bytes,
        "Docker artifact archive",
    )
}

fn copy_stream_bounded(
    mut source: impl Read,
    mut destination: impl Write,
    maximum_bytes: usize,
    subject: &str,
) -> Result<(), EvalExecutionError> {
    let mut buffer = [0_u8; 8192];
    let mut copied = 0_usize;
    loop {
        let remaining = maximum_bytes.saturating_sub(copied);
        let read_size = if remaining == 0 {
            1
        } else {
            remaining.min(buffer.len())
        };
        let read = source
            .read(&mut buffer[..read_size])
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if read == 0 {
            return Ok(());
        }
        if read > remaining {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "{subject} exceeds {maximum_bytes} bytes"
            )));
        }
        destination
            .write_all(&buffer[..read])
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        copied += read;
    }
}

fn read_file_bounded(
    path: &std::path::Path,
    maximum_bytes: usize,
    subject: &str,
) -> Result<Vec<u8>, EvalExecutionError> {
    let metadata = fs::metadata(path)
        .map_err(|error| EvalExecutionError::ProcessFailure(error.to_string()))?;
    if metadata.len() > maximum_bytes as u64 {
        return Err(EvalExecutionError::ProcessFailure(format!(
            "{subject} exceeds {maximum_bytes} bytes"
        )));
    }
    fs::read(path).map_err(|error| EvalExecutionError::ProcessFailure(error.to_string()))
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
    let mut no_remove = |_target: &str, _: Duration| Ok(());
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

fn ensure_public_network_bounded(
    clock: Rc<dyn Clock>,
    deadline: Duration,
) -> Result<(), EvalExecutionError> {
    let deadline_ns = provider_deadline_ns(&clock, deadline);
    let inspect = || -> Result<bool, EvalExecutionError> {
        match docker_output_bounded(
            clock.clone(),
            [
                "network".to_owned(),
                "inspect".to_owned(),
                PUBLIC_NETWORK_LEASE.to_owned(),
            ],
            PUBLIC_NETWORK_LEASE,
            Some(remaining_provider_deadline(
                &clock,
                deadline_ns,
                PUBLIC_NETWORK_LEASE,
            )?),
        ) {
            Ok(_) => Ok(true),
            Err(EvalExecutionError::ProcessFailure(_)) => Ok(false),
            Err(error) => Err(error),
        }
    };
    if inspect()? {
        return Ok(());
    }
    let create = docker_command_bounded(
        clock.clone(),
        [
            "network".to_owned(),
            "create".to_owned(),
            PUBLIC_NETWORK_LEASE.to_owned(),
        ],
        PUBLIC_NETWORK_LEASE,
        Some(remaining_provider_deadline(
            &clock,
            deadline_ns,
            PUBLIC_NETWORK_LEASE,
        )?),
    );
    if create.is_ok() || inspect()? {
        Ok(())
    } else {
        create
    }
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
    deadline: Option<Duration>,
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
    let request = match deadline {
        Some(deadline) => DockerCreateRequest::new(arguments)
            .with_network_lease(network_lease)
            .with_deadline(deadline),
        None => DockerCreateRequest::new(arguments).with_network_lease(network_lease),
    };
    runtime.create(&request)
}

fn prepare_workdir_with_deadline(
    runtime: &dyn DockerRuntime,
    container: &str,
    environment: &super::EnvironmentPlan,
    phase: &super::PhasePlan,
    execution_phase: EvalExecutionPhase,
    workdir: Option<&str>,
    network_lease: &str,
    deadline: Option<Duration>,
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
            prepare_workdir_arguments(workdir, user),
            Default::default(),
            Default::default(),
        )
        .with_phase(
            execution_phase,
            Some("root"),
            Some(workdir),
            network_lease,
            deadline.or(phase.timeout()),
        ),
    )
}

fn prepare_verifier_files_with_deadline(
    runtime: &dyn DockerRuntime,
    container: &str,
    network_lease: &str,
    deadline: Option<Duration>,
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
            deadline,
        ),
    )
}

fn clear_verifier_files(
    runtime: &dyn DockerRuntime,
    container: &str,
    network_lease: &str,
    deadline: Duration,
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
            Some(deadline),
        ),
    )
}

fn verifier_cleanup_deadline(deadline: &Option<Deadline>) -> Duration {
    deadline
        .as_ref()
        .and_then(|deadline| deadline.remaining().ok())
        .unwrap_or(super::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE)
}

fn transfer_verifier_artifacts(
    runtime: &dyn DockerRuntime,
    container: &str,
    source: &std::path::Path,
    workdir: Option<&str>,
    network_lease: &str,
    deadline: Option<&Deadline>,
) -> Result<(), EvalExecutionError> {
    let target = match workdir {
        Some(workdir) => workdir.to_owned(),
        None => match deadline.map(Deadline::remaining).transpose()? {
            Some(remaining) => runtime.container_workdir_bounded(container, remaining)?,
            None => runtime.container_workdir(container)?,
        },
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
            deadline.map(Deadline::remaining).transpose()?,
        ),
    )?;
    let request = DockerCopyRequest::new([
        "cp".to_owned(),
        format!("{}/.", source.display()),
        format!("{container}:{target}"),
    ]);
    let request = match deadline.map(Deadline::remaining).transpose()? {
        Some(deadline) => request.with_deadline(deadline),
        None => request,
    };
    runtime.copy(&request)
}

fn execute_planned_phase_with_deadline(
    runtime: &dyn DockerRuntime,
    container: &str,
    execution_phase: EvalExecutionPhase,
    command: &[String],
    environment: &super::EnvironmentPlan,
    phase: &super::PhasePlan,
    workdir: Option<&str>,
    secrets: &dyn SecretProvider,
    deadline: Option<Duration>,
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
            deadline.or(phase.timeout()),
        ),
    )
}

fn read_reward_with_runtime(
    runtime: &dyn DockerRuntime,
    container: &str,
    _: &TempDir,
    deadline: Option<&Deadline>,
) -> Result<RewardDocument, EvalExecutionError> {
    read_reward_archive_with_runtime(runtime, container, deadline)
}

fn read_reward_from_lease(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    deadline: Option<&Deadline>,
) -> Result<RewardDocument, EvalExecutionError> {
    let json = read_optional_service_file(lease, service, "/logs/verifier/reward.json", deadline)?;
    let text = read_optional_service_file(lease, service, "/logs/verifier/reward.txt", deadline)?;
    parse_reward(json.as_deref(), text.as_deref(), deadline)
}

const MAX_REWARD_BYTES: u64 = 1024 * 1024;
const MAX_REWARD_ARCHIVE_BYTES: u64 = MAX_REWARD_BYTES + 1024 * 1024;

fn parse_reward(
    json: Option<&[u8]>,
    text: Option<&[u8]>,
    deadline: Option<&Deadline>,
) -> Result<RewardDocument, EvalExecutionError> {
    if let Some(deadline) = deadline {
        deadline.remaining()?;
    }
    let reward = RewardDocument::parse(json, text)
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))?;
    if let Some(deadline) = deadline {
        deadline.remaining()?;
    }
    Ok(reward)
}

fn read_optional_service_file(
    lease: &mut dyn TaskEnvironmentLease,
    service: &super::ComposeServiceName,
    source: &str,
    deadline: Option<&Deadline>,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let remaining = deadline.map(Deadline::remaining).transpose()?;
    let archive = match lease.archive(ServiceArchiveRequest {
        service,
        source,
        deadline: remaining.unwrap_or(Duration::from_secs(10)),
        phase: EvalExecutionPhase::Verifier,
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
            if entry.size() > MAX_REWARD_BYTES {
                return Err(EvalExecutionError::ArtifactCollection(
                    "verifier reward exceeds the maximum size".to_owned(),
                ));
            }
            let mut bytes = Vec::new();
            read_reward_bytes(&mut entry, &mut bytes, deadline)?;
            return Ok(Some(bytes));
        }
    }
    Ok(None)
}

fn read_reward_archive_with_runtime(
    runtime: &dyn DockerRuntime,
    container: &str,
    deadline: Option<&Deadline>,
) -> Result<RewardDocument, EvalExecutionError> {
    let json =
        read_optional_reward_archive(runtime, container, "/logs/verifier/reward.json", deadline)?;
    let text =
        read_optional_reward_archive(runtime, container, "/logs/verifier/reward.txt", deadline)?;
    parse_reward(json.as_deref(), text.as_deref(), deadline)
}

fn read_optional_reward_archive(
    runtime: &dyn DockerRuntime,
    container: &str,
    source: &str,
    deadline: Option<&Deadline>,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let remaining = deadline
        .map(Deadline::remaining)
        .transpose()?
        .unwrap_or(Duration::from_secs(10));
    let archive = runtime.copy_archive_bounded(container, source, remaining);
    let archive = match archive {
        Ok(archive) => archive,
        Err(EvalExecutionError::ProcessFailure(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let expected = std::path::Path::new(source)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(EvalExecutionError::InvalidRecipe("verifier reward path"))?;
    let mut archive = tar::Archive::new(RewardArchiveReader::new(archive));
    for entry in archive
        .entries()
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?
    {
        let mut entry =
            entry.map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        let path = entry
            .path()
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if path.file_name().and_then(|name| name.to_str()) != Some(expected) {
            continue;
        }
        if !entry.header().entry_type().is_file() {
            return Err(EvalExecutionError::ArtifactCollection(
                "verifier reward must be a regular file".to_owned(),
            ));
        }
        if entry.size() > MAX_REWARD_BYTES {
            return Err(EvalExecutionError::ArtifactCollection(
                "verifier reward exceeds the maximum size".to_owned(),
            ));
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        read_reward_bytes(&mut entry, &mut bytes, deadline)?;
        return Ok(Some(bytes));
    }
    Ok(None)
}

struct RewardArchiveReader<R> {
    source: R,
    bytes: u64,
}

impl<R> RewardArchiveReader<R> {
    fn new(source: R) -> Self {
        Self { source, bytes: 0 }
    }
}

impl<R: Read> Read for RewardArchiveReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.source.read(buffer)?;
        self.bytes = self
            .bytes
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::other("verifier reward archive exceeds the maximum size"))?;
        if self.bytes > MAX_REWARD_ARCHIVE_BYTES {
            return Err(io::Error::other(
                "verifier reward archive exceeds the maximum size",
            ));
        }
        Ok(read)
    }
}

fn read_reward_bytes(
    source: &mut dyn Read,
    destination: &mut Vec<u8>,
    deadline: Option<&Deadline>,
) -> Result<(), EvalExecutionError> {
    let mut buffer = [0_u8; 8192];
    loop {
        if let Some(deadline) = deadline {
            deadline.remaining()?;
        }
        let read = source
            .read(&mut buffer)
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
        if read == 0 {
            return Ok(());
        }
        destination.extend_from_slice(&buffer[..read]);
        if destination.len() as u64 > MAX_REWARD_BYTES {
            return Err(EvalExecutionError::ArtifactCollection(
                "verifier reward exceeds the maximum size".to_owned(),
            ));
        }
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
    let cleanup_clock = clock.clone();
    let mut remove = move |container: &str, deadline| {
        remove_timed_out_container(cleanup_clock.clone(), container, deadline)
    };
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
    F: for<'a> FnMut(&'a str, Duration) -> Result<(), EvalExecutionError>,
{
    // Real provider processes are polled synchronously. This keeps the
    // synchronous evaluator usable from a Tokio task without re-entering a
    // Tokio runtime. The virtual path below retains SimClock's event pump and
    // deterministic registration order.
    if !clock.is_virtual() {
        return drive_real_docker_exec(clock, process, container, phase, timeout, remove);
    }
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(EvalExecutionError::RuntimeContext(
            "virtual-clock Docker operation from an entered Tokio runtime",
        ));
    }
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

fn drive_real_docker_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str, Duration) -> Result<(), EvalExecutionError>,
{
    let deadline = clock
        .now_ns()
        .saturating_add(timeout.as_nanos().min(i64::MAX as u128) as i64);
    loop {
        if clock.now_ns() >= deadline {
            return terminate_timed_out_real_exec(
                clock, process, container, phase, timeout, false, remove,
            );
        }
        let state = process.try_wait().map_err(|reason| {
            EvalExecutionError::ProcessFailure(format!("docker exec process check: {reason}"))
        })?;
        if clock.now_ns() >= deadline {
            return terminate_timed_out_real_exec(
                clock,
                process,
                container,
                phase,
                timeout,
                state.is_terminal(),
                remove,
            );
        }
        match state {
            DockerExecState::Succeeded => return Ok(()),
            DockerExecState::Failed(status) => {
                return Err(EvalExecutionError::ProcessFailure(format!(
                    "docker exec exited with {status}"
                )));
            }
            DockerExecState::Running => {
                let remaining_ns = deadline.saturating_sub(clock.now_ns());
                std::thread::sleep(Duration::from_nanos(
                    remaining_ns.min(DOCKER_EXEC_POLL_NS) as u64
                ));
            }
        }
    }
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
    F: for<'a> FnMut(&'a str, Duration) -> Result<(), EvalExecutionError>,
{
    let deadline = clock
        .now_ns()
        .saturating_add(timeout.as_nanos().min(i64::MAX as u128) as i64);
    loop {
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(
                clock, process, container, phase, timeout, false, remove,
            )
            .await;
        }
        let state = process.try_wait().map_err(|reason| {
            EvalExecutionError::ProcessFailure(format!("docker exec process check: {reason}"))
        })?;
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(
                clock,
                process,
                container,
                phase,
                timeout,
                state.is_terminal(),
                remove,
            )
            .await;
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

async fn terminate_timed_out_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    has_observed_terminal_client: bool,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str, Duration) -> Result<(), EvalExecutionError>,
{
    let cleanup_deadline = clock.now_ns().saturating_add(TERMINAL_DOCKER_CLEANUP_NS);
    let kill = (!has_observed_terminal_client).then(|| process.kill());
    let reap = wait_for_docker_client_exit(clock.clone(), process, cleanup_deadline).await;
    let removal_deadline = remaining_provider_deadline(&clock, cleanup_deadline, container)
        .unwrap_or(Duration::from_nanos(1));
    let removal = remove(container, removal_deadline);
    removal?;
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

const TERMINAL_DOCKER_CLEANUP_NS: i64 = 10_000_000_000;

fn terminate_timed_out_real_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    has_observed_terminal_client: bool,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str, Duration) -> Result<(), EvalExecutionError>,
{
    let cleanup_deadline = clock.now_ns().saturating_add(TERMINAL_DOCKER_CLEANUP_NS);
    let kill = (!has_observed_terminal_client).then(|| process.kill());
    let reap = wait_for_real_docker_client_exit(clock.clone(), process, cleanup_deadline);
    let removal_deadline = remaining_provider_deadline(&clock, cleanup_deadline, container)
        .unwrap_or(Duration::from_nanos(1));
    remove(container, removal_deadline)?;
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

fn wait_for_real_docker_client_exit<P: DockerExecProcess>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    deadline_ns: i64,
) -> Result<(), String> {
    loop {
        match process.try_wait()? {
            DockerExecState::Succeeded | DockerExecState::Failed(_) => return Ok(()),
            DockerExecState::Running => {}
        }
        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return Err("timed out reaping docker exec client".to_owned());
        }
        std::thread::sleep(Duration::from_nanos(
            remaining_ns.min(DOCKER_EXEC_POLL_NS) as u64
        ));
    }
}

async fn wait_for_docker_client_exit<P: DockerExecProcess>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    deadline_ns: i64,
) -> Result<(), String> {
    loop {
        match process.try_wait()? {
            DockerExecState::Succeeded | DockerExecState::Failed(_) => return Ok(()),
            DockerExecState::Running => {}
        }
        let remaining_ns = deadline_ns.saturating_sub(clock.now_ns());
        if remaining_ns <= 0 {
            return Err("timed out reaping docker exec client".to_owned());
        }
        clock
            .clone()
            .sleep(remaining_ns.min(DOCKER_EXEC_POLL_NS))
            .await;
    }
}

fn remove_timed_out_container(
    clock: Rc<dyn Clock>,
    container: &str,
    deadline: Duration,
) -> Result<(), EvalExecutionError> {
    let deadline_ns = provider_deadline_ns(&clock, deadline);
    docker_remove_bounded(
        clock.clone(),
        vec!["rm", "--force", "--volumes", container],
        remaining_provider_deadline(&clock, deadline_ns, container)?,
    )?;
    let inspect_deadline = remaining_provider_deadline(&clock, deadline_ns, container)?;
    match docker_output_bounded(
        clock,
        [
            "container".to_owned(),
            "inspect".to_owned(),
            container.to_owned(),
        ],
        container,
        Some(inspect_deadline),
    ) {
        Ok(_) => Err(EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: "docker container inspect found the container after forced removal".to_owned(),
        }),
        Err(EvalExecutionError::ProcessFailure(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

fn reports_absent_container(stderr: &[u8]) -> bool {
    let diagnostic = String::from_utf8_lossy(stderr).to_ascii_lowercase();
    diagnostic.contains("no such container") || diagnostic.contains("no such object")
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        collections::VecDeque,
        io::{self, Read},
        process::Command,
        rc::Rc,
        time::Duration,
    };

    use super::{
        DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecProcess,
        DockerExecState, DockerRemoveRequest, DockerRuntime, DockerStartRequest,
        EvalExecutionError, EvalExecutionPhase, classify_bounded_remove_result,
        compose_ownership_filters, compose_stop_arguments, copy_archive_stream_bounded,
        docker_container_name, docker_image_name, drain_output_bounded, drive_docker_exec,
        ensure_network_exists, read_optional_reward_archive, read_reward_with_runtime,
        redact_secret_values, reports_absent_container, run_docker_exec_without_deadline,
    };
    use crate::{
        clock::SimClock,
        eval::{ComposeProjectId, ProviderCapabilities},
    };

    #[tokio::test]
    async fn virtual_bounded_exec_refuses_an_entered_tokio_runtime() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(clock.clone(), [DockerExecState::Running]);
        let mut remove = |_: &str, _: Duration| Ok(());
        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_secs(1),
            &mut remove,
        );
        assert_eq!(
            result,
            Err(EvalExecutionError::RuntimeContext(
                "virtual-clock Docker operation from an entered Tokio runtime"
            ))
        );
    }

    #[tokio::test]
    async fn real_bounded_exec_completes_inside_an_entered_tokio_runtime() {
        let clock = crate::clock::RealClock::new();
        let mut process = CompletedDockerExec;
        let mut remove = |_: &str, _: Duration| Ok(());

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_secs(1),
            &mut remove,
        );

        assert_eq!(result, Ok(()));
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
    fn bounded_compose_stop_forces_the_container_before_the_collection_deadline() {
        assert_eq!(
            compose_stop_arguments("main-container", Some(Duration::from_millis(200))),
            vec![
                "stop".to_owned(),
                "--time".to_owned(),
                "0".to_owned(),
                "main-container".to_owned(),
            ]
        );
        assert_eq!(
            compose_stop_arguments("main-container", Some(Duration::from_millis(1900))),
            vec![
                "stop".to_owned(),
                "--time".to_owned(),
                "0".to_owned(),
                "main-container".to_owned(),
            ]
        );
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
    fn bounded_archive_copy_rejects_bytes_before_the_temporary_file_exceeds_its_cap() {
        struct EndlessArchive;

        impl Read for EndlessArchive {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                buffer.fill(b'x');
                Ok(buffer.len())
            }
        }

        let cap = 16 * 1024;
        let mut archive = EndlessArchive;
        let mut temporary = Vec::new();

        let error = copy_archive_stream_bounded(&mut archive, &mut temporary, cap)
            .expect_err("the Docker archive must be capped before it can exhaust temporary disk");

        assert!(
            matches!(error, EvalExecutionError::ArtifactCollection(message) if message.contains("exceeds"))
        );
        assert!(temporary.len() <= cap);
    }

    #[test]
    fn bounded_command_output_keeps_draining_after_its_capture_cap() {
        struct ChunkedOutput {
            chunks: VecDeque<Vec<u8>>,
            reads: Rc<Cell<usize>>,
        }

        impl Read for ChunkedOutput {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.reads.set(self.reads.get() + 1);
                let Some(chunk) = self.chunks.pop_front() else {
                    return Ok(0);
                };
                buffer[..chunk.len()].copy_from_slice(&chunk);
                Ok(chunk.len())
            }
        }

        let reads = Rc::new(Cell::new(0));
        let output = ChunkedOutput {
            chunks: VecDeque::from([vec![b'a'; 4], vec![b'b'; 4], vec![b'c'; 4]]),
            reads: reads.clone(),
        };

        let error = drain_output_bounded(output, 5)
            .expect_err("the capture cap must reject excess command output after draining it");

        assert!(error.to_string().contains("exceeds"));
        assert_eq!(reads.get(), 4);
    }

    #[test]
    fn unbounded_docker_exec_discards_stdout_and_bounds_stderr() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "head -c 1048577 /dev/zero; head -c 1048577 /dev/zero >&2; exit 1",
        ]);

        let error = run_docker_exec_without_deadline(&mut command, &Default::default())
            .expect_err("Docker exec diagnostics must remain bounded");

        assert!(matches!(
            error,
            EvalExecutionError::ProcessFailure(message)
                if message.contains("docker command output exceeds the maximum size")
        ));
    }

    #[test]
    fn legacy_reward_archive_rejects_a_verifier_authored_symlink() {
        struct ArchiveRuntime;

        impl DockerRuntime for ArchiveRuntime {
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::none()
            }

            fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not build")
            }

            fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not create")
            }

            fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not start")
            }

            fn exec(&self, _: &super::DockerExecRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not execute")
            }

            fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not copy")
            }

            fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
                let mut archive = tar::Builder::new(Vec::new());
                archive
                    .append_link(
                        &mut tar::Header::new_gnu(),
                        "reward.json",
                        "/verifier-controlled",
                    )
                    .unwrap();
                Ok(Box::new(io::Cursor::new(archive.into_inner().unwrap())))
            }

            fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not remove")
            }
        }

        let error = read_optional_reward_archive(
            &ArchiveRuntime,
            "verifier",
            "/logs/verifier/reward.json",
            None,
        )
        .expect_err("reward symlinks must never be followed from a legacy archive");

        assert!(
            matches!(error, EvalExecutionError::ArtifactCollection(message) if message.contains("regular file"))
        );
    }

    #[test]
    fn standard_reward_collection_reads_the_bounded_archive_without_host_copy() {
        struct ArchiveRuntime;

        impl DockerRuntime for ArchiveRuntime {
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::none()
            }

            fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not build")
            }

            fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not create")
            }

            fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not start")
            }

            fn exec(&self, _: &super::DockerExecRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not execute")
            }

            fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
                panic!("standard verifier rewards must not be copied onto the host")
            }

            fn copy_archive(
                &self,
                _: &str,
                source: &str,
            ) -> Result<Box<dyn Read>, EvalExecutionError> {
                let name = std::path::Path::new(source)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                let mut archive = tar::Builder::new(Vec::new());
                let body = if name == "reward.json" {
                    b"{\"reward\":1}".as_slice()
                } else {
                    b"".as_slice()
                };
                let mut header = tar::Header::new_gnu();
                header.set_size(body.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive.append_data(&mut header, name, body).unwrap();
                Ok(Box::new(io::Cursor::new(archive.into_inner().unwrap())))
            }

            fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not remove")
            }
        }

        let workspace = tempfile::tempdir().unwrap();
        let reward = read_reward_with_runtime(&ArchiveRuntime, "verifier", &workspace, None)
            .expect("standard verifier rewards must use the bounded archive path");

        assert_eq!(reward.metrics.get("reward"), Some(&1.0));
    }

    #[test]
    fn standard_reward_collection_uses_a_bounded_archive_without_an_authored_timeout() {
        struct BoundedArchiveRuntime;

        impl DockerRuntime for BoundedArchiveRuntime {
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::none()
            }

            fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not build")
            }

            fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not create")
            }

            fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not start")
            }

            fn exec(&self, _: &super::DockerExecRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not execute")
            }

            fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not copy")
            }

            fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
                panic!("standard verifier rewards must use a bounded archive")
            }

            fn copy_archive_bounded(
                &self,
                _: &str,
                source: &str,
                deadline: Duration,
            ) -> Result<Box<dyn Read>, EvalExecutionError> {
                assert_eq!(deadline, Duration::from_secs(10));
                if source.ends_with("reward.txt") {
                    return Err(EvalExecutionError::ProcessFailure(
                        "optional reward is absent".to_owned(),
                    ));
                }
                let mut archive = tar::Builder::new(Vec::new());
                let body = b"{\"reward\":1}";
                let mut header = tar::Header::new_gnu();
                header.set_size(body.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive
                    .append_data(&mut header, "reward.json", body.as_slice())
                    .unwrap();
                Ok(Box::new(io::Cursor::new(archive.into_inner().unwrap())))
            }

            fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
                unreachable!("reward archive test does not remove")
            }
        }

        let workspace = tempfile::tempdir().unwrap();
        let reward = read_reward_with_runtime(&BoundedArchiveRuntime, "verifier", &workspace, None)
            .expect("an absent optional reward.txt must not reject reward.json");

        assert_eq!(reward.metrics.get("reward"), Some(&1.0));
    }

    #[test]
    fn completed_command_observed_after_deadline_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(clock.clone(), [DockerExecState::Succeeded])
            .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str, _: Duration| {
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
    fn failed_command_observed_after_deadline_also_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(
            clock.clone(),
            [DockerExecState::Failed("exit status: 1".to_owned())],
        )
        .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str, _: Duration| {
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
        let mut remove = |_: &str, _: Duration| {
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
    }

    #[test]
    fn reap_deadline_returns_typed_terminal_uncertainty_after_removal() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(clock.clone(), [DockerExecState::Running]);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str, _: Duration| {
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
        terminal_observed: bool,
        was_killed: Cell<bool>,
    }

    struct CompletedDockerExec;

    impl DockerExecProcess for CompletedDockerExec {
        fn try_wait(&mut self) -> Result<DockerExecState, String> {
            Ok(DockerExecState::Succeeded)
        }

        fn kill(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    impl FakeDockerExec {
        fn new(clock: Rc<SimClock>, states: impl IntoIterator<Item = DockerExecState>) -> Self {
            Self {
                clock,
                states: states.into_iter().collect(),
                advance_terminal_to: None,
                kill_fails: false,
                terminal_observed: false,
                was_killed: Cell::new(false),
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
    }
}
