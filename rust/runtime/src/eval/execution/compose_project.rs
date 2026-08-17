// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! State-bounded lifecycle management for a task-owned Compose project.

use std::{collections::BTreeSet, io::Read, rc::Rc, time::Duration};

use crate::clock::Clock;

use super::task_environment::{ServiceArchiveRequest, ServiceExecRequest, TaskEnvironmentLease};
use super::{
    ComposeProjectId, ComposeProjectPlan, ComposeServiceName, DockerComposeArchiveRequest,
    DockerComposeBuildRequest, DockerComposeCopyRequest, DockerComposeDownRequest,
    DockerComposeExecRequest, DockerComposeRuntime, DockerComposeStopRequest,
    DockerComposeUpRequest, DockerRemoveRequest, EvalExecutionError, OwnedComposeResources,
};

/// Terminal benchmark failures get at most this host-side cleanup allowance.
///
/// This is intentionally independent from an expired phase deadline: cleanup must
/// still reclaim exact task-owned resources, but must never turn a sub-second
/// phase timeout into the former minute-long teardown wait.
pub(crate) const TERMINAL_COMPOSE_CLEANUP_DEADLINE: Duration = Duration::from_secs(10);

/// One host deadline shared by terminal cleanup provider operations.
struct CleanupDeadline {
    clock: Rc<dyn Clock>,
    deadline_ns: i64,
}

impl CleanupDeadline {
    fn new(clock: Rc<dyn Clock>, timeout: Duration) -> Self {
        let timeout_ns = timeout.as_nanos().min(i64::MAX as u128) as i64;
        let deadline_ns = clock.now_ns().saturating_add(timeout_ns);
        Self { clock, deadline_ns }
    }
    fn remaining(&self) -> Result<Duration, EvalExecutionError> {
        let remaining_ns = self.deadline_ns.saturating_sub(self.clock.now_ns());
        if remaining_ns <= 0 {
            return Err(EvalExecutionError::ContainerTeardown {
                container: "Compose project".to_owned(),
                reason: "terminal cleanup deadline elapsed".to_owned(),
            });
        }
        Ok(Duration::from_nanos(remaining_ns as u64))
    }
}

/// The monotonic lifecycle state of a Compose task environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComposeLeaseState {
    Reserved,
    Built,
    Started,
    MainStopped,
    Down,
}

/// A registered task-owned Compose project lease.
pub(crate) struct ComposeProjectLease<'a> {
    runtime: &'a dyn DockerComposeRuntime,
    clock: Rc<dyn Clock>,
    project: ComposeProjectId,
    project_directory: String,
    services: BTreeSet<ComposeServiceName>,
    main: ComposeServiceName,
    main_image: String,
    build_timeout: std::time::Duration,
    startup_timeout: std::time::Duration,
    state: ComposeLeaseState,
    recorded: OwnedComposeResources,
    terminal_cleanup_deadline: Option<Duration>,
}

impl<'a> ComposeProjectLease<'a> {
    pub(crate) fn reserve_with_clock(
        runtime: &'a dyn DockerComposeRuntime,
        clock: Rc<dyn Clock>,
        plan: &ComposeProjectPlan,
        source_digest: &str,
        project_directory: impl Into<String>,
        main_image: impl Into<String>,
    ) -> Result<Self, EvalExecutionError> {
        let prefix: String = source_digest
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(12)
            .collect();
        if prefix.is_empty() {
            return Err(EvalExecutionError::InvalidRecipe("source digest"));
        }
        let project =
            ComposeProjectId::new(format!("aiperf-{prefix}-{}", uuid::Uuid::new_v4().simple()));
        Ok(Self {
            runtime,
            clock,
            project,
            project_directory: project_directory.into(),
            services: plan.services().clone(),
            main: ComposeServiceName::main(),
            main_image: main_image.into(),
            build_timeout: plan.build_timeout(),
            startup_timeout: plan.startup_timeout(),
            state: ComposeLeaseState::Reserved,
            recorded: OwnedComposeResources::default(),
            terminal_cleanup_deadline: None,
        })
    }

    pub(crate) fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    pub(crate) fn start(&mut self) -> Result<(), EvalExecutionError> {
        if self.state != ComposeLeaseState::Reserved {
            return Err(EvalExecutionError::InvalidRecipe("Compose lease state"));
        }
        let build = DockerComposeBuildRequest::new(self.project.clone(), &self.project_directory)
            .with_deadline(self.build_timeout);
        if let Err(error) = self.runtime.compose_build(&build) {
            return self.finish_start_failure(error);
        }
        self.state = ComposeLeaseState::Built;
        // Startup is one provider window: discovery is part of proving that
        // `compose up --wait` created only this project, not a fresh timeout.
        let startup_deadline = CleanupDeadline::new(self.clock.clone(), self.startup_timeout);
        let up_deadline = match startup_deadline.remaining() {
            Ok(deadline) => deadline,
            Err(error) => return self.finish_start_failure(error),
        };
        let up = DockerComposeUpRequest::new(self.project.clone(), &self.project_directory)
            .with_deadline(up_deadline);
        if let Err(error) = self.runtime.compose_up(&up) {
            return self.finish_start_failure(error);
        }
        let discovery_deadline = match startup_deadline.remaining() {
            Ok(deadline) => deadline,
            Err(error) => return self.finish_start_failure(error),
        };
        match self
            .runtime
            .compose_owned_resources(&self.project, discovery_deadline)
        {
            Ok(resources) => {
                self.recorded = resources;
                self.state = ComposeLeaseState::Started;
                Ok(())
            }
            Err(error) => self.finish_start_failure(error),
        }
    }

    fn finish_start_failure(
        &mut self,
        phase_error: EvalExecutionError,
    ) -> Result<(), EvalExecutionError> {
        let cleanup_timeout = *self
            .terminal_cleanup_deadline
            .get_or_insert(TERMINAL_COMPOSE_CLEANUP_DEADLINE);
        let cleanup_deadline = CleanupDeadline::new(self.clock.clone(), cleanup_timeout);
        self.record_after_failure(&cleanup_deadline);
        match self.teardown_with_cleanup_deadline(&cleanup_deadline, true) {
            Ok(()) => Err(phase_error),
            Err(cleanup_error) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: format!("{phase_error}; cleanup: {cleanup_error}"),
            }),
        }
    }

    fn record_after_failure(&mut self, cleanup_deadline: &CleanupDeadline) {
        if let Ok(deadline) = cleanup_deadline.remaining()
            && let Ok(resources) = self
                .runtime
                .compose_owned_resources(&self.project, deadline)
        {
            self.recorded = resources;
        }
    }

    fn ensure_started(&self) -> Result<(), EvalExecutionError> {
        match self.state {
            ComposeLeaseState::Started | ComposeLeaseState::MainStopped => Ok(()),
            _ => Err(EvalExecutionError::InvalidRecipe("Compose lease state")),
        }
    }

    fn force_recorded_resources(
        &self,
        deadline: &CleanupDeadline,
    ) -> Result<(), EvalExecutionError> {
        let mut first_error = None;
        for container in self.recorded.containers() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["rm", "--force", "--volumes", container])
                    .with_deadline(deadline.remaining()?),
            );
            first_error = first_error.or(removal.err());
        }
        for network in self.recorded.networks() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["network", "rm", network])
                    .with_deadline(deadline.remaining()?),
            );
            first_error = first_error.or(removal.err());
        }
        for volume in self.recorded.volumes() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["volume", "rm", "--force", volume])
                    .with_deadline(deadline.remaining()?),
            );
            first_error = first_error.or(removal.err());
        }
        first_error.map_or(Ok(()), Err)
    }

    fn teardown_with_terminal_failure(
        &mut self,
        deadline: Duration,
        is_terminal_failure: bool,
    ) -> Result<(), EvalExecutionError> {
        if self.state == ComposeLeaseState::Down {
            return Ok(());
        }
        let cleanup_deadline = CleanupDeadline::new(self.clock.clone(), deadline);
        self.teardown_with_cleanup_deadline(&cleanup_deadline, is_terminal_failure)
    }

    fn teardown_with_cleanup_deadline(
        &mut self,
        cleanup_deadline: &CleanupDeadline,
        is_terminal_failure: bool,
    ) -> Result<(), EvalExecutionError> {
        if self.state == ComposeLeaseState::Down {
            return Ok(());
        }
        let request = DockerComposeDownRequest::new(self.project.clone(), &self.project_directory)
            .with_deadline(cleanup_deadline.remaining()?);
        let request = if is_terminal_failure {
            request.with_terminal_failure()
        } else {
            request
        };
        let down = self.runtime.compose_down(&request);
        let remaining_after_down = self
            .runtime
            .compose_owned_resources(&self.project, cleanup_deadline.remaining()?);
        if let Ok(resources) = &remaining_after_down {
            self.recorded = resources.clone();
        }
        let needs_force = matches!(&remaining_after_down, Ok(resources) if resources != &OwnedComposeResources::default());
        let forced = if needs_force {
            self.force_recorded_resources(cleanup_deadline)
        } else {
            Ok(())
        };
        let remaining = if needs_force {
            self.runtime
                .compose_owned_resources(&self.project, cleanup_deadline.remaining()?)
        } else {
            remaining_after_down
        };
        let result = match (down, forced, remaining) {
            (Err(down), Err(forced), Err(discovery)) => {
                Err(EvalExecutionError::ContainerTeardown {
                    container: self.project.as_str().to_owned(),
                    reason: format!(
                        "Compose down: {down}; forced removal: {forced}; discovery: {discovery}"
                    ),
                })
            }
            (Err(down), Err(forced), _) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: format!("Compose down: {down}; forced removal: {forced}"),
            }),
            (Err(down), _, Err(discovery)) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: format!("Compose down: {down}; discovery: {discovery}"),
            }),
            (Err(error), _, _) => Err(error),
            (Ok(_), Err(forced), Err(discovery)) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: format!("forced removal: {forced}; discovery: {discovery}"),
            }),
            (Ok(_), Err(error), _) => Err(error),
            (Ok(_), Ok(_), Ok(resources)) if resources == OwnedComposeResources::default() => {
                Ok(())
            }
            (Ok(_), Ok(_), Ok(_)) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: "Compose resources remain after teardown".to_owned(),
            }),
            (Ok(_), Ok(_), Err(error)) => Err(error),
        };
        if result.is_ok() {
            self.state = ComposeLeaseState::Down;
        }
        result
    }
}

impl TaskEnvironmentLease for ComposeProjectLease<'_> {
    fn main_service(&self) -> &ComposeServiceName {
        &self.main
    }
    fn exec(&mut self, request: ServiceExecRequest<'_>) -> Result<(), EvalExecutionError> {
        self.ensure_started()?;
        if !self.services.contains(request.service) {
            return Err(EvalExecutionError::InvalidRecipe("Compose service"));
        }
        self.runtime.compose_exec(
            &DockerComposeExecRequest::new(
                self.project.clone(),
                request.service.clone(),
                request.arguments.iter().cloned(),
                request.public_environment,
                request.secret_environment,
            )
            .with_phase(
                request.phase,
                request.user,
                request.workdir,
                request.deadline,
            ),
        )
    }
    fn archive(
        &mut self,
        request: ServiceArchiveRequest<'_>,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.ensure_started()?;
        if !self.services.contains(request.service) {
            return Err(EvalExecutionError::InvalidRecipe("Compose service"));
        }
        self.runtime.compose_copy_archive_bounded(
            &DockerComposeArchiveRequest::new(
                self.project.clone(),
                request.service.clone(),
                request.source,
            ),
            request.phase,
            request.deadline,
        )
    }
    fn copy_into(
        &mut self,
        service: &ComposeServiceName,
        source: &str,
        destination: &str,
    ) -> Result<(), EvalExecutionError> {
        self.ensure_started()?;
        if !self.services.contains(service) {
            return Err(EvalExecutionError::InvalidRecipe("Compose service"));
        }
        self.runtime
            .compose_copy_into(&DockerComposeCopyRequest::new(
                self.project.clone(),
                service.clone(),
                source,
                destination,
            ))
    }
    fn copy_into_bounded(
        &mut self,
        service: &ComposeServiceName,
        source: &str,
        destination: &str,
        deadline: Duration,
    ) -> Result<(), EvalExecutionError> {
        self.ensure_started()?;
        if !self.services.contains(service) {
            return Err(EvalExecutionError::InvalidRecipe("Compose service"));
        }
        self.runtime.compose_copy_into(
            &DockerComposeCopyRequest::new(
                self.project.clone(),
                service.clone(),
                source,
                destination,
            )
            .with_deadline(deadline),
        )
    }
    fn stop_main(&mut self, deadline: Duration) -> Result<(), EvalExecutionError> {
        self.ensure_started()?;
        self.runtime.compose_stop_service_bounded(
            &DockerComposeStopRequest::new(self.project.clone(), self.main.clone())
                .with_deadline(deadline),
        )?;
        self.state = ComposeLeaseState::MainStopped;
        Ok(())
    }
    fn main_image_id(&self) -> Result<&str, EvalExecutionError> {
        Ok(&self.main_image)
    }
    fn teardown(&mut self) -> Result<(), EvalExecutionError> {
        match self.terminal_cleanup_deadline {
            Some(deadline) => self.teardown_with_terminal_failure(deadline, true),
            None => self.teardown_with_terminal_failure(Duration::from_secs(60), false),
        }
    }

    fn teardown_after_terminal_failure(
        &mut self,
        deadline: Duration,
    ) -> Result<(), EvalExecutionError> {
        let deadline = *self.terminal_cleanup_deadline.get_or_insert(deadline);
        self.teardown_with_terminal_failure(deadline, true)
    }
}

impl Drop for ComposeProjectLease<'_> {
    fn drop(&mut self) {
        if self.state != ComposeLeaseState::Down {
            let _ = self.teardown();
        }
    }
}
