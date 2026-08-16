// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! State-bounded lifecycle management for a task-owned Compose project.

use std::{
    collections::BTreeSet,
    io::Read,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use super::task_environment::{
    ServiceArchiveRequest, ServiceExecRequest, ServiceHandle, TaskEnvironmentLease,
};
use super::{
    ComposeProjectId, ComposeProjectPlan, ComposeServiceName, DockerComposeArchiveRequest,
    DockerComposeBuildRequest, DockerComposeDownRequest, DockerComposeExecRequest,
    DockerComposeRuntime, DockerComposeStopRequest, DockerComposeUpRequest, DockerRemoveRequest,
    EvalExecutionError, OwnedComposeResources,
};

static NEXT_COMPOSE_RUN_ID: AtomicU64 = AtomicU64::new(1);

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
pub(crate) struct ComposeProjectLease {
    runtime: Rc<dyn DockerComposeRuntime>,
    project: ComposeProjectId,
    project_directory: String,
    services: BTreeSet<ComposeServiceName>,
    main: ComposeServiceName,
    main_image: String,
    build_timeout: std::time::Duration,
    startup_timeout: std::time::Duration,
    state: ComposeLeaseState,
    recorded: OwnedComposeResources,
}

impl ComposeProjectLease {
    pub(crate) fn reserve(
        runtime: Rc<dyn DockerComposeRuntime>,
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
        let run = NEXT_COMPOSE_RUN_ID.fetch_add(1, Ordering::Relaxed);
        let project =
            ComposeProjectId::new(format!("aiperf-{prefix}-{}-{run}", std::process::id()));
        Ok(Self {
            runtime,
            project,
            project_directory: project_directory.into(),
            services: plan.services().clone(),
            main: ComposeServiceName::main(),
            main_image: main_image.into(),
            build_timeout: plan.build_timeout(),
            startup_timeout: plan.startup_timeout(),
            state: ComposeLeaseState::Reserved,
            recorded: OwnedComposeResources::default(),
        })
    }

    pub(crate) fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    pub(crate) const fn state(&self) -> ComposeLeaseState {
        self.state
    }

    pub(crate) fn start(&mut self) -> Result<(), EvalExecutionError> {
        if self.state != ComposeLeaseState::Reserved {
            return Err(EvalExecutionError::InvalidRecipe("Compose lease state"));
        }
        let build = DockerComposeBuildRequest::new(self.project.clone(), &self.project_directory)
            .with_deadline(self.build_timeout);
        if let Err(error) = self.runtime.compose_build(&build) {
            self.record_after_failure();
            return self.finish_start_failure(error);
        }
        self.state = ComposeLeaseState::Built;
        let up = DockerComposeUpRequest::new(self.project.clone(), &self.project_directory)
            .with_deadline(self.startup_timeout);
        if let Err(error) = self.runtime.compose_up(&up) {
            self.record_after_failure();
            return self.finish_start_failure(error);
        }
        match self.runtime.compose_owned_resources(&self.project) {
            Ok(resources) => {
                self.recorded = resources;
                self.state = ComposeLeaseState::Started;
                Ok(())
            }
            Err(error) => {
                self.record_after_failure();
                self.finish_start_failure(error)
            }
        }
    }

    fn finish_start_failure(
        &mut self,
        phase_error: EvalExecutionError,
    ) -> Result<(), EvalExecutionError> {
        match self.teardown() {
            Ok(()) => Err(phase_error),
            Err(cleanup_error) => Err(EvalExecutionError::ContainerTeardown {
                container: self.project.as_str().to_owned(),
                reason: format!("{phase_error}; cleanup: {cleanup_error}"),
            }),
        }
    }

    fn record_after_failure(&mut self) {
        if let Ok(resources) = self.runtime.compose_owned_resources(&self.project) {
            self.recorded = resources;
        }
    }

    fn ensure_started(&self) -> Result<(), EvalExecutionError> {
        match self.state {
            ComposeLeaseState::Started | ComposeLeaseState::MainStopped => Ok(()),
            _ => Err(EvalExecutionError::InvalidRecipe("Compose lease state")),
        }
    }

    fn force_recorded_resources(&self) -> Result<(), EvalExecutionError> {
        let mut first_error = None;
        for container in self.recorded.containers() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["rm", "--force", "--volumes", container])
                    .with_deadline(std::time::Duration::from_secs(10)),
            );
            first_error = first_error.or(removal.err());
        }
        for network in self.recorded.networks() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["network", "rm", network])
                    .with_deadline(std::time::Duration::from_secs(10)),
            );
            first_error = first_error.or(removal.err());
        }
        for volume in self.recorded.volumes() {
            let removal = self.runtime.remove(
                &DockerRemoveRequest::new(["volume", "rm", "--force", volume])
                    .with_deadline(std::time::Duration::from_secs(10)),
            );
            first_error = first_error.or(removal.err());
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl TaskEnvironmentLease for ComposeProjectLease {
    fn main_service(&self) -> &ComposeServiceName {
        &self.main
    }
    fn service(&self, name: &ComposeServiceName) -> Result<ServiceHandle, EvalExecutionError> {
        self.ensure_started()?;
        if self.services.contains(name) {
            Ok(ServiceHandle::new(name.clone()))
        } else {
            Err(EvalExecutionError::InvalidRecipe("Compose service"))
        }
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
            request.deadline,
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
        if self.state == ComposeLeaseState::Down {
            return Ok(());
        }
        let down = self.runtime.compose_down(
            &DockerComposeDownRequest::new(self.project.clone(), &self.project_directory)
                .with_deadline(std::time::Duration::from_secs(60)),
        );
        let remaining_after_down = self.runtime.compose_owned_resources(&self.project);
        if let Ok(resources) = &remaining_after_down {
            self.recorded = resources.clone();
        }
        let needs_force = matches!(&remaining_after_down, Ok(resources) if resources != &OwnedComposeResources::default());
        let forced = if needs_force {
            self.force_recorded_resources()
        } else {
            Ok(())
        };
        let remaining = if needs_force {
            self.runtime.compose_owned_resources(&self.project)
        } else {
            remaining_after_down
        };
        let result = match (down, forced, remaining) {
            (Err(error), _, _) => Err(error),
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

impl Drop for ComposeProjectLease {
    fn drop(&mut self) {
        if self.state != ComposeLeaseState::Down {
            let _ = self.teardown();
        }
    }
}
