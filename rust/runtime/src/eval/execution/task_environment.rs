// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral leases for task-owned benchmark environments.

use std::{collections::BTreeMap, io::Read, time::Duration};

use super::{
    ComposeServiceName, DockerCopyRequest, DockerExecRequest, DockerRuntime, EnvName,
    EvalExecutionError, EvalExecutionPhase, SecretValue,
};

/// An opaque reference to one service owned by a task environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ServiceHandle {
    service: ComposeServiceName,
}

impl ServiceHandle {
    pub(crate) fn new(service: ComposeServiceName) -> Self {
        Self { service }
    }
    pub(crate) fn service(&self) -> &ComposeServiceName {
        &self.service
    }
}

/// A command directed to one service without a shell boundary.
pub(crate) struct ServiceExecRequest<'a> {
    pub(crate) service: &'a ComposeServiceName,
    pub(crate) arguments: &'a [String],
    pub(crate) public_environment: BTreeMap<EnvName, String>,
    pub(crate) secret_environment: BTreeMap<EnvName, SecretValue>,
    pub(crate) phase: EvalExecutionPhase,
    pub(crate) user: Option<&'a str>,
    pub(crate) workdir: Option<&'a str>,
    pub(crate) network_lease: &'a str,
    pub(crate) deadline: Option<std::time::Duration>,
}

/// A bounded archive request directed to one task service.
pub(crate) struct ServiceArchiveRequest<'a> {
    pub(crate) service: &'a ComposeServiceName,
    pub(crate) source: &'a str,
    pub(crate) deadline: Duration,
}

/// A live, task-owned environment whose services can be used by benchmark phases.
pub(crate) trait TaskEnvironmentLease {
    fn main_service(&self) -> &ComposeServiceName;
    fn service(&self, name: &ComposeServiceName) -> Result<ServiceHandle, EvalExecutionError>;
    fn exec(&mut self, request: ServiceExecRequest<'_>) -> Result<(), EvalExecutionError>;
    fn archive(
        &mut self,
        request: ServiceArchiveRequest<'_>,
    ) -> Result<Box<dyn Read>, EvalExecutionError>;
    fn copy_into(
        &mut self,
        service: &ComposeServiceName,
        source: &str,
        destination: &str,
    ) -> Result<(), EvalExecutionError>;
    fn stop_main(&mut self, deadline: Duration) -> Result<(), EvalExecutionError>;
    fn main_image_id(&self) -> Result<&str, EvalExecutionError>;
    fn teardown(&mut self) -> Result<(), EvalExecutionError>;
    /// Tears down after a terminal benchmark failure within the provider deadline.
    fn teardown_after_terminal_failure(
        &mut self,
        deadline: Duration,
    ) -> Result<(), EvalExecutionError>;
}

/// Adapts the established one-container Dockerfile environment to the lease seam.
pub(crate) struct DockerfileEnvironmentLease<'a> {
    runtime: &'a dyn DockerRuntime,
    container: String,
    image: String,
    main: ComposeServiceName,
}

impl<'a> DockerfileEnvironmentLease<'a> {
    pub(crate) fn new(runtime: &'a dyn DockerRuntime, container: String, image: String) -> Self {
        Self {
            runtime,
            container,
            image,
            main: ComposeServiceName::main(),
        }
    }
}

impl TaskEnvironmentLease for DockerfileEnvironmentLease<'_> {
    fn main_service(&self) -> &ComposeServiceName {
        &self.main
    }
    fn service(&self, name: &ComposeServiceName) -> Result<ServiceHandle, EvalExecutionError> {
        if name == &self.main {
            Ok(ServiceHandle::new(name.clone()))
        } else {
            Err(EvalExecutionError::InvalidRecipe("Dockerfile service"))
        }
    }
    fn exec(&mut self, request: ServiceExecRequest<'_>) -> Result<(), EvalExecutionError> {
        if request.service != &self.main {
            return Err(EvalExecutionError::InvalidRecipe("Dockerfile service"));
        }
        self.runtime.exec(
            &DockerExecRequest::new(
                &self.container,
                request.arguments.iter().cloned(),
                request.public_environment,
                request.secret_environment,
            )
            .with_phase(
                request.phase,
                request.user,
                request.workdir,
                request.network_lease,
                request.deadline,
            ),
        )
    }
    fn archive(
        &mut self,
        request: ServiceArchiveRequest<'_>,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        if request.service != &self.main {
            return Err(EvalExecutionError::InvalidRecipe("Dockerfile service"));
        }
        self.runtime
            .copy_archive_bounded(&self.container, request.source, request.deadline)
    }
    fn copy_into(
        &mut self,
        service: &ComposeServiceName,
        source: &str,
        destination: &str,
    ) -> Result<(), EvalExecutionError> {
        if service != &self.main {
            return Err(EvalExecutionError::InvalidRecipe("Dockerfile service"));
        }
        self.runtime.copy(&DockerCopyRequest::new([
            "cp".to_owned(),
            source.to_owned(),
            format!("{}:{destination}", self.container),
        ]))
    }
    fn stop_main(&mut self, _: Duration) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn main_image_id(&self) -> Result<&str, EvalExecutionError> {
        Ok(&self.image)
    }
    fn teardown(&mut self) -> Result<(), EvalExecutionError> {
        Ok(())
    }
    fn teardown_after_terminal_failure(
        &mut self,
        _: Duration,
    ) -> Result<(), EvalExecutionError> {
        self.teardown()
    }
}
