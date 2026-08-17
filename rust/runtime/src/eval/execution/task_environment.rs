// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral leases for task-owned benchmark environments.

use std::{collections::BTreeMap, io::Read, time::Duration};

use super::{ComposeServiceName, EnvName, EvalExecutionError, EvalExecutionPhase, SecretValue};

/// A command directed to one service without a shell boundary.
pub(crate) struct ServiceExecRequest<'a> {
    pub(crate) service: &'a ComposeServiceName,
    pub(crate) arguments: &'a [String],
    pub(crate) public_environment: BTreeMap<EnvName, String>,
    pub(crate) secret_environment: BTreeMap<EnvName, SecretValue>,
    pub(crate) phase: EvalExecutionPhase,
    pub(crate) user: Option<&'a str>,
    pub(crate) workdir: Option<&'a str>,
    pub(crate) deadline: Option<std::time::Duration>,
}

/// A bounded archive request directed to one task service.
pub(crate) struct ServiceArchiveRequest<'a> {
    pub(crate) service: &'a ComposeServiceName,
    pub(crate) source: &'a str,
    pub(crate) deadline: Duration,
    pub(crate) phase: EvalExecutionPhase,
}

/// A live, task-owned environment whose services can be used by benchmark phases.
pub(crate) trait TaskEnvironmentLease {
    fn main_service(&self) -> &ComposeServiceName;
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
    /// Copies snapshot data while consuming the caller's phase deadline.
    fn copy_into_bounded(
        &mut self,
        service: &ComposeServiceName,
        source: &str,
        destination: &str,
        _: Duration,
    ) -> Result<(), EvalExecutionError> {
        self.copy_into(service, source, destination)
    }
    fn stop_main(&mut self, deadline: Duration) -> Result<(), EvalExecutionError>;
    fn main_image_id(&self) -> Result<&str, EvalExecutionError>;
    fn teardown(&mut self) -> Result<(), EvalExecutionError>;
    /// Tears down after a terminal benchmark failure within the provider deadline.
    fn teardown_after_terminal_failure(
        &mut self,
        deadline: Duration,
    ) -> Result<(), EvalExecutionError>;
}
