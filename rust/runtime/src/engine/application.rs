// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen application composition for one runner process.
//!
//! This is the executable's only implementation-universe boundary. A stock or
//! custom statically linked distribution supplies explicit factories once;
//! capabilities, protocol-v2 validation, and protocol-v2 execution then borrow
//! the exact same registries.

use std::sync::Arc;

use crate::extensions::{AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory};
use anyhow::Result;

use crate::engine::coordinator::{Coordinator, ProcessResultV2};
use crate::engine::dataset_input::{
    BuiltinRunnerDatasetInputAdapterResolver, DatasetInputAdapterResolver,
};
use crate::engine::execution_factories::{ExecutionFactories, native_execution_factories};
use crate::engine::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, GraphInputAdapterResolver,
};
use crate::engine::protocol::Catalog;
use crate::engine::protocol_v2::EnvelopeV2;
use crate::engine::registry::{RunContext, ValidatedEndpointProfileV2};
use crate::engine::sidecar_input::{
    BuiltinRunnerSidecarInputAdapterResolver, SidecarInputAdapterResolver,
};
use crate::graph::driver::TraceProgramDriverFactory;
use crate::graph::tools::{
    ContainerRuntime, ResolvedTraceEnvironment, ToolExecutionBackend, preflight_docker_sandbox,
};

/// Preflight every Docker-backed recorded-agent recipe before trace provisioning.
pub async fn preflight_recorded_agent_docker_environments<'a>(
    runtime: &dyn ContainerRuntime,
    environments: impl IntoIterator<Item = &'a ResolvedTraceEnvironment>,
) -> Result<()> {
    for environment in environments {
        if environment.backend != ToolExecutionBackend::Docker {
            continue;
        }
        preflight_docker_sandbox(runtime, environment)
            .await
            .map_err(anyhow::Error::from)?;
    }
    Ok(())
}

/// One frozen implementation universe for a fresh runner child.
///
/// Rust has no ambient implementation discovery. Custom distributions link
/// extension crates and pass their explicit factories here; no core dispatcher,
/// scheduler, validator, or reporter changes are required.
pub struct Application {
    distribution_id: String,
    coordinator: Coordinator,
}

impl Application {
    /// Compose an explicitly linked runner distribution exactly once.
    pub fn new(
        distribution_id: impl Into<String>,
        product_registry_factory: &dyn AIPerfRegistryFactory,
        execution_factories: ExecutionFactories,
        graph_inputs: Arc<dyn GraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn DatasetInputAdapterResolver>,
        sidecar_inputs: Arc<dyn SidecarInputAdapterResolver>,
    ) -> Result<Self> {
        crate::endpoints::implementation::capture_endpoint_policy();
        let distribution_id = distribution_id.into();
        let coordinator = Coordinator::new(
            distribution_id.clone(),
            product_registry_factory,
            execution_factories,
            graph_inputs,
            dataset_inputs,
            sidecar_inputs,
        )?;
        Ok(Self {
            distribution_id,
            coordinator,
        })
    }

    /// Compose the stock in-tree distribution around an already computed image ID.
    pub fn stock(distribution_id: impl Into<String>) -> Result<Self> {
        Self::new(
            distribution_id,
            &BuiltinAIPerfRegistryFactory,
            native_execution_factories(),
            Arc::new(BuiltinRunnerGraphInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerSidecarInputAdapterResolver::new()),
        )
    }

    /// Return the identity of the complete executable image serving this process.
    pub fn distribution_id(&self) -> &str {
        &self.distribution_id
    }

    /// Return the plugins.yaml-shaped catalog for this exact linked runner.
    pub fn catalog(&self) -> Catalog {
        self.coordinator.catalog()
    }

    /// Validate or execute one protocol-v2 envelope through the frozen coordinator.
    pub fn handle_v2(&self, envelope: EnvelopeV2) -> ProcessResultV2 {
        self.coordinator.handle(envelope)
    }

    /// Borrow the exact product registry used by capabilities and execution.
    pub fn product_registry(&self) -> &AIPerfRegistry {
        self.coordinator.product_registry()
    }

    /// Borrow the trace-driver registry frozen with this application image.
    pub fn trace_driver_factory(&self) -> &dyn TraceProgramDriverFactory {
        self.coordinator.execution_factories().trace_driver()
    }

    /// Builds a model-stage context from this application's frozen composition.
    pub fn native_graph_context(
        &self,
        endpoint_profiles: Vec<ValidatedEndpointProfileV2>,
    ) -> Result<RunContext> {
        self.coordinator.native_graph_context(endpoint_profiles)
    }
}
