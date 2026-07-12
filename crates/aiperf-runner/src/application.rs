// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen application composition for one runner process.
//!
//! This is the executable's only implementation-universe boundary. A stock or
//! custom statically linked distribution supplies explicit factories once;
//! capabilities, protocol-v2 validation, and protocol-v2 execution then borrow
//! the exact same registries. Protocol v1 remains a compatibility adapter over
//! the same product registry and native dispatcher.

use std::sync::Arc;

use aiperf_extensions::{
    AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory, ExtensionError,
};
use aiperf_graph::input::{GraphInputAdapterRegistry, GraphInputAdapterResolver};
use anyhow::Result;

use crate::coordinator::{RunnerProcessResultV2, RunnerV2Coordinator};
use crate::dataset_input::{
    BuiltinRunnerDatasetInputAdapterResolver, RunnerDatasetInputAdapterResolver,
};
use crate::execute::execute_run_with_all_factories;
use crate::execution_factories::{RunnerExecutionFactories, native_execution_factories};
use crate::graph_input::{BuiltinRunnerGraphInputAdapterResolver, RunnerGraphInputAdapterResolver};
use crate::protocol::{RunRequest, RunTerminal, RunnerCapabilities};
use crate::protocol_v2::RunnerEnvelopeV2;
use crate::registry::{BuiltinRunnerRegistryFactory, RunnerRegistryFactory};
use crate::sidecar_input::{
    BuiltinRunnerSidecarInputAdapterResolver, RunnerSidecarInputAdapterResolver,
};

/// One frozen implementation universe for a fresh runner child.
///
/// Rust has no ambient implementation discovery. Custom distributions link
/// extension crates and pass their explicit factories here; no core dispatcher,
/// scheduler, validator, or reporter changes are required.
pub struct RunnerApplication {
    distribution_id: String,
    coordinator: RunnerV2Coordinator,
    execution_factories: RunnerExecutionFactories,
    legacy_graph_inputs: Arc<dyn GraphInputAdapterResolver>,
}

impl RunnerApplication {
    /// Compose an explicitly linked runner distribution exactly once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        distribution_id: impl Into<String>,
        runner_registry_factory: &dyn RunnerRegistryFactory,
        product_registry_factory: &dyn AiperfRegistryFactory,
        execution_factories: RunnerExecutionFactories,
        legacy_graph_inputs: Arc<dyn GraphInputAdapterResolver>,
        graph_inputs: Arc<dyn RunnerGraphInputAdapterResolver>,
        dataset_inputs: Arc<dyn RunnerDatasetInputAdapterResolver>,
        sidecar_inputs: Arc<dyn RunnerSidecarInputAdapterResolver>,
    ) -> Result<Self> {
        let distribution_id = distribution_id.into();
        let coordinator = RunnerV2Coordinator::new(
            distribution_id.clone(),
            runner_registry_factory,
            product_registry_factory,
            execution_factories.clone(),
            graph_inputs,
            dataset_inputs,
            sidecar_inputs,
        )?;
        Ok(Self {
            distribution_id,
            coordinator,
            execution_factories,
            legacy_graph_inputs,
        })
    }

    /// Compose the stock in-tree distribution around an already computed image ID.
    pub fn stock(distribution_id: impl Into<String>) -> Result<Self> {
        Self::new(
            distribution_id,
            &BuiltinRunnerRegistryFactory,
            &BuiltinAiperfRegistryFactory,
            native_execution_factories(),
            Arc::new(GraphInputAdapterRegistry::with_builtin_adapters()),
            Arc::new(BuiltinRunnerGraphInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerSidecarInputAdapterResolver::new()),
        )
    }

    /// Return the identity of the complete executable image serving this process.
    pub fn distribution_id(&self) -> &str {
        &self.distribution_id
    }

    /// Advertise capabilities from the exact frozen linked registries.
    pub fn capabilities(&self) -> RunnerCapabilities {
        self.coordinator.capabilities()
    }

    /// Execute one protocol-v1 compatibility request through the shared dispatcher.
    ///
    /// `AiperfRegistry` is an immutable aggregate of cloneable frozen registries;
    /// this adapter clones that value but never reruns linked extension
    /// registration. Protocol v2 borrows the original value directly.
    pub fn execute_v1(&self, request: RunRequest) -> Result<RunTerminal> {
        let frozen_registry = FrozenRegistryFactory(self.coordinator.product_registry());
        execute_run_with_all_factories(
            request,
            self.execution_factories.http(),
            self.legacy_graph_inputs.as_ref(),
            self.execution_factories.graph(),
            &frozen_registry,
        )
    }

    /// Validate or execute one protocol-v2 envelope through the frozen coordinator.
    pub fn handle_v2(&self, envelope: RunnerEnvelopeV2) -> RunnerProcessResultV2 {
        self.coordinator.handle(envelope)
    }

    /// Borrow the exact product registry used by capabilities and execution.
    pub fn product_registry(&self) -> &AiperfRegistry {
        self.coordinator.product_registry()
    }
}

struct FrozenRegistryFactory<'a>(&'a AiperfRegistry);

impl AiperfRegistryFactory for FrozenRegistryFactory<'_> {
    fn build(&self) -> std::result::Result<AiperfRegistry, ExtensionError> {
        Ok(self.0.clone())
    }
}
