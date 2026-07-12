// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen application composition for one runner process.
//!
//! This is the executable's only implementation-universe boundary. A stock or
//! custom statically linked distribution supplies explicit factories once;
//! capabilities, protocol-v2 validation, and protocol-v2 execution then borrow
//! the exact same registries.

use std::sync::Arc;

use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory};
use anyhow::Result;

use crate::coordinator::{RunnerProcessResultV2, RunnerV2Coordinator};
use crate::dataset_input::{
    BuiltinRunnerDatasetInputAdapterResolver, RunnerDatasetInputAdapterResolver,
};
use crate::execution_factories::{RunnerExecutionFactories, native_execution_factories};
use crate::graph_input::{BuiltinRunnerGraphInputAdapterResolver, RunnerGraphInputAdapterResolver};
use crate::protocol::RunnerCapabilities;
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
}

impl RunnerApplication {
    /// Compose an explicitly linked runner distribution exactly once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        distribution_id: impl Into<String>,
        runner_registry_factory: &dyn RunnerRegistryFactory,
        product_registry_factory: &dyn AiperfRegistryFactory,
        execution_factories: RunnerExecutionFactories,
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
        })
    }

    /// Compose the stock in-tree distribution around an already computed image ID.
    pub fn stock(distribution_id: impl Into<String>) -> Result<Self> {
        Self::new(
            distribution_id,
            &BuiltinRunnerRegistryFactory,
            &BuiltinAiperfRegistryFactory,
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

    /// Advertise capabilities from the exact frozen linked registries.
    pub fn capabilities(&self) -> RunnerCapabilities {
        self.coordinator.capabilities()
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
