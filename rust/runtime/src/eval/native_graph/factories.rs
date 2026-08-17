// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, independently selectable factories for NativeGraph-only behavior.
//!
//! Endpoint, transport, tokenizer, graph placement, clock, segment, observer,
//! and exporter selection remain in their existing AIPerf registries. This
//! module contains only NativeGraph behavior that needs an explicit extension
//! seam of its own.

use std::{
    fmt::{self, Display, Formatter},
    rc::Rc,
    sync::Arc,
};

use crate::eval::semantic::GraphLowererFactory;
use crate::eval::{AdapterSpec, ProviderRecovery};

use super::{
    AdapterProtocolConfig, AdapterProtocolFactory, AdapterRuntimeFactory, AdapterSpawner,
    NativeGraphLowererFactory, NativeGraphLoweringReport, NativeGraphPackagePlan,
    ProtocolAdapterRuntimeFactory,
};

/// Failure while selecting or binding one NativeGraph-only extension seam.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphFactoryError(String);

impl NativeGraphFactoryError {
    /// Creates a stable, redacted factory-selection failure.
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl Display for NativeGraphFactoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeGraphFactoryError {}

/// Stateless provider that binds a lowerer only after package import freezes.
pub trait NativeGraphLowererProvider: Send + Sync {
    /// Binds one lowerer to the imported package snapshot and declared authority.
    fn bind(
        &self,
        package: &NativeGraphPackagePlan,
    ) -> Result<Arc<dyn GraphLowererFactory>, NativeGraphFactoryError>;
}

/// Built-in package-bound lowerer provider for schema-1.1 NativeGraph sources.
#[derive(Clone, Copy, Debug, Default)]
pub struct PackageNativeGraphLowererProvider;

impl NativeGraphLowererProvider for PackageNativeGraphLowererProvider {
    fn bind(
        &self,
        package: &NativeGraphPackagePlan,
    ) -> Result<Arc<dyn GraphLowererFactory>, NativeGraphFactoryError> {
        Ok(Arc::new(NativeGraphLowererFactory::new(package)))
    }
}

/// Factory that creates one package-scoped, protocol-validated adapter runtime.
///
/// The selected protocol and process spawner are explicit constructor inputs;
/// the provider does not retain an ambient adapter or Docker capability.
pub trait NativeGraphAdapterRuntimeProvider {
    /// Binds one validated protocol configuration and task-owned spawner.
    fn bind(
        &self,
        config: AdapterProtocolConfig,
        protocol_factory: Rc<dyn AdapterProtocolFactory>,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError>;
}

/// Built-in constructor for the existing strict supervised adapter runtime.
#[derive(Clone, Copy, Debug, Default)]
pub struct StrictAdapterRuntimeProvider;

impl NativeGraphAdapterRuntimeProvider for StrictAdapterRuntimeProvider {
    fn bind(
        &self,
        config: AdapterProtocolConfig,
        protocol_factory: Rc<dyn AdapterProtocolFactory>,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(ProtocolAdapterRuntimeFactory::new(
            config,
            protocol_factory,
            spawner,
        )))
    }
}

/// Host-owned stepper for a declared environment adapter.
pub trait NativeGraphEnvironmentStepper {
    /// Advances the environment by one already-authorized operation.
    fn step(&mut self) -> Result<(), NativeGraphFactoryError>;
}

/// Factory for package-bound environment stepping implementations.
pub trait NativeGraphEnvironmentStepperFactory: Send + Sync {
    /// Binds an implementation to one declared environment adapter.
    fn bind(
        &self,
        adapter: &AdapterSpec,
    ) -> Result<Box<dyn NativeGraphEnvironmentStepper>, NativeGraphFactoryError>;
}

/// Explicit refusal until a later slice supplies an environment-step protocol.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefusingEnvironmentStepperFactory;

impl NativeGraphEnvironmentStepperFactory for RefusingEnvironmentStepperFactory {
    fn bind(
        &self,
        _: &AdapterSpec,
    ) -> Result<Box<dyn NativeGraphEnvironmentStepper>, NativeGraphFactoryError> {
        Err(NativeGraphFactoryError::new(
            "NativeGraph environment stepping is not available in the acyclic model slice",
        ))
    }
}

/// Compatibility driver selected only for explicitly externally driven packages.
pub trait NativeGraphExternalDriver {
    /// Runs one bounded externally driven episode.
    fn run(&mut self) -> Result<(), NativeGraphFactoryError>;
}

/// Factory for compatibility-only externally driven package support.
pub trait NativeGraphExternalDriverFactory: Send + Sync {
    /// Binds an external driver to the imported package authority.
    fn bind(
        &self,
        package: &NativeGraphPackagePlan,
    ) -> Result<Box<dyn NativeGraphExternalDriver>, NativeGraphFactoryError>;
}

/// Explicit refusal until the externally driven compatibility slice is enabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefusingExternalDriverFactory;

impl NativeGraphExternalDriverFactory for RefusingExternalDriverFactory {
    fn bind(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<Box<dyn NativeGraphExternalDriver>, NativeGraphFactoryError> {
        Err(NativeGraphFactoryError::new(
            "externally driven NativeGraph packages are not enabled by the acyclic model slice",
        ))
    }
}

/// Observer that validates the fidelity account emitted by source lowering.
pub trait NativeGraphFidelityObserver {
    /// Records or refuses one immutable lowering report before dispatch.
    fn observe(&self, report: &NativeGraphLoweringReport) -> Result<(), NativeGraphFactoryError>;
}

/// Factory for a fidelity observer selected through application composition.
pub trait NativeGraphFidelityObserverFactory: Send + Sync {
    /// Creates one observer for an immutable episode preparation.
    fn create(&self) -> Rc<dyn NativeGraphFidelityObserver>;
}

/// Built-in observer requiring exact lowering for every source node.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactNativeGraphFidelityObserverFactory;

impl NativeGraphFidelityObserverFactory for ExactNativeGraphFidelityObserverFactory {
    fn create(&self) -> Rc<dyn NativeGraphFidelityObserver> {
        Rc::new(ExactNativeGraphFidelityObserver)
    }
}

struct ExactNativeGraphFidelityObserver;

impl NativeGraphFidelityObserver for ExactNativeGraphFidelityObserver {
    fn observe(&self, report: &NativeGraphLoweringReport) -> Result<(), NativeGraphFactoryError> {
        if report.nodes().all(super::NativeGraphNodeLowering::is_exact) {
            Ok(())
        } else {
            Err(NativeGraphFactoryError::new(
                "NativeGraph lowering contains a non-exact source node",
            ))
        }
    }
}

/// Resolves a provider's terminal recovery fact without inventing cleanup state.
pub trait NativeGraphProviderRecoveryFactory: Send + Sync {
    /// Validates a provider-owned recovery disposition at an episode boundary.
    fn resolve(&self, recovery: ProviderRecovery) -> Result<(), NativeGraphFactoryError>;
}

/// Built-in recovery policy preserving only provider-confirmed terminal facts.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConfirmedNativeGraphProviderRecoveryFactory;

impl NativeGraphProviderRecoveryFactory for ConfirmedNativeGraphProviderRecoveryFactory {
    fn resolve(&self, recovery: ProviderRecovery) -> Result<(), NativeGraphFactoryError> {
        match recovery {
            ProviderRecovery::Recovered | ProviderRecovery::RequiresFreshAdapter => Ok(()),
            ProviderRecovery::Failed { reason } => Err(NativeGraphFactoryError::new(format!(
                "NativeGraph provider did not establish terminal recovery: {reason}"
            ))),
        }
    }
}
