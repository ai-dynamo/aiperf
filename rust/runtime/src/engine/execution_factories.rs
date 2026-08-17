// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen execution-placement composition for one runner process.
//!
//! The aggregate is intentionally a value, not a mode enum: every replaceable
//! member is an object-safe trait, keeping placement and readiness independent
//! from pair selection, scheduling, metrics, and reporting.

use std::fmt;
use std::sync::Arc;

use crate::engine::control_plane_http::{
    ControlPlaneHttpProviderFactory, NativeControlPlaneHttpProviderFactory,
};
use crate::engine::graph_execution::{GraphPlacementFactory, NativeRunnerGraphPlacementFactory};
use crate::engine::readiness::{
    NativeHttpReadinessPlanFactory, NativeHttpReadinessTransportFactory,
    OnlineReadinessPlanFactory, ReadinessTransportFactory,
};
use crate::engine::turn_execution::{HttpExecutionFactory, RequestExecutorFactory};
use crate::eval::NativeGraphLiveTraceProgramDriverFactory;
use crate::extensions::TransactionalRegistry;
use crate::graph::driver::{
    RecordedReplayTraceProgramDriverFactory, StaticGraphTraceProgramDriverFactory,
    TraceDriverCapabilities, TraceDriverError, TraceDriverSpec, TraceIdentity, TraceProgramDriver,
    TraceProgramDriverFactory, WorkerIdentity,
};

/// Frozen name-keyed selection over trace-program driver families.
///
/// Registration failures remain explicit even though stock names are compiled
/// into this binary. This keeps application construction on the same
/// transactional selection boundary used by injected distributions.
#[derive(Clone)]
struct RegisteredTraceProgramDrivers {
    factories: TransactionalRegistry<Arc<dyn TraceProgramDriverFactory>>,
    registration_error: Option<TraceDriverError>,
}

impl RegisteredTraceProgramDrivers {
    fn stock() -> Self {
        let mut factories = TransactionalRegistry::new();
        let registration_error = factories
            .commit(|staged| {
                staged.insert(
                    "static_graph",
                    Arc::new(StaticGraphTraceProgramDriverFactory)
                        as Arc<dyn TraceProgramDriverFactory>,
                )?;
                staged.insert(
                    "recorded_replay",
                    Arc::new(RecordedReplayTraceProgramDriverFactory::default())
                        as Arc<dyn TraceProgramDriverFactory>,
                )?;
                staged.insert(
                    "native_graph_live",
                    Arc::new(NativeGraphLiveTraceProgramDriverFactory)
                        as Arc<dyn TraceProgramDriverFactory>,
                )?;
                Ok::<(), crate::extensions::DuplicateName>(())
            })
            .err()
            .map(|error| TraceDriverError::new(error.to_string()));
        Self {
            factories,
            registration_error,
        }
    }

    fn resolve(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<&Arc<dyn TraceProgramDriverFactory>, TraceDriverError> {
        if let Some(error) = &self.registration_error {
            return Err(error.clone());
        }
        self.factories.get(&spec.kind).ok_or_else(|| {
            TraceDriverError::new(format!(
                "no linked trace program driver for {:?}",
                spec.kind
            ))
        })
    }
}

impl TraceProgramDriverFactory for RegisteredTraceProgramDrivers {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError> {
        self.resolve(spec)?.capabilities(spec)
    }

    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
        self.resolve(spec)?.create(worker, trace, spec)
    }
}

/// Exact execution-factory universe retained from coordinator construction.
///
/// This holds only the seams shared across native transports: the base HTTP
/// turn-placement factory (the primary injectable placement seam), graph
/// placement, readiness, and the isolated control-plane HTTP provider. A
/// transport's own turn-placement — the gRPC Tonic backend, the `dry_run` fake
/// leaf — is owned by its [`NativeTransportExecution`] binding, not named here,
/// so a transport is added by registering it, never by growing this struct.
///
/// [`NativeTransportExecution`]: crate::engine::registry::NativeTransportExecution
#[derive(Clone)]
pub struct ExecutionFactories {
    http: Arc<dyn RequestExecutorFactory>,
    graph: Arc<dyn GraphPlacementFactory>,
    trace_driver: Arc<dyn TraceProgramDriverFactory>,
    readiness_plans: Arc<dyn OnlineReadinessPlanFactory>,
    readiness_transport: Arc<dyn ReadinessTransportFactory>,
    control_plane_http: Arc<dyn ControlPlaneHttpProviderFactory>,
}

impl ExecutionFactories {
    /// Compose one frozen set of independently replaceable execution seams.
    pub fn new(
        http: Arc<dyn RequestExecutorFactory>,
        graph: Arc<dyn GraphPlacementFactory>,
        readiness_plans: Arc<dyn OnlineReadinessPlanFactory>,
        readiness_transport: Arc<dyn ReadinessTransportFactory>,
    ) -> Self {
        Self {
            http,
            graph,
            trace_driver: Arc::new(RegisteredTraceProgramDrivers::stock()),
            readiness_plans,
            readiness_transport,
            control_plane_http: Arc::new(NativeControlPlaneHttpProviderFactory::default()),
        }
    }

    /// Borrow the HTTP turn-placement factory used below the one dispatcher.
    pub fn http(&self) -> &dyn RequestExecutorFactory {
        self.http.as_ref()
    }

    /// Retain the HTTP turn-placement factory in a prepared operation.
    pub fn http_handle(&self) -> Arc<dyn RequestExecutorFactory> {
        self.http.clone()
    }

    /// Borrow the whole-trace graph-placement factory.
    pub fn graph(&self) -> &dyn GraphPlacementFactory {
        self.graph.as_ref()
    }

    /// Retain the whole-trace graph-placement factory in a prepared operation.
    pub fn graph_handle(&self) -> Arc<dyn GraphPlacementFactory> {
        self.graph.clone()
    }

    /// Borrow the frozen trace-driver registry used for capability preflight.
    pub fn trace_driver(&self) -> &dyn TraceProgramDriverFactory {
        self.trace_driver.as_ref()
    }

    /// Retain the frozen trace-driver registry in a prepared graph operation.
    pub fn trace_driver_handle(&self) -> Arc<dyn TraceProgramDriverFactory> {
        self.trace_driver.clone()
    }

    /// Replace the linked trace-driver registry before the application freezes composition.
    pub fn with_trace_driver_factory(
        mut self,
        trace_driver: Arc<dyn TraceProgramDriverFactory>,
    ) -> Self {
        self.trace_driver = trace_driver;
        self
    }

    /// Borrow the side-effect-free readiness-plan factory.
    pub fn readiness_plans(&self) -> &dyn OnlineReadinessPlanFactory {
        self.readiness_plans.as_ref()
    }

    /// Borrow the readiness control-transport factory.
    pub fn readiness_transport(&self) -> &dyn ReadinessTransportFactory {
        self.readiness_transport.as_ref()
    }

    /// Borrow the factory for run-local, inference-isolated control handles.
    pub fn control_plane_http(&self) -> &dyn ControlPlaneHttpProviderFactory {
        self.control_plane_http.as_ref()
    }

    /// Retain the factory for run-local, inference-isolated control handles.
    pub fn control_plane_http_handle(&self) -> Arc<dyn ControlPlaneHttpProviderFactory> {
        self.control_plane_http.clone()
    }
}

impl fmt::Debug for ExecutionFactories {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionFactories")
            .finish_non_exhaustive()
    }
}

/// Compose the stock in-process HTTP, graph, and readiness implementations.
///
/// gRPC and `dry_run` turn placement are not composed here: each is owned by its
/// transport's [`NativeTransportExecution`] binding, resolved from the registry
/// at prepare time.
///
/// [`NativeTransportExecution`]: crate::engine::registry::NativeTransportExecution
pub fn native_execution_factories() -> ExecutionFactories {
    ExecutionFactories::new(
        Arc::new(HttpExecutionFactory),
        Arc::new(NativeRunnerGraphPlacementFactory),
        Arc::new(NativeHttpReadinessPlanFactory),
        Arc::new(NativeHttpReadinessTransportFactory),
    )
}
