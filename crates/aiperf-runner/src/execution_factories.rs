// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen execution-placement composition for one runner process.
//!
//! The aggregate is intentionally a value, not a mode enum: every replaceable
//! member is an object-safe trait. A future RPC or ZMQ distribution can supply
//! remote HTTP-turn and whole-graph placement, plus its own readiness control
//! transport, without changing pair selection, scheduling, metrics, or report
//! code.

use std::fmt;
use std::sync::Arc;

use crate::graph_execution::{NativeRunnerGraphPlacementFactory, RunnerGraphPlacementFactory};
use crate::grpc_turn_execution::NativeGrpcExecutionBackendFactory;
use crate::readiness::{
    NativeHttpReadinessPlanFactory, NativeHttpReadinessTransportFactory,
    OnlineReadinessPlanFactory, ReadinessTransportFactory,
};
use crate::turn_execution::{HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory};

/// Exact execution-factory universe retained from coordinator construction.
#[derive(Clone)]
pub struct RunnerExecutionFactories {
    http: Arc<dyn HttpExecutionBackendFactory>,
    grpc: Arc<dyn HttpExecutionBackendFactory>,
    graph: Arc<dyn RunnerGraphPlacementFactory>,
    readiness_plans: Arc<dyn OnlineReadinessPlanFactory>,
    readiness_transport: Arc<dyn ReadinessTransportFactory>,
}

impl RunnerExecutionFactories {
    /// Compose one frozen set of independently replaceable execution seams.
    pub fn new(
        http: Arc<dyn HttpExecutionBackendFactory>,
        graph: Arc<dyn RunnerGraphPlacementFactory>,
        readiness_plans: Arc<dyn OnlineReadinessPlanFactory>,
        readiness_transport: Arc<dyn ReadinessTransportFactory>,
    ) -> Self {
        let grpc = http.clone();
        Self {
            http,
            grpc,
            graph,
            readiness_plans,
            readiness_transport,
        }
    }

    /// Override gRPC turn placement independently from HTTP placement.
    ///
    /// The base constructor aliases both to the same generic turn-placement
    /// factory, which keeps remote implementations transport-neutral. Native
    /// composition installs the Tonic-specific factory explicitly.
    pub fn with_grpc(mut self, grpc: Arc<dyn HttpExecutionBackendFactory>) -> Self {
        self.grpc = grpc;
        self
    }

    /// Borrow the HTTP turn-placement factory used below the one dispatcher.
    pub fn http(&self) -> &dyn HttpExecutionBackendFactory {
        self.http.as_ref()
    }

    /// Retain the HTTP turn-placement factory in a prepared operation.
    pub fn http_handle(&self) -> Arc<dyn HttpExecutionBackendFactory> {
        self.http.clone()
    }

    /// Borrow the gRPC turn-placement factory used below the same dispatcher.
    pub fn grpc(&self) -> &dyn HttpExecutionBackendFactory {
        self.grpc.as_ref()
    }

    /// Retain the gRPC turn-placement factory in a prepared operation.
    pub fn grpc_handle(&self) -> Arc<dyn HttpExecutionBackendFactory> {
        self.grpc.clone()
    }

    /// Borrow the whole-trace graph-placement factory.
    pub fn graph(&self) -> &dyn RunnerGraphPlacementFactory {
        self.graph.as_ref()
    }

    /// Retain the whole-trace graph-placement factory in a prepared operation.
    pub fn graph_handle(&self) -> Arc<dyn RunnerGraphPlacementFactory> {
        self.graph.clone()
    }

    /// Borrow the side-effect-free readiness-plan factory.
    pub fn readiness_plans(&self) -> &dyn OnlineReadinessPlanFactory {
        self.readiness_plans.as_ref()
    }

    /// Retain the readiness-plan factory in a prepared operation.
    pub fn readiness_plans_handle(&self) -> Arc<dyn OnlineReadinessPlanFactory> {
        self.readiness_plans.clone()
    }

    /// Borrow the readiness control-transport factory.
    pub fn readiness_transport(&self) -> &dyn ReadinessTransportFactory {
        self.readiness_transport.as_ref()
    }

    /// Retain the readiness transport factory in a prepared operation.
    pub fn readiness_transport_handle(&self) -> Arc<dyn ReadinessTransportFactory> {
        self.readiness_transport.clone()
    }
}

impl fmt::Debug for RunnerExecutionFactories {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunnerExecutionFactories")
            .finish_non_exhaustive()
    }
}

/// Compose the stock in-process HTTP, graph, and readiness implementations.
pub fn native_execution_factories() -> RunnerExecutionFactories {
    RunnerExecutionFactories::new(
        Arc::new(NativeHttpExecutionBackendFactory),
        Arc::new(NativeRunnerGraphPlacementFactory),
        Arc::new(NativeHttpReadinessPlanFactory),
        Arc::new(NativeHttpReadinessTransportFactory),
    )
    .with_grpc(Arc::new(NativeGrpcExecutionBackendFactory::default()))
}
