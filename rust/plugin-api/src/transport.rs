// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport execution boundary.
//!
//! A transport occupies exactly one of two execution shapes.
//! [`TransportExecutionShapeV1::Request`] transports plug into the host's shared
//! worker kernel: they build one [`RequestExecutor`] per placement and the host
//! owns scheduling, admission, drain, and cancellation.
//! [`TransportExecutionShapeV1::Direct`] transports drive their own execution
//! and reach back through the narrow `aiperf_core::services` traits instead.
//! Declaring both is refused — the two shapes place the same transport at two
//! incompatible points in the run's control flow.
//!
//! What is deliberately **absent** here is the generic execution capsule.
//! `ExecutionSinkBuilder`, `WorkerSink`, `B::Sink`, and every sink-erasure trait
//! belong to `aiperf-transport-sdk`, are monomorphized into each transport
//! plugin, and never appear in a boundary signature. If a concrete sink type
//! crossed this boundary, the worker loop would dispatch through a vtable on the
//! per-request path; keeping the capsule plugin-private is what preserves the
//! static specialization the monolithic baseline has.
//!
//! [`RequestTransportExecution::build_executor`] is therefore called once per
//! placement construction, takes one owned context, and returns the same
//! `Rc<dyn RequestExecutor>` shape the baseline produces. The host wraps the
//! result in nothing.

use core::fmt::{self, Display, Formatter};
use core::future::Future;
use core::pin::Pin;
use std::rc::Rc;

use aiperf_core::clock::Clock;

use crate::id::RegistryId;
use crate::prepared::Endpoint;
use crate::validation::ValidationError;

/// Which execution position a transport occupies. Exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransportExecutionShapeV1 {
    /// Builds a [`RequestExecutor`] and runs inside the host's worker kernel.
    Request,
    /// Drives its own execution through the narrow host service traits.
    Direct,
}

impl TransportExecutionShapeV1 {
    /// The lowercase label used in receipts and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Direct => "direct",
        }
    }

    /// Select the single declared shape, refusing zero or two declarations.
    pub fn exactly_one(declared: &[TransportExecutionShapeV1]) -> Result<Self, ValidationError> {
        match declared {
            [shape] => Ok(*shape),
            _ => Err(ValidationError::AmbiguousExecutionShape),
        }
    }
}

impl Display for TransportExecutionShapeV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Whether a transport can answer an endpoint readiness probe before traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessCapabilityV1 {
    /// No probe; the host starts the phase without waiting.
    Unsupported,
    /// The transport can probe the endpoint and report reachability.
    Supported,
}

/// Whether a transport speaks WebSocket, and under what session constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketCapabilityV1 {
    /// The transport has no WebSocket path.
    Unsupported,
    /// The transport speaks WebSocket.
    Supported {
        /// A continuation turn must reuse the logical session's bound socket.
        ///
        /// When true the host may not freely rebalance a session across
        /// connections: server-side session state lives on one socket.
        needs_session_affinity: bool,
    },
}

/// Everything a request transport needs to build one placement's executor.
///
/// The context is owned and consumed: a placement is constructed once, and
/// holding a borrow of host state across the run would put a lifetime on the
/// executor for no benefit.
#[derive(Clone)]
pub struct RequestExecutionBuildContextV1 {
    clock: Rc<dyn Clock>,
    endpoint: Rc<dyn Endpoint>,
    worker_index: u32,
    worker_count: u32,
}

impl RequestExecutionBuildContextV1 {
    /// Bind one placement's clock, endpoint, and worker coordinates.
    pub fn new(
        clock: Rc<dyn Clock>,
        endpoint: Rc<dyn Endpoint>,
        worker_index: u32,
        worker_count: u32,
    ) -> Self {
        Self {
            clock,
            endpoint,
            worker_index,
            worker_count,
        }
    }

    /// The run's single time source.
    pub fn clock(&self) -> Rc<dyn Clock> {
        self.clock.clone()
    }

    /// The validated endpoint this placement composes requests for.
    pub fn endpoint(&self) -> Rc<dyn Endpoint> {
        self.endpoint.clone()
    }

    /// Zero-based index of this worker within the cell.
    pub const fn worker_index(&self) -> u32 {
        self.worker_index
    }

    /// Total workers in the cell.
    pub const fn worker_count(&self) -> u32 {
        self.worker_count
    }

    /// Consume the context into its parts, avoiding a refcount bump per field.
    pub fn into_parts(self) -> (Rc<dyn Clock>, Rc<dyn Endpoint>, u32, u32) {
        (
            self.clock,
            self.endpoint,
            self.worker_index,
            self.worker_count,
        )
    }
}

/// One request as the execution boundary sees it.
///
/// The body is borrowed pre-serialized bytes: composing it is the endpoint's
/// job, and copying it here would add an allocation to every request.
pub trait BoundaryRequest {
    /// Host-assigned identifier correlating this request with its record.
    fn correlation_id(&self) -> u64;

    /// The composed request body.
    fn body(&self) -> &[u8];
}

/// The terminal outcome of one request as the execution boundary sees it.
pub trait BoundaryTerminal {
    /// Whether the request reached a successful terminal.
    fn is_success(&self) -> bool;

    /// The `error.type` classification, or `None` on success.
    fn error_type(&self) -> Option<&str>;
}

/// A worker-local executor that drives one request to terminal.
///
/// `!Send` by construction: an executor lives on one current-thread runtime with
/// its worker-local connection pool and measurement state, so it needs no
/// synchronization.
pub trait RequestExecutor {
    /// Drive one request to terminal.
    fn execute<'a>(
        &'a self,
        request: &'a dyn BoundaryRequest,
    ) -> Pin<Box<dyn Future<Output = Box<dyn BoundaryTerminal>> + 'a>>;
}

/// A transport that runs inside the host's shared worker kernel.
pub trait RequestTransportExecution {
    /// Whether the transport can probe endpoint readiness.
    fn readiness(&self) -> ReadinessCapabilityV1;

    /// Whether and how the transport speaks WebSocket.
    fn websocket(&self) -> WebSocketCapabilityV1;

    /// Build this placement's executor. Called once per placement.
    fn build_executor(
        &self,
        context: RequestExecutionBuildContextV1,
    ) -> Result<Rc<dyn RequestExecutor>, ValidationError>;
}

/// One resolved gRPC method binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpcBindingV1 {
    /// Fully qualified `/package.Service/Method` path.
    pub path: String,
    /// Whether the client streams requests.
    pub is_client_streaming: bool,
    /// Whether the server streams responses.
    pub is_server_streaming: bool,
}

/// An optional endpoint-supplied gRPC method binding factory.
///
/// Only gRPC endpoint families implement this. The binding is resolved once at
/// startup; the concrete codec and the generated service stubs stay inside the
/// plugin, so no generated type crosses the boundary.
pub trait GrpcEndpointBindingFactory {
    /// The registered identifier of the endpoint family supplying the binding.
    fn id(&self) -> &RegistryId;

    /// Resolve one service/method pair into its wire binding.
    fn bind(&self, service: &str, method: &str) -> Result<GrpcBindingV1, ValidationError>;
}
