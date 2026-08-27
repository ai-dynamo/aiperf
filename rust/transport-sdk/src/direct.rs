// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The plugin-private execution shape a request transport is written against.
//!
//! [`ExecutionSinkBuilder`] and [`WorkerSink`] are deliberately **not** part of
//! `aiperf-plugin-api`. They are statically linked and monomorphized into each
//! transport plugin, so the worker loop calls the concrete sink directly. If
//! either trait appeared in a boundary signature the per-request path would
//! acquire a vtable hop the monolithic baseline does not have, and the plugin
//! split would show up as a measurable regression rather than as a refactor.
//!
//! The division of labour is:
//!
//! - [`ExecutionSinkBuilder`] is `Send + Sync` and crosses to a worker OS thread.
//!   It is the only thing that does. It builds one worker-local
//!   [`ExecutionSinkBuilder::Sink`] on the worker's own reactor, so `!Send`
//!   worker state (a hyper client, an `Rc<dyn Clock>`, a connection pool) never
//!   has to move between threads.
//! - [`WorkerSink`] is the worker-local dispatch half. It is `?Send` for the
//!   same reason.
//! - The associated `Sink` is bound `WorkerSink + RequestExecutor` because the
//!   `workers == 1` placement returns the sink *itself* as the executor: there is
//!   no co-located wrapper type, and adding one would put an indirection on the
//!   single-worker path that carries most runs.
//!
//! [`WorkerRequest`] and [`WorkerTerminal`] exist because the boundary
//! request/terminal traits are borrowed and `!Send`. A cross-thread hop needs
//! owned `Send` data at the channel, exactly as the baseline's `PreparedTurn`
//! does. The co-located path never constructs either.

use std::rc::Rc;

use aiperf_core::clock::Clock;
use aiperf_core::dispatch::RequestObserver;
use aiperf_plugin_api::transport::{BoundaryRequest, BoundaryTerminal, RequestExecutor};
use anyhow::Result;
use bytes::Bytes;
use uuid::Uuid;

/// One owned request, sendable across the worker hop.
///
/// The body is `Bytes`, so cloning the request into the command queue is a
/// refcount bump rather than a copy of the composed payload.
#[derive(Clone, Debug)]
pub struct WorkerRequest {
    correlation_id: u64,
    uuid: Uuid,
    body: Bytes,
    input_length: usize,
    max_output_tokens: usize,
    /// Session key used by the affinity routing policies, absent for a
    /// single-turn request that never needs to return to the same worker.
    session_key: Option<String>,
    is_final_turn: bool,
}

impl WorkerRequest {
    /// Bind one composed body to its correlation identity.
    pub fn new(correlation_id: u64, uuid: Uuid, body: Bytes) -> Self {
        Self {
            correlation_id,
            uuid,
            body,
            input_length: 0,
            max_output_tokens: 0,
            session_key: None,
            is_final_turn: true,
        }
    }

    /// Copy a borrowed boundary request into owned, sendable form.
    pub fn from_boundary(request: &dyn BoundaryRequest) -> Self {
        Self::new(
            request.correlation_id(),
            Uuid::nil(),
            Bytes::copy_from_slice(request.body()),
        )
    }

    /// Attach the measurement lengths the observer records on arrival.
    #[must_use]
    pub const fn with_lengths(mut self, input_length: usize, max_output_tokens: usize) -> Self {
        self.input_length = input_length;
        self.max_output_tokens = max_output_tokens;
        self
    }

    /// Bind this request to a logical session for affinity routing.
    ///
    /// `is_final_turn` releases the binding: without it the sticky map grows one
    /// entry per session for the lifetime of the run and never reclaims them.
    #[must_use]
    pub fn with_session(mut self, session_key: impl Into<String>, is_final_turn: bool) -> Self {
        self.session_key = Some(session_key.into());
        self.is_final_turn = is_final_turn;
        self
    }

    /// The per-request measurement identity.
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Prompt length in tokens.
    pub const fn input_length(&self) -> usize {
        self.input_length
    }

    /// Requested output length in tokens.
    pub const fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    /// The affinity key, when this request belongs to a multi-turn session.
    pub fn session_key(&self) -> Option<&str> {
        self.session_key.as_deref()
    }

    /// Whether this is the session's last turn.
    pub const fn is_final_turn(&self) -> bool {
        self.is_final_turn
    }
}

impl BoundaryRequest for WorkerRequest {
    fn correlation_id(&self) -> u64 {
        self.correlation_id
    }

    fn body(&self) -> &[u8] {
        &self.body
    }
}

/// One owned terminal outcome, sendable back across the worker hop.
#[derive(Clone, Debug, Default)]
pub struct WorkerTerminal {
    is_success: bool,
    error_type: Option<String>,
}

impl WorkerTerminal {
    /// A successful terminal.
    pub const fn success() -> Self {
        Self {
            is_success: true,
            error_type: None,
        }
    }

    /// A failed terminal carrying its `error.type` classification.
    pub fn failed(error_type: impl Into<String>) -> Self {
        Self {
            is_success: false,
            error_type: Some(error_type.into()),
        }
    }

    /// Copy a borrowed boundary terminal into owned, sendable form.
    pub fn from_boundary(terminal: &dyn BoundaryTerminal) -> Self {
        Self {
            is_success: terminal.is_success(),
            error_type: terminal.error_type().map(str::to_owned),
        }
    }
}

impl BoundaryTerminal for WorkerTerminal {
    fn is_success(&self) -> bool {
        self.is_success
    }

    fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }
}

/// The worker-local dispatch half of a request transport.
///
/// `?Send`: a sink lives on one current-thread runtime with its worker-local
/// client and measurement state, so it needs no synchronization and its dispatch
/// future is not required to be `Send`.
///
/// `dispatch_measured` deliberately does **not** wrap itself in
/// [`crate::measure::measure_dispatch`]. The capsule applies measurement once
/// around this call; a sink that measured internally as well would double-count
/// arrival and terminal facts and change recorded TTFT and latency.
#[async_trait::async_trait(?Send)]
pub trait WorkerSink {
    /// Anchor this worker's measurement to the run origin, in clock ns.
    fn set_run_origin(&self, origin_ns: i64);

    /// The worker-local clock every measurement and firing gate routes through.
    fn clock(&self) -> &dyn Clock;

    /// Whether this transport can stream partial responses.
    fn supports_response_streaming(&self) -> bool {
        false
    }

    /// Drive one request to terminal, emitting observations through `observer`.
    async fn dispatch_measured(
        &self,
        observer: &dyn RequestObserver,
        request: &dyn BoundaryRequest,
    ) -> Result<Box<dyn BoundaryTerminal>>;

    /// Warm connections before the measured phase. Unmeasured by construction.
    async fn prewarm(&self, _request: &dyn BoundaryRequest) -> Result<()> {
        Ok(())
    }

    /// Release worker-local resources at end of run.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Materializes a full request from an identity-only credit, worker-side.
///
/// This is the push-dispatch seam: the coordinator routes a credit carrying only
/// an identity, and the worker builds the body. That keeps the issuer out of
/// both the request's lifetime and its body construction.
///
/// `Send` because the materializer is built coordinator-side and moved onto the
/// worker thread once, alongside the builder.
pub trait CreditMaterializer: Send {
    /// Build the request this credit identifies.
    fn materialize(&self, correlation_id: u64) -> Result<WorkerRequest>;
}

/// Builds one worker-local sink per execution worker.
///
/// `Send + Sync + 'static` so a single builder can be shared by every worker
/// thread; the sinks it produces are not required to be either.
pub trait ExecutionSinkBuilder: Send + Sync + 'static {
    /// The concrete worker-local sink. Never named at the plugin boundary.
    ///
    /// Bound `RequestExecutor` as well as `WorkerSink` because the `workers == 1`
    /// placement returns the sink directly as the executor.
    type Sink: WorkerSink + RequestExecutor + 'static;

    /// A stable label for diagnostics; not a registry identifier.
    fn label(&self) -> &'static str;

    /// Build worker `worker_id`'s sink on that worker's own reactor.
    fn build_sink(&self, clock: Rc<dyn Clock>, worker_id: usize) -> Result<Self::Sink>;

    /// Build this worker's credit materializer, when the transport pushes.
    fn build_credit_materializer(&self) -> Result<Option<Box<dyn CreditMaterializer>>> {
        Ok(None)
    }
}
