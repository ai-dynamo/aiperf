// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-category boundary for native AIPerf plugins.
//!
//! This crate holds the plugin-private half of a request transport: everything a
//! transport author reuses that is *not* the boundary itself. The boundary —
//! `RequestTransportExecution`, `RequestExecutionBuildContextV1`, and
//! `RequestExecutor` — lives in `aiperf-plugin-api`, and the transport-neutral
//! values those contracts carry live in `aiperf-core`. Nothing defined here may
//! appear in an exported boundary signature.
//!
//! - [`direct`] — [`ExecutionSinkBuilder`] and [`WorkerSink`], the plugin-private
//!   execution shape, plus the owned request/terminal values the worker hop needs.
//! - [`execution`] — the generic execution capsule:
//!   [`build_native_request_executor`], [`ThreadPerCoreExecutor`], and the worker
//!   loop, monomorphized into each transport plugin.
//! - [`measure`] — worker-local measurement plumbing shared by every sink.
//! - [`reduce`] — transport-neutral reduction of decoded responses into observer
//!   facts.
//! - [`retry`] — the shared connect-phase retry policy.
//! - [`service_helpers`] — adapters over the narrow `aiperf_core::services`
//!   traits a direct transport receives in place of the runtime's `RunContext`.
//!
//! This crate must never depend on `aiperf-runtime`.

#![deny(missing_docs)]

pub mod direct;
pub mod execution;
pub mod measure;
pub mod reduce;
pub mod retry;

pub use direct::{
    CreditMaterializer, ExecutionSinkBuilder, WorkerRequest, WorkerSink, WorkerTerminal,
};
pub use execution::{
    HopRouting, NoopObserver, ThreadPerCoreExecutor, WORKER_COMMAND_QUEUE_DEPTH, WorkerClockFactory,
    WorkerCommand, WorkerLoad, WorkerMessage, WorkerObserverFactory,
    build_native_request_executor, execute_worker_command, pick_worker, run_worker,
    run_worker_thread,
};
pub use measure::{ArrivalFacts, WorkerMeasurement, measure_dispatch};
pub use reduce::{
    TokenEmitter, absorb_observed_endpoint_metrics, absorb_observed_usage, append_text,
};
pub use retry::retry_connect;

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
