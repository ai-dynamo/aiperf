// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-owned WebSocket transport sink leaf (Phase 3 stub).
//!
//! This module establishes the module path and candidate-inventory entry for the
//! native WebSocket transport sink.  The types listed below are currently defined
//! in `engine::ws_execution` and re-exported here; the full move of their
//! definitions into this module is deferred to the Task 33 plugin extraction.
//!
//! Re-exported from `engine::ws_execution`:
//! - [`WebSocketSinkBuilder`] — [`crate::engine::turn_execution::ExecutionSinkBuilder`]
//!   that prepares per-run state
//! - [`WebSocketTransportSink`] — [`crate::engine::turn_execution::WorkerSink`] and
//!   [`crate::transport::core::RequestExecutor`] that executes individual requests
//!
//! The host adapter (`engine/ws_execution.rs`) retains until Task 33:
//! - `WebSocketNativeExecution` — `NativeTransportExecution` impl
//! - `WebSocketExecutionFactory` — `RequestExecutorFactory` impl
//! - Socket-pool management and endpoint wiring

// Stub re-export: definitions live in engine::ws_execution until Task 33 extraction.
pub(crate) use crate::engine::ws_execution::{WebSocketSinkBuilder, WebSocketTransportSink};
