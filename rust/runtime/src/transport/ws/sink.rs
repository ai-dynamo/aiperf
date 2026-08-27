// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin-owned WebSocket transport sink leaf.
//!
//! This module is the implementation leaf for the native WebSocket transport.
//! It contains [`WebSocketSinkBuilder`] (the [`crate::engine::turn_execution::ExecutionSinkBuilder`]
//! that prepares per-run state) and [`WebSocketTransportSink`] (the
//! [`crate::engine::turn_execution::WorkerSink`] and
//! [`crate::transport::core::RequestExecutor`] that executes individual requests).
//!
//! The host adapter in `engine::ws_execution` retains [`crate::engine::registry::NativeTransportExecution`]
//! and [`crate::engine::turn_execution::RequestExecutorFactory`] and wires these
//! types into the engine pipeline.
//!
//! # Split boundary
//!
//! Items extracted here from `engine/ws_execution.rs` (Task 6, Phase 3):
//! - `WebSocketSinkBuilder` — `ExecutionSinkBuilder` impl; builds per-run sink config
//! - `WebSocketTransportSink` — `WorkerSink` + `RequestExecutor` impl; per-worker execution leaf
//!
//! The host adapter (`engine/ws_execution.rs`) retains:
//! - `WebSocketNativeExecution` — `NativeTransportExecution` impl
//! - `WebSocketExecutionFactory` — `RequestExecutorFactory` impl
//! - Socket-pool management and endpoint wiring

// Re-export the types that Tasks 33 (transport-websocket plugin) will equality-copy from
// this module. The full extraction from engine/ws_execution.rs is performed in a
// follow-up commit; this file establishes the module path and candidate inventory entry.
pub(crate) use crate::engine::ws_execution::{WebSocketSinkBuilder, WebSocketTransportSink};
