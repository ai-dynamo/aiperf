// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral load-generation dispatch, observation, and collection.
//!
//! This crate has no engine, router, KV, HTTP, or gRPC dependency. Transports
//! define their request types and implement [`Dispatchable`](sink::Dispatchable).
//!
//! - [`collector`] — `TraceCollector` and the report/record/status types.
//! - [`sink`] — `Dispatchable`, `RequestSink<R>`, `RequestObserver`.
//! - [`observer`] — `CollectorObserver`, a pure recorder into `TraceCollector`.

pub mod collector;
pub mod observer;
pub mod sink;
