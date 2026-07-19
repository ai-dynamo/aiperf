// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral load-generation dispatch, observation, and collection.
//!
//! This module has no direct engine, router, KV, HTTP, or gRPC dependency of
//! its own; transports elsewhere in this crate define their request types and
//! implement [`Dispatchable`](sink::Dispatchable) against it. The separation
//! is a module-level convention (not a compiler-enforced crate boundary) —
//! see `specs/loadgen-core-merge.md` for why this was formerly a standalone
//! crate and why that boundary was retired.
//!
//! - [`collector`] — `TraceCollector` and the report/record/status types.
//! - [`sink`] — `Dispatchable`, `RequestSink<R>`, `RequestObserver`.
//! - [`observer`] — `CollectorObserver`, a pure recorder into `TraceCollector`.

pub mod collector;
pub mod observer;
pub mod sink;
