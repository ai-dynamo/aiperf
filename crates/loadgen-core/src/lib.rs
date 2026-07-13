// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral load-generation seam plus the trace collector, extracted
//! lean from `dynamo-mocker`.
//!
//! This crate has **no** dependency on any engine, router, KV, or sim types.
//! The dispatch seam ([`sink`]) is generic over a [`Dispatchable`](sink::Dispatchable)
//! request; concrete request types (a slim HTTP request online, the mocker's
//! `DirectRequest` for the simulated engine) implement the trait in their own
//! crates, so nothing heavy leaks in here.
//!
//! - [`collector`] — `TraceCollector` and the report/record/status types.
//! - [`sink`] — `Dispatchable`, `RequestSink<R>`, `RequestObserver`.
//! - [`observer`] — `CollectorObserver`, a pure recorder into `TraceCollector`.

pub mod collector;
pub mod observer;
pub mod sink;
