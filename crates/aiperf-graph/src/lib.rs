// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph-IR async-dataflow workload driver.
//!
//! Runs a DAG of chat requests with fan-out/fan-in dependencies and firing-gate
//! timing: nodes fire when their input channels are ready, dispatch through the
//! extensible [`sink::GraphSink`] (for example, HTTP or an in-process engine), and
//! measurement flows to `loadgen_core`'s shared `TraceCollector`. Prompts are
//! materialized from a content-addressed [`segment::SegmentStore`] plus dynamic
//! predecessor replies.
//!
//! This crate stays independent of application backends. Its
//! [`runtime::drive_sim_with_source`] pump merges the [`aiperf_clock::SimClock`]
//! sleeper heap with any injected [`runtime::SimEventSource`]. The optional
//! Dynamo adapter lives in the `aiperf` application crate, so neither this
//! graph engine nor its clock/measurement leaves acquire a mocker dependency.

pub mod bench;
pub mod channel_store;
pub mod channels;
pub mod context;
pub mod dataset_lowering;
pub mod errors;
pub mod executor;
pub mod materialize;
pub mod model;
pub mod policy;
pub mod reducers;
pub mod run;
pub mod runtime;
pub mod scheduler;
pub mod segment;
pub mod sink;
mod syslimits;
pub mod transport_bench;
pub mod transport_sink;
pub mod validate;
pub mod wire;
pub mod workload;
