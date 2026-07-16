// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-protocol layer: the v2 protocol / registry / execution modules
//! relocated out of the `aiperf-runner` crate.
//!
//! This module hosts the ~30k-line v2 execution substrate — protocol envelopes,
//! the frozen transport/workload/pair registries, the execution factories and
//! drivers, dataset/graph input resolution, the coordinator/application
//! composition root, and the ancillary side-channel accumulators — so that the
//! `aiperf-runner` binary is reduced to a thin process shell (`main.rs`, the
//! cellular controller/cell, the control-plane HTTP surface, and signal
//! handling).
//!
//! It is gated behind the `runner-protocol` Cargo feature: only `aiperf-runner`
//! opts in, so `aiperf-mock-server`, `e2e`, and other library consumers pull
//! `aiperf` with default features and never compile this layer or its
//! dependency surface.
//!
//! The relocation tasks `git mv` the runner modules in here leaf-first
//! (protocol → registry → drivers → side-channels) and rewrite their
//! references.

pub mod application;
// Cross-host (k8s) cellular per-record artifact shipping over HTTP + streaming
// zstd (Stage E). Only reachable on the velo cellular path (it reuses the velo
// controller's bootstrap/DNS addressing), and its zstd streaming core is gated on
// the `zstd` dep the `velo` feature pulls in.
#[cfg(feature = "velo")]
pub mod artifact_shipping;
pub mod cellular_cell;
// Tier-T2 hierarchical merge: the aggregator role between cells and the controller.
#[cfg(feature = "velo")]
pub mod cellular_aggregator;
// The controller orchestration (cell launch + velo transport + merge) is only
// reachable with the `velo` feature; `owned_positions` (needed by the non-velo
// sharded runtime) lives in `cell_launcher` instead.
pub mod cell_launcher;
#[cfg(feature = "velo")]
pub mod cellular_controller;
pub mod control_plane_http;
pub mod coordinator;
pub mod dataset_input;
pub mod distribution_identity;
pub mod execute;
pub mod execution_factories;
pub mod gpu_telemetry;
pub mod graph_execution;
pub mod graph_input;
pub mod graph_phase_runtime;
pub mod grpc_execution;
pub mod grpc_turn_execution;
pub mod heartbeat_lane;
pub mod live_streaming;
pub mod network_latency;
#[cfg(feature = "dynosim")]
pub mod offline_execution;
pub mod online_execution;
pub mod protocol;
pub mod protocol_v2;
pub mod readiness;
pub mod record_lane;
pub mod records;
pub mod redaction;
pub mod registry;
pub mod server_metrics;
pub mod shard_artifacts;
pub mod sharded_scheduled;
pub mod sidecar_input;
pub mod turn_execution;
