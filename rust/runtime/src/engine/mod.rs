// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 registry, preparation, execution, reporting, and cellular
//! coordination. Available under the `engine` feature.

pub mod application;
// Cross-host cellular artifact shipping reuses Velo bootstrap addressing and zstd.
#[cfg(feature = "cellular")]
pub mod artifact_shipping;
// Velo-streaming sibling of `artifact_shipping`: ships the same per-record artifact
// bytes over the shared cellular velo plane (no second port), backpressured + zstd.
#[cfg(feature = "cellular")]
pub mod artifact_stream_velo;
pub mod cell_launcher;
#[cfg(feature = "cellular")]
pub mod cellular_aggregator;
pub mod cellular_cell;
// Barrier-synchronized cross-cell timing origin (opt-in, feature-agnostic): a
// cell zeroes its record timeline at the velo START barrier instead of its
// post-setup local run start.
pub mod cell_origin;
#[cfg(feature = "cellular")]
pub mod cellular_controller;
pub mod cellular_kind;
pub mod control_hooks;
pub mod control_plane_http;
pub mod coordinator;
pub mod dataset_analysis_writer;
pub mod dataset_input;
pub mod distribution_identity;
pub mod dry_run;
pub mod execute;
pub mod execution_factories;
pub mod global_hop;
pub mod global_push;
pub mod gpu_telemetry;
pub mod graph_execution;
pub mod graph_input;
pub mod graph_phase_runtime;
#[cfg(feature = "grpc")]
pub mod grpc_execution;
#[cfg(feature = "grpc")]
pub mod grpc_turn_execution;
pub mod heartbeat_lane;
/// Legacy AgentX weka execution path (byte-exact agentic replay), selected by
/// `--weka-semantics legacy`.
pub mod legacy_agentx_execution;
pub mod live_streaming;
pub mod network_latency;
#[cfg(feature = "dynosim")]
pub mod offline_execution;
pub mod online_execution;
pub mod phase_identity;
pub mod phase_manifest;
pub mod protocol;
pub mod protocol_v2;
pub mod readiness;
pub mod record_lane;
pub mod records;
pub mod redaction;
pub mod registry;
pub mod server_metrics;
pub mod server_profiler;
pub mod shard_artifacts;
pub mod sharded_scheduled;
pub mod sidecar_input;
pub mod slurm_topology;
pub mod turn_execution;
#[cfg(test)]
mod workers_characterization;
#[cfg(feature = "websocket")]
pub mod ws_execution;
