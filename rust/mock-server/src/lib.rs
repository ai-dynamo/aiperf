// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic inference server for AIPerf benchmarks and tests.
//!
//! Provides OpenAI-compatible APIs, multi-backend Prometheus metrics simulation,
//! configurable latency, and deterministic responses for benchmarking.

pub mod accuracy;
pub mod app;
pub mod balancer;
pub mod config;
pub mod dcgm;
pub mod fastmock;
pub mod grpc;
pub mod grpc_riva;
pub mod handlers;
pub mod latency;
pub mod listener;
pub mod metrics;
pub mod models;
pub mod prefix_cache;
pub mod prom;
pub mod scheduler;
pub mod state;
pub mod throughput;
pub mod tls;
pub mod tokens;

pub use app::{AppState, build_router};
pub use config::MockServerConfig;
