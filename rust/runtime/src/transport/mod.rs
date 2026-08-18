// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire clients, sinks, and transport-neutral contracts.
//!
//! `http` contains Hyper and SSE support, `grpc` contains feature-gated Tonic
//! support, and ungated `core` contains shared dispatch vocabulary.
pub mod core;
#[cfg(feature = "grpc")]
pub mod grpc;
pub mod http;
pub(crate) mod measure;
pub(crate) mod reduce;
pub(crate) mod retry;
#[cfg(feature = "websocket")]
pub mod ws;
