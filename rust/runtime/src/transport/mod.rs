// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport plane: the wire clients and sinks, grouped by transport.
//!
//! `http` = hyper HTTP client + SSE; `grpc` = Tonic gRPC client (feature-gated).
//! (`core` — the transport-neutral dispatch vocabulary — is added in a later stage.)
#[cfg(feature = "grpc")]
pub mod grpc;
pub mod http;
