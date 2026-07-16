// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Idiomatic Rust data models for the transport layer.

pub mod request;
pub mod sse;

pub use request::{HttpVersion, RequestConfig};
pub use sse::{SseField, SseFieldName, SseMessage};
