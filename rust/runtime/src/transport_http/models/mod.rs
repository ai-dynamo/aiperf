// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Idiomatic Rust data models for the transport layer.

pub mod error;
pub mod record;
pub mod request;
pub mod response;
pub mod sse;
pub mod trace;

pub use error::{ErrorDetails, ErrorKind};
pub use record::RequestRecord;
pub use request::{ConnectionReuseStrategy, HttpVersion, RequestConfig};
pub use response::{Response, TextResponse};
pub use sse::{SseField, SseFieldName, SseMessage};
pub use trace::{TraceData, TraceExport, TraceReference};
