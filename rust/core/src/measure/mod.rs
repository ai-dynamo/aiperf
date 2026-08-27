// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned transport-neutral measurement vocabulary.
//!
//! These types carry no wire-client state. Every transport and the metrics
//! plane share the same response, record, trace, error, server-sent-event,
//! AWS event-stream, and connection-reuse contracts, so a transport plugin is
//! authored against this module rather than against the runtime.
//!
//! Server-sent-event framing stays byte-oriented until complete lines are
//! available: a UTF-8 code point may span network chunks.

pub mod error;
pub mod eventstream;
pub mod record;
pub mod response;
pub mod reuse;
pub mod sse;
pub mod trace;

pub use error::{ErrorDetails, ErrorKind};
pub use eventstream::{
    EventStreamDecodeError, EventStreamDecoder, EventStreamEncodeError, EventStreamMessage,
};
pub use record::RequestRecord;
pub use response::{Response, TextResponse};
pub use reuse::ConnectionReuseStrategy;
pub use sse::{SseField, SseFieldName, SseMessage};
pub use trace::{TraceData, TraceExport, TraceReference};
