// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch vocabulary.
//!
//! These types carry no transport-specific state and are reused verbatim by
//! every transport (http, grpc, dynosim, dry_run) plus the metrics plane. They
//! were extracted out of `transport::http` so a non-HTTP transport can depend on
//! the shared request/record/response/trace/error vocabulary without pulling in
//! the hyper client. The module is ungated (generic); gRPC maps
//! [`ConnectionReuseStrategy`] and consumes [`RequestRecord`]/[`Response`]/
//! [`TraceData`]/[`ErrorDetails`] with no HTTP dependency.

pub mod dispatch;
pub mod error;
pub mod record;
pub mod response;
pub mod reuse;
pub mod sse;
pub mod trace;

pub use dispatch::{
    DispatchResult, Dispatcher, MeasuredContext, MeasuredOutcome, PreparedEndpoint, PreparedTurn,
    Request, RequestExecutor,
};
pub use error::{ErrorDetails, ErrorKind};
pub use record::RequestRecord;
pub use response::{Response, TextResponse};
pub use reuse::ConnectionReuseStrategy;
pub use sse::{SseField, SseFieldName, SseMessage};
pub use trace::{TraceData, TraceExport, TraceReference};
