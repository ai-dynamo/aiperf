// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch vocabulary.
//!
//! These types carry no wire-client state. The module is ungated so every
//! transport and the metrics plane can share request, record, response, trace,
//! error, and connection-reuse contracts.

pub mod dispatch;
pub mod error;
pub mod record;
pub mod response;
pub mod reuse;
pub mod sse;
pub mod trace;

pub use dispatch::{
    DispatchResult, Dispatcher, MeasuredContext, MeasuredOutcome, PreparedEndpointBinding,
    PreparedTurn, Request, RequestExecutor,
};
pub use error::{ErrorDetails, ErrorKind};
pub use record::RequestRecord;
pub use response::{Response, TextResponse};
pub use reuse::ConnectionReuseStrategy;
pub use sse::{SseField, SseFieldName, SseMessage};
pub use trace::{TraceData, TraceExport, TraceReference};
