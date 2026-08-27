// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch vocabulary.
//!
//! These types carry no wire-client state. The module is ungated so every
//! transport and the metrics plane can share request, record, response, trace,
//! error, and connection-reuse contracts.

pub mod dispatch;

// The value half of this vocabulary is boundary-owned and lives in
// `aiperf_core::measure`; these paths remain for runtime code and downstream
// crates that already import `crate::transport::core::*`.
pub use aiperf_core::measure::{error, eventstream, record, response, reuse, sse, trace};

pub use aiperf_core::measure::{
    ConnectionReuseStrategy, ErrorDetails, ErrorKind, EventStreamDecodeError, EventStreamDecoder,
    EventStreamEncodeError, EventStreamMessage, RequestRecord, Response, SseField, SseFieldName,
    SseMessage, TextResponse, TraceData, TraceExport, TraceReference,
};
pub use dispatch::{
    BoundedDecisionAdmission, BoundedDecisionMode, BoundedDecisionReader, CreditReportKind,
    DecisionAdmissionError, DispatchResult, Dispatcher, MeasuredContext, MeasuredOutcome,
    PreparedEndpointBinding, PreparedTurn, Request, RequestExecutor, WorkerCreditReport,
};
