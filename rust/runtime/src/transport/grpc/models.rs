// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC request, response, error, and trace models.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How channels are reused across requests.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectionReuseStrategy {
    /// One multiplexed channel per configured target.
    #[default]
    Pooled,
    /// A fresh channel per request.
    Never,
    /// One channel per correlated user session.
    StickyUserSessions,
}

/// Worker-local gRPC client policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcClientConfig {
    /// Maximum uncompressed inbound protobuf message size.
    pub max_receive_message_size: usize,
    /// Maximum uncompressed outbound protobuf message size.
    pub max_send_message_size: usize,
    /// Maximum Clock time spent establishing a ready channel.
    pub channel_ready_timeout_ns: i64,
    /// Number of additional attempts to establish a channel after a
    /// connect-phase failure (the pre-RPC `Endpoint::connect`). Retries apply
    /// only before any RPC is dispatched, so a request the server may have
    /// observed is never re-issued. Channel-ready timeouts, whole-request
    /// timeouts, and every post-send RPC status are never retried. `0` (the
    /// default) preserves the historical single-attempt behavior. This mirrors
    /// the HTTP transport's `ClientConfig::max_connect_retries`.
    pub max_connect_retries: u32,
    /// Base linear backoff, in Clock-nanoseconds, slept between connect
    /// retries. Retry `n` (1-based) waits `connect_retry_backoff_ns * n`, so
    /// successive waits grow linearly. The sleep is driven through the injected
    /// [`crate::clock::Clock`] so virtual-time replay stays deterministic.
    /// Non-positive (the default) disables the wait.
    pub connect_retry_backoff_ns: i64,
    /// Optional default whole-request timeout.
    pub total_timeout_ns: Option<i64>,
    /// Record per-message size/timestamp pairs in addition to totals.
    pub trace_chunks: bool,
    /// Verify the server's TLS certificate chain + hostname on `grpcs`. When
    /// `false`, any certificate is accepted (self-signed / untrusted test
    /// servers) — the gRPC equivalent of the HTTP transport's `ssl_verify=false`.
    /// Handshake signatures remain cryptographically verified.
    pub ssl_verify: bool,
}

impl Default for GrpcClientConfig {
    fn default() -> Self {
        Self {
            max_receive_message_size: 256 * 1024 * 1024,
            max_send_message_size: 256 * 1024 * 1024,
            channel_ready_timeout_ns: 30_000_000_000,
            max_connect_retries: 0,
            connect_retry_backoff_ns: 0,
            total_timeout_ns: None,
            trace_chunks: true,
            ssl_verify: true,
        }
    }
}

/// Per-dispatch gRPC configuration.
#[derive(Clone, Debug)]
pub struct GrpcRequestConfig {
    /// Endpoint-owned metadata, normalized to lowercase by the transport.
    pub metadata: BTreeMap<String, String>,
    /// Cancellation delay armed after the RPC future is first submitted.
    pub cancel_after_ns: Option<i64>,
    /// Correlation ID used by sticky channel reuse and metadata.
    pub correlation_id: Option<String>,
    /// Request ID carried in metadata and the dialect protobuf request.
    pub request_id: Option<String>,
    /// Whether this is the final turn in a correlated session.
    pub is_final_turn: bool,
    /// Channel reuse policy.
    pub reuse: ConnectionReuseStrategy,
    /// Selected configured URL index.
    pub url_index: Option<u32>,
    /// Effective model name.
    pub model_name: String,
    /// Whether server streaming is requested.
    pub streaming: bool,
    /// Optional per-request whole-lifecycle timeout override.
    pub total_timeout_ns: Option<i64>,
}

impl GrpcRequestConfig {
    /// Construct request policy for one model.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            metadata: BTreeMap::new(),
            cancel_after_ns: None,
            correlation_id: None,
            request_id: None,
            is_final_turn: true,
            reuse: ConnectionReuseStrategy::Pooled,
            url_index: None,
            model_name: model_name.into(),
            streaming: false,
            total_timeout_ns: None,
        }
    }

    /// Add one metadata entry.
    pub fn metadata(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(name.into(), value.into());
        self
    }

    /// Select streaming dispatch.
    pub const fn streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Select channel reuse policy.
    pub const fn reuse(mut self, reuse: ConnectionReuseStrategy) -> Self {
        self.reuse = reuse;
        self
    }

    /// Select one URL index.
    pub const fn url_index(mut self, index: u32) -> Self {
        self.url_index = Some(index);
        self
    }

    /// Set a request ID.
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Set a correlation ID.
    pub fn correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Arm cancellation after send submission.
    pub const fn cancel_after_ns(mut self, duration_ns: i64) -> Self {
        self.cancel_after_ns = Some(duration_ns);
        self
    }

    /// Mark whether this request ends its correlated session.
    pub const fn final_turn(mut self, is_final_turn: bool) -> Self {
        self.is_final_turn = is_final_turn;
        self
    }
}

/// One canonical response received from gRPC.
#[derive(Clone, Debug, PartialEq)]
pub struct GrpcResponse {
    /// Clock timestamp when the message was decoded.
    pub perf_ns: i64,
    /// Canonical endpoint JSON.
    pub json: Value,
    /// Unframed protobuf message size.
    pub wire_size: usize,
}

/// Stable gRPC failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrpcErrorKind {
    /// URL, metadata, endpoint binding, or serialization is invalid.
    InvalidRequest,
    /// A channel did not become ready within its capped send timeout.
    RequestSendTimeout,
    /// User-authored post-send cancellation fired.
    RequestCancellation,
    /// Whole-request timeout fired.
    RequestTimeout,
    /// Tonic returned a gRPC status.
    Rpc,
    /// A streaming envelope contained an in-band error.
    Stream,
    /// Canonical response decoding failed.
    Decode,
    /// Other transport failure.
    Other,
}

/// Structured request error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcErrorDetails {
    /// Stable failure category.
    pub kind: GrpcErrorKind,
    /// Human-readable diagnostic.
    pub message: String,
    /// HTTP-equivalent status used by common metrics and reporting.
    pub code: u16,
    /// Native gRPC status code when one exists.
    pub grpc_status_code: Option<i32>,
}

/// Per-request gRPC trace. Every timestamp is from the injected Clock.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GrpcTraceData {
    /// Transport discriminator for artifacts.
    pub trace_type: &'static str,
    /// Request metadata.
    pub request_metadata: BTreeMap<String, String>,
    /// Response initial/trailing metadata.
    pub response_metadata: BTreeMap<String, String>,
    /// Channel connect start.
    pub connect_start_ns: Option<i64>,
    /// Channel connect end.
    pub connect_end_ns: Option<i64>,
    /// Timestamp when an existing multiplexed channel was selected.
    pub channel_reused_ns: Option<i64>,
    /// Request submission start.
    pub request_send_start_ns: Option<i64>,
    /// Request metadata submission timestamp.
    pub request_headers_sent_ns: Option<i64>,
    /// Best available request-send completion anchor.
    pub request_send_end_ns: Option<i64>,
    /// Request protobuf message count.
    pub request_chunks_count: u32,
    /// Request protobuf bytes.
    pub request_bytes_total: u64,
    /// Optional request message timestamp/size pairs.
    pub request_chunks: Vec<(i64, u64)>,
    /// First response message timestamp.
    pub response_receive_start_ns: Option<i64>,
    /// Initial response metadata timestamp.
    pub response_headers_received_ns: Option<i64>,
    /// Last response/stream completion timestamp.
    pub response_receive_end_ns: Option<i64>,
    /// Response protobuf message count.
    pub response_chunks_count: u32,
    /// Response protobuf bytes.
    pub response_bytes_total: u64,
    /// Optional response message timestamp/size pairs.
    pub response_chunks: Vec<(i64, u64)>,
    /// HTTP-equivalent status.
    pub response_status_code: Option<u16>,
    /// Human-readable native status name.
    pub response_reason: Option<String>,
    /// Native gRPC status code.
    pub grpc_status_code: Option<i32>,
    /// Native gRPC status message.
    pub grpc_status_message: Option<String>,
    /// Error timestamp.
    pub error_timestamp_ns: Option<i64>,
}

impl GrpcTraceData {
    /// Construct a trace with its stable discriminator.
    pub fn new() -> Self {
        Self {
            trace_type: "grpc",
            ..Self::default()
        }
    }

    /// Best available request send duration.
    pub fn sending_ns(&self) -> Option<i64> {
        Some(self.request_send_end_ns? - self.request_send_start_ns?)
    }

    /// Send-complete to first response message.
    pub fn waiting_ns(&self) -> Option<i64> {
        Some(self.response_receive_start_ns? - self.request_send_end_ns?)
    }

    /// Response stream duration.
    pub fn receiving_ns(&self) -> Option<i64> {
        match self.response_chunks_count {
            0 => None,
            1 => Some(0),
            _ => Some(self.response_receive_end_ns? - self.response_receive_start_ns?),
        }
    }

    /// Request-send start through response completion.
    pub fn duration_ns(&self) -> Option<i64> {
        Some(self.response_receive_end_ns? - self.request_send_start_ns?)
    }
}

/// Complete result of one gRPC dispatch.
#[derive(Clone, Debug)]
pub struct GrpcRequestRecord {
    /// Dispatch start timestamp.
    pub start_ns: i64,
    /// Terminal timestamp.
    pub end_ns: Option<i64>,
    /// Cancellation timestamp.
    pub cancellation_ns: Option<i64>,
    /// HTTP-equivalent status.
    pub status: Option<u16>,
    /// Decoded response messages in arrival order.
    pub responses: Vec<GrpcResponse>,
    /// Structured terminal error.
    pub error: Option<GrpcErrorDetails>,
    /// Fine-grained trace.
    pub trace: GrpcTraceData,
    /// Exact unframed request protobuf bytes for artifacts.
    pub request_body: Bytes,
    /// Exact ordered unframed protobuf request messages.
    ///
    /// Unary and server-streaming requests contain one entry. Bidirectional
    /// requests retain the config message followed by every client chunk.
    pub request_messages: Vec<Bytes>,
}

impl GrpcRequestRecord {
    /// Construct an empty started record.
    pub fn started(start_ns: i64) -> Self {
        Self {
            start_ns,
            end_ns: None,
            cancellation_ns: None,
            status: None,
            responses: Vec::new(),
            error: None,
            trace: GrpcTraceData::new(),
            request_body: Bytes::new(),
            request_messages: Vec::new(),
        }
    }
}
