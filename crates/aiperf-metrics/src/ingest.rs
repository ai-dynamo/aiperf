// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral record-ingest DTOs for the metrics plane.
//!
//! Ingest records deliberately carry raw measurement facts and optional explicit
//! metric overrides. Accumulators decide which catalog rows are computable for a
//! given export context; producers only fill this shape.

use crate::catalog::MetricTag;
use crate::value::MetricValue;
use crate::window::Phase;

/// Token counts attached to a completed request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCounts {
    /// Input tokens sent to the model.
    pub input: Option<u64>,
    /// Output tokens produced by the model, excluding reasoning tokens.
    pub output: Option<u64>,
    /// Reasoning tokens reported by the model.
    pub reasoning: Option<u64>,
    /// Requested output sequence length, when known from request parameters.
    pub requested_output: Option<u64>,
}

impl TokenCounts {
    /// Returns OSL including reasoning, preserving absent-vs-zero semantics.
    pub fn output_sequence_length(self) -> Option<u64> {
        (self.output.is_some() || self.reasoning.is_some())
            .then_some(self.output.unwrap_or(0) + self.reasoning.unwrap_or(0))
    }
}

/// Usage fields reported by an endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageMetrics {
    /// Prompt tokens from endpoint usage.
    pub prompt_tokens: Option<u64>,
    /// Completion tokens from endpoint usage.
    pub completion_tokens: Option<u64>,
    /// Total tokens from endpoint usage.
    pub total_tokens: Option<u64>,
    /// Reasoning tokens from endpoint usage.
    pub reasoning_tokens: Option<u64>,
    /// Prompt audio tokens from endpoint usage.
    pub prompt_audio_tokens: Option<u64>,
    /// Completion audio tokens from endpoint usage.
    pub completion_audio_tokens: Option<u64>,
    /// Accepted prediction tokens from endpoint usage.
    pub accepted_prediction_tokens: Option<u64>,
    /// Rejected prediction tokens from endpoint usage.
    pub rejected_prediction_tokens: Option<u64>,
    /// Prompt cache-read tokens from endpoint usage.
    pub prompt_cache_read_tokens: Option<u64>,
    /// Prompt cache-write tokens from endpoint usage.
    pub prompt_cache_write_tokens: Option<u64>,
    /// Prompt cache-miss tokens from endpoint usage.
    pub prompt_cache_miss_tokens: Option<u64>,
    /// Tool-use prompt tokens from endpoint usage.
    pub tool_use_prompt_tokens: Option<u64>,
    /// Prompt audio duration from endpoint usage.
    pub prompt_audio_seconds: Option<f64>,
}

/// HTTP timing trace attached to a request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HttpTrace {
    /// Time from dispatch start until the streaming response is established.
    pub stream_setup_ns: Option<i64>,
    /// Queue or connector blocked duration in nanoseconds.
    pub blocked_ns: Option<i64>,
    /// DNS lookup duration in nanoseconds.
    pub dns_lookup_ns: Option<i64>,
    /// TCP/TLS connection duration in nanoseconds.
    pub connecting_ns: Option<i64>,
    /// Request send duration in nanoseconds.
    pub sending_ns: Option<i64>,
    /// Server wait duration in nanoseconds.
    pub waiting_ns: Option<i64>,
    /// Response receive duration in nanoseconds.
    pub receiving_ns: Option<i64>,
    /// Total HTTP duration in nanoseconds.
    pub duration_ns: Option<i64>,
    /// Whether the request reused a connection.
    pub connection_reused: Option<bool>,
    /// Request bytes sent.
    pub data_sent_bytes: Option<u64>,
    /// Response bytes received.
    pub data_received_bytes: Option<u64>,
    /// Request chunks sent.
    pub chunks_sent: Option<u64>,
    /// Response chunks received.
    pub chunks_received: Option<u64>,
}

/// One completed request record ready for metric ingestion.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordIngest {
    /// Request correlation id.
    pub correlation_id: String,
    /// Session sequence number within the run.
    pub session_num: u64,
    /// Zero-based turn index within the session.
    pub turn_index: u32,
    /// Optional worker identity for per-worker analysis.
    pub worker_id: Option<String>,
    /// Optional conversation identity for multi-turn analysis.
    pub conversation_id: Option<String>,
    /// Record phase.
    pub phase: Phase,
    /// Request start timestamp in nanoseconds.
    pub start_ns: i64,
    /// Request end timestamp in nanoseconds.
    pub end_ns: i64,
    /// Admit timestamp in nanoseconds.
    pub admit_ns: Option<i64>,
    /// First token timestamp in nanoseconds.
    pub first_token_ns: Option<i64>,
    /// Second token timestamp in nanoseconds.
    pub second_token_ns: Option<i64>,
    /// First non-reasoning output token timestamp in nanoseconds.
    pub first_output_token_ns: Option<i64>,
    /// Token or chunk arrival timestamps in nanoseconds.
    pub token_arrival_ns: Vec<i64>,
    /// Whether the request ended in an error.
    pub errored: bool,
    /// Whether the request was canceled by policy.
    pub canceled: bool,
    /// Token counts measured locally.
    pub tokens: TokenCounts,
    /// Optional endpoint usage.
    pub usage: UsageMetrics,
    /// Optional HTTP trace.
    pub http: HttpTrace,
    /// Audio input duration in seconds.
    pub audio_duration_s: Option<f64>,
    /// Number of images in the request.
    pub num_images: Option<u64>,
    /// Video inference time in seconds as reported by the endpoint.
    pub video_inference_seconds: Option<f64>,
    /// Video peak memory in megabytes.
    pub video_peak_memory_mb: Option<f64>,
    /// Explicit metric values injected by endpoint, telemetry, or tests.
    pub metric_overrides: Vec<(MetricTag, MetricValue)>,
}

impl RecordIngest {
    /// Builds the minimal record used by append-only store tests.
    pub fn minimal(start_ns: i64, end_ns: i64, phase: Phase) -> Self {
        Self {
            correlation_id: format!("record-{start_ns}-{end_ns}"),
            session_num: 0,
            turn_index: 0,
            worker_id: None,
            conversation_id: None,
            phase,
            start_ns,
            end_ns,
            admit_ns: None,
            first_token_ns: None,
            second_token_ns: None,
            first_output_token_ns: None,
            token_arrival_ns: Vec::new(),
            errored: false,
            canceled: false,
            tokens: TokenCounts::default(),
            usage: UsageMetrics::default(),
            http: HttpTrace::default(),
            audio_duration_s: None,
            num_images: None,
            video_inference_seconds: None,
            video_peak_memory_mb: None,
            metric_overrides: Vec::new(),
        }
    }

    /// Returns request latency in nanoseconds.
    pub fn latency_ns(&self) -> i64 {
        self.end_ns - self.start_ns
    }

    /// Returns adjacent token/chunk inter-arrival latencies in nanoseconds.
    pub fn inter_chunk_latencies_ns(&self) -> Vec<i64> {
        self.token_arrival_ns
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect()
    }
}
