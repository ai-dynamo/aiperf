// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed Config-v2 native streaming dataset resources.
//!
//! These are the user-authored `dataset_streams:` and `shadow_replay:` sections.
//! They are strict (`deny_unknown_fields`) because a misspelled streaming key
//! would otherwise be silently ignored by the deliberately lenient
//! [`BenchmarkConfig`](super::config::BenchmarkConfig) and would then execute a
//! different benchmark than the one authored.
//!
//! This module is compiled in every build, including builds without the
//! `streaming` feature, so it names no type from `crate::streaming`. The
//! protocol-v2 layer owns the bridge.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

/// One named component selection inside a stream.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingComponent {
    /// Stable registry identifier of the selected implementation.
    pub id: String,
    /// Factory-owned configuration, retained verbatim.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub config: serde_json::Map<String, serde_json::Value>,
}

/// Retention capacities for one stream.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamLimits {
    /// Simultaneously acquired immutable partitions.
    pub acquired_partitions: u64,
    /// Simultaneously retained decoded fragments.
    pub decoded_fragments: u64,
    /// Simultaneously retained decoded bytes.
    pub decoded_bytes: u64,
    /// In-memory session/decoder state bytes.
    pub state_memory: u64,
    /// Validated spill bytes for session/decoder state.
    pub state_disk: u64,
}

/// One authored dataset stream.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetStream {
    /// Run-unique stream name.
    pub id: String,
    /// Selected source implementation.
    pub source: StreamingComponent,
    /// Selected format implementation.
    pub format: StreamingComponent,
    /// Selected session program.
    pub session_program: StreamingComponent,
    /// Retention capacities.
    pub limits: StreamLimits,
}

/// The authored `dataset_streams:` section.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetStreams {
    /// Authored streams in authored order.
    pub items: Vec<DatasetStream>,
    /// Fault-handling policy for every stream in this resource.
    #[serde(default)]
    pub reliability: StreamingReliabilityPolicy,
}

/// The authored `dataset_streams.reliability:` block.
///
/// Reliability-first by construction: there is no authored disposition field of
/// any kind, so an ordinary data, endpoint, checkpoint, or export fault can
/// never be configured to fail the run. Only the private host classifier can
/// reach a terminal outcome, and it does so from verified invariants rather
/// than from authored policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingReliabilityPolicy {
    /// Retries before an immutable partition becomes a durable hole.
    #[serde(default = "default_partition_retry_limit")]
    pub partition_retry_limit: u32,
    /// Retries before one endpoint action is finalized as a failed terminal
    /// receipt. Zero unless the selected action sink proves retry safety.
    #[serde(default)]
    pub endpoint_retry_limit: u32,
    /// Retries before a checkpoint attempt applies backpressure and fences
    /// admission.
    #[serde(default = "default_checkpoint_retry_limit")]
    pub checkpoint_retry_limit: u32,
    /// Retries before a derived export is marked incomplete.
    #[serde(default = "default_export_retry_limit")]
    pub export_retry_limit: u32,
    /// Clock-driven delay between retry attempts, in milliseconds.
    #[serde(default = "default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,
    /// Cumulative partition holes before admission is fenced.
    #[serde(default)]
    pub partition_holes_before_admission_fence: Option<NonZeroU64>,
    /// Cumulative record/session quarantines before admission is fenced.
    #[serde(default)]
    pub quarantines_before_admission_fence: Option<NonZeroU64>,
    /// Cumulative committed failed-action receipts before admission is fenced.
    ///
    /// Cumulative, never consecutive: a consecutive counter would silently
    /// reset on an interleaved success and would therefore report a different
    /// threshold for the same fault set under a different arrival order.
    #[serde(default)]
    pub endpoint_failures_before_admission_fence: Option<NonZeroU64>,
    /// Cumulative checkpoint-attempt failures before admission is fenced.
    #[serde(default = "default_checkpoint_fence")]
    pub checkpoint_failures_before_admission_fence: Option<NonZeroU64>,
}

impl Default for StreamingReliabilityPolicy {
    fn default() -> Self {
        Self {
            partition_retry_limit: default_partition_retry_limit(),
            endpoint_retry_limit: 0,
            checkpoint_retry_limit: default_checkpoint_retry_limit(),
            export_retry_limit: default_export_retry_limit(),
            retry_backoff_ms: default_retry_backoff_ms(),
            partition_holes_before_admission_fence: None,
            quarantines_before_admission_fence: None,
            endpoint_failures_before_admission_fence: None,
            checkpoint_failures_before_admission_fence: default_checkpoint_fence(),
        }
    }
}

const fn default_partition_retry_limit() -> u32 {
    3
}

const fn default_checkpoint_retry_limit() -> u32 {
    3
}

const fn default_export_retry_limit() -> u32 {
    3
}

const fn default_retry_backoff_ms() -> u64 {
    100
}

fn default_checkpoint_fence() -> Option<NonZeroU64> {
    NonZeroU64::new(3)
}

/// Action family a session program may emit.
///
/// Mirrors the gated `crate::streaming::unit::DatasetActionKind` so this module
/// compiles without the `streaming` feature; the protocol-v2 layer owns the
/// one-way bridge.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetActionKind {
    /// Materialize and issue one endpoint request.
    Request,
    /// Execute one host-owned graph node.
    GraphNode,
    /// Publish a terminal session update.
    SessionTerminal,
}

/// Replay time interpretation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayTimeMode {
    /// Offsets are relative to the replay origin.
    Relative,
    /// Recorded event times are absolute UTC instants.
    Absolute,
}

/// The authored `shadow_replay.time` block.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayTimeConfig {
    /// Selected time interpretation.
    pub mode: ReplayTimeMode,
}

/// Completeness signal advancing the watermark.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WatermarkSource {
    /// Source partition order proves completeness.
    SourceOrder,
    /// Decoded event time proves completeness.
    EventTime,
}

/// Disposition for units behind the watermark.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LateUnitPolicy {
    /// A late unit fails the run.
    Fail,
    /// A late unit is dropped with explicitly lossy semantics.
    Drop,
}

/// The authored `shadow_replay.ordering` block.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OrderingConfig {
    /// Selected watermark source.
    pub watermark: WatermarkSource,
    /// Selected late-unit disposition.
    pub late: LateUnitPolicy,
}

/// Behavior when admission cannot keep up.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OverloadMode {
    /// Stall acquisition until capacity is returned.
    Backpressure,
    /// Shed admitted work with explicitly lossy semantics.
    Shed,
}

/// The authored `shadow_replay.overload` block.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverloadConfig {
    /// Selected overload behavior.
    pub mode: OverloadMode,
}

/// Checkpoint cadence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointMode {
    /// No checkpoint backend is selected.
    None,
    /// Commit one atomic generation on a fixed cadence.
    Periodic,
}

/// The authored `shadow_replay.checkpoint` block.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    /// Selected checkpoint cadence.
    pub mode: CheckpointMode,
    /// Commit cadence in seconds; `periodic` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_seconds: Option<f64>,
    /// Selected checkpoint backend; `periodic` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<StreamingComponent>,
}

/// The authored `shadow_replay:` section.
///
/// Duplicate action-kind rejection is deliberately not implemented here: this
/// map is re-encoded and re-decoded through the strict protocol-v2
/// `ShadowReplaySpecV2`, whose `unique_action_bindings` is the single
/// enforcement point.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowReplay {
    /// Name of the selected [`DatasetStream::id`].
    pub stream: String,
    /// Action-sink binding per emitted action kind.
    pub actions: BTreeMap<DatasetActionKind, StreamingComponent>,
    /// Time interpretation.
    pub time: ReplayTimeConfig,
    /// Ordering and late-unit policy.
    pub ordering: OrderingConfig,
    /// Overload behavior.
    pub overload: OverloadConfig,
    /// Checkpoint policy.
    pub checkpoint: CheckpointConfig,
}
