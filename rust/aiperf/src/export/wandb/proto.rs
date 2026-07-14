// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal hand-written `prost` mirror of the W&B `wandb_internal.proto`
//! `Record` messages we emit into the offline `.wandb` transaction log.
//!
//! Only the field numbers and wire types matter for byte-fidelity, so we vendor
//! exactly the subset the offline table upload needs rather than compiling the
//! whole schema. Every number here was read off the installed SDK descriptor
//! (`wandb.proto.wandb_internal_pb2`, wandb 0.28.0):
//!
//! - `Record` oneof `record_type` shares one field-number space; because we set
//!   at most one arm per record, modeling each arm as an independent `optional`
//!   field produces wire bytes identical to the real oneof.
//! - Field numbers cited inline (e.g. `run` = 17, `history` = 2) match the
//!   descriptor dump used to author this module.

use prost::Message;

/// Top-level datastore envelope. Exactly one `record_type` arm is set per
/// record; unset optional message fields encode to nothing.
#[derive(Clone, PartialEq, Message)]
pub struct Record {
    /// Monotonic record number stamped by the writer (1-based).
    #[prost(int64, tag = "1")]
    pub num: i64,
    /// History (logged step) record — oneof arm 2.
    #[prost(message, optional, tag = "2")]
    pub history: Option<HistoryRecord>,
    /// Summary (final values) record — oneof arm 3.
    #[prost(message, optional, tag = "3")]
    pub summary: Option<SummaryRecord>,
    /// Files-to-upload record — oneof arm 6.
    #[prost(message, optional, tag = "6")]
    pub files: Option<FilesRecord>,
    /// Control metadata (outside the oneof) — field 16.
    #[prost(message, optional, tag = "16")]
    pub control: Option<Control>,
    /// Run header/identity record — oneof arm 17.
    #[prost(message, optional, tag = "17")]
    pub run: Option<RunRecord>,
    /// Run-exit record — oneof arm 18.
    #[prost(message, optional, tag = "18")]
    pub exit: Option<RunExitRecord>,
    /// Datastore version header — oneof arm 21.
    #[prost(message, optional, tag = "21")]
    pub header: Option<HeaderRecord>,
    /// Deduplication uuid (outside the oneof) — field 19.
    #[prost(string, tag = "19")]
    pub uuid: String,
    /// Internal routing info (outside the oneof) — field 200.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

impl Record {
    /// Serialize to the length-delimited body the datastore frames.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode_to_vec()
    }
}

/// Per-record internal routing info; only `stream_id` is meaningful offline.
#[derive(Clone, PartialEq, Message)]
pub struct RecordInfo {
    /// Owning run/stream id.
    #[prost(string, tag = "1")]
    pub stream_id: String,
}

/// Datastore version header (first record in every `.wandb` file).
#[derive(Clone, PartialEq, Message)]
pub struct HeaderRecord {
    /// Producer / minimum-consumer version stamps.
    #[prost(message, optional, tag = "1")]
    pub version_info: Option<VersionInfo>,
    /// Internal routing info.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

/// Producer/consumer compatibility stamps carried by [`HeaderRecord`].
#[derive(Clone, PartialEq, Message)]
pub struct VersionInfo {
    /// Producing SDK version string.
    #[prost(string, tag = "1")]
    pub producer: String,
    /// Minimum consumer version able to read this stream.
    #[prost(string, tag = "2")]
    pub min_consumer: String,
}

/// Well-known `google.protobuf.Timestamp` (seconds + nanos).
#[derive(Clone, PartialEq, Message)]
pub struct Timestamp {
    /// Seconds since the Unix epoch.
    #[prost(int64, tag = "1")]
    pub seconds: i64,
    /// Nanosecond fraction in `[0, 1e9)`.
    #[prost(int32, tag = "2")]
    pub nanos: i32,
}

/// Run identity, config, tags, and start time.
#[derive(Clone, PartialEq, Message)]
pub struct RunRecord {
    /// Run id (matches the `.wandb` filename stem).
    #[prost(string, tag = "1")]
    pub run_id: String,
    /// Entity (team/user); empty defers to the account default.
    #[prost(string, tag = "2")]
    pub entity: String,
    /// Project name.
    #[prost(string, tag = "3")]
    pub project: String,
    /// Initial config payload.
    #[prost(message, optional, tag = "4")]
    pub config: Option<ConfigRecord>,
    /// Human-facing run name.
    #[prost(string, tag = "8")]
    pub display_name: String,
    /// Run tags.
    #[prost(string, repeated, tag = "10")]
    pub tags: Vec<String>,
    /// Host name.
    #[prost(string, tag = "13")]
    pub host: String,
    /// Wall-clock run start.
    #[prost(message, optional, tag = "17")]
    pub start_time: Option<Timestamp>,
    /// Internal routing info.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

/// Config delta (`update` adds/replaces keys; `remove` drops them).
#[derive(Clone, PartialEq, Message)]
pub struct ConfigRecord {
    /// Config keys to set.
    #[prost(message, repeated, tag = "1")]
    pub update: Vec<ConfigItem>,
    /// Internal routing info.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

/// One config entry; value is a JSON-encoded scalar/object string.
#[derive(Clone, PartialEq, Message)]
pub struct ConfigItem {
    /// Flat key (mutually exclusive with `nested_key`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Nested key path.
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    /// JSON-encoded value.
    #[prost(string, tag = "16")]
    pub value_json: String,
}

/// A single logged history step.
#[derive(Clone, PartialEq, Message)]
pub struct HistoryRecord {
    /// Logged items for this step.
    #[prost(message, repeated, tag = "1")]
    pub item: Vec<HistoryItem>,
    /// Step index.
    #[prost(message, optional, tag = "2")]
    pub step: Option<HistoryStep>,
}

/// History step index wrapper.
#[derive(Clone, PartialEq, Message)]
pub struct HistoryStep {
    /// Zero-based step number.
    #[prost(int64, tag = "1")]
    pub num: i64,
}

/// One history entry; value is a JSON-encoded scalar/object string.
#[derive(Clone, PartialEq, Message)]
pub struct HistoryItem {
    /// Flat key (mutually exclusive with `nested_key`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Nested key path.
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    /// JSON-encoded value.
    #[prost(string, tag = "16")]
    pub value_json: String,
}

/// Final run-summary values.
#[derive(Clone, PartialEq, Message)]
pub struct SummaryRecord {
    /// Summary keys to set.
    #[prost(message, repeated, tag = "1")]
    pub update: Vec<SummaryItem>,
}

/// One summary entry; value is a JSON-encoded scalar/object string.
#[derive(Clone, PartialEq, Message)]
pub struct SummaryItem {
    /// Flat key (mutually exclusive with `nested_key`).
    #[prost(string, tag = "1")]
    pub key: String,
    /// Nested key path.
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    /// JSON-encoded value.
    #[prost(string, tag = "16")]
    pub value_json: String,
}

/// Set of run files to persist/upload.
#[derive(Clone, PartialEq, Message)]
pub struct FilesRecord {
    /// Files, each relative to the run `files/` directory.
    #[prost(message, repeated, tag = "1")]
    pub files: Vec<FilesItem>,
    /// Internal routing info.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

/// One run file plus its upload policy/type enums.
#[derive(Clone, PartialEq, Message)]
pub struct FilesItem {
    /// Path relative to the run `files/` directory.
    #[prost(string, tag = "1")]
    pub path: String,
    /// Upload policy (`0=NOW`, `1=END`, `2=LIVE`).
    #[prost(int32, tag = "2")]
    pub policy: i32,
    /// File type (`0=OTHER`, `1=WANDB`, `2=MEDIA`, `3=ARTIFACT`).
    #[prost(int32, tag = "3")]
    pub r#type: i32,
}

/// Run termination record.
#[derive(Clone, PartialEq, Message)]
pub struct RunExitRecord {
    /// Process exit code (0 == success).
    #[prost(int32, tag = "1")]
    pub exit_code: i32,
    /// Wall-clock runtime seconds.
    #[prost(int32, tag = "2")]
    pub runtime: i32,
    /// Internal routing info.
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

/// Record control metadata; `always_send` on the exit record forces a flush.
#[derive(Clone, PartialEq, Message)]
pub struct Control {
    /// Force the record to be sent even under flow control.
    #[prost(bool, tag = "5")]
    pub always_send: bool,
}
