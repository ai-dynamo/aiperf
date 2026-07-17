// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native HTTP serving for generated multimodal content.
//!
//! One run-owned server streams a path-confined directory,
//! exposes `/healthz`, and retains bounded full-lifecycle request records. The
//! separate [`ContentServerMediaPublisher`] implements AIPerf's synthetic-media
//! publication seam so generated images and videos become small HTTP URLs while
//! audio remains inline as required by the OpenAI `input_audio` shape.

mod error;
mod media_tag;
mod model;
mod publisher;
mod server;
mod tracker;

pub use error::{ContentServerError, Result};
pub use media_tag::{MediaTag, parse_media_tag, tag_media_urls};
pub use model::{
    ContentRecordSender, ContentRequestRecord, ContentServerStatus, RequestTrackerSnapshot,
};
pub use publisher::ContentServerMediaPublisher;
pub use server::{
    ContentServerConfig, ContentServerFactory, ContentServerRuntime, NativeContentServerFactory,
};
pub use tracker::{ContentServerClock, RequestTracker, SystemContentServerClock};
