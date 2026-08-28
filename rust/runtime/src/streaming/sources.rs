// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete streaming source implementations and their built-in registration.
//!
//! Each source is gated by the feature that owns its dependency graph, so a
//! build without `streaming-s3` links none of the AWS SDK.

pub mod hf_rows;
pub mod local;

/// Narrow provider-neutral S3 listing and object-read seam.
#[cfg(feature = "streaming-s3")]
pub mod s3_client;

/// S3 reconciliation policy, identity, and finite/follow source.
#[cfg(feature = "streaming-s3")]
pub mod s3;

use super::source::StreamingDatasetSourceFactory;

/// Return every streaming source factory compiled into this build.
///
/// Feature-gated sources append themselves here; the lightweight `streaming`
/// build contains only `local`.
#[must_use]
pub fn builtin_source_factories() -> Vec<Box<dyn StreamingDatasetSourceFactory>> {
    #[allow(unused_mut)]
    let mut factories: Vec<Box<dyn StreamingDatasetSourceFactory>> =
        vec![Box::new(local::LocalSourceFactory)];

    #[cfg(feature = "streaming-s3")]
    factories.push(Box::new(s3::S3SourceFactory));

    factories
}
