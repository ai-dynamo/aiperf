// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in streaming source implementations.
//!
//! Each source is gated by the feature that owns its dependency graph, so a
//! build without `streaming-s3` links none of the AWS SDK. Registration into
//! `AIPerfRegistry` is a separate concern and is not declared here.

/// Narrow provider-neutral S3 listing and object-read seam.
#[cfg(feature = "streaming-s3")]
pub mod s3_client;

/// S3 reconciliation policy, identity, and finite/follow source.
#[cfg(feature = "streaming-s3")]
pub mod s3;
