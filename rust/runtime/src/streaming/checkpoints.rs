// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference and persistent checkpoint backend implementations.

pub(crate) mod budget;
#[cfg(feature = "streaming-s3")]
pub mod aws_object_store;
pub mod lease_gc;
pub mod local;
pub mod memory;
pub mod none;
#[cfg(feature = "streaming-s3")]
pub mod object_store;
