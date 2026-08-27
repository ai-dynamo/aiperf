// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native streaming dataset contracts and execution support.

pub mod blocking;
pub mod budget;
pub mod checkpoint;
pub mod checkpoint_backend;
pub mod checkpoints;
pub mod identity;
pub mod results;
pub mod unit;

/// Whether the lightweight native streaming runtime is compiled.
pub const STREAMING_RUNTIME_COMPILED: bool = true;

/// Whether S3 source and object-store support are compiled.
pub const STREAMING_S3_COMPILED: bool = cfg!(feature = "streaming-s3");
