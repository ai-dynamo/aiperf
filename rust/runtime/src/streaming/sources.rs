// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in streaming source registration.

pub mod local;

use super::source::StreamingDatasetSourceFactory;

/// Return every streaming source factory compiled into this build.
///
/// Feature-gated sources append themselves here; the lightweight `streaming`
/// build contains only `local`.
#[must_use]
pub fn builtin_source_factories() -> Vec<Box<dyn StreamingDatasetSourceFactory>> {
    vec![Box::new(local::LocalSourceFactory)]
}
