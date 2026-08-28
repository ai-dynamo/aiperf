// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiled streaming dataset format implementations.
//!
//! Each format is gated on the Cargo features its readers need, so a build that
//! drops those dependencies also drops the format from the registry inventory
//! instead of failing at selection time.

/// Baseten literal-prompt Parquet trace decoding requires the Arrow/Parquet readers.
#[cfg(feature = "parquet")]
pub mod baseten;
pub mod streaming_dynamo;
pub mod synthesis;

use std::sync::Arc;

use super::format::StreamingDatasetFormatFactory;

/// Return every streaming format factory compiled into this build.
///
/// Only formats whose decode authority is fully compiled in are built-ins.
/// `synthesis` binds the run's resolved tokenizer and its receipt, and
/// `streaming_dynamo` binds a host-prepared synthesis-profile digest, so both
/// are constructed by the run's composition root and registered as extensions
/// once that authority exists — a startup-only registry cannot fabricate it.
#[must_use]
pub fn builtin_format_factories() -> Vec<Arc<dyn StreamingDatasetFormatFactory>> {
    #[allow(unused_mut)]
    let mut factories: Vec<Arc<dyn StreamingDatasetFormatFactory>> = Vec::new();

    #[cfg(feature = "parquet")]
    factories.push(Arc::new(baseten::BasetenFormatFactory));

    factories
}
