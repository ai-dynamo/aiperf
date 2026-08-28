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
