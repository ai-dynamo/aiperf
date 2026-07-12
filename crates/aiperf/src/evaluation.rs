// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral evaluation workload and Rust-owned host-effect boundary.
//!
//! Evaluator providers own benchmark semantics and canonical aggregation.
//! AIPerf owns bounded unit/operation admission, logical route resolution,
//! upstream transport, retries, cancellation, accounting, and artifact sealing.
//! The implementation is split into policy-focused modules while this module
//! exposes the composition seams consumed by the sole `aiperf-runner` product
//! path.

pub mod arbiter;
pub mod host;
pub mod inference;
pub mod ledger;
pub mod retry;
