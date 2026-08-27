// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared value boundary for native AIPerf plugins.
//!
//! Every type and trait reachable from a plugin category SDK lives here, not in
//! `aiperf-runtime`. The crate deliberately has no dependency on the runtime:
//! the measured plugin ABI closure is exactly what this crate exposes, so a
//! leak edge back into runtime implementation is a compile error rather than a
//! review finding.
//!
//! - [`artifact`] — the capability-limited artifact access every SDK consumes.
//! - [`clock`] — the sleepable time source all measurement routes through.
//! - [`dispatch`] — the transport-neutral request/observer/sink seam.
//! - [`endpoint`] — segment handles, the narrow segment reader, authored
//!   overrides, and store-free WebSocket operation values.
//! - [`measure`] — transport-neutral response, record, trace, and framing values.
//! - [`report`] — the atomic commit of a finalized report projection.

pub mod artifact;
pub mod clock;
pub mod dispatch;
pub mod endpoint;
pub mod measure;
pub mod report;

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
