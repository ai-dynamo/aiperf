// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared value boundary for native AIPerf plugins.
//!
//! Every type and trait reachable from a plugin category SDK lives here, not in
//! `aiperf-runtime`. The crate deliberately has no `[dependencies]` edge to the
//! runtime, and `aiperf-runtime` now depends on this crate, so Cargo's
//! normal-dependency cycle rejection makes a leak edge back into runtime
//! implementation a compile error rather than a review finding. Cargo does
//! permit dev-dependency cycles, so that one edge is a review matter rather
//! than a structural one.
//!
//! - [`artifact`] — the capability-limited artifact access every SDK consumes.
//! - [`capture`] — the schema-tagged capture projections an exporter requires.
//! - [`clock`] — the sleepable time source all measurement routes through.
//! - [`dispatch`] — the transport-neutral request/observer/sink seam.
//! - [`endpoint`] — segment handles, the narrow segment reader, authored
//!   overrides, and store-free WebSocket operation values.
//! - [`histogram`] — the one GenAI histogram vocabulary every exporter shares.
//! - [`measure`] — transport-neutral response, record, trace, and framing values.
//! - [`report`] — the atomic commit of a finalized report projection.

// `missing_docs` is not enabled yet: 63 public items moved byte-identically
// out of `aiperf-runtime` into `measure/` predate the boundary-documentation
// rule, and documenting them here would destroy the byte-identical-move
// property this extraction is verified against. Enabling the lint is tracked
// as its own change against `measure/`.

pub mod artifact;
pub mod capture;
pub mod clock;
pub mod dispatch;
pub mod endpoint;
pub mod histogram;
pub mod measure;
pub mod report;

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
