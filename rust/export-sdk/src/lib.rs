// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exporter-category boundary for native AIPerf plugins.
//!
//! This crate owns the shared, pure, plugin-private helpers every exporter
//! implementation leaf would otherwise re-derive. It defines no boundary type:
//! the factory traits, prepared handles, capture requirements, and error
//! vocabulary belong to `aiperf-plugin-api`, and the capture projections and
//! artifact capability belong to `aiperf-core`. Everything here consumes and
//! produces those types.
//!
//! - [`helpers`] — CSV dialect, endpoint display, run naming, and the two
//!   report-value finiteness policies.
//! - [`capture_helpers`] — navigation of the finalized report projection and the
//!   exact/folded capture projections.
//! - [`artifact_helpers`] — writing artifacts through
//!   [`aiperf_core::artifact::ArtifactAccess`] alone.
//! - [`prepared_helpers`] — requiring captures and building a
//!   [`aiperf_plugin_api::PreparedExporterV1`] from a closure.
//!
//! # What this crate must not do
//!
//! No helper accepts a raw artifact directory, a runtime-private report or
//! config type, or an `aiperf-runtime` value. A type defined here never occurs
//! in an exported boundary signature, trait-object vtable, or host-owned stored
//! value: the compiled artifact is a plugin-private input, statically linked
//! into each exporter package.

#![deny(missing_docs)]

pub mod artifact_helpers;
pub mod capture_helpers;
pub mod helpers;
pub mod prepared_helpers;

pub use artifact_helpers::{append_bytes, write_bytes, write_csv, write_json, write_text};
pub use capture_helpers::{
    CanonicalStats, SummarySeries, flatten_stats, histogram_series, record_metric, report_metric,
    report_metrics, successful_records, summary_series,
};
pub use helpers::{
    crlf_csv_writer, default_run_name, finite_guarded, finite_passthrough,
    normalize_endpoint_display,
};
pub use prepared_helpers::{
    ClosureExporter, require_exact_records, require_histograms, require_report,
};

/// The source API version exposed by provisional plugin crate shells.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";
