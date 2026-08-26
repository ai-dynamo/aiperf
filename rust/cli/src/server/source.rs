// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The pluggable historical-results source the dashboard server browses.
//!
//! The dashboard's live plane is always the in-process session index; its
//! historical plane is whatever this trait resolves. `aiperf serve` and
//! `aiperf profile --serve` resolve it from a local results directory
//! ([`DiskSource`]); `aiperf kube dashboard` resolves it from the operator's
//! results API. Nothing else about the server, its routes, or its SPA changes
//! between the two.

use std::path::PathBuf;

use serde_json::Value;

use super::index::{RunEntry, scan_disk};

/// Historical runs the dashboard browses alongside the live session index.
///
/// Implementations are shared across the server's worker threads, so they must
/// be `Send + Sync`. Both methods fail soft: a source that cannot reach its
/// backing store returns an empty list or `None` rather than erroring the
/// request, so a transient backend outage degrades the run list instead of
/// breaking the dashboard.
pub trait HistoricalSource: Send + Sync {
    /// Every historical run entry this source can currently see.
    fn list(&self) -> Vec<RunEntry>;

    /// The run's full `native-v2.json` report, when the source can produce one.
    fn read_report(&self, run: &RunEntry) -> Option<Value>;
}

/// A filesystem-backed historical source: the bounded results-root walk that
/// `aiperf serve` and `aiperf profile --serve` have always used.
pub struct DiskSource {
    /// Results root walked for `native-v2.json` reports.
    pub root: PathBuf,
    /// Maximum walk depth, so a stray deep tree cannot stall the scan.
    pub max_depth: usize,
}

impl HistoricalSource for DiskSource {
    fn list(&self) -> Vec<RunEntry> {
        scan_disk(&self.root, self.max_depth)
    }

    fn read_report(&self, run: &RunEntry) -> Option<Value> {
        crate::sweep::aggregate::read_report_path(run.report_path.as_deref()?)
    }
}
