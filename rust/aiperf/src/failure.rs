// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified failure behavior for the scheduled and graph online execution paths.
//!
//! The production-convergence audit
//! (`specs/2026-07-13-scheduled-graph-production-convergence.md`) found that the
//! two online paths share their entire substrate except one thing: **what
//! happens when a request fails**. Scheduled online is resilient (record the
//! failed record, keep going); graph online is fail-fast (abort the trace, fail
//! the whole run on the first non-cancellation failure). That is a product
//! choice, not a necessity, so it becomes one configurable knob rather than two
//! hard-coded worlds.
//!
//! This module owns the greenfield-vocabulary
//! (`specs/2026-07-13-greenfield-execution-vocabulary.md`) selector for that
//! choice — [`OnFailure`] — deliberately at the crate root (not under `graph`)
//! because both paths share it. It is a plain `Copy` enum threaded by value; the
//! per-path failure *traits* (`graph::policy::{NodeFailurePolicy,
//! RunFailurePolicy}` and the scheduled latch) stay, because they carry the
//! async admission gate and abort-latch semantics the executors need. `OnFailure`
//! merely selects which concrete impl each path installs, replacing the previously
//! hard-coded picks — so a third failure discipline is still a new trait impl, and
//! the extension seam survives.
//!
//! Cancellation is never a failure on either path: it is an authored outcome, not
//! an error. Both the graph `FailFastRunFailurePolicy` (which ignores
//! `TraceError::Cancelled`) and the scheduled cancel arm treat it as resilient
//! regardless of the selected [`OnFailure`].

use serde::{Deserialize, Serialize};

/// How a run reacts to a failed request/node.
///
/// The wire form is the lowercase string `"continue"` / `"abort"`. The field is
/// optional on the request; when absent, each path applies its historical
/// default via [`OnFailure::for_scheduled_default`] /
/// [`OnFailure::for_graph_default`], so an unmodified request behaves exactly as
/// before this seam existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnFailure {
    /// Record the failed request/node and keep running (today's *scheduled*
    /// behavior). On the graph path this installs the dormant
    /// `ResilientNodeFailurePolicy` + `ContinueRunFailurePolicy`.
    Continue,
    /// Fail the whole benchmark on the first non-cancellation failure (today's
    /// *graph* behavior). On the graph path this installs
    /// `AbortTraceNodeFailurePolicy` + `FailFastRunFailurePolicy`; on the
    /// scheduled path it latches a run-level failure on a `Failed` terminal.
    Abort,
}

impl OnFailure {
    /// The scheduled path's historical default when no policy is configured:
    /// resilient (record-and-continue).
    pub const fn for_scheduled_default() -> Self {
        OnFailure::Continue
    }

    /// The graph path's historical default when no policy is configured:
    /// fail-fast (abort the run on the first failure), matching Python DAG
    /// `FAIL_FAST` parity.
    pub const fn for_graph_default() -> Self {
        OnFailure::Abort
    }

    /// Resolve an optional configured value against the scheduled default.
    pub fn scheduled_or_default(configured: Option<OnFailure>) -> Self {
        configured.unwrap_or_else(OnFailure::for_scheduled_default)
    }

    /// Resolve an optional configured value against the graph default.
    pub fn graph_or_default(configured: Option<OnFailure>) -> Self {
        configured.unwrap_or_else(OnFailure::for_graph_default)
    }

    /// Whether the first non-cancellation failure must fail the whole run.
    pub const fn is_abort(self) -> bool {
        matches!(self, OnFailure::Abort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_form_is_lowercase() {
        assert_eq!(
            serde_json::from_str::<OnFailure>("\"continue\"").unwrap(),
            OnFailure::Continue
        );
        assert_eq!(
            serde_json::from_str::<OnFailure>("\"abort\"").unwrap(),
            OnFailure::Abort
        );
        assert_eq!(
            serde_json::to_string(&OnFailure::Abort).unwrap(),
            "\"abort\""
        );
    }

    #[test]
    fn absent_config_uses_per_path_defaults() {
        assert_eq!(
            OnFailure::scheduled_or_default(None),
            OnFailure::Continue,
            "scheduled default is resilient"
        );
        assert_eq!(
            OnFailure::graph_or_default(None),
            OnFailure::Abort,
            "graph default is fail-fast"
        );
    }

    #[test]
    fn configured_value_overrides_default() {
        assert_eq!(
            OnFailure::scheduled_or_default(Some(OnFailure::Abort)),
            OnFailure::Abort
        );
        assert_eq!(
            OnFailure::graph_or_default(Some(OnFailure::Continue)),
            OnFailure::Continue
        );
    }
}
