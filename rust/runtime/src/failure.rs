// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified failure behavior for the scheduled and graph online execution paths.
//!
//! Scheduled execution records failures and continues by default; graph
//! execution aborts on the first non-cancellation failure by default.
//!
//! Cancellation is never a failure on either path: it is an authored outcome, not
//! an error, regardless of the selected [`OnFailure`].

use serde::{Deserialize, Serialize};

/// How a run reacts to a failed request/node.
///
/// The wire form is the lowercase string `"continue"` / `"abort"`. The field is
/// optional on the request; when absent, each path applies its default via
/// [`OnFailure::for_scheduled_default`] or [`OnFailure::for_graph_default`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnFailure {
    /// Record the failed request or node and keep running.
    Continue,
    /// Fail the benchmark on the first non-cancellation failure.
    Abort,
}

impl OnFailure {
    /// The scheduled default: record the failure and continue.
    pub const fn for_scheduled_default() -> Self {
        OnFailure::Continue
    }

    /// The graph default: abort on the first failure.
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
