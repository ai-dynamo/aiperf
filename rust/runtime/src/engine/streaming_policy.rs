// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Projection of the authored streaming reliability policy onto the frozen
//! host issue policy.
//!
//! The public Config-v2 surface exposes retry limits and cumulative admission
//! thresholds. It exposes no disposition, because the exhausted disposition of
//! every authorable `(scope, class)` pair is a property of the scope, not of the
//! operator's preference: a partition becomes a hole, a record or session is
//! quarantined, an action is finalized as a truthful terminal receipt, an export
//! is marked incomplete, and a checkpoint attempt applies backpressure. There is
//! no expansion of this table that yields `FailRun`, and
//! [`StreamingIssueThresholdRule::new`] rejects it independently.

use std::num::NonZeroU64;

use crate::engine::protocol_v2::StreamingReliabilityPolicyV2;
use crate::streaming::reliability::{
    PreparedStreamingIssuePolicy, StreamingIssueClass, StreamingIssueComponentId,
    StreamingIssueDisposition, StreamingIssueScopeKind, StreamingIssueThresholdRule,
    StreamingReliabilityError,
};

/// Every authorable issue class.
///
/// `Invariant` is deliberately absent: it is reachable only through the private
/// host classifier, and `is_allowed_authored_disposition` refuses every authored
/// rule for it.
const AUTHORABLE_CLASSES: [StreamingIssueClass; 3] = [
    StreamingIssueClass::Retryable,
    StreamingIssueClass::Permanent,
    StreamingIssueClass::Capacity,
];

/// One authorable scope and the policy fields that govern it.
struct ScopePolicy {
    scope: StreamingIssueScopeKind,
    name: &'static str,
    retry_limit: u32,
    exhausted: StreamingIssueDisposition,
    fence: Option<NonZeroU64>,
}

/// Project the authored policy onto one complete frozen rule set.
///
/// Emits exactly one wildcard rule per authorable `(scope, class)` pair — six
/// scopes by three classes — so `rule_for` always resolves and
/// [`PreparedStreamingIssuePolicy::new`]'s "every exact key needs a wildcard"
/// invariant holds for any exact rule a later task adds.
pub fn prepare_streaming_policy(
    policy: &StreamingReliabilityPolicyV2,
) -> Result<PreparedStreamingIssuePolicy, StreamingReliabilityError> {
    let scopes = [
        ScopePolicy {
            scope: StreamingIssueScopeKind::Partition,
            name: "partition",
            retry_limit: policy.partition_retry_limit,
            exhausted: StreamingIssueDisposition::Hole,
            fence: policy.partition_holes_before_admission_fence,
        },
        ScopePolicy {
            scope: StreamingIssueScopeKind::Record,
            name: "record",
            retry_limit: policy.partition_retry_limit,
            exhausted: StreamingIssueDisposition::Quarantine,
            fence: policy.quarantines_before_admission_fence,
        },
        ScopePolicy {
            scope: StreamingIssueScopeKind::Session,
            name: "session",
            retry_limit: policy.partition_retry_limit,
            exhausted: StreamingIssueDisposition::Quarantine,
            fence: policy.quarantines_before_admission_fence,
        },
        ScopePolicy {
            scope: StreamingIssueScopeKind::Action,
            name: "action",
            retry_limit: policy.endpoint_retry_limit,
            exhausted: StreamingIssueDisposition::TerminalActionReceipt,
            fence: policy.endpoint_failures_before_admission_fence,
        },
        ScopePolicy {
            scope: StreamingIssueScopeKind::Export,
            name: "export",
            retry_limit: policy.export_retry_limit,
            exhausted: StreamingIssueDisposition::ExportIncomplete,
            // An export fence is not authorable: fencing admission on a derived
            // sink fault would let a derived failure rewrite an execution
            // outcome.
            fence: None,
        },
        ScopePolicy {
            scope: StreamingIssueScopeKind::CheckpointAttempt,
            name: "checkpoint_attempt",
            retry_limit: policy.checkpoint_retry_limit,
            exhausted: StreamingIssueDisposition::Backpressure,
            fence: policy.checkpoint_failures_before_admission_fence,
        },
    ];

    let mut rules = Vec::with_capacity(scopes.len() * AUTHORABLE_CLASSES.len());
    for scope in scopes {
        for class in AUTHORABLE_CLASSES {
            let exhausted = match class {
                // Capacity is transient by definition; exhausting retries means
                // wait for capacity, never discard membership.
                StreamingIssueClass::Capacity => StreamingIssueDisposition::Backpressure,
                _ => scope.exhausted,
            };
            let rule_id =
                StreamingIssueComponentId::new(format!("{}_{}", scope.name, class_name(class)))?;
            rules.push(StreamingIssueThresholdRule::new(
                rule_id,
                scope.scope,
                class,
                None,
                scope.retry_limit,
                exhausted,
                scope.fence,
            )?);
        }
    }
    PreparedStreamingIssuePolicy::new(rules)
}

/// Stable lowercase rule-id fragment for one authorable class.
const fn class_name(class: StreamingIssueClass) -> &'static str {
    match class {
        StreamingIssueClass::Retryable => "retryable",
        StreamingIssueClass::Permanent => "permanent",
        StreamingIssueClass::Capacity => "capacity",
        // Unreachable through `AUTHORABLE_CLASSES`; kept total so adding a
        // variant is a compile error rather than a silent miscategorization.
        StreamingIssueClass::Invariant => "invariant",
    }
}
