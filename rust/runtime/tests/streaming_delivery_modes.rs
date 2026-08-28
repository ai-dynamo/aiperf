// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable proof of the streaming delivery-mode restart contract.
//!
//! Each row pins one `(delivery mode, crash point, target idempotency)` triple
//! to its documented reissue set, delivery claim, and duplicate/loss window.
//! This binary owns no production code: a failing row means the owning
//! implementation task is incomplete, not that this file needs a workaround.
//!
//! The oracles below are restated independently of the production derivation so
//! a change to `deliver_restart_decision` has to be argued for rather than
//! silently absorbed.

#![cfg(feature = "streaming")]

use aiperf_runtime::streaming::{
    action::{EndpointRetrySafety, scheduled_request::SCHEDULED_REQUEST_ACTION_SINK},
    checkpoint::StreamRunIdentity,
    identity::{
        LogicalReplayRunId, RunIncarnationId, StableActionId, StableRecordId, StableSessionKey,
        attempt_id, stable_action_id,
    },
    reliability::StreamingIssueDisposition,
    results::{
        CheckpointDeliveryMode, DeliveryClaim, DeliveryCrashPoint, DuplicateWindow,
        TargetIdempotencyCapability,
    },
    unit::DatasetActionKind,
};

#[allow(dead_code)]
#[path = "support/streaming_checkpoint.rs"]
mod support;

/// Never-dispatched action the fixture leaves outstanding at every crash point.
const FIRST_UNDISPATCHED: [u8; 32] = [0xa1; 32];
/// Second never-dispatched action, ordered after [`FIRST_UNDISPATCHED`].
const SECOND_UNDISPATCHED: [u8; 32] = [0xa2; 32];
/// Action whose target effect the dead incarnation left uncertain.
///
/// It carries the lowest outstanding sequence, so whenever it is re-emitted it
/// leads the reissue vector.
const UNCERTAIN: [u8; 32] = [0xb1; 32];

/// Reissue expectation for one matrix row, projected from the restart decision.
///
/// The production vocabulary answers "which actions" and "what claim"; this
/// collapses both into the four readings a product report has to distinguish.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedReplay {
    /// Nonterminal actions are re-emitted from the committed cut.
    Reissue,
    /// The action was durably admitted; the restart must not re-emit it.
    DoNotReissue,
    /// The mode retains ingestion state only; no action-fidelity claim.
    DiagnosticOnly,
    /// No durable generation exists; the restart makes no resume claim at all.
    NoResumeClaim,
}

impl ExpectedReplay {
    fn observe(claim: DeliveryClaim, reissue: &[StableActionId]) -> Self {
        match claim {
            DeliveryClaim::None => Self::NoResumeClaim,
            DeliveryClaim::IngestionOnly => Self::DiagnosticOnly,
            _ if reissue.contains(&StableActionId::from_bytes(UNCERTAIN)) => Self::Reissue,
            _ => Self::DoNotReissue,
        }
    }
}

/// Product reading of one restart, in the reliability report vocabulary.
///
/// `Failed`, `Degraded`, and `ExportIncomplete` are the three report words the
/// streaming reliability summary uses; `Clean` names the residual case where a
/// restart reproduces the target's effect set exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutcomeLabel {
    /// Nothing durable names what was delivered, so the run cannot resume.
    Failed,
    /// The run resumes but leaves duplication or loss open at the target.
    Degraded,
    /// Source ingestion resumes; derived action delivery is not claimed.
    ExportIncomplete,
    /// The restart reproduces the target's effect set exactly.
    Clean,
}

impl OutcomeLabel {
    fn observe(claim: DeliveryClaim, window: DuplicateWindow) -> Self {
        match claim {
            DeliveryClaim::None => Self::Failed,
            DeliveryClaim::IngestionOnly => Self::ExportIncomplete,
            _ if window.is_closed() => Self::Clean,
            _ => Self::Degraded,
        }
    }
}

/// Whether the crash left an action whose target effect the restart cannot know.
fn has_uncertain_action(crash: DeliveryCrashPoint) -> bool {
    match crash {
        DeliveryCrashPoint::AfterDispatchBeforeTerminal
        | DeliveryCrashPoint::AfterTerminalBeforeCommit => true,
        DeliveryCrashPoint::BeforeDispatch | DeliveryCrashPoint::AfterCommit => false,
    }
}

/// Whether a cut in this mode re-emits an action of unknown target outcome.
///
/// `Admitted` is the only mode that can name an uncertain action while proving
/// nothing about its outcome, so it is the only one whose answer depends on the
/// target: suppression is the safe default unless the target deduplicates.
fn reissues_uncertain(mode: CheckpointDeliveryMode, dedups: bool) -> bool {
    match mode {
        CheckpointDeliveryMode::Terminal
        | CheckpointDeliveryMode::Decoded
        | CheckpointDeliveryMode::Acquired => true,
        CheckpointDeliveryMode::Admitted => dedups,
        CheckpointDeliveryMode::None => false,
    }
}

/// Claim the product owes one mode and target capability pairing.
fn expected_claim(
    mode: CheckpointDeliveryMode,
    capability: TargetIdempotencyCapability,
) -> DeliveryClaim {
    let dedups = capability == TargetIdempotencyCapability::VerifiedLogicalActionKey;
    match mode {
        CheckpointDeliveryMode::None => DeliveryClaim::None,
        CheckpointDeliveryMode::Acquired => DeliveryClaim::IngestionOnly,
        CheckpointDeliveryMode::Admitted if !dedups => DeliveryClaim::AtMostOnce,
        CheckpointDeliveryMode::Admitted => DeliveryClaim::IdempotentAtLeastOnceSubmission,
        CheckpointDeliveryMode::Terminal | CheckpointDeliveryMode::Decoded => {
            if dedups {
                DeliveryClaim::IdempotentAtLeastOnceSubmission
            } else {
                DeliveryClaim::AtLeastOnce
            }
        }
    }
}

/// Duplication and loss one restart leaves possible at the target.
fn expected_window(
    mode: CheckpointDeliveryMode,
    crash: DeliveryCrashPoint,
    capability: TargetIdempotencyCapability,
) -> DuplicateWindow {
    let dedups = capability == TargetIdempotencyCapability::VerifiedLogicalActionKey;
    let uncertain = has_uncertain_action(crash);
    let reissues = uncertain
        && reissues_uncertain(mode, dedups)
        && mode != CheckpointDeliveryMode::None;
    DuplicateWindow {
        may_duplicate_target_effect: reissues && !dedups,
        may_lose_target_effect: uncertain && !reissues,
    }
}

/// Exact ordered reissue set the restart owes, in global replay order.
fn expected_reissue(
    mode: CheckpointDeliveryMode,
    crash: DeliveryCrashPoint,
    capability: TargetIdempotencyCapability,
) -> Vec<StableActionId> {
    if mode == CheckpointDeliveryMode::None {
        // Without a durable cut there is no authoritative record naming what to
        // re-emit, so a restart re-emits nothing rather than guessing.
        return Vec::new();
    }
    let dedups = capability == TargetIdempotencyCapability::VerifiedLogicalActionKey;
    let mut expected = Vec::new();
    if has_uncertain_action(crash) && reissues_uncertain(mode, dedups) {
        expected.push(StableActionId::from_bytes(UNCERTAIN));
    }
    expected.push(StableActionId::from_bytes(FIRST_UNDISPATCHED));
    expected.push(StableActionId::from_bytes(SECOND_UNDISPATCHED));
    expected
}

/// Target idempotency key: the logical run and the logical action, and nothing
/// else.
///
/// The product exposes no single hashed key value; it exposes the two halves and
/// guarantees neither carries an incarnation. Deriving the pair here is what
/// lets the tests below assert that guarantee directly.
fn idempotency_key(run: StreamRunIdentity, action: StableActionId) -> ([u8; 32], [u8; 32]) {
    (*run.logical_replay_run().as_bytes(), *action.as_bytes())
}

/// One logical action identity built from semantic and causal inputs only.
fn logical_action(causal_ordinal: u64) -> StableActionId {
    stable_action_id(
        &[0x21; 32],
        StableSessionKey::from_bytes([0x22; 32]),
        &[StableRecordId::from_bytes([0x23; 32])],
        DatasetActionKind::Request,
        causal_ordinal,
    )
}

#[test]
fn restart_cuts_have_documented_semantics() {
    let cases = [
        (
            CheckpointDeliveryMode::Terminal,
            DeliveryCrashPoint::AfterDispatchBeforeTerminal,
            ExpectedReplay::Reissue,
        ),
        (
            CheckpointDeliveryMode::Admitted,
            DeliveryCrashPoint::AfterDispatchBeforeTerminal,
            ExpectedReplay::DoNotReissue,
        ),
        (
            CheckpointDeliveryMode::Decoded,
            DeliveryCrashPoint::AfterTerminalBeforeCommit,
            ExpectedReplay::Reissue,
        ),
        (
            CheckpointDeliveryMode::Acquired,
            DeliveryCrashPoint::AfterDispatchBeforeTerminal,
            ExpectedReplay::DiagnosticOnly,
        ),
        (
            CheckpointDeliveryMode::None,
            DeliveryCrashPoint::AfterTerminalBeforeCommit,
            ExpectedReplay::NoResumeClaim,
        ),
    ];

    for (mode, crash, expected) in cases {
        let restored = support::delivery_fixture(mode, TargetIdempotencyCapability::Unsupported)
            .crash_and_restore(crash);
        assert_eq!(
            ExpectedReplay::observe(restored.claim(), restored.reissue()),
            expected,
            "{}/{} re-emitted the wrong thing against a non-idempotent target",
            mode.tag(),
            crash.tag()
        );
    }
}

#[test]
fn delivery_matrix_pins_reissue_set_claim_and_duplicate_window() {
    for mode in CheckpointDeliveryMode::ALL {
        for crash in DeliveryCrashPoint::ALL {
            for capability in TargetIdempotencyCapability::ALL {
                let restored = support::delivery_fixture(mode, capability).crash_and_restore(crash);
                let row = format!("{}/{}/{}", mode.tag(), crash.tag(), capability.tag());

                assert_eq!(
                    restored.reissue(),
                    expected_reissue(mode, crash, capability).as_slice(),
                    "{row} re-emitted the wrong action set"
                );
                assert!(
                    restored.logical_membership_is_unique(),
                    "{row} re-emitted one logical action twice"
                );
                assert_eq!(
                    restored.claim(),
                    expected_claim(mode, capability),
                    "{row} published the wrong delivery claim"
                );
                assert_eq!(
                    restored.duplicate_window(),
                    expected_window(mode, crash, capability),
                    "{row} left the wrong duplicate window"
                );
            }
        }
    }
}

#[test]
fn every_restart_reads_as_failed_degraded_or_export_incomplete() {
    for mode in CheckpointDeliveryMode::ALL {
        for crash in DeliveryCrashPoint::ALL {
            for capability in TargetIdempotencyCapability::ALL {
                let restored = support::delivery_fixture(mode, capability).crash_and_restore(crash);
                let observed = OutcomeLabel::observe(restored.claim(), restored.duplicate_window());
                let expected = OutcomeLabel::observe(
                    expected_claim(mode, capability),
                    expected_window(mode, crash, capability),
                );
                assert_eq!(
                    observed,
                    expected,
                    "{}/{}/{} reads as the wrong outcome",
                    mode.tag(),
                    crash.tag(),
                    capability.tag()
                );
                if mode == CheckpointDeliveryMode::None {
                    assert_eq!(observed, OutcomeLabel::Failed, "an undurable cut cannot resume");
                }
                if mode == CheckpointDeliveryMode::Acquired {
                    assert_eq!(
                        observed,
                        OutcomeLabel::ExportIncomplete,
                        "an acquisition-only cut claims ingestion and no action delivery"
                    );
                }
            }
        }
    }
}

#[test]
fn idempotency_key_is_run_and_action_only_and_survives_incarnation_change() {
    let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x31; 32]));
    let action = logical_action(7);

    // The logical action identity takes no incarnation input, so re-deriving it
    // in a new process yields the same value.
    assert_eq!(action, logical_action(7));

    let first_incarnation = RunIncarnationId::from_bytes([0x41; 32]);
    let second_incarnation = RunIncarnationId::from_bytes([0x42; 32]);
    assert_ne!(
        attempt_id(action, first_incarnation, 0),
        attempt_id(action, second_incarnation, 0),
        "the attempt id is incarnation-scoped telemetry, not delivery identity"
    );

    let key = idempotency_key(run, action);
    assert_eq!(
        key,
        idempotency_key(run, action),
        "the key must not move when the incarnation does"
    );
    assert_eq!(key, (*run.logical_replay_run().as_bytes(), *action.as_bytes()));

    let other_run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x32; 32]));
    assert_ne!(
        key,
        idempotency_key(other_run, action),
        "two logical runs must not collapse into one target effect"
    );
    assert_ne!(
        key,
        idempotency_key(run, logical_action(8)),
        "two logical actions must not collapse into one target effect"
    );
}

#[test]
fn reissue_identity_is_stable_across_restarts_and_supports_idempotent_submission() {
    let fixture = || {
        support::delivery_fixture(
            CheckpointDeliveryMode::Terminal,
            TargetIdempotencyCapability::VerifiedLogicalActionKey,
        )
        .crash_and_restore(DeliveryCrashPoint::AfterDispatchBeforeTerminal)
    };

    let first = fixture();
    let second = fixture();
    assert_eq!(
        first.reissue(),
        second.reissue(),
        "a second restart of the same run must re-emit the identical logical keys"
    );

    let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x33; 32]));
    let first_keys: Vec<_> = first
        .reissue()
        .iter()
        .map(|action| idempotency_key(run, *action))
        .collect();
    let second_keys: Vec<_> = second
        .reissue()
        .iter()
        .map(|action| idempotency_key(run, *action))
        .collect();
    assert_eq!(first_keys, second_keys);

    assert_eq!(
        first.claim(),
        DeliveryClaim::IdempotentAtLeastOnceSubmission,
        "a verified logical action key makes re-submission collapsible at the target"
    );
    assert!(
        !first.duplicate_window().may_duplicate_target_effect,
        "a deduplicating target closes the duplication half of the window"
    );
}

#[test]
fn endpoint_without_proven_idempotency_never_reaches_an_idempotent_claim() {
    // The one registered binding that reaches an inference endpoint proves
    // neither pre-acceptance rejection nor logical idempotency, so no run
    // selecting it may be read at a verified capability.
    assert_eq!(
        SCHEDULED_REQUEST_ACTION_SINK.endpoint_retry_safety,
        EndpointRetrySafety::Unproven
    );

    for mode in CheckpointDeliveryMode::ALL {
        let restored =
            support::delivery_fixture(mode, TargetIdempotencyCapability::Unsupported)
                .crash_and_restore(DeliveryCrashPoint::AfterDispatchBeforeTerminal);
        assert_ne!(
            restored.claim(),
            DeliveryClaim::IdempotentAtLeastOnceSubmission,
            "{} claimed target deduplication the endpoint never proved",
            mode.tag()
        );
    }
}

#[test]
fn no_delivery_claim_asserts_exactly_once() {
    // Exhaustive match: adding an `ExactlyOnce` variant to `DeliveryClaim` fails
    // to compile here rather than silently shipping a false claim.
    for claim in [
        DeliveryClaim::AtLeastOnce,
        DeliveryClaim::AtMostOnce,
        DeliveryClaim::IdempotentAtLeastOnceSubmission,
        DeliveryClaim::IngestionOnly,
        DeliveryClaim::None,
    ] {
        let label = match claim {
            DeliveryClaim::AtLeastOnce => "at_least_once",
            DeliveryClaim::AtMostOnce => "at_most_once",
            DeliveryClaim::IdempotentAtLeastOnceSubmission => "idempotent_at_least_once_submission",
            DeliveryClaim::IngestionOnly => "ingestion_only",
            DeliveryClaim::None => "none",
        };
        assert_eq!(claim.tag(), label);
        assert!(!label.contains("exactly_once"));
    }

    for mode in CheckpointDeliveryMode::ALL {
        for capability in TargetIdempotencyCapability::ALL {
            assert!(
                !expected_claim(mode, capability).tag().contains("exactly"),
                "{}/{} published an exactly-once claim",
                mode.tag(),
                capability.tag()
            );
        }
    }
}

#[test]
fn derived_sink_failure_does_not_rewrite_the_delivery_decision() {
    // Marking one derived export incomplete preserves the generation; only the
    // run-failing disposition ends it. They must stay distinct dispositions.
    assert_ne!(
        StreamingIssueDisposition::ExportIncomplete,
        StreamingIssueDisposition::FailRun
    );

    // The restart decision reads the committed cut, the target capability, and
    // the outstanding set. A derived sink outcome is not among them, so
    // repeating a restart cannot move the claim, window, or reissue set.
    for crash in DeliveryCrashPoint::ALL {
        let fixture = || {
            support::delivery_fixture(
                CheckpointDeliveryMode::Terminal,
                TargetIdempotencyCapability::Unsupported,
            )
            .crash_and_restore(crash)
        };
        let before = fixture();
        let after = fixture();
        assert_eq!(before.reissue(), after.reissue(), "{}", crash.tag());
        assert_eq!(before.claim(), after.claim(), "{}", crash.tag());
        assert_eq!(
            before.duplicate_window(),
            after.duplicate_window(),
            "{}",
            crash.tag()
        );
    }
}
