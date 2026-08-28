// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delivery restart policy across the committed-cut, crash-point, and
//! target-idempotency matrix.

#![cfg(feature = "streaming")]

use aiperf_runtime::streaming::{
    identity::ContentDigest,
    results::{
        CheckpointDeliveryMode, DeliveryClaim, DeliveryCrashPoint, DeliveryRestartError,
        DuplicateWindow, ResultProjectionId, TargetIdempotencyCapability,
    },
};

#[path = "support/streaming_checkpoint.rs"]
mod support;

/// Claim the policy owes each mode and capability pairing.
///
/// Restated independently of the production derivation: an admitted cut is the
/// only one that can name an uncertain action without proving its outcome, so
/// without target deduplication it suppresses and becomes at-most-once, while a
/// decoded cut has no suppression handle and stays at-least-once.
fn expected_claim(
    mode: CheckpointDeliveryMode,
    capability: TargetIdempotencyCapability,
) -> DeliveryClaim {
    let deduplicates = capability == TargetIdempotencyCapability::VerifiedLogicalActionKey;
    match mode {
        CheckpointDeliveryMode::None => DeliveryClaim::None,
        CheckpointDeliveryMode::Acquired => DeliveryClaim::IngestionOnly,
        CheckpointDeliveryMode::Terminal | CheckpointDeliveryMode::Decoded => {
            if deduplicates {
                DeliveryClaim::IdempotentAtLeastOnceSubmission
            } else {
                DeliveryClaim::AtLeastOnce
            }
        }
        CheckpointDeliveryMode::Admitted => {
            if deduplicates {
                DeliveryClaim::IdempotentAtLeastOnceSubmission
            } else {
                DeliveryClaim::AtMostOnce
            }
        }
    }
}

/// Duplication and loss the policy leaves open for one restart.
fn expected_window(
    mode: CheckpointDeliveryMode,
    crash: DeliveryCrashPoint,
    capability: TargetIdempotencyCapability,
) -> DuplicateWindow {
    let deduplicates = capability == TargetIdempotencyCapability::VerifiedLogicalActionKey;
    let has_uncertain = matches!(
        crash,
        DeliveryCrashPoint::AfterDispatchBeforeTerminal
            | DeliveryCrashPoint::AfterTerminalBeforeCommit
    );
    let reissues = match mode {
        CheckpointDeliveryMode::None => false,
        CheckpointDeliveryMode::Admitted => deduplicates,
        CheckpointDeliveryMode::Terminal
        | CheckpointDeliveryMode::Decoded
        | CheckpointDeliveryMode::Acquired => true,
    };
    DuplicateWindow {
        may_duplicate_target_effect: has_uncertain && reissues && !deduplicates,
        may_lose_target_effect: has_uncertain && !reissues,
    }
}

#[test]
fn delivery_mode_crash_matrix_has_stable_logical_membership() {
    for mode in CheckpointDeliveryMode::ALL {
        for crash in DeliveryCrashPoint::ALL {
            for capability in TargetIdempotencyCapability::ALL {
                let restored = support::delivery_fixture(mode, capability).crash_and_restore(crash);
                assert!(
                    restored.logical_membership_is_unique(),
                    "{}/{}/{} re-emitted a logical action twice",
                    mode.tag(),
                    crash.tag(),
                    capability.tag()
                );
                assert_eq!(
                    restored.claim(),
                    expected_claim(mode, capability),
                    "{}/{}/{} published the wrong delivery claim",
                    mode.tag(),
                    crash.tag(),
                    capability.tag()
                );
                assert_eq!(
                    restored.duplicate_window(),
                    expected_window(mode, crash, capability),
                    "{}/{}/{} left the wrong duplicate window",
                    mode.tag(),
                    crash.tag(),
                    capability.tag()
                );
            }
        }
    }
}

#[test]
fn undurable_mode_reissues_nothing_while_a_durable_cut_replays_its_suffix() {
    let crash = DeliveryCrashPoint::AfterDispatchBeforeTerminal;

    let undurable = support::delivery_fixture(
        CheckpointDeliveryMode::None,
        TargetIdempotencyCapability::Unsupported,
    )
    .crash_and_restore(crash);
    assert!(
        undurable.reissue().is_empty(),
        "a cut recording nothing has no authoritative record to re-emit from"
    );

    let durable = support::delivery_fixture(
        CheckpointDeliveryMode::Terminal,
        TargetIdempotencyCapability::Unsupported,
    )
    .crash_and_restore(crash);
    assert_eq!(
        durable.reissue().len(),
        3,
        "a terminal cut re-emits its undispatched suffix and the uncertain action"
    );

    let suppressing = support::delivery_fixture(
        CheckpointDeliveryMode::Admitted,
        TargetIdempotencyCapability::Unsupported,
    )
    .crash_and_restore(crash);
    assert_eq!(
        suppressing.reissue().len(),
        2,
        "an admitted cut suppresses the uncertain action against a non-idempotent target"
    );
}

#[test]
fn restart_rejects_changed_topology_projection_or_membership_scheme() {
    let fixture = support::delivery_fixture(
        CheckpointDeliveryMode::Terminal,
        TargetIdempotencyCapability::VerifiedLogicalActionKey,
    );

    assert_eq!(
        fixture.restart_with_binding(|binding| {
            binding.topology_digest = ContentDigest::from_bytes([0x01; 32]);
        }),
        Err(DeliveryRestartError::TopologyChanged)
    );
    assert_eq!(
        fixture.restart_with_binding(|binding| {
            binding.projection =
                ResultProjectionId::new("aiperf.records.raw").expect("nonempty projection");
        }),
        Err(DeliveryRestartError::ProjectionChanged)
    );
    assert_eq!(
        fixture.restart_with_binding(|binding| {
            binding.membership_scheme_digest = ContentDigest::from_bytes([0x02; 32]);
        }),
        Err(DeliveryRestartError::MembershipSchemeChanged)
    );
    assert!(
        fixture.restart_with_binding(|_| {}).is_ok(),
        "an identical binding resumes"
    );
}
