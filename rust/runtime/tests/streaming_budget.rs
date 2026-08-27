// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use aiperf_runtime::streaming::{
    budget::{BudgetError, BudgetLimits, StreamingResourceBudget},
    identity::{
        ContentDigest, ImmutableObjectIdentity, StableActionId, StableOrderKey, StableSessionKey,
    },
    unit::{
        ActionContentLeaseSet, DatasetActionV1, ExecutableDatasetAction, SessionFragmentLease,
        SessionGraphAction, SessionRequestAction, SessionTerminalAction, SourcePosition,
        UnitProvenance,
    },
};

fn budget(max_items: usize, max_bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes,
    })
    .expect("valid limits")
}

fn executable_action(
    payload: DatasetActionV1,
    content_leases: ActionContentLeaseSet,
) -> Result<ExecutableDatasetAction, BudgetError> {
    executable_action_with_predecessors(payload, Default::default(), content_leases)
}

fn executable_action_with_predecessors(
    payload: DatasetActionV1,
    predecessors: smallvec::SmallVec<[StableActionId; 2]>,
    content_leases: ActionContentLeaseSet,
) -> Result<ExecutableDatasetAction, BudgetError> {
    ExecutableDatasetAction::new(
        StableActionId::from_bytes([1; 32]),
        StableSessionKey::from_bytes([2; 32]),
        predecessors,
        None,
        StableOrderKey::from_bytes([3; 32]),
        SourcePosition::new(4),
        UnitProvenance {
            source_partition: ImmutableObjectIdentity::from_bytes([5; 32]),
            source_position: SourcePosition::new(4),
            format_semantic_digest: ContentDigest::from_bytes([6; 32]),
        },
        payload,
        content_leases,
    )
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_owned_lease_returns_item_and_bytes() {
    let budget = budget(1, 64);
    let lease = budget.acquire(1, 64).await.expect("first lease");
    assert_eq!(budget.snapshot().used_items, 1);
    assert_eq!(budget.snapshot().used_bytes, 64);

    let moved = lease;
    assert_eq!(
        budget.snapshot().used_items,
        1,
        "moving cannot mint permits"
    );
    drop(moved);

    assert_eq!(budget.snapshot().used_items, 0);
    assert_eq!(budget.snapshot().used_bytes, 0);
    assert!(budget.acquire(1, 64).await.is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn close_wakes_waiters_blocked_on_either_semaphore() {
    let item_blocked_budget = budget(1, 2);
    let _items_held = item_blocked_budget.acquire(1, 1).await.expect("held item");
    let mut item_wait = Box::pin(item_blocked_budget.acquire(1, 1));
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut item_wait)
            .await
            .is_err()
    );
    item_blocked_budget.close();
    assert!(matches!(item_wait.await, Err(BudgetError::Closed)));

    let byte_blocked_budget = budget(2, 1);
    let _bytes_held = byte_blocked_budget.acquire(1, 1).await.expect("held byte");
    let mut byte_wait = Box::pin(byte_blocked_budget.acquire(1, 1));
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut byte_wait)
            .await
            .is_err()
    );
    byte_blocked_budget.close();
    assert!(matches!(byte_wait.await, Err(BudgetError::Closed)));
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_after_item_acquisition_rolls_back_the_item_permit() {
    let budget = budget(2, 1);
    let held = budget.acquire(1, 1).await.expect("held byte");
    let mut byte_wait = Box::pin(budget.acquire(1, 1));
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut byte_wait)
            .await
            .is_err()
    );
    drop(byte_wait);
    drop(held);

    tokio::time::timeout(Duration::from_millis(100), budget.acquire(2, 1))
        .await
        .expect("cancelled acquisition must return its partial item permit")
        .expect("budget remains open");
}

#[tokio::test(flavor = "current_thread")]
async fn limits_requests_and_permit_conversions_fail_before_waiting() {
    assert!(matches!(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 0,
            max_bytes: 1,
        }),
        Err(BudgetError::ZeroCapacity)
    ));
    assert!(matches!(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: 0,
        }),
        Err(BudgetError::ZeroCapacity)
    ));

    let budget = budget(2, 8);
    assert!(matches!(
        budget.acquire(3, 1).await,
        Err(BudgetError::RequestExceedsCapacity)
    ));
    assert!(matches!(
        budget.acquire(1, 9).await,
        Err(BudgetError::RequestExceedsCapacity)
    ));

    if usize::BITS > u32::BITS {
        let unrepresentable = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize");
        assert!(matches!(
            StreamingResourceBudget::new(BudgetLimits {
                max_items: unrepresentable,
                max_bytes: 1,
            }),
            Err(BudgetError::PermitCountTooLarge)
        ));
        assert!(matches!(
            budget.acquire(unrepresentable, 1).await,
            Err(BudgetError::PermitCountTooLarge)
        ));
        assert!(matches!(
            budget.acquire(1, usize::MAX).await,
            Err(BudgetError::PermitCountTooLarge)
        ));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn shrink_releases_each_dimension_and_rejects_growth_atomically() {
    let budget = budget(3, 30);
    let mut lease = budget.acquire(3, 30).await.expect("full lease");
    lease.shrink_to(2, 10).expect("valid shrink");
    assert_eq!(lease.charged_items(), 2);
    assert_eq!(lease.charged_bytes(), 10);
    assert_eq!(budget.snapshot().used_items, 2);
    assert_eq!(budget.snapshot().used_bytes, 10);

    assert_eq!(lease.shrink_to(1, 11), Err(BudgetError::CannotGrowLease));
    assert_eq!(lease.charged_items(), 2, "failed shrink is atomic");
    assert_eq!(lease.charged_bytes(), 10, "failed shrink is atomic");

    let other = budget.acquire(1, 20).await.expect("returned capacity");
    assert_eq!(budget.snapshot().used_items, 3);
    assert_eq!(budget.snapshot().used_bytes, 30);
    drop(other);
    lease
        .shrink_to(0, 0)
        .expect("zero retained charge is valid");
    assert_eq!(budget.snapshot().used_items, 0);
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn snapshots_track_independent_high_water_marks() {
    let budget = budget(3, 100);
    let first = budget.acquire(1, 100).await.expect("byte peak");
    drop(first);
    let second = budget.acquire(3, 1).await.expect("item peak");
    let snapshot = budget.snapshot();
    assert_eq!(snapshot.used_items, 3);
    assert_eq!(snapshot.used_bytes, 1);
    assert_eq!(snapshot.high_water_items, 3);
    assert_eq!(snapshot.high_water_bytes, 100);
    drop(second);
}

#[tokio::test(flavor = "current_thread")]
async fn content_charge_releases_only_after_every_owner() {
    let first_budget = budget(1, 32);
    let second_budget = budget(1, 32);
    let first_fragment =
        SessionFragmentLease::try_from(first_budget.acquire(1, 32).await.expect("first fragment"))
            .expect("one-item fragment");
    let second_fragment = SessionFragmentLease::try_from(
        second_budget.acquire(1, 32).await.expect("second fragment"),
    )
    .expect("one-item fragment");
    let first_retained = first_fragment.into_retained();
    let second_retained = second_fragment.into_retained();
    let first_original_owner = first_retained.clone();
    let second_original_owner = second_retained.clone();
    let duplicate = first_retained.clone();
    let mut content_leases = ActionContentLeaseSet::from_retained(first_retained);
    assert!(
        !content_leases.insert(duplicate),
        "Rc-identical lease deduplicates"
    );
    assert!(content_leases.insert(second_retained));
    assert_eq!(content_leases.len(), 2);
    assert_eq!(content_leases.charged_items(), Ok(2));
    assert_eq!(content_leases.charged_bytes(), Ok(64));
    let action = executable_action(
        DatasetActionV1::Request(SessionRequestAction {
            request: vec![0; 12],
        }),
        content_leases,
    )
    .expect("two-fragment action is fully charged");
    let continuation = action.content_leases().retain_for_continuation();
    let raw_capture = action.content_leases().retain_for_continuation();
    let receipt_owner = action.content_leases().retain_for_continuation();

    drop(first_original_owner);
    drop(second_original_owner);
    assert_eq!(first_budget.snapshot().used_bytes, 32);
    assert_eq!(second_budget.snapshot().used_bytes, 32);
    drop(action);
    drop(continuation);
    drop(raw_capture);
    assert_eq!(first_budget.snapshot().used_bytes, 32);
    assert_eq!(second_budget.snapshot().used_bytes, 32);
    drop(receipt_owner);
    assert_eq!(first_budget.snapshot().used_items, 0);
    assert_eq!(first_budget.snapshot().used_bytes, 0);
    assert_eq!(second_budget.snapshot().used_items, 0);
    assert_eq!(second_budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn fragment_creation_rejects_non_unit_item_charges() {
    let budget = budget(2, 8);
    let zero = budget.acquire(0, 0).await.expect("generic zero lease");
    assert!(SessionFragmentLease::try_from(zero).is_err());

    let two = budget.acquire(2, 8).await.expect("generic two-item lease");
    assert!(SessionFragmentLease::try_from(two).is_err());
    assert_eq!(budget.snapshot().used_items, 0);
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn live_fragment_cannot_shrink_away_its_item_charge() {
    let budget = budget(1, 8);
    let mut fragment =
        SessionFragmentLease::try_from(budget.acquire(1, 8).await.expect("fragment-sized lease"))
            .expect("one-item fragment");
    fragment.shrink_bytes_to(4).expect("byte-only shrink");
    assert_eq!(fragment.charged_items(), 1);
    assert_eq!(fragment.charged_bytes(), 4);
    assert_eq!(budget.snapshot().used_items, 1);
    assert_eq!(budget.snapshot().used_bytes, 4);
}

#[tokio::test(flavor = "current_thread")]
async fn every_payload_variant_rejects_undercharged_content() {
    let payloads = [
        DatasetActionV1::Request(SessionRequestAction {
            request: vec![0; 2],
        }),
        DatasetActionV1::GraphNode(SessionGraphAction {
            node_key: "n".into(),
            request: vec![0; 1],
        }),
        DatasetActionV1::SessionTerminal(SessionTerminalAction {
            reason: "é".into()
        }),
    ];

    for payload in payloads {
        let budget = budget(1, 1);
        let fragment =
            SessionFragmentLease::try_from(budget.acquire(1, 1).await.expect("one-byte fragment"))
                .expect("one-item fragment");
        let error = executable_action(
            payload,
            ActionContentLeaseSet::from_retained(fragment.into_retained()),
        )
        .expect_err("two payload bytes cannot use one retained byte");
        assert!(matches!(
            error,
            BudgetError::ActionPayloadUndercharged {
                required_bytes,
                retained_bytes: 1,
            } if required_bytes >= 2
        ));
        assert_eq!(budget.snapshot().used_items, 0, "rejection drops leases");
        assert_eq!(budget.snapshot().used_bytes, 0, "rejection drops leases");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn reserved_request_capacity_requires_lease_coverage() {
    let budget = budget(1, 1);
    let fragment =
        SessionFragmentLease::try_from(budget.acquire(1, 1).await.expect("logical request byte"))
            .expect("one-item fragment");
    let mut request = Vec::with_capacity(8);
    request.push(1);

    executable_action(
        DatasetActionV1::Request(SessionRequestAction { request }),
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect_err("reserved request capacity must be charged");
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn reserved_graph_string_and_request_capacity_require_lease_coverage() {
    let budget = budget(1, 2);
    let fragment =
        SessionFragmentLease::try_from(budget.acquire(1, 2).await.expect("logical graph bytes"))
            .expect("one-item fragment");
    let mut node_key = String::with_capacity(4);
    node_key.push('n');
    let mut request = Vec::with_capacity(4);
    request.push(1);

    executable_action(
        DatasetActionV1::GraphNode(SessionGraphAction { node_key, request }),
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect_err("reserved graph capacities must be charged");
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn reserved_terminal_string_capacity_requires_lease_coverage() {
    let budget = budget(1, 1);
    let fragment =
        SessionFragmentLease::try_from(budget.acquire(1, 1).await.expect("logical terminal byte"))
            .expect("one-item fragment");
    let mut reason = String::with_capacity(8);
    reason.push('x');

    executable_action(
        DatasetActionV1::SessionTerminal(SessionTerminalAction { reason }),
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect_err("reserved terminal capacity must be charged");
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn spilled_predecessor_storage_requires_lease_coverage() {
    let budget = budget(1, 1);
    let fragment =
        SessionFragmentLease::try_from(budget.acquire(1, 1).await.expect("minimal action lease"))
            .expect("one-item fragment");
    let predecessors = smallvec::smallvec![
        StableActionId::from_bytes([10; 32]),
        StableActionId::from_bytes([11; 32]),
        StableActionId::from_bytes([12; 32]),
    ];

    executable_action_with_predecessors(
        DatasetActionV1::Request(SessionRequestAction {
            request: Vec::new(),
        }),
        predecessors,
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect_err("spilled predecessor allocation must be charged");
    assert_eq!(budget.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn spilled_lease_set_storage_requires_lease_coverage() {
    let budget = budget(3, 3);
    let mut leases = None;
    for _ in 0..3 {
        let fragment =
            SessionFragmentLease::try_from(budget.acquire(1, 1).await.expect("distinct lease"))
                .expect("one-item fragment");
        let retained = fragment.into_retained();
        match &mut leases {
            Some(existing) => assert!(ActionContentLeaseSet::insert(existing, retained)),
            None => leases = Some(ActionContentLeaseSet::from_retained(retained)),
        }
    }

    executable_action(
        DatasetActionV1::Request(SessionRequestAction {
            request: Vec::new(),
        }),
        leases.expect("three-lease set"),
    )
    .expect_err("spilled lease-set allocation must be charged");
    assert_eq!(budget.snapshot().used_bytes, 0);
}
