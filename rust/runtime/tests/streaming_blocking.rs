// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "support/streaming_checkpoint.rs"]
mod support;

use std::{
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use aiperf_runtime::{
    clock::{Clock, SimClock},
    streaming::{
        blocking::{
            BLOCKING_CHECKPOINT_SCHEMA_ID, BLOCKING_CHECKPOINT_SCHEMA_VERSION,
            BlockingCheckpointState, BlockingWorkBudget, BlockingWorkClass, BlockingWorkError,
            StreamingBlockingExecutor,
        },
        budget::{BudgetLimits, StreamingResourceBudget},
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointError, CommittedParticipantState,
            PreparedParticipantState, StreamingCheckpointParticipant,
        },
        unit::StateBudgetFailureCode,
    },
};
use bytes::Bytes;

#[derive(Debug)]
struct CustomOutput {
    logical_len: usize,
    retained: Vec<u8>,
}

fn held_work(
    started: tokio::sync::oneshot::Sender<()>,
    release: Arc<AtomicBool>,
) -> impl FnOnce(
    aiperf_runtime::streaming::blocking::BlockingCancellation,
) -> Result<Vec<u8>, BlockingWorkError> {
    move |cancellation| {
        let _ = started.send(());
        while !release.load(Ordering::Acquire) {
            if cancellation.is_cancelled() {
                return Err(BlockingWorkError::Cancelled);
            }
            std::thread::yield_now();
        }
        Ok(vec![0_u8; 1])
    }
}

async fn committed_blocking_state(
    owner: &StreamingBlockingExecutor,
    state: BlockingCheckpointState,
) -> CommittedParticipantState {
    let bytes = Bytes::from(state.encode().to_vec());
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: bytes.len(),
    })
    .expect("state budget");
    let lease = budget.acquire(1, bytes.len()).await.expect("state lease");
    let payload = BudgetedCheckpointBytes::new(bytes.clone(), lease).expect("state payload");
    let prepared = PreparedParticipantState::new(
        support::run_id(1),
        owner.participant_id(),
        BLOCKING_CHECKPOINT_SCHEMA_ID,
        BLOCKING_CHECKPOINT_SCHEMA_VERSION,
        support::cut_at(state.completed_horizon().get().get()),
        1,
        payload,
    )
    .expect("prepared state");
    let descriptor = prepared.descriptor().clone();
    drop(prepared);

    let lease = budget
        .acquire(1, bytes.len())
        .await
        .expect("committed state lease");
    let payload = BudgetedCheckpointBytes::new(bytes, lease).expect("committed state payload");
    CommittedParticipantState::new(support::run_id(1), descriptor, payload)
        .expect("verified committed state")
}

#[tokio::test(flavor = "current_thread")]
async fn full_authored_output_reservation_lives_with_arbitrary_typed_output() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 16).expect("executor");
    let output = executor
        .run(
            BlockingWorkClass::Decode,
            BlockingWorkBudget {
                input_bytes: 1,
                output_bytes: 16,
            },
            |_cancellation| {
                Ok(CustomOutput {
                    logical_len: 1,
                    retained: Vec::with_capacity(4),
                })
            },
        )
        .await
        .expect("output");

    assert_eq!(output.logical_len, 1);
    assert_eq!(output.retained.capacity(), 4);
    assert_eq!(executor.snapshot().output_bytes, 16);
    drop(output);
    assert_eq!(executor.snapshot().output_bytes, 0);
    executor.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn accepted_job_capacity_blocks_before_spawn_blocking_enqueue() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let release = Arc::new(AtomicBool::new(false));
    let (first_started_tx, first_started_rx) = tokio::sync::oneshot::channel();
    let mut first = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        held_work(first_started_tx, Arc::clone(&release)),
    ));
    tokio::select! {
        result = &mut first => panic!("held job completed early: {result:?}"),
        result = first_started_rx => result.expect("first job started"),
    }

    let second_started = Arc::new(AtomicBool::new(false));
    let second_started_in_work = Arc::clone(&second_started);
    let mut second = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        move |_cancellation| {
            second_started_in_work.store(true, Ordering::Release);
            Ok(vec![1_u8])
        },
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut second)
            .await
            .is_err()
    );
    assert!(!second_started.load(Ordering::Acquire));
    assert_eq!(executor.snapshot().accepted_jobs, 1);

    release.store(true, Ordering::Release);
    drop(first.await.expect("first job"));
    drop(
        tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .expect("second job admitted after capacity returns")
            .expect("second job"),
    );
    executor.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_run_is_reaped_and_capacity_one_admits_the_next_job() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let release = Arc::new(AtomicBool::new(false));
    let release_in_work = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    let mut abandoned = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        move |_cancellation| {
            let _ = started_tx.send(());
            while !release_in_work.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            let _ = finished_tx.send(());
            Ok(vec![0_u8])
        },
    ));
    tokio::select! {
        result = &mut abandoned => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("first job started"),
    }
    drop(abandoned);
    release.store(true, Ordering::Release);
    finished_rx.await.expect("abandoned worker completed");

    let output = tokio::time::timeout(
        Duration::from_secs(1),
        executor.run(
            BlockingWorkClass::Decode,
            BlockingWorkBudget {
                input_bytes: 1,
                output_bytes: 1,
            },
            |_cancellation| Ok(vec![1_u8]),
        ),
    )
    .await
    .expect("completed abandoned work must release accepted capacity")
    .expect("second job");
    drop(output);
    executor.cancel_and_join().await.expect("clean shutdown");
    assert_eq!(executor.snapshot().accepted_jobs, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_the_last_owner_cancels_an_abandoned_accepted_job() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (cancelled_tx, cancelled_rx) = tokio::sync::oneshot::channel();
    let mut abandoned = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        move |cancellation| {
            let _ = started_tx.send(());
            while !cancellation.is_cancelled() {
                std::thread::yield_now();
            }
            let _ = cancelled_tx.send(());
            Err::<Vec<u8>, _>(BlockingWorkError::Cancelled)
        },
    ));
    tokio::select! {
        result = &mut abandoned => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("job started"),
    }

    drop(abandoned);
    drop(executor);
    tokio::time::timeout(Duration::from_secs(1), cancelled_rx)
        .await
        .expect("last owner drop must cancel accepted work")
        .expect("worker observed cancellation");
}

#[tokio::test(flavor = "current_thread")]
async fn cancel_and_join_waits_for_cooperative_worker_exit() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let release_after_cancel = Arc::new(AtomicBool::new(false));
    let release_in_work = Arc::clone(&release_after_cancel);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (cancel_seen_tx, cancel_seen_rx) = tokio::sync::oneshot::channel();
    let mut run = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        move |cancellation| {
            let _ = started_tx.send(());
            while !cancellation.is_cancelled() {
                std::thread::yield_now();
            }
            let _ = cancel_seen_tx.send(());
            while !release_in_work.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Err::<Vec<u8>, _>(BlockingWorkError::Cancelled)
        },
    ));
    tokio::select! {
        result = &mut run => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("job started"),
    }

    let mut shutdown = Box::pin(executor.cancel_and_join());
    tokio::select! {
        result = &mut shutdown => panic!("shutdown returned before cancellation was observed: {result:?}"),
        result = cancel_seen_rx => result.expect("worker observed cancellation"),
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut shutdown)
            .await
            .is_err(),
        "shutdown must join, not merely signal"
    );

    release_after_cancel.store(true, Ordering::Release);
    shutdown.await.expect("joined cancelled worker");
    assert!(matches!(run.await, Err(BlockingWorkError::Cancelled)));
    assert_eq!(executor.snapshot().accepted_jobs, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_cancel_and_join_does_not_lose_later_join_authority() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let release_after_cancel = Arc::new(AtomicBool::new(false));
    let release_in_work = Arc::clone(&release_after_cancel);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (cancel_seen_tx, cancel_seen_rx) = tokio::sync::oneshot::channel();
    let mut run = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        move |cancellation| {
            let _ = started_tx.send(());
            while !cancellation.is_cancelled() {
                std::thread::yield_now();
            }
            let _ = cancel_seen_tx.send(());
            while !release_in_work.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Err::<Vec<u8>, _>(BlockingWorkError::Cancelled)
        },
    ));
    tokio::select! {
        result = &mut run => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("job started"),
    }

    let mut abandoned_shutdown = Box::pin(executor.cancel_and_join());
    tokio::select! {
        result = &mut abandoned_shutdown => panic!("shutdown returned before cancellation was observed: {result:?}"),
        result = cancel_seen_rx => result.expect("worker observed cancellation"),
    }
    drop(abandoned_shutdown);
    release_after_cancel.store(true, Ordering::Release);

    tokio::time::timeout(Duration::from_secs(1), executor.cancel_and_join())
        .await
        .expect("a later shutdown retains join authority")
        .expect("joined cancelled worker");
    assert!(matches!(run.await, Err(BlockingWorkError::Cancelled)));
    assert_eq!(executor.snapshot().accepted_jobs, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_refuses_inflight_work_and_advances_only_its_prepared_view() {
    let mut owner =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let worker = owner.clone();
    let release = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let mut run = Box::pin(worker.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        held_work(started_tx, Arc::clone(&release)),
    ));
    tokio::select! {
        result = &mut run => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("job started"),
    }

    let barrier = support::barrier_at(7);
    assert!(matches!(
        owner.checkpoint_view(&barrier).await,
        Err(CheckpointError::CutBlockedByInflight { job_count: 1, .. })
    ));
    assert_eq!(owner.snapshot().completed_horizon, None);

    release.store(true, Ordering::Release);
    drop(run.await.expect("completed held job"));
    let prepared = owner
        .checkpoint_view(&barrier)
        .await
        .expect("quiescent checkpoint view");
    let checkpoint = BlockingCheckpointState::decode(prepared.payload_bytes())
        .expect("blocking checkpoint payload");
    assert_eq!(checkpoint.inflight_job_count(), 0);
    assert_eq!(checkpoint.completed_horizon(), &barrier.cut.decoded);
    assert_eq!(
        owner.snapshot().completed_horizon,
        None,
        "preparation is non-destructive until commit"
    );
    drop(prepared);
    owner.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn foreign_barrier_is_rejected_before_blocking_owner_fences() {
    let mut owner =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let before = owner.snapshot();

    assert!(matches!(
        owner.checkpoint_view(&support::barrier_for_run(2, 7)).await,
        Err(CheckpointError::ObjectVerification)
    ));
    assert_eq!(owner.snapshot(), before);
    assert!(owner.snapshot().is_accepting);

    owner.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn restore_rejects_any_claimed_inflight_closure() {
    let mut owner =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let state = BlockingCheckpointState::new(support::cut_at(3).decoded, 1);
    let committed = committed_blocking_state(&owner, state).await;

    assert_eq!(
        owner.initialize(Some(committed)).await,
        Err(CheckpointError::ObjectVerification)
    );
    assert_eq!(owner.snapshot().completed_horizon, None);
    owner.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn restored_horizon_refuses_a_lower_checkpoint_barrier() {
    let mut owner =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let committed = committed_blocking_state(
        &owner,
        BlockingCheckpointState::new(support::cut_at(7).decoded, 0),
    )
    .await;
    owner
        .initialize(Some(committed))
        .await
        .expect("restore completed horizon");

    assert!(matches!(
        owner.checkpoint_view(&support::barrier_at(3)).await,
        Err(CheckpointError::DecodeHorizonRegression {
            completed,
            proposed,
            ..
        }) if completed == support::cut_at(7).decoded && proposed == support::cut_at(3).decoded
    ));
    assert_eq!(
        owner.snapshot().completed_horizon,
        Some(support::cut_at(7).decoded)
    );
    owner.cancel_and_join().await.expect("clean shutdown");
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_clone_contention_is_immediate_and_shutdown_is_terminal() {
    let mut owner =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let mut duplicate = owner.clone();
    let barrier = support::barrier_at(4);
    let prepared = owner
        .checkpoint_view(&barrier)
        .await
        .expect("first prepared view");

    let duplicate_result = tokio::time::timeout(
        Duration::from_millis(20),
        duplicate.checkpoint_view(&barrier),
    )
    .await
    .expect("duplicate view must refuse rather than wait");
    assert!(matches!(
        duplicate_result,
        Err(CheckpointError::StateBudget {
            code: StateBudgetFailureCode::ItemCapacity,
            ..
        })
    ));

    owner.cancel_and_join().await.expect("shutdown");
    drop(prepared);
    assert!(matches!(
        duplicate.checkpoint_view(&barrier).await,
        Err(CheckpointError::ParticipantUnavailable { .. })
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn blocking_work_does_not_stall_sim_clock_progress() {
    let executor =
        StreamingBlockingExecutor::for_test(support::run_id(1), 1, 8, 8).expect("executor");
    let release = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let mut run = Box::pin(executor.run(
        BlockingWorkClass::Decode,
        BlockingWorkBudget {
            input_bytes: 1,
            output_bytes: 1,
        },
        held_work(started_tx, Arc::clone(&release)),
    ));
    tokio::select! {
        result = &mut run => panic!("held job completed early: {result:?}"),
        result = started_rx => result.expect("job started"),
    }

    let clock = Rc::new(SimClock::new());
    let sleep = Rc::clone(&clock).sleep(100);
    tokio::pin!(sleep);
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut sleep)
            .await
            .is_err()
    );
    clock.advance_to(100);
    sleep.await;
    assert_eq!(clock.now_ns(), 100);
    assert_eq!(executor.snapshot().accepted_jobs, 1);

    release.store(true, Ordering::Release);
    drop(run.await.expect("held job"));
    executor.cancel_and_join().await.expect("clean shutdown");
}
