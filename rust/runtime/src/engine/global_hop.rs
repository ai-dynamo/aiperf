// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-dispatcher (`GlobalHop`) execution for `workers > 1`.
//!
//! Unlike [`DispatchMode::Sharded`]/[`DispatchMode::Global`] — which spawn `W`
//! independent per-thread scheduling loops (each running its own
//! [`execute_scheduled_shard`](super::execute::execute_scheduled_shard) over a
//! `1/W` partition) — `GlobalHop` runs ONE coordinator-owned scheduling loop.
//! That single loop issues every turn in exact global order and hops each
//! individual [`RequestExecutor::execute_measured`](crate::transport::core::RequestExecutor)
//! call round-robin to a worker OS thread over a bounded mpsc command queue,
//! awaiting a oneshot reply — the cross-thread
//! [`ThreadPerCoreExecutor`](super::turn_execution) hop that
//! [`build_native`](super::turn_execution) constructs for `workers > 1`.
//!
//! This reproduces exact request-to-thread assignment order (turn `i` -> worker
//! `i % W`) and, as a consequence, exact arrival-pattern statistics for jittered
//! rate phases — the gap that `Global` mode's shared-admission-only fix cannot
//! close, because its `W` loops race independently.
//!
//! # Why `GlobalHop` does NOT consume [`GlobalAdmission`](super::execute::GlobalAdmission)
//!
//! [`DispatchMode::Global`] needs `GlobalAdmission` (shared cross-thread
//! concurrency/rate gates) precisely because it has `W` independent scheduling
//! loops that would otherwise each enforce a `1/W`-sliced local cap; the shared
//! gate makes their union globally exact. `GlobalHop` has NO such race: there is
//! ONE coordinator loop, driven with the FULL (un-thread-sliced) cell-level
//! concurrency cap and rate through the ordinary local
//! [`native_scheduled_resources`](super::execute::native_scheduled_resources)
//! `SlotPool` and the local per-phase rate grid. One loop holding the full cap
//! IS the global cap, so aggregate concurrency and rate are exact WITHOUT any
//! cross-thread admission gate. `shared.global_admission` is therefore left
//! `None` for `GlobalHop` (see the build site in
//! [`run_execute`](super::execute)); the coordinator's exactness comes from
//! "one loop, one full-cap local `SlotPool`", not from `GlobalAdmission`.
//!
//! [`DispatchMode::Sharded`]: crate::engine::protocol::DispatchMode::Sharded
//! [`DispatchMode::Global`]: crate::engine::protocol::DispatchMode::Global

use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};

use crate::clock::Clock;
use crate::engine::protocol::PhaseSpec;
use crate::engine::turn_execution::{ExecutionBackendConfig, PreparedEndpointTableFactory};
use crate::phase_runtime::ScheduledPhaseSidecar;

use super::execute::{
    ScheduledShardOutcome, ShardRecords, ShardedShared, execute_scheduled_pipeline,
};

/// Run the whole cell's schedule from ONE coordinator-owned dispatch loop,
/// hopping each issued turn round-robin to a worker thread through the
/// thread-per-core hop executor.
///
/// Signature-compatible with
/// [`run_sharded_scheduled`](super::sharded_scheduled::run_sharded_scheduled)
/// so the caller in `execute.rs` selects between the two purely by
/// `dispatch_mode`.
pub(crate) async fn run_global_hop(
    shared: Arc<ShardedShared>,
    profiling_sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    coordinator_clock: Rc<dyn Clock>,
) -> Result<ScheduledShardOutcome> {
    run_single_coordinator(shared, profiling_sidecars, coordinator_clock).await
}

/// The single-coordinator pipeline itself, shared by [`run_global_hop`] and
/// [`run_global_push`](super::global_push::run_global_push).
///
/// Both modes make the same structural promise — ONE scheduling loop holding the
/// full cell-level cap, `W` worker threads under it — and differ only in how a
/// dispatched turn gets to a worker and back, which is selected per phase from
/// `shared.dispatch_mode` when the phase plan is built. Keeping one body means a
/// change to partitioning, sidecar spanning, or merge ordering cannot drift
/// between them.
pub(crate) async fn run_single_coordinator(
    shared: Arc<ShardedShared>,
    profiling_sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    coordinator_clock: Rc<dyn Clock>,
) -> Result<ScheduledShardOutcome> {
    let workers = shared.workers as usize;
    ensure!(
        workers >= 1,
        "single-coordinator execution requires at least one worker"
    );

    // The single coordinator scheduling loop runs on the caller's reactor, so it
    // uses the injected `coordinator_clock` directly — the reactor-local clock
    // the caller already built and drives. Honoring the injected clock (rather
    // than reconstructing a `RealClock` from the anchor) keeps the coordinator on
    // whichever timeline the run selected: a `SimClock` for a virtual-clock
    // `workers == 1` global-hop run drives the same object the pump advances,
    // instead of a live `RealClock` that would silently ignore virtual time. The
    // per-turn hop backend still hands each worker thread its own reactor-local
    // `RealClock` from `shared.real_clock_anchor` (a `!Send` clock cannot cross
    // the thread boundary); only this coordinator loop follows the injected clock.
    let clock = coordinator_clock.clone();

    // Full cell partition (thread_id 0 of 1 worker): the single coordinator
    // pipeline owns the whole cell's ordinal share, tiling this cell's slice of
    // 0..total, identical to a `workers == 1` sharded run.
    let partition =
        crate::engine::sharded_scheduled::two_level_partition(shared.cell_id, shared.cells, 0, 1)?;
    // Un-thread-sliced (full cell-level) phase budgets/caps/rate: `workers == 1`
    // makes `slice_phase_for_thread` a no-op split, leaving the full cap and rate
    // for the single coordinator loop to enforce locally.
    let sliced_phases: Vec<PhaseSpec> = shared
        .phases
        .iter()
        .map(|phase| {
            crate::engine::sharded_scheduled::slice_phase_for_thread(
                phase,
                0,
                1,
                shared.dispatch_mode,
            )
        })
        .collect();

    // The hop backend: `workers > 1` builds the cross-thread `ThreadPerCoreExecutor`
    // (bounded mpsc + oneshot per turn); `workers == 1` degrades to a co-located
    // sink (a lone-worker GlobalHop is just a single-thread run, still correct).
    let prepared_endpoints: Arc<dyn PreparedEndpointTableFactory> = shared.table_factory.clone();
    let execution_backend = shared.transport_factory.build(ExecutionBackendConfig {
        workers,
        coordinator_clock: clock.clone(),
        real_clock_anchor: shared.real_clock_anchor,
        base_urls: shared.endpoint_urls.clone(),
        model: shared.primary_model.clone(),
        transport: shared.transport_config.clone(),
        raw_enabled: shared.raw_enabled,
        prepared_endpoints: Some(prepared_endpoints),
        hop_routing: shared.hop_routing,
        virtual_worker_width: None,
    })?;

    // Sidecars span the whole coordinator window, matching `run_sharded_scheduled`.
    for sidecar in &profiling_sidecars {
        sidecar
            .start()
            .await
            .map_err(|error| anyhow!("starting single-coordinator sidecar: {error:#}"))?;
        sidecar.on_phase_start(shared.start_ns);
    }

    // The single coordinator scheduling loop, dispatching every turn in exact
    // global order through the hop executor (the executor's round-robin over the
    // worker threads gives exact request-to-thread assignment).
    let outcome = execute_scheduled_pipeline(
        &shared,
        0,
        partition,
        sliced_phases,
        clock,
        execution_backend,
    )
    .await;

    let end_ns = coordinator_clock.now_ns();
    for sidecar in &profiling_sidecars {
        sidecar.on_phase_end(end_ns);
    }
    for sidecar in &profiling_sidecars {
        sidecar
            .finish()
            .await
            .map_err(|error| anyhow!("finishing single-coordinator sidecar: {error:#}"))?;
    }

    let mut outcome = outcome?;
    // The single-coordinator pipeline already carries this cell's globally-unique
    // two-level ordinals; apply the same deterministic finalization ordering
    // `merge_shards` applies for the multi-shard paths so retained record order is
    // independent of completion timing. inputs.json is no longer ordered here: it is
    // projected from the resident dataset up front, not harvested from replies.
    if let ShardRecords::Retained(records) = &mut outcome.records {
        records.sort_by_key(|record| record.ingest.request_index);
    }
    Ok(outcome)
}
