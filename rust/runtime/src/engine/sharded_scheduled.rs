// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-per-core sharded scheduled execution.
//!
//! Each worker thread owns its scheduler, admission state, transport, capture,
//! current-thread runtime, and `LocalSet`. Workers stamp global ordinals through
//! [`IssuanceAuthority`](crate::cellular::IssuanceAuthority), then their record
//! shards are merged.
//!
//! # The two-level `(cell × thread)` partition
//!
//! For cell `c` of `cells` and thread `t` of `W`, the nested partition is
//!
//! ```text
//!     partition_t = ModuloCellPartition::new(c + cells*t, cells*W)
//! ```
//!
//! Since `c + cells*t ≡ c (mod cells)`, each thread's ownership is a subset of
//! its cell's ownership. The `W` residues tile that ownership exactly. The
//! alternative `c*W + t` does not preserve the cell residue when `cells > 1`.
//!
//! # Per-thread workload partition
//!
//! Each thread partitions the cell-local phase budget by `W`:
//! - `requests → owned_positions(cell_requests, t, W)` (its round-robin share);
//! - `concurrency`/`prefill_concurrency → owned_positions(cap, t, W).max(1)`;
//! - `rate → cell_rate / W`.
//!
//! This nested request count equals `owned_positions(global, c + cells*t,
//! cells*W)`, so all threads together stamp a permutation of `0..total`, as
//! required by [`merge_records_in_global_order`](crate::cellular).
//!
//! # Cell-level split under `Global`/`GlobalHop`
//!
//! Under [`DispatchMode::Global`]/[`DispatchMode::GlobalHop`] dispatch, the
//! cell-level split (`owned_positions(global, cell_id, cells)`, applied by the
//! cellular controller's `build_cell_envelope` before this cell's process ever
//! starts) is preserved exactly as [`DispatchMode::Sharded`] mode's is; only
//! the thread-level `/workers` split is replaced by a shared
//! [`GlobalAdmission`](crate::engine::execute::GlobalAdmission) gate scoped to
//! this cell alone. Cells remain separate OS processes with no cross-process
//! shared state (no shared memory, no IPC admission channel), so global
//! exactness is per-cell, not per-run, when `cells > 1`: each cell
//! independently enforces its own `1/cells` share of the authored
//! concurrency/rate, and the union across cells sums back to the authored
//! global value. See the `global_admission_is_scoped_per_cell_not_across_cells`
//! test below for the regression guard.
//!
//! # Ordinal bases
//!
//! The autonomous issuer stamps `phase_base + within*(cells*W) + (c + cells*t)`.
//! Controller children receive `phase_base` values through
//! `AIPERF_CELL_PHASE_ORDINAL_BASES`; single-process runs derive them from phase
//! request budgets with `compute_phase_ordinal_bases`.
//!
//! # Cell-local responsibilities
//!
//! The main thread owns artifacts, side-channel telemetry, shard merging,
//! reporting, and controller transport. Worker threads only execute workloads,
//! dispatch requests, and capture records.

use std::sync::Arc;

use crate::clock::Clock;
use crate::metrics_core::Phase;
use crate::phase_runtime::ScheduledPhaseSidecar;
use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::engine::cell_launcher::owned_positions;
#[cfg(test)]
use crate::engine::execute::GlobalAdmission;
use crate::engine::execute::{
    ScheduledShardOutcome, ShardRecords, ShardedShared, execute_scheduled_shard, metrics_phase,
};
use crate::engine::protocol::{DispatchMode, PhaseSpec};

/// Return a thread's nested dataset and ordinal partition.
pub(crate) fn two_level_partition(
    cell_id: u32,
    cells: u32,
    thread_id: usize,
    workers: u32,
) -> Result<crate::cellular::ModuloCellPartition> {
    let thread_id = u32::try_from(thread_id)
        .map_err(|_| anyhow!("thread id {thread_id} exceeds the cell grid index width"))?;
    let index = cell_id
        .checked_add(
            cells
                .checked_mul(thread_id)
                .ok_or_else(|| anyhow!("cell grid index cells*thread overflow"))?,
        )
        .ok_or_else(|| anyhow!("cell grid index overflow"))?;
    let modulus = cells
        .checked_mul(workers)
        .ok_or_else(|| anyhow!("cell grid modulus cells*workers overflow"))?;
    crate::cellular::ModuloCellPartition::new(index, modulus)
        .map_err(|error| anyhow!("building thread-per-core partition: {error}"))
}

/// Partition a phase across the cell's worker threads.
///
/// `fixed_schedule` is partitioned by conversation in
/// [`NativeDatasetConversationSource`](crate::multiturn).
///
/// The split follows one rule: **work budgets are partitioned, admission caps are
/// gated**.
///
/// `requests` and `sessions` are work budgets, and are ALWAYS sliced into this
/// thread's `1/workers` share in every dispatch mode. `GlobalAdmission` carries
/// no shared total-dispatched-request counter, so leaving them unsliced would
/// make a `Global` run attempt `workers` DUPLICATE copies of the full budget
/// (`workers`x over-dispatch) instead of the authored total. Slicing them costs
/// nothing: `owned_positions` tiles exactly, and because concurrency is gated
/// rather than partitioned, a thread that exhausts its own budget early simply
/// stops asking while any thread still holding budget can occupy the whole
/// shared pool — an uneven budget never becomes idle capacity.
///
/// `concurrency` and `prefill_concurrency` are admission caps, and under
/// [`DispatchMode::Global`]/[`DispatchMode::GlobalHop`] both are left at the
/// cell-local (unsliced-by-thread) value so every thread admits against the
/// shared per-cell `GlobalAdmission` gate. Partitioning a cap is what strands
/// capacity — the share a finished thread still owns cannot be lent to a busy
/// one — and `owned_cap`'s floor of one over-subscribes whenever the cap is
/// below the thread count. Both slice under [`DispatchMode::Sharded`] exactly
/// as before.
///
/// Caps are gated only on the request-rate phase shapes
/// (`Concurrency`/`Poisson`/`Constant`/`Gamma`). `UserCentric` builds its own
/// internal session pool independent of this seam, and `FixedSchedule` has no
/// concurrency or rate concept at all; both are unaffected by dispatch mode.
pub(crate) fn slice_phase_for_thread(
    phase: &PhaseSpec,
    thread_id: usize,
    workers: u32,
    dispatch_mode: DispatchMode,
) -> PhaseSpec {
    let t = thread_id as u32;
    let owned_budget = |value: u64| owned_positions(value, t, workers);
    // Admission caps split by the same round-robin share, floored to 1 so a cap
    // below the thread count over-subscribes to `workers` rather than starving a
    // thread — the same bounded trade the cellular per-cell split accepts.
    let owned_cap = |value: usize| owned_positions(value as u64, t, workers).max(1) as usize;
    let scaled_rate = |rate: f64| rate / workers as f64;
    // Global/GlobalHop admit concurrency+rate from the shared per-cell gate on
    // the request-rate phase shapes, so this thread's LOCAL slice of those two
    // fields is left at the cell-local (unsliced) value instead of `1/workers`.
    let global_admits_concurrency_and_rate = !matches!(dispatch_mode, DispatchMode::Sharded);

    let mut sliced = phase.clone();
    match &mut sliced {
        PhaseSpec::Concurrency {
            common,
            concurrency,
        } => {
            slice_common(
                common,
                &owned_budget,
                &owned_cap,
                global_admits_concurrency_and_rate,
            );
            if !global_admits_concurrency_and_rate {
                *concurrency = owned_cap(*concurrency);
            }
        }
        PhaseSpec::Poisson {
            common,
            rate,
            concurrency,
        }
        | PhaseSpec::Constant {
            common,
            rate,
            concurrency,
        } => {
            slice_common(
                common,
                &owned_budget,
                &owned_cap,
                global_admits_concurrency_and_rate,
            );
            if !global_admits_concurrency_and_rate {
                *rate = scaled_rate(*rate);
                if let Some(cap) = concurrency {
                    *cap = owned_cap(*cap);
                }
            }
        }
        PhaseSpec::Gamma {
            common,
            rate,
            concurrency,
            ..
        } => {
            slice_common(
                common,
                &owned_budget,
                &owned_cap,
                global_admits_concurrency_and_rate,
            );
            if !global_admits_concurrency_and_rate {
                *rate = scaled_rate(*rate);
                if let Some(cap) = concurrency {
                    *cap = owned_cap(*cap);
                }
            }
        }
        // Open-loop churn is aggregate-equivalent across shards, but the
        // per-turn-shape split is timing-dependent. Unaffected by dispatch
        // mode: `UserCentric` owns its own session pool (see the function doc).
        PhaseSpec::UserCentric {
            common,
            rate,
            users,
            concurrency,
        } => {
            slice_common(
                common,
                &owned_budget,
                &owned_cap,
                global_admits_concurrency_and_rate,
            );
            *rate = scaled_rate(*rate);
            *users = owned_cap(*users);
            if let Some(cap) = concurrency {
                *cap = owned_cap(*cap);
            }
        }
        PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. } => {}
    }
    sliced
}

/// Slice a phase's shared `requests`/`sessions` budgets and `prefill_concurrency`
/// cap in place.
fn slice_common(
    common: &mut crate::engine::protocol::PhaseCommonSpec,
    owned_budget: &impl Fn(u64) -> u64,
    owned_cap: &impl Fn(usize) -> usize,
    global_admits_prefill: bool,
) {
    if let Some(requests) = common.requests {
        common.requests = Some(owned_budget(requests));
    }
    // `sessions` is a total work budget exactly like `requests`, and
    // `GlobalAdmission` gates concurrency/rate only — it carries no shared
    // started-session counter. Leaving it unsliced makes every thread run the
    // FULL conversation budget out of its own dataset partition, i.e. `workers`x
    // over-dispatch on a multi-turn session-bounded run.
    if let Some(sessions) = common.sessions {
        common.sessions = Some(owned_budget(sessions));
    }
    // `prefill_concurrency` is an admission cap, not a work budget. Under `Global`/`GlobalHop`
    // the cell builds one shared `GlobalSlotPool` for it, so this thread keeps the full cell-level
    // value and admits against that gate — the same treatment `concurrency` gets, and for the same
    // reason: a per-thread share strands prefill capacity a busy thread cannot borrow, and the
    // `owned_cap` floor over-subscribes when the cap is below the thread count.
    if let Some(prefill) = common.prefill_concurrency
        && !global_admits_prefill
    {
        common.prefill_concurrency = Some(owned_cap(prefill));
    }
}

/// Compute global ordinal bases for a single-process sharded run.
///
/// Controller children must use `AIPERF_CELL_PHASE_ORDINAL_BASES`; cell-local
/// request budgets cannot reconstruct global bases.
pub(crate) fn compute_phase_ordinal_bases(
    phases: &[PhaseSpec],
) -> Result<std::collections::HashMap<Phase, usize>> {
    let mut bases = std::collections::HashMap::new();
    let mut base = 0usize;
    for phase in phases {
        let metric_phase = metrics_phase(phase)?;
        bases.insert(metric_phase, base);
        base += phase.common().requests.unwrap_or(0) as usize;
    }
    Ok(bases)
}

/// Run the scheduled pipeline across worker threads and merge their shards.
///
/// Sidecars stay on the main thread and span worker execution.
pub(crate) async fn run_sharded_scheduled(
    shared: Arc<ShardedShared>,
    profiling_sidecars: Vec<std::rc::Rc<dyn ScheduledPhaseSidecar>>,
    coordinator_clock: std::rc::Rc<dyn Clock>,
) -> Result<ScheduledShardOutcome> {
    let workers = shared.workers as usize;
    if workers == 0 {
        bail!("sharded scheduled execution requires at least one worker");
    }

    // Fail closed on the clock-seam invariant this path silently depends on: each
    // worker thread reconstructs a reactor-local `RealClock` from
    // `shared.real_clock_anchor` (a `!Send` `Rc<dyn Clock>` cannot cross the spawn
    // boundary, and a `SimClock`'s virtual time is driven by ONE reactor's pump so
    // it cannot be shared across `W` independent worker reactors). The engine
    // upstream already forces `workers == 1` for a virtual-clock run, so this path
    // must never see a virtual coordinator clock; assert it rather than leave the
    // worker threads to silently run on live wall time under a virtual run.
    ensure!(
        !coordinator_clock.is_virtual(),
        "sharded scheduled execution reached a virtual coordinator clock with workers={workers}; \
         thread-per-core workers reconstruct reactor-local RealClocks and cannot share virtual \
         time — a virtual-clock run must collapse to workers == 1"
    );

    // Synchronous unbounded sends let worker threads return results without a
    // runtime while the receiver keeps main-thread sidecars progressing.
    let (result_tx, mut result_rx) =
        tokio::sync::mpsc::unbounded_channel::<(usize, Result<ScheduledShardOutcome>)>();
    let mut handles = Vec::with_capacity(workers);
    for worker_id in 0..workers {
        let worker_shared = shared.clone();
        let worker_tx = result_tx.clone();
        let handle = std::thread::Builder::new()
            .name(format!("aiperf-sched-{worker_id}"))
            .spawn(move || {
                let outcome = run_worker_thread(&worker_shared, worker_id);
                let _ = worker_tx.send((worker_id, outcome));
            })
            .map_err(|error| {
                anyhow!("failed to spawn sharded scheduled worker {worker_id}: {error}")
            })?;
        handles.push(handle);
    }
    // Drop the main-thread sender so the channel closes once every worker's clone
    // is gone — a worker that panics before sending then surfaces as a short recv.
    drop(result_tx);

    // Sidecars span the whole worker window, so warmup scrapes may be attributed
    // to profiling when both phases are present.
    for sidecar in &profiling_sidecars {
        sidecar
            .start()
            .await
            .map_err(|error| anyhow!("starting sharded scheduled sidecar: {error:#}"))?;
        sidecar.on_phase_start(shared.start_ns);
    }

    let mut shards: Vec<Option<Result<ScheduledShardOutcome>>> =
        (0..workers).map(|_| None).collect();
    let mut received = 0usize;
    while received < workers {
        match result_rx.recv().await {
            Some((worker_id, outcome)) => {
                if shards[worker_id].replace(outcome).is_some() {
                    bail!("sharded scheduled worker {worker_id} reported its shard twice");
                }
                received += 1;
            }
            None => bail!(
                "a sharded scheduled worker exited before delivering its shard \
                 ({received}/{workers} received) — a worker thread panicked"
            ),
        }
    }

    let end_ns = coordinator_clock.now_ns();
    for sidecar in &profiling_sidecars {
        sidecar.on_phase_end(end_ns);
    }
    for sidecar in &profiling_sidecars {
        sidecar
            .finish()
            .await
            .map_err(|error| anyhow!("finishing sharded scheduled sidecar: {error:#}"))?;
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("a sharded scheduled worker thread panicked"))?;
    }

    merge_shards(shards)
}

/// Run one shard on a worker-local current-thread runtime and `LocalSet`.
fn run_worker_thread(shared: &ShardedShared, worker_id: usize) -> Result<ScheduledShardOutcome> {
    // IO + time only; see the note in turn_execution::run_worker_thread. The
    // signal driver enable_all() would add exists only to reap child-process
    // orphans, which a shard worker never creates.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .with_context(|| format!("creating sharded scheduled worker {worker_id} runtime"))?;
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(execute_scheduled_shard(shared, worker_id)))
}

/// Merge worker shards in global dispatch order.
///
/// The shards already carry globally-unique two-level ordinals (a permutation of
/// `0..total`), so concatenation needs no renumber. Sorting by `request_index`
/// makes the per-record store placement and the deterministic-per-topology row
/// order independent of the (racy) thread completion order, and the input-session
/// list is re-sorted by session id so `inputs.json` is stable across runs.
fn merge_shards(
    shards: Vec<Option<Result<ScheduledShardOutcome>>>,
) -> Result<ScheduledShardOutcome> {
    // Folded shards must merge into an existing folded accumulator rather than
    // the empty retained default.
    let mut shards = shards.into_iter().enumerate();
    let (first_id, first) = shards
        .next()
        .ok_or_else(|| anyhow!("sharded scheduled run produced no shards to merge"))?;
    let mut combined =
        first.ok_or_else(|| anyhow!("sharded scheduled worker {first_id} produced no shard"))??;
    for (worker_id, shard) in shards {
        let shard = shard
            .ok_or_else(|| anyhow!("sharded scheduled worker {worker_id} produced no shard"))??;
        combined.absorb(shard)?;
    }
    // Dense global ordinals make retained row order independent of thread
    // completion. Folded shards retain only grouped errors.
    if let ShardRecords::Retained(records) = &mut combined.records {
        records.sort_by_key(|record| record.ingest.request_index);
    }
    combined
        .input_sessions
        .sort_by(|a, b| a.session_id.cmp(&b.session_id));
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::CellPartition;

    #[test]
    fn two_level_partition_nests_and_tiles() {
        for cells in 1..=4u32 {
            for workers in 1..=4u32 {
                let mut seen = vec![0u32; (cells * workers) as usize];
                for cell_id in 0..cells {
                    for thread_id in 0..workers {
                        let partition =
                            two_level_partition(cell_id, cells, thread_id as usize, workers)
                                .unwrap();
                        assert_eq!(
                            partition.cell_id() % cells,
                            cell_id,
                            "cells={cells} workers={workers}: thread residue must nest in its cell"
                        );
                        assert_eq!(partition.cell_count(), cells * workers);
                        seen[partition.cell_id() as usize] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|&count| count == 1),
                    "cells={cells} workers={workers}: residues must tile exactly once, got {seen:?}"
                );
            }
        }
    }

    #[test]
    fn per_thread_slice_counts_match_global_two_level() {
        for global in [0u64, 1, 5, 10, 11, 37, 100] {
            for cells in 1..=4u32 {
                for workers in 1..=4u32 {
                    let mut total = 0u64;
                    for cell_id in 0..cells {
                        let cell_requests = owned_positions(global, cell_id, cells);
                        for thread_id in 0..workers {
                            let nested = owned_positions(cell_requests, thread_id, workers);
                            let flat = owned_positions(
                                global,
                                cell_id + cells * thread_id,
                                cells * workers,
                            );
                            assert_eq!(
                                nested, flat,
                                "global={global} cells={cells} workers={workers} \
                                 cell={cell_id} thread={thread_id}: nested slice must equal \
                                 the flat two-level share"
                            );
                            total += nested;
                        }
                    }
                    assert_eq!(
                        total, global,
                        "global={global} cells={cells} workers={workers}: shares must tile the total"
                    );
                }
            }
        }
    }

    #[test]
    fn concurrency_phase_slices_requests_and_concurrency_by_w() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
            "prefill_concurrency": 4,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 0, 4, DispatchMode::Sharded);
        assert_eq!(sliced.common().requests, Some(25));
        assert_eq!(sliced.common().prefill_concurrency, Some(1));
        assert_eq!(sliced.concurrency(), Some(2));
        let total: u64 = (0..4)
            .map(|t| {
                slice_phase_for_thread(&phase, t, 4, DispatchMode::Sharded)
                    .common()
                    .requests
                    .unwrap()
            })
            .sum();
        assert_eq!(total, 100);
    }

    /// A `sessions`-bounded multi-turn phase must partition its conversation
    /// budget across threads in EVERY dispatch mode. Leaving it unsliced ran the
    /// full budget per thread (`workers`x over-dispatch).
    #[test]
    fn sessions_budget_slices_across_threads_in_every_dispatch_mode() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "sessions": 12,
            "concurrency": 6,
        }))
        .unwrap();
        for mode in [
            DispatchMode::Sharded,
            DispatchMode::Global,
            DispatchMode::GlobalHop,
        ] {
            let total: u64 = (0..4)
                .map(|t| {
                    slice_phase_for_thread(&phase, t, 4, mode)
                        .common()
                        .sessions
                        .unwrap()
                })
                .sum();
            assert_eq!(total, 12, "sessions must sum to the authored budget");
            assert_eq!(
                slice_phase_for_thread(&phase, 0, 4, mode).common().sessions,
                Some(3)
            );
        }
    }

    #[test]
    fn global_mode_does_not_slice_concurrency_but_still_slices_requests() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 0, 4, DispatchMode::Global);
        // `requests` still slices under Global: GlobalAdmission has no shared
        // total-dispatched-request counter, only concurrency/rate gates (see
        // this function's doc comment) — leaving `requests` unsliced would
        // make every thread attempt the FULL budget, a workers-x over-dispatch.
        assert_eq!(sliced.common().requests, Some(25));
        // `concurrency` is the field GlobalAdmission actually gates, so it
        // stays at the cell-local (unsliced) value.
        assert_eq!(sliced.concurrency(), Some(8));
    }

    #[test]
    fn global_hop_mode_also_does_not_slice_rate_but_still_slices_requests() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "poisson",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 50,
            "rate": 10.0,
            "concurrency": 2,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 3, 4, DispatchMode::GlobalHop);
        assert_eq!(sliced.common().requests, Some(12));
        assert_eq!(sliced.rate(), Some(10.0));
        assert_eq!(sliced.concurrency(), Some(2));
    }

    #[test]
    fn global_mode_leaves_prefill_concurrency_to_the_shared_gate() {
        // `prefill_concurrency` is an admission cap, not a work budget, so it gets the same
        // treatment as `concurrency`: the cell builds one shared gate and this thread keeps the
        // full cell-level value.
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
            "prefill_concurrency": 6,
        }))
        .unwrap();
        for mode in [DispatchMode::Global, DispatchMode::GlobalHop] {
            let sliced = slice_phase_for_thread(&phase, 0, 4, mode);
            assert_eq!(sliced.common().prefill_concurrency, Some(6), "{mode:?}");
        }
    }

    #[test]
    fn sharded_mode_still_slices_prefill_concurrency_per_thread() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
            "prefill_concurrency": 6,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 0, 4, DispatchMode::Sharded);
        assert_eq!(sliced.common().prefill_concurrency, Some(2));
    }

    #[test]
    fn shared_prefill_gate_removes_the_floor_over_subscription() {
        // Under Sharded, `owned_cap` floors every share at one, so a prefill cap of 3 across 4
        // threads admits 4 — more than authored. The shared gate cannot do this: there is one
        // counter, so the sum across threads is the cap itself.
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
            "prefill_concurrency": 3,
        }))
        .unwrap();
        let sharded_total: usize = (0..4)
            .map(|t| {
                slice_phase_for_thread(&phase, t, 4, DispatchMode::Sharded)
                    .common()
                    .prefill_concurrency
                    .unwrap()
            })
            .sum();
        assert_eq!(sharded_total, 4, "sharded over-subscribes the authored 3");
        let global = slice_phase_for_thread(&phase, 0, 4, DispatchMode::Global);
        assert_eq!(global.common().prefill_concurrency, Some(3));
    }

    #[test]
    fn sharded_mode_still_slices_as_before() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 100,
            "concurrency": 8,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 0, 4, DispatchMode::Sharded);
        assert_eq!(sliced.common().requests, Some(25));
        assert_eq!(sliced.concurrency(), Some(2));
    }

    #[test]
    fn rate_phase_splits_rate_by_w_and_floors_caps() {
        let phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "poisson",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 50,
            "rate": 10.0,
            "concurrency": 2,
        }))
        .unwrap();
        let sliced = slice_phase_for_thread(&phase, 3, 4, DispatchMode::Sharded);
        assert_eq!(sliced.rate(), Some(2.5));
        assert_eq!(sliced.concurrency(), Some(1));
    }

    #[test]
    fn phase_ordinal_bases_are_cumulative_prior_requests() {
        let phases: Vec<PhaseSpec> = serde_json::from_value(serde_json::json!([
            {"type": "concurrency", "name": "warmup", "exclude_from_results": true, "requests": 8, "concurrency": 2},
            {"type": "concurrency", "name": "profiling", "exclude_from_results": false, "requests": 200, "concurrency": 4},
        ]))
        .unwrap();
        let bases = compute_phase_ordinal_bases(&phases).unwrap();
        assert_eq!(bases.get(&Phase::Warmup), Some(&0));
        assert_eq!(bases.get(&Phase::Profiling), Some(&8));
    }

    /// Regression guard for `--cells N` combined with `Global`/`GlobalHop`
    /// dispatch: `GlobalAdmission` must be built from the CELL-LOCAL phase
    /// budget (`owned_positions(global, cell_id, cells)`, applied by the
    /// cellular controller's `build_cell_envelope` before this process ever
    /// starts — see `execute.rs::run_execute`'s `global_admission` build site
    /// consuming `request.phases`), never from the raw unsliced global phase
    /// spec. If it were built from the unsliced spec, each of the `cells`
    /// separate OS processes would independently enforce a concurrency cap of
    /// the FULL global value instead of its `1/cells` share, over-subscribing
    /// the true global cap by a factor of `cells`.
    #[test]
    fn global_admission_is_scoped_per_cell_not_across_cells() {
        // A 2-cell, workers=1, Global-mode phase with global concurrency=10 must
        // give each cell's GlobalAdmission a concurrency cap of 5 (its
        // owned_positions share), not 10 (the unsliced global value) — proving
        // the gate composes with cellular tiling instead of bypassing it.
        let global_phase: PhaseSpec = serde_json::from_value(serde_json::json!({
            "type": "concurrency", "name": "profiling", "exclude_from_results": false,
            "requests": 100, "concurrency": 10,
        }))
        .unwrap();
        for cell_id in 0..2u32 {
            let cell_requests = owned_positions(100, cell_id, 2);
            let cell_concurrency = owned_positions(10, cell_id, 2).max(1);
            // Mirror `build_cell_envelope`'s per-cell slice (requests/concurrency),
            // which runs in the cellular controller BEFORE this cell's process
            // starts, and hence before `GlobalAdmission::build` ever sees the
            // phase list. This is the same cell-level narrowing `Sharded` mode's
            // per-cell budget already receives; only the thread-level split
            // differs under `Global`/`GlobalHop`.
            let cell_phase: PhaseSpec = serde_json::from_value(serde_json::json!({
                "type": "concurrency", "name": "profiling", "exclude_from_results": false,
                "requests": cell_requests, "concurrency": cell_concurrency,
            }))
            .unwrap();

            let admission = GlobalAdmission::build(std::slice::from_ref(&cell_phase)).unwrap();
            let phase_key = metrics_phase(&global_phase).unwrap();
            assert_eq!(
                admission
                    .concurrency
                    .get(&phase_key)
                    .unwrap()
                    .current_limit(),
                cell_concurrency as usize,
                "cell {cell_id}: GlobalAdmission must cap at this cell's owned share, not the \
                 unsliced global concurrency"
            );
        }
    }
}
