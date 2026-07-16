// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-per-core sharded scheduled execution (design P3, "a thread is a
//! sub-cell").
//!
//! When a scheduled run requests `runtime.workers > 1`, this module runs the
//! **entire** scheduled pipeline — arrival pacing, `SlotPool` admission,
//! per-request dispatch, transport, and record capture — independently on `W`
//! self-contained OS threads over a `1/W` partition of the run, then merges the
//! per-thread record shards. Each thread is a *sub-cell*: its own
//! `current_thread` runtime + `LocalSet`, its own `workers == 1` transport sink
//! (so the scheduler and transport are co-located and the old per-request
//! `mpsc`/`oneshot`/`Notify` hop to a transport-worker pool is *gone*), its own
//! [`RunCapture`](crate::runner_protocol::execute), and its own injected
//! [`IssuanceAuthority`](crate::cellular::IssuanceAuthority) stamping **global**
//! two-level ordinals. The `workers == 1` path stays byte-unchanged on the
//! original single-thread code in [`crate::runner_protocol::execute`].
//!
//! # The two-level `(cell × thread)` partition
//!
//! A process is cell `c` of `cells` (from `AIPERF_CELL_ID`/`_COUNT`, default
//! `(0, 1)`); within it, thread `t` of `W` owns a nested slice. The design note
//! writes this partition as `(c*W + t, cells*W)`, but that flat index does **not**
//! nest inside the controller's per-cell envelope: when this process is a
//! controller child, [`build_cell_envelope`](crate::runner_protocol::cellular_controller) has
//! **already** sliced each phase's `requests`/`concurrency`/`rate` to cell `c`'s
//! round-robin share of the global stream (`i % cells == c`), and each thread must
//! draw a subset of *that* share, not of the whole global stream. The unique
//! partition family that (a) is a modulo partition, (b) nests inside cell `c`'s
//! `(c, cells)` ownership, and (c) tiles the global `cells*W` grid is
//!
//! ```text
//!     partition_t = ModuloCellPartition::new(c + cells*t, cells*W)
//! ```
//!
//! because `c + cells*t ≡ c (mod cells)`, so `{i : i % (cells*W) == c + cells*t}`
//! is a subset of cell `c`'s `{i : i % cells == c}`, and the `W` threads' residues
//! `{c, c+cells, …, c+(W-1)*cells}` partition cell `c`'s ownership exactly. The
//! flat `c*W + t` fails (b): its residue `≡ c*W + t (mod cells)` is generally not
//! `c`, so a thread would draw instances belonging to *other* cells and the merge
//! would overflow its residue class. For a single process (`cells == 1`) the two
//! formulas coincide (`0 + 1*t == 0*W + t == t`, modulus `W`), so this correction
//! only matters for the multi-process (`cells > 1`) grid — verified against
//! [`owned_positions`](crate::runner_protocol::cell_launcher) below.
//!
//! # Per-thread workload slicing (slice by `W`, not `cells*W`)
//!
//! Because the cell envelope is *already* cell-sliced (a controller child) or
//! carries the whole run (`cells == 1`), each thread slices the **cell's** phase
//! share across its `W` threads — never `cells*W`, which would double-slice a
//! controller child. Concretely, thread `t` of `W`:
//! - `requests → owned_positions(cell_requests, t, W)` (its round-robin share);
//! - `concurrency`/`prefill_concurrency → owned_positions(cap, t, W).max(1)`;
//! - `rate → cell_rate / W`.
//!
//! This is exactly [`build_cell_envelope`](crate::runner_protocol::cellular_controller)'s per-cell
//! arithmetic, re-applied one level down. The key invariant proven below:
//! slicing the cell's already-sliced `requests` by `owned_positions(·, t, W)`
//! yields per-thread dispatch counts that **equal** `owned_positions(global, c +
//! cells*t, cells*W)`, so each thread `(c, t)` stamps exactly the ordinals of its
//! residue class `c + cells*t (mod cells*W)` and the union across every cell and
//! thread is a permutation of `0..total` — the precondition
//! [`merge_records_in_global_order`](crate::cellular) tiles on.
//!
//! # Per-phase ordinal bases (`cells == 1`)
//!
//! The autonomous issuer stamps `phase_base + within*(cells*W) + (c + cells*t)`.
//! `phase_base` (the turns the run's prior phases dispatched globally) keeps
//! profiling ordinals from colliding with warmup's `[0, W)` block. A controller
//! child receives the bases in `AIPERF_CELL_PHASE_ORDINAL_BASES`; a single process
//! has no such env var, so [`compute_phase_ordinal_bases`] derives them from the
//! phase `requests` budgets (mirroring
//! [`crate::runner_protocol::cellular_controller::phase_ordinal_bases`]) and every thread's
//! `RunCapture` is injected with the same partition-independent map.
//!
//! # Once-per-cell vs per-thread (D5)
//!
//! Worker threads run **only** workload execution + dispatch + record capture.
//! Everything else stays once-per-cell on the main thread: the artifact tree,
//! side-channel telemetry (server-metrics / GPU / network), the shard merge, the
//! report build, and the controller ship. The worker threads never scrape a
//! sidecar or write an artifact.

use std::sync::Arc;

use crate::clock::Clock;
use crate::metrics_core::Phase;
use crate::phase_runtime::ScheduledPhaseSidecar;
use anyhow::{Context, Result, anyhow, bail};

use crate::runner_protocol::cell_launcher::owned_positions;
use crate::runner_protocol::execute::{
    ScheduledShardOutcome, ShardRecords, ShardedShared, execute_scheduled_shard, metrics_phase,
};
use crate::runner_protocol::protocol::PhaseSpec;

/// This thread's nested two-level dataset/ordinal partition within the run.
///
/// See the module docs for why the index is `cell_id + cells*thread_id` (nesting
/// inside cell `cell_id`'s ownership) rather than the design note's flat
/// `cell_id*workers + thread_id` (which does not nest under a controller child's
/// already-cell-sliced envelope). `workers` is this cell's thread-per-core count,
/// so the modulus is the full `cells*workers` grid.
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

/// One thread's copy of a phase with its `requests`/`concurrency`/`rate` sliced to
/// this thread's `1/W` share of the **cell's** budget (never `1/(cells*W)` — the
/// cell envelope is already cell-sliced for a controller child; see the module
/// docs). Mirrors [`build_cell_envelope`](crate::runner_protocol::cellular_controller) one level
/// down: round-robin `owned_positions` for the request budget and the admission
/// caps (floored to 1 so every thread makes progress when a cap `< W`), and an
/// even `rate / W` split of the arrival rate (the accepted aggregate-offered-rate
/// approximation, now at thread granularity). `user_centric`/`fixed_schedule`
/// phases are returned unchanged: they are trace-driven and rejected upstream for
/// `workers > 1` (see [`crate::runner_protocol::execute`]'s sharded branch), so they never reach a
/// worker.
pub(crate) fn slice_phase_for_thread(
    phase: &PhaseSpec,
    thread_id: usize,
    workers: u32,
) -> PhaseSpec {
    let t = thread_id as u32;
    let owned_budget = |value: u64| owned_positions(value, t, workers);
    // Admission caps split by the same round-robin share, floored to 1 so a cap
    // below the thread count over-subscribes to `workers` rather than starving a
    // thread — the same bounded trade the cellular per-cell split accepts.
    let owned_cap = |value: usize| owned_positions(value as u64, t, workers).max(1) as usize;
    let scaled_rate = |rate: f64| rate / workers as f64;

    let mut sliced = phase.clone();
    match &mut sliced {
        PhaseSpec::Concurrency {
            common,
            concurrency,
        } => {
            slice_common(common, &owned_budget, &owned_cap);
            *concurrency = owned_cap(*concurrency);
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
            slice_common(common, &owned_budget, &owned_cap);
            *rate = scaled_rate(*rate);
            if let Some(cap) = concurrency {
                *cap = owned_cap(*cap);
            }
        }
        PhaseSpec::Gamma {
            common,
            rate,
            concurrency,
            ..
        } => {
            slice_common(common, &owned_budget, &owned_cap);
            *rate = scaled_rate(*rate);
            if let Some(cap) = concurrency {
                *cap = owned_cap(*cap);
            }
        }
        // Trace-driven phases are rejected upstream for workers > 1; leave them
        // untouched rather than mis-slice a schedule the thread will never run.
        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => {}
    }
    sliced
}

/// Slice a phase's shared `requests` budget and `prefill_concurrency` cap in place.
fn slice_common(
    common: &mut crate::runner_protocol::protocol::PhaseCommonSpec,
    owned_budget: &impl Fn(u64) -> u64,
    owned_cap: &impl Fn(usize) -> usize,
) {
    if let Some(requests) = common.requests {
        common.requests = Some(owned_budget(requests));
    }
    if let Some(prefill) = common.prefill_concurrency {
        common.prefill_concurrency = Some(owned_cap(prefill));
    }
}

/// Each phase's global ordinal base for a single-process (`cells == 1`) sharded
/// run, keyed by metric phase — the turns the run's prior phases dispatch, so a
/// phase's base is the running sum of prior phases' `requests`. Mirrors
/// [`crate::runner_protocol::cellular_controller::phase_ordinal_bases`] over the typed phase specs.
///
/// A controller child instead reads the (global, already-correct) bases from
/// `AIPERF_CELL_PHASE_ORDINAL_BASES`; recomputing them here from a cell's *local*
/// sliced `requests` would understate them, so [`crate::runner_protocol::execute`]'s sharded branch
/// prefers the env map when present and only falls back to this for a lone
/// process.
pub(crate) fn compute_phase_ordinal_bases(
    phases: &[PhaseSpec],
) -> Result<std::collections::HashMap<Phase, usize>> {
    let mut bases = std::collections::HashMap::new();
    let mut base = 0usize;
    for phase in phases {
        let metric_phase = metrics_phase(phase)?;
        // Warmup precedes profiling; distinct phase names keep the map 1:1 with the
        // runner's two-phase structure (matches the controller's assumption).
        bases.insert(metric_phase, base);
        base += phase.common().requests.unwrap_or(0) as usize;
    }
    Ok(bases)
}

/// Run the scheduled pipeline sharded across `shared.workers` self-contained
/// sub-cell OS threads and merge their record shards into one cell-level outcome.
///
/// The caller (the main / cell thread) owns everything once-per-cell: it built
/// `shared`, created the artifact tree, and passes the already-built
/// profiling-phase side-channel sidecars (`profiling_sidecars`) which this
/// function drives on the main thread for the run window while the worker threads
/// execute — the worker threads never touch a sidecar (D5). The returned
/// [`ScheduledShardOutcome`] carries the concatenated, globally-ordinal-sorted
/// records the caller then ships / summarizes / persists exactly as the
/// single-thread path does.
///
/// `coordinator_clock` is the main thread's clock (the same real-clock timeline
/// the workers derive from [`shared.real_clock_anchor`](ShardedShared)); it is
/// used only to close the sidecar window after the last shard arrives.
pub(crate) async fn run_sharded_scheduled(
    shared: Arc<ShardedShared>,
    profiling_sidecars: Vec<std::rc::Rc<dyn ScheduledPhaseSidecar>>,
    coordinator_clock: std::rc::Rc<dyn Clock>,
) -> Result<ScheduledShardOutcome> {
    let workers = shared.workers as usize;
    if workers == 0 {
        bail!("sharded scheduled execution requires at least one worker");
    }

    // Each worker returns its shard over an unbounded channel. `send` on an
    // unbounded tokio channel is synchronous, so a worker never needs a runtime to
    // hand its result back; the main thread `recv().await`s, which keeps its
    // LocalSet pumping so the once-per-cell sidecar cadence loops run concurrently.
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
                // A closed receiver means the main thread already errored out; drop
                // the result rather than panic the worker.
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

    // Open the once-per-cell side-channel telemetry window on the main thread. The
    // sharded path brackets the sidecars over the whole worker-execution span; for
    // a profiling-only run this is exact, and when a warmup phase precedes
    // profiling the window is coarser (warmup-span scrapes attribute to profiling)
    // — a documented approximation of the single-thread per-phase bracketing, not
    // a silent drop.
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

    // Close the telemetry window before the workers' summaries are read.
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

    // Every worker has delivered; the joins are immediate and surface any panic.
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("a sharded scheduled worker thread panicked"))?;
    }

    merge_shards(shards)
}

/// Build and run one sub-cell thread's entire scheduled pipeline to completion.
///
/// Each thread owns a fresh `current_thread` runtime + `LocalSet` (the graph
/// thread-per-core model, `crate::graph::placement`) so its whole `!Send` stack —
/// clock, transport, capture, dispatcher, `SlotPool`s, plans — is thread-local and
/// contends nothing on the hot path.
fn run_worker_thread(shared: &ShardedShared, worker_id: usize) -> Result<ScheduledShardOutcome> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .with_context(|| format!("creating sharded scheduled worker {worker_id} runtime"))?;
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(execute_scheduled_shard(shared, worker_id)))
}

/// Concatenate the workers' shards and order them by their global dispatch
/// ordinal.
///
/// The shards already carry globally-unique two-level ordinals (a permutation of
/// `0..total`), so concatenation needs no renumber. Sorting by `request_index`
/// makes the per-record store placement and the deterministic-per-topology row
/// order independent of the (racy) thread completion order, and the input-session
/// list is re-sorted by session id so `inputs.json` is stable across runs.
fn merge_shards(
    shards: Vec<Option<Result<ScheduledShardOutcome>>>,
) -> Result<ScheduledShardOutcome> {
    // Take the first delivered shard as the merge base: a Folded (sketch) shard must
    // start from a real per-shard accumulator, not the empty `Retained` default that
    // would then refuse to merge with folded siblings.
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
    // Order retained records by the dense global dispatch ordinal every issuer stamped
    // so the combined shard tiles the store's `insert_record_at` slots without
    // collision and the row order is topology-deterministic (not thread-race
    // dependent). Folded shards keep only errored records — order is irrelevant to
    // error grouping — so there is nothing to sort there.
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

    /// The two-level partition nests inside the cell's ownership AND the union
    /// across every cell and thread tiles the global `cells*W` residue space — the
    /// precondition the records merge relies on. This is the property the flat
    /// `c*W + t` index fails.
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
                        // Nests: this thread's residue belongs to cell `cell_id`.
                        assert_eq!(
                            partition.cell_id() % cells,
                            cell_id,
                            "cells={cells} workers={workers}: thread residue must nest in its cell"
                        );
                        assert_eq!(partition.cell_count(), cells * workers);
                        seen[partition.cell_id() as usize] += 1;
                    }
                }
                // Tiles: every residue of the cells*W grid is owned exactly once.
                assert!(
                    seen.iter().all(|&count| count == 1),
                    "cells={cells} workers={workers}: residues must tile exactly once, got {seen:?}"
                );
            }
        }
    }

    /// Slicing a cell's already-cell-sliced `requests` budget by `W` yields
    /// per-thread counts that equal the flat global two-level `owned_positions`, so
    /// every thread stamps exactly its residue class and the ordinals tile
    /// `0..global`. This is the arithmetic that makes the nested slice correct.
    #[test]
    fn per_thread_slice_counts_match_global_two_level() {
        for global in [0u64, 1, 5, 10, 11, 37, 100] {
            for cells in 1..=4u32 {
                for workers in 1..=4u32 {
                    let mut total = 0u64;
                    for cell_id in 0..cells {
                        // The controller slices the global budget to this cell.
                        let cell_requests = owned_positions(global, cell_id, cells);
                        for thread_id in 0..workers {
                            // This thread slices the cell's share by W.
                            let nested = owned_positions(cell_requests, thread_id, workers);
                            // The flat global two-level share for residue c + cells*t.
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
        // Thread 0 of 4 owns ceil(100/4)=25 requests, ceil(8/4)=2 concurrency.
        let sliced = slice_phase_for_thread(&phase, 0, 4);
        assert_eq!(sliced.common().requests, Some(25));
        assert_eq!(sliced.common().prefill_concurrency, Some(1));
        assert_eq!(sliced.concurrency(), Some(2));
        // The four threads' request shares sum back to the phase budget.
        let total: u64 = (0..4)
            .map(|t| {
                slice_phase_for_thread(&phase, t, 4)
                    .common()
                    .requests
                    .unwrap()
            })
            .sum();
        assert_eq!(total, 100);
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
        // Rate splits evenly; a concurrency cap below W floors to 1 per thread.
        let sliced = slice_phase_for_thread(&phase, 3, 4);
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
}
