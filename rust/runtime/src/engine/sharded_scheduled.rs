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
use anyhow::{Context, Result, anyhow, bail};

use crate::engine::cell_launcher::owned_positions;
use crate::engine::execute::{
    ScheduledShardOutcome, ShardRecords, ShardedShared, execute_scheduled_shard, metrics_phase,
};
use crate::engine::protocol::PhaseSpec;

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
/// Admission caps are floored to one. `fixed_schedule` is partitioned by
/// conversation in [`NativeDatasetConversationSource`](crate::multiturn).
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
        // Open-loop churn is aggregate-equivalent across shards, but the
        // per-turn-shape split is timing-dependent.
        PhaseSpec::UserCentric {
            common,
            rate,
            users,
            concurrency,
        } => {
            slice_common(common, &owned_budget, &owned_cap);
            *rate = scaled_rate(*rate);
            *users = owned_cap(*users);
            if let Some(cap) = concurrency {
                *cap = owned_cap(*cap);
            }
        }
        PhaseSpec::FixedSchedule { .. } => {}
    }
    sliced
}

/// Slice a phase's shared `requests` budget and `prefill_concurrency` cap in place.
fn slice_common(
    common: &mut crate::engine::protocol::PhaseCommonSpec,
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
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
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
        let sliced = slice_phase_for_thread(&phase, 0, 4);
        assert_eq!(sliced.common().requests, Some(25));
        assert_eq!(sliced.common().prefill_concurrency, Some(1));
        assert_eq!(sliced.concurrency(), Some(2));
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
