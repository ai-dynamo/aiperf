// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cellular controller — the Phase-2 multi-process topology.
//!
//! When a run requests `cfg.runtime.cells > 1`, the receiving runner becomes the
//! controller rather than executing in-process. It partitions the request budget by
//! `(cell_id, cell_count)`, spawns one `aiperf-runner --cell` child per cell (each a
//! separate OS process, wired with the autonomous issuer and per-cell sampler),
//! serves the [`transport`](aiperf::cellular::transport) endpoint the cells ship
//! their records-shard partitions and heartbeats back over, merges every cell's
//! records in global dispatch-ordinal order into the single authoritative
//! `native-v2.json`, and fails the run loudly if any cell exits non-zero. To the
//! Python orchestrator this is still one run behind one v2 request.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aiperf::cellular::{
    CellMessage, ControllerTransport, MetricsHeartbeat, RecordsShardPartition, TDigest,
    TcpControllerTransport, merge_records_in_global_order,
};
use aiperf::metrics_core::report::NativeReport;
use aiperf::metrics_core::{ExportContext, MetricsConfig, PERCENTILES};
use anyhow::{Context, Result, bail, ensure};

use crate::cellular_cell::CellLaunchSpec;

/// The outcome of a cellular run: the merged report path plus a live view of the
/// last heartbeat each cell reported (for diagnostics/logging).
pub struct CellularRunOutcome {
    /// Path of the written merged `native-v2.json`.
    pub report_path: PathBuf,
    /// The number of cells the run was partitioned across.
    pub cell_count: u32,
    /// The merged record count across all cells.
    pub record_count: usize,
}

/// Removes the controller's per-cell scratch tree on any exit path — a normal
/// return or a bailed run — so a failed cellular run never leaks a `/tmp` tree.
struct ScratchTreeGuard(PathBuf);

impl Drop for ScratchTreeGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs one benchmark across `cell_count` cell subprocesses and writes the merged
/// report to `report_path`. Blocks until every cell finishes.
pub fn run_cellular(
    envelope: &serde_json::Value,
    cell_count: u32,
    report_path: &Path,
) -> Result<CellularRunOutcome> {
    ensure!(cell_count >= 1, "cell_count must be at least 1");
    validate_cellular_run_shape(envelope)?;
    validate_cellular_phase_budgets(envelope, cell_count)?;
    // Ensure a profiling phase exists; every phase's `requests >= cell_count` (so no
    // cell owns zero) is already checked by validate_cellular_phase_budgets.
    profiling_request_budget(envelope)?;
    // Derive the metrics policy from the envelope so the merge reproduces the
    // authored SLOs / timeslices, exactly as the single-process path does.
    let metrics_config = cellular_metrics_config(envelope)?;

    // The controller is off the per-request hot path; a small multi-thread runtime
    // drives the transport accept/read tasks and the child processes concurrently.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("building controller runtime")?;

    runtime.block_on(async move {
        let mut transport = TcpControllerTransport::bind("127.0.0.1:0", cell_count as usize)
            .await
            .context("binding controller transport")?;
        let controller_addr = transport.local_addr().to_string();

        let temp_root =
            std::env::temp_dir().join(format!("aiperf-cellular-{}", std::process::id()));
        // Cleans the scratch tree on every exit path, including a bail. On a bail this
        // guard drops (removing `temp_root`) as the async block returns, a moment
        // BEFORE `runtime` drops and kill_on_drop SIGKILLs the cells; a surviving cell
        // could briefly recreate part of its `cell_dir` in that window. That is benign
        // — a cell's artifacts are discarded, and its records were already shipped if
        // it got far enough to matter — leaving at worst a small orphaned `/tmp` subtree
        // the OS reclaims. (A crashed run's data is not trusted regardless.)
        let _scratch = ScratchTreeGuard(temp_root.clone());
        // Each phase's global dispatch base (turns dispatched by prior phases): a
        // cell's sampler restarts each phase, so the cell adds this to its phase-local
        // slot to stamp the single-cell absolute slot. Same for every cell.
        let phase_ordinal_bases = phase_ordinal_bases(envelope)?;
        let mut children = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let cell_dir = temp_root.join(format!("cell-{cell_id}"));
            std::fs::create_dir_all(&cell_dir)
                .with_context(|| format!("creating cell {cell_id} artifact dir"))?;
            let cell_envelope = build_cell_envelope(envelope, cell_id, cell_count, &cell_dir)?;
            let spec = CellLaunchSpec {
                cell_id,
                cell_count,
                controller_addr: controller_addr.clone(),
                phase_ordinal_bases: phase_ordinal_bases.clone(),
                envelope: cell_envelope,
            };
            children.push(spawn_cell(&spec).await?);
        }

        // Watch each cell in a background task and forward the first hard failure, so
        // a cell that dies BEFORE connecting aborts the run rather than hanging the
        // transport's accept loop (which awaits `expected_cells` connections).
        let (failure_tx, mut failure_rx) =
            tokio::sync::mpsc::channel::<String>(cell_count as usize);
        for (cell_id, mut child) in children.into_iter().enumerate() {
            let failure_tx = failure_tx.clone();
            tokio::spawn(async move {
                let report = match child.wait().await {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(format!("cell {cell_id} exited with {status}")),
                    Err(error) => Some(format!("cell {cell_id} could not be waited on: {error}")),
                };
                if let Some(report) = report {
                    let _ = failure_tx.send(report).await;
                }
            });
        }
        drop(failure_tx);

        // Collect exactly one partition per cell (plus the latest heartbeat). The
        // `select!` is `biased`, so within a single poll a ready cell message is taken
        // before a cell-exit failure — the ship-then-exit race resolves in the cell's
        // favour when both land together. This is NOT blanket immunity for a cell that
        // already shipped: if a cell ships its partition and only LATER exits nonzero
        // while sibling partitions are still outstanding, the failure branch fires and
        // aborts the run even though that cell's records were already collected. For
        // this off-product-path experimental feature that is accepted — a cell's only
        // post-ship work is throwaway-temp-dir artifact writes (the controller, not the
        // cell, assembles the authoritative report), so a post-ship nonzero exit is
        // rare and the direction is fail-loud, never silent corruption or a parity
        // break. A cell that fails WITHOUT shipping leaves the transport with nothing
        // pending, so the failure branch fires and aborts — the crash-before-connecting
        // case that would otherwise hang the accept loop. A cell that connects but hangs
        // indefinitely without shipping or exiting is NOT covered (no per-cell deadline
        // yet — the failure watcher only fires on a cell exit); that bound belongs with
        // the cross-host transport work.
        let mut partitions: Vec<RecordsShardPartition> = Vec::with_capacity(cell_count as usize);
        let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
        while partitions.len() < cell_count as usize {
            tokio::select! {
                biased;
                message = transport.recv() => match message.context("receiving from cell")? {
                    Some(CellMessage::Partition(partition)) => partitions.push(partition),
                    Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                        heartbeats.insert(cell_id, *heartbeat);
                    }
                    None => bail!(
                        "transport closed with {} of {cell_count} cell partitions",
                        partitions.len()
                    ),
                },
                Some(failure) = failure_rx.recv() => bail!("{failure}"),
            }
        }

        // Records-first merge in global dispatch-ordinal order → the single report.
        let merged = merge_records_in_global_order(metrics_config, partitions)
            .context("merging cell partitions")?;
        let record_count = merged.record_count();
        let summary = merged.export_results(&ExportContext::phase(
            aiperf::metrics_core::Phase::Profiling,
        ));
        // Assemble the report so its metric data matches a 1-cell run: the profiling
        // metrics, the warmup section (carried only when a warmup phase actually ran,
        // so a profiling-only run stays byte-identical to the plain builder), plus the
        // run mode/model and configured endpoints. `was_cancelled` is left false — the
        // controller has no cross-cell cancellation.
        //
        // Three blocks a 1-cell report can carry are intentionally NOT reproduced here
        // (cells ship only their metric records + a heartbeat, so nothing else has a
        // channel back, and the merged report simply omits them):
        // (1) the coordinator's finalize_run provenance (distribution_id / workload /
        //     alias-resolved endpoint_profiles / extensions) — the controller carries
        //     transport/workload/cells/record_count in its terminal envelope instead
        //     of replaying the coordinator's alias resolution;
        // (2) the grouped per-error detail — cells ship the error/cancel flags (so
        //     error COUNTS are in the metrics) but not the messages group_record_errors
        //     needs, and a cross-cell regroup could not reproduce the 1-cell order; and
        // (3) side-channel sidecar data — server_metrics, GPU-telemetry-derived
        //     metrics, and network-RTT-adjusted metrics — which each cell would scrape
        //     locally but does not ship. All three are report-fidelity gaps, not metric
        //     corruption; the record-derived distributions stay byte-identical.
        let warmup =
            merged.export_results(&ExportContext::phase(aiperf::metrics_core::Phase::Warmup));
        let outcome = aiperf::metrics_core::report::RunOutcome {
            run: aiperf::metrics_core::report::ReportRunInfo {
                mode: Some("online".to_owned()),
                model: cellular_model_name(envelope),
            },
            summary: aiperf::metrics_core::report::ReportSummary {
                endpoints_configured: cellular_endpoint_urls(envelope),
                ..Default::default()
            },
            warmup: (!warmup.result_map().is_empty()).then_some(warmup),
            ..Default::default()
        };
        let report = NativeReport::from_outcome(&summary, &outcome);
        let json = serde_json::to_string_pretty(&report).context("serializing merged report")?;
        // Mirror the single-process path (execute.rs create_dir_all(&run.artifact_dir)):
        // create the report's parent so a fresh artifact_dir the orchestrator has not
        // pre-made does not fail the write. Cells write only to the throwaway scratch
        // tree, so nothing else creates this directory on the cellular path.
        if let Some(parent) = report_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        std::fs::write(report_path, json)
            .with_context(|| format!("writing merged report to {}", report_path.display()))?;

        // Aggregate the cells' final heartbeats (counters summed, sketches t-digest
        // merged) into one run-wide view written beside the report. The exact report
        // stays authoritative from S2; this is the cross-cell live-lane aggregate.
        let mut aggregate = heartbeats.into_values();
        if let Some(mut merged_heartbeat) = aggregate.next() {
            for heartbeat in aggregate {
                merged_heartbeat.merge(&heartbeat);
            }
            write_heartbeat_sidecar(report_path, &merged_heartbeat)
                .context("writing merged cellular heartbeat")?;
        }

        // `_scratch` removes `temp_root` on drop.
        Ok(CellularRunOutcome {
            report_path: report_path.to_path_buf(),
            cell_count,
            record_count,
        })
    })
}

/// Spawns one `aiperf-runner --cell` child and pipes its [`CellLaunchSpec`] to stdin.
async fn spawn_cell(spec: &CellLaunchSpec) -> Result<tokio::process::Child> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;
    let exe = std::env::current_exe().context("resolving the runner executable for a cell")?;
    let spec_bytes = serde_json::to_vec(spec).context("serializing cell launch spec")?;
    let mut child = tokio::process::Command::new(exe)
        .arg("--cell")
        .stdin(Stdio::piped())
        // A cell's records flow over the transport; keep stderr for its diagnostics
        // and drop stdout (its would-be terminal envelope is unused by the controller).
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        // On any controller abort the runtime is dropped, cancelling the watcher
        // tasks that own these children; kill_on_drop then SIGKILLs each cell so a
        // failed run never leaves cells generating load against the target.
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawning cell {}", spec.cell_id))?;
    let mut stdin = child.stdin.take().context("cell child stdin unavailable")?;
    stdin
        .write_all(&spec_bytes)
        .await
        .with_context(|| format!("piping launch spec to cell {}", spec.cell_id))?;
    stdin.shutdown().await.ok();
    Ok(child)
}

/// Builds the protocol-v2 envelope for one cell: the same run with its phase
/// budgets sliced to the cell's owned share and its own scratch artifact dir. The
/// dataset and seed are unchanged — the cell's `PartitionedSampler` selects its
/// owned instances from the shared space.
///
/// The runner rebuilds each cell's sampler fresh at every phase boundary (the
/// dataset RNG re-seeds per phase), so a cell draws its owned instances of *each
/// phase* from position 0 and the per-cell issuer stamps a *phase-local* ordinal.
/// Each phase is therefore partitioned independently: cell `k` takes
/// `owned_positions(phase_requests, k, cell_count)` — its share of that phase's
/// `{0, 1, …, phase_requests-1}` instance space — so the cells' per-phase shares sum
/// to the phase budget and their phase-local ordinals tile `0..phase_requests`, and
/// the union of each phase's instances equals the single-cell run's. (A cumulative
/// base offset would draw the wrong instances, since the sampler is not continuous
/// across phases.)
fn build_cell_envelope(
    envelope: &serde_json::Value,
    cell_id: u32,
    cell_count: u32,
    cell_dir: &Path,
) -> Result<serde_json::Value> {
    let mut cell = envelope.clone();
    let run = cell
        .get_mut("run")
        .and_then(serde_json::Value::as_object_mut)
        .context("envelope has no run object")?;
    run.insert(
        "artifact_dir".to_owned(),
        serde_json::Value::String(cell_dir.to_string_lossy().into_owned()),
    );
    // The cell runs single-process (its slice); its autonomous behaviour comes from
    // the env vars the controller sets, not from re-entering the controller path.
    if let Some(runtime) = run
        .get_mut("cfg")
        .and_then(|cfg| cfg.get_mut("runtime"))
        .and_then(serde_json::Value::as_object_mut)
    {
        runtime.insert("cells".to_owned(), serde_json::Value::from(1));
    }
    let phases = run
        .get_mut("cfg")
        .and_then(|cfg| cfg.get_mut("phases"))
        .and_then(serde_json::Value::as_array_mut)
        .context("run cfg has no phases array")?;
    for phase in phases.iter_mut() {
        let Some(phase) = phase.as_object_mut() else {
            continue;
        };
        if let Some(requests) = phase.get("requests").and_then(serde_json::Value::as_u64) {
            let owned = owned_positions(requests, cell_id, cell_count);
            phase.insert("requests".to_owned(), serde_json::Value::from(owned));
        }
        // Split the global concurrency cap by the same round-robin share as the
        // request budget so the cells' caps sum to the requested aggregate in-flight.
        // Concurrency is a per-phase saturation cap, not a budget that tiles the
        // stream, so it needs no base offset. `.max(1)` keeps every cell able to make
        // progress when `concurrency < cell_count`, a bounded over-subscription.
        // `prefill_concurrency` is the same kind of cap and is split identically.
        for cap in ["concurrency", "prefill_concurrency"] {
            if let Some(value) = phase.get(cap).and_then(serde_json::Value::as_u64) {
                let cell_cap = owned_positions(value, cell_id, cell_count).max(1);
                phase.insert(cap.to_owned(), serde_json::Value::from(cell_cap));
            }
        }
        // Split the arrival RATE (requests/sec) evenly: the cells run concurrently, so
        // each cell paces at `rate / cell_count` and their aggregate offered rate — and
        // thus the merged report's throughput and duration — matches the single-cell
        // run. Without this every cell would fire at the full authored rate (N× load).
        if let Some(rate) = phase.get("rate").and_then(serde_json::Value::as_f64) {
            phase.insert(
                "rate".to_owned(),
                serde_json::Value::from(rate / cell_count as f64),
            );
        }
    }
    Ok(cell)
}

/// Whitelists a cellular run to the exact shape the partition/issuance seam is sound
/// for: the scheduled HTTP transport over **synthetic, single-turn** datasets. Two
/// invariants underpin the byte-parity contract and each fails closed here:
///
/// - **Records ship only from the scheduled HTTP executor.** A gRPC/dynosim transport
///   or a non-synthetic (`file`/`public`, incl. graph-program) dataset runs a
///   different executor that never ships a partition, so the controller would hang.
/// - **One sampler draw must equal one dispatched turn.** [`PartitionedSampler`]
///   partitions by conversation *draw*, but the issuer stamps a per-*turn* ordinal
///   ([`CellularAutonomousIssuer`]); a multi-turn conversation makes the two diverge,
///   so the merged report silently reorders (or, for variable turn counts, draws a
///   different instance set). Only `turns == 1` (the default) is sound.
///
/// [`PartitionedSampler`]: aiperf::dataset::sampler
/// [`CellularAutonomousIssuer`]: aiperf::cellular::CellularAutonomousIssuer
fn validate_cellular_run_shape(envelope: &serde_json::Value) -> Result<()> {
    if let Some(transport) = envelope
        .pointer("/run/cfg/transport/type")
        .and_then(serde_json::Value::as_str)
    {
        ensure!(
            transport == "http",
            "cellular runs support only the scheduled HTTP transport; got transport.type={transport:?}"
        );
    }
    let datasets = envelope
        .pointer("/run/cfg/datasets")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no datasets array")?;
    for dataset in datasets {
        let kind = dataset.get("type").and_then(serde_json::Value::as_str);
        ensure!(
            kind == Some("synthetic"),
            "cellular runs support only synthetic datasets (whose conversations the sampler can partition one-draw-per-turn); got dataset type {kind:?}"
        );
        // Single-turn only: `turns` absent defaults to a fixed 1; a fixed `{value: 1}`
        // is the only other single-turn form. Anything else (a larger value or a
        // distribution) makes a conversation dispatch multiple turns.
        let single_turn = dataset.get("turns").is_none_or(|turns| {
            turns.get("value").and_then(serde_json::Value::as_f64) == Some(1.0)
        });
        ensure!(
            single_turn,
            "cellular runs support only single-turn conversations (turns == 1); a multi-turn dataset diverges the sampler draw index from the issuer's per-turn ordinal and silently breaks byte parity"
        );
    }
    // Byte-parity requires every cell to compose the SAME dataset space. Each cell is a
    // separate process; without a concrete run seed it entropy-seeds an independent
    // random dataset (different prompts / ISL-OSL draws), voiding the partition
    // invariant. Require a run-level seed (the cell envelopes inherit it verbatim).
    ensure!(
        envelope
            .pointer("/run/random_seed")
            .and_then(serde_json::Value::as_u64)
            .is_some(),
        "cellular runs require a concrete run-level random_seed; without it each cell composes an independent random dataset and the merged report is not reproducible"
    );
    // A single request-per-turn URL selection is reproducible; multiple URLs round-robin
    // in cell-local order, so a heterogeneous-backend run would diverge per cell.
    let url_count = envelope
        .pointer("/run/cfg/endpoint/urls")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    ensure!(
        url_count <= 1,
        "cellular runs support a single endpoint URL; {url_count} URLs would round-robin differently per cell and diverge on heterogeneous backends"
    );
    Ok(())
}

/// Phase `type`s whose dispatch count is exactly the `requests` budget and whose
/// turns are drawn one at a time through the (partitionable) sampler — the only
/// shapes cellular can partition. The trace-driven types (`fixed_schedule`,
/// `user_centric`) build their schedule from the *full* conversation list and set
/// `enforce_stop = false`, so every cell would replay the entire trace (N× load and
/// N× records) — a merge the completeness check would accept silently.
const CELLULAR_REQUEST_BOUNDED_PHASE_TYPES: [&str; 4] =
    ["concurrency", "poisson", "gamma", "constant"];

/// Rejects a cellular run whose phases are not exactly request-bounded. The
/// dense-ordinal tiling requires every phase's *actual* dispatch count to equal its
/// sliced `requests` budget. A phase mis-partitions — running unpartitioned and/or
/// leaving gaps, or silently N×-ing the load and record count — if it:
/// - has a `type` outside [`CELLULAR_REQUEST_BOUNDED_PHASE_TYPES`] (a trace-driven
///   `fixed_schedule`/`user_centric` phase ignores `requests` and replays the full
///   trace per cell);
/// - lacks a `requests` budget, or has one below `cell_count` (a cell would own zero);
/// - carries a `duration`/`sessions`/`adaptive_scale` bound that can stop it early;
/// - drives a concurrency/prefill/rate **ramp**, which the controller cannot slice
///   per cell (each cell would ramp to the full authored target, N×-ing the aggregate); or
/// - has a `concurrency`/`prefill_concurrency` cap below `cell_count` — the `.max(1)`
///   per-cell floor would then over-subscribe the aggregate in-flight to `cell_count`.
///
/// Fail closed rather than silently corrupt. The static `concurrency`/
/// `prefill_concurrency`/`rate` caps at or above `cell_count` ARE sliced per cell (see
/// [`build_cell_envelope`]). Two supported knobs are only *statistically* equivalent to
/// a 1-cell run, not byte-identical: rate-based phases match the aggregate offered rate
/// but not the per-turn arrival sample path, and a post-send `cancellation` policy
/// matches the aggregate cancellation rate but not the exact cancelled subset (its RNG
/// runs in cell-local dispatch order). Byte-parity is exact for a seeded `concurrency`
/// phase without cancellation; the other two are intentional approximations.
fn validate_cellular_phase_budgets(envelope: &serde_json::Value, cell_count: u32) -> Result<()> {
    let phases = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no phases array")?;
    let cells = cell_count as u64;
    for phase in phases {
        let name = phase
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        let phase_type = phase.get("type").and_then(serde_json::Value::as_str);
        ensure!(
            phase_type.is_some_and(|t| CELLULAR_REQUEST_BOUNDED_PHASE_TYPES.contains(&t)),
            "cellular runs support only request-bounded phase types ({}); phase {name:?} has type {phase_type:?}, whose dispatch count is trace-driven and would replay the full trace per cell",
            CELLULAR_REQUEST_BOUNDED_PHASE_TYPES.join("/")
        );
        let requests = phase.get("requests").and_then(serde_json::Value::as_u64);
        ensure!(
            requests.is_some_and(|r| r >= cells),
            "cellular runs require every phase to have a `requests` budget >= cell_count ({cell_count}); phase {name:?} has {requests:?}"
        );
        ensure!(
            phase.get("duration").is_none()
                && phase.get("sessions").is_none()
                && phase.get("adaptive_scale").is_none(),
            "cellular runs require a fixed per-phase request budget; phase {name:?} carries a `duration`/`sessions`/`adaptive_scale` bound whose actual dispatch count can diverge from `requests` and break the merge"
        );
        ensure!(
            phase.get("concurrency_ramp").is_none()
                && phase.get("prefill_ramp").is_none()
                && phase.get("rate_ramp").is_none(),
            "cellular runs do not support concurrency/prefill/rate ramps; the controller cannot slice a ramp target per cell, so every cell would ramp to the full value and N× the aggregate load; phase {name:?}"
        );
        // A post-send `cancellation` policy is allowed: each cell applies the same
        // per-request probability, so the aggregate cancellation rate matches, though
        // the exact cancelled subset differs (cell-local RNG order) — an accepted
        // non-byte-parity approximation, not a rejected shape.
        for cap in ["concurrency", "prefill_concurrency"] {
            if let Some(value) = phase.get(cap).and_then(serde_json::Value::as_u64) {
                ensure!(
                    value >= cells,
                    "cellular runs require a `{cap}` cap >= cell_count ({cell_count}) so it splits evenly (a smaller cap floors to 1 per cell and over-subscribes the aggregate to cell_count); phase {name:?} has {value}"
                );
            }
        }
    }
    Ok(())
}

/// The number of dispatch-stream positions in `[0, total)` that cell `k` owns under
/// round-robin ownership (`position % cell_count == cell_id`) — `ceil((total-k)/C)`.
/// A phase's per-cell slice is the difference of this over the phase's `[base,
/// base+len)` window (see [`build_cell_envelope`]); over a single phase (`base=0`)
/// it is just each cell's share, summing to `total`.
fn owned_positions(total: u64, cell_id: u32, cell_count: u32) -> u64 {
    let count = cell_count as u64;
    let k = cell_id as u64;
    if k >= total {
        return 0;
    }
    (total - k).div_ceil(count)
}

/// The native metrics policy for the merge, derived from the v2 envelope exactly as
/// the single-process path does — `cfg.metrics` (SLOs + slice duration) plus
/// `cfg.endpoint.use_server_token_count`. Passing `MetricsConfig::default()` would
/// silently drop authored goodput SLOs and timeslice sweep-lines from the merged
/// report. Mirrors [`crate::protocol::BenchmarkRunConfigWireV2`]'s
/// `from_value(cfg.metrics).unwrap_or_default()` so an absent/loose `metrics` block
/// falls back the same way (`metrics_config` still validates any SLO names present).
fn cellular_metrics_config(envelope: &serde_json::Value) -> Result<MetricsConfig> {
    let spec: crate::protocol::MetricsSpec = envelope
        .pointer("/run/cfg/metrics")
        .cloned()
        .map(|value| serde_json::from_value(value).unwrap_or_default())
        .unwrap_or_default();
    let use_server_token_count = envelope
        .pointer("/run/cfg/endpoint/use_server_token_count")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    crate::execute::metrics_config(&spec, use_server_token_count)
}

/// Each phase's global ordinal base (`phase name -> turns dispatched by prior
/// phases`) from the v2 envelope. Phases execute in array order, so a phase's base is
/// the running sum of prior phases' `requests`. Assumes distinct phase names (the
/// runner's `warmup`/`profiling` structure), since the cell keys the base by metric
/// phase. `validate_cellular_phase_budgets` has already ensured every phase is
/// request-bounded.
fn phase_ordinal_bases(envelope: &serde_json::Value) -> Result<BTreeMap<String, u64>> {
    let phases = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no phases array")?;
    let mut bases = BTreeMap::new();
    let mut base: u64 = 0;
    for phase in phases {
        let name = phase
            .get("name")
            .and_then(serde_json::Value::as_str)
            .context("cellular phase has no name")?;
        bases.insert(name.to_owned(), base);
        let requests = phase
            .get("requests")
            .and_then(serde_json::Value::as_u64)
            .context("cellular phase has no request budget")?;
        base += requests;
    }
    Ok(bases)
}

/// The profiling phase's request budget from the v2 envelope.
fn profiling_request_budget(envelope: &serde_json::Value) -> Result<u64> {
    let phases = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no phases array")?;
    for phase in phases {
        let is_profiling = phase
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(|name| name == "profiling")
            .unwrap_or(false);
        if is_profiling
            && let Some(requests) = phase.get("requests").and_then(serde_json::Value::as_u64)
        {
            return Ok(requests);
        }
    }
    bail!("cellular runs require a profiling phase with a request budget")
}

/// The primary model name from the v2 envelope's model list, for the merged
/// report's run info (matching the single-process report).
fn cellular_model_name(envelope: &serde_json::Value) -> Option<String> {
    envelope
        .pointer("/run/cfg/models/items/0/name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// The configured endpoint URLs from the v2 envelope, for the merged report's
/// summary (matching the single-process report).
fn cellular_endpoint_urls(envelope: &serde_json::Value) -> Vec<String> {
    envelope
        .pointer("/run/cfg/endpoint/urls")
        .and_then(serde_json::Value::as_array)
        .map(|urls| {
            urls.iter()
                .filter_map(|url| url.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Reads `cfg.runtime.cells` from a v2 envelope, defaulting to 1 (single process).
pub fn cell_count_from_envelope(envelope: &serde_json::Value) -> u32 {
    envelope
        .pointer("/run/cfg/runtime/cells")
        .and_then(serde_json::Value::as_u64)
        .map(|cells| cells.clamp(1, 1024) as u32)
        .unwrap_or(1)
}

/// Writes the merged cross-cell live heartbeat beside the report as a JSON-safe
/// percentile projection (a raw t-digest anchors `min = +inf`, which JSON cannot
/// encode). The exact report percentiles stay authoritative from S2.
fn write_heartbeat_sidecar(report_path: &Path, heartbeat: &MetricsHeartbeat) -> Result<()> {
    let quantiles: Vec<f64> = PERCENTILES.iter().map(|&p| p as f64 / 100.0).collect();
    let project = |sketch: &TDigest| -> serde_json::Value {
        let percentiles: serde_json::Map<String, serde_json::Value> = PERCENTILES
            .iter()
            .zip(sketch.quantiles(&quantiles))
            .filter_map(|(&p, value)| {
                value.map(|value| (format!("p{p}"), serde_json::json!(value)))
            })
            .collect();
        serde_json::json!({ "count": sketch.count(), "percentiles": percentiles })
    };
    let document = serde_json::json!({
        "event": "cellular_heartbeat_merged",
        "counters": {
            "issued": heartbeat.counters.issued,
            "completed": heartbeat.counters.completed,
            "errored": heartbeat.counters.errored,
        },
        "ttft_ms": project(&heartbeat.ttft_ms),
        "itl_ms": project(&heartbeat.itl_ms),
        "latency_ms": project(&heartbeat.latency_ms),
    });
    let path = report_path.with_file_name("cellular-heartbeat.json");
    std::fs::write(&path, serde_json::to_string_pretty(&document)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_shipping_run_shapes() {
        // Supported: http (or default) transport, synthetic single-turn dataset, a run
        // seed, and at most one endpoint URL.
        for ok in [
            serde_json::json!({"run": {"random_seed": 42, "cfg": {
                "transport": {"type": "http"},
                "datasets": [{"type": "synthetic", "turns": {"value": 1}}],
                "endpoint": {"urls": ["http://x/v1"]},
            }}}),
            serde_json::json!({"run": {"random_seed": 1, "cfg": {
                "datasets": [{"type": "synthetic"}],
            }}}),
        ] {
            assert!(
                validate_cellular_run_shape(&ok).is_ok(),
                "should accept {ok}"
            );
        }
        // Fail closed on each unsupported aspect (all else valid + seeded):
        for bad in [
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "transport": {"type": "grpc"}}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "dag_jsonl"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "public"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"value": 3}}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"mean": 2.0, "stddev": 1.0}}]}}}),
            // Missing run seed.
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}}),
            // Multiple endpoint URLs.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "endpoint": {"urls": ["http://a", "http://b"]}}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&bad).is_err(),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn derives_metrics_config_from_the_envelope() {
        // Authored SLOs + slice duration + server-token-count flow into the merge
        // config (the merge would silently drop goodput/timeslices under
        // MetricsConfig::default()).
        let env = serde_json::json!({"run": {"cfg": {
            "metrics": {"slos": {"request_latency": 60.0}, "slice_duration_seconds": 2.0},
            "endpoint": {"use_server_token_count": true},
        }}});
        let config = cellular_metrics_config(&env).expect("valid metrics");
        assert_eq!(config.slos.len(), 1);
        assert_eq!(config.slice_duration_ns, Some(2_000_000_000));
        assert!(config.use_server_token_count);
        // Absent metrics/endpoint → the default policy (empty SLOs, no timeslicing).
        let bare =
            cellular_metrics_config(&serde_json::json!({"run": {"cfg": {}}})).expect("default");
        assert!(bare.slos.is_empty());
        assert_eq!(bare.slice_duration_ns, None);
        assert!(!bare.use_server_token_count);
        // An SLO metric absent from the catalog is rejected, like the 1-cell path.
        assert!(
            cellular_metrics_config(&serde_json::json!({"run": {"cfg": {
                "metrics": {"slos": {"not_a_real_metric": 1.0}},
            }}}))
            .is_err()
        );
    }

    #[test]
    fn rejects_non_request_bounded_phases() {
        // Request-bounded (arrival-pattern) phase types with requests + caps >= cells
        // pass; a post-send cancellation policy is allowed (approximate).
        let ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "warmup", "requests": 10, "concurrency": 8},
            {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8, "cancellation": {"rate": 25.0, "delay": 0.5}},
        ]}}});
        assert!(validate_cellular_phase_budgets(&ok, 4).is_ok());
        // Fail closed (cell_count 4) on: a trace-driven or missing phase type; requests
        // absent or below cell_count; a duration/sessions/adaptive_scale bound; a ramp;
        // a post-send cancellation; or a concurrency/prefill cap below cell_count.
        for bad in [
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "fixed_schedule", "name": "profiling", "requests": 100},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "user_centric", "name": "profiling", "requests": 100},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"name": "profiling", "requests": 100},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [{"type": "concurrency", "name": "profiling"}]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 2},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 100, "duration": 5.0},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 100, "sessions": 3},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 100, "adaptive_scale": {"controller": "ramp_until_fail"}},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency_ramp": {"start": 1, "end": 100}},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "constant", "name": "profiling", "requests": 100, "rate_ramp": {"start": 1.0, "end": 50.0}},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 2},
            ]}}}),
        ] {
            assert!(
                validate_cellular_phase_budgets(&bad, 4).is_err(),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn owned_positions_sum_to_total_and_tile() {
        for total in [1_u64, 7, 100, 500, 501] {
            for count in 1..=8u32 {
                let sum: u64 = (0..count).map(|k| owned_positions(total, k, count)).sum();
                assert_eq!(sum, total, "total {total} count {count}");
            }
        }
    }

    #[test]
    fn each_phase_partitions_and_tiles_independently() {
        // The per-cell sampler restarts each phase, so each phase is partitioned on
        // its OWN instance space `0..phase_requests`: cell k's per-phase count is
        // `owned_positions(phase_requests, k, count)` and the phase-local ordinals
        // `within*count + cell_id` tile `0..phase_requests` densely — the invariant
        // the phase-aware merge relies on, and what makes the merged per-phase
        // instance set equal a 1-cell run's. A cumulative base offset (the earlier
        // bug) would draw the wrong instances since the sampler is not continuous.
        let dir = Path::new("/tmp/aiperf-cellular-envelope-test");
        for (warmup, profiling) in [(100u64, 1000u64), (3, 3), (1, 7), (10, 250), (7, 13)] {
            for count in 1..=5u32 {
                let mut warmup_seen = vec![false; warmup as usize];
                let mut profiling_seen = vec![false; profiling as usize];
                for cell_id in 0..count {
                    let envelope = serde_json::json!({"run": {"cfg": {"phases": [
                        {"name": "warmup", "requests": warmup},
                        {"name": "profiling", "requests": profiling},
                    ]}}});
                    let cell = build_cell_envelope(&envelope, cell_id, count, dir).unwrap();
                    let phases = cell
                        .pointer("/run/cfg/phases")
                        .and_then(serde_json::Value::as_array)
                        .unwrap();
                    for (seen, phase) in [
                        (&mut warmup_seen, &phases[0]),
                        (&mut profiling_seen, &phases[1]),
                    ] {
                        let owned = phase
                            .get("requests")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap();
                        for within in 0..owned {
                            let ordinal = (within * count as u64 + cell_id as u64) as usize;
                            assert!(ordinal < seen.len(), "phase ordinal {ordinal} out of range");
                            assert!(!seen[ordinal], "duplicate phase ordinal {ordinal}");
                            seen[ordinal] = true;
                        }
                    }
                }
                assert!(
                    warmup_seen.iter().all(|&s| s),
                    "warmup gap (w{warmup} c{count})"
                );
                assert!(
                    profiling_seen.iter().all(|&s| s),
                    "profiling gap (p{profiling} c{count})"
                );
            }
        }
    }

    #[test]
    fn slices_rate_and_concurrency_caps_per_cell() {
        // The arrival rate is divided evenly (the cells' rates sum to the authored
        // aggregate), and the concurrency/prefill caps are per-cell round-robin shares.
        let dir = Path::new("/tmp/aiperf-cellular-envelope-test");
        let envelope = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "constant", "name": "profiling", "requests": 100, "rate": 40.0,
             "concurrency": 8, "prefill_concurrency": 4},
        ]}}});
        let cell_count = 4u32;
        let mut rate_sum = 0.0;
        for cell_id in 0..cell_count {
            let cell = build_cell_envelope(&envelope, cell_id, cell_count, dir).unwrap();
            let phase = &cell
                .pointer("/run/cfg/phases")
                .and_then(serde_json::Value::as_array)
                .unwrap()[0];
            rate_sum += phase
                .get("rate")
                .and_then(serde_json::Value::as_f64)
                .unwrap();
            assert!(
                phase
                    .get("concurrency")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap()
                    <= 8
            );
            assert!(
                phase
                    .get("prefill_concurrency")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap()
                    <= 4
            );
        }
        assert!(
            (rate_sum - 40.0).abs() < 1e-9,
            "per-cell rates must sum to the authored aggregate, got {rate_sum}"
        );
    }

    #[test]
    fn cell_count_reads_runtime_cells() {
        let envelope = serde_json::json!({"run": {"cfg": {"runtime": {"cells": 4}}}});
        assert_eq!(cell_count_from_envelope(&envelope), 4);
        let single = serde_json::json!({"run": {"cfg": {"runtime": {"workers": 1}}}});
        assert_eq!(cell_count_from_envelope(&single), 1);
    }
}
