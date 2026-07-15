// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cellular controller — the Phase-2 multi-process topology.
//!
//! When a run requests `cfg.runtime.cells > 1`, the receiving runner becomes the
//! controller rather than executing in-process. It partitions the request budget by
//! `(cell_id, cell_count)`, spawns one `aiperf-runner --cell` child per cell (each a
//! separate OS process, wired with the autonomous issuer and per-cell sampler),
//! serves the [`transport`](crate::cellular::transport) endpoint the cells ship
//! their records-shard partitions and heartbeats back over, merges every cell's
//! records in global dispatch-ordinal order into the single authoritative
//! `native-v2.json`, and fails the run loudly if any cell exits non-zero. To the
//! Python orchestrator this is still one run behind one v2 request.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cellular::{
    MetricsHeartbeat, RecordsShardPartition, TDigest, merge_records_by_concatenation,
    merge_records_in_global_order,
};
use crate::metrics_core::report::NativeReport;
use crate::metrics_core::{ExportContext, MetricsAccumulator, MetricsConfig, PERCENTILES};
use anyhow::{Context, Result, bail, ensure};

use crate::runner_protocol::cell_launcher::owned_positions;

// The velo transport + launcher wiring is the only part of the controller that
// needs the `velo` feature; the validation, budget-slicing, merge, and report
// assembly below are plain envelope/metric logic reused by the non-velo build.
#[cfg(feature = "velo")]
use crate::cellular::transport::connect::{BindSpec, BootstrapSource, build_velo, serve_bootstrap};
#[cfg(feature = "velo")]
use crate::cellular::{CellMessage, ControllerTransport, SpecFor, VeloControllerTransport};
#[cfg(feature = "velo")]
use crate::runner_protocol::cell_launcher::{CellLaunchContext, select_launcher};

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

/// Whether the run targets a graph program (`dag_jsonl` / `weka_trace` / `dynamo_trace`),
/// as opposed to a scheduled synthetic dataset. Graph programs partition cleanly by whole
/// trace, so they take the concatenation merge and bypass the scheduled request-budget guards.
fn is_graph_dataset(envelope: &serde_json::Value) -> bool {
    envelope
        .pointer("/run/cfg/datasets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|datasets| {
            datasets.iter().any(|dataset| {
                matches!(
                    dataset.get("format").and_then(serde_json::Value::as_str),
                    Some("dag_jsonl" | "weka_trace" | "dynamo_trace")
                )
            })
        })
}

/// Which execution path a cellular run drives. The scheduled arrival-paced executor and
/// the graph trace executor differ in exactly three ways — how the phases are validated,
/// whether a per-phase global ordinal base applies, and how the cells' records merge — so
/// each kind answers those three. A future kind (e.g. gRPC) is one variant plus three arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellularRunKind {
    /// Synthetic scheduled runs: request-bounded phases, pre-tiled global dispatch ordinals.
    Scheduled,
    /// Graph programs (dag_jsonl/weka_trace/dynamo_trace): trace-partitioned, concatenation-merged.
    Graph,
}

impl CellularRunKind {
    /// A graph-format dataset selects the graph path; anything else is scheduled.
    fn detect(envelope: &serde_json::Value) -> Self {
        if is_graph_dataset(envelope) {
            Self::Graph
        } else {
            Self::Scheduled
        }
    }

    /// Validate the phases for this kind (scheduled: request-bounded budgets + a profiling
    /// budget; graph: no static `requests` budget, caps >= cell_count).
    fn validate_phases(&self, envelope: &serde_json::Value, cell_count: u32) -> Result<()> {
        match self {
            Self::Scheduled => {
                validate_cellular_phase_budgets(envelope, cell_count)?;
                profiling_request_budget(envelope)?;
                Ok(())
            }
            Self::Graph => validate_graph_cellular_phases(envelope, cell_count),
        }
    }

    /// Each phase's global ordinal base — scheduled cells add it to stamp the single-cell
    /// absolute slot; graph cells partition by trace and never read it (empty).
    fn phase_ordinal_bases(&self, envelope: &serde_json::Value) -> Result<BTreeMap<String, u64>> {
        match self {
            Self::Scheduled => phase_ordinal_bases(envelope),
            Self::Graph => Ok(BTreeMap::new()),
        }
    }

    /// Merge the cells' record partitions into one accumulator (scheduled: pre-tiled global
    /// order; graph: concatenate by cell_id + re-number local indices densely).
    fn merge(
        &self,
        config: MetricsConfig,
        partitions: Vec<RecordsShardPartition>,
    ) -> Result<MetricsAccumulator> {
        match self {
            Self::Scheduled => {
                merge_records_in_global_order(config, partitions).context("merging cell partitions")
            }
            Self::Graph => Ok(merge_records_by_concatenation(config, partitions)),
        }
    }
}

/// Runs one benchmark across `cell_count` cells and writes the merged report to
/// `report_path`. Blocks until every cell ships. Requires the `velo` feature (the
/// cell transport).
#[cfg(feature = "velo")]
pub fn run_cellular(
    envelope: &serde_json::Value,
    cell_count: u32,
    report_path: &Path,
    exporters: &crate::export::ExporterRegistry,
) -> Result<CellularRunOutcome> {
    ensure!(cell_count >= 1, "cell_count must be at least 1");
    validate_cellular_run_shape(envelope)?;
    // The dataset-shape gate above runs before the kind is known; the kind then names
    // the scheduled-vs-graph run once and owns its three differing behaviours (phase
    // validation, ordinal bases, record merge). The scheduled path folds the profiling
    // budget check in — graph phases carry sessions/duration, not a `requests` budget.
    let kind = CellularRunKind::detect(envelope);
    kind.validate_phases(envelope, cell_count)?;
    warn_dropped_sidecar_telemetry(envelope);
    warn_cellular_approximations(envelope);
    // One shared seed for every cell when the author gave none (else `None` and each
    // cell inherits the authored `run.random_seed` verbatim).
    let injected_seed = resolve_cellular_seed(envelope);
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
        let temp_root =
            std::env::temp_dir().join(format!("aiperf-cellular-{}", std::process::id()));
        // Cleans the scratch tree on every exit path, including a bail. On a bail this
        // guard drops (removing `temp_root`) as the async block returns, a moment
        // BEFORE `runtime` drops and kill_on_drop SIGKILLs the cells; a surviving cell
        // could briefly recreate part of its `cell_dir` in that window. That is benign
        // — a cell's artifacts are discarded, and its records were already shipped if
        // it got far enough to matter. (A crashed run's data is not trusted regardless.)
        let _scratch = ScratchTreeGuard(temp_root.clone());
        std::fs::create_dir_all(&temp_root)
            .with_context(|| format!("creating cellular scratch {}", temp_root.display()))?;

        // Build the controller's velo transport and publish its PeerInfo so cells
        // reach it from the one operator-hardcoded coordinate (zero discovery).
        let is_k8s = matches!(
            std::env::var(crate::runner_protocol::cell_launcher::CELL_LAUNCHER_ENV).as_deref(),
            Ok("k8s")
        );
        let velo = build_velo(controller_velo_bind(is_k8s, &temp_root))
            .await
            .context("building controller velo")?;
        // The run-wide synchronized-START event: cells await it after registering,
        // and the controller triggers it once every cell has registered so they all
        // begin dispatching together. Created before `velo` moves into the transport;
        // held here so a bail (before trigger) drop-poisons it and unblocks every
        // waiting cell with an error rather than a hang.
        let start_event = velo
            .event_manager()
            .new_event()
            .context("creating cellular start event")?;
        let start_handle = start_event.handle();
        let (serve_source, cell_coordinate) = controller_bootstrap(is_k8s, &temp_root)?;
        let _bootstrap = serve_bootstrap(&serve_source, &velo.peer_info())
            .await
            .context("serving controller bootstrap PeerInfo")?;

        // Each phase's global dispatch base (turns dispatched by prior phases): a
        // cell's sampler restarts each phase, so the cell adds this to its phase-local
        // slot to stamp the single-cell absolute slot. Same for every cell. Graph cells
        // partition by trace and never read it, so the graph kind returns an empty map.
        let phase_ordinal_bases = kind.phase_ordinal_bases(envelope)?;

        // Precompute each cell's sliced execute envelope; the register handler serves
        // it as that cell's spec (replacing the stdin pipe).
        let mut specs: Vec<Vec<u8>> = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let cell_dir = temp_root.join(format!("cell-{cell_id}"));
            std::fs::create_dir_all(&cell_dir)
                .with_context(|| format!("creating cell {cell_id} artifact dir"))?;
            let cell_envelope =
                build_cell_envelope(envelope, cell_id, cell_count, &cell_dir, injected_seed)?;
            specs.push(
                serde_json::to_vec(&cell_envelope)
                    .with_context(|| format!("serializing cell {cell_id} envelope"))?,
            );
        }
        let specs = std::sync::Arc::new(specs);
        let spec_for: SpecFor = {
            let specs = specs.clone();
            std::sync::Arc::new(move |cell_id: u32| specs.get(cell_id as usize).cloned())
        };
        let mut transport =
            VeloControllerTransport::bind_controller(velo, spec_for, cell_count, start_handle)
                .context("binding controller transport")?;

        // Launch (local subprocesses) or expect (k8s pods) the cells.
        let launch_ctx = CellLaunchContext {
            cell_count,
            controller_coordinate: cell_coordinate,
            phase_ordinal_bases,
        };
        let mut handles = select_launcher()
            .launch(&launch_ctx)
            .context("launching cells")?;

        // Watch each cell for a hard failure and forward it, so a cell that dies
        // BEFORE registering aborts the run rather than hanging the collect. A local
        // child exit resolves; a k8s handle never resolves (the collect deadline
        // below is the k8s backstop).
        let (failure_tx, mut failure_rx) =
            tokio::sync::mpsc::channel::<String>(cell_count.max(1) as usize);
        for mut handle in handles.drain(..) {
            let failure_tx = failure_tx.clone();
            tokio::spawn(async move {
                let report = handle.wait_failure().await;
                let _ = failure_tx.send(report).await;
            });
        }
        drop(failure_tx);

        // Synchronized start: wait for every cell to register (bounded — cells only
        // fetch their envelope, no work yet), then trigger the START event so all
        // cells begin dispatching together. A cell that dies before registering, or a
        // registration timeout, aborts the run (dropping `start_event` poisons it, so
        // any already-waiting cell unblocks with an error rather than hanging).
        tokio::select! {
            biased;
            () = transport.await_all_registered() => {}
            Some(failure) = failure_rx.recv() => bail!("{failure}"),
            () = tokio::time::sleep(register_timeout()) => {
                bail!("cells did not all register within the registration timeout")
            }
        }
        start_event
            .trigger()
            .context("triggering cellular benchmark start")?;

        // Collect exactly one partition per cell (plus the latest heartbeat), with a
        // generous deadline so a cell that never ships (a k8s pod with no child to
        // watch) aborts loudly instead of hanging forever. The `select!` is `biased`,
        // so a ready cell message is taken before a cell-exit failure — the
        // ship-then-exit race resolves in the cell's favour when both land together.
        let deadline = tokio::time::sleep(collect_timeout());
        tokio::pin!(deadline);
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
                _ = &mut deadline => bail!(
                    "cellular run timed out with {} of {cell_count} cell partitions",
                    partitions.len()
                ),
            }
        }

        // Records-first merge → the single report. Scheduled cells pre-tile a global
        // dispatch ordinal (merged in that order); graph records carry a LOCAL per-cell
        // request_index (wall-clock start order), concatenated by cell_id and re-numbered
        // densely — deterministic-per-topology. The kind selects between the two.
        let merged = kind.merge(metrics_config, partitions)?;
        let record_count = merged.record_count();
        let summary =
            merged.export_results(&ExportContext::phase(crate::metrics_core::Phase::Profiling));
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
            merged.export_results(&ExportContext::phase(crate::metrics_core::Phase::Warmup));
        let outcome = crate::metrics_core::report::RunOutcome {
            run: crate::metrics_core::report::ReportRunInfo {
                mode: Some("online".to_owned()),
                model: cellular_model_name(envelope),
            },
            summary: crate::metrics_core::report::ReportSummary {
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

        // Run the native export plane on the merged report, exactly as the
        // single-process path does in coordinator::persist_prepared_report. The
        // exporter-plane cutover made the runner the sole emitter of the user-facing
        // sink outputs (genai-perf-v1 aiperf.json/CSV, console.txt, timeslice, OTLP,
        // MLflow, W&B); native-v2.json above is the committed authority but is NOT one
        // of them. Without this a cellular run would emit only native-v2.json and the
        // frontend would find no aiperf.json. Best-effort by contract: run_exporters
        // logs per-sink and never fails the run.
        if let Some(artifact_dir) = report_path.parent() {
            exporters.run(&report, artifact_dir, &cellular_export_config(envelope));
        }

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

/// The controller's velo messaging bind. k8s binds an ephemeral routable TCP port
/// (its address is advertised to cells via the bootstrap `PeerInfo`); a co-located
/// launcher binds UDS on unix (a lower-overhead local socket) or loopback elsewhere.
#[cfg(feature = "velo")]
fn controller_velo_bind(is_k8s: bool, temp_root: &Path) -> BindSpec {
    if is_k8s {
        return BindSpec::TcpBind("0.0.0.0:0".parse().expect("valid ephemeral bind addr"));
    }
    #[cfg(unix)]
    {
        BindSpec::UdsPath(temp_root.join("controller.sock"))
    }
    #[cfg(not(unix))]
    {
        let _ = temp_root;
        BindSpec::TcpLoopback
    }
}

/// The controller's bootstrap publication and the coordinate cells fetch it from.
/// - **k8s**: serve the `PeerInfo` on the operator-hardcoded bootstrap bind
///   (`AIPERF_CONTROLLER_BOOTSTRAP_BIND`, default `0.0.0.0:9500`); the cell fetch
///   coordinate is supplied to the pods by the operator (`AIPERF_CELL_CONTROLLER_ADDR`),
///   so the launcher's copy is unused here (empty).
/// - **local**: write the `PeerInfo` to a file in the scratch tree; cells (same host)
///   read it via a `file:` coordinate the local launcher injects.
#[cfg(feature = "velo")]
fn controller_bootstrap(is_k8s: bool, temp_root: &Path) -> Result<(BootstrapSource, String)> {
    if is_k8s {
        let bind = std::env::var("AIPERF_CONTROLLER_BOOTSTRAP_BIND")
            .unwrap_or_else(|_| "0.0.0.0:9500".to_owned());
        Ok((BootstrapSource::Tcp(bind), String::new()))
    } else {
        let path = temp_root.join("controller-peer.rmp");
        let coordinate = format!("file:{}", path.display());
        Ok((BootstrapSource::File(path), coordinate))
    }
}

/// The deadline for collecting every cell's partition. Covers the whole run (cells
/// execute the benchmark before shipping), so it is generous and env-overridable
/// (`AIPERF_CELL_COLLECT_TIMEOUT_SECS`, default 2 hours). Primarily a k8s backstop —
/// a local run's per-child exit watcher catches a dead cell far sooner.
#[cfg(feature = "velo")]
fn collect_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_COLLECT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(7200);
    std::time::Duration::from_secs(secs)
}

/// The deadline for every cell to REGISTER (before the synchronized start). Cells
/// only fetch their envelope here — no benchmark work yet — so this is a short
/// startup bound (env `AIPERF_CELL_REGISTER_TIMEOUT_SECS`, default 5 minutes),
/// unlike [`collect_timeout`] which must span the whole run.
#[cfg(feature = "velo")]
fn register_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_REGISTER_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    std::time::Duration::from_secs(secs)
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
    injected_seed: Option<u64>,
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
    // Share the controller-derived seed with every cell when the author gave none, so
    // all cells compose the identical dataset space (the same value goes to each cell).
    if let Some(seed) = injected_seed {
        run.insert("random_seed".to_owned(), serde_json::Value::from(seed));
    }
    // The cell runs single-process (its slice); its autonomous behaviour comes from
    // the env vars the controller sets, not from re-entering the controller path.
    if let Some(runtime) = run
        .get_mut("cfg")
        .and_then(|cfg| cfg.get_mut("runtime"))
        .and_then(serde_json::Value::as_object_mut)
    {
        runtime.insert("cells".to_owned(), serde_json::Value::from(1));
        // Divide the thread-per-core worker count uniformly across the cells so N cell
        // processes on one host target ~`workers` total threads, not N×`workers`
        // (core over-subscription). UNIFORM integer division is required: the two-level
        // (cell × thread) partition `(c + cells*t, cells*W)` assumes every cell has the
        // same W, so round-robin (`owned_positions`) is wrong here — the remainder
        // (`workers % cell_count`) is dropped, `.max(1)` keeps every cell threaded.
        // Applies to both kinds: scheduled reads W in the sharded runtime, graph as its
        // thread-per-core `worker_count`.
        if let Some(workers) = runtime.get("workers").and_then(serde_json::Value::as_u64) {
            let per_cell = (workers / u64::from(cell_count)).max(1);
            runtime.insert("workers".to_owned(), serde_json::Value::from(per_cell));
        }
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

/// Whitelists a cellular run to the shape the cell topology is currently *wired* for:
/// the shared online-scheduled executor over the `http` transport, on **synthetic,
/// single-turn** datasets. There is no bespoke HTTP layer — a cell runs the same
/// `execute.rs` path as any single-process run, differing only by an injected
/// [`IssuanceAuthority`] and an env-gated records sink (`CellRecordsShipper`). The
/// partition/issuance seam is transport-neutral; this whitelist reflects wiring
/// coverage, not an HTTP special case. Two invariants underpin byte parity and each
/// fails closed here:
///
/// - **Only the online-scheduled executor ships a partition today.** The gRPC, graph,
///   and offline executors are separate paths that do not yet inject the cell issuer
///   or ship records, so a `grpc`/`dynosim` transport or a non-synthetic
///   (`file`/`public`, incl. graph-program) dataset would run an unwired executor and
///   hang the controller. Threading the same seam through those executors is the
///   natural extension; until then they are rejected, not silently divergent.
/// - **One sampler draw must equal one dispatched turn.** [`PartitionedSampler`]
///   partitions by conversation *draw*, but the issuer stamps a per-*turn* ordinal
///   ([`CellularAutonomousIssuer`]); a multi-turn conversation makes the two diverge,
///   so the merged report silently reorders (or, for variable turn counts, draws a
///   different instance set). Only `turns == 1` (the default) is sound.
///
/// [`PartitionedSampler`]: crate::dataset::sampler
/// [`CellularAutonomousIssuer`]: crate::cellular::CellularAutonomousIssuer
fn validate_cellular_run_shape(envelope: &serde_json::Value) -> Result<()> {
    if let Some(transport) = envelope
        .pointer("/run/cfg/transport/type")
        .and_then(serde_json::Value::as_str)
    {
        ensure!(
            transport == "http",
            "cellular is currently wired only for transport.type=\"http\"; got {transport:?}. \
             A cell reuses the shared online-scheduled executor + hyper transport (not a \
             bespoke HTTP layer); the partition/issuance seam is transport-neutral, but \
             records-shipping is only threaded through the HTTP path so far"
        );
    }
    let datasets = envelope
        .pointer("/run/cfg/datasets")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no datasets array")?;
    for dataset in datasets {
        // Graph programs (dag_jsonl / weka_trace / dynamo_trace) partition by whole
        // trace via PartitionedGraphTraceSource, so they bypass the scheduled
        // synthetic/single-turn requirement.
        if matches!(
            dataset.get("format").and_then(serde_json::Value::as_str),
            Some("dag_jsonl" | "weka_trace" | "dynamo_trace")
        ) {
            continue;
        }
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
    // A run seed is no longer *required*: every cell must compose the SAME dataset
    // space, but when the author gives no `run.random_seed` the controller derives one
    // shared seed and injects it into every cell envelope (see [`resolve_cellular_seed`]
    // / [`build_cell_envelope`]) — coherent partition without forcing the flag.
    //
    // Multiple endpoint URLs are likewise allowed, not rejected: cells round-robin the
    // URL pool in cell-local order, so the exact per-request URL assignment differs from
    // a 1-cell run, but the aggregate load across the pool matches — an intentional
    // aggregate-equivalent approximation (see [`warn_cellular_approximations`]), in the
    // same family as rate pacing and cancellation. Hitting a backend pool from N nodes
    // is a first-class multi-node workload, so byte-parity does not gate it.
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
/// - carries a `duration`/`sessions`/`adaptive_scale` bound that can stop it early (a
///   duration bound needs the ragged-count merge that the graph-mode cellular path
///   provides; `adaptive_scale` needs cross-cell scaling consensus — both future work); or
/// - has a `concurrency`/`prefill_concurrency` cap below `cell_count` — the `.max(1)`
///   per-cell floor would then over-subscribe the aggregate in-flight to `cell_count`.
///
/// Fail closed rather than silently corrupt. The static `concurrency`/
/// `prefill_concurrency`/`rate` caps at or above `cell_count` ARE sliced per cell (see
/// [`build_cell_envelope`]). Several supported knobs are only *aggregate-equivalent* to a
/// 1-cell run, not byte-identical (warned, not rejected — the cellular bargain trades
/// exact reproducibility for multi-node scale): rate-based phases match the aggregate
/// offered rate but not the per-turn arrival sample path; a post-send `cancellation`
/// policy matches the aggregate rate but not the exact cancelled subset; and
/// concurrency/prefill/rate **ramps** ramp each cell to its sliced target so the
/// aggregate reaches the full authored target but starts near `cell_count` rather than 1.
/// Byte-parity is exact only for a seeded `concurrency` phase with none of these.
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
        // Concurrency/prefill/rate ramps are allowed, not rejected. A `RampSpec` is only
        // `{duration, strategy}`: it ramps *to* the phase's `concurrency`/`rate` target,
        // which build_cell_envelope already slices per cell. So each cell ramps to its
        // sliced target over the same duration and the aggregate ramps to the full
        // authored target — aggregate-equivalent, not byte-identical (the aggregate
        // starts near `cell_count` rather than 1, since every cell's ramp starts at 1).
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

/// Rejects a cellular *graph* run whose phases cannot partition cleanly across cells.
///
/// A graph cell is partitioned by [`PartitionedGraphTraceSource`] (selected in
/// `graph_phase_runtime.rs`), which slices the **SESSION** space — the trace instances
/// bounded by `sessions` / `--num-conversations` or a duration — round-robin across
/// cells. That source is chosen ONLY when the phase carries no `requests` budget. A
/// phase that instead carries a static-node `requests` budget falls back to the
/// single-cell [`CyclingGraphTraceSource`], so *every* cell replays the FULL
/// un-partitioned cycle (N× load and N× records, or an overlapping low-index-biased
/// trace subset) — a silent mis-partition the completeness check would accept. Until
/// the request-budget partition is built, fail closed: a graph phase must bound its
/// work by the session space (or a duration), never by `requests`.
///
/// The session/prefill concurrency caps are still split round-robin per cell, so the
/// same `>= cell_count` floor as the scheduled path applies (below it every cell floors
/// to 1 and the aggregate over-subscribes to `cell_count`).
///
/// [`PartitionedGraphTraceSource`]: crate::graph::workload::PartitionedGraphTraceSource
/// [`CyclingGraphTraceSource`]: crate::graph::workload::CyclingGraphTraceSource
fn validate_graph_cellular_phases(envelope: &serde_json::Value, cell_count: u32) -> Result<()> {
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
        ensure!(
            phase.get("requests").is_none(),
            "cellular graph runs do not yet partition a static-node `requests` budget (phase {name:?}); \
             use `sessions` / `--num-conversations` or a duration bound so the trace instances partition across cells"
        );
        // Session/prefill concurrency caps split round-robin per cell; below cell_count
        // they floor to 1 and over-subscribe the aggregate, so require >= cell_count.
        for cap in ["concurrency", "prefill_concurrency"] {
            if let Some(value) = phase.get(cap).and_then(serde_json::Value::as_u64) {
                ensure!(
                    value >= cells,
                    "cellular graph runs require a `{cap}` cap >= cell_count ({cell_count}) so it splits evenly; phase {name:?} has {value}"
                );
            }
        }
    }
    Ok(())
}

/// The native metrics policy for the merge, derived from the v2 envelope exactly as
/// the single-process path does — `cfg.metrics` (SLOs + slice duration) plus
/// `cfg.endpoint.use_server_token_count`. Passing `MetricsConfig::default()` would
/// silently drop authored goodput SLOs and timeslice sweep-lines from the merged
/// report. Mirrors [`crate::runner_protocol::protocol::BenchmarkRunConfigWireV2`]'s
/// `from_value(cfg.metrics).unwrap_or_default()` so an absent/loose `metrics` block
/// falls back the same way (`metrics_config` still validates any SLO names present).
fn cellular_metrics_config(envelope: &serde_json::Value) -> Result<MetricsConfig> {
    let spec: crate::runner_protocol::protocol::MetricsSpec = envelope
        .pointer("/run/cfg/metrics")
        .cloned()
        .map(|value| serde_json::from_value(value).unwrap_or_default())
        .unwrap_or_default();
    let use_server_token_count = envelope
        .pointer("/run/cfg/endpoint/use_server_token_count")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    crate::runner_protocol::execute::metrics_config(&spec, use_server_token_count)
}

/// Warns, once at controller startup, when a cellular run carries side-channel
/// telemetry sidecars (`server_metrics` / `gpu_telemetry` / `network_latency`). Each
/// cell scrapes them into its own scratch tree, which the controller discards — so
/// these metrics are omitted from the merged report (the documented report-fidelity
/// gap), unlike a single-process run. This is surfaced as a loud runtime warning
/// rather than a silent drop or a fail-closed rejection: `gpu_telemetry` and
/// `server_metrics` default *on*, so rejecting any present sidecar would refuse nearly
/// every cellular run. Cross-cell sidecar aggregation is future wiring.
fn warn_dropped_sidecar_telemetry(envelope: &serde_json::Value) {
    const SIDECARS: [&str; 3] = ["server_metrics", "gpu_telemetry", "network_latency"];
    let is_active = |value: Option<&serde_json::Value>| match value {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Object(map)) => !map.is_empty(),
        Some(_) => true,
    };
    let present: Vec<&str> = SIDECARS
        .into_iter()
        .filter(|key| {
            is_active(envelope.pointer(&format!("/run/cfg/sidecars/{key}")))
                || is_active(envelope.pointer(&format!("/run/cfg/{key}")))
        })
        .collect();
    if !present.is_empty() {
        tracing::warn!(
            sidecars = present.join(","),
            "cellular mode does not aggregate side-channel telemetry across cells; \
             these sidecar metrics are omitted from the merged report (a single-process \
             run emits them). Run without --cells to collect them."
        );
    }
}

/// One shared run seed for every cell when the author gave none, or `None` when
/// `run.random_seed` is already set (cells then inherit it verbatim). Derived
/// deterministically from the run identity so every cell composes the *identical*
/// dataset space (the coherence a shared seed provides), reproducible per `benchmark_id`
/// — a strictly friendlier default than rejecting a seedless `--cells` run.
fn resolve_cellular_seed(envelope: &serde_json::Value) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    if envelope
        .pointer("/run/random_seed")
        .and_then(serde_json::Value::as_u64)
        .is_some()
    {
        return None;
    }
    let identity = envelope
        .pointer("/run/benchmark_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("aiperf-cellular");
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    identity.hash(&mut hasher);
    Some(hasher.finish())
}

/// Warns, once at controller startup, about the aggregate-equivalent (not
/// byte-identical) knobs a cellular run carries, so divergence from a 1-cell run is
/// loud rather than silent: multiple endpoint URLs (round-robined cell-locally),
/// concurrency/prefill/rate ramps (each cell ramps to its sliced target, so the
/// aggregate starts near `cell_count` not 1), and an auto-derived shared seed when the
/// author gave none. These are allowed by design — the cellular bargain trades exact
/// reproducibility for multi-node scale — but the operator should know.
fn warn_cellular_approximations(envelope: &serde_json::Value) {
    let url_count = envelope
        .pointer("/run/cfg/endpoint/urls")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if url_count > 1 {
        tracing::warn!(
            urls = url_count,
            "cellular round-robins multiple endpoint URLs in cell-local order; aggregate \
             load across the pool matches a 1-cell run but the exact per-request URL \
             assignment differs (aggregate-equivalent, not byte-identical)"
        );
    }
    let has_ramp = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|phases| {
            phases.iter().any(|phase| {
                ["concurrency_ramp", "prefill_ramp", "rate_ramp"]
                    .iter()
                    .any(|key| phase.get(key).is_some_and(|value| !value.is_null()))
            })
        });
    if has_ramp {
        tracing::warn!(
            "cellular slices ramp targets per cell; the aggregate ramps to the full \
             authored target but starts near cell_count rather than 1 \
             (aggregate-equivalent, not byte-identical)"
        );
    }
    if resolve_cellular_seed(envelope).is_some() {
        tracing::warn!(
            "cellular run has no run.random_seed; the controller derived one shared seed \
             from the run identity so all cells compose the same dataset space \
             (reproducible per benchmark_id)"
        );
    }
}

/// The native export policy for the merged report, parsed from the envelope's
/// `cfg.export` exactly as the single-process path does
/// (`protocol_v2::RunConfig::export`), defaulting when absent so the cellular run
/// emits the same user-facing sink outputs (genai-perf-v1 JSON/CSV, console.txt, …).
fn cellular_export_config(envelope: &serde_json::Value) -> crate::export::ExportConfig {
    envelope
        .pointer("/run/cfg/export")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
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
        // Supported: http (or default) transport, synthetic single-turn dataset. A run
        // seed and a single endpoint URL are preferred but no longer required — a
        // seedless run auto-derives one shared seed and multiple URLs round-robin
        // cell-locally (both aggregate-equivalent, warned not rejected).
        for ok in [
            // No run seed — allowed (controller derives a shared seed).
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}}),
            // Multiple endpoint URLs — allowed (cell-local round-robin).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "endpoint": {"urls": ["http://a", "http://b"]}}}}),
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
            // A `file` dataset in a NON-graph trace format is still rejected; the graph
            // formats (dag_jsonl/weka_trace/dynamo_trace) are admitted instead — see
            // admits_graph_datasets_but_still_rejects_linear_non_synthetic.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "mooncake_trace"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "public"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"value": 3}}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"mean": 2.0, "stddev": 1.0}}]}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&bad).is_err(),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn admits_graph_datasets_but_still_rejects_linear_non_synthetic() {
        // Graph programs (dag_jsonl / weka_trace / dynamo_trace) partition by whole
        // trace via PartitionedGraphTraceSource, so validate_cellular_run_shape admits
        // them past the synthetic / single-turn guards even though they are `file`
        // datasets. The http transport ensure is unchanged (graph dispatches HTTP too).
        for graph_format in ["dag_jsonl", "weka_trace", "dynamo_trace"] {
            let graph = serde_json::json!({"run": {"cfg": {
                "transport": {"type": "http"},
                "datasets": [{"type": "file", "format": graph_format}],
            }}});
            assert!(
                validate_cellular_run_shape(&graph).is_ok(),
                "should admit graph dataset {graph_format}"
            );
            assert!(
                is_graph_dataset(&graph),
                "is_graph_dataset true for {graph_format}"
            );
        }
        // A NON-graph linear trace format (mooncake_trace is not one of the three graph
        // formats) is still a `file` non-synthetic dataset and must stay rejected — the
        // graph bypass is scoped to exactly the three graph formats.
        let linear = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "file", "format": "mooncake_trace"}],
        }}});
        assert!(
            validate_cellular_run_shape(&linear).is_err(),
            "should still reject non-graph linear trace"
        );
        assert!(
            !is_graph_dataset(&linear),
            "mooncake_trace is not a graph dataset"
        );
        // is_graph_dataset is false for a synthetic (scheduled) dataset.
        let synthetic = serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}});
        assert!(
            !is_graph_dataset(&synthetic),
            "synthetic is not a graph dataset"
        );
    }

    #[test]
    fn run_kind_detects_and_dispatches() {
        // detect: a graph-format dataset is the Graph kind; anything else is Scheduled.
        let graph_env = serde_json::json!(
            {"run": {"cfg": {"datasets": [{"type": "file", "format": "dag_jsonl"}]}}}
        );
        let synthetic_env =
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}});
        assert_eq!(CellularRunKind::detect(&graph_env), CellularRunKind::Graph);
        assert_eq!(
            CellularRunKind::detect(&synthetic_env),
            CellularRunKind::Scheduled
        );

        // validate_phases (graph): a sessions-bounded phase with a concurrency cap
        // >= cell_count and no `requests` passes; a phase carrying `requests` is rejected.
        let graph_ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "profiling", "sessions": 100, "concurrency": 8},
        ]}}});
        assert!(CellularRunKind::Graph.validate_phases(&graph_ok, 4).is_ok());
        let graph_requests = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8},
        ]}}});
        assert!(
            CellularRunKind::Graph
                .validate_phases(&graph_requests, 4)
                .is_err()
        );

        // phase_ordinal_bases: the scheduled kind computes a per-phase base map from a
        // request-bounded envelope; the graph kind always returns an empty map.
        let two_phase = serde_json::json!({"run": {"cfg": {"phases": [
            {"name": "warmup", "requests": 10},
            {"name": "profiling", "requests": 100},
        ]}}});
        assert!(
            !CellularRunKind::Scheduled
                .phase_ordinal_bases(&two_phase)
                .unwrap()
                .is_empty()
        );
        assert!(
            CellularRunKind::Graph
                .phase_ordinal_bases(&two_phase)
                .unwrap()
                .is_empty()
        );
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
        // pass; a post-send cancellation policy AND concurrency/rate ramps are allowed
        // (aggregate-equivalent approximations, warned not rejected).
        let ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "warmup", "requests": 10, "concurrency": 8},
            {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8, "cancellation": {"rate": 25.0, "delay": 0.5}, "concurrency_ramp": {"start": 1, "end": 100}},
        ]}}});
        assert!(validate_cellular_phase_budgets(&ok, 4).is_ok());
        // Fail closed (cell_count 4) on: a trace-driven or missing phase type; requests
        // absent or below cell_count; a duration/sessions/adaptive_scale bound; or a
        // concurrency/prefill cap below cell_count. (Ramps and cancellation are allowed.)
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
    fn rejects_graph_requests_budget_but_allows_sessions() {
        // PartitionedGraphTraceSource partitions the SESSION space (sessions /
        // --num-conversations / a duration), not a static-node `requests` budget: a
        // `requests` phase falls back to the single-cell CyclingGraphTraceSource and
        // every cell replays the full un-partitioned cycle. Fail closed on `requests`
        // (and on a concurrency cap below cell_count), accept a session/duration bound.
        //
        // REJECT: a graph phase carrying a static-node `requests` budget.
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8},
                ]}}}),
                4
            )
            .is_err()
        );
        // REJECT: a concurrency cap below cell_count (floors to 1 per cell, over-subscribes).
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "sessions": 100, "concurrency": 2},
                ]}}}),
                4
            )
            .is_err()
        );
        // ACCEPT: a `sessions`-bounded phase with a concurrency cap >= cell_count and no `requests`.
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "sessions": 100, "concurrency": 8},
                ]}}}),
                4
            )
            .is_ok()
        );
        // ACCEPT: a `duration`-bounded phase with no `requests` budget.
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "duration": 30.0, "concurrency": 8},
                ]}}}),
                4
            )
            .is_ok()
        );
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
                    let cell = build_cell_envelope(&envelope, cell_id, count, dir, None).unwrap();
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
            let cell = build_cell_envelope(&envelope, cell_id, cell_count, dir, None).unwrap();
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
}
