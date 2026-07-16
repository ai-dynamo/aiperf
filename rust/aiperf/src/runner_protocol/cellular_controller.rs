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
    ColumnStorePartition, MetricsHeartbeat, RecordsShardPartition, TDigest,
    merge_records_by_concatenation, merge_records_in_global_order, merge_store_partitions,
};
use crate::metrics_core::report::NativeReport;
use crate::metrics_core::{ExportContext, MetricsAccumulator, MetricsConfig, PERCENTILES};
use anyhow::{Context, Result, bail, ensure};

use crate::runner_protocol::cell_launcher::owned_positions;

// The velo transport + launcher wiring is the only part of the controller that
// needs the `velo` feature; the validation, budget-slicing, merge, and report
// assembly below are plain envelope/metric logic reused by the non-velo build.
#[cfg(feature = "velo")]
use crate::cellular::transport::connect::{BindSpec, build_velo};
#[cfg(feature = "velo")]
use crate::cellular::{CellMessage, ControllerTransport, SpecFor, VeloControllerTransport};
#[cfg(feature = "velo")]
use crate::runner_protocol::cell_launcher::{CellLaunchContext, select_launcher};

/// Env toggle (tier T3) for the master-less, barrier-free start: the controller
/// triggers START immediately instead of gathering all N cell registrations first
/// (the O(N) fan-in rendezvous). Default off (the tight synchronized start). Cells
/// registering after the trigger see the completed event instantly (velo's
/// completed-event cache), so each starts on its own registration.
pub const CELL_BARRIER_FREE_ENV: &str = "AIPERF_CELL_BARRIER_FREE";

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
    // Same-host (local launcher) cells write their per-record artifacts into
    // controller-local `temp_root/cell-{id}` dirs, which the controller concatenates into
    // the real artifact dir at finalize (Stage D). A cross-host (k8s) pod writes to its
    // own filesystem, so those files stay unreachable by the controller — still dropped.
    let is_k8s = matches!(
        std::env::var(crate::runner_protocol::cell_launcher::CELL_LAUNCHER_ENV).as_deref(),
        Ok("k8s")
    );
    // Tier-T3 master-less start: skip the O(N) register rendezvous (see the start
    // policy below). Default off (the tight synchronized start).
    let barrier_free = matches!(
        std::env::var(CELL_BARRIER_FREE_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    // The run's artifact spec, parsed once (the same `AuthoredRunSpecV2` shape the
    // cell's execute path reads), for the Stage E shipping decision and the Stage D
    // concat below.
    let artifacts: crate::runner_protocol::protocol::ArtifactSpec = envelope
        .pointer("/run/cfg/artifacts")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    // Stage E (reopened): a CROSS-HOST (k8s) run whose cells write per-record
    // artifacts to their own pod filesystems now ships those files to the controller
    // over HTTP + streaming zstd, so the controller runs the SAME Stage D concat on
    // `temp_root/cell-{id}`. Same-host (`--cells N`) keeps the shared-FS concat (no
    // HTTP). Gated on the operator toggle and on the run actually requesting
    // shippable files (per-record artifacts or inputs.json); a metrics-only run ships
    // nothing.
    // The test/dev HTTP-force seam ([`CELL_ARTIFACT_HTTP_FORCE_ENV`]) drives the
    // cross-host HTTP artifact path over loopback for a SAME-HOST run so a
    // multi-process test can prove the shipping mechanism end-to-end. Off by default,
    // so a normal `--cells N` run keeps the shared-FS Stage D concat unchanged.
    let force_http = crate::runner_protocol::cellular_cell::artifact_http_force_enabled();
    let http_shipping = (is_k8s || force_http)
        && crate::runner_protocol::cellular_cell::http_artifact_shipping_enabled()
        && !crate::runner_protocol::artifact_shipping::shippable_relatives(&artifacts).is_empty();
    // Stage G: a cross-host cell cannot read a controller-local `file`/`path` dataset
    // source, so the controller serves it over the SAME HTTP+zstd plane and the cell
    // recompiles it locally. Only a cross-host (k8s / force) run with a `file`/`path`
    // dataset and HTTP shipping enabled needs the serve; same-host cells read the
    // controller-local path directly, and synthetic/inline-records/public need no serve.
    let dataset_source =
        crate::runner_protocol::cellular_cell::cellular_file_dataset_path(envelope);
    let dataset_ship = (is_k8s || force_http)
        && crate::runner_protocol::cellular_cell::http_artifact_shipping_enabled()
        && dataset_source.is_some();
    // The controller's artifact HTTP server carries BOTH per-record uploads (Stage E)
    // and dataset serving (Stage G); stand it up when either is needed.
    let need_artifact_server = http_shipping || dataset_ship;
    // The force seam only applies to the same-host launcher (k8s already ships): when
    // true, cells write to their own controller-local `temp_root/cell-{id}` scratch AND
    // ship those files over HTTP to a SEPARATE loopback landing dir, from which the
    // concat reads — so the shipped bytes (not the local writes) feed the merged report.
    // Dataset serving reuses the same loopback bind + injected authority.
    let force_local_http = need_artifact_server && !is_k8s;
    warn_dropped_sidecar_telemetry(envelope);
    // Warn about DROPPED per-record artifacts only when they genuinely cannot be
    // delivered — cross-host AND HTTP shipping disabled. When HTTP shipping is active
    // the files ARE collected, so the boundary warning would be misleading.
    warn_dropped_per_record_artifacts(envelope, is_k8s && !http_shipping);
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

        // Stage E: start the artifact upload server BEFORE launching cells (a k8s pod
        // may start and upload before the controller's collect loop). Cells POST their
        // per-record artifact files here with streaming zstd; each file lands at
        // `temp_root/cell-{id}/{rel}` — exactly where the Stage D concat reads. The
        // allowlist is the run's shippable relative paths, so a cell can only land
        // known artifacts inside its own cell dir.
        // Where uploaded artifact files land (`landing_root/cell-{id}/{rel}`). k8s
        // lands directly in `temp_root/cell-{id}` (the concat's own dirs — the cell fs
        // is a different host, no collision). The same-host force seam MUST land in a
        // SEPARATE subtree, because there the cell's own artifact_dir already IS
        // `temp_root/cell-{id}`; landing there would overwrite each file with itself.
        let landing_root = if force_local_http {
            temp_root.join("http-landing")
        } else {
            temp_root.clone()
        };
        // The same-host force seam binds the upload server on an ephemeral loopback
        // port and injects that concrete authority into each locally-launched cell
        // (below). k8s uses the fixed operator-exposed routable bind and the cells
        // derive the authority from their `tcp://` velo coordinate instead.
        let artifact_bind = if force_local_http {
            std::net::SocketAddr::from(([127, 0, 0, 1], 0))
        } else {
            controller_artifact_bind()
        };
        let artifact_server = if need_artifact_server {
            // Upload allowlist: the run's per-record artifact relatives when Stage E
            // shipping is active, else empty (a dataset-serve-only run accepts no
            // uploads, so every POST is rejected).
            let allowed: std::collections::HashSet<String> = if http_shipping {
                crate::runner_protocol::artifact_shipping::shippable_relatives(&artifacts)
                    .into_iter()
                    .collect()
            } else {
                std::collections::HashSet::new()
            };
            // Dataset serve allowlist (Stage G): the run's single `file`/`path` source,
            // keyed by its file name (the name a cell requests). Empty otherwise.
            let datasets: std::collections::HashMap<String, PathBuf> = match dataset_source
                .as_ref()
                .filter(|_| dataset_ship)
            {
                Some(path) => {
                    let name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .context("cellular file dataset path has no file name")?
                        .to_owned();
                    std::iter::once((name, path.clone())).collect()
                }
                None => std::collections::HashMap::new(),
            };
            Some(
                crate::runner_protocol::artifact_shipping::ArtifactUploadServer::start_with_datasets(
                    artifact_bind,
                    landing_root.clone(),
                    allowed,
                    datasets,
                )
                .await
                .context("starting cellular artifact server")?,
            )
        } else {
            None
        };

        // Bind the controller's velo transport at a known endpoint cells `connect`
        // to (zero discovery — velo's `_hello` handshake resolves identity on dial).
        // `is_k8s` is resolved once above and moved in here.
        let (bind, cell_coordinate) = controller_bind_and_endpoint(is_k8s, &temp_root)?;
        let velo = build_velo(bind).await.context("building controller velo")?;
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

        // Tier-T2 hierarchical merge: insert `M = ceil(cells / fanout)` aggregators
        // between the cells and the controller (local only — k8s aggregator placement is
        // the operator's concern, a follow-on). Each cell ships to its round-robin
        // aggregator; each aggregator merges its subtree and ships ONE store up, so the
        // controller collects `M` partitions instead of `cells`. Fold-only (sketch /
        // exact-fold): the retain path keeps the star topology (needs global order).
        let aggregator_count =
            crate::runner_protocol::cellular_aggregator::aggregator_count(cell_count);
        if aggregator_count.is_some() && is_k8s {
            tracing::warn!(
                "AIPERF_CELL_AGG_FANOUT is set but k8s aggregator placement is not yet wired; \
                 falling back to the flat star topology"
            );
        }
        let aggregator_count = if is_k8s { None } else { aggregator_count };
        let aggregator_base_port =
            crate::runner_protocol::cellular_aggregator::aggregator_base_port();
        // The controller collects one partition per aggregator (tree) or per cell (flat).
        let expected_partitions = aggregator_count.unwrap_or(cell_count);
        // Spawn the aggregator subprocesses before the cells so they are bound and
        // collecting by the time cells ship (cell `connect` also retries). Each gets the
        // run envelope on stdin (for the merge config) and its subtree parameters via env.
        let mut aggregator_children = if let Some(agg_count) = aggregator_count {
            spawn_aggregators(
                envelope,
                agg_count,
                cell_count,
                aggregator_base_port,
                &cell_coordinate,
            )
            .await
            .context("spawning tier-T2 aggregators")?
        } else {
            Vec::new()
        };

        // Launch (local subprocesses) or expect (k8s pods) the cells.
        let launch_ctx = CellLaunchContext {
            cell_count,
            controller_coordinate: cell_coordinate,
            phase_ordinal_bases,
            aggregator_count,
            aggregator_base_port,
            // k8s pods derive the artifact authority from their operator-injected
            // `tcp://` controller coordinate + artifact port (the controller cannot
            // know its own routable host), so nothing is injected there. The same-host
            // HTTP-force seam DOES know its own loopback address, so it injects the
            // bound server authority into each local cell (there is no `tcp://`
            // coordinate for a same-host cell to derive it from).
            artifact_authority: if force_local_http {
                artifact_server.as_ref().map(|server| server.local_addr().to_string())
            } else {
                None
            },
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
        // Watch each aggregator (tier T2) for a hard failure the same way, so a dead
        // aggregator aborts the run rather than hanging the controller's collect on a
        // subtree partition that will never arrive.
        for (agg_id, mut child) in aggregator_children.drain(..).enumerate() {
            let failure_tx = failure_tx.clone();
            tokio::spawn(async move {
                if let Ok(status) = child.wait().await
                    && !status.success()
                {
                    let _ = failure_tx
                        .send(format!("aggregator {agg_id} exited with {status}"))
                        .await;
                }
            });
        }
        drop(failure_tx);

        // Start policy. The default is a SYNCHRONIZED start: wait for every cell to
        // register (bounded — cells only fetch their envelope, no work yet), then
        // trigger the START event so all cells begin dispatching together. This is a
        // tight O(N) fan-in rendezvous that fights unbounded horizontal scale.
        //
        // Tier T3 (`AIPERF_CELL_BARRIER_FREE=1`) is the master-less alternative k6 uses:
        // the controller triggers START IMMEDIATELY, without gathering all N
        // registrations. A cell that registers *after* the trigger sees the completed
        // event instantly (velo's completed-event cache), so each cell starts as soon
        // as it has its envelope — no O(N) rendezvous. The tradeoff is looser start
        // correlation across cells (arrival-epoch jitter), which is aggregate-equivalent
        // (the same bar as rate/ramp) and does not affect data-deterministic metrics.
        // A failed cell is still caught by the collect loop's failure watch below.
        if barrier_free {
            tracing::info!(
                "tier-T3 barrier-free start: triggering immediately without the O(N) register \
                 rendezvous (cells start on their own registration; looser cross-cell start sync)"
            );
        } else {
            tokio::select! {
                biased;
                () = transport.await_all_registered() => {}
                Some(failure) = failure_rx.recv() => bail!("{failure}"),
                () = tokio::time::sleep(register_timeout()) => {
                    bail!("cells did not all register within the registration timeout")
                }
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
        // A cell ships EXACTLY ONE terminal partition, of one of two kinds depending on
        // the mode it ran (the whole run is uniform — every cell got the same envelope
        // + env, so they never mix): a `Partition` (retain path: raw records for the
        // byte-exact global-order/concatenation merge) or a `StorePartition` (Stage-C
        // metrics-only exact-fold: the cell's folded EXACT store, no record Vec). Count
        // BOTH toward the one-per-cell termination barrier; the merge below dispatches on
        // which kind arrived.
        let mut partitions: Vec<RecordsShardPartition> = Vec::with_capacity(cell_count as usize);
        let mut store_partitions: Vec<ColumnStorePartition> =
            Vec::with_capacity(cell_count as usize);
        let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
        let collected = |records: &[RecordsShardPartition], stores: &[ColumnStorePartition]| {
            records.len() + stores.len()
        };
        // In the flat topology this is one partition per cell; under tier-T2 it is one
        // MERGED partition per aggregator (`expected_partitions == aggregator count`).
        while collected(&partitions, &store_partitions) < expected_partitions as usize {
            tokio::select! {
                biased;
                message = transport.recv() => match message.context("receiving from cell")? {
                    Some(CellMessage::Partition(partition)) => partitions.push(partition),
                    Some(CellMessage::StorePartition(partition)) => store_partitions.push(*partition),
                    Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                        heartbeats.insert(cell_id, *heartbeat);
                    }
                    None => bail!(
                        "transport closed with {} of {expected_partitions} partitions",
                        collected(&partitions, &store_partitions)
                    ),
                },
                Some(failure) = failure_rx.recv() => bail!("{failure}"),
                _ = &mut deadline => bail!(
                    "cellular run timed out with {} of {expected_partitions} partitions",
                    collected(&partitions, &store_partitions)
                ),
            }
        }

        // Merge → the single report. A metrics-only exact-fold run shipped folded stores
        // (Stage C): append them by cell_id (`merge_store_partitions`) — within-tolerance
        // (counts/percentiles/min/max exact; sums/means a few ULPs), the same bar the
        // in-process sharded exact-fold merge meets. Otherwise the cells shipped raw
        // records: scheduled cells pre-tile a global dispatch ordinal (byte-exact global
        // order); graph records carry a LOCAL per-cell request_index (concatenated by
        // cell_id, densely re-numbered). Cells never mix the two kinds in one run.
        let merged = if !store_partitions.is_empty() {
            ensure!(
                partitions.is_empty(),
                "cellular run mixed folded store partitions with raw record partitions \
                 ({} store, {} record) — every cell must ship the same kind",
                store_partitions.len(),
                partitions.len()
            );
            merge_store_partitions(metrics_config, store_partitions)
        } else {
            kind.merge(metrics_config, partitions)?
        };
        // `ingested_count()` — not `record_count()` — so a merged SKETCH store (which
        // retains no rows, `record_count() == 0`) reports its true total; identical to
        // `record_count()` for the retain/exact-fold merges.
        let record_count = merged.ingested_count() as usize;
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

        // Stage E artifact barrier: when cross-host HTTP shipping is active, wait for
        // every cell to POST its files AND its `/done` marker before concatenating, so
        // `temp_root/cell-{id}` is complete. (Same-host cells write their files locally
        // before shipping their velo partition, so the partition-collection loop above
        // is already their barrier.)
        // Only when per-record uploads are active (Stage E): a dataset-serve-only
        // run (Stage G, no per-record artifacts) never POSTs files or `/done`, so
        // there is nothing to wait for and the barrier would spuriously time out.
        if http_shipping
            && let Some(server) = artifact_server.as_ref()
        {
            server
                .wait_for_cells(cell_count, artifact_upload_timeout())
                .await
                .context("waiting for cellular artifact uploads")?;
        }

        // Per-record artifact concat + inputs.json copy. Each cell ran its ordinary
        // execute path with a per-cell `temp_root/cell-{id}` dir as its artifact_dir and
        // wrote its merged per-record artifacts (records/raw/CSV/parquet/outputs) there.
        // The controller concatenates them into the real artifact dir (the per-cell dirs
        // are the "shards"), reusing the Stage B concat (row SET-identical, completion
        // order accepted), before `_scratch` removes `temp_root`. `inputs.json` is NOT
        // concatenated (a single FULL-dataset document, not per-record rows): every cell
        // generated the identical up-front (S4) inputs.json over the same resident
        // dataset, so the controller copies ONE cell's copy verbatim
        // (`copy_cell_inputs_json`). inputs.json is always-on (`rust_wire`), so without
        // this the cellular run would silently drop it / break GenAI-Perf compat.
        //
        // The files are controller-local in two cases, both handled here:
        // - SAME-HOST (`!is_k8s`): every cell wrote directly to its controller-local
        //   `temp_root/cell-{id}` dir (Stage D).
        // - CROSS-HOST (k8s) with `http_shipping`: each pod wrote to its OWN fs, then
        //   shipped every file to the controller over HTTP + streaming zstd (Stage E),
        //   landing at the SAME `temp_root/cell-{id}/{rel}` paths.
        // Cross-host with shipping DISABLED still skips the concat (the files never
        // reach the controller) — the shared-storage product boundary, warned at start.
        if (!is_k8s || http_shipping)
            && let Some(artifact_dir) = report_path.parent()
        {
            // Read the SHIPPED copies from the landing subtree when HTTP shipping is
            // active (k8s: landing_root == temp_root; same-host force: a separate
            // `http-landing` subtree), else the cells' own local writes under
            // `temp_root/cell-{id}` (default same-host Stage D).
            let concat_source_root = if http_shipping { &landing_root } else { &temp_root };
            let cell_dirs: Vec<PathBuf> = (0..cell_count)
                .map(|cell_id| concat_source_root.join(format!("cell-{cell_id}")))
                .collect();
            // Per-record artifacts (records/raw/CSV/parquet/outputs) are concatenated only
            // when requested; inputs.json (a single full-dataset doc) is copied whenever a
            // cell produced one, independent of the per-record request set. `artifacts` was
            // parsed once at the top of the run (identically to the cell's execute path).
            if !requested_per_record_artifacts(envelope).is_empty() {
                crate::runner_protocol::shard_artifacts::concatenate_cell_artifacts(
                    &cell_dirs,
                    artifact_dir,
                    &artifacts,
                )
                .context("concatenating per-cell per-record artifacts")?;
            }
            crate::runner_protocol::shard_artifacts::copy_cell_inputs_json(
                &cell_dirs,
                artifact_dir,
                &artifacts,
            )
            .context("copying per-cell inputs.json")?;
        }

        // Stop the upload server (also dropped on any bail path).
        if let Some(server) = artifact_server {
            server.shutdown().await;
        }

        // `_scratch` removes `temp_root` on drop.
        Ok(CellularRunOutcome {
            report_path: report_path.to_path_buf(),
            cell_count,
            record_count,
        })
    })
}

/// The controller's HTTP artifact-upload bind (Stage E). A fixed routable port
/// (`AIPERF_CONTROLLER_ARTIFACT_BIND`, default `0.0.0.0:9600`) the operator exposes
/// on the controller pod; cells derive the matching authority from their `tcp://`
/// velo coordinate host + the artifact port. Distinct from the velo messaging bind
/// (control plane) — this carries bulk artifact bytes, not coordination.
#[cfg(feature = "velo")]
fn controller_artifact_bind() -> std::net::SocketAddr {
    use crate::runner_protocol::cellular_cell::DEFAULT_ARTIFACT_PORT;
    std::env::var("AIPERF_CONTROLLER_ARTIFACT_BIND")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| std::net::SocketAddr::from(([0, 0, 0, 0], DEFAULT_ARTIFACT_PORT)))
}

/// The controller's velo bind plus the `tcp://HOST:PORT` endpoint string cells
/// `velo.connect` to. The coordinate stays `tcp://` everywhere so the HTTP artifact
/// plane (which derives its authority by swapping the port on this coordinate) keeps
/// working — hence local uses a known loopback port, not UDS.
/// - **k8s**: bind the operator-known port (`AIPERF_CONTROLLER_PORT`, default 9500)
///   on all interfaces; the cell endpoint is injected into the pods by the operator
///   (`AIPERF_CELL_CONTROLLER_ADDR`), so the launcher's copy is unused (empty).
/// - **local**: pre-bind a loopback TCP listener so the actual port is known before
///   build; cells connect to `tcp://127.0.0.1:<port>`.
#[cfg(feature = "velo")]
fn controller_bind_and_endpoint(is_k8s: bool, temp_root: &Path) -> Result<(BindSpec, String)> {
    let _ = temp_root;
    if is_k8s {
        let port: u16 = std::env::var("AIPERF_CONTROLLER_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9500);
        return Ok((
            BindSpec::TcpBind(std::net::SocketAddr::from(([0, 0, 0, 0], port))),
            String::new(),
        ));
    }
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").context("binding controller loopback")?;
    let addr = listener
        .local_addr()
        .context("controller loopback local_addr")?;
    Ok((BindSpec::TcpListener(listener), format!("tcp://{addr}")))
}

/// The deadline for collecting every cell's partition. Covers the whole run (cells
/// execute the benchmark before shipping), so it is generous and env-overridable
/// (`AIPERF_CELL_COLLECT_TIMEOUT_SECS`, default 2 hours). Primarily a k8s backstop —
/// a local run's per-child exit watcher catches a dead cell far sooner.
#[cfg(feature = "velo")]
pub(crate) fn collect_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_COLLECT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(7200);
    std::time::Duration::from_secs(secs)
}

/// The deadline for the Stage E artifact-upload barrier ([`ArtifactUploadServer::
/// wait_for_cells`](crate::runner_protocol::artifact_shipping::ArtifactUploadServer::wait_for_cells)),
/// distinct from [`collect_timeout`]. By the time this barrier runs every cell has
/// already shipped its velo partition (metrics), so only the per-record artifact
/// bytes remain in flight — a few minutes is ample, and a much tighter bound than
/// the whole-run `collect_timeout` (default 2h). Env-overridable
/// (`AIPERF_CELL_ARTIFACT_UPLOAD_TIMEOUT`, seconds; default 5 minutes), so a cell
/// that dies mid-upload fails the run in minutes rather than hours.
#[cfg(feature = "velo")]
fn artifact_upload_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_ARTIFACT_UPLOAD_TIMEOUT")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
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

/// Dataset `format`s proven — in `crate::dataset::loader` — to compile exactly ONE turn
/// per conversation, so a cellular run over a `file`/`public` source keeps the sampler's
/// per-conversation draw index aligned with the issuer's per-turn ordinal. Verified
/// per-format:
/// - `single_turn` (`loader/simple.rs`: `SingleTurnComposer`, one turn per JSONL row).
///   Residual: its composer groups rows sharing an explicit `session_id` into one
///   multi-turn conversation (`simple.rs` ~:345-372) — a file that reuses `session_id`
///   is the same divergence class as multi-turn and is a documented residual, not gated
///   here (the canonical one-request-per-row format is admitted).
/// - `raw_payload` (`loader/raw_payload.rs`: loader stamps a unique `group_key = row:{i}`
///   ~:62, so `RawPayloadComposer` ~:116-140 makes one conversation per row, one turn).
/// - `accuracy` (`loader/public.rs`: `AccuracyComposer` ~:237-243 builds a fresh
///   conversation with a single turn per row, unconditionally — no session grouping).
/// - `hf_instruction_response` (`loader/public.rs`: `HfInstructionComposer` ~:382-386,
///   one turn per row, fresh conversation).
///
/// Every OTHER linear format is rejected (fail closed). Proven multi-turn / session-
/// grouping: `multi_turn` (`simple.rs` ~:377-417), `mooncake_trace`/`bailian_trace`/
/// `burst_gpt`/`sagemaker_data_capture` (`trace.rs`, session-keyed grouping, e.g. mooncake
/// ~:232-247), `inputs_json` (`raw_payload.rs` ~:457-472, many payloads per session),
/// `sharegpt` (`public.rs` ~:297-318), `hf_conversation` (`public.rs` ~:405 `multi_turn`
/// option), `mt_bench`. Unverified turn semantics (`mmvu`, `spec_bench`, `speed_bench`,
/// `hf_asr`, `random_pool`, `exgentic`, `exgentic_v2`, `synthetic_rankings`) are rejected
/// conservatively. The graph formats (`dag_jsonl`/`weka_trace`/`dynamo_trace`) never reach
/// this list — they short-circuit to the whole-trace partition above.
const CELLULAR_SINGLE_TURN_FILE_FORMATS: [&str; 4] = [
    "single_turn",
    "raw_payload",
    "accuracy",
    "hf_instruction_response",
];

/// Whitelists a cellular run to the shape the cell topology is currently *wired* for:
/// the shared online-scheduled executor over the `http` transport, on **synthetic,
/// file, or public single-turn** datasets. There is no bespoke HTTP layer — a cell
/// runs the same `execute.rs` path as any single-process run, differing only by an
/// injected [`IssuanceAuthority`] and an env-gated records sink (`CellRecordsShipper`).
/// The partition/issuance seam is transport-neutral; this whitelist reflects wiring
/// coverage, not an HTTP special case. Two invariants underpin byte parity and each
/// fails closed here:
///
/// - **Only the online-scheduled executor ships a partition today.** The gRPC and
///   offline executors are separate paths that do not yet inject the cell issuer or
///   ship records, so a `grpc`/`dynosim` transport runs an unwired executor and hangs
///   the controller; it is rejected, not silently divergent. Synthetic, `file`, and
///   `public` linear datasets ARE wired: synthetic regenerates from the shared seed,
///   a cross-host `file`/`path` source ships controller->cell over HTTP+zstd (Stage G)
///   and recompiles deterministically per cell, and `public` URL/HF each cell fetches
///   itself. Graph programs (dag_jsonl/weka_trace/dynamo_trace) take the whole-trace
///   partition below. Multi-turn `file` traces (per-conversation partition, like the
///   graph path) remain a documented follow-up, rejected by the single-turn guard.
/// - **One sampler draw must equal one dispatched turn.** [`PartitionedSampler`]
///   partitions by conversation *draw*, but the issuer stamps a per-*turn* ordinal
///   ([`CellularAutonomousIssuer`]); a multi-turn conversation makes the two diverge,
///   so the merged report silently reorders (or, for variable turn counts, draws a
///   different instance set). Only `turns == 1` (the default) is sound.
///
/// For a `file`/`public` dataset the turn count is NOT driven by the top-level `turns`
/// config field — it is compiled by the dataset FORMAT plus `session_id` grouping in the
/// loader. `multi_turn` (`dataset/loader/simple.rs`), the trace formats (`mooncake_trace`,
/// `bailian_trace`, `burst_gpt`, `sagemaker_data_capture` in `dataset/loader/trace.rs`),
/// `inputs_json` (`dataset/loader/raw_payload.rs`), and the multi-turn public shapes
/// (`sharegpt`, `hf_conversation`, `mt_bench`, ... in `dataset/loader/public.rs`) all
/// group rows into MULTI-turn conversations regardless of `turns`. So the top-level
/// `turns == 1` check alone does NOT prove single-turn for a file/public dataset — it is
/// backstopped by [`CELLULAR_SINGLE_TURN_FILE_FORMATS`], an explicit allowlist of the
/// formats proven (in the loader code) to compile exactly one turn per conversation. The
/// whitelist fails closed: an absent or unlisted format is rejected, so a session-grouping
/// or ambiguous format can never slip through. Per-conversation cellular partition (which
/// would admit the multi-turn formats, like the graph path) is a documented follow-up.
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
            matches!(kind, Some("synthetic" | "file" | "public")),
            "cellular runs support synthetic, file, or public datasets (whose conversations the \
             sampler can partition one-draw-per-turn); got dataset type {kind:?}. A cross-host \
             `file`/`path` dataset is shipped controller->cell over HTTP+zstd (Stage G) and \
             recompiled per cell; inline `records` and `public` URL/HF each cell resolves itself"
        );
        // Format whitelist (file/public only): the turn count of a file/public dataset is
        // compiled by the FORMAT + `session_id` grouping in the loader, NOT the top-level
        // `turns` field below. Only the formats proven to emit exactly one turn per
        // conversation preserve the one-draw==one-turn invariant; every other (including
        // session-grouping trace formats like `mooncake_trace`) is rejected fail-closed.
        // Synthetic regenerates single-turn conversations from the shared seed and needs
        // no format check; graph formats already `continue`d above.
        if matches!(kind, Some("file" | "public")) {
            let format = dataset.get("format").and_then(serde_json::Value::as_str);
            ensure!(
                format.is_some_and(|format| CELLULAR_SINGLE_TURN_FILE_FORMATS.contains(&format)),
                "cellular file/public datasets support only strictly single-turn formats ({}); got \
                 format {format:?}. A file/public dataset's turn count is driven by its FORMAT and \
                 `session_id` grouping, not the top-level `turns` field, so session-grouping / \
                 multi-turn formats (multi_turn, mooncake_trace, bailian_trace, burst_gpt, \
                 sagemaker_data_capture, inputs_json, sharegpt, hf_conversation, mt_bench, ...) \
                 compile multi-turn conversations whose per-turn issuer ordinal diverges from the \
                 sampler's per-conversation draw index and silently break the merged report. \
                 Per-conversation cellular partition (like the graph path) is a documented follow-up",
                CELLULAR_SINGLE_TURN_FILE_FORMATS.join("/")
            );
        }
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
/// Spawns the tier-T2 aggregator subprocesses (`aiperf-runner --aggregator`), one per
/// aggregator, each fed the run envelope on stdin (for the merge `MetricsConfig`) and
/// its subtree parameters via env: its id, the fixed loopback `tcp://` coordinate it
/// binds (its cells dial it there), how many cells ship to it, and the controller
/// coordinate it ships its one merged store up to. Returns the children so the
/// controller watches them for hard failure. `kill_on_drop` tears them down on any
/// controller abort.
async fn spawn_aggregators(
    envelope: &serde_json::Value,
    agg_count: u32,
    cell_count: u32,
    base_port: u16,
    controller_coordinate: &str,
) -> Result<Vec<tokio::process::Child>> {
    use crate::runner_protocol::cellular_aggregator::{
        AGG_BIND_ENV, AGG_CHILD_COUNT_ENV, AGG_ID_ENV, children_of,
    };
    use crate::runner_protocol::cellular_cell::CELL_CONTROLLER_ADDR_ENV;
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;

    let envelope_bytes =
        serde_json::to_vec(envelope).context("serializing envelope for aggregators")?;
    let exe = std::env::current_exe().unwrap_or_else(|_| "aiperf-runner".into());
    let mut children = Vec::with_capacity(agg_count as usize);
    for agg_id in 0..agg_count {
        let child_count = children_of(agg_id, agg_count, cell_count);
        let mut child = tokio::process::Command::new(&exe)
            .arg("--aggregator")
            .env(AGG_ID_ENV, agg_id.to_string())
            .env(
                AGG_BIND_ENV,
                format!("tcp://127.0.0.1:{}", base_port + agg_id as u16),
            )
            .env(AGG_CHILD_COUNT_ENV, child_count.to_string())
            .env(CELL_CONTROLLER_ADDR_ENV, controller_coordinate)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning aggregator {agg_id}"))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&envelope_bytes)
                .await
                .with_context(|| format!("writing envelope to aggregator {agg_id}"))?;
            // `stdin` drops here → EOF, so the aggregator's `read_to_end` returns.
        }
        children.push(child);
    }
    Ok(children)
}

pub(crate) fn cellular_metrics_config(envelope: &serde_json::Value) -> Result<MetricsConfig> {
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

/// The per-record FILE artifacts a cellular run will DROP because the deployment
/// cannot deliver a cell's files to the controller — empty unless this is a
/// CROSS-HOST (k8s) run that requested them. This is the documented Stage E product
/// boundary, and the single source of truth the concat gate (`!is_k8s`) and the
/// operator warning ([`warn_dropped_per_record_artifacts`]) both derive from, so the
/// two never drift.
///
/// Same-host (`--cells N`, local launcher) drops nothing: every cell runs its ordinary
/// execute path with a controller-local `temp_root/cell-{id}` dir as its artifact_dir,
/// writes its merged per-record artifacts there, and the controller concatenates them
/// into the real artifact dir at finalize (`concatenate_cell_artifacts`) — so this
/// returns empty for `!is_k8s` even when files are requested. A k8s pod's cell dir
/// lives on its OWN pod filesystem, unreachable by the controller, so its per-record
/// files are dropped.
///
/// **Why not ship the bytes over velo (the deliberate boundary, not a TODO).** k8s
/// cellular is the substrate for the largest distributed runs (300k–1M concurrency in
/// the project's own durability ramps); per-record artifacts (records.jsonl, one row
/// per request) scale with total request count and reach tens–hundreds of GB summed
/// across cells. velo is a control plane, and its transparent large-payload path
/// (`velo::…::rendezvous::DataStore`, `StageMode::InMemory`; the RDMA arena is a Phase-2
/// placeholder) buffers each staged payload whole in RAM on BOTH ends — so shipping it
/// as-is would put the sum of every shard's largest file in the single controller's RAM.
/// Even hand-rolled sub-threshold chunk-to-disk (which would bound memory) would still
/// funnel every cell's bulk artifact bytes through one controller node and require that
/// node to hold the SUM of all shards on local disk — coupling the latency-sensitive
/// coordination plane (heartbeats + small metric partitions) to a bulk-data plane, and
/// reinventing, worse, what a shared filesystem provides natively. The intended
/// cross-host mechanism is **shared object storage** (a ReadWriteMany PVC or S3-style
/// bucket the operator mounts into every cell pod AND the controller): the cell's
/// existing local-write path then lands each shard in the shared location and the
/// controller's existing Stage D concat runs unchanged, with no bulk data on the
/// control plane.
fn dropped_cross_host_artifacts(envelope: &serde_json::Value, is_k8s: bool) -> Vec<&'static str> {
    if !is_k8s {
        // Same-host concatenates them (`concatenate_cell_artifacts`); nothing dropped.
        return Vec::new();
    }
    requested_per_record_artifacts(envelope)
}

/// Warns, once at controller startup, when a CROSS-HOST (k8s) cellular run requests a
/// per-record FILE artifact the controller cannot collect (`records_path`/`raw_path`/
/// `records_csv_path`/`records_parquet_path`/`outputs_path`). Thin logging wrapper over
/// the pure [`dropped_cross_host_artifacts`] boundary; the accompanying rationale
/// (why bulk bytes are NOT shipped over the velo control plane, and that shared object
/// storage is the intended mechanism) lives on that function.
fn warn_dropped_per_record_artifacts(envelope: &serde_json::Value, is_k8s: bool) {
    let dropped = dropped_cross_host_artifacts(envelope, is_k8s);
    if !dropped.is_empty() {
        tracing::warn!(
            artifacts = dropped.join(","),
            "cross-host (k8s) cellular mode does not collect per-record file artifacts \
             from cell pods: each pod writes them to its own local filesystem, which the \
             controller cannot read, so these files will NOT appear in the run artifact \
             dir. The merged report and native exporter outputs (genai-perf-v1 JSON/CSV, \
             console.txt, timeslice) ARE still produced. To collect per-record files \
             across hosts, mount shared object storage (a ReadWriteMany PVC or S3-style \
             bucket) into every cell pod and the controller so each cell writes its shard \
             to the shared path; a same-host --cells run also emits them directly. Bulk \
             artifact bytes are intentionally not streamed over the velo control plane."
        );
    }
}

/// The per-record FILE artifacts a run requests that a cellular run will NOT emit
/// (records/raw/CSV/parquet/outputs), read from the envelope's `cfg.artifacts`. Pure
/// (no logging) so the detection is unit-testable; `warn_dropped_per_record_artifacts`
/// is the logging wrapper. `inputs_path` is deliberately excluded — it is a per-SESSION
/// artifact, not a per-record one.
fn requested_per_record_artifacts(envelope: &serde_json::Value) -> Vec<&'static str> {
    const ARTIFACTS: [&str; 5] = [
        "records_path",
        "raw_path",
        "records_csv_path",
        "records_parquet_path",
        "outputs_path",
    ];
    ARTIFACTS
        .into_iter()
        .filter(|key| {
            envelope
                .pointer(&format!("/run/cfg/artifacts/{key}"))
                .is_some_and(|value| !value.is_null())
        })
        .collect()
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
            // Stage G: a single-turn `file`/`path` dataset is now accepted — the
            // controller ships the source cross-host and each cell recompiles it.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "single_turn", "path": "/data/prompts.jsonl"}]}}}),
            // An inline-records `file` dataset (no path) is accepted — it already
            // rides in the envelope, so no ship is needed.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "single_turn", "records": []}]}}}),
            // A single-turn `public` dataset is accepted — each cell fetches the URL/HF
            // source itself. A whitelisted single-turn format is required (below).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "public", "format": "accuracy"}]}}}),
            // `raw_payload` is strictly single-turn (unique per-row group key).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "raw_payload", "path": "/data/p.jsonl"}]}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&ok).is_ok(),
                "should accept {ok}"
            );
        }
        // Fail closed on each unsupported aspect (all else valid + seeded):
        for bad in [
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "transport": {"type": "grpc"}}}}),
            // An unknown dataset type is still rejected (only synthetic/file/public wired).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "agentic"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"value": 3}}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic", "turns": {"mean": 2.0, "stddev": 1.0}}]}}}),
            // A multi-turn `file` dataset (explicit turns > 1) is rejected — the
            // per-conversation partition is a documented follow-up (Stage G non-goal).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "multi_turn", "path": "/data/t.jsonl", "turns": {"value": 3}}]}}}),
            // A session-grouping trace format is rejected by the FORMAT whitelist even
            // with NO top-level `turns` override: the file itself compiles multi-turn
            // conversations (grouped by session_id), which the `turns` guard cannot see.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "mooncake_trace", "path": "/data/t.jsonl"}]}}}),
            // `multi_turn` with no `turns` override is still rejected by the whitelist.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "multi_turn", "path": "/data/t.jsonl"}]}}}),
            // A `file` dataset with an unknown/unspecified format fails closed.
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "path": "/data/t.jsonl"}]}}}),
            // A bare `public` dataset (no format) fails closed — its default format could
            // be a multi-turn public shape (sharegpt / mt_bench / hf_conversation).
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "public"}]}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&bad).is_err(),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn admits_graph_and_linear_file_datasets() {
        // Graph programs (dag_jsonl / weka_trace / dynamo_trace) partition by whole
        // trace via PartitionedGraphTraceSource, so validate_cellular_run_shape admits
        // them past the single-turn guard even though they are `file` datasets. The
        // http transport ensure is unchanged (graph dispatches HTTP too).
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
        // Stage G + C1 fix: a PROVEN single-turn linear file format (single_turn) is
        // ADMITTED — the controller ships the source cross-host and each cell recompiles
        // it. It is NOT a graph dataset (takes the scheduled partition).
        let linear = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "file", "format": "single_turn", "path": "/data/t.jsonl"}],
        }}});
        assert!(
            validate_cellular_run_shape(&linear).is_ok(),
            "should admit single-turn linear file dataset"
        );
        assert!(
            !is_graph_dataset(&linear),
            "single_turn is not a graph dataset (scheduled partition)"
        );
        // C1 fix: `mooncake_trace` is a NON-graph, session-grouping trace format that
        // compiles MULTI-turn conversations. It takes the scheduled partition (not the
        // graph carve-out) where one-draw-per-turn is load-bearing, so the format
        // whitelist must REJECT it — restoring the pre-Stage-G safety for the multi-turn
        // case. (Before this fix it was wrongly admitted: a silent draw/ordinal
        // divergence regression.)
        let multi_turn_trace = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "file", "format": "mooncake_trace", "path": "/data/t.jsonl"}],
        }}});
        assert!(
            validate_cellular_run_shape(&multi_turn_trace).is_err(),
            "mooncake_trace (session-grouping multi-turn) must be rejected by the format whitelist"
        );
        assert!(
            !is_graph_dataset(&multi_turn_trace),
            "mooncake_trace is not a graph dataset (would take the scheduled partition)"
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

    #[test]
    fn detects_requested_per_record_artifacts() {
        // A metrics-only cellular run (no per-record file artifacts) → nothing detected,
        // so the warn stays silent. `inputs_path` is per-session, not per-record, so it
        // must NOT trigger the per-record warning either.
        let metrics_only = serde_json::json!({"run": {"cfg": {"artifacts": {
            "inputs_path": "inputs.json",
        }}}});
        assert!(
            requested_per_record_artifacts(&metrics_only).is_empty(),
            "metrics-only (+ per-session inputs.json) must not flag per-record artifacts"
        );
        // An empty artifacts object and a wholly absent one are both silent.
        assert!(requested_per_record_artifacts(&serde_json::json!({})).is_empty());

        // Each per-record file artifact is detected independently.
        for key in [
            "records_path",
            "raw_path",
            "records_csv_path",
            "records_parquet_path",
            "outputs_path",
        ] {
            let env = serde_json::json!({"run": {"cfg": {"artifacts": {key: "x"}}}});
            assert_eq!(
                requested_per_record_artifacts(&env),
                vec![key],
                "{key} should be detected as a per-record file artifact"
            );
        }

        // A run requesting several reports them all (order matches the scan order).
        let many = serde_json::json!({"run": {"cfg": {"artifacts": {
            "records_path": "profile_export.jsonl",
            "records_parquet_path": "profile_export.parquet",
            "outputs_path": "profile_export.json",
        }}}});
        assert_eq!(
            requested_per_record_artifacts(&many),
            vec!["records_path", "records_parquet_path", "outputs_path"],
        );
    }

    #[test]
    fn cross_host_artifact_boundary_is_precise() {
        // Stage E product boundary: the per-record artifact drop is EXACTLY
        // "cross-host (k8s) run that requested per-record files". `dropped_cross_host_artifacts`
        // is the single source of truth for both the concat gate (`!is_k8s`) and the operator
        // warning, so this pins that they can never disagree.
        let with_files = serde_json::json!({"run": {"cfg": {"artifacts": {
            "records_path": "profile_export.jsonl",
            "records_parquet_path": "profile_export.parquet",
            "outputs_path": "profile_export.json",
        }}}});

        // Same-host (`!is_k8s`) drops NOTHING even when per-record files are requested:
        // the controller concatenates each cell's controller-local dir (`concatenate_cell_artifacts`),
        // so the gate runs and the warn stays silent.
        assert!(
            dropped_cross_host_artifacts(&with_files, false).is_empty(),
            "same-host concatenates per-record artifacts; nothing is dropped"
        );

        // Cross-host (k8s) with per-record files → exactly those files are dropped (each
        // pod writes to its own filesystem, unreachable by the controller). Order matches
        // the scan order so the warn's `artifacts=` field is stable.
        assert_eq!(
            dropped_cross_host_artifacts(&with_files, true),
            vec!["records_path", "records_parquet_path", "outputs_path"],
            "cross-host drops exactly the requested per-record files"
        );

        // Cross-host but metrics-only (only the per-session inputs.json) → nothing dropped,
        // so the k8s warn does NOT fire spuriously on a run that produces no per-record files.
        let metrics_only = serde_json::json!({"run": {"cfg": {"artifacts": {
            "inputs_path": "inputs.json",
        }}}});
        assert!(
            dropped_cross_host_artifacts(&metrics_only, true).is_empty(),
            "cross-host metrics-only run drops no per-record files (inputs.json is per-session)"
        );
        // A cross-host run with no artifacts block at all is likewise silent.
        assert!(dropped_cross_host_artifacts(&serde_json::json!({}), true).is_empty());
    }
}
