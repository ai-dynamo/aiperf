// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-process cellular controller.
//!
//! When a run requests `cfg.runtime.cells > 1`, the receiving runner becomes the
//! controller rather than executing in-process. It partitions the request budget by
//! `(cell_id, cell_count)`, spawns one `aiperf --cell` child per cell (each a
//! separate OS process, wired with the autonomous issuer and per-cell sampler),
//! serves the [`transport`](crate::cellular::transport) endpoint the cells ship
//! their records-shard partitions and heartbeats back over, merges every cell's
//! records in global dispatch-ordinal order into the single authoritative
//! `native-v2.json`, and fails the run loudly if any cell exits non-zero. The
//! controller exposes cellular execution as one protocol-v2 run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cellular::{
    ColumnStorePartition, MetricsHeartbeat, RecordsShardPartition, TDigest,
    merge_records_by_concatenation, merge_records_in_global_order, merge_store_partitions,
};
use crate::metrics_core::report::NativeReport;
use crate::metrics_core::{ExportContext, MetricsAccumulator, MetricsConfig, PERCENTILES};
use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::engine::cell_launcher::owned_positions;
use crate::engine::cellular_kind::CellularRunKind;

// The velo transport + launcher wiring is the only part of the controller that
// needs the `velo` feature; the validation, budget-slicing, merge, and report
// assembly below are plain envelope/metric logic reused by the non-velo build.
#[cfg(feature = "cellular")]
use crate::cellular::transport::connect::{BindSpec, build_velo};
#[cfg(feature = "cellular")]
use crate::cellular::{CellMessage, ControllerTransport, SpecFor, VeloControllerTransport};
#[cfg(feature = "cellular")]
use crate::engine::cell_launcher::{CellLaunchContext, select_launcher};

/// Env toggle for barrier-free start: the controller
/// triggers START immediately instead of gathering all N cell registrations first
/// (the O(N) fan-in rendezvous). Default off (the tight synchronized start). Cells
/// registering after the trigger see the completed event instantly (velo's
/// completed-event cache), so each starts on its own registration.
pub const CELL_BARRIER_FREE_ENV: &str = "AIPERF_CELL_BARRIER_FREE";

/// Env toggle routing the run-wide START through the monotonic
/// phaser control plane instead of the single-shot velo event: the controller binds a
/// `PhaserServer` and `advance`s `Started`; cells subscribe with `PhaserClient` and
/// await generation 1. Default off (the event-based START). The phaser generalizes
/// START to phase transitions and dataset-availability signals.
pub const CELL_PHASER_START_ENV: &str = "AIPERF_CELL_PHASER_START";

/// Env toggle for the dataset fan-out data plane: the controller
/// generates the dataset request-ids once and broadcasts them; each cell builds its
/// owned index over velo and dispatches its owned requests. Default off
/// (per-cell seed regeneration or controller file serving).
pub const CELL_DATASET_FANOUT_ENV: &str = "AIPERF_CELL_DATASET_FANOUT";

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

impl CellularRunKind {
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

/// Appends one live cross-cell aggregate `metrics_heartbeat` line to
/// `AIPERF_CELLULAR_HEARTBEAT_LOG` while the run is in flight, so the `aiperf
/// controller` frontend can tail it and patch a native-v2-level snapshot into the
/// AIPerfJob `.status` (counters into `.status.phases.profiling`, the metric
/// percentiles into `.status.snapshot`). A cross-host controller has no single
/// load-gen process to run the [`crate::engine::heartbeat_lane`] lane, so
/// it emits the running aggregate of every cell's latest heartbeat here instead.
///
/// Emits the FULL native-v2-level snapshot: counters summed and the TTFT/ITL/latency
/// t-digests merged across cells (via [`MetricsHeartbeat::merge`]), serialized to the
/// identical NDJSON shape the single-process lane writes ([`heartbeat_event_line`]),
/// so the live CR snapshot converges to the final `native-v2.json` metrics. Best-effort
/// — no log path (env unset, e.g. a local `--cells` run whose frontend does not set
/// it) or a transient write error just skips the tick; the authoritative report still
/// comes from the merged partitions at finalize.
#[cfg(feature = "cellular")]
fn emit_live_progress(log_path: Option<&Path>, heartbeats: &BTreeMap<u32, MetricsHeartbeat>) {
    use std::io::Write as _;
    let Some(path) = log_path else {
        return;
    };
    // Merge every cell's latest heartbeat into the cross-cell aggregate: counters
    // summed, latency sketches t-digest-merged — the same fold the finalize path does.
    let mut merged: Option<MetricsHeartbeat> = None;
    for heartbeat in heartbeats.values() {
        match merged {
            Some(ref mut aggregate) => aggregate.merge(heartbeat),
            None => merged = Some(heartbeat.clone()),
        }
    }
    let Some(merged) = merged else {
        return;
    };
    let Some(mut line) = crate::engine::heartbeat_lane::heartbeat_event_line(&merged) else {
        return;
    };
    line.push(b'\n');
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(&line);
    }
}

/// Runs one benchmark across `cell_count` cells and writes the merged report to
/// `report_path`. Blocks until every cell ships. Requires the `velo` feature (the
/// cell transport).
#[cfg(feature = "cellular")]
pub fn run_cellular(
    envelope: &serde_json::Value,
    cell_count: u32,
    report_path: &Path,
    exporters: &crate::export::ExporterRegistry,
) -> Result<CellularRunOutcome> {
    ensure!(cell_count >= 1, "cell_count must be at least 1");
    validate_cellular_run_shape(envelope)?;
    // The dataset-shape gate above runs before the kind is known; the kind then names
    // the scheduled-vs-graph run once and owns its four differing behaviours (phase
    // validation, ordinal bases, record merge, session-budget slicing). The scheduled
    // path folds the profiling budget check in — graph phases carry sessions/duration,
    // not a `requests` budget.
    let kind = CellularRunKind::detect(envelope);
    kind.validate_phases(envelope, cell_count)?;
    // Same-host (local launcher) cells write their per-record artifacts into
    // controller-local `temp_root/cell-{id}` dirs, which the controller concatenates into
    // the real artifact dir at finalize. A cross-host (k8s) pod writes to its
    // own filesystem, so those files stay unreachable by the controller — still dropped.
    let is_k8s = matches!(
        std::env::var(crate::engine::cell_launcher::CELL_LAUNCHER_ENV).as_deref(),
        Ok("k8s")
    );
    // Barrier-free start skips the O(N) registration rendezvous.
    let barrier_free = matches!(
        std::env::var(CELL_BARRIER_FREE_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    // Phaser-driven START is opt-in; the default uses the event.
    let phaser_start = matches!(
        std::env::var(CELL_PHASER_START_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    // Dataset fan-out is opt-in. The controller generates the dataset's
    // request-ids once and broadcasts them; each cell builds its owned index over velo
    // and dispatches its owned requests (default off = per-cell seed
    // regeneration or controller file serving).
    let dataset_fanout = matches!(
        std::env::var(CELL_DATASET_FANOUT_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    // Parse the artifact policy once for upload and concatenation.
    let artifacts: crate::engine::protocol::ArtifactSpec = envelope
        .pointer("/run/cfg/artifacts")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    // Cross-host cells upload requested artifacts over HTTP with streaming zstd.
    // Same-host cells use shared-filesystem concatenation.
    // The test/dev HTTP-force seam ([`CELL_ARTIFACT_HTTP_FORCE_ENV`]) drives the
    // cross-host HTTP artifact path over loopback for a SAME-HOST run so a
    // multi-process tests can exercise the shipping mechanism. Off by default,
    // so a normal `--cells N` run keeps shared-filesystem concatenation.
    let force_http = crate::engine::cellular_cell::artifact_http_force_enabled();
    // `http_shipping` here means "per-record artifacts are shipped cross-host"
    // (over some transport), gating the barrier + concat-from-landing. The transport
    // toggle then decides HTTP vs velo for the actual byte movement.
    let http_shipping = (is_k8s || force_http)
        && crate::engine::cellular_cell::http_artifact_shipping_enabled()
        && !crate::engine::artifact_shipping::shippable_relatives(&artifacts).is_empty();
    // When selected, per-record artifacts ride the shared velo endpoint instead of the
    // HTTP artifact server/port (dataset serving stays on HTTP for now).
    let velo_artifacts = crate::engine::cellular_cell::artifact_transport_is_velo();
    // The HTTP upload server is only needed for artifacts when NOT using velo.
    let http_upload = http_shipping && !velo_artifacts;
    // A cross-host cell cannot read a controller-local `file`/`path` dataset
    // source, so the controller serves it over the HTTP+zstd plane and the cell
    // recompiles it locally. Only a cross-host (k8s / force) run with a `file`/`path`
    // dataset and HTTP shipping enabled needs the serve; same-host cells read the
    // controller-local path directly, and synthetic/inline-records/public need no serve.
    let dataset_source = crate::engine::cellular_cell::cellular_file_dataset_path(envelope);
    let dataset_ship = (is_k8s || force_http)
        && crate::engine::cellular_cell::http_artifact_shipping_enabled()
        && dataset_source.is_some();
    // Compute the serve plan before binding so an
    // unreadable/missing/unsupported source fails the run closed here rather than
    // half-shipping. A single-file trace (or scheduled `file`/`path` dataset) ships as
    // one file; a graph trace whose `path` is a DIRECTORY or SEGMENTED-PREFIX ships
    // every shard the loader would read (`enumerate_recorded_trace_files`), reconstructed
    // per cell from the manifest. dag_jsonl reads a single file only, so a dag_jsonl
    // directory/prefix still fails closed.
    let dataset_plan = if dataset_ship {
        let source = dataset_source
            .as_ref()
            .expect("dataset_ship implies a source");
        let format = envelope
            .pointer("/run/cfg/datasets/0/format")
            .and_then(serde_json::Value::as_str);
        Some(build_dataset_serve_plan(format, source)?)
    } else {
        None
    };
    // One HTTP server handles per-record uploads (HTTP transport only) and dataset serving.
    let need_artifact_server = http_upload || dataset_ship;
    // The force seam only applies to the same-host launcher (k8s already ships): when
    // true, cells write to their own controller-local `temp_root/cell-{id}` scratch AND
    // ship those files to a SEPARATE loopback landing dir, from which the concat reads —
    // so the shipped bytes (not the local writes) feed the merged report. This holds for
    // BOTH transports: velo same-host shipping needs the separate landing subtree too,
    // otherwise the velo receiver would overwrite each cell file with itself in place.
    let force_local_http = need_artifact_server && !is_k8s;
    let force_local_landing =
        (need_artifact_server || (http_shipping && velo_artifacts)) && !is_k8s;
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

        // Start the artifact upload server before launching cells; a k8s pod
        // may start and upload before the controller's collect loop). Cells POST their
        // per-record artifact files here with streaming zstd; each file lands at
        // `temp_root/cell-{id}/{rel}`, where concatenation reads. The
        // allowlist is the run's shippable relative paths, so a cell can only land
        // known artifacts inside its own cell dir.
        // Where uploaded artifact files land (`landing_root/cell-{id}/{rel}`). k8s
        // lands directly in `temp_root/cell-{id}` (the concat's own dirs — the cell fs
        // is a different host, no collision). The same-host force seam MUST land in a
        // SEPARATE subtree, because there the cell's own artifact_dir already IS
        // `temp_root/cell-{id}`; landing there would overwrite each file with itself.
        let landing_root = if force_local_landing {
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
            // Upload allowlist: the run's requested per-record artifact paths when
            // shipping is active, else empty (a dataset-serve-only run accepts no
            // uploads, so every POST is rejected).
            let allowed: std::collections::HashSet<String> = if http_shipping {
                crate::engine::artifact_shipping::shippable_relatives(&artifacts)
                    .into_iter()
                    .collect()
            } else {
                std::collections::HashSet::new()
            };
            // Dataset serve plan: the run's `file`/`path` source(s), keyed
            // by flat relative name, plus the manifest a cell fetches to reconstruct a
            // directory / segmented-prefix trace. Empty (`None` manifest) otherwise.
            let (datasets, manifest) = match dataset_plan {
                Some((map, manifest)) => (map, Some(manifest)),
                None => (std::collections::HashMap::new(), None),
            };
            Some(
                crate::engine::artifact_shipping::ArtifactUploadServer::start_with_dataset_plan(
                    artifact_bind,
                    landing_root.clone(),
                    allowed,
                    datasets,
                    manifest,
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
        // When artifacts ride velo, hang the artifact receive handlers off THIS same
        // control-plane velo instance (no second port): cells stream their zstd chunks
        // here and the receiver lands them at `landing_root/cell-{id}/{rel}`, exactly
        // where the HTTP server would, so the downstream barrier + concat are unchanged.
        let velo_artifact_receiver = if http_shipping && velo_artifacts {
            let allowed: std::collections::HashSet<String> =
                crate::engine::artifact_shipping::shippable_relatives(&artifacts)
                    .into_iter()
                    .collect();
            Some(
                crate::engine::artifact_stream_velo::ArtifactVeloReceiver::register(
                    velo.clone(),
                    landing_root.clone(),
                    allowed,
                )
                .context("registering velo artifact receiver")?,
            )
        } else {
            None
        };
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

        // When phaser START is selected, bind the phaser control plane
        // on the controller velo BEFORE it moves into the transport, so cells can
        // subscribe. `advance(Started)` below drives the run-wide START through the
        // monotonic phaser. The server (held for the run) keeps its handlers alive via its
        // own velo clone independent of the transport.
        let phaser = phaser_start.then(crate::cellular::phaser::Phaser::new);
        let _phaser_server = match &phaser {
            Some(phaser) => Some(
                crate::cellular::transport::phaser_velo::PhaserServer::bind(
                    velo.clone(),
                    phaser.clone(),
                )
                .context("binding phaser control plane")?,
            ),
            None => None,
        };

        // Bind the dataset service, generate the
        // dataset's request-ids once, broadcast them chunk-by-chunk (advancing the phaser
        // `ShardsAvailable` per chunk when the phaser is active), and finalize. A
        // bounded run distributes fully before dispatch; each
        // cell then subscribes and builds its owned index over velo. Held for the run so
        // the service's handlers/pumps stay alive.
        let _dataset_server = if dataset_fanout {
            let publisher =
                crate::cellular::dataset_session::DatasetPublisher::<Vec<u8>>::new();
            let server = crate::cellular::transport::dataset_velo::DatasetServer::bind(
                velo.clone(),
                publisher.clone(),
            )
            .context("binding dataset fan-out plane")?;
            let total = profiling_request_budget(envelope).unwrap_or(0);
            // Build each request's endpoint-ready body once on the controller so a cell
            // POSTs exactly what the controller
            // published — the fan-out is the real dispatch source. Chat-completions body
            // against the run's endpoint URL + model.
            let url = envelope
                .pointer("/run/cfg/endpoint/urls")
                .and_then(serde_json::Value::as_array)
                .and_then(|urls| urls.first())
                .and_then(serde_json::Value::as_str)
                .context("dataset fan-out: run cfg has no endpoint url")?;
            let chat_url = format!("{}/v1/chat/completions", url.trim_end_matches('/'));
            let model = envelope
                .pointer("/run/cfg/models/items/0/name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("model");
            const CHUNK: u64 = 16;
            let mut start = 0;
            while start < total {
                let end = (start + CHUNK).min(total);
                let requests = (start..end)
                    .map(|request_id| {
                        let body = serde_json::json!({
                            "model": model,
                            "messages": [{"role": "user", "content": format!("benchmark request {request_id}")}],
                            "max_tokens": 8,
                            "stream": false,
                        });
                        let wire = crate::cellular::dispatch_state::WireRequest {
                            url: chat_url.clone(),
                            body: serde_json::to_vec(&body).unwrap_or_default(),
                        };
                        crate::cellular::dataset_session::DatasetRequest {
                            request_id,
                            payload: rmp_serde::to_vec(&wire).unwrap_or_default(),
                        }
                    })
                    .collect();
                let chunk_id = publisher.add(requests);
                if let Some(phaser) = &phaser {
                    phaser.advance(
                        crate::cellular::phaser::PhaseTransition::ShardsAvailable(chunk_id + 1),
                    );
                }
                start = end;
            }
            publisher.finalize();
            tracing::info!(
                total,
                chunks = publisher.chunk_count(),
                endpoint = %chat_url,
                "dataset fan-out: broadcast the endpoint-ready request bodies to the cells"
            );
            Some(server)
        } else {
            None
        };

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
                build_cell_envelope(envelope, kind, cell_id, cell_count, &cell_dir, injected_seed)?;
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

        // Insert a reduction TREE of aggregators between the cells and the controller.
        // Each cell ships to its round-robin tier-1 aggregator; each aggregator merges
        // its subtree and ships ONE store up (to a parent aggregator for a lower tier,
        // the controller for the top tier), so the controller collects only the top
        // tier's partitions instead of `cells`. Fold-only (sketch / exact-fold): the
        // retain path keeps the star topology (needs global order).
        //
        // Placement differs by deployment, exactly like the cells:
        // - SAME-HOST (`!is_k8s`): the controller spawns every tier's `aiperf
        //   --aggregator` subprocess at a distinct fixed loopback port
        //   ([`aggregator_nodes`]) and injects each cell's tier-1 loopback ship address
        //   (via `CellLaunchContext::aggregator_count`). Depth `>= 2` tiers reduce a
        //   large cell count geometrically.
        // - K8S: the operator created a SINGLE tier of aggregator pods and injected each
        //   cell pod's ship-DNS, so the controller must NOT spawn and must NOT inject
        //   loopback ship addresses (`K8sLauncher` ignores `aggregator_count` — cell env
        //   is the pod spec's). It sizes `expected_partitions = M` and collects the M
        //   merged stores. This k8s "expect, don't spawn" path is gated on the operator
        //   having signalled it wired the aggregators ([`AGG_DNS_TEMPLATE_ENV`]); a
        //   fanout-set k8s run without that signal fails closed to the flat star.
        //   Multi-tier k8s is a TODO — the operator builds one tier today (see
        //   `src/aiperf/kubernetes/jobset.py`).
        use crate::engine::cellular_aggregator::{
            aggregator_base_port as agg_base_port, aggregator_count as requested_agg_count,
            effective_aggregator_count, tier_counts_from_env, AGG_DNS_TEMPLATE_ENV,
        };
        let aggregator_base_port = agg_base_port();
        // Same-host uses the full multi-tier plan; k8s stays single-tier (operator-built).
        let tiers: Vec<u32> = if is_k8s {
            let requested = requested_agg_count(cell_count);
            let k8s_wired = std::env::var_os(AGG_DNS_TEMPLATE_ENV).is_some();
            let effective = effective_aggregator_count(is_k8s, k8s_wired, requested);
            if requested.is_some() && effective.is_none() {
                tracing::warn!(
                    "AIPERF_CELL_AGG_FANOUT requests aggregators but the operator did not wire \
                     the k8s aggregators (AIPERF_CELL_AGG_DNS_TEMPLATE unset); falling back to \
                     the flat star topology"
                );
            }
            effective.into_iter().collect()
        } else {
            tier_counts_from_env(cell_count)
        };
        // Cells ship to tier 1 (the first tier); the controller collects the top tier.
        // Both equal `cell_count` for the flat star.
        let aggregator_count = tiers.first().copied();
        let expected_partitions = tiers.last().copied().unwrap_or(cell_count);
        if tiers.len() > 1 {
            tracing::info!(?tiers, "cellular multi-tier aggregation tree");
        }
        // Spawn every tier's aggregator subprocess (same-host only) before the cells so
        // they are bound and collecting by the time cells ship (cell `connect` also
        // retries). Each gets the run envelope on stdin (for the merge config) and its
        // placement (id, bind port, collect barrier, parent ship coordinate) via env. On
        // k8s the aggregators are operator-created pods, so the controller expects rather
        // than spawns — `aggregator_children` stays empty and pod liveness is the
        // operator's concern.
        let mut aggregator_children = if !is_k8s && !tiers.is_empty() {
            spawn_aggregators(
                envelope,
                cell_count,
                aggregator_base_port,
                &cell_coordinate,
            )
            .await
            .context("spawning aggregators")?
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
        // Watch each aggregator for hard failure, so a dead
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
        // `AIPERF_CELL_BARRIER_FREE=1` triggers START without gathering all registrations:
        // the controller triggers START IMMEDIATELY, without gathering all N
        // registrations. A cell that registers *after* the trigger sees the completed
        // event instantly (velo's completed-event cache), so each cell starts as soon
        // as it has its envelope — no O(N) rendezvous. The tradeoff is looser start
        // correlation across cells (arrival-epoch jitter), which is aggregate-equivalent
        // (the same bar as rate/ramp) and does not affect data-deterministic metrics.
        // A failed cell is still caught by the collect loop's failure watch below.
        if barrier_free {
            tracing::info!(
                "barrier-free start: triggering immediately without the O(N) register \
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
        // Drive run-wide START through the monotonic phaser
        // (generation 1 = Started). Cells that subscribed with `PhaserClient` wake here;
        // a cell registering after this sees the completed generation via replay.
        if let Some(phaser) = &phaser {
            phaser.advance(crate::cellular::phaser::PhaseTransition::Started);
        }

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
        // byte-exact global-order/concatenation merge) or a `StorePartition`
        // (metrics-only exact-fold: the cell's folded exact store, no record Vec). Count
        // BOTH toward the one-per-cell termination barrier; the merge below dispatches on
        // which kind arrived.
        let mut partitions: Vec<RecordsShardPartition> = Vec::with_capacity(cell_count as usize);
        let mut store_partitions: Vec<ColumnStorePartition> =
            Vec::with_capacity(cell_count as usize);
        let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
        // The frontend tails this file into AIPerfJob CR status.
        let live_progress_log = std::env::var_os("AIPERF_CELLULAR_HEARTBEAT_LOG")
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from);
        let collected = |records: &[RecordsShardPartition], stores: &[ColumnStorePartition]| {
            records.len() + stores.len()
        };
        // In the flat topology this is one partition per cell; with aggregators it is one
        // MERGED partition per aggregator (`expected_partitions == aggregator count`).
        while collected(&partitions, &store_partitions) < expected_partitions as usize {
            tokio::select! {
                biased;
                message = transport.recv() => match message.context("receiving from cell")? {
                    Some(CellMessage::Partition(partition)) => partitions.push(partition),
                    Some(CellMessage::StorePartition(partition)) => store_partitions.push(*partition),
                    Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                        heartbeats.insert(cell_id, *heartbeat);
                        // Emit the running cross-cell aggregate for live CR-status progress.
                        emit_live_progress(live_progress_log.as_deref(), &heartbeats);
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

        // A metrics-only exact-fold run ships folded stores, appended by cell_id
        // (`merge_store_partitions`) within tolerance
        // (counts/percentiles/min/max exact; sums/means a few ULPs), the same bar the
        // in-process sharded exact-fold merge meets. Otherwise the cells shipped raw
        // records: scheduled cells pre-tile a global dispatch ordinal (byte-exact global
        // order); graph records carry a LOCAL per-cell request_index (concatenated by
        // cell_id, densely re-numbered). Cells never mix the two kinds in one run.
        // Multi-turn backstop: a multi-turn run is sound ONLY on the exact-fold concat
        // merge (cells ship folded StorePartitions). The gate predicts exact-fold from the
        // envelope, but one cell-side disqualifier — a live-reply multi-turn `inputs.json`
        // captured during the run — needs the compiled dataset the controller never loads,
        // so a cell can still fall to retain and ship raw records. Merging those in global
        // dispatch order would silently reorder / re-sample a multi-turn report, so fail
        // loud instead. Scheduled-only: a graph run partitions by whole trace and merges by
        // concatenation regardless of turn count.
        if kind.enforces_multiturn_retain_backstop()
            && cellular_run_is_multi_turn(envelope)
            && !partitions.is_empty()
        {
            bail!(
                "multi-turn cellular run received {} raw-record (retain-path) partition(s): \
                 multi-turn cellular requires the exact-fold merge (every cell must ship a \
                 folded store). A cell fell to the retain path — most likely a live-reply \
                 multi-turn inputs.json captured during the run, or AIPERF_RUNTIME_EXACT_FOLD=0. \
                 Run single-turn, disable inputs.json, or remove the retain-forcing option",
                partitions.len()
            );
        }
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
        // (1) coordinator metadata (distribution_id / workload /
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
        // Create the report's parent so a fresh artifact_dir the orchestrator has not
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
        // stays authoritative; this is the cross-cell live-lane aggregate.
        let mut aggregate = heartbeats.into_values();
        if let Some(mut merged_heartbeat) = aggregate.next() {
            for heartbeat in aggregate {
                merged_heartbeat.merge(&heartbeat);
            }
            write_heartbeat_sidecar(report_path, &merged_heartbeat)
                .context("writing merged cellular heartbeat")?;
        }

        // When cross-host HTTP shipping is active, wait for
        // every cell to POST its files AND its `/done` marker before concatenating, so
        // `temp_root/cell-{id}` is complete. (Same-host cells write their files locally
        // before shipping their velo partition, so the partition-collection loop above
        // is already their barrier.)
        // A dataset-serve-only run never POSTs files or `/done`, so
        // there is nothing to wait for and the barrier would spuriously time out.
        if http_shipping {
            if let Some(receiver) = velo_artifact_receiver.as_ref() {
                receiver
                    .wait_for_cells(cell_count, artifact_upload_timeout())
                    .await
                    .context("waiting for cellular velo artifact streams")?;
            } else if let Some(server) = artifact_server.as_ref() {
                server
                    .wait_for_cells(cell_count, artifact_upload_timeout())
                    .await
                    .context("waiting for cellular artifact uploads")?;
            }
        }

        // Per-record artifact concat + inputs.json copy. Each cell ran its ordinary
        // execute path with a per-cell `temp_root/cell-{id}` dir as its artifact_dir and
        // wrote its merged per-record artifacts (records/raw/CSV/parquet/outputs) there.
        // The controller concatenates them into the real artifact dir (the per-cell dirs
        // are the shards), preserving row-set identity with completion
        // order accepted), before `_scratch` removes `temp_root`. `inputs.json` is NOT
        // concatenated (a single FULL-dataset document, not per-record rows): every cell
        // generated the identical inputs.json over the same resident
        // dataset, so the controller copies ONE cell's copy verbatim
        // (`copy_cell_inputs_json`). inputs.json is always-on (`rust_wire`), so without
        // this the cellular run would silently drop it / break GenAI-Perf compat.
        //
        // The files are controller-local in two cases, both handled here:
        // - SAME-HOST (`!is_k8s`): every cell wrote directly to its controller-local
        //   `temp_root/cell-{id}` dir.
        // - CROSS-HOST (k8s) with `http_shipping`: each pod wrote to its OWN fs, then
        //   shipped every file to the controller over HTTP + streaming zstd,
        //   landing at the SAME `temp_root/cell-{id}/{rel}` paths.
        // Cross-host with shipping DISABLED still skips the concat (the files never
        // reach the controller) — the shared-storage product boundary, warned at start.
        if (!is_k8s || http_shipping)
            && let Some(artifact_dir) = report_path.parent()
        {
            // Read the SHIPPED copies from the landing subtree when HTTP shipping is
            // active (k8s: landing_root == temp_root; same-host force: a separate
            // `http-landing` subtree), else the cells' own local writes under
            // `temp_root/cell-{id}` (default same-host path).
            let concat_source_root = if http_shipping { &landing_root } else { &temp_root };
            let cell_dirs: Vec<PathBuf> = (0..cell_count)
                .map(|cell_id| concat_source_root.join(format!("cell-{cell_id}")))
                .collect();
            // Per-record artifacts (records/raw/CSV/parquet/outputs) are concatenated only
            // when requested; inputs.json (a single full-dataset doc) is copied whenever a
            // cell produced one, independent of the per-record request set. `artifacts` was
            // parsed once at the top of the run (identically to the cell's execute path).
            if !requested_per_record_artifacts(envelope).is_empty() {
                crate::engine::shard_artifacts::concatenate_cell_artifacts(
                    &cell_dirs,
                    artifact_dir,
                    &artifacts,
                )
                .context("concatenating per-cell per-record artifacts")?;
            }
            crate::engine::shard_artifacts::copy_cell_inputs_json(
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

/// The controller's HTTP artifact-upload bind. A fixed routable port
/// (`AIPERF_CONTROLLER_ARTIFACT_BIND`, default `0.0.0.0:9600`) the operator exposes
/// on the controller pod; cells derive the matching authority from their `tcp://`
/// velo coordinate host + the artifact port. Distinct from the velo messaging bind
/// (control plane) — this carries bulk artifact bytes, not coordination.
#[cfg(feature = "cellular")]
fn controller_artifact_bind() -> std::net::SocketAddr {
    use crate::engine::cellular_cell::DEFAULT_ARTIFACT_PORT;
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
#[cfg(feature = "cellular")]
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
#[cfg(feature = "cellular")]
pub(crate) fn collect_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_COLLECT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(7200);
    std::time::Duration::from_secs(secs)
}

/// The deadline for the artifact-upload barrier ([`ArtifactUploadServer::
/// wait_for_cells`](crate::engine::artifact_shipping::ArtifactUploadServer::wait_for_cells)),
/// distinct from [`collect_timeout`]. By the time this barrier runs every cell has
/// already shipped its velo partition (metrics), so only the per-record artifact
/// bytes remain in flight — a few minutes is ample, and a much tighter bound than
/// the whole-run `collect_timeout` (default 2h). Env-overridable
/// (`AIPERF_CELL_ARTIFACT_UPLOAD_TIMEOUT`, seconds; default 5 minutes), so a cell
/// that dies mid-upload fails the run in minutes rather than hours.
#[cfg(feature = "cellular")]
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
#[cfg(feature = "cellular")]
fn register_timeout() -> std::time::Duration {
    let secs = std::env::var("AIPERF_CELL_REGISTER_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    std::time::Duration::from_secs(secs)
}

/// Builds the protocol-v2 envelope for one cell: the same run with its phase
/// budgets sliced to the cell's owned share and its own scratch artifact dir.
/// All cells receive the same dataset and seed; `PartitionedSampler` selects each
/// cell's owned instances from the shared space.
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
    kind: CellularRunKind,
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
    // A graph run's cells partition the trace at runtime (`PartitionedGraphTraceSource`
    // over the SESSION space), so its `sessions` budget must reach every cell WHOLE —
    // slicing it here would double-partition. A scheduled run has no such runtime
    // partition, so its `sessions` budget (multi-turn / `--num-conversations`) IS sliced
    // per cell below. The `kind` names which — see `CellularRunKind::slices_session_budget`.
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
        // Slice the SESSION (conversation) budget per cell for a scheduled multi-turn run,
        // aligned with [`PartitionedSampler`]'s per-conversation stride: cell k owns
        // `owned_positions(total, k, C)` conversations — its share of the first `total`
        // conversation draws `{k, k+C, ...}`. `owned_positions` tiles exactly, so the
        // shares sum to `total` and no cell is handed a short budget. The sampler
        // recycles silently (wraparound),
        // so an off-by-one budget would resample a conversation instead of stopping — a
        // silent correctness trap the exact tiling avoids. Graph cells skip this (they get
        // the whole budget and partition the trace themselves).
        //
        // [`PartitionedSampler`]: crate::dataset::sampler::PartitionedSampler
        if kind.slices_session_budget()
            && let Some(sessions) = phase.get("sessions").and_then(serde_json::Value::as_u64)
        {
            debug_assert_eq!(
                (0..cell_count)
                    .map(|k| owned_positions(sessions, k, cell_count))
                    .sum::<u64>(),
                sessions,
                "cellular session budget must tile exactly across cells"
            );
            let owned = owned_positions(sessions, cell_id, cell_count);
            phase.insert("sessions".to_owned(), serde_json::Value::from(owned));
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

/// Dataset `format`s whose loaders compile exactly one turn
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
/// Every other linear format is rejected. Known multi-turn or session-
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

/// Known MULTI-turn / session-grouping `file`/`public` formats — each compiles rows into
/// multi-turn conversations (by `session_id` grouping or an explicit turn array) in its
/// loader, but partitions cleanly by CONVERSATION and merges correctly on the exact-fold
/// concat path. Admitted to multi-turn cellular ONLY on that path (see
/// [`validate_cellular_run_shape`]); a format outside BOTH this set and
/// [`CELLULAR_SINGLE_TURN_FILE_FORMATS`] still fails closed (unknown / unwired shapes).
const CELLULAR_MULTI_TURN_FILE_FORMATS: [&str; 9] = [
    "multi_turn",
    "mooncake_trace",
    "bailian_trace",
    "burst_gpt",
    "sagemaker_data_capture",
    "inputs_json",
    "sharegpt",
    "hf_conversation",
    "mt_bench",
];

/// Whether a single dataset value compiles STRICTLY one turn per conversation, so its
/// sampler draw index equals the issuer's per-turn dispatch ordinal (the invariant the
/// retain-path global-ordinal merge relies on). A `file`/`public` dataset's turn count is
/// driven by its FORMAT + `session_id` grouping (NOT the top-level `turns` field), so
/// those are single-turn only for a whitelisted single-turn format
/// ([`CELLULAR_SINGLE_TURN_FILE_FORMATS`]); a `synthetic` dataset is single-turn unless
/// its top-level `turns` says otherwise. A dataset that is NOT single-turn is a MULTI-turn
/// conversation, admitted to cellular only on the exact-fold merge.
fn dataset_is_single_turn(dataset: &serde_json::Value) -> bool {
    match dataset.get("type").and_then(serde_json::Value::as_str) {
        Some("file" | "public") => dataset
            .get("format")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|format| CELLULAR_SINGLE_TURN_FILE_FORMATS.contains(&format)),
        // Synthetic (and any non-file/public shape) is single-turn unless `turns` overrides.
        _ => dataset.get("turns").is_none_or(|turns| {
            turns.get("value").and_then(serde_json::Value::as_f64) == Some(1.0)
        }),
    }
}

/// Whether a dataset value targets a graph program (`dag_jsonl`/`weka_trace`/
/// `dynamo_trace`). Graph datasets partition by whole trace and take their own cellular
/// path, so the linear multi-turn gate skips them.
fn is_graph_dataset_value(dataset: &serde_json::Value) -> bool {
    matches!(
        dataset.get("format").and_then(serde_json::Value::as_str),
        Some("dag_jsonl" | "conditional_graph" | "weka_trace" | "dynamo_trace")
    )
}

/// Whether a scheduled (non-graph) cellular run dispatches MULTI-turn conversations —
/// any linear dataset that is not strictly single-turn, or any phase carrying a
/// `sessions` (`--num-conversations`) budget. Multi-turn continuation runs single-process
/// per cell ([`crate::request_rate`]); it is sound in cellular ONLY on the exact-fold
/// concat merge, which the gate ([`validate_cellular_run_shape`]) enforces up front and
/// the merge-time backstop in [`run_cellular`] re-checks against the cells' shipped kind.
fn cellular_run_is_multi_turn(envelope: &serde_json::Value) -> bool {
    let dataset_multi = envelope
        .pointer("/run/cfg/datasets")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|datasets| {
            datasets
                .iter()
                .any(|dataset| !is_graph_dataset_value(dataset) && !dataset_is_single_turn(dataset))
        });
    let session_bounded = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|phases| {
            phases.iter().any(|phase| {
                phase
                    .get("sessions")
                    .and_then(serde_json::Value::as_u64)
                    .is_some()
            })
        });
    dataset_multi || session_bounded
}

/// Whether an envelope carries a phase with an `adaptive_scale` bound (which retains
/// per-turn records per control window, forcing the retain path).
fn cellular_has_adaptive_phase(envelope: &serde_json::Value) -> bool {
    envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|phases| {
            phases.iter().any(|phase| {
                phase
                    .get("adaptive_scale")
                    .is_some_and(|value| !value.is_null())
            })
        })
}

/// Whether a cellular run will merge its cells on the EXACT-FOLD path — each cell folds
/// its records into a dense-LOCAL exact store and ships a `CellMessage::StorePartition`,
/// merged order-independently by `merge_store_partitions` — rather than the RETAIN path
/// (raw records shipped and merged in byte-exact GLOBAL dispatch order by
/// `merge_records_in_global_order`).
///
/// This is the gate for MULTI-turn cellular: a multi-turn conversation dispatches a
/// variable number of turns, so the per-turn global ordinal the retain merge orders by
/// diverges from the sampler's per-conversation draw index (the documented single-turn
/// restriction). The concat merge is order-independent, so multi-turn merges correctly
/// there and only there.
///
/// This applies the cell's `execute::exact_fold` decision inputs from the shared
/// envelope and process environment. Every cell uses identical inputs, so the controller can
/// predict the fold path they all take. It reads only the retain-forcing signals that
/// are RELIABLE at the controller without loading the dataset: the env force-switch
/// (`AIPERF_RUNTIME_EXACT_FOLD=0`), the heartbeat lane, sketch storage, an adaptive-scale
/// phase, and — on a lite build — a requested Parquet sidecar.
///
/// The one cell-side disqualifier it deliberately does NOT model is a live-reply
/// multi-turn `inputs.json` that must be captured DURING the run
/// (`execute::wants_per_record_artifacts` via `inputs_need_retain`), because deciding it
/// needs the compiled dataset's per-conversation context mode, which the controller does
/// not load. That case is caught instead by the merge-time backstop in [`run_cellular`]
/// (a multi-turn run whose cells shipped retain partitions bails), so a false "exact-fold"
/// here can never silently corrupt a merge — at worst it defers a clear gate error to an
/// equally clear merge error.
fn cellular_will_use_exact_fold(envelope: &serde_json::Value) -> bool {
    // The env force-switch routes every path to retain for A/B.
    if !crate::engine::execute::exact_fold_enabled_by_env() {
        return false;
    }
    // The single-process cellular heartbeat lane keeps a per-record clone the fold drops.
    if crate::engine::heartbeat_lane::HeartbeatLane::enabled_by_env() {
        return false;
    }
    // Sketch storage has its own bounded t-digest fold; it ships no StorePartition yet.
    if cellular_metrics_config(envelope).is_ok_and(|config| {
        matches!(
            config.storage_mode,
            crate::metrics_core::MetricsStorageMode::Sketch { .. }
        )
    }) {
        return false;
    }
    // An adaptive-scale phase retains per-turn records per control window.
    if cellular_has_adaptive_phase(envelope) {
        return false;
    }
    // A Parquet sidecar streams (staying fold-eligible) only under the `parquet` feature;
    // a lite runner keeps the run on retain to write it from retained records.
    #[cfg(not(feature = "parquet"))]
    if envelope
        .pointer("/run/cfg/artifacts/records_parquet_path")
        .is_some_and(|value| !value.is_null())
    {
        return false;
    }
    true
}

// Reject cellular shapes without supported issuance and partitioning. HTTP and
// gRPC support single-turn linear datasets; graph programs use whole-trace
// partitioning. File and public datasets require an allowlisted single-turn
// format because loader grouping is independent of the top-level turn count.
fn validate_cellular_run_shape(envelope: &serde_json::Value) -> Result<()> {
    if let Some(transport) = envelope
        .pointer("/run/cfg/transport/type")
        .and_then(serde_json::Value::as_str)
    {
        ensure!(
            matches!(transport, "http" | "grpc"),
            "cellular is wired for transport.type=\"http\" or \"grpc\"; got {transport:?}. \
             Both run the shared online-scheduled executor (the cell issuer + records \
             shipper live above the transport, so gRPC ships partitions exactly as HTTP); \
             the `dynosim_*` offline/online SimClock executors are a separate driver without \
             cell issuance or record shipping, so they fail closed here"
        );
    }
    // Multi-turn conversations merge correctly only on the exact-fold concat path;
    // computed once so the per-dataset gate below can admit them there and reject them on
    // retain. Single-turn runs never read it (they take the `continue`).
    let exact_fold = cellular_will_use_exact_fold(envelope);
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
            Some("dag_jsonl" | "conditional_graph" | "weka_trace" | "dynamo_trace")
        ) {
            continue;
        }
        let kind = dataset.get("type").and_then(serde_json::Value::as_str);
        ensure!(
            matches!(kind, Some("synthetic" | "file" | "public")),
            "cellular runs support synthetic, file, or public datasets (whose conversations the \
             sampler can partition one-draw-per-turn); got dataset type {kind:?}. A cross-host \
             `file`/`path` dataset is shipped controller->cell over HTTP+zstd and \
             recompiled per cell; inline `records` and `public` URL/HF each cell resolves itself"
        );
        // A strictly single-turn dataset (whitelisted file/public format + `turns == 1`,
        // or a synthetic without a `turns` override) keeps the established one-draw==one-turn
        // shape both merge paths support. This is the common case; everything below is the
        // multi-turn extension.
        if dataset_is_single_turn(dataset) {
            continue;
        }
        // MULTI-turn dataset. A multi-turn conversation dispatches a variable number of
        // turns, so its per-turn dispatch ordinal diverges from the sampler's
        // per-conversation draw index — which the RETAIN merge orders by. It is sound in
        // cellular ONLY on the exact-fold concat merge (order-independent store
        // concatenation), where per-turn order is irrelevant to the merged report. The
        // per-cell partition unit (conversation, via [`PartitionedSampler`]) matches
        // the draw unit, and the session budget is sliced by conversation
        // (`build_cell_envelope`), so each cell single-passes its owned conversation slice.
        ensure!(
            exact_fold,
            "cellular runs support multi-turn conversations only on the exact-fold merge \
             path (metrics-only, order-independent store concatenation); this run selected \
             the RETAIN path (raw records merged in global dispatch order), where a \
             multi-turn conversation's variable per-turn dispatch ordinal diverges from the \
             sampler's per-conversation draw index and silently reorders / re-samples the \
             merged report. Remove the retain-forcing options (sketch metrics, an \
             adaptive-scale phase, the cellular heartbeat lane, AIPERF_RUNTIME_EXACT_FOLD=0, \
             or a Parquet sidecar on a lite runner), or run single-turn (turns == 1)"
        );
        // A file/public multi-turn dataset must carry a KNOWN multi-turn format so an
        // unknown / unwired format still fails closed (it may not compile, or may not
        // partition by conversation). Synthetic multi-turn is format-free (regenerated
        // from the shared seed), so it skips this check.
        if matches!(kind, Some("file" | "public")) {
            let format = dataset.get("format").and_then(serde_json::Value::as_str);
            ensure!(
                format.is_some_and(|format| CELLULAR_MULTI_TURN_FILE_FORMATS.contains(&format)),
                "cellular multi-turn file/public datasets support only known multi-turn formats \
                 ({}); got format {format:?}. An unknown or single-turn-only format cannot be \
                 admitted as multi-turn",
                CELLULAR_MULTI_TURN_FILE_FORMATS.join("/")
            );
        }
        // Determinism scope: only the sequential/shuffle samplers give a deterministic
        // single-pass per-cell partition — each cell replays the shared inner draw
        // sequence and keeps its `position % cell_count == cell_id` stride. Random
        // sampling WITH REPLACEMENT has no stable single-pass conversation set to
        // partition, so a multi-turn cellular random run cannot reproduce the 1-cell
        // instance space. Reject it (sequential/shuffle only). Single-turn random cellular
        // is unaffected — it takes the `continue` above.
        let sampling = dataset
            .get("sampling")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("sequential");
        ensure!(
            !sampling.eq_ignore_ascii_case("random"),
            "multi-turn cellular runs require a deterministic single-pass sampler \
             (sequential or shuffle) so each cell owns a fixed conversation slice; got \
             sampling {sampling:?} (random-with-replacement has no stable per-cell \
             partition). Use sequential/shuffle sampling, or run single-turn"
        );
    }
    // Every cell must compose the same dataset space. When `run.random_seed` is absent,
    // the controller derives one
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

/// The serve plan for a cross-host `file`/`path` dataset source: the exact
/// file set the controller registers (keyed by flat relative name) plus the
/// [`DatasetManifest`](crate::engine::artifact_shipping::DatasetManifest)
/// a cell fetches to reconstruct the tree and rewrite `datasets/0.path`.
///
/// A graph trace (`dag_jsonl` / `weka_trace` / `dynamo_trace` / `aiperf_trace`) is
/// enumerated by the graph loader's OWN read set
/// ([`enumerate_recorded_trace_files`](crate::graph::recorded::enumerate_recorded_trace_files)),
/// so a DIRECTORY or SEGMENTED-PREFIX trace ships every shard the 1-cell run would
/// read — no over/under-ship. Any other (scheduled) `file`/`path` dataset is a
/// single file (its multi-file directory shapes, e.g. `raw_payload`, are out of
/// scope and still fail closed).
///
/// Fails closed on a missing/unreadable path, an empty trace directory, an
/// unmatched segmented-prefix, a `dag_jsonl` directory/prefix (its loader reads one
/// file), or a duplicate shard file name — exactly the errors the loader would
/// raise, surfaced before the run launches cells. Only invoked when the run is
/// genuinely cross-host and shipping-enabled, where the controller holds the source
/// locally so its on-disk shape is authoritative.
fn build_dataset_serve_plan(
    format: Option<&str>,
    source: &Path,
) -> Result<(
    std::collections::HashMap<String, PathBuf>,
    crate::engine::artifact_shipping::DatasetManifest,
)> {
    use crate::engine::artifact_shipping::DatasetManifest;
    use crate::graph::recorded::{RecordedTracePathKind, enumerate_recorded_trace_files};

    let file_name = |path: &Path| -> Result<String> {
        Ok(path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("dataset path {} has no file name", path.display()))?
            .to_owned())
    };

    let is_graph_format = matches!(
        format,
        Some("dag_jsonl" | "conditional_graph" | "weka_trace" | "dynamo_trace" | "aiperf_trace")
    );
    if is_graph_format {
        let (kind, base_name, files) =
            enumerate_recorded_trace_files(format.expect("matched Some above"), source)
                .map_err(|error| anyhow!(error.to_string()))
                .with_context(|| {
                    format!(
                        "enumerating cross-host graph trace files for {}",
                        source.display()
                    )
                })?;
        let kind_str = match kind {
            RecordedTracePathKind::File => "file",
            RecordedTracePathKind::Directory => "dir",
            RecordedTracePathKind::SegmentedPrefix => "prefix",
        };
        let mut map = std::collections::HashMap::with_capacity(files.len());
        let mut names = Vec::with_capacity(files.len());
        for path in files {
            let name = file_name(&path)?;
            ensure!(
                map.insert(name.clone(), path).is_none(),
                "cross-host graph trace has two shards with the same file name {name:?}; \
                 shard names must be unique to reconstruct the tree"
            );
            names.push(name);
        }
        Ok((
            map,
            DatasetManifest {
                kind: kind_str.to_owned(),
                base_name,
                files: names,
            },
        ))
    } else {
        // A scheduled `file`/`path` dataset (single_turn, mooncake_trace, ...) always
        // ships as one file. Its multi-file directory shapes fail closed here.
        ensure!(
            source.is_file(),
            "cross-host cellular runs support a single-file dataset for this format; the path \
             {} is not a single readable file (directory or missing). Ship a single file, mount \
             a shared volume, or use an inline `records` dataset",
            source.display()
        );
        let name = file_name(source)?;
        let map = std::iter::once((name.clone(), source.to_path_buf())).collect();
        Ok((
            map,
            DatasetManifest {
                kind: "file".to_owned(),
                base_name: name.clone(),
                files: vec![name],
            },
        ))
    }
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
///   duration bound needs the graph path's ragged-count merge;
///   `adaptive_scale` needs cross-cell scaling consensus); or
/// - has a `concurrency`/`prefill_concurrency` cap below `cell_count` — the `.max(1)`
///   per-cell floor would then over-subscribe the aggregate in-flight to `cell_count`
///   (enforced by [`ensure_cellular_cap_floor`], shared with the graph path).
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
    // A `sessions` (multi-turn / --num-conversations) budget merges cleanly only on the
    // exact-fold concat path (like the multi-turn dataset gate); on retain its per-turn
    // dispatch ordinal diverges from the sampler's per-conversation draw index.
    let exact_fold = cellular_will_use_exact_fold(envelope);
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
        // Every phase must be bounded by a partitionable budget: a `requests` (turn) budget
        // — the single-turn shape both merge paths support — OR, on the exact-fold merge, a
        // `sessions` (conversation) budget for multi-turn continuation. Both tile per cell
        // via `owned_positions` (`build_cell_envelope`), so each must be >= cell_count or a
        // cell owns zero (the silently-recycling sampler would then resample instead of
        // stopping).
        let requests = phase.get("requests").and_then(serde_json::Value::as_u64);
        let sessions = phase.get("sessions").and_then(serde_json::Value::as_u64);
        ensure!(
            requests.is_some() || sessions.is_some(),
            "cellular runs require every phase to carry a `requests` budget (single-turn) or a `sessions` budget (multi-turn, exact-fold only); phase {name:?} has neither"
        );
        if let Some(requests) = requests {
            ensure!(
                requests >= cells,
                "cellular runs require a `requests` budget >= cell_count ({cell_count}); phase {name:?} has {requests}"
            );
        }
        if let Some(sessions) = sessions {
            ensure!(
                exact_fold,
                "cellular runs support a `sessions` (multi-turn / --num-conversations) budget only on the exact-fold merge path (order-independent store concatenation); phase {name:?} carries `sessions` on the RETAIN path, where a multi-turn conversation's per-turn dispatch ordinal diverges from the sampler's per-conversation draw index. Remove the retain-forcing options, or bound the phase by `requests`"
            );
            ensure!(
                sessions >= cells,
                "cellular runs require a `sessions` budget >= cell_count ({cell_count}) so each cell owns at least one conversation (a shorter budget makes the silently-recycling sampler resample instead of stopping); phase {name:?} has {sessions}"
            );
        }
        // A `duration`/`adaptive_scale` bound still breaks the merge: a duration bound
        // needs the graph path's ragged-count merge and `adaptive_scale` needs
        // cross-cell scaling consensus. (`sessions` is allowed on
        // exact-fold, rejected on retain — rather than blanket-rejected here.)
        ensure!(
            phase.get("duration").is_none() && phase.get("adaptive_scale").is_none(),
            "cellular runs require a fixed per-phase budget; phase {name:?} carries a `duration`/`adaptive_scale` bound whose actual dispatch count can diverge from the sliced budget and break the merge"
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
        ensure_cellular_cap_floor(phase, name, cell_count)?;
    }
    Ok(())
}

/// Ensures a phase's `concurrency` / `prefill_concurrency` caps are each at least
/// `cell_count` — the shared floor both the scheduled and graph cellular paths enforce.
///
/// [`build_cell_envelope`] splits every cap round-robin per cell with a `.max(1)` floor,
/// so a cap below `cell_count` collapses to 1 per cell and the cells' caps sum to
/// `cell_count` — silently over-subscribing the aggregate in-flight. `name` labels the
/// offending phase. The reason is identical for both run kinds, so the message does not
/// name graph-vs-scheduled.
fn ensure_cellular_cap_floor(phase: &serde_json::Value, name: &str, cell_count: u32) -> Result<()> {
    let cells = u64::from(cell_count);
    for cap in ["concurrency", "prefill_concurrency"] {
        if let Some(value) = phase.get(cap).and_then(serde_json::Value::as_u64) {
            ensure!(
                value >= cells,
                "cellular runs require a `{cap}` cap >= cell_count ({cell_count}) so it splits evenly (a smaller cap floors to 1 per cell and over-subscribes the aggregate to cell_count); phase {name:?} has {value}"
            );
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
        ensure_cellular_cap_floor(phase, name, cell_count)?;
    }
    Ok(())
}

/// Spawns one `aiperf --aggregator` subprocess per aggregator node across EVERY tier of
/// the same-host reduction tree ([`aggregator_nodes`]). Each receives the run envelope
/// on stdin for `MetricsConfig` and its placement via env: its id, the fixed loopback
/// `tcp://` coordinate it binds (its children dial it there), its collect barrier, and
/// where it ships its one merged store — a parent aggregator's loopback coordinate for
/// a lower tier, or the controller for the top tier. Returns the children so the
/// controller watches them for hard failure. `kill_on_drop` tears them down on any
/// controller abort. The fanout is read from the environment; a single-tier tree spawns
/// exactly the original topology.
async fn spawn_aggregators(
    envelope: &serde_json::Value,
    cell_count: u32,
    base_port: u16,
    controller_coordinate: &str,
) -> Result<Vec<tokio::process::Child>> {
    use crate::engine::cellular_aggregator::{
        aggregator_nodes, tier_counts_from_env, ShipTarget, AGG_BIND_ENV, AGG_CHILD_COUNT_ENV,
        AGG_ID_ENV, AGG_SHIP_ADDR_ENV,
    };
    use crate::engine::cellular_cell::CELL_CONTROLLER_ADDR_ENV;
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;

    // Recover the fanout the tier plan was sized with (the plan is derived from the same
    // env), so `aggregator_nodes` computes the identical port/parent layout.
    let fanout: u32 = std::env::var(crate::engine::cellular_aggregator::CELL_AGG_FANOUT_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .context("spawn_aggregators called without AIPERF_CELL_AGG_FANOUT")?;
    debug_assert!(!tier_counts_from_env(cell_count).is_empty());
    let nodes = aggregator_nodes(cell_count, fanout, base_port);

    let envelope_bytes =
        serde_json::to_vec(envelope).context("serializing envelope for aggregators")?;
    let exe = std::env::current_exe().unwrap_or_else(|_| "aiperf runner".into());
    let mut children = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let mut command = tokio::process::Command::new(&exe);
        command
            .arg("--aggregator")
            .env(AGG_ID_ENV, node.id.to_string())
            .env(AGG_BIND_ENV, format!("tcp://127.0.0.1:{}", node.bind_port))
            .env(AGG_CHILD_COUNT_ENV, node.child_count.to_string())
            // The controller coordinate is carried for the top tier (which ships there);
            // a lower tier ships to its parent via AGG_SHIP_ADDR below and ignores it.
            .env(CELL_CONTROLLER_ADDR_ENV, controller_coordinate);
        if let ShipTarget::Aggregator(port) = node.ship {
            command.env(AGG_SHIP_ADDR_ENV, format!("tcp://127.0.0.1:{port}"));
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning aggregator tier {} id {}", node.tier, node.id))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&envelope_bytes)
                .await
                .with_context(|| format!("writing envelope to aggregator {}", node.id))?;
            // `stdin` drops here → EOF, so the aggregator's `read_to_end` returns.
        }
        children.push(child);
    }
    Ok(children)
}

/// Builds the native metrics policy for a cellular merge from `cfg.metrics` and
/// `cfg.endpoint.use_server_token_count`.
///
/// An absent or loose `metrics` block uses its defaults. The resulting policy still
/// validates configured SLO names and preserves authored SLOs and timeslice intervals.
pub(crate) fn cellular_metrics_config(envelope: &serde_json::Value) -> Result<MetricsConfig> {
    let spec: crate::engine::protocol::MetricsSpec = envelope
        .pointer("/run/cfg/metrics")
        .cloned()
        .map(|value| serde_json::from_value(value).unwrap_or_default())
        .unwrap_or_default();
    let use_server_token_count = envelope
        .pointer("/run/cfg/endpoint/use_server_token_count")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    crate::engine::execute::metrics_config(&spec, use_server_token_count)
}

/// Warns, once at controller startup, when a cellular run carries side-channel
/// telemetry sidecars (`server_metrics` / `gpu_telemetry` / `network_latency`). Each
/// cell scrapes them into its own scratch tree, which the controller discards — so
/// these metrics are omitted from the merged report (the documented report-fidelity
/// gap), unlike a single-process run. This is surfaced as a loud runtime warning
/// rather than a silent drop or a fail-closed rejection: `gpu_telemetry` and
/// `server_metrics` default *on*, so rejecting any present sidecar would refuse nearly
/// every cellular run.
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

// Cross-host cells cannot return local per-record files through the control
// plane; bulk artifacts require shared storage.
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
        // The base accumulates prior phases' TURN counts to stamp the retain-path global
        // dispatch ordinal. A session-bounded (multi-turn) phase has no fixed turn count
        // up front and runs exact-fold-only, where the cell uses its own dense-LOCAL fold
        // ordinal and never reads this base — so a missing `requests` contributes 0 rather
        // than failing. (A retain-path phase always carries `requests`, so its base stays
        // exact.)
        let requests = phase
            .get("requests")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        base += requests;
    }
    Ok(bases)
}

/// The profiling phase's dispatch budget from the v2 envelope — its `requests` (turn)
/// budget for a single-turn run, or its `sessions` (conversation) budget for a multi-turn
/// exact-fold run. Used only to require a bounded profiling phase (the caller
/// discards the value); the per-phase budget/shape checks live in
/// [`validate_cellular_phase_budgets`].
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
            && let Some(budget) = phase
                .get("requests")
                .or_else(|| phase.get("sessions"))
                .and_then(serde_json::Value::as_u64)
        {
            return Ok(budget);
        }
    }
    bail!("cellular runs require a profiling phase with a `requests` or `sessions` budget")
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
/// encode). The exact report percentiles remain authoritative.
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
    use crate::engine::cellular_kind::is_graph_dataset;

    #[test]
    fn rejects_non_shipping_run_shapes() {
        for ok in [
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "endpoint": {"urls": ["http://a", "http://b"]}}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {
                "transport": {"type": "http"},
                "datasets": [{"type": "synthetic", "turns": {"value": 1}}],
                "endpoint": {"urls": ["http://x/v1"]},
            }}}),
            serde_json::json!({"run": {"random_seed": 1, "cfg": {
                "datasets": [{"type": "synthetic"}],
            }}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "single_turn", "path": "/data/prompts.jsonl"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "single_turn", "records": []}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "public", "format": "accuracy"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "format": "raw_payload", "path": "/data/p.jsonl"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "transport": {"type": "grpc"}}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&ok).is_ok(),
                "should accept {ok}"
            );
        }
        for bad in [
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "synthetic"}], "transport": {"type": "dynosim_offline"}}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "agentic"}]}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {}}}),
            serde_json::json!({"run": {"random_seed": 42, "cfg": {"datasets": [{"type": "file", "path": "/data/t.jsonl"}]}}}),
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
        let multi_turn_trace = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "file", "format": "mooncake_trace", "path": "/data/t.jsonl"}],
        }}});
        assert!(
            validate_cellular_run_shape(&multi_turn_trace).is_ok(),
            "mooncake_trace (known multi-turn format) must be admitted on the exact-fold merge"
        );
        assert!(
            !is_graph_dataset(&multi_turn_trace),
            "mooncake_trace is not a graph dataset (takes the scheduled partition)"
        );
        let synthetic = serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}});
        assert!(
            !is_graph_dataset(&synthetic),
            "synthetic is not a graph dataset"
        );
    }

    #[test]
    fn admits_multi_turn_on_exact_fold() {
        for ok in [
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic", "turns": {"value": 3}}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic", "turns": {"mean": 2.0, "stddev": 1.0}}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "multi_turn", "path": "/data/t.jsonl"}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "mooncake_trace", "path": "/data/t.jsonl"}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "inputs_json", "path": "/data/t.jsonl"}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "public", "format": "sharegpt"}]}}}),
        ] {
            assert!(
                validate_cellular_run_shape(&ok).is_ok(),
                "multi-turn should be admitted on exact-fold: {ok}"
            );
            assert!(
                cellular_run_is_multi_turn(&ok),
                "run should be detected as multi-turn: {ok}"
            );
        }
        let unknown = serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "not_a_real_format", "path": "/data/t.jsonl"}]}}});
        assert!(
            validate_cellular_run_shape(&unknown).is_err(),
            "an unknown multi-turn file format must fail closed"
        );
    }

    #[test]
    fn rejects_multi_turn_on_retain() {
        let retain_multi = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "synthetic", "turns": {"value": 3}}],
            "metrics": {"sketch": true},
        }}});
        assert!(
            !cellular_will_use_exact_fold(&retain_multi),
            "sketch metric storage must force the retain (non-exact-fold) path"
        );
        assert!(
            validate_cellular_run_shape(&retain_multi).is_err(),
            "multi-turn on the retain path must be rejected"
        );
        let retain_single = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "synthetic"}],
            "metrics": {"sketch": true},
        }}});
        assert!(
            validate_cellular_run_shape(&retain_single).is_ok(),
            "single-turn on the retain path is unaffected"
        );
    }

    #[test]
    fn build_dataset_serve_plan_ships_single_dir_and_prefix() {
        let tmp = tempfile::TempDir::new().unwrap();

        let file = tmp.path().join("trace.dag.jsonl");
        std::fs::write(&file, b"{}\n").unwrap();
        let (map, manifest) = build_dataset_serve_plan(Some("dag_jsonl"), &file).unwrap();
        assert_eq!(manifest.kind, "file");
        assert_eq!(manifest.files, vec!["trace.dag.jsonl".to_owned()]);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("trace.dag.jsonl"), Some(&file));

        let dir = tmp.path().join("shards");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.json"), b"{}").unwrap();
        std::fs::write(dir.join("b.json"), b"{}").unwrap();
        let (map, manifest) = build_dataset_serve_plan(Some("weka_trace"), &dir).unwrap();
        assert_eq!(manifest.kind, "dir");
        let mut names = manifest.files.clone();
        names.sort();
        assert_eq!(names, vec!["a.json".to_owned(), "b.json".to_owned()]);
        assert_eq!(map.len(), 2);

        std::fs::write(tmp.path().join("seg.000000.jsonl.gz"), b"{}\n").unwrap();
        std::fs::write(tmp.path().join("seg.000001.jsonl.gz"), b"{}\n").unwrap();
        let prefix = tmp.path().join("seg.jsonl.gz");
        let (map, manifest) = build_dataset_serve_plan(Some("dynamo_trace"), &prefix).unwrap();
        assert_eq!(manifest.kind, "prefix");
        assert_eq!(manifest.base_name, "seg.jsonl.gz");
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("seg.000000.jsonl.gz"));

        assert!(
            build_dataset_serve_plan(Some("dag_jsonl"), &dir).is_err(),
            "a dag_jsonl directory must fail closed (single-file loader)"
        );
        let missing = tmp.path().join("nope.jsonl");
        assert!(build_dataset_serve_plan(Some("weka_trace"), &missing).is_err());
        assert!(build_dataset_serve_plan(Some("single_turn"), &missing).is_err());
        let scheduled = tmp.path().join("prompts.jsonl");
        std::fs::write(&scheduled, b"{}\n").unwrap();
        let (_map, manifest) = build_dataset_serve_plan(Some("single_turn"), &scheduled).unwrap();
        assert_eq!(manifest.kind, "file");
    }

    #[test]
    fn rejects_multi_turn_random_sampler() {
        let random_multi = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "synthetic", "turns": {"value": 3}, "sampling": "random"}],
        }}});
        assert!(
            validate_cellular_run_shape(&random_multi).is_err(),
            "multi-turn cellular with a random sampler must be rejected"
        );
        for sampling in ["sequential", "shuffle", "SEQUENTIAL"] {
            let ok = serde_json::json!({"run": {"cfg": {
                "datasets": [{"type": "synthetic", "turns": {"value": 3}, "sampling": sampling}],
            }}});
            assert!(
                validate_cellular_run_shape(&ok).is_ok(),
                "multi-turn cellular with {sampling:?} sampling must be admitted"
            );
        }
        let random_single = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "synthetic", "sampling": "random"}],
        }}});
        assert!(
            validate_cellular_run_shape(&random_single).is_ok(),
            "single-turn random sampling is unaffected"
        );
    }

    #[test]
    fn session_budget_tiles_exactly_across_cells() {
        let dir = Path::new("/tmp/aiperf-cellular-session-tiling-test");
        for total in [4u64, 7, 12, 60, 101] {
            for count in 1..=6u32 {
                let envelope = serde_json::json!({"run": {"cfg": {
                    "datasets": [{"type": "synthetic", "turns": {"value": 3}}],
                    "phases": [{"type": "concurrency", "name": "profiling", "sessions": total, "concurrency": 8}],
                }}});
                let mut sum = 0u64;
                for cell_id in 0..count {
                    let cell = build_cell_envelope(
                        &envelope,
                        CellularRunKind::detect(&envelope),
                        cell_id,
                        count,
                        dir,
                        None,
                    )
                    .unwrap();
                    let owned = cell
                        .pointer("/run/cfg/phases/0/sessions")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap();
                    sum += owned;
                    if total >= count as u64 {
                        assert!(
                            owned >= 1,
                            "cell {cell_id} owns zero sessions (total {total} count {count})"
                        );
                    }
                }
                assert_eq!(
                    sum, total,
                    "session shares must sum to total (total {total} count {count})"
                );
            }
        }
        let graph = serde_json::json!({"run": {"cfg": {
            "datasets": [{"type": "file", "format": "dag_jsonl"}],
            "phases": [{"type": "concurrency", "name": "profiling", "sessions": 60, "concurrency": 8}],
        }}});
        for cell_id in 0..4u32 {
            let cell = build_cell_envelope(
                &graph,
                CellularRunKind::detect(&graph),
                cell_id,
                4,
                dir,
                None,
            )
            .unwrap();
            assert_eq!(
                cell.pointer("/run/cfg/phases/0/sessions")
                    .and_then(serde_json::Value::as_u64),
                Some(60),
                "graph cell {cell_id} must keep the whole sessions budget"
            );
        }
    }

    #[test]
    fn phase_budgets_accept_sessions_on_exact_fold() {
        let ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "profiling", "sessions": 60, "concurrency": 8},
        ]}}});
        assert!(validate_cellular_phase_budgets(&ok, 4).is_ok());
        let too_few = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "profiling", "sessions": 3, "concurrency": 8},
        ]}}});
        assert!(validate_cellular_phase_budgets(&too_few, 4).is_err());
        let retain = serde_json::json!({"run": {"cfg": {
            "metrics": {"sketch": true},
            "phases": [{"type": "concurrency", "name": "profiling", "sessions": 60, "concurrency": 8}],
        }}});
        assert!(validate_cellular_phase_budgets(&retain, 4).is_err());
        let unbounded = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "profiling", "concurrency": 8},
        ]}}});
        assert!(validate_cellular_phase_budgets(&unbounded, 4).is_err());
    }

    #[test]
    fn run_kind_detects_and_dispatches() {
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
        let env = serde_json::json!({"run": {"cfg": {
            "metrics": {"slos": {"request_latency": 60.0}, "slice_duration_seconds": 2.0},
            "endpoint": {"use_server_token_count": true},
        }}});
        let config = cellular_metrics_config(&env).expect("valid metrics");
        assert_eq!(config.slos.len(), 1);
        assert_eq!(config.slice_duration_ns, Some(2_000_000_000));
        assert!(config.use_server_token_count);
        let bare =
            cellular_metrics_config(&serde_json::json!({"run": {"cfg": {}}})).expect("default");
        assert!(bare.slos.is_empty());
        assert_eq!(bare.slice_duration_ns, None);
        assert!(!bare.use_server_token_count);
        assert!(
            cellular_metrics_config(&serde_json::json!({"run": {"cfg": {
                "metrics": {"slos": {"not_a_real_metric": 1.0}},
            }}}))
            .is_err()
        );
    }

    #[test]
    fn rejects_non_request_bounded_phases() {
        let ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "concurrency", "name": "warmup", "requests": 10, "concurrency": 8},
            {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8, "cancellation": {"rate": 25.0, "delay": 0.5}, "concurrency_ramp": {"start": 1, "end": 100}},
        ]}}});
        assert!(validate_cellular_phase_budgets(&ok, 4).is_ok());
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
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "requests": 100, "concurrency": 8},
                ]}}}),
                4
            )
            .is_err()
        );
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "sessions": 100, "concurrency": 2},
                ]}}}),
                4
            )
            .is_err()
        );
        assert!(
            validate_graph_cellular_phases(
                &serde_json::json!({"run": {"cfg": {"phases": [
                    {"type": "concurrency", "name": "profiling", "sessions": 100, "concurrency": 8},
                ]}}}),
                4
            )
            .is_ok()
        );
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
                    let cell = build_cell_envelope(
                        &envelope,
                        CellularRunKind::detect(&envelope),
                        cell_id,
                        count,
                        dir,
                        None,
                    )
                    .unwrap();
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
        let dir = Path::new("/tmp/aiperf-cellular-envelope-test");
        let envelope = serde_json::json!({"run": {"cfg": {"phases": [
            {"type": "constant", "name": "profiling", "requests": 100, "rate": 40.0,
             "concurrency": 8, "prefill_concurrency": 4},
        ]}}});
        let cell_count = 4u32;
        let mut rate_sum = 0.0;
        for cell_id in 0..cell_count {
            let cell = build_cell_envelope(
                &envelope,
                CellularRunKind::detect(&envelope),
                cell_id,
                cell_count,
                dir,
                None,
            )
            .unwrap();
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
        let metrics_only = serde_json::json!({"run": {"cfg": {"artifacts": {
            "inputs_path": "inputs.json",
        }}}});
        assert!(
            requested_per_record_artifacts(&metrics_only).is_empty(),
            "metrics-only (+ per-session inputs.json) must not flag per-record artifacts"
        );
        assert!(requested_per_record_artifacts(&serde_json::json!({})).is_empty());

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
        let with_files = serde_json::json!({"run": {"cfg": {"artifacts": {
            "records_path": "profile_export.jsonl",
            "records_parquet_path": "profile_export.parquet",
            "outputs_path": "profile_export.json",
        }}}});

        assert!(
            dropped_cross_host_artifacts(&with_files, false).is_empty(),
            "same-host concatenates per-record artifacts; nothing is dropped"
        );

        assert_eq!(
            dropped_cross_host_artifacts(&with_files, true),
            vec!["records_path", "records_parquet_path", "outputs_path"],
            "cross-host drops exactly the requested per-record files"
        );

        let metrics_only = serde_json::json!({"run": {"cfg": {"artifacts": {
            "inputs_path": "inputs.json",
        }}}});
        assert!(
            dropped_cross_host_artifacts(&metrics_only, true).is_empty(),
            "cross-host metrics-only run drops no per-record files (inputs.json is per-session)"
        );
        assert!(dropped_cross_host_artifacts(&serde_json::json!({}), true).is_empty());
    }
}
