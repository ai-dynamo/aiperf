// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cell-mode support for the multi-cell cellular runtime.
//!
//! A *cell* is one `aiperf-runner --cell` process (a local subprocess or a k8s
//! pod). It runs the ordinary online scheduled execution over its budget slice,
//! but with the `CellularAutonomousIssuer` assigning dense global dispatch
//! ordinals from its `(cell_id, cell_count)` partition, and it ships its captured
//! records to the controller over the velo [`transport`](crate::cellular::transport)
//! seam instead of writing a report. The controller re-ingests every cell's
//! records in global ordinal order for the single authoritative `native-v2.json`.
//!
//! Cell behaviour is injected through environment variables so the ordinary
//! execute path is reused unchanged: [`CELL_ID_ENV`](crate::cellular::partition::CELL_ID_ENV)
//! / [`CELL_COUNT_ENV`](crate::cellular::partition::CELL_COUNT_ENV) select the
//! issuer's partition, [`CELL_CONTROLLER_ADDR_ENV`] carries the controller's
//! bootstrap coordinate, and [`CELL_PHASE_ORDINAL_BASES_ENV`] carries each phase's
//! global ordinal base. The launcher sets them on the child; the cell fetches its
//! sliced execute envelope over velo (there is no stdin spec pipe).
//!
//! velo runtimes: the cell touches velo twice — once to fetch its envelope at
//! startup, once to ship its records at the end — and each uses its **own**
//! short-lived runtime (fetch on the caller's runtime; ship on a dedicated thread),
//! so a velo instance is never shared across the cell's thread-per-core execute
//! runtime. Because the ship uses a fresh velo instance, the partition ship carries
//! that instance's `PeerInfo` (see `crate::cellular::CellPartitionShip`).

use std::collections::HashMap;

use crate::cellular::partition::CellPartition;
use crate::cellular::{CellularAutonomousIssuer, IssuanceAuthority, ModuloCellPartition};
use crate::metrics_core::Phase;
use anyhow::Result;

/// Env var carrying the controller's bootstrap coordinate (`file:PATH` locally,
/// `tcp://HOST:PORT` in k8s) — where a cell fetches the controller's `PeerInfo`.
/// The cell id and count live in [`crate::cellular::partition`]'s env vars.
pub const CELL_CONTROLLER_ADDR_ENV: &str = "AIPERF_CELL_CONTROLLER_ADDR";

/// Env var (tier-T2 hierarchical merge) carrying the velo coordinate a cell ships its
/// terminal partition + heartbeat to, when it is NOT the controller. In the flat
/// (star) topology this is unset and a cell ships to the controller
/// ([`CELL_CONTROLLER_ADDR_ENV`]); in the tree topology the controller sets it to the
/// cell's assigned aggregator (`tcp://HOST:PORT`), so the aggregator merges its
/// subtree's stores and ships one merged partition up. Only the ship target changes —
/// the cell still fetches its envelope and awaits START from the controller — so a
/// cell's partition/issuer/sampler behaviour is byte-identical to the flat topology.
pub const CELL_SHIP_ADDR_ENV: &str = "AIPERF_CELL_SHIP_ADDR";

/// Env var carrying the per-phase global ordinal bases as JSON (`{name: base}`), so a
/// cell's issuer recovers each turn's single-cell absolute slot from its phase-local
/// slot (the cell's sampler restarts each phase; see [`phase_ordinal_bases_from_env`]).
pub const CELL_PHASE_ORDINAL_BASES_ENV: &str = "AIPERF_CELL_PHASE_ORDINAL_BASES";

/// Env var carrying the controller's artifact upload `host:port` (Stage E). The
/// operator injects this into k8s pods (or the local launcher sets it to the
/// controller's bound server) so a cell knows where to POST its per-record artifact
/// files. When absent, a cell on a `tcp://` (k8s) controller coordinate derives the
/// host from that coordinate and the port from [`CELL_ARTIFACT_PORT_ENV`].
pub const CELL_ARTIFACT_ADDR_ENV: &str = "AIPERF_CELL_ARTIFACT_ADDR";

/// Env var overriding the controller's artifact-server port when a cell derives the
/// artifact `host:port` from its `tcp://HOST:PORT` velo coordinate (default `9600`,
/// matching the controller's `AIPERF_CONTROLLER_ARTIFACT_BIND` default).
pub const CELL_ARTIFACT_PORT_ENV: &str = "AIPERF_CELL_ARTIFACT_PORT";

/// Env toggle disabling cross-host HTTP artifact shipping (Stage E). Default ON;
/// set to `0`/`false`/`off` to fall back to the shared-storage product boundary (the
/// controller then warns the per-record files are dropped, as before). For operators
/// on a shared-FS (ReadWriteMany PVC) setup who prefer the cells' own writes.
pub const CELL_HTTP_ARTIFACT_SHIPPING_ENV: &str = "AIPERF_CELL_HTTP_ARTIFACT_SHIPPING";

/// The default controller artifact-server port (server bind + cell-derived fetch).
pub const DEFAULT_ARTIFACT_PORT: u16 = 9600;

/// **Test/dev-only force seam.** Env flag that makes a LOCAL (`--cells N`, same-host)
/// run drive the CROSS-HOST HTTP artifact path over loopback instead of the default
/// shared-FS Stage D concat: the controller binds its artifact upload server on
/// `127.0.0.1:0`, injects that authority into each locally-launched cell, and the
/// cells POST their per-record artifact files back over real TCP + streaming zstd —
/// exercising the exact production shipping/upload/concat code that k8s uses, but
/// without a second host. Set to `1`/`true`/`on`/`yes` to enable.
///
/// This exists ONLY so a same-host multi-PROCESS test can prove the HTTP+zstd
/// shipping mechanism end-to-end (see `rust/e2e/tests/test_cellular_http_shipping.rs`).
/// It is NOT a product mode: default local `--cells N` (flag unset) is byte-unchanged
/// — cells write directly to the controller-local scratch and the controller
/// concatenates those writes with no HTTP, exactly as before.
pub const CELL_ARTIFACT_HTTP_FORCE_ENV: &str = "AIPERF_CELL_ARTIFACT_HTTP_FORCE";

/// Whether the [test/dev HTTP-force seam](CELL_ARTIFACT_HTTP_FORCE_ENV) is enabled
/// (default OFF). When ON, the controller routes a same-host cellular run through the
/// real HTTP+zstd artifact-shipping path over loopback rather than the shared-FS
/// concat, so a same-host multi-process test drives the production shipping code.
pub fn artifact_http_force_enabled() -> bool {
    matches!(
        std::env::var(CELL_ARTIFACT_HTTP_FORCE_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// Whether cross-host HTTP artifact shipping is enabled ([`CELL_HTTP_ARTIFACT_SHIPPING_ENV`],
/// default ON). Shared by the controller (whether to start the upload server + run the
/// concat) and the cell (whether to ship), so the two never disagree.
pub fn http_artifact_shipping_enabled() -> bool {
    !matches!(
        std::env::var(CELL_HTTP_ARTIFACT_SHIPPING_ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

/// The controller's artifact upload `host:port` this cell should POST to, or `None`
/// when HTTP shipping is off or this is a same-host launcher (Stage D concatenates
/// the cell's own local writes instead — no HTTP). Resolution order:
/// 1. shipping disabled → `None`;
/// 2. [`CELL_ARTIFACT_ADDR_ENV`] set (operator/launcher) → that authority;
/// 3. a `tcp://HOST:PORT` velo controller coordinate with a **routable** `HOST`
///    (k8s) → `HOST` + the [`CELL_ARTIFACT_PORT_ENV`] port (default
///    [`DEFAULT_ARTIFACT_PORT`]);
/// 4. otherwise (a `tcp://` **loopback** or `uds://` local coordinate) → `None`
///    (a co-located run concatenates the cell's own local writes; no HTTP).
pub fn cell_artifact_authority() -> Option<String> {
    if !http_artifact_shipping_enabled() {
        return None;
    }
    if let Ok(addr) = std::env::var(CELL_ARTIFACT_ADDR_ENV)
        && !addr.is_empty()
    {
        return Some(addr);
    }
    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()?;
    let host_port = coordinate.strip_prefix("tcp://")?;
    // The velo coordinate host, with the artifact-server port (the velo port is a
    // different service).
    let host = host_port
        .rsplit_once(':')
        .map_or(host_port, |(host, _)| host);
    // A loopback coordinate is a co-located (local) run — the controller runs no
    // HTTP upload server there (Stage D concatenates the cells' shared-FS writes),
    // so unless an explicit `CELL_ARTIFACT_ADDR` forced it above, ship nothing.
    if host
        .parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(host.eq_ignore_ascii_case("localhost"))
    {
        return None;
    }
    let port = std::env::var(CELL_ARTIFACT_PORT_ENV)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(DEFAULT_ARTIFACT_PORT);
    Some(format!("{host}:{port}"))
}

/// The per-phase global ordinal bases for this cell, keyed by metric phase — the
/// turns the run's prior phases dispatched globally — or empty when the process is
/// not a cell (the single-process path stamps the flat cumulative slot). A cell's
/// sampler restarts each phase at 0, so the autonomous issuer adds a phase's base to
/// a turn's phase-local slot to recover the absolute slot a single-cell run assigns.
pub fn phase_ordinal_bases_from_env() -> HashMap<Phase, usize> {
    let Ok(raw) = std::env::var(CELL_PHASE_ORDINAL_BASES_ENV) else {
        return HashMap::new();
    };
    let by_name: std::collections::BTreeMap<String, u64> =
        serde_json::from_str(&raw).unwrap_or_default();
    by_name
        .into_iter()
        .filter_map(|(name, base)| phase_from_name(&name).map(|phase| (phase, base as usize)))
        .collect()
}

/// Maps a v2 phase name to its metric phase (the two the scheduled runner supports).
fn phase_from_name(name: &str) -> Option<Phase> {
    match name {
        "warmup" => Some(Phase::Warmup),
        "profiling" => Some(Phase::Profiling),
        _ => None,
    }
}

/// The autonomous issuer for this cell, or [`DirectIssuanceAuthority`] when the
/// process is not a cell. The partition is read from the environment (via
/// [`ModuloCellPartition::from_env`]), so the single-process default stays
/// byte-unchanged.
///
/// [`DirectIssuanceAuthority`]: crate::cellular::DirectIssuanceAuthority
pub fn issuance_authority_from_env() -> std::rc::Rc<dyn IssuanceAuthority> {
    match ModuloCellPartition::from_env() {
        Some(partition) => std::rc::Rc::new(CellularAutonomousIssuer::new(partition)),
        None => std::rc::Rc::new(crate::cellular::DirectIssuanceAuthority::new()),
    }
}

/// The autonomous issuer for an explicitly supplied partition, for a caller that
/// derives a per-worker partition itself rather than from the process environment.
/// Always the autonomous issuer (never [`DirectIssuanceAuthority`]): a caller
/// supplies a partition only when it wants global-ordinal stamping.
///
/// [`DirectIssuanceAuthority`]: crate::cellular::DirectIssuanceAuthority
pub fn issuance_authority_for(
    partition: ModuloCellPartition,
) -> std::rc::Rc<dyn IssuanceAuthority> {
    std::rc::Rc::new(CellularAutonomousIssuer::new(partition))
}

/// Ship this cell's per-record artifact files (+ `inputs.json`) to the controller's
/// HTTP upload server with streaming zstd (Stage E, cross-host path), when shipping
/// is enabled and an artifact authority is resolvable ([`cell_artifact_authority`]).
/// A no-op on the same-host launcher (Stage D concatenates the cell's own writes) or
/// when shipping is disabled.
///
/// Called at cell finalize AFTER the cell has written its artifacts to its own
/// `artifact_dir`, before process exit. Blocking by design (off the hot path); the
/// async HTTP work runs on a dedicated thread + runtime so it never touches the
/// caller's (possibly `current_thread`) execute runtime — mirroring
/// [`CellRecordsShipper::ship`]. The controller waits for every cell's `/done` marker
/// (posted last by [`crate::runner_protocol::artifact_shipping::ship_cell_artifacts`])
/// before running its concat, so this must complete before the process exits.
#[cfg(feature = "velo")]
pub fn ship_http_artifacts_if_enabled(
    cell_dir: &std::path::Path,
    artifacts: &crate::runner_protocol::protocol::ArtifactSpec,
) -> Result<()> {
    let Some(partition) = ModuloCellPartition::from_env() else {
        return Ok(()); // not a cell (single-process path)
    };
    let Some(authority) = cell_artifact_authority() else {
        return Ok(()); // same-host or shipping disabled
    };
    let relatives = crate::runner_protocol::artifact_shipping::shippable_relatives(artifacts);
    if relatives.is_empty() {
        return Ok(()); // metrics-only run with no files to ship
    }
    let cell_id = partition.cell_id();
    let cell_dir = cell_dir.to_path_buf();
    std::thread::spawn(move || -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        runtime.block_on(
            crate::runner_protocol::artifact_shipping::ship_cell_artifacts(
                &authority, cell_id, &cell_dir, &relatives,
            ),
        )
    })
    .join()
    .map_err(|_| anyhow::anyhow!("cell artifact-shipping thread panicked"))?
}

/// The controller-local absolute path of a `file`-type dataset with a `path`
/// source (the only non-synthetic dataset a cross-host cell cannot reach), or
/// `None` for synthetic, inline-`records` `file`, or `public` (URL/HF each cell
/// fetches independently). This is FORMAT-BLIND: it keys only on `type == "file"`
/// plus a non-empty `path`, so a single-file GRAPH trace (`dag_jsonl`, or a
/// single-file `weka_trace`/`dynamo_trace`) ALSO returns its path and rides the same
/// Stage G serve/download/rewrite plane. (A graph trace whose `path` is a
/// DIRECTORY or segmented-prefix ships every shard the loader reads over the same
/// plane via the manifest — see [`crate::runner_protocol::cellular_controller`]'s
/// `build_dataset_serve_plan` and [`download_cell_dataset_if_needed`].) Reads the
/// canonical single-dataset list at
/// `/run/cfg/datasets/0`, matching the controller's own detection so the serve
/// allowlist and the cell request name can never disagree.
pub fn cellular_file_dataset_path(envelope: &serde_json::Value) -> Option<std::path::PathBuf> {
    let dataset = envelope.pointer("/run/cfg/datasets/0")?;
    if dataset.get("type").and_then(serde_json::Value::as_str) != Some("file") {
        return None;
    }
    dataset
        .get("path")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
}

/// The cell-local directory shipped dataset sources land in (`aiperf-cell-dataset-{pid}`
/// under the system temp dir), created on demand. Distinct from the velo scratch so
/// a shipped dataset never collides with the cell's fetch/ship sockets.
fn cell_dataset_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("aiperf-cell-dataset-{}", std::process::id()))
}

/// Before the cell compiles its dataset, ship the controller's `file`/`path`
/// dataset source to the cell over HTTP + streaming zstd (Stage G) and rewrite the
/// cell's envelope to point at the landed cell-local copy, so `build_file_dataset`
/// reads a local file rather than the unreachable controller path.
///
/// A no-op that returns the envelope unchanged when any of these hold (each is the
/// correct behaviour, not a skip):
/// - the process is not a cell (single-process path);
/// - the dataset is not a `file`/`path` source (synthetic regenerates from the
///   shared seed; inline `records` already ride in the envelope; `public` URL/HF
///   each cell fetches independently);
/// - no artifact authority resolves ([`cell_artifact_authority`]) — a same-host
///   cell (the controller-local path is directly readable) or an operator on a
///   shared filesystem with HTTP shipping disabled (the path is shared too).
///
/// The download runs on a dedicated thread + runtime (mirroring
/// [`CellRecordsShipper::ship`]) so it never touches the caller's runtime.
#[cfg(feature = "velo")]
pub fn download_cell_dataset_if_needed(envelope_bytes: Vec<u8>) -> Result<Vec<u8>> {
    use anyhow::Context;

    if ModuloCellPartition::from_env().is_none() {
        return Ok(envelope_bytes); // not a cell (single-process path)
    }
    let mut envelope: serde_json::Value = serde_json::from_slice(&envelope_bytes)
        .context("parsing cell envelope for dataset download")?;
    let Some(source_path) = cellular_file_dataset_path(&envelope) else {
        // synthetic / inline-records / public — nothing to ship. A `file`/`path` graph
        // trace (dag_jsonl / weka_trace / dynamo_trace) IS shipped (the predicate is
        // format-blind), whether its path is a single file, a directory of shards, or a
        // segmented-prefix — the controller's manifest carries the whole file set.
        return Ok(envelope_bytes);
    };
    let Some(authority) = cell_artifact_authority() else {
        // Same-host cell, or shared-FS with shipping disabled: the controller-local
        // path is directly readable, so leave the envelope pointing at it.
        return Ok(envelope_bytes);
    };
    // The controller cannot know this cell's on-disk layout, so it publishes a
    // manifest describing the trace file set (single file, directory of shards, or
    // segmented-prefix). The cell fetches the manifest, streams every file over the
    // same HTTP+zstd plane, reconstructs the tree under a cell-local dir preserving
    // the (flat) relative names, and rewrites `datasets/0.path` to the local
    // file/dir/prefix stem — so the graph loader reads the reconstructed tree.
    let _ = &source_path; // presence gated the ship; the controller owns the file set
    let dest_dir = cell_dataset_dir();
    let fetch_authority = authority.clone();
    let local_path = std::thread::spawn(move || -> Result<std::path::PathBuf> {
        use crate::runner_protocol::artifact_shipping::{
            fetch_dataset_manifest, reconstruct_shipped_dataset,
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        runtime.block_on(async move {
            let manifest = fetch_dataset_manifest(&fetch_authority)
                .await
                .context("cell fetching dataset manifest from controller")?;
            reconstruct_shipped_dataset(&fetch_authority, &manifest, &dest_dir)
                .await
                .context("cell reconstructing shipped dataset from controller")
        })
    })
    .join()
    .map_err(|_| anyhow::anyhow!("cell dataset-download thread panicked"))??;

    // Rewrite the cell's envelope to compile from the landed cell-local copy.
    envelope
        .pointer_mut("/run/cfg/datasets/0")
        .and_then(serde_json::Value::as_object_mut)
        .context("cell envelope dataset is not an object")?
        .insert(
            "path".to_owned(),
            serde_json::Value::String(local_path.to_string_lossy().into_owned()),
        );
    serde_json::to_vec(&envelope).context("re-serializing cell envelope after dataset download")
}

// -- velo cell transport (fetch spec + ship records) ------------------------------

/// The velo bind for this cell, chosen from the controller coordinate: a `tcp://`
/// coordinate whose host is **loopback** is a co-located (local launcher) run — the
/// cell binds loopback too so it advertises a loopback endpoint the loopback-bound
/// controller can route back to; a `tcp://` coordinate with a **routable** host is
/// k8s — the cell binds all interfaces so the controller reaches the pod IP. A
/// `uds://` coordinate is a pure-local unix run. `role` disambiguates the cell's
/// fetch vs ship velo instances so their UDS paths do not collide.
#[cfg(feature = "velo")]
fn cell_bind(coordinate: &str, role: &str) -> crate::cellular::transport::connect::BindSpec {
    use crate::cellular::transport::connect::BindSpec;
    if let Some(addr) = coordinate.strip_prefix("tcp://") {
        let loopback = addr
            .parse::<std::net::SocketAddr>()
            .map(|socket| socket.ip().is_loopback())
            .unwrap_or(false);
        if loopback {
            let _ = role;
            return BindSpec::TcpLoopback;
        }
        return BindSpec::TcpBind("0.0.0.0:0".parse().expect("valid ephemeral bind addr"));
    }
    #[cfg(unix)]
    {
        let dir = std::env::temp_dir().join(format!("aiperf-velo-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        BindSpec::UdsPath(dir.join(format!("cell-{role}.sock")))
    }
    #[cfg(not(unix))]
    {
        let _ = role;
        BindSpec::TcpLoopback
    }
}

/// Fetch this cell's sliced execute envelope (protocol-v2 JSON bytes) from the
/// controller over velo, replacing the stdin spec pipe. Reads the controller
/// coordinate + cell id from the env the launcher set. Runs on the caller's
/// runtime; the velo instance is dropped on return.
#[cfg(feature = "velo")]
pub async fn fetch_cell_envelope() -> Result<Vec<u8>> {
    use crate::cellular::VeloCellClient;
    use crate::cellular::transport::connect::{build_velo, connect_controller};
    use anyhow::Context;

    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV)
        .context("cell has no AIPERF_CELL_CONTROLLER_ADDR")?;
    let cell_id = ModuloCellPartition::from_env()
        .context("cell has no partition env (AIPERF_CELL_ID/_COUNT)")?
        .cell_id();
    let velo = build_velo(cell_bind(&coordinate, "fetch")).await?;
    // Discovery-free: dial the controller's known endpoint; velo's `_hello`
    // handshake learns its identity and mutually registers us.
    let controller = connect_controller(&velo, &coordinate)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect controller: {error}"))?;
    // Ultimate spec §4: opt-in phaser-driven START. Capture the velo + controller peer
    // before the client consumes them, so the cell can subscribe to the phaser control
    // plane over the same fetch instance and await generation 1 instead of the event.
    let phaser_start = matches!(
        std::env::var("AIPERF_CELL_PHASER_START")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    let phaser_handles = phaser_start.then(|| (velo.clone(), controller.clone()));
    let client = VeloCellClient::connect(velo, controller)
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect: {error}"))?;
    let reply = client
        .register(cell_id)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} register: {error}"))?;
    // Block until START. Either the phaser reaches generation 1 (§4 control plane) or the
    // single-shot event triggers — every cell resumes together once the controller has
    // seen the registrations (or immediately, barrier-free). A poisoned event / finalized
    // phaser (the controller aborted before starting) surfaces here as an error.
    if let Some((phaser_velo, phaser_controller)) = phaser_handles {
        let mut sub = crate::cellular::transport::phaser_velo::PhaserClient::subscribe(
            phaser_velo,
            &phaser_controller,
        )
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} phaser subscribe: {error}"))?;
        sub.await_started()
            .await
            .map_err(|error| anyhow::anyhow!("cell {cell_id} phaser await start: {error}"))?;
    } else {
        client
            .await_start(reply.start_event)
            .await
            .map_err(|error| anyhow::anyhow!("cell {cell_id} await start: {error}"))?;
    }
    Ok(reply.envelope)
}

/// Ultimate spec §3 + §4.5: build this cell's owned dataset index over the fan-out
/// broadcast, then run the dispatch state machine over its owned slice. When
/// `AIPERF_CELL_DATASET_FANOUT` is set, the controller has broadcast the dataset's
/// request-ids; the cell subscribes over velo (replay-on-attach → its full owned shard),
/// builds an index keyed by request_id filtered to its round-robin owned positions
/// (§3.4 → O(1/N) RAM), and issues each owned request through the `DispatchTracker`
/// (§4.5: exactly-once, counted `DistributionMiss`). A no-op when the flag is unset. This
/// proves the §3 fan-out delivers each cell its owned shard and the §4.5 dispatch state
/// machine drives it, over real velo, in a real run — fail-closed on any miss (an
/// incomplete fan-out).
#[cfg(feature = "velo")]
pub async fn verify_dataset_fanout() -> Result<()> {
    use crate::cellular::dispatch_state::{DispatchDecision, DispatchTracker};
    use crate::cellular::transport::connect::{build_velo, connect_controller};
    use crate::cellular::transport::dataset_velo::DatasetClient;
    use anyhow::Context;

    let enabled = matches!(
        std::env::var("AIPERF_CELL_DATASET_FANOUT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on" | "yes"
    );
    if !enabled {
        return Ok(());
    }
    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV)
        .context("dataset fan-out: cell has no AIPERF_CELL_CONTROLLER_ADDR")?;
    let partition =
        ModuloCellPartition::from_env().context("dataset fan-out: cell has no partition env")?;
    let cell_id = partition.cell_id() as u64;
    let cell_count = partition.cell_count() as u64;

    let velo = build_velo(cell_bind(&coordinate, "dataset")).await?;
    let controller = connect_controller(&velo, &coordinate)
        .await
        .map_err(|error| anyhow::anyhow!("dataset fan-out connect controller: {error}"))?;
    let index =
        DatasetClient::build_owned_index(velo, &controller, move |id| id % cell_count == cell_id)
            .await
            .map_err(|error| anyhow::anyhow!("dataset fan-out build index: {error}"))?;

    // Run the §4.5 dispatch state machine over the owned slice.
    let mut tracker = DispatchTracker::new();
    for id in index.owned_ids() {
        match tracker.on_issue(id, &index) {
            DispatchDecision::Issue(_) => tracker.on_complete(id),
            other => tracing::warn!(request_id = id, ?other, "unexpected dispatch decision"),
        }
    }
    tracing::info!(
        cell_id,
        cell_count,
        owned = index.len(),
        issued = tracker.issued(),
        completed = tracker.completed(),
        misses = tracker.distribution_misses(),
        "dataset fan-out + dispatch state machine verified over velo"
    );
    anyhow::ensure!(
        tracker.distribution_misses() == 0,
        "dataset fan-out delivered incompletely: {} distribution misses on cell {cell_id}",
        tracker.distribution_misses()
    );
    Ok(())
}

/// Substitute a cell's round-robin aggregator id into the operator's ship-DNS
/// template. The template is a concrete `tcp://…svc.cluster.local:PORT` coordinate
/// with a single `{agg_id}` placeholder (jobset/namespace already resolved by the
/// operator); the cell fills in `cell_id % agg_count`. Pure so the pod-side ship-target
/// derivation is unit-testable without a velo runtime.
pub(crate) fn k8s_agg_ship_coordinate(template: &str, cell_id: u32, agg_count: u32) -> String {
    let agg_id = cell_id % agg_count.max(1);
    template.replace("{agg_id}", &agg_id.to_string())
}

/// Ships a cell's final records-shard partition + heartbeat to the controller over
/// velo, when this process is a cell (the controller coordinate is set).
#[cfg(feature = "velo")]
pub struct CellRecordsShipper {
    cell_id: u32,
    coordinate: String,
}

#[cfg(feature = "velo")]
impl CellRecordsShipper {
    /// Builds a shipper when the controller coordinate and cell partition env vars
    /// are set, else `None` (the ordinary single-process path).
    pub fn from_env() -> Option<Self> {
        // Requires a controller address to exist (i.e. this is a real cell); the ship
        // address alone never activates cellular shipping on a single-process run.
        std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()?;
        let partition = ModuloCellPartition::from_env()?;
        let cell_id = partition.cell_id();
        let coordinate = Self::ship_target(cell_id, partition.cell_count())?;
        Some(Self {
            cell_id,
            coordinate,
        })
    }

    /// This cell's terminal ship coordinate, in precedence order:
    /// 1. `AIPERF_CELL_SHIP_ADDR` (same-host tree — the controller injected each
    ///    local cell's assigned loopback aggregator coordinate directly);
    /// 2. the k8s aggregator derived from the operator's DNS template
    ///    ([`AGG_DNS_TEMPLATE_ENV`]) + this cell's round-robin aggregator
    ///    (`cell_id % M`) — a JobSet indexed replicatedJob shares one env template, so
    ///    the per-cell ship target must be computed pod-side from the shared template;
    /// 3. the controller directly ([`CELL_CONTROLLER_ADDR_ENV`], flat star).
    fn ship_target(cell_id: u32, cell_count: u32) -> Option<String> {
        if let Ok(addr) = std::env::var(CELL_SHIP_ADDR_ENV)
            && !addr.is_empty()
        {
            return Some(addr);
        }
        if let Ok(template) =
            std::env::var(crate::runner_protocol::cellular_aggregator::AGG_DNS_TEMPLATE_ENV)
            && !template.is_empty()
            && let Some(agg_count) =
                crate::runner_protocol::cellular_aggregator::aggregator_count(cell_count)
        {
            return Some(k8s_agg_ship_coordinate(&template, cell_id, agg_count));
        }
        std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()
    }

    /// Builds a shipper that sends to an explicit velo `coordinate` under `cell_id`.
    /// Used by a tier-T2 aggregator to ship its one merged store up to the controller
    /// (the `cell_id` is the aggregator's id, which orders the controller's
    /// `merge_store_partitions` deterministically).
    pub fn to_coordinate(cell_id: u32, coordinate: String) -> Self {
        Self {
            cell_id,
            coordinate,
        }
    }

    /// The cell's identifier.
    pub fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// Builds this cell's terminal heartbeat from its final records and ships the
    /// records-shard partition + heartbeat to the controller (the RETAIN path). Shared
    /// by the scheduled and graph cell paths, which differ only in how they derive
    /// `records` and the run-span `epoch_ns`. One end-of-run aggregate (not a
    /// per-tick snapshot); saturation is zero (the run has drained).
    ///
    /// Blocking by design (called once, off the hot path). The velo async work runs
    /// on a dedicated thread + runtime so it never touches the caller's
    /// (possibly `current_thread`) execute runtime.
    pub fn ship_records(
        &self,
        records: Vec<crate::metrics_core::RecordIngest>,
        epoch_ns: i64,
    ) -> Result<()> {
        use crate::cellular::{
            CellMessage, HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation,
            RecordsShardPartition,
        };

        let mut heartbeat = HeartbeatAccumulator::new();
        let mut completed = 0_u64;
        let mut errored = 0_u64;
        for record in &records {
            crate::runner_protocol::heartbeat_lane::observe_ingest(&mut heartbeat, record);
            if record.errored || record.canceled {
                errored += 1;
            } else {
                completed += 1;
            }
        }
        let counters = HeartbeatCounters {
            issued: records.len() as u64,
            completed,
            errored,
        };
        let heartbeat = heartbeat.snapshot(epoch_ns, counters, HeartbeatSaturation::default());
        let partition = RecordsShardPartition::new(self.cell_id, records);
        self.ship(heartbeat, CellMessage::Partition(partition))
    }

    /// Stage C: ships this cell's folded EXACT column store + a counters-only heartbeat
    /// (the metrics-only exact-fold path). A cell running exact-fold folded every record
    /// into its own accumulator and DROPPED the per-record data, so it has no record
    /// `Vec` to ship — it ships the folded store instead, which the controller appends
    /// across cells (`merge_store_partitions`) into the merged report.
    ///
    /// The heartbeat carries EXACT counters (`issued`/`completed`/`errored`, supplied by
    /// the caller from the accumulator's record count and the retained errored subset)
    /// but EMPTY latency sketches: the fold dropped the per-record TTFT/ITL/latency
    /// samples the sketches need. The heartbeat is a live-lane diagnostic; the
    /// authoritative percentiles come from the merged store, so empty sketches are honest
    /// rather than a fidelity loss. The counters keep the controller's
    /// `cellular-heartbeat.json` sidecar populated (proving the controller path ran).
    pub fn ship_store(
        &self,
        store: crate::metrics_core::store::ColumnStore,
        counters: crate::cellular::HeartbeatCounters,
        epoch_ns: i64,
    ) -> Result<()> {
        use crate::cellular::{
            CellMessage, ColumnStorePartition, HeartbeatAccumulator, HeartbeatSaturation,
        };
        let heartbeat = HeartbeatAccumulator::new().snapshot(
            epoch_ns,
            counters,
            HeartbeatSaturation::default(),
        );
        let partition = ColumnStorePartition::from_store(self.cell_id, store);
        self.ship(heartbeat, CellMessage::StorePartition(Box::new(partition)))
    }

    /// Ships one heartbeat then one terminal partition message to the controller over a
    /// fresh, dedicated velo runtime. Shared by [`ship_records`](Self::ship_records) and
    /// [`ship_store`](Self::ship_store): they differ only in the heartbeat's sketches and
    /// whether the terminal message is a records or a store partition.
    ///
    /// Blocking by design (called once, off the hot path). The velo async work runs on a
    /// dedicated thread + runtime so it never touches the caller's (possibly
    /// `current_thread`) execute runtime.
    fn ship(
        &self,
        heartbeat: crate::cellular::MetricsHeartbeat,
        terminal: crate::cellular::CellMessage,
    ) -> Result<()> {
        use crate::cellular::transport::connect::{build_velo, connect_controller};
        use crate::cellular::{CellClient, CellMessage, VeloCellClient};

        let coordinate = self.coordinate.clone();
        let cell_id = self.cell_id;

        // A dedicated thread + multi-thread runtime for the velo ship: velo builds
        // and drives its own tasks here, isolated from the execute runtime.
        std::thread::spawn(move || -> Result<()> {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?;
            runtime.block_on(async move {
                let velo = build_velo(cell_bind(&coordinate, "ship")).await?;
                let controller = connect_controller(&velo, &coordinate)
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!("cell {cell_id} ship connect controller: {error}")
                    })?;
                let mut client = VeloCellClient::connect(velo, controller)
                    .map_err(|error| anyhow::anyhow!("cell {cell_id} ship connect: {error}"))?;
                client
                    .send(&CellMessage::Heartbeat {
                        cell_id,
                        heartbeat: Box::new(heartbeat),
                    })
                    .await
                    .map_err(|error| anyhow::anyhow!("cell {cell_id} ship heartbeat: {error}"))?;
                client
                    .send(&terminal)
                    .await
                    .map_err(|error| anyhow::anyhow!("cell {cell_id} ship partition: {error}"))?;
                Ok(())
            })
        })
        .join()
        .map_err(|_| anyhow::anyhow!("cell {} ship thread panicked", self.cell_id))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k8s_ship_coordinate_round_robins_cells_to_aggregators() {
        let template = "tcp://js-aggregators-{agg_id}-0.js.ns.svc.cluster.local:9700";
        // cells 0..6 over M=2 aggregators: even→agg0, odd→agg1 (cell_id % M).
        assert_eq!(
            k8s_agg_ship_coordinate(template, 0, 2),
            "tcp://js-aggregators-0-0.js.ns.svc.cluster.local:9700"
        );
        assert_eq!(
            k8s_agg_ship_coordinate(template, 1, 2),
            "tcp://js-aggregators-1-0.js.ns.svc.cluster.local:9700"
        );
        assert_eq!(
            k8s_agg_ship_coordinate(template, 4, 2),
            "tcp://js-aggregators-0-0.js.ns.svc.cluster.local:9700"
        );
        // M=3: cell 5 → agg 2.
        assert_eq!(
            k8s_agg_ship_coordinate(template, 5, 3),
            "tcp://js-aggregators-2-0.js.ns.svc.cluster.local:9700"
        );
    }

    #[test]
    fn detects_only_file_path_datasets_for_ship() {
        // A `file` dataset with a `path` is the one non-synthetic source a cross-host
        // cell cannot reach — the only shape the controller must ship (Stage G).
        let file_path = serde_json::json!({"run": {"cfg": {"datasets": [
            {"type": "file", "format": "single_turn", "path": "/data/prompts.jsonl"}
        ]}}});
        assert_eq!(
            cellular_file_dataset_path(&file_path),
            Some(std::path::PathBuf::from("/data/prompts.jsonl"))
        );

        // A single-file GRAPH trace (`dag_jsonl`/`weka_trace`/`dynamo_trace`) is likewise
        // shipped: the predicate is format-blind (it keys only on `type == "file"` + a
        // non-empty `path`), so a graph `{type:file, format:dag_jsonl, path}` also returns
        // the path and rides the SAME Stage G serve/download/rewrite plane. The controller
        // separately fails closed if that graph path is a directory/segmented-prefix.
        for graph_format in ["dag_jsonl", "weka_trace", "dynamo_trace"] {
            let graph = serde_json::json!({"run": {"cfg": {"datasets": [
                {"type": "file", "format": graph_format, "path": "/traces/graph.jsonl"}
            ]}}});
            assert_eq!(
                cellular_file_dataset_path(&graph),
                Some(std::path::PathBuf::from("/traces/graph.jsonl")),
                "single-file graph trace {graph_format} must ship over Stage G"
            );
        }

        // Everything else yields None (no ship): synthetic regenerates from the seed;
        // an inline-`records` file already rides in the envelope; `public` URL/HF each
        // cell fetches itself; and an empty path is not a shippable source.
        for none in [
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "synthetic"}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "single_turn", "records": []}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "public", "source": {"type": "url", "url": "http://x"}}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": [{"type": "file", "format": "single_turn", "path": ""}]}}}),
            serde_json::json!({"run": {"cfg": {"datasets": []}}}),
            serde_json::json!({"run": {"cfg": {}}}),
        ] {
            assert_eq!(
                cellular_file_dataset_path(&none),
                None,
                "should not ship {none}"
            );
        }
    }
}
