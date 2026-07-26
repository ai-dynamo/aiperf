// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cell-mode support for the multi-cell cellular runtime.
//!
//! A *cell* is one `aiperf --cell` process (a local subprocess or a k8s
//! pod). It runs the ordinary online scheduled execution over its budget slice,
//! but with the `CellularAutonomousIssuer` assigning dense global dispatch
//! ordinals from its `(cell_id, cell_count)` partition, and it ships its captured
//! records to the controller over the velo [`transport`](crate::cellular::transport)
//! seam instead of writing a report. The controller re-ingests every cell's
//! records in global ordinal order for the single authoritative `native-v2.json`.
//!
//! Cell behaviour is injected through environment variables:
//! [`CELL_ID_ENV`](crate::cellular::partition::CELL_ID_ENV)
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

/// Env var carrying the velo coordinate a cell ships its
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

/// Env var carrying the controller's artifact upload `host:port`. The
/// operator injects this into k8s pods (or the local launcher sets it to the
/// controller's bound server) so a cell knows where to POST its per-record artifact
/// files. When absent, a cell on a `tcp://` (k8s) controller coordinate derives the
/// host from that coordinate and the port from [`CELL_ARTIFACT_PORT_ENV`].
pub const CELL_ARTIFACT_ADDR_ENV: &str = "AIPERF_CELL_ARTIFACT_ADDR";

/// Env var overriding the controller's artifact-server port when a cell derives the
/// artifact `host:port` from its `tcp://HOST:PORT` velo coordinate (default `9600`,
/// matching the controller's `AIPERF_CONTROLLER_ARTIFACT_BIND` default).
pub const CELL_ARTIFACT_PORT_ENV: &str = "AIPERF_CELL_ARTIFACT_PORT";

/// Env toggle disabling cross-host HTTP artifact shipping. Default ON; set to
/// `0`/`false`/`off` to use cell-local or shared-filesystem writes.
pub const CELL_HTTP_ARTIFACT_SHIPPING_ENV: &str = "AIPERF_CELL_HTTP_ARTIFACT_SHIPPING";

/// The default controller artifact-server port (server bind + cell-derived fetch).
pub const DEFAULT_ARTIFACT_PORT: u16 = 9600;

/// Env selecting the per-record artifact transport: `http` (default) uses the
/// raw-hyper HTTP/1 + streaming-zstd plane on the artifact port; `velo` collapses
/// artifact shipping onto the shared cellular velo instance/endpoint (no second
/// port), streaming zstd chunks over velo's native backpressured stream primitive.
/// Any unrecognized value keeps the default HTTP behavior (fail-safe, additive).
pub const ARTIFACT_TRANSPORT_ENV: &str = "AIPERF_ARTIFACT_TRANSPORT";

/// Whether per-record artifact shipping should ride the velo plane
/// ([`ARTIFACT_TRANSPORT_ENV`] == `velo`). Default (unset / `http` / anything else)
/// is `false` — the existing HTTP plane — so this is additive and non-regressing.
pub fn artifact_transport_is_velo() -> bool {
    std::env::var(ARTIFACT_TRANSPORT_ENV)
        .map(|value| value.eq_ignore_ascii_case("velo"))
        .unwrap_or(false)
}

/// **Test/dev-only force seam.** Env flag that makes a LOCAL (`--cells N`, same-host)
/// run drive the cross-host HTTP artifact path over loopback instead of direct
/// shared-filesystem concatenation: the controller binds its artifact upload server on
/// `127.0.0.1:0`, injects that authority into each locally-launched cell, and the
/// cells POST their per-record artifact files back over real TCP + streaming zstd —
/// exercising the exact production shipping/upload/concat code that k8s uses, but
/// without a second host. Set to `1`/`true`/`on`/`yes` to enable.
///
/// This lets same-host multi-process tests exercise HTTP+zstd shipping
/// (see `rust/e2e-tests/tests/test_cellular_http_shipping.rs`).
/// It is not a product mode; local `--cells N` writes directly to
/// controller-local scratch when the flag is unset.
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
/// when HTTP shipping is off or this is a same-host launcher (which concatenates
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
    // HTTP upload server there because cells use shared-filesystem writes,
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
/// [`ModuloCellPartition::from_env`]).
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
/// HTTP upload server with streaming zstd, when shipping
/// is enabled and an artifact authority is resolvable ([`cell_artifact_authority`]).
/// A no-op on the same-host launcher, which concatenates the cell's own writes, or
/// when shipping is disabled.
///
/// Called at cell finalize AFTER the cell has written its artifacts to its own
/// `artifact_dir`, before process exit. Blocking by design (off the hot path); the
/// async HTTP work runs on a dedicated thread and runtime so it never touches the
/// caller's execute runtime. The controller waits for every cell's `/done` marker
/// (posted last by [`crate::engine::artifact_shipping::ship_cell_artifacts`])
/// before running its concat, so this must complete before the process exits.
#[cfg(feature = "cellular")]
pub fn ship_http_artifacts_if_enabled(
    cell_dir: &std::path::Path,
    artifacts: &crate::engine::protocol::ArtifactSpec,
) -> Result<()> {
    let Some(partition) = ModuloCellPartition::from_env() else {
        return Ok(()); // not a cell (single-process path)
    };
    let Some(authority) = cell_artifact_authority() else {
        return Ok(()); // same-host or shipping disabled
    };
    let relatives = crate::engine::artifact_shipping::shippable_relatives(artifacts);
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
        runtime.block_on(crate::engine::artifact_shipping::ship_cell_artifacts(
            &authority, cell_id, &cell_dir, &relatives,
        ))
    })
    .join()
    .map_err(|_| anyhow::anyhow!("cell artifact-shipping thread panicked"))?
}

/// Ship this cell's per-record artifact files (+ `inputs.json`) to the controller
/// over the **velo** streaming plane (the shared cellular velo endpoint), when
/// shipping is enabled and this process is a cell. Unlike the HTTP path this needs
/// no separate artifact host/port: it dials the controller's velo coordinate
/// ([`CELL_SHIP_ADDR_ENV`] if set, else [`CELL_CONTROLLER_ADDR_ENV`]) — the exact
/// same endpoint the control plane already uses — and streams zstd chunks over
/// velo's native backpressured stream primitive (bounded memory).
///
/// Runs on a dedicated thread + runtime (off the caller's execute runtime), mirroring
/// [`ship_http_artifacts_if_enabled`]. A no-op when not a cell, when shipping is
/// disabled, or when no controller coordinate is set.
#[cfg(feature = "cellular")]
pub fn ship_velo_artifacts_if_enabled(
    cell_dir: &std::path::Path,
    artifacts: &crate::engine::protocol::ArtifactSpec,
) -> Result<()> {
    use anyhow::Context as _;

    use crate::cellular::transport::connect::{build_velo, connect_controller};

    let Some(partition) = ModuloCellPartition::from_env() else {
        return Ok(()); // not a cell (single-process path)
    };
    if !http_artifact_shipping_enabled() {
        return Ok(()); // shipping disabled
    }
    // Prefer the tree-topology ship target, else the controller coordinate.
    let coordinate = std::env::var(CELL_SHIP_ADDR_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var(CELL_CONTROLLER_ADDR_ENV).ok())
        .filter(|value| !value.is_empty());
    let Some(coordinate) = coordinate else {
        return Ok(()); // no controller coordinate — nothing to ship to
    };
    // Same-host gate (mirrors the HTTP `cell_artifact_authority` loopback rule): a
    // loopback controller coordinate is a co-located run whose cells use shared-FS
    // concatenation — ship nothing unless the test/dev force seam is engaged. A
    // routable (k8s) coordinate always ships.
    let host = coordinate.strip_prefix("tcp://").map(|host_port| {
        host_port
            .rsplit_once(':')
            .map_or(host_port, |(host, _)| host)
    });
    let is_loopback = host
        .map(|host| {
            host.parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(host.eq_ignore_ascii_case("localhost"))
        })
        // A `uds://` coordinate is always a co-located local run.
        .unwrap_or(true);
    if is_loopback && !artifact_http_force_enabled() {
        return Ok(());
    }
    let relatives = crate::engine::artifact_stream_velo::shippable_relatives_velo(artifacts);
    if relatives.is_empty() {
        return Ok(()); // metrics-only run with no files to ship
    }
    let cell_id = partition.cell_id();
    tracing::debug!(
        target: "aiperf_cellular_artifact",
        cell_id,
        coordinate = %coordinate,
        files = relatives.len(),
        "velo artifact shipping starting"
    );
    let cell_dir = cell_dir.to_path_buf();
    std::thread::spawn(move || -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let result = runtime.block_on(async move {
            // A fresh, short-lived shipping velo instance (the same lifecycle the
            // partition ship uses): bind, dial the controller by endpoint, stream.
            let bind = cell_bind(&coordinate, "artifact");
            let velo = build_velo(bind)
                .await
                .context("building cell artifact velo")?;
            let controller = connect_controller(&velo, &coordinate)
                .await
                .context("connecting to controller for velo artifact shipping")?;
            crate::engine::artifact_stream_velo::ship_cell_artifacts_velo(
                &velo,
                &controller,
                cell_id,
                &cell_dir,
                &relatives,
            )
            .await
        });
        match &result {
            Ok(()) => tracing::debug!(
                target: "aiperf_cellular_artifact",
                cell_id,
                "velo artifact shipping completed"
            ),
            Err(error) => {
                tracing::error!(cell_id, "velo artifact shipping failed: {error:#}")
            }
        }
        result
    })
    .join()
    .map_err(|_| anyhow::anyhow!("cell velo artifact-shipping thread panicked"))?
}

/// Ship this cell's per-record artifacts over the configured transport
/// ([`ARTIFACT_TRANSPORT_ENV`]): velo when selected, else the default HTTP plane.
/// The single dispatch point the execute finalize tail calls, so the transport
/// choice lives in one place.
#[cfg(feature = "cellular")]
pub fn ship_artifacts_if_enabled(
    cell_dir: &std::path::Path,
    artifacts: &crate::engine::protocol::ArtifactSpec,
) -> Result<()> {
    if artifact_transport_is_velo() {
        ship_velo_artifacts_if_enabled(cell_dir, artifacts)
    } else {
        ship_http_artifacts_if_enabled(cell_dir, artifacts)
    }
}

/// The controller-local absolute path of a `file`-type dataset with a `path`
/// source (the only non-synthetic dataset a cross-host cell cannot reach), or
/// `None` for synthetic, inline-`records` `file`, or `public` (URL/HF each cell
/// fetches independently). This is FORMAT-BLIND: it keys only on `type == "file"`
/// plus a non-empty `path`, so a single-file GRAPH trace (`dag_jsonl`, or a
/// single-file `weka_trace`/`dynamo_trace`) ALSO returns its path and rides the same
/// serve/download/rewrite plane. (A graph trace whose `path` is a
/// DIRECTORY or segmented-prefix ships every shard the loader reads over the same
/// plane via the manifest — see [`crate::engine::cellular_controller`]'s
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
/// dataset source to the cell over HTTP + streaming zstd and rewrite the
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
/// The download runs on a dedicated thread and runtime, isolated from the caller's
/// runtime.
#[cfg(feature = "cellular")]
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
        use crate::engine::artifact_shipping::{
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

/// The velo bind for this cell, chosen from the controller coordinate: a `tcp://`
/// coordinate whose host is **loopback** is a co-located (local launcher) run — the
/// cell binds loopback too so it advertises a loopback endpoint the loopback-bound
/// controller can route back to; a `tcp://` coordinate with a **routable** host is
/// k8s — the cell binds all interfaces so the controller reaches the pod IP. A
/// `uds://` coordinate is a pure-local unix run. `role` disambiguates the cell's
/// fetch vs ship velo instances so their UDS paths do not collide.
#[cfg(feature = "cellular")]
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
#[cfg(feature = "cellular")]
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
    // Keep handles before constructing the client so the phaser can subscribe over
    // the same fetch instance.
    let phaser_start = matches!(
        std::env::var(crate::engine::cellular_controller::CELL_PHASER_START_ENV)
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
    // Block until START. Either the phaser reaches generation 1 or the
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
    // The velo START barrier has released for every cell together: capture THIS
    // instant as the shared cross-cell timing origin (opt-in), before the cell's
    // per-shard dataset download + run setup skews each cell's local run start.
    crate::engine::cell_origin::capture_cell_shared_origin();
    Ok(reply.envelope)
}

/// Await a named controller-owned phase gate over the cellular phaser.
///
/// This opens a short-lived velo control-plane client on the caller's runtime, so later
/// phase-bound hooks can block on replay/live phaser semantics without sharing a velo
/// instance across the cell's execute runtime.
#[cfg(feature = "cellular")]
pub async fn await_controller_phase_advance(phase: &str) -> Result<()> {
    use crate::cellular::transport::connect::{build_velo, connect_controller};
    use crate::cellular::transport::phaser_velo::PhaserClient;
    use anyhow::Context;

    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV)
        .context("cell has no AIPERF_CELL_CONTROLLER_ADDR")?;
    let cell_id = ModuloCellPartition::from_env()
        .context("cell has no partition env (AIPERF_CELL_ID/_COUNT)")?
        .cell_id();
    let velo = build_velo(cell_bind(&coordinate, "phase-await")).await?;
    let controller = connect_controller(&velo, &coordinate)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect controller: {error}"))?;
    let mut sub = PhaserClient::subscribe(velo, &controller)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} phaser subscribe: {error}"))?;
    sub.await_phase_advance(phase)
        .await
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("cell {cell_id} phaser await phase {phase}: {error}"))
}

/// Send a named per-cell phase signal back to the controller over the cellular control
/// transport.
#[cfg(feature = "cellular")]
pub async fn send_controller_phase_signal(
    phase: &str,
    signal: crate::cellular::transport::CellPhaseSignal,
) -> Result<()> {
    use crate::cellular::transport::connect::{build_velo, connect_controller};
    use crate::cellular::{CellClient, CellMessage, VeloCellClient};
    use anyhow::Context;

    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV)
        .context("cell has no AIPERF_CELL_CONTROLLER_ADDR")?;
    let cell_id = ModuloCellPartition::from_env()
        .context("cell has no partition env (AIPERF_CELL_ID/_COUNT)")?
        .cell_id();
    let velo = build_velo(cell_bind(&coordinate, "phase-signal")).await?;
    let controller = connect_controller(&velo, &coordinate)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect controller: {error}"))?;
    let mut client = VeloCellClient::connect(velo, controller)
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect: {error}"))?;
    client
        .send(&CellMessage::PhaseSignal {
            cell_id,
            phase: phase.to_owned(),
            signal,
        })
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} ship phase signal: {error}"))
}

/// Build this cell's owned dataset index and dispatch its owned slice. When
/// `AIPERF_CELL_DATASET_FANOUT` is set, the controller has broadcast the dataset's
/// request-ids; the cell subscribes over velo (replay-on-attach → its full owned shard),
/// builds an index keyed by request_id filtered to its round-robin owned positions
/// using O(1/N) RAM, and issues each request exactly once through `DispatchTracker`.
/// A distribution miss fails the run. The function is a no-op when the flag is unset.
#[cfg(feature = "cellular")]
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

    // `DispatchTracker` enforces exactly-once issue and counts missing payloads.
    let mut tracker = DispatchTracker::new();
    let mut ok_2xx: u64 = 0;
    for id in index.owned_ids() {
        match tracker.on_issue(id, &index) {
            DispatchDecision::Issue(payload) => {
                let wire: crate::cellular::dispatch_state::WireRequest =
                    rmp_serde::from_slice(&payload)
                        .map_err(|error| anyhow::anyhow!("decode WireRequest {id}: {error}"))?;
                match http_post_json(&wire.url, wire.body).await {
                    Ok(status) if (200..300).contains(&status) => ok_2xx += 1,
                    Ok(status) => {
                        tracing::warn!(request_id = id, status, "controlled-issue non-2xx")
                    }
                    Err(error) => {
                        tracing::warn!(request_id = id, error = %error, "controlled-issue send failed")
                    }
                }
                tracker.on_complete(id);
            }
            other => tracing::warn!(request_id = id, ?other, "unexpected dispatch decision"),
        }
    }
    tracing::info!(
        cell_id,
        cell_count,
        owned = index.len(),
        issued = tracker.issued(),
        completed = tracker.completed(),
        dispatched_2xx = ok_2xx,
        misses = tracker.distribution_misses(),
        "ControlledIssuer: dispatched this cell's owned shard from the fan-out index"
    );
    anyhow::ensure!(
        tracker.distribution_misses() == 0,
        "dataset fan-out delivered incompletely: {} distribution misses on cell {cell_id}",
        tracker.distribution_misses()
    );
    Ok(())
}

/// Minimal HTTP/1.1 POST of `body` (as `application/json`) to `url`, returning the
/// response status code. Off the per-request measurement hot path (this is the
/// controlled-issue dispatch of a cell's owned shard), so a raw hyper client is
/// sufficient and no `Clock` is threaded.
#[cfg(feature = "cellular")]
async fn http_post_json(url: &str, body: Vec<u8>) -> anyhow::Result<u16> {
    use anyhow::Context;
    let rest = url
        .strip_prefix("http://")
        .context("controlled-issue URL must be http://")?;
    let (authority, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (host, port.parse::<u16>().unwrap_or(80)),
        None => (authority, 80),
    };
    let stream = tokio::net::TcpStream::connect((host, port))
        .await
        .with_context(|| format!("connect {host}:{port}"))?;
    let io = hyper_util::rt::TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
        .await
        .context("http1 handshake")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let request = hyper::Request::builder()
        .method(hyper::Method::POST)
        .uri(path)
        .header(hyper::header::HOST, authority)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(http_body_util::Full::new(bytes::Bytes::from(body)))
        .context("build request")?;
    let response = sender.send_request(request).await.context("send request")?;
    Ok(response.status().as_u16())
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
#[cfg(feature = "cellular")]
pub struct CellRecordsShipper {
    cell_id: u32,
    coordinate: String,
}

#[cfg(feature = "cellular")]
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
            std::env::var(crate::engine::cellular_aggregator::AGG_DNS_TEMPLATE_ENV)
            && !template.is_empty()
            && let Some(agg_count) =
                crate::engine::cellular_aggregator::aggregator_count(cell_count)
        {
            return Some(k8s_agg_ship_coordinate(&template, cell_id, agg_count));
        }
        std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()
    }

    /// Builds a shipper that sends to an explicit velo `coordinate` under `cell_id`.
    /// Used by an aggregator to ship its merged store to the controller
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
            crate::engine::heartbeat_lane::observe_ingest(&mut heartbeat, record);
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

    /// Ships this cell's folded exact column store with a counters-only heartbeat.
    /// Exact-fold mode retains no per-record data, so it has no record
    /// `Vec` to ship — it ships the folded store instead, which the controller appends
    /// across cells (`merge_store_partitions`) into the merged report.
    ///
    /// The heartbeat carries EXACT counters (`issued`/`completed`/`errored`, supplied by
    /// the caller from the accumulator's record count and the retained errored subset)
    /// but EMPTY latency sketches: the fold dropped the per-record TTFT/ITL/latency
    /// samples the sketches need. The heartbeat is a live-lane diagnostic; the
    /// authoritative percentiles come from the merged store, so empty sketches are honest
    /// rather than a fidelity loss. The counters keep the controller's
    /// `cellular-heartbeat.json` sidecar populated.
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
        assert_eq!(
            k8s_agg_ship_coordinate(template, 5, 3),
            "tcp://js-aggregators-2-0.js.ns.svc.cluster.local:9700"
        );
    }

    #[test]
    fn detects_only_file_path_datasets_for_ship() {
        let file_path = serde_json::json!({"run": {"cfg": {"datasets": [
            {"type": "file", "format": "single_turn", "path": "/data/prompts.jsonl"}
        ]}}});
        assert_eq!(
            cellular_file_dataset_path(&file_path),
            Some(std::path::PathBuf::from("/data/prompts.jsonl"))
        );

        for graph_format in ["dag_jsonl", "weka_trace", "dynamo_trace"] {
            let graph = serde_json::json!({"run": {"cfg": {"datasets": [
                {"type": "file", "format": graph_format, "path": "/traces/graph.jsonl"}
            ]}}});
            assert_eq!(
                cellular_file_dataset_path(&graph),
                Some(std::path::PathBuf::from("/traces/graph.jsonl")),
                "single-file graph trace {graph_format} must be shipped"
            );
        }

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
