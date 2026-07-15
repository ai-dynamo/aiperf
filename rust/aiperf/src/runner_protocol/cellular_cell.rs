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

/// Env var carrying the per-phase global ordinal bases as JSON (`{name: base}`), so a
/// cell's issuer recovers each turn's single-cell absolute slot from its phase-local
/// slot (the cell's sampler restarts each phase; see [`phase_ordinal_bases_from_env`]).
pub const CELL_PHASE_ORDINAL_BASES_ENV: &str = "AIPERF_CELL_PHASE_ORDINAL_BASES";

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

// -- velo cell transport (fetch spec + ship records) ------------------------------

/// The velo bind for this cell, chosen from the controller coordinate scheme: a
/// `tcp://` coordinate is k8s (bind an ephemeral routable TCP port); anything else
/// is a co-located launcher (UDS on unix, loopback elsewhere). `role` disambiguates
/// the cell's fetch vs ship velo instances so their UDS paths do not collide.
#[cfg(feature = "velo")]
fn cell_bind(coordinate: &str, role: &str) -> crate::cellular::transport::connect::BindSpec {
    use crate::cellular::transport::connect::BindSpec;
    if coordinate.starts_with("tcp://") {
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
    use crate::cellular::transport::connect::{
        BootstrapSource, build_velo, resolve_controller_peer,
    };
    use anyhow::Context;

    let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV)
        .context("cell has no AIPERF_CELL_CONTROLLER_ADDR")?;
    let cell_id = ModuloCellPartition::from_env()
        .context("cell has no partition env (AIPERF_CELL_ID/_COUNT)")?
        .cell_id();
    let velo = build_velo(cell_bind(&coordinate, "fetch")).await?;
    let source = BootstrapSource::parse(&coordinate)?;
    let controller = resolve_controller_peer(&source).await?;
    let client = VeloCellClient::connect(velo, controller)
        .map_err(|error| anyhow::anyhow!("cell {cell_id} connect: {error}"))?;
    let reply = client
        .register(cell_id)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} register: {error}"))?;
    // Block until the controller triggers the synchronized START — every cell
    // resumes together once all cells have registered. A poisoned event (the
    // controller aborted before starting) surfaces here as an error.
    client
        .await_start(reply.start_event)
        .await
        .map_err(|error| anyhow::anyhow!("cell {cell_id} await start: {error}"))?;
    Ok(reply.envelope)
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
        let coordinate = std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()?;
        let cell_id = ModuloCellPartition::from_env()?.cell_id();
        Some(Self {
            cell_id,
            coordinate,
        })
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
        use crate::cellular::transport::connect::{
            BootstrapSource, build_velo, resolve_controller_peer,
        };
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
                let source = BootstrapSource::parse(&coordinate)?;
                let controller = resolve_controller_peer(&source).await?;
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
