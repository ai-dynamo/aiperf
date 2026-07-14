// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cell-mode support for the multi-process cellular runtime (Phase 2).
//!
//! A *cell* is one `aiperf-runner --cell` child of the controller. It runs the
//! ordinary online scheduled execution over its budget slice, but with the
//! `CellularAutonomousIssuer` assigning dense global dispatch ordinals from its
//! `(cell_id, cell_count)` partition, and it ships its captured records to the
//! controller over the [`transport`](aiperf::cellular::transport) seam instead of
//! writing a report. The controller re-ingests every cell's records in global
//! ordinal order for the single authoritative `native-v2.json`.
//!
//! Cell behaviour is injected through four environment variables so the ordinary
//! execute path is reused unchanged: [`CELL_ID_ENV`] / [`CELL_COUNT_ENV`] select the
//! issuer's partition (read by `RunCapture`), [`CELL_CONTROLLER_ADDR_ENV`] points the
//! records shipper at the controller, and [`CELL_PHASE_ORDINAL_BASES_ENV`] carries
//! each phase's global ordinal base so the issuer stamps single-cell-equivalent
//! absolute slots. They are set once, before any runtime exists, from the
//! [`CellLaunchSpec`] the controller pipes in.

use std::collections::{BTreeMap, HashMap};

use aiperf::cellular::partition::CellPartition;
use aiperf::cellular::{
    CellClient, CellMessage, CellularAutonomousIssuer, HeartbeatAccumulator, HeartbeatCounters,
    HeartbeatSaturation, IssuanceAuthority, MetricsHeartbeat, ModuloCellPartition,
    RecordsShardPartition, TcpCellClient,
};
use aiperf::metrics_core::{Phase, RecordIngest};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Env var carrying the controller's `host:port` transport address (the cell id and
/// count live in [`aiperf::cellular::partition`]'s env vars).
pub const CELL_CONTROLLER_ADDR_ENV: &str = "AIPERF_CELL_CONTROLLER_ADDR";

/// Env var carrying the per-phase global ordinal bases as JSON (`{name: base}`), so a
/// cell's issuer recovers each turn's single-cell absolute slot from its phase-local
/// slot (the cell's sampler restarts each phase; see [`phase_ordinal_bases_from_env`]).
pub const CELL_PHASE_ORDINAL_BASES_ENV: &str = "AIPERF_CELL_PHASE_ORDINAL_BASES";

/// The launch descriptor the controller serializes to each cell child's stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellLaunchSpec {
    /// This cell's zero-based identifier.
    pub cell_id: u32,
    /// Total number of cells the run is partitioned across.
    pub cell_count: u32,
    /// The controller's transport address the cell connects back to.
    pub controller_addr: String,
    /// Each phase's global ordinal base (`phase name -> turns dispatched by prior
    /// phases`), so the cell stamps single-cell-equivalent absolute slots.
    pub phase_ordinal_bases: BTreeMap<String, u64>,
    /// The full protocol-v2 `execute` envelope for this cell's budget slice.
    pub envelope: serde_json::Value,
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
    let by_name: BTreeMap<String, u64> = serde_json::from_str(&raw).unwrap_or_default();
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
/// [`DirectIssuanceAuthority`]: aiperf::cellular::DirectIssuanceAuthority
pub fn issuance_authority_from_env() -> std::rc::Rc<dyn IssuanceAuthority> {
    match ModuloCellPartition::from_env() {
        Some(partition) => std::rc::Rc::new(CellularAutonomousIssuer::new(partition)),
        None => std::rc::Rc::new(aiperf::cellular::DirectIssuanceAuthority::new()),
    }
}

/// Ships a cell's final records-shard partition to the controller over the
/// transport, when this process is a cell (the controller address is set).
pub struct CellRecordsShipper {
    cell_id: u32,
    controller_addr: String,
}

impl CellRecordsShipper {
    /// Builds a shipper when the controller address and cell partition env vars are
    /// set, else `None` (the ordinary single-process path).
    pub fn from_env() -> Option<Self> {
        let controller_addr = std::env::var(CELL_CONTROLLER_ADDR_ENV).ok()?;
        let cell_id = ModuloCellPartition::from_env()?.cell_id();
        Some(Self {
            cell_id,
            controller_addr,
        })
    }

    /// The cell's identifier.
    pub fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// Connects to the controller and ships this cell's final heartbeat then its
    /// records-shard partition, and closes. Blocking, called once after the run
    /// completes — never on the hot path. The controller merges the cells'
    /// heartbeats (counters by sum, sketches by t-digest merge) into one run-wide
    /// view and their partitions into the single report.
    pub fn ship(
        &self,
        partition: RecordsShardPartition,
        heartbeat: MetricsHeartbeat,
    ) -> Result<()> {
        let mut client = TcpCellClient::connect(&self.controller_addr)
            .with_context(|| format!("cell {} connecting to controller", self.cell_id))?;
        client
            .send(&CellMessage::Heartbeat {
                cell_id: self.cell_id,
                heartbeat: Box::new(heartbeat),
            })
            .context("cell shipping heartbeat")?;
        client
            .send(&CellMessage::Partition(partition))
            .context("cell shipping partition")?;
        Ok(())
    }

    /// Builds this cell's terminal heartbeat from its final records and ships the
    /// records-shard partition + heartbeat to the controller. Shared by the scheduled and
    /// graph cell paths, which differ only in how they derive `records` and the run-span
    /// `epoch_ns`. One end-of-run aggregate (not a per-tick snapshot); saturation is zero
    /// (the run has drained).
    pub fn ship_records(&self, records: Vec<RecordIngest>, epoch_ns: i64) -> Result<()> {
        let mut heartbeat = HeartbeatAccumulator::new();
        let mut completed = 0_u64;
        let mut errored = 0_u64;
        for record in &records {
            crate::heartbeat_lane::observe_ingest(&mut heartbeat, record);
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
        self.ship(
            RecordsShardPartition::new(self.cell_id(), records),
            heartbeat,
        )
    }
}
