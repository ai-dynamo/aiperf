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
//! Cell behaviour is injected through three environment variables so the ordinary
//! execute path is reused unchanged: [`CELL_ID_ENV`] / [`CELL_COUNT_ENV`] select
//! the issuer's partition (read by `RunCapture`), and [`CELL_CONTROLLER_ADDR_ENV`]
//! points the records shipper at the controller. They are set once, before any
//! runtime exists, from the [`CellLaunchSpec`] the controller pipes in.

use aiperf::cellular::partition::CellPartition;
use aiperf::cellular::{
    CellClient, CellMessage, CellularAutonomousIssuer, IssuanceAuthority, MetricsHeartbeat,
    ModuloCellPartition, RecordsShardPartition, TcpCellClient,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Env var carrying the controller's `host:port` transport address (the cell id and
/// count live in [`aiperf::cellular::partition`]'s env vars).
pub const CELL_CONTROLLER_ADDR_ENV: &str = "AIPERF_CELL_CONTROLLER_ADDR";

/// The launch descriptor the controller serializes to each cell child's stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellLaunchSpec {
    /// This cell's zero-based identifier.
    pub cell_id: u32,
    /// Total number of cells the run is partitioned across.
    pub cell_count: u32,
    /// The controller's transport address the cell connects back to.
    pub controller_addr: String,
    /// The full protocol-v2 `execute` envelope for this cell's budget slice.
    pub envelope: serde_json::Value,
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
}
