// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cellular-runtime seams — one measurement/execution model at two scales.
//!
//! A *cell* is a unit of autonomous scale: a thread, a process, or a node. The
//! single process is a **cell of one**; a cluster is `N` cells. Nothing on the hot
//! path changes shape between them. This module owns the traits the fully
//! autonomous cellular runtime consumes, each shipped today with exactly the
//! basic ("Direct", single-process, in-process-sharded) concrete impl, so the
//! distributed runtime is a **drop-in behind the seam**, not a rearchitecture.
//!
//! Design record: `specs/2026-07-12-cellular-ready-seams-and-roadmap.md`. Four of
//! the five seams are built here today, each with its Direct impl:
//!
//! - **S1 [`issuance`]** — [`issuance::IssuanceAuthority`]: assign the dense global
//!   dispatch ordinal for each issued turn through one seam
//!   ([`issuance::DirectIssuanceAuthority`] today; a per-cell autonomous issuer
//!   later).
//! - **S2 [`shard`]** — [`shard::RecordsShard`] + the serializable
//!   [`shard::RecordsShardPartition`] (byte-exact) / [`shard::ColumnStorePartition`]
//!   (summary): local per-record capture with a mergeable, wire-serializable
//!   partition ([`shard::DirectRecordsShard`] in-process; the Phase-2 controller ships
//!   each cell's partition across the transport and merges them in global order).
//! - **S3 [`heartbeat`] + [`sketch`]** — [`heartbeat::MetricsHeartbeat`]: a
//!   bounded-cadence live snapshot of counters + associatively-mergeable
//!   [`sketch::TDigest`] sketches (TTFT / ITL / latency), aggregated across shards
//!   ([`heartbeat::HeartbeatAccumulator`]). Report percentiles stay exact from S2;
//!   the sketch is live-only. The single-process live lane is built in the runner, and
//!   the Phase-2 controller aggregates every cell's shipped heartbeat over the transport
//!   into one run-wide snapshot (counters summed, sketches t-digest-merged).
//! - **S4 [`partition`]** — [`partition::CellPartition`]: the deterministic
//!   `(cell_id, cell_count)` work partition ([`partition::ModuloCellPartition`],
//!   identity `(0, 1)` today).
//!
//! - **Transport [`transport`]** — [`transport::ControllerTransport`] /
//!   [`transport::CellClient`]: the abstracted cross-node seam carrying
//!   [`transport::CellMessage`]s (heartbeats, partitions) from a cell to the
//!   controller. The concrete impl is velo-backed (`velo` feature): a
//!   `VeloControllerTransport` / `VeloCellClient` pair over rmp raw payloads, with
//!   discovery-free connection via [`transport::connect`]
//!   (`specs/2026-07-15-velo-cell-transport-design.md`). The multi-process /
//!   multi-pod controller/cell topology that drives it lives in `aiperf-runner`.
//!
//! Everything here is object-safe where it crosses a `dyn` boundary and generic
//! where it is hot-path monomorphized, per the crate's extensibility discipline.

pub mod broadcast;
pub mod dataset_session;
pub mod dispatch_state;
pub mod heartbeat;
pub mod issuance;
pub mod partition;
pub mod phaser;
pub mod shard;
pub mod sketch;
pub mod transport;

pub use heartbeat::{
    HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation, MetricsHeartbeat,
};
pub use issuance::{CellularAutonomousIssuer, DirectIssuanceAuthority, IssuanceAuthority};
pub use partition::{CellPartition, CellPartitionError, ModuloCellPartition};
pub use shard::{
    ColumnStorePartition, DirectRecordsShard, PartitionCodecError, RecordsMergeError, RecordsShard,
    RecordsShardPartition, merge_records_by_concatenation, merge_records_in_global_order,
    merge_store_partitions,
};
pub use sketch::TDigest;
#[cfg(feature = "velo")]
pub use transport::velo_transport::{SpecFor, VeloCellClient, VeloControllerTransport};
pub use transport::{
    CellAck, CellClient, CellMessage, CellPartitionShip, CellRegister, CellStorePartitionShip,
    CellTransportError, ControllerTransport, HANDLER_HEARTBEAT, HANDLER_PARTITION,
    HANDLER_REGISTER, HANDLER_STORE_PARTITION,
};
