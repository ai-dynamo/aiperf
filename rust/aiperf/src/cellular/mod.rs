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
//! Design record: `specs/2026-07-12-cellular-ready-seams-and-roadmap.md`. Three of
//! the five seams are built here today, each with its Direct impl:
//!
//! - **S1 [`issuance`]** — [`issuance::IssuanceAuthority`]: assign the dense global
//!   dispatch ordinal for each issued turn through one seam
//!   ([`issuance::DirectIssuanceAuthority`] today; a per-cell autonomous issuer
//!   later).
//! - **S2 [`shard`]** — [`shard::RecordsShard`] + the serializable
//!   [`shard::RecordsShardPartition`] (byte-exact) / [`shard::ColumnStorePartition`]
//!   (summary): local per-record capture with a mergeable, wire-serializable
//!   partition ([`shard::DirectRecordsShard`] today; a per-cell records-shard moved
//!   across a transport later).
//! - **S4 [`partition`]** — [`partition::CellPartition`]: the deterministic
//!   `(cell_id, cell_count)` work partition ([`partition::ModuloCellPartition`],
//!   identity `(0, 1)` today).
//!
//! The remaining two seams are **designed, not yet built** (see the roadmap): an
//! S3 `MetricsHeartbeat` (a bounded-cadence snapshot of counters plus
//! associatively-mergeable t-digest sketches, aggregated across shards; report
//! percentiles stay exact from S2, the sketch is live-only) and a `CellTransport`
//! (the abstracted cross-node seam carrying heartbeats and partitions from a cell
//! to the controller). Both land in later phases.
//!
//! Everything here is object-safe where it crosses a `dyn` boundary and generic
//! where it is hot-path monomorphized, per the crate's extensibility discipline.

pub mod heartbeat;
pub mod issuance;
pub mod partition;
pub mod shard;
pub mod sketch;

pub use heartbeat::{
    HeartbeatAccumulator, HeartbeatCounters, HeartbeatSaturation, MetricsHeartbeat,
};
pub use issuance::{CellularAutonomousIssuer, DirectIssuanceAuthority, IssuanceAuthority};
pub use partition::{CellPartition, CellPartitionError, ModuloCellPartition};
pub use shard::{
    ColumnStorePartition, DirectRecordsShard, PartitionCodecError, RecordsMergeError, RecordsShard,
    RecordsShardPartition, merge_records_in_global_order, merge_store_partitions,
};
pub use sketch::TDigest;
