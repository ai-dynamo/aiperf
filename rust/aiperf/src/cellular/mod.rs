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
//! Design record: `specs/2026-07-12-cellular-ready-seams-and-roadmap.md`. The five
//! seams and their Direct impls:
//!
//! - **S1 [`issuance`]** — [`issuance::IssuanceAuthority`]: acquire a credit +
//!   admission slot and the dense global dispatch ordinal through one seam
//!   ([`issuance::DirectIssuanceAuthority`] today; a per-cell autonomous issuer
//!   later).
//! - **S2 [`shard`]** — [`shard::RecordsShard`] + the serializable
//!   [`shard::ColumnStorePartition`]: local per-record accumulation with a
//!   mergeable, wire-serializable partition ([`shard::DirectRecordsShard`] today;
//!   a per-cell records-shard moved across a transport later).
//! - **S3 [`heartbeat`]** — [`heartbeat::MetricsHeartbeat`]: a bounded-cadence
//!   snapshot of counters + associatively-mergeable t-digest sketches, aggregated
//!   across shards (in-process merge today; cross-cell heartbeat aggregation
//!   later). Report percentiles stay exact from S2; the sketch is live-only.
//! - **S4 [`partition`]** — [`partition::CellPartition`]: the deterministic
//!   `(cell_id, cell_count)` work partition ([`partition::ModuloCellPartition`],
//!   identity `(0, 1)` today).
//! - **Transport [`transport`]** — [`transport::CellTransport`]: the abstracted
//!   cross-node communication seam that carries heartbeats and partitions from a
//!   cell to the controller (in-process today; framed TCP for real multi-process).
//!
//! Everything here is object-safe where it crosses a `dyn` boundary and generic
//! where it is hot-path monomorphized, per the crate's extensibility discipline.

pub mod partition;
pub mod shard;

pub use partition::{CellPartition, CellPartitionError, ModuloCellPartition};
pub use shard::{
    ColumnStorePartition, DirectRecordsShard, PartitionCodecError, RecordsShard,
    RecordsShardPartition, merge_records_in_global_order, merge_store_partitions,
};
