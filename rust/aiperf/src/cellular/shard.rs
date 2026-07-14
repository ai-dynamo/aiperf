// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S2 — the records-shard seam and its serializable partitions.
//!
//! A [`RecordsShard`] captures a cell's completed records locally and exports a
//! serializable [`RecordsShardPartition`] at a phase boundary
//! (`specs/2026-07-12-cellular-ready-seams-and-roadmap.md`, S2). The controller
//! merges every cell's partition into the final report.
//!
//! Two partition forms, because two properties are wanted:
//!
//! - [`RecordsShardPartition`] carries the cell's raw [`RecordIngest`] records.
//!   The controller re-ingests every cell's records **in global dispatch-ordinal
//!   order** ([`merge_records_in_global_order`]) — the exact mechanism the
//!   single-process path uses to make its report independent of worker count. This
//!   is the **authoritative, byte-exact** path: because both the numeric columns
//!   (summed in absolute-slot order) and the ragged columns (summed in
//!   ingest/`append_order`) then match a single-cell run, the merged report is
//!   byte-identical to the same run executed as one cell.
//! - [`ColumnStorePartition`] carries a cell's pre-accumulated [`ColumnStore`] —
//!   the roadmap's serializable "the store *is* the partition" form. Its merge
//!   ([`merge_store_partitions`]) is [`ColumnStore::append_store`]: associative and
//!   **deterministic at a fixed topology**, but because it concatenates rows its
//!   floating-point summation order differs from a single-cell run, so its
//!   summaries match only up to the last ULP. It is the cheap live/summary form;
//!   the byte-exact report uses [`RecordsShardPartition`].
//!
//! Both wire forms use MessagePack: a self-describing binary format that preserves
//! the NaN/`+inf` sentinels JSON cannot and round-trips the untagged `MetricValue`
//! encoding a non-self-describing format cannot.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::metrics_core::accumulator::{MetricsAccumulator, MetricsConfig};
use crate::metrics_core::ingest::RecordIngest;
use crate::metrics_core::store::ColumnStore;

/// One cell's captured records, ready for global-order re-ingestion by the
/// controller. This is the authoritative, byte-exact records-shard partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordsShardPartition {
    cell_id: u32,
    records: Vec<RecordIngest>,
}

impl RecordsShardPartition {
    /// Wraps a cell's captured records as its partition.
    pub fn new(cell_id: u32, records: Vec<RecordIngest>) -> Self {
        Self { cell_id, records }
    }

    /// The identifier of the cell that produced this partition.
    pub fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// Borrows the captured records.
    pub fn records(&self) -> &[RecordIngest] {
        &self.records
    }

    /// Consumes the partition, returning the owned records.
    pub fn into_records(self) -> Vec<RecordIngest> {
        self.records
    }

    /// The number of captured records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the cell captured no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Serializes to the MessagePack wire form.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PartitionCodecError> {
        rmp_serde::to_vec(self).map_err(|error| PartitionCodecError::Encode(error.to_string()))
    }

    /// Deserializes from the MessagePack wire form.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PartitionCodecError> {
        rmp_serde::from_slice(bytes).map_err(|error| PartitionCodecError::Decode(error.to_string()))
    }
}

/// Merges every cell's records into one accumulator for the final report.
///
/// Records are re-ingested in ascending global dispatch-ordinal
/// ([`request_index`](RecordIngest::request_index)) order, so both absolute-slot
/// placement and ragged `append_order` match a single-cell run: the summary is
/// byte-identical to the same records accumulated as one cell. Run-level scalars
/// (network RTT, injected side-channel scalars) are applied by the caller.
///
/// The union of every cell's global ordinals must be a permutation of `0..total`.
/// This is validated before any record is placed, so a misframed or overlapping
/// wire partition (a duplicate, missing, or out-of-range ordinal — the input here
/// arrives off [`RecordsShardPartition::from_bytes`]) returns a structured
/// [`RecordsMergeError`] instead of an `insert_record_at` panic or an
/// O(max-ordinal) allocation.
pub fn merge_records_in_global_order(
    config: MetricsConfig,
    partitions: Vec<RecordsShardPartition>,
) -> Result<MetricsAccumulator, RecordsMergeError> {
    let mut records: Vec<RecordIngest> = partitions
        .into_iter()
        .flat_map(RecordsShardPartition::into_records)
        .collect();

    let total = records.len();
    let mut seen = vec![false; total];
    for record in &records {
        match record.request_index {
            Some(ordinal) if ordinal < total => {
                if std::mem::replace(&mut seen[ordinal], true) {
                    return Err(RecordsMergeError::DuplicateOrdinal(ordinal));
                }
            }
            Some(ordinal) => return Err(RecordsMergeError::OrdinalOutOfRange { ordinal, total }),
            None => return Err(RecordsMergeError::MissingOrdinal),
        }
    }

    // Stable sort by the dense global dispatch ordinal so the re-ingested order —
    // and therefore every order-sensitive floating-point reduction — reproduces a
    // single-cell run exactly. Validated above as a permutation of 0..total.
    records.sort_by_key(|record| record.request_index);

    let mut accumulator = MetricsAccumulator::with_config(config);
    for record in &records {
        accumulator.process_record(record);
    }
    Ok(accumulator)
}

/// Error merging cell records: the union of global ordinals was not a permutation
/// of `0..total` (dense, unique, in range), so re-ingestion could not proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordsMergeError {
    /// A record carried no global dispatch ordinal.
    MissingOrdinal,
    /// Two records claimed the same global ordinal.
    DuplicateOrdinal(usize),
    /// A global ordinal fell outside `0..total`.
    OrdinalOutOfRange {
        /// The offending ordinal.
        ordinal: usize,
        /// The record count it was checked against.
        total: usize,
    },
}

impl Display for RecordsMergeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingOrdinal => write!(f, "a cell record carried no global dispatch ordinal"),
            Self::DuplicateOrdinal(ordinal) => {
                write!(f, "two cell records claimed global ordinal {ordinal}")
            }
            Self::OrdinalOutOfRange { ordinal, total } => write!(
                f,
                "global ordinal {ordinal} is out of range for {total} merged records"
            ),
        }
    }
}

impl std::error::Error for RecordsMergeError {}

/// S2 seam: a shard captures a cell's completed records and exports them as a
/// serializable, mergeable partition at a phase boundary.
///
/// Object-safe so the runtime can hold `Box<dyn RecordsShard>`; the deferred
/// per-cell records-shard drops in behind this trait unchanged.
pub trait RecordsShard {
    /// Captures one completed record into this shard.
    fn capture(&mut self, record: RecordIngest);

    /// Exports the shard's captured records as its partition.
    fn export_partition(&self) -> RecordsShardPartition;

    /// The identifier of the cell this shard belongs to.
    fn cell_id(&self) -> u32;
}

/// The Tier-0 in-process records shard: one cell's captured record buffer.
#[derive(Debug)]
pub struct DirectRecordsShard {
    cell_id: u32,
    records: Vec<RecordIngest>,
}

impl DirectRecordsShard {
    /// Builds an empty shard for `cell_id`.
    pub fn new(cell_id: u32) -> Self {
        Self {
            cell_id,
            records: Vec::new(),
        }
    }

    /// The number of records captured so far.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the shard has captured no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Borrows the captured records.
    pub fn records(&self) -> &[RecordIngest] {
        &self.records
    }
}

impl RecordsShard for DirectRecordsShard {
    fn capture(&mut self, record: RecordIngest) {
        self.records.push(record);
    }

    fn export_partition(&self) -> RecordsShardPartition {
        RecordsShardPartition::new(self.cell_id, self.records.clone())
    }

    fn cell_id(&self) -> u32 {
        self.cell_id
    }
}

/// One cell's pre-accumulated column store — the roadmap's serializable "store is
/// the partition" form, for cheap live/summary shipping.
///
/// Merge ([`merge_store_partitions`]) is [`ColumnStore::append_store`]:
/// deterministic at a fixed topology. It concatenates rows, so its summaries match
/// a single-cell run only up to floating-point summation order; the byte-exact
/// report path is [`RecordsShardPartition`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStorePartition {
    cell_id: u32,
    store: ColumnStore,
}

impl ColumnStorePartition {
    /// Wraps a cell's populated column store as its partition.
    pub fn from_store(cell_id: u32, store: ColumnStore) -> Self {
        Self { cell_id, store }
    }

    /// Builds an accumulator's store into a partition for `cell_id`.
    pub fn from_accumulator(cell_id: u32, accumulator: &MetricsAccumulator) -> Self {
        Self {
            cell_id,
            store: accumulator.column_store().clone(),
        }
    }

    /// The identifier of the cell that produced this partition.
    pub fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// The number of populated request records the partition carries.
    pub fn record_count(&self) -> usize {
        self.store.record_count()
    }

    /// Borrows the underlying column store.
    pub fn store(&self) -> &ColumnStore {
        &self.store
    }

    /// Consumes the partition, returning the owned column store.
    pub fn into_store(self) -> ColumnStore {
        self.store
    }

    /// Appends another partition's rows into this one (ascending `cell_id` order).
    pub fn append(&mut self, other: &ColumnStorePartition) {
        self.store.append_store(&other.store);
    }

    /// Serializes to the MessagePack wire form.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PartitionCodecError> {
        rmp_serde::to_vec(self).map_err(|error| PartitionCodecError::Encode(error.to_string()))
    }

    /// Deserializes from the MessagePack wire form.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PartitionCodecError> {
        rmp_serde::from_slice(bytes).map_err(|error| PartitionCodecError::Decode(error.to_string()))
    }
}

/// Merges store partitions in ascending `cell_id` order into one accumulator.
///
/// The reduction is [`ColumnStore::append_store`]: deterministic at a fixed
/// topology (see [`ColumnStorePartition`] for the byte-parity caveat). Empty input
/// yields an empty accumulator.
pub fn merge_store_partitions(
    config: MetricsConfig,
    mut partitions: Vec<ColumnStorePartition>,
) -> MetricsAccumulator {
    partitions.sort_by_key(ColumnStorePartition::cell_id);
    let mut merged: Option<ColumnStorePartition> = None;
    for partition in partitions {
        match merged.as_mut() {
            Some(accumulated) => accumulated.append(&partition),
            None => merged = Some(partition),
        }
    }
    match merged {
        Some(partition) => MetricsAccumulator::from_column_store(config, partition.into_store()),
        None => MetricsAccumulator::with_config(config),
    }
}

/// Error encoding or decoding a partition on the wire.
///
/// A plain enum with a hand-written [`Display`] per the crate's error convention;
/// the underlying codec error is captured as a string so this type carries no
/// serde-library type in its public surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionCodecError {
    /// Serialization to the wire form failed.
    Encode(String),
    /// Deserialization from the wire form failed.
    Decode(String),
}

impl Display for PartitionCodecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(error) => write!(f, "failed to encode shard partition: {error}"),
            Self::Decode(error) => write!(f, "failed to decode shard partition: {error}"),
        }
    }
}

impl std::error::Error for PartitionCodecError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::ingest::TokenCounts;
    use crate::metrics_core::store::NumericColumn;
    use crate::metrics_core::value::MetricValue;
    use crate::metrics_core::window::Phase;

    /// A completed record at global dispatch ordinal `idx`, with real
    /// latency/TTFT/ITL/OSL so summaries are non-trivial.
    fn record(idx: u64) -> RecordIngest {
        let start = 1_000_000_000 + idx as i64 * 10_000_000;
        let end = start + 5_000_000 + idx as i64 * 100_000;
        let mut record = RecordIngest::minimal(start, end, Phase::Profiling);
        record.request_index = Some(idx as usize);
        record.first_token_ns = Some(start + 1_000_000);
        record.token_arrival_ns = vec![
            start + 1_000_000,
            start + 2_000_000,
            start + 3_100_000,
            start + 4_300_000,
        ];
        record.tokens = TokenCounts {
            output: Some(4),
            ..TokenCounts::default()
        };
        record
    }

    fn accumulator_over(records: &[RecordIngest]) -> MetricsAccumulator {
        let mut accumulator = MetricsAccumulator::new();
        for record in records {
            accumulator.process_record(record);
        }
        accumulator
    }

    #[test]
    fn numeric_column_nan_absence_survives_the_wire() {
        // The absence sentinel is NaN; MessagePack must round-trip the raw bits so
        // a wire-shipped store keeps present-vs-absent semantics exactly.
        let mut column = NumericColumn::new();
        column.push_f64(1.5);
        column.push_absent();
        column.push_f64(3.5);

        let bytes = rmp_serde::to_vec(&column).expect("encode");
        let restored: NumericColumn = rmp_serde::from_slice(&bytes).expect("decode");

        assert_eq!(restored.get(0), Some(1.5));
        assert!(restored.values()[1].is_nan(), "absent slot must stay NaN");
        assert_eq!(restored.get(1), None, "absent slot reads as absent");
        assert_eq!(restored.get(2), Some(3.5));
        assert_eq!(restored.present_count(), 2);
    }

    #[test]
    fn metric_value_round_trips_through_messagepack() {
        for value in [
            MetricValue::Finite(3.5),
            MetricValue::Finite(-0.0),
            MetricValue::PosInf,
            MetricValue::Absent,
        ] {
            let bytes = rmp_serde::to_vec(&value).expect("encode");
            let restored: MetricValue = rmp_serde::from_slice(&bytes).expect("decode");
            assert_eq!(restored, value, "MetricValue must round-trip on the wire");
        }
    }

    #[test]
    fn records_partition_wire_round_trip_is_lossless_and_stable() {
        let mut shard = DirectRecordsShard::new(0);
        for idx in 0..16 {
            let mut record = record(idx);
            // Exercise the MetricValue wire on a populated override.
            record.metric_overrides.push((
                crate::metrics_core::catalog::MetricTag::NumImages,
                MetricValue::Finite(2.0),
            ));
            shard.capture(record);
        }
        let partition = shard.export_partition();
        assert_eq!(partition.cell_id(), 0);
        assert_eq!(partition.len(), 16);

        let bytes = partition.to_bytes().expect("encode");
        let restored = RecordsShardPartition::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.cell_id(), 0);
        assert_eq!(
            restored.records(),
            partition.records(),
            "records round-trip exactly"
        );
        let bytes_again = restored.to_bytes().expect("re-encode");
        assert_eq!(
            bytes, bytes_again,
            "record wire must be byte-stable across a round trip"
        );
    }

    #[test]
    fn merged_cell_records_are_byte_identical_to_a_single_cell_run() {
        // The flagship S2 contract: re-ingesting every cell's records in global
        // dispatch order reproduces a single-cell run byte-for-byte, including the
        // last-ULP floating-point reductions.
        let records: Vec<_> = (0..24).map(record).collect();
        let direct = accumulator_over(&records);

        // Round-robin ownership (cell k owns i % 3 == k), as the S4 partition would.
        let mut cells = [
            DirectRecordsShard::new(0),
            DirectRecordsShard::new(1),
            DirectRecordsShard::new(2),
        ];
        for (index, record) in records.iter().enumerate() {
            cells[index % 3].capture(record.clone());
        }

        // Ship each partition over the wire before merging (as a transport would).
        let partitions: Vec<_> = cells
            .iter()
            .map(|cell| {
                let bytes = cell.export_partition().to_bytes().expect("encode");
                RecordsShardPartition::from_bytes(&bytes).expect("decode")
            })
            .collect();

        let merged = merge_records_in_global_order(MetricsConfig::default(), partitions)
            .expect("cell ordinals tile 0..24");
        assert_eq!(merged.record_count(), 24);
        assert_eq!(
            merged.summarize(),
            direct.summarize(),
            "merge of cell records must be byte-identical to the single-cell run"
        );
    }

    #[test]
    fn merge_rejects_ordinals_that_are_not_a_permutation() {
        let dup = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![
                RecordsShardPartition::new(0, vec![record(0)]),
                RecordsShardPartition::new(1, vec![record(0)]),
            ],
        );
        assert_eq!(dup.unwrap_err(), RecordsMergeError::DuplicateOrdinal(0));

        let out_of_range = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![RecordsShardPartition::new(0, vec![record(9)])],
        );
        assert_eq!(
            out_of_range.unwrap_err(),
            RecordsMergeError::OrdinalOutOfRange {
                ordinal: 9,
                total: 1
            }
        );

        let mut indexless = record(0);
        indexless.request_index = None;
        let missing = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![RecordsShardPartition::new(0, vec![indexless])],
        );
        assert_eq!(missing.unwrap_err(), RecordsMergeError::MissingOrdinal);
    }

    #[test]
    fn record_merge_is_independent_of_partition_arrival_order() {
        let records: Vec<_> = (0..18).map(record).collect();
        let mut cells = [
            DirectRecordsShard::new(0),
            DirectRecordsShard::new(1),
            DirectRecordsShard::new(2),
        ];
        for (index, record) in records.iter().enumerate() {
            cells[index % 3].capture(record.clone());
        }
        let p: Vec<_> = cells
            .iter()
            .map(DirectRecordsShard::export_partition)
            .collect();

        let ascending = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![p[0].clone(), p[1].clone(), p[2].clone()],
        )
        .expect("cell ordinals tile 0..18");
        let shuffled = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![p[2].clone(), p[0].clone(), p[1].clone()],
        )
        .expect("cell ordinals tile 0..18");
        assert_eq!(ascending.summarize(), shuffled.summarize());
    }

    #[test]
    fn empty_records_partition_merges_to_empty_accumulator() {
        let merged = merge_records_in_global_order(
            MetricsConfig::default(),
            vec![
                RecordsShardPartition::new(0, Vec::new()),
                RecordsShardPartition::new(1, Vec::new()),
            ],
        )
        .expect("no records is a trivial permutation");
        assert_eq!(merged.record_count(), 0);
    }

    #[test]
    fn store_partition_wire_round_trip_preserves_summary() {
        // The store partition is a lossy-order but lossless-content wire form: a
        // round-tripped store summarizes identically to its source.
        let records: Vec<_> = (0..12).map(record).collect();
        let source = accumulator_over(&records);
        let partition = ColumnStorePartition::from_accumulator(3, &source);
        assert_eq!(partition.cell_id(), 3);

        let bytes = partition.to_bytes().expect("encode");
        let restored = ColumnStorePartition::from_bytes(&bytes).expect("decode");
        let restored_acc =
            MetricsAccumulator::from_column_store(MetricsConfig::default(), restored.into_store());
        assert_eq!(restored_acc.summarize(), source.summarize());
    }

    #[test]
    fn store_partition_merge_is_deterministic_at_a_fixed_topology() {
        let records: Vec<_> = (0..20).map(record).collect();
        let mut cell0 = MetricsAccumulator::new();
        let mut cell1 = MetricsAccumulator::new();
        for (index, record) in records.iter().enumerate() {
            let mut record = record.clone();
            // Dense per-cell slots so append_store's dense-store precondition holds.
            record.request_index = None;
            if index % 2 == 0 {
                cell0.process_record(&record);
            } else {
                cell1.process_record(&record);
            }
        }
        let p0 = ColumnStorePartition::from_accumulator(0, &cell0);
        let p1 = ColumnStorePartition::from_accumulator(1, &cell1);

        let first = merge_store_partitions(MetricsConfig::default(), vec![p0.clone(), p1.clone()]);
        let second = merge_store_partitions(MetricsConfig::default(), vec![p1, p0]);
        // cell_id-ordered reduction is identical regardless of input order.
        assert_eq!(first.summarize(), second.summarize());
        assert_eq!(first.record_count(), 20);
    }
}
