// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records-shard capture and serializable partitions.
//!
//! A [`RecordsShard`] captures a cell's completed records locally and exports a
//! serializable [`RecordsShardPartition`] at a phase boundary. The controller
//! merges every cell partition into the final report.
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
//! - [`ColumnStorePartition`] carries a cell's pre-accumulated [`ColumnStore`]. Its merge
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

use crate::graph::supplement::GraphCellSupplement;
use crate::metrics_core::accumulator::{MetricsAccumulator, MetricsConfig};
use crate::metrics_core::ingest::RecordIngest;
use crate::metrics_core::store::ColumnStore;

/// One cell's captured records, ready for global-order re-ingestion by the
/// controller. This is the authoritative, byte-exact records-shard partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordsShardPartition {
    cell_id: u32,
    records: Vec<RecordIngest>,
    #[serde(default)]
    graph_supplement: Option<GraphCellSupplement>,
}

impl RecordsShardPartition {
    /// Wraps a cell's captured records as its partition.
    pub fn new(cell_id: u32, records: Vec<RecordIngest>) -> Self {
        Self {
            cell_id,
            records,
            graph_supplement: None,
        }
    }

    /// Attach the bounded replay facts produced by this graph cell. These facts are
    /// folded only by the controller after every terminal partition arrives.
    pub fn with_graph_supplement(mut self, supplement: GraphCellSupplement) -> Self {
        self.graph_supplement = Some(supplement);
        self
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

    /// Borrows the graph replay facts carried with this terminal partition.
    pub fn graph_supplement(&self) -> Option<&GraphCellSupplement> {
        self.graph_supplement.as_ref()
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
/// Each record carries the **global cumulative dispatch ordinal**
/// ([`request_index`](RecordIngest::request_index)) the per-cell issuer stamped — the
/// exact absolute slot a single-cell run assigns (warmup block `[0, W)`, then
/// profiling `[W, W+P)`), because the cell adds its phase's global base to its
/// phase-local index. Records are re-ingested in ascending ordinal order, so both
/// absolute-slot placement and ragged `append_order` match a single-cell run: the
/// summary is byte-identical to the same records accumulated as one cell. Run-level
/// scalars (network RTT, injected side-channel scalars) are applied by the caller.
///
/// The union of every cell's global ordinals must be a permutation of `0..total`.
/// This is validated before any record is placed, so a misframed or overlapping wire
/// partition (a duplicate, missing, or out-of-range ordinal — the input here arrives
/// off [`RecordsShardPartition::from_bytes`]) returns a structured
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

/// Merges every cell's records by **dense re-numbering**, for graph-mode cellular
/// where cells cannot pre-tile a global dispatch ordinal.
///
/// The byte-exact sibling [`merge_records_in_global_order`] requires every record's
/// [`request_index`](RecordIngest::request_index) to already be a permutation of
/// `0..total` — the absolute slot a single-cell run assigns. Graph cells cannot
/// supply that: each cell stamps a **local** `request_index` (`0..N` per cell,
/// ordered by wall-clock start), so the ordinals collide across cells. This merge
/// instead concatenates the cells' records in ascending `cell_id` order and assigns a
/// fresh dense global slot to each, preserving every cell's own start-time order
/// through the local index it stamped.
///
/// This is **numerically correct**. `request_index` only selects which absolute
/// column slot a record occupies in the store — it never feeds a metric formula — and
/// phase separation rides each record's [`phase`](RecordIngest::phase) field, not its
/// slot (see [`ExportContext`](crate::metrics_core::window::ExportContext): "phase
/// masks are authoritative over wall-clock bounds"). So re-numbering can neither move
/// a record between phases nor change any per-record value; it only changes which
/// store column the record lands in.
///
/// It is **deterministic-per-topology, not byte-identical** to a single-cell run.
/// Because concatenation interleaves the cells' records in a different order than a
/// single cell's dispatch sequence, the order-sensitive floating-point reductions sum
/// in a different order and can disagree in the last ULP. Sorting the partitions by
/// `cell_id` makes the output independent of partition arrival order, so at a fixed
/// topology the result is bit-stable. Unlike [`merge_records_in_global_order`] this
/// cannot fail: any set of records renumbers into a dense `0..total`, so there is no
/// permutation precondition to reject.
pub fn merge_records_by_concatenation(
    config: MetricsConfig,
    mut partitions: Vec<RecordsShardPartition>,
) -> MetricsAccumulator {
    // Deterministic regardless of arrival order: concatenate in ascending cell_id.
    partitions.sort_by_key(RecordsShardPartition::cell_id);
    let mut accumulator = MetricsAccumulator::with_config(config);
    let mut slot = 0usize;
    for partition in partitions {
        // Preserve each cell's own (start-time) order via the local request_index it
        // stamped, then assign the dense global slot.
        let mut records = partition.into_records();
        records.sort_by_key(|record| record.request_index);
        for mut record in records {
            record.request_index = Some(slot);
            accumulator.process_record(&record);
            slot += 1;
        }
    }
    accumulator
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

/// Captures a cell's completed records as a serializable, mergeable partition.
pub trait RecordsShard {
    /// Captures one completed record into this shard.
    fn capture(&mut self, record: RecordIngest);

    /// Exports the shard's captured records as its partition.
    fn export_partition(&self) -> RecordsShardPartition;

    /// The identifier of the cell this shard belongs to.
    fn cell_id(&self) -> u32;
}

/// One cell's in-process record buffer.
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

/// One cell's pre-accumulated column store for summary shipping.
///
/// Merge ([`merge_store_partitions`]) is [`ColumnStore::append_store`]:
/// deterministic at a fixed topology. It concatenates rows, so its summaries match
/// a single-cell run only up to floating-point summation order; the byte-exact
/// report path is [`RecordsShardPartition`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStorePartition {
    cell_id: u32,
    store: ColumnStore,
    #[serde(default)]
    graph_supplement: Option<GraphCellSupplement>,
}

impl ColumnStorePartition {
    /// Wraps a cell's populated column store as its partition.
    pub fn from_store(cell_id: u32, store: ColumnStore) -> Self {
        Self {
            cell_id,
            store,
            graph_supplement: None,
        }
    }

    /// Attach the bounded replay facts produced by this graph cell or aggregator.
    pub fn with_graph_supplement(mut self, supplement: GraphCellSupplement) -> Self {
        self.graph_supplement = Some(supplement);
        self
    }

    /// Builds an accumulator's store into a partition for `cell_id`.
    pub fn from_accumulator(cell_id: u32, accumulator: &MetricsAccumulator) -> Self {
        Self {
            cell_id,
            store: accumulator.column_store().clone(),
            graph_supplement: None,
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

    /// Borrows the graph replay facts carried with this terminal partition.
    pub fn graph_supplement(&self) -> Option<&GraphCellSupplement> {
        self.graph_supplement.as_ref()
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
    use crate::dispatch::sink::ObservedSpecDecodeAcceptance;
    use crate::metrics_core::catalog::MetricTag;
    use crate::metrics_core::ingest::TokenCounts;
    use crate::metrics_core::store::NumericColumn;
    use crate::metrics_core::value::MetricValue;
    use crate::metrics_core::window::{ExportContext, Phase};
    use std::collections::BTreeMap;

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

    fn spec_decode_partition_record(phase_index: usize, accepted: &[u64]) -> RecordIngest {
        let mut record =
            RecordIngest::minimal(phase_index as i64, phase_index as i64 + 1, Phase::Profiling);
        record.phase_index = Some(phase_index);
        let steps = accepted.len() as u64;
        let accepted_total = accepted.iter().sum::<u64>();
        let mut histogram = BTreeMap::new();
        for value in accepted {
            *histogram.entry(*value).or_insert(0) += 1;
        }
        record.spec_decode_acceptance = Some(ObservedSpecDecodeAcceptance {
            engine: "vllm".to_string(),
            mean_acceptance_length: 1.0 + accepted_total as f64 / steps as f64,
            draft_acceptance_rate: accepted_total as f64 / (steps * 4) as f64,
            acceptance_histogram: histogram,
            num_accepted_draft_tokens: accepted_total,
            num_draft_tokens: steps * 4,
            num_spec_steps: steps,
            num_spec_tokens: Some(4),
            completion_tokens: Some(accepted_total + steps),
            per_step_accepted: None,
            per_step_drafted: None,
        });
        record
    }

    #[test]
    fn spec_decode_pool_survives_exact_and_sketch_column_partition_codecs() {
        for storage_mode in [
            crate::metrics_core::MetricsStorageMode::Exact,
            crate::metrics_core::MetricsStorageMode::Sketch { compression: 100.0 },
        ] {
            let config = MetricsConfig {
                storage_mode,
                ..MetricsConfig::default()
            };
            let mut first = MetricsAccumulator::with_config(config.clone());
            let mut second = MetricsAccumulator::with_config(config.clone());
            first.process_record(&spec_decode_partition_record(0, &[2, 3, 1, 4, 2, 0, 3, 3]));
            second.process_record(&spec_decode_partition_record(1, &[1, 1, 0]));
            let partitions = [
                ColumnStorePartition::from_accumulator(0, &first),
                ColumnStorePartition::from_accumulator(1, &second),
            ]
            .into_iter()
            .map(|partition| {
                ColumnStorePartition::from_bytes(&partition.to_bytes().expect("encode"))
                    .expect("decode")
            })
            .collect();

            let merged = merge_store_partitions(config, partitions);
            let phase = merged.export_results(&ExportContext::phase(Phase::Profiling));
            assert_eq!(
                phase.pooled_spec_decode_acceptance_histogram(),
                Some(&BTreeMap::from([(0, 2), (1, 3), (2, 2), (3, 3), (4, 1)]))
            );
            let indexed = merged.export_results(&ExportContext::phase_index(Phase::Profiling, 1));
            assert_eq!(
                indexed.pooled_spec_decode_acceptance_histogram(),
                Some(&BTreeMap::from([(0, 1), (1, 2)]))
            );
            assert_eq!(
                indexed.finite_value(MetricTag::TotalSpecDecodeSteps),
                Some(3.0)
            );
        }
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
        // Global dispatch-order re-ingestion must reproduce single-cell reductions.
        let records: Vec<_> = (0..24).map(record).collect();
        let direct = accumulator_over(&records);

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
    fn multi_phase_merge_is_byte_identical_to_a_single_cell_run() {
        // A cell stamps the GLOBAL cumulative slot (its phase's global base plus its
        // phase-local index), so warmup fills `[0, W)` and profiling `[W, W+P)` — the
        // same absolute slots a single-cell run assigns. Warmup is intentionally
        // not a multiple of the cell count.
        let (warmup_n, profiling_n, cell_count) = (5usize, 7usize, 2usize);
        // Distinct content per (phase, phase-relative position).
        let content = |warmup: bool, pos: usize| {
            if warmup {
                pos as u64
            } else {
                1000 + pos as u64
            }
        };
        // Global cumulative slot for a phase-relative position.
        let slot = |warmup: bool, pos: usize| if warmup { pos } else { warmup_n + pos };

        // Single-cell reference: every record at its global slot.
        let mut reference = Vec::new();
        for (warmup, count) in [(true, warmup_n), (false, profiling_n)] {
            for pos in 0..count {
                let mut r = record(content(warmup, pos));
                r.phase = if warmup {
                    Phase::Warmup
                } else {
                    Phase::Profiling
                };
                r.request_index = Some(slot(warmup, pos));
                reference.push(r);
            }
        }
        let direct = accumulator_over(&reference);

        // Cells: each owns its round-robin share of each phase (cell k owns
        // phase-relative pos where pos % C == k) and stamps that same global slot.
        let mut cells: Vec<DirectRecordsShard> = (0..cell_count as u32)
            .map(DirectRecordsShard::new)
            .collect();
        for (warmup, count) in [(true, warmup_n), (false, profiling_n)] {
            for pos in 0..count {
                let mut r = record(content(warmup, pos));
                r.phase = if warmup {
                    Phase::Warmup
                } else {
                    Phase::Profiling
                };
                r.request_index = Some(slot(warmup, pos));
                cells[pos % cell_count].capture(r);
            }
        }
        let partitions: Vec<_> = cells
            .iter()
            .map(|cell| {
                let bytes = cell.export_partition().to_bytes().expect("encode");
                RecordsShardPartition::from_bytes(&bytes).expect("decode")
            })
            .collect();

        let merged = merge_records_in_global_order(MetricsConfig::default(), partitions)
            .expect("global ordinals tile 0..W+P");
        assert_eq!(merged.record_count(), warmup_n + profiling_n);
        assert_eq!(
            merged.summarize(),
            direct.summarize(),
            "multi-phase merge must be byte-identical to the single-cell run"
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

    #[test]
    fn sketch_store_partitions_merge_matches_single_sketch_and_carry_the_count() {
        use crate::metrics_core::MetricsStorageMode;
        // Each sketch cell folds records into a bounded t-digest store with no rows.
        // The controller merges the stores associatively.
        // Counts/sums/min/max stay exact; percentiles are t-digest-approximate; and the
        // true record total travels WITH the store — a sketch store's `record_count()`
        // is 0, so `ingested_count()` is the only surviving total across ship+merge.
        let sketch_cfg = || MetricsConfig {
            storage_mode: MetricsStorageMode::Sketch { compression: 100.0 },
            ..MetricsConfig::default()
        };
        let records: Vec<_> = (0..30).map(record).collect();

        // Compare against one sketch over every record.
        let mut single = MetricsAccumulator::with_config(sketch_cfg());
        for r in &records {
            single.process_record(r);
        }

        // Three cells, each folding a disjoint round-robin third into its own sketch,
        // shipped through the msgpack wire form exactly as a cell ships to the controller.
        let mut partitions = Vec::new();
        for cell in 0..3u32 {
            let mut acc = MetricsAccumulator::with_config(sketch_cfg());
            for (idx, r) in records.iter().enumerate() {
                if idx as u32 % 3 == cell {
                    acc.process_record(r);
                }
            }
            // The store folded-and-cleared every row; only the counter survives.
            assert_eq!(acc.record_count(), 0, "a sketch store retains no rows");
            assert_eq!(
                acc.ingested_count(),
                10,
                "per-cell fold count survives the clear"
            );
            let bytes = ColumnStorePartition::from_accumulator(cell, &acc)
                .to_bytes()
                .expect("encode");
            partitions.push(ColumnStorePartition::from_bytes(&bytes).expect("decode"));
        }

        let merged = merge_store_partitions(sketch_cfg(), partitions);

        // The true total survives fold-and-clear + serialize + associative merge, even
        // though the merged store retains no rows (this is what the controller's outcome
        // `record_count` now reads via `ingested_count`).
        assert_eq!(merged.record_count(), 0);
        assert_eq!(merged.ingested_count(), 30);

        let single_sum = single.summarize();
        let merged_sum = merged.summarize();
        assert_eq!(
            merged_sum.finite_value(MetricTag::RequestCount),
            single_sum.finite_value(MetricTag::RequestCount),
            "request count is exact across the sketch merge",
        );
        let single_lat = single_sum
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        let merged_lat = merged_sum
            .result(MetricTag::RequestLatency)
            .unwrap()
            .distribution()
            .unwrap();
        // Exact: count, min, max (t-digest anchors the extrema exactly).
        assert_eq!(merged_lat.count, single_lat.count);
        assert_eq!(merged_lat.min, single_lat.min);
        assert_eq!(merged_lat.max, single_lat.max);
        // Within a few ULPs: the sum/avg (reordered Welford combine on the merge).
        if let (Some(m), Some(s)) = (merged_lat.sum.as_f64(), single_lat.sum.as_f64()) {
            rel_close_f64(m, s, "sketch merge latency sum");
        }
        if let (Some(m), Some(s)) = (merged_lat.avg.as_f64(), single_lat.avg.as_f64()) {
            rel_close_f64(m, s, "sketch merge latency avg");
        }
    }

    /// Relative-tolerance float comparison for store-merge parity:
    /// sums/means may drift a few ULPs from the reordered f64 summation (`~1e-9`).
    fn rel_close_f64(a: f64, b: f64, context: &str) {
        if a == b {
            return;
        }
        let denom = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() / denom <= 1e-9,
            "{context}: {a} vs {b} exceeds 1e-9 relative tolerance"
        );
    }

    /// A finite [`MetricValue`] pair within tolerance; a non-finite pair must match
    /// exactly (a `+inf`/`NaN` sentinel is a present-vs-absent distinction, not a sum).
    fn rel_close_mv(a: MetricValue, b: MetricValue, context: &str) {
        match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => rel_close_f64(a, b, context),
            _ => assert_eq!(a, b, "{context}: non-finite values must match exactly"),
        }
    }

    #[test]
    fn cell_message_store_partition_round_trips_through_messagepack() {
        // A folded ColumnStorePartition carried inside a
        // CellMessage must survive the exact rmp-serde path the velo transport uses,
        // preserving the store's NaN-sparse columns (present-vs-absent semantics).
        use crate::cellular::transport::CellMessage;
        let records: Vec<_> = (0..10).map(record).collect();
        let source = accumulator_over(&records);
        let message = CellMessage::StorePartition(Box::new(
            ColumnStorePartition::from_accumulator(2, &source),
        ));

        let bytes = rmp_serde::to_vec(&message).expect("encode CellMessage::StorePartition");
        let restored: CellMessage = rmp_serde::from_slice(&bytes).expect("decode");
        let CellMessage::StorePartition(partition) = restored else {
            panic!("round-trip must preserve the StorePartition variant");
        };
        assert_eq!(partition.cell_id(), 2);
        assert_eq!(partition.record_count(), 10);
        let restored_acc =
            MetricsAccumulator::from_column_store(MetricsConfig::default(), partition.into_store());
        assert_eq!(
            restored_acc.summarize(),
            source.summarize(),
            "a wire-shipped store partition summarizes identically to its source"
        );
    }

    #[test]
    fn n_store_partitions_merge_within_tolerance_of_the_union() {
        // N cells each fold their round-robin share into a
        // LOCAL-dense EXACT accumulator, ship the folded store over the wire, and the
        // controller appends them. The merged summary must be WITHIN TOLERANCE of a
        // single accumulator fed the union — counts/min/max/percentiles/std bit-exact
        // (order-independent sorted reductions), sums/means within `1e-9` (the
        // concatenated f64 summation order differs from the union's row order).
        let records: Vec<_> = (0..24).map(record).collect();

        // Union reference: every record folded into one accumulator, dense-appended.
        let mut union = MetricsAccumulator::new();
        for record in &records {
            let mut record = record.clone();
            record.request_index = None;
            union.process_record(&record);
        }

        // N cells: round-robin ownership (cell k owns i % N == k), each dense-appended.
        let cell_count = 4usize;
        let mut cells: Vec<MetricsAccumulator> =
            (0..cell_count).map(|_| MetricsAccumulator::new()).collect();
        for (index, record) in records.iter().enumerate() {
            let mut record = record.clone();
            record.request_index = None;
            cells[index % cell_count].process_record(&record);
        }
        // Ship each partition over the wire before merging (as the transport would).
        let partitions: Vec<_> = cells
            .iter()
            .enumerate()
            .map(|(cell_id, accumulator)| {
                let partition = ColumnStorePartition::from_accumulator(cell_id as u32, accumulator);
                let bytes = partition.to_bytes().expect("encode");
                ColumnStorePartition::from_bytes(&bytes).expect("decode")
            })
            .collect();
        let merged = merge_store_partitions(MetricsConfig::default(), partitions);
        assert_eq!(merged.record_count(), 24, "every cell's records are merged");

        let union_out = union.export_results(&ExportContext::phase(Phase::Profiling));
        let merged_out = merged.export_results(&ExportContext::phase(Phase::Profiling));
        assert!(
            !union_out.result_map().is_empty(),
            "the union export must be non-trivial"
        );
        for (tag, u) in union_out.result_map() {
            let m = merged_out
                .result_map()
                .get(tag)
                .unwrap_or_else(|| panic!("merged export missing metric {tag}"));
            match (u.distribution(), m.distribution()) {
                (Some(ud), Some(md)) => {
                    // Count/min/max/percentiles are order-independent set operations over
                    // the SORTED values, so they stay bit-exact across the reordered
                    // concatenation. avg/sum/std ride the row-order f64 sum (std subtracts
                    // avg, so it inherits avg's drift), so they land within `1e-9`.
                    assert_eq!(ud.count, md.count, "{tag} count must be exact");
                    assert_eq!(ud.min, md.min, "{tag} min must be bit-exact (sorted)");
                    assert_eq!(ud.max, md.max, "{tag} max must be bit-exact (sorted)");
                    assert_eq!(
                        ud.percentiles, md.percentiles,
                        "{tag} percentiles must be bit-exact (sorted)"
                    );
                    rel_close_mv(ud.avg, md.avg, &format!("{tag} avg"));
                    rel_close_mv(ud.sum, md.sum, &format!("{tag} sum"));
                    match (ud.std, md.std) {
                        (Some(a), Some(b)) => rel_close_f64(a, b, &format!("{tag} std")),
                        (a, b) => assert_eq!(a, b, "{tag} non-finite std must match"),
                    }
                }
                _ => match (u.finite_value(), m.finite_value()) {
                    (Some(a), Some(b)) => rel_close_f64(a, b, &format!("{tag} scalar")),
                    (a, b) => assert_eq!(a, b, "{tag} non-finite scalar must match"),
                },
            }
        }
    }

    /// A completed record with an explicit phase and the **local** (per-cell)
    /// request_index a graph cell stamps — deliberately colliding across cells.
    fn local_record(seed: u64, phase: Phase, local: usize) -> RecordIngest {
        let mut record = record(seed);
        record.phase = phase;
        record.request_index = Some(local);
        record
    }

    #[test]
    fn concatenation_merges_all_cells_and_renumbers_densely() {
        // Two cells stamp colliding LOCAL ordinals: cell 0 has [0, 1], cell 1 has
        // [0, 1, 2]. If the merge honored those local indices via insert_record_at,
        // the overlap would land five records in three slots; dense re-numbering must
        // place all five at unique global slots.
        let cell0 = RecordsShardPartition::new(
            0,
            vec![
                local_record(10, Phase::Profiling, 0),
                local_record(11, Phase::Profiling, 1),
            ],
        );
        let cell1 = RecordsShardPartition::new(
            1,
            vec![
                local_record(20, Phase::Profiling, 0),
                local_record(21, Phase::Profiling, 1),
                local_record(22, Phase::Profiling, 2),
            ],
        );

        let merged = merge_records_by_concatenation(MetricsConfig::default(), vec![cell0, cell1]);
        assert_eq!(
            merged.record_count(),
            5,
            "every record must occupy a unique dense slot; none dropped or overwritten"
        );
    }

    #[test]
    fn concatenation_is_deterministic_regardless_of_partition_order() {
        // cell_id-ordered concatenation must ignore partition arrival order.
        let p0 = RecordsShardPartition::new(
            0,
            vec![
                local_record(100, Phase::Profiling, 0),
                local_record(101, Phase::Profiling, 1),
            ],
        );
        let p1 = RecordsShardPartition::new(
            1,
            vec![
                local_record(200, Phase::Profiling, 0),
                local_record(201, Phase::Profiling, 1),
                local_record(202, Phase::Profiling, 2),
            ],
        );

        let forward =
            merge_records_by_concatenation(MetricsConfig::default(), vec![p0.clone(), p1.clone()]);
        let reversed = merge_records_by_concatenation(MetricsConfig::default(), vec![p1, p0]);
        assert_eq!(
            forward.export_results(&ExportContext::phase(Phase::Profiling)),
            reversed.export_results(&ExportContext::phase(Phase::Profiling)),
            "sorting partitions by cell_id makes the merge independent of arrival order"
        );
    }

    #[test]
    fn concatenation_preserves_phase_separation() {
        // Phase separation rides each record's `phase` field, not its slot, so
        // re-numbering must never leak a record across the phase boundary. Each cell
        // owns a mix of warmup and profiling records under its own local ordinals.
        let cell0 = RecordsShardPartition::new(
            0,
            vec![
                local_record(10, Phase::Warmup, 0),
                local_record(11, Phase::Profiling, 1),
            ],
        );
        let cell1 = RecordsShardPartition::new(
            1,
            vec![
                local_record(20, Phase::Warmup, 0),
                local_record(21, Phase::Warmup, 1),
                local_record(22, Phase::Profiling, 2),
            ],
        );

        let merged = merge_records_by_concatenation(MetricsConfig::default(), vec![cell0, cell1]);
        assert_eq!(merged.record_count(), 5, "all five records placed");

        let profiling = merged.export_results(&ExportContext::phase(Phase::Profiling));
        let warmup = merged.export_results(&ExportContext::phase(Phase::Warmup));
        assert_eq!(
            profiling.finite_value(MetricTag::RequestCount),
            Some(2.0),
            "profiling export sees only the two profiling records"
        );
        assert_eq!(
            warmup.finite_value(MetricTag::RequestCount),
            Some(3.0),
            "warmup export sees only the three warmup records"
        );
    }
}
