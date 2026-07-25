// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Absolute-request-index-addressed columnar storage for inference metric records.
//!
//! Numeric columns keep absolute request-index alignment with NaN absence sentinels.
//! Metadata dimensions are stored separately and categorical values receive dense
//! first-appearance codes. The sparse-column and query semantics and the exact
//! CSR list replay are implemented in this module.

use crate::cellular::sketch::TDigest;
use crate::metrics_core::catalog::MetricTag;
use crate::metrics_core::ingest::{InferenceDimensions, RecordIngest, RequestTrace, UsageMetrics};
use crate::metrics_core::value::MetricValue;
use crate::metrics_core::window::{ExportContext, Phase};
use rustc_hash::{FxHashMap, FxHasher};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// A NaN-sparse numeric column aligned by absolute request index.
///
/// Serialized with a binary serde format only: the NaN absence sentinels are not
/// representable in JSON, so a wire-shipped [`ColumnStorePartition`] uses a format
/// that round-trips raw `f64` bits (see `crate::cellular::shard`).
///
/// [`ColumnStorePartition`]: crate::cellular::shard::ColumnStorePartition
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NumericColumn {
    values: Vec<f64>,
}

impl NumericColumn {
    /// Builds an empty numeric column.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a column with `rows` absent entries.
    pub fn with_absent_rows(rows: usize) -> Self {
        Self {
            values: vec![f64::NAN; rows],
        }
    }

    /// Appends a raw value; NaN denotes absence while infinity remains present.
    pub fn push_f64(&mut self, value: f64) {
        self.values.push(value);
    }

    /// Appends an explicit absent value.
    pub fn push_absent(&mut self) {
        self.values.push(f64::NAN);
    }

    /// Appends a boundary-safe metric value.
    pub fn push_metric_value(&mut self, value: MetricValue) {
        self.push_f64(raw_metric_value(value));
    }

    /// Replaces one absolute request-index row.
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the column.
    pub fn set_f64(&mut self, row: usize, value: f64) {
        self.values[row] = value;
    }

    /// Replaces one row with a boundary-safe metric value.
    pub fn set_metric_value(&mut self, row: usize, value: MetricValue) {
        self.set_f64(row, raw_metric_value(value));
    }

    /// Returns a present finite value, treating every non-finite value as absent.
    pub fn get(&self, row: usize) -> Option<f64> {
        self.values
            .get(row)
            .copied()
            .filter(|value| value.is_finite())
    }

    /// Returns the raw index-aligned values, including NaN sentinels.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the stable absolute-request-index-order sum of present values.
    pub fn running_sum(&self) -> f64 {
        self.values.iter().filter(|value| !value.is_nan()).sum()
    }

    /// Returns the number of present values.
    pub fn present_count(&self) -> usize {
        self.values.iter().filter(|value| !value.is_nan()).count()
    }

    /// Returns present values selected by an index-aligned mask.
    ///
    /// # Panics
    ///
    /// Panics when `mask` has a different row count.
    pub fn masked_values(&self, mask: &[bool]) -> Vec<f64> {
        assert_eq!(self.values.len(), mask.len());
        self.values
            .iter()
            .zip(mask)
            .filter_map(|(value, selected)| (*selected && value.is_finite()).then_some(*value))
            .collect()
    }

    /// Returns the absolute-request-index-order sum and count for selected present rows.
    pub fn masked_sum_count(&self, mask: &[bool]) -> (f64, usize) {
        assert_eq!(self.values.len(), mask.len());
        self.values
            .iter()
            .zip(mask)
            .filter(|(value, selected)| **selected && value.is_finite())
            .fold((0.0, 0), |(sum, count), (value, _)| {
                (sum + *value, count + 1)
            })
    }

    /// Returns the row count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true when the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn resize_absent(&mut self, rows: usize) {
        if self.values.len() < rows {
            self.values.resize(rows, f64::NAN);
        }
    }

    /// Drops every row while retaining the backing allocation (sketch-mode reuse).
    fn clear(&mut self) {
        self.values.clear();
    }
}

/// Borrowed exact replay data exposed by a list-metric backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaggedReplay<'a> {
    /// Concatenated list values.
    pub values: &'a [f64],
    /// First flat-value offset for every record; unused rows contain zero.
    pub offsets: &'a [usize],
    /// Flat-value count for every record; unused rows contain zero.
    pub lengths: &'a [usize],
    /// Record indices in the order their contiguous value slices were appended.
    pub append_order: &'a [usize],
}

impl RaggedReplay<'_> {
    /// Expands record ownership lazily without retaining one index per value.
    pub fn record_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.append_order
            .iter()
            .copied()
            .flat_map(|row| std::iter::repeat_n(row, self.lengths[row]))
    }
}

/// Extension seam for exact or bounded-memory list-valued metric storage.
pub trait ListMetricBackend: Debug + Default {
    /// Prepares index metadata for an absolute request-slot span.
    fn prepare_rows(&mut self, _rows: usize) {}

    /// Appends one record's list values.
    fn add_for_record(&mut self, row: usize, values: &[f64]);

    /// Appends one record's generated list values without requiring the caller
    /// to allocate a temporary contiguous buffer.
    fn add_for_record_iter(&mut self, row: usize, values: &mut dyn Iterator<Item = f64>) {
        let values = values.collect::<Vec<_>>();
        self.add_for_record(row, &values);
    }

    /// Returns all values selected by a record mask.
    fn values_for_mask(&self, record_mask: &[bool]) -> Vec<f64>;

    /// Returns exact per-record replay data when retained by this backend.
    fn replay(&self) -> Option<RaggedReplay<'_>>;

    /// Appends another backend after shifting its record indices by `row_offset`.
    /// `other_rows` includes rows with no list value so later appends stay aligned.
    fn append_shifted(&mut self, other: &Self, row_offset: usize, other_rows: usize);
}

/// Exact CSR-style list backend used for ICL distributions and sweep replay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RaggedSeries {
    values: Vec<f64>,
    offsets: Vec<usize>,
    lengths: Vec<usize>,
    append_order: Vec<usize>,
    present: Vec<bool>,
}

impl RaggedSeries {
    /// Builds an empty exact ragged series.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all concatenated values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Expands owning record indices lazily in flat-value order.
    pub fn record_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.append_order
            .iter()
            .copied()
            .flat_map(|row| std::iter::repeat_n(row, self.lengths[row]))
    }

    /// Returns one start offset per record; absent rows contain zero.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns one value count per absolute record slot.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Returns the values stored for one record.
    pub fn values_for_record(&self, row: usize) -> Option<&[f64]> {
        if !self.present.get(row).copied().unwrap_or(false) {
            return None;
        }
        let start = self.offsets[row];
        let end = start + self.lengths[row];
        Some(&self.values[start..end])
    }

    /// Computes cumulative sums that reset at each record boundary.
    pub fn grouped_cumsum(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.values.len());
        for &row in &self.append_order {
            let mut cumulative = 0.0;
            for &value in self
                .values_for_record(row)
                .expect("append order contains only present rows")
            {
                cumulative += value;
                result.push(cumulative);
            }
        }
        result
    }
}

impl ListMetricBackend for RaggedSeries {
    fn prepare_rows(&mut self, rows: usize) {
        self.offsets.resize(rows, 0);
        self.lengths.resize(rows, 0);
        self.present.resize(rows, false);
    }

    fn add_for_record(&mut self, row: usize, values: &[f64]) {
        if self.offsets.len() <= row {
            self.offsets.resize(row + 1, 0);
            self.lengths.resize(row + 1, 0);
            self.present.resize(row + 1, false);
        }
        assert!(!self.present[row], "a ragged row may only be appended once");
        if values.is_empty() {
            return;
        }
        self.offsets[row] = self.values.len();
        self.lengths[row] = values.len();
        self.present[row] = true;
        self.append_order.push(row);
        self.values.extend_from_slice(values);
    }

    fn add_for_record_iter(&mut self, row: usize, values: &mut dyn Iterator<Item = f64>) {
        if self.offsets.len() <= row {
            self.offsets.resize(row + 1, 0);
            self.lengths.resize(row + 1, 0);
            self.present.resize(row + 1, false);
        }
        assert!(!self.present[row], "a ragged row may only be appended once");
        let start = self.values.len();
        self.values.extend(values);
        let added = self.values.len() - start;
        if added == 0 {
            return;
        }
        self.offsets[row] = start;
        self.lengths[row] = added;
        self.present[row] = true;
        self.append_order.push(row);
    }

    fn values_for_mask(&self, record_mask: &[bool]) -> Vec<f64> {
        let mut selected = Vec::new();
        for &row in &self.append_order {
            if record_mask.get(row).copied().unwrap_or(false) {
                selected.extend_from_slice(
                    self.values_for_record(row)
                        .expect("append order contains only present rows"),
                );
            }
        }
        selected
    }

    fn replay(&self) -> Option<RaggedReplay<'_>> {
        Some(RaggedReplay {
            values: &self.values,
            offsets: &self.offsets,
            lengths: &self.lengths,
            append_order: &self.append_order,
        })
    }

    fn append_shifted(&mut self, other: &Self, row_offset: usize, other_rows: usize) {
        if self.offsets.len() < row_offset {
            self.offsets.resize(row_offset, 0);
            self.lengths.resize(row_offset, 0);
            self.present.resize(row_offset, false);
        }
        for row in 0..other_rows {
            let Some(values) = other.values_for_record(row) else {
                continue;
            };
            self.add_for_record(row_offset + row, values);
        }
        if self.offsets.len() < row_offset + other_rows {
            self.offsets.resize(row_offset + other_rows, 0);
            self.lengths.resize(row_offset + other_rows, 0);
            self.present.resize(row_offset + other_rows, false);
        }
    }
}

/// Dense, first-appearance categorical interner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoryInterner<T: Eq + Hash> {
    codes_by_hash: FxHashMap<u64, HashCodes>,
    values: Vec<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HashCodes {
    One(u32),
    Collision(Vec<u32>),
}

impl<T: Eq + Hash> Default for CategoryInterner<T> {
    fn default() -> Self {
        Self {
            codes_by_hash: FxHashMap::default(),
            values: Vec::new(),
        }
    }
}

impl<T> CategoryInterner<T>
where
    T: Clone + Eq + Hash,
{
    /// Returns the existing dense code or inserts the value at the next code.
    pub fn intern(&mut self, value: T) -> u32 {
        let hash = category_hash(&value);
        if let Some(code) = self.code_with_hash(hash, &value) {
            return code;
        }
        self.insert_new(hash, value)
    }

    /// Returns the existing dense code, cloning only when the value is new.
    pub fn intern_ref(&mut self, value: &T) -> u32 {
        let hash = category_hash(value);
        if let Some(code) = self.code_with_hash(hash, value) {
            return code;
        }
        self.insert_new(hash, value.clone())
    }

    /// Looks up a previously interned value.
    pub fn code<Q>(&self, value: &Q) -> Option<u32>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.code_with_hash(category_hash(value), value)
    }

    /// Returns the interned value for a dense code.
    pub fn value(&self, code: u32) -> Option<&T> {
        self.values.get(code as usize)
    }

    /// Returns values in first-appearance order.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the number of unique values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true when no value has been interned.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn insert_new(&mut self, hash: u64, value: T) -> u32 {
        let code = u32::try_from(self.values.len()).expect("category cardinality exceeds u32");
        self.values.push(value);
        match self.codes_by_hash.get_mut(&hash) {
            Some(slot) => match slot {
                HashCodes::One(existing) => {
                    let existing = *existing;
                    *slot = HashCodes::Collision(vec![existing, code]);
                }
                HashCodes::Collision(codes) => codes.push(code),
            },
            None => {
                self.codes_by_hash.insert(hash, HashCodes::One(code));
            }
        }
        code
    }

    fn code_with_hash<Q>(&self, hash: u64, value: &Q) -> Option<u32>
    where
        T: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let matches = |code: u32| <T as Borrow<Q>>::borrow(&self.values[code as usize]) == value;
        match self.codes_by_hash.get(&hash)? {
            HashCodes::One(code) => matches(*code).then_some(*code),
            HashCodes::Collision(codes) => codes.iter().copied().find(|code| matches(*code)),
        }
    }
}

// The interner serializes as its dense first-appearance `values` alone; the
// `codes_by_hash` index is a rebuildable acceleration structure, not primary
// state. Re-interning the values in order reproduces byte-identical dense codes
// (code = insertion position) and an equal index, so the wire form is lossless
// while carrying no redundant hash table.
impl<T> Serialize for CategoryInterner<T>
where
    T: Eq + Hash + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.values.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for CategoryInterner<T>
where
    T: Clone + Eq + Hash + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let values = Vec::<T>::deserialize(deserializer)?;
        let mut interner = Self::default();
        for value in values {
            interner.intern(value);
        }
        Ok(interner)
    }
}

fn category_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Selects how a [`ColumnStore`] retains per-record metric values.
///
/// Exact retention keeps every value for exact percentiles at the cost of memory
/// linear in the record count. Sketch retention streams each Record-metric value
/// into a bounded-memory t-digest (approximate percentiles; exact count/sum/min/max)
/// so memory stays O(1) in the record count — the opt-in high-request-rate mode.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MetricsStorageMode {
    /// Retain every per-record value (exact percentiles, O(records) memory).
    #[default]
    Exact,
    /// Stream each value into a per-tag t-digest sketch (approximate percentiles,
    /// exact count/sum/min/max, O(1) memory).
    Sketch {
        /// t-digest compression (δ); larger keeps more centroids, finer quantiles.
        compression: f64,
    },
}

/// One tag's streaming aggregate: a t-digest for approximate percentiles plus an
/// exact running count/sum/min/max and a Welford mean/M2 for the standard
/// deviation. Every scalar except the percentiles is exact; the percentiles track
/// the exact linear-interpolation band to well under a percent on broad
/// distributions (see the t-digest convergence test in `cellular::sketch`).
///
/// The running sum is summed in record-arrival order rather than the exact path's
/// absolute-row order, so an integer-valued metric's sum is bitwise identical while
/// a float metric's sum may differ by a few ULPs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TagSketch {
    digest: TDigest,
    sum: f64,
    count: u64,
    min: f64,
    max: f64,
    mean: f64,
    m2: f64,
}

impl TagSketch {
    /// Builds an empty sketch with an explicit t-digest compression.
    fn with_compression(compression: f64) -> Self {
        Self {
            digest: TDigest::with_compression(compression),
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Ingests one finite value, updating every exact aggregate and the digest.
    fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.digest.add(value);
    }

    /// Merges another shard's sketch: exact aggregates combine by the parallel
    /// (Chan) Welford update and the digests merge associatively.
    fn merge(&mut self, other: &TagSketch) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let (count_a, count_b) = (self.count as f64, other.count as f64);
        let total = count_a + count_b;
        let delta = other.mean - self.mean;
        self.mean += delta * count_b / total;
        self.m2 += other.m2 + delta * delta * count_a * count_b / total;
        self.count += other.count;
        self.sum += other.sum;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.digest.merge(&other.digest);
    }

    /// The number of ingested values.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The exact record-arrival-order sum of ingested values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// The exact minimum ingested value (`+inf` when empty).
    pub fn min(&self) -> f64 {
        self.min
    }

    /// The exact maximum ingested value (`-inf` when empty).
    pub fn max(&self) -> f64 {
        self.max
    }

    /// The standard deviation with the given delta-degrees-of-freedom, from the
    /// streaming Welford M2 (population std at `ddof == 0`).
    pub fn std(&self, ddof: usize) -> f64 {
        let denom = (self.count as usize).saturating_sub(ddof);
        if denom == 0 {
            0.0
        } else {
            (self.m2 / denom as f64).sqrt()
        }
    }

    /// Estimates several quantiles (each `q` in `[0, 1]`) from the digest.
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<Option<f64>> {
        self.digest.quantiles(quantiles)
    }
}

/// Per-`(phase, tag)` bounded-memory streaming sketches.
///
/// Phase separation preserves the warmup/profiling phase
/// mask the exact path applies over rows, and the whole structure merges
/// associatively so cell partitions combine the same way single-process workers do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SketchColumns {
    compression: f64,
    tags: FxHashMap<(Phase, u16), TagSketch>,
}

impl SketchColumns {
    /// Builds an empty sketch column set with the given digest compression.
    pub fn new(compression: f64) -> Self {
        Self {
            compression,
            tags: FxHashMap::default(),
        }
    }

    /// Ingests one finite value for a `(phase, tag)`.
    pub fn add(&mut self, phase: Phase, tag: MetricTag, value: f64) {
        let compression = self.compression;
        self.tags
            .entry((phase, tag.index() as u16))
            .or_insert_with(|| TagSketch::with_compression(compression))
            .add(value);
    }

    /// Returns the sketch for one `(phase, tag)`, if any value was ingested.
    pub fn tag(&self, phase: Phase, tag: MetricTag) -> Option<&TagSketch> {
        self.tags.get(&(phase, tag.index() as u16))
    }

    /// Resolves a tag sketch for an optional phase context: a specific phase
    /// returns that phase's sketch; `None` merges every phase's sketch for the tag.
    pub fn resolve(&self, phase: Option<Phase>, tag: MetricTag) -> Option<TagSketch> {
        let index = tag.index() as u16;
        match phase {
            Some(phase) => self.tags.get(&(phase, index)).cloned(),
            None => {
                let mut merged: Option<TagSketch> = None;
                for ((_, tag_index), sketch) in &self.tags {
                    if *tag_index != index {
                        continue;
                    }
                    match &mut merged {
                        Some(accumulated) => accumulated.merge(sketch),
                        None => merged = Some(sketch.clone()),
                    }
                }
                merged
            }
        }
    }

    /// Merges another shard's sketch columns into this one.
    pub fn merge(&mut self, other: &SketchColumns) {
        for (key, sketch) in &other.tags {
            let compression = self.compression;
            self.tags
                .entry(*key)
                .or_insert_with(|| TagSketch::with_compression(compression))
                .merge(sketch);
        }
    }
}

/// Absolute-request-index-aligned metric and metadata columns.
///
/// A worker's store is directly serializable and mergeable as a
/// [`ColumnStorePartition`]. Serde bounds are conditional on the list backend `B`, so the trait
/// [`ListMetricBackend`] stays serde-free and only stores whose `B` is
/// serializable gain the wire form. Use a binary serde format — the NaN-sparse
/// numeric columns are not JSON-representable.
///
/// [`ColumnStorePartition`]: crate::cellular::shard::ColumnStorePartition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStore<B: ListMetricBackend = RaggedSeries> {
    occupied: Vec<bool>,
    occupied_count: usize,
    start_ns: Vec<f64>,
    end_ns: Vec<f64>,
    generation_start_ns: Vec<f64>,
    observed_output_sequence_length: Vec<f64>,
    session_nums: Vec<u64>,
    turn_indices: Vec<u32>,
    phase_codes: Vec<u32>,
    correlation_codes: Vec<u32>,
    dimension_codes: Vec<u32>,
    worker_codes: Vec<Option<u32>>,
    conversation_codes: Vec<Option<u32>>,
    errored: Vec<bool>,
    canceled: Vec<bool>,
    phases: CategoryInterner<Phase>,
    correlations: CategoryInterner<String>,
    dimensions: CategoryInterner<InferenceDimensions>,
    workers: CategoryInterner<String>,
    conversations: CategoryInterner<String>,
    // Dense columns use `MetricTag::index()` because the tag set is a small fixed
    // enum; the `*_present` lists keep iteration limited to populated tags.
    numeric: Vec<Option<NumericColumn>>,
    numeric_present: Vec<MetricTag>,
    ragged: Vec<Option<B>>,
    ragged_present: Vec<MetricTag>,
    // Bounded-memory sketch retention. `Some` only under
    // [`MetricsStorageMode::Sketch`]; the accumulator then uses the row columns as
    // a transient single-record scratch, harvesting each finished record into this
    // sketch and clearing the rows so memory stays O(1) in the record count.
    #[serde(default)]
    sketch: Option<SketchColumns>,
    // Monotonic count of every record ever ingested, unaffected by [`Self::clear_rows`]
    // (which resets `occupied_count` after each sketch harvest) and summed on
    // [`Self::append_store`]. It is the true record total on the sketch path, where the
    // store retains no rows; on exact/exact-fold stores it equals `occupied_count`. It
    // travels with the serialized store so a cellular controller's merged sketch store
    // reports the real record count, not 0. `#[serde(default)]` keeps wire compatibility.
    #[serde(default)]
    ingested_total: u64,
}

impl<B: ListMetricBackend> Default for ColumnStore<B> {
    fn default() -> Self {
        Self {
            occupied: Vec::new(),
            occupied_count: 0,
            start_ns: Vec::new(),
            end_ns: Vec::new(),
            generation_start_ns: Vec::new(),
            observed_output_sequence_length: Vec::new(),
            session_nums: Vec::new(),
            turn_indices: Vec::new(),
            phase_codes: Vec::new(),
            correlation_codes: Vec::new(),
            dimension_codes: Vec::new(),
            worker_codes: Vec::new(),
            conversation_codes: Vec::new(),
            errored: Vec::new(),
            canceled: Vec::new(),
            phases: CategoryInterner::default(),
            correlations: CategoryInterner::default(),
            dimensions: CategoryInterner::default(),
            workers: CategoryInterner::default(),
            conversations: CategoryInterner::default(),
            numeric: (0..MetricTag::COUNT).map(|_| None).collect(),
            numeric_present: Vec::new(),
            ragged: (0..MetricTag::COUNT).map(|_| None).collect(),
            ragged_present: Vec::new(),
            sketch: None,
            ingested_total: 0,
        }
    }
}

impl<B: ListMetricBackend> ColumnStore<B> {
    /// Builds an empty column store with the selected list backend.
    pub fn with_list_backend() -> Self {
        Self::default()
    }

    /// Builds an empty column store honoring the requested retention mode.
    ///
    /// Under [`MetricsStorageMode::Sketch`] the store allocates the per-tag sketch
    /// columns; the row columns then serve only as a one-record scratch that the
    /// accumulator harvests and clears.
    pub fn with_storage_mode(mode: MetricsStorageMode) -> Self {
        let mut store = Self::default();
        if let MetricsStorageMode::Sketch { compression } = mode {
            store.sketch = Some(SketchColumns::new(compression));
        }
        store
    }

    /// Returns the bounded-memory sketch columns when the store is in sketch mode.
    pub fn sketch(&self) -> Option<&SketchColumns> {
        self.sketch.as_ref()
    }

    /// Harvests one populated row's finite metric values into the sketch columns,
    /// keyed by `phase`, so the row storage can then be cleared. A no-op when the
    /// store is not in sketch mode.
    pub fn harvest_row_to_sketch(&mut self, row: usize, phase: Phase) {
        let Some(mut sketch) = self.sketch.take() else {
            return;
        };
        for &tag in &self.numeric_present {
            if let Some(column) = &self.numeric[tag.index()]
                && let Some(value) = column.get(row)
            {
                sketch.add(phase, tag, value);
            }
        }
        if !self.ragged_present.is_empty() {
            let mut mask = vec![false; self.row_count()];
            if let Some(slot) = mask.get_mut(row) {
                *slot = true;
            }
            for &tag in &self.ragged_present {
                if let Some(backend) = &self.ragged[tag.index()] {
                    for value in backend.values_for_mask(&mask) {
                        sketch.add(phase, tag, value);
                    }
                }
            }
        }
        self.sketch = Some(sketch);
    }

    /// Clears every row-addressed column and categorical interner while retaining
    /// the sketch columns and allocated capacity. Used between records in sketch
    /// mode so the transient row scratch never grows.
    pub fn clear_rows(&mut self) {
        self.occupied.clear();
        self.occupied_count = 0;
        self.start_ns.clear();
        self.end_ns.clear();
        self.generation_start_ns.clear();
        self.observed_output_sequence_length.clear();
        self.session_nums.clear();
        self.turn_indices.clear();
        self.phase_codes.clear();
        self.correlation_codes.clear();
        self.dimension_codes.clear();
        self.worker_codes.clear();
        self.conversation_codes.clear();
        self.errored.clear();
        self.canceled.clear();
        self.phases = CategoryInterner::default();
        self.correlations = CategoryInterner::default();
        self.dimensions = CategoryInterner::default();
        self.workers = CategoryInterner::default();
        self.conversations = CategoryInterner::default();
        for column in self.numeric.iter_mut().flatten() {
            column.clear();
        }
        for backend in self.ragged.iter_mut().flatten() {
            *backend = B::default();
        }
    }

    /// Prepares every index-aligned column for an absolute request-slot span.
    ///
    /// Subsequent inserts still decide which slots are occupied; this only
    /// removes per-record vector growth from known-size ingestion paths. A no-op
    /// in sketch mode, where the store never retains more than one record.
    pub fn prepare_request_slots(&mut self, rows: usize) {
        if self.sketch.is_some() {
            return;
        }
        self.ensure_row_count(rows);
        for backend in self.ragged.iter_mut().flatten() {
            backend.prepare_rows(rows);
        }
    }

    /// Appends one raw record for producers without an authored absolute index.
    pub fn push_record(&mut self, record: &RecordIngest) -> usize {
        let row = self.row_count();
        self.insert_record_at(row, record);
        row
    }

    /// Appends one record while borrowing its token-arrival storage.
    pub fn push_record_with_token_arrivals(
        &mut self,
        record: &RecordIngest,
        token_arrivals_ns: &[i64],
    ) -> usize {
        let row = self.row_count();
        self.insert_record_at_with_token_arrivals(row, record, token_arrivals_ns);
        row
    }

    /// Inserts one raw record into its absolute zero-based request slot.
    ///
    /// # Panics
    ///
    /// Panics when the slot was already populated.
    pub fn insert_record_at(&mut self, row: usize, record: &RecordIngest) {
        self.insert_record_at_with_token_arrivals(row, record, &record.token_arrival_ns);
    }

    /// Inserts one raw record while borrowing its token-arrival storage.
    pub fn insert_record_at_with_token_arrivals(
        &mut self,
        row: usize,
        record: &RecordIngest,
        token_arrivals_ns: &[i64],
    ) {
        self.ensure_row_count(row.saturating_add(1));
        assert!(
            !self.occupied[row],
            "request slot {row} was already populated"
        );
        self.occupied_count += 1;
        // Monotonic total surviving `clear_rows` (sketch mode's per-record fold-and-clear).
        // Every ingest path (`push_record_*`, `insert_record_at*`) funnels through here.
        self.ingested_total += 1;
        self.populate_dimensions(row, record);
        self.populate_raw_metrics(row, record, token_arrivals_ns);
        for (tag, value) in &record.metric_overrides {
            self.set_metric_value(row, *tag, *value);
        }
    }

    /// Appends every row and allocated metric column from another worker store.
    ///
    /// Categorical codes are re-interned in append order, numeric absence remains
    /// index-aligned, and list-valued record indices are shifted exactly once.
    pub fn append_store(&mut self, other: &Self) {
        // Carry the true record total across the merge (it is the only surviving count
        // on the sketch path, where both stores retain no rows).
        self.ingested_total = self.ingested_total.saturating_add(other.ingested_total);
        // Sketch partitions merge associatively; a sketch-mode store retains no
        // rows, so this is the whole merge on that path.
        if let (Some(sketch), Some(other_sketch)) = (self.sketch.as_mut(), other.sketch.as_ref()) {
            sketch.merge(other_sketch);
        }
        let row_offset = self.row_count();
        let other_rows = other.row_count();
        if other_rows == 0 {
            return;
        }
        assert!(
            other.occupied.iter().all(|occupied| *occupied),
            "worker stores must be dense before append"
        );
        self.occupied_count += other_rows;

        // Remap each categorical column's dense codes once per unique value.
        // `values()` is in dense first-appearance order, which matches the
        // row-walk first-appearance order, so interning it up front yields the
        // same self codes as per-row interning would — but at one clone/lookup
        // per unique value instead of per row (~1M clones for low-cardinality
        // columns over 1M rows).
        let phase_remap: Vec<u32> = other
            .phases
            .values()
            .iter()
            .map(|phase| self.phases.intern_ref(phase))
            .collect();
        let correlation_remap: Vec<u32> = other
            .correlations
            .values()
            .iter()
            .map(|correlation| self.correlations.intern_ref(correlation))
            .collect();
        let dimension_remap: Vec<u32> = other
            .dimensions
            .values()
            .iter()
            .map(|dimensions| self.dimensions.intern_ref(dimensions))
            .collect();
        let worker_remap: Vec<u32> = other
            .workers
            .values()
            .iter()
            .map(|worker| self.workers.intern_ref(worker))
            .collect();
        let conversation_remap: Vec<u32> = other
            .conversations
            .values()
            .iter()
            .map(|conversation| self.conversations.intern_ref(conversation))
            .collect();

        for row in 0..other_rows {
            self.occupied.push(true);
            self.start_ns.push(other.start_ns[row]);
            self.end_ns.push(other.end_ns[row]);
            self.generation_start_ns
                .push(other.generation_start_ns[row]);
            self.observed_output_sequence_length
                .push(other.observed_output_sequence_length[row]);
            self.session_nums.push(other.session_nums[row]);
            self.turn_indices.push(other.turn_indices[row]);

            self.phase_codes
                .push(phase_remap[other.phase_codes[row] as usize]);
            self.correlation_codes
                .push(correlation_remap[other.correlation_codes[row] as usize]);
            self.dimension_codes
                .push(dimension_remap[other.dimension_codes[row] as usize]);
            self.worker_codes
                .push(other.worker_codes[row].map(|code| worker_remap[code as usize]));
            self.conversation_codes
                .push(other.conversation_codes[row].map(|code| conversation_remap[code as usize]));
            self.errored.push(other.errored[row]);
            self.canceled.push(other.canceled[row]);
        }

        for column in self.numeric.iter_mut().flatten() {
            for _ in 0..other_rows {
                column.push_absent();
            }
        }
        for &tag in &other.numeric_present {
            let Some(other_column) = &other.numeric[tag.index()] else {
                continue;
            };
            let column = self.numeric_column_or_insert(tag, row_offset + other_rows);
            for (row, value) in other_column.values().iter().copied().enumerate() {
                if !value.is_nan() {
                    column.set_f64(row_offset + row, value);
                }
            }
        }
        for &tag in &other.ragged_present {
            let Some(other_backend) = &other.ragged[tag.index()] else {
                continue;
            };
            self.ragged_backend_or_insert(tag).append_shifted(
                other_backend,
                row_offset,
                other_rows,
            );
        }
    }

    /// Returns the numeric column for `tag`, allocating an absent-filled column
    /// (and recording the tag as present) on first use.
    fn numeric_column_or_insert(&mut self, tag: MetricTag, rows: usize) -> &mut NumericColumn {
        let index = tag.index();
        if self.numeric[index].is_none() {
            self.numeric[index] = Some(NumericColumn::with_absent_rows(rows));
            self.numeric_present.push(tag);
        }
        self.numeric[index].as_mut().unwrap()
    }

    /// Returns the ragged backend for `tag`, allocating a default (and recording
    /// the tag as present) on first use.
    fn ragged_backend_or_insert(&mut self, tag: MetricTag) -> &mut B {
        let index = tag.index();
        if self.ragged[index].is_none() {
            self.ragged[index] = Some(B::default());
            self.ragged_present.push(tag);
        }
        self.ragged[index].as_mut().unwrap()
    }

    /// Like [`Self::ragged_backend_or_insert`] but sizes a freshly created
    /// backend to the current row span — the live-ingest set path, where a new
    /// column must already cover the rows inserted before it first appeared.
    fn ragged_backend_prepared(&mut self, tag: MetricTag, rows: usize) -> &mut B {
        let index = tag.index();
        if self.ragged[index].is_none() {
            let mut backend = B::default();
            backend.prepare_rows(rows);
            self.ragged[index] = Some(backend);
            self.ragged_present.push(tag);
        }
        self.ragged[index].as_mut().unwrap()
    }

    /// Returns the absolute slot span, including any not-yet-populated holes.
    pub fn row_count(&self) -> usize {
        self.start_ns.len()
    }

    /// Returns the number of populated request slots.
    pub fn record_count(&self) -> usize {
        self.occupied_count
    }

    /// Returns the monotonic total of every record ever ingested. Unlike
    /// [`Self::record_count`] this survives [`Self::clear_rows`] (sketch mode's
    /// per-record fold-and-clear) and is summed across [`Self::append_store`], so it is
    /// the true record total even when the store retains no rows. Equals
    /// [`Self::record_count`] for exact/exact-fold stores.
    pub fn ingested_count(&self) -> u64 {
        self.ingested_total
    }

    /// Returns true when no request slot has been populated.
    pub fn is_empty(&self) -> bool {
        self.occupied_count == 0
    }

    /// Returns request start timestamps as f64 nanoseconds.
    pub fn start_ns(&self) -> &[f64] {
        &self.start_ns
    }

    /// Returns request end timestamps as f64 nanoseconds.
    pub fn end_ns(&self) -> &[f64] {
        &self.end_ns
    }

    /// Returns first-token timestamps, with NaN for absent values.
    pub fn generation_start_ns(&self) -> &[f64] {
        &self.generation_start_ns
    }

    /// Returns locally observed OSL before endpoint-usage reconciliation.
    ///
    /// This private measurement plane preserves the client/server discrepancy
    /// diagnostic after the public OSL column becomes authoritative server
    /// usage. It is never used to fabricate token-arrival timestamps.
    pub fn observed_output_sequence_length(&self, row: usize) -> Option<f64> {
        self.observed_output_sequence_length
            .get(row)
            .copied()
            .filter(|value| value.is_finite())
    }

    /// Returns session sequence numbers by row.
    pub fn session_nums(&self) -> &[u64] {
        &self.session_nums
    }

    /// Returns zero-based turn indices by row.
    pub fn turn_indices(&self) -> &[u32] {
        &self.turn_indices
    }

    /// Returns error markers by row.
    pub fn errored(&self) -> &[bool] {
        &self.errored
    }

    /// Returns cancellation markers by row.
    pub fn canceled(&self) -> &[bool] {
        &self.canceled
    }

    /// Returns a numeric metric column.
    pub fn numeric_column(&self, tag: MetricTag) -> Option<&NumericColumn> {
        self.numeric[tag.index()].as_ref()
    }

    /// Returns one present numeric value by row and tag.
    pub fn metric_f64(&self, row: usize, tag: MetricTag) -> Option<f64> {
        self.numeric_column(tag)?.get(row)
    }

    /// Returns a mutable numeric metric column.
    pub fn numeric_column_mut(&mut self, tag: MetricTag) -> Option<&mut NumericColumn> {
        self.numeric[tag.index()].as_mut()
    }

    /// Returns a list-valued metric backend.
    pub fn ragged_column(&self, tag: MetricTag) -> Option<&B> {
        self.ragged[tag.index()].as_ref()
    }

    /// Returns all numeric metric tags currently allocated.
    pub fn numeric_tags(&self) -> impl Iterator<Item = MetricTag> + '_ {
        self.numeric_present.iter().copied()
    }

    /// Returns the dense code for a phase when it has appeared.
    pub fn phase_code(&self, phase: Phase) -> Option<u32> {
        self.phases.code(&phase)
    }

    /// Returns the dense code for a correlation id when it has appeared.
    pub fn correlation_code(&self, correlation_id: &str) -> Option<u32> {
        self.correlations.code(correlation_id)
    }

    /// Returns correlation ids in first-appearance order.
    pub fn correlation_ids(&self) -> &[String] {
        self.correlations.values()
    }

    /// Returns model/endpoint dimension pairs in first-appearance order.
    pub fn inference_dimensions(&self) -> &[InferenceDimensions] {
        self.dimensions.values()
    }

    /// Returns worker ids in first-appearance order.
    pub fn worker_ids(&self) -> &[String] {
        self.workers.values()
    }

    /// Returns conversation ids in first-appearance order.
    pub fn conversation_ids(&self) -> &[String] {
        self.conversations.values()
    }

    /// Sets one numeric metric value on an existing row.
    ///
    /// New metric columns are backfilled with NaN so every column remains aligned.
    ///
    /// # Panics
    ///
    /// Panics when `row` does not exist.
    pub fn set_metric_f64(&mut self, row: usize, tag: MetricTag, value: f64) {
        assert!(row < self.row_count() && self.occupied[row]);
        let rows = self.row_count();
        self.numeric_column_or_insert(tag, rows).set_f64(row, value);
    }

    /// Sets one boundary-safe metric value on an existing row.
    pub fn set_metric_value(&mut self, row: usize, tag: MetricTag, value: MetricValue) {
        self.set_metric_f64(
            row,
            tag,
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .unwrap_or(f64::NAN),
        );
    }

    /// Appends list values for a metric on an existing row.
    pub fn set_ragged_values(&mut self, row: usize, tag: MetricTag, values: &[f64]) {
        assert!(row < self.row_count() && self.occupied[row]);
        let rows = self.row_count();
        self.ragged_backend_prepared(tag, rows)
            .add_for_record(row, values);
    }

    /// Appends generated list values for a metric on an existing row.
    pub fn set_ragged_values_iter(
        &mut self,
        row: usize,
        tag: MetricTag,
        values: impl IntoIterator<Item = f64>,
    ) {
        assert!(row < self.row_count() && self.occupied[row]);
        let rows = self.row_count();
        let mut values = values.into_iter();
        self.ragged_backend_prepared(tag, rows)
            .add_for_record_iter(row, &mut values);
    }

    /// Builds the phase-authoritative or phase-less half-open start-time mask.
    pub fn mask_for(&self, context: &ExportContext) -> Vec<bool> {
        if let Some(phase) = context.phase {
            let expected = self.phase_code(phase);
            return self
                .phase_codes
                .iter()
                .zip(&self.occupied)
                .map(|(code, occupied)| *occupied && Some(*code) == expected)
                .collect();
        }
        self.mask_started_in(context.start_ns, context.end_ns)
    }

    /// Selects records whose start timestamp is in the half-open optional bounds.
    pub fn mask_started_in(&self, start_ns: Option<i64>, end_ns: Option<i64>) -> Vec<bool> {
        self.start_ns
            .iter()
            .map(|timestamp| {
                !timestamp.is_nan()
                    && start_ns.is_none_or(|start| *timestamp >= start as f64)
                    && end_ns.is_none_or(|end| *timestamp < end as f64)
            })
            .collect()
    }

    /// Selects records overlapping inclusive `[start_ns, end_ns]` bounds.
    pub fn mask_overlaps(&self, start_ns: i64, end_ns: i64) -> Vec<bool> {
        self.start_ns
            .iter()
            .zip(&self.end_ns)
            .map(|(record_start, record_end)| {
                !record_start.is_nan()
                    && !record_end.is_nan()
                    && *record_start <= end_ns as f64
                    && *record_end >= start_ns as f64
            })
            .collect()
    }

    /// Selects all rows for a correlation id; an unknown id returns all false.
    pub fn mask_for_correlation(&self, correlation_id: &str) -> Vec<bool> {
        let expected = self.correlation_code(correlation_id);
        self.correlation_codes
            .iter()
            .zip(&self.occupied)
            .map(|(code, occupied)| *occupied && Some(*code) == expected)
            .collect()
    }

    /// Selects rows for one exact model/endpoint pair.
    pub fn mask_for_inference_dimensions(&self, dimensions: &InferenceDimensions) -> Vec<bool> {
        let expected = self.dimensions.code(dimensions);
        self.dimension_codes
            .iter()
            .zip(&self.occupied)
            .map(|(code, occupied)| *occupied && Some(*code) == expected)
            .collect()
    }

    /// Selects all rows belonging to one session number.
    pub fn mask_for_session(&self, session_num: u64) -> Vec<bool> {
        self.session_nums
            .iter()
            .zip(&self.occupied)
            .map(|(value, occupied)| *occupied && *value == session_num)
            .collect()
    }

    /// Selects all rows with one zero-based turn index.
    pub fn mask_for_turn(&self, turn_index: u32) -> Vec<bool> {
        self.turn_indices
            .iter()
            .zip(&self.occupied)
            .map(|(value, occupied)| *occupied && *value == turn_index)
            .collect()
    }

    /// Selects all rows for a worker id; an unknown id returns all false.
    pub fn mask_for_worker(&self, worker_id: &str) -> Vec<bool> {
        let expected = self.workers.code(worker_id);
        self.worker_codes
            .iter()
            .map(|code| code.is_some() && *code == expected)
            .collect()
    }

    /// Selects all rows for a conversation id; an unknown id returns all false.
    pub fn mask_for_conversation(&self, conversation_id: &str) -> Vec<bool> {
        let expected = self.conversations.code(conversation_id);
        self.conversation_codes
            .iter()
            .map(|code| code.is_some() && *code == expected)
            .collect()
    }

    fn ensure_row_count(&mut self, rows: usize) {
        if self.row_count() >= rows {
            return;
        }
        self.occupied.resize(rows, false);
        self.start_ns.resize(rows, f64::NAN);
        self.end_ns.resize(rows, f64::NAN);
        self.generation_start_ns.resize(rows, f64::NAN);
        self.observed_output_sequence_length.resize(rows, f64::NAN);
        self.session_nums.resize(rows, 0);
        self.turn_indices.resize(rows, 0);
        self.phase_codes.resize(rows, u32::MAX);
        self.correlation_codes.resize(rows, u32::MAX);
        self.dimension_codes.resize(rows, u32::MAX);
        self.worker_codes.resize(rows, None);
        self.conversation_codes.resize(rows, None);
        self.errored.resize(rows, false);
        self.canceled.resize(rows, false);
        for column in self.numeric.iter_mut().flatten() {
            column.resize_absent(rows);
        }
    }

    fn populate_dimensions(&mut self, row: usize, record: &RecordIngest) {
        self.occupied[row] = true;
        self.start_ns[row] = record.start_ns as f64;
        self.end_ns[row] = record.end_ns as f64;
        self.generation_start_ns[row] =
            record.first_token_ns.map_or(f64::NAN, |value| value as f64);
        self.observed_output_sequence_length[row] = record
            .tokens
            .output_sequence_length()
            .map_or(f64::NAN, |value| value as f64);
        self.session_nums[row] = record.session_num;
        self.turn_indices[row] = record.turn_index;
        let phase = self.phases.intern(record.phase);
        self.phase_codes[row] = phase;
        let correlation = self.correlations.intern_ref(&record.correlation_id);
        self.correlation_codes[row] = correlation;
        let dimensions = self.dimensions.intern_ref(&record.dimensions);
        self.dimension_codes[row] = dimensions;
        self.worker_codes[row] = record
            .worker_id
            .as_ref()
            .map(|worker| self.workers.intern_ref(worker));
        self.conversation_codes[row] = record
            .conversation_id
            .as_ref()
            .map(|conversation| self.conversations.intern_ref(conversation));
        self.errored[row] = record.errored;
        self.canceled[row] = record.canceled;
    }

    fn populate_raw_metrics(
        &mut self,
        row: usize,
        record: &RecordIngest,
        token_arrivals_ns: &[i64],
    ) {
        let valid = !record.errored && !record.canceled;
        if valid {
            self.set_metric_f64(row, MetricTag::RequestCount, 1.0);
            self.set_metric_f64(row, MetricTag::MinRequestTimestamp, record.start_ns as f64);
            if record.end_ns >= record.start_ns {
                self.set_metric_f64(row, MetricTag::MaxResponseTimestamp, record.end_ns as f64);
                self.set_metric_f64(row, MetricTag::RequestLatency, record.latency_ns() as f64);
            }
            self.set_nonnegative_i64(
                row,
                MetricTag::TimeToFirstToken,
                record
                    .first_token_ns
                    .map(|timestamp| timestamp - record.start_ns),
            );
            self.set_nonnegative_i64(
                row,
                MetricTag::TimeToSecondToken,
                record
                    .second_token_ns
                    .zip(record.first_token_ns)
                    .map(|(second, first)| second - first),
            );
            self.set_nonnegative_i64(
                row,
                MetricTag::TimeToFirstOutputToken,
                record
                    .first_output_token_ns
                    .map(|timestamp| timestamp - record.start_ns),
            );
            // OSL, output-token, and reasoning counts are pure passthroughs of the
            // per-mode-reconciled `token_counts` (client tokenization by default;
            // server `usage` under `use_server_token_count`, reconciled in
            // `metrics.rs`), byte-exact with the Python `output_sequence_length` /
            // `output_token_count` / `reasoning_token_count` record metrics.
            // Endpoint `usage` stays on the record for the `usage_*` metrics and the
            // client/server discrepancy diagnostics (`observed_output_sequence_length`)
            // only; it is NOT authoritative for these visible counts.
            self.set_optional_u64(
                row,
                MetricTag::OutputSequenceLength,
                record.tokens.output_sequence_length(),
            );
            self.set_optional_u64(row, MetricTag::InputSequenceLength, record.tokens.input);
            if record.tokens.output.is_some_and(|tokens| tokens > 0) {
                self.set_optional_u64(row, MetricTag::OutputTokenCount, record.tokens.output);
            }
            self.set_optional_u64(row, MetricTag::ReasoningTokenCount, record.tokens.reasoning);
            // ICL remains an observed content-chunk metric. Endpoint usage may
            // change OSL/TPOT/throughput but never pads this timestamp vector;
            // ICL is defined by adjacent content responses.
            let arrivals = token_arrivals_ns;
            // Single pass over adjacent arrivals: collect the inter-chunk deltas
            // while verifying the timestamps are non-decreasing, bailing on the
            // first inversion. ICL is only defined for monotonic content-chunk
            // arrivals, so an out-of-order pair suppresses the metric entirely.
            let mut deltas = Vec::with_capacity(arrivals.len().saturating_sub(1));
            let mut monotonic = true;
            for pair in arrivals.windows(2) {
                if pair[1] < pair[0] {
                    monotonic = false;
                    break;
                }
                deltas.push((pair[1] - pair[0]) as f64);
            }
            if monotonic {
                self.set_ragged_values_iter(row, MetricTag::InterChunkLatency, deltas);
            }
            self.populate_usage_metrics(row, record.usage);
            self.populate_http_metrics(row, record.http);
            self.set_optional_u64(
                row,
                MetricTag::RequestedOutputSequenceLength,
                record.tokens.requested_output,
            );
            self.set_optional_f64(
                row,
                MetricTag::AudioDuration,
                record.audio_duration_s.filter(|value| *value > 0.0),
            );
            self.set_optional_u64(
                row,
                MetricTag::NumImages,
                record.num_images.filter(|value| *value > 0),
            );
            self.set_optional_f64(
                row,
                MetricTag::VideoInferenceTime,
                record.video_inference_seconds,
            );
            self.set_optional_f64(row, MetricTag::VideoPeakMemory, record.video_peak_memory_mb);
        } else {
            self.set_metric_f64(row, MetricTag::ErrorRequestCount, 1.0);
            self.set_optional_u64(
                row,
                MetricTag::ErrorInputSequenceLength,
                record.tokens.input,
            );
        }

        if let Some(credit_ns) = record.admit_ns {
            let queue_ns = (record.start_ns - credit_ns).max(0);
            let queue_ms = queue_ns as f64 / 1_000_000.0;
            let effective_ms = ((record.end_ns - credit_ns).max(0) as f64) / 1_000_000.0;
            self.set_metric_f64(row, MetricTag::CreditDropLatency, queue_ns as f64);
            self.set_metric_f64(row, MetricTag::CreditToStartLatency, queue_ms);
            self.set_metric_f64(row, MetricTag::EffectiveLatency, effective_ms);
        }
    }

    fn populate_usage_metrics(&mut self, row: usize, usage: UsageMetrics) {
        self.set_optional_u64(row, MetricTag::UsagePromptTokens, usage.prompt_tokens);
        self.set_optional_u64(
            row,
            MetricTag::UsageCompletionTokens,
            usage.completion_tokens,
        );
        self.set_optional_u64(row, MetricTag::UsageTotalTokens, usage.total_tokens);
        self.set_optional_u64(row, MetricTag::UsageReasoningTokens, usage.reasoning_tokens);
        self.set_optional_u64(
            row,
            MetricTag::UsagePromptAudioTokens,
            usage.prompt_audio_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsageCompletionAudioTokens,
            usage.completion_audio_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsageAcceptedPredictionTokens,
            usage.accepted_prediction_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsageRejectedPredictionTokens,
            usage.rejected_prediction_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsagePromptCacheReadTokens,
            usage.prompt_cache_read_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsagePromptCacheWriteTokens,
            usage.prompt_cache_write_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsagePromptCacheMissTokens,
            usage.prompt_cache_miss_tokens,
        );
        self.set_optional_u64(
            row,
            MetricTag::UsageToolUsePromptTokens,
            usage.tool_use_prompt_tokens,
        );
        self.set_optional_f64(
            row,
            MetricTag::UsagePromptAudioSeconds,
            usage.prompt_audio_seconds,
        );
    }

    fn populate_http_metrics(&mut self, row: usize, trace: RequestTrace) {
        self.set_nonnegative_i64(row, MetricTag::StreamSetupLatency, trace.stream_setup_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqBlocked, trace.blocked_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqDnsLookup, trace.dns_lookup_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqConnecting, trace.connecting_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqSending, trace.sending_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqWaiting, trace.waiting_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqReceiving, trace.receiving_ns);
        self.set_nonnegative_i64(row, MetricTag::HttpReqDuration, trace.duration_ns);
        self.set_optional_bool(
            row,
            MetricTag::HttpReqConnectionReused,
            trace.connection_reused,
        );
        self.set_optional_u64(row, MetricTag::HttpReqDataSent, trace.data_sent_bytes);
        self.set_optional_u64(
            row,
            MetricTag::HttpReqDataReceived,
            trace.data_received_bytes,
        );
        self.set_optional_u64(row, MetricTag::HttpReqChunksSent, trace.chunks_sent);
        self.set_optional_u64(row, MetricTag::HttpReqChunksReceived, trace.chunks_received);
    }

    fn set_optional_f64(&mut self, row: usize, tag: MetricTag, value: Option<f64>) {
        if let Some(value) = value.filter(|value| value.is_finite()) {
            self.set_metric_f64(row, tag, value);
        }
    }

    fn set_optional_i64(&mut self, row: usize, tag: MetricTag, value: Option<i64>) {
        self.set_optional_f64(row, tag, value.map(|value| value as f64));
    }

    fn set_nonnegative_i64(&mut self, row: usize, tag: MetricTag, value: Option<i64>) {
        self.set_optional_i64(row, tag, value.filter(|value| *value >= 0));
    }

    fn set_optional_u64(&mut self, row: usize, tag: MetricTag, value: Option<u64>) {
        self.set_optional_f64(row, tag, value.map(|value| value as f64));
    }

    fn set_optional_bool(&mut self, row: usize, tag: MetricTag, value: Option<bool>) {
        self.set_optional_f64(row, tag, value.map(f64::from));
    }
}

impl ColumnStore<RaggedSeries> {
    /// Builds an empty column store using exact ragged list storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns exact ICL replay arrays when the metric has been ingested.
    pub fn inter_chunk_latency_replay(&self) -> Option<RaggedReplay<'_>> {
        self.ragged_column(MetricTag::InterChunkLatency)?.replay()
    }
}

fn raw_metric_value(value: MetricValue) -> f64 {
    match value {
        MetricValue::Finite(value) => value,
        MetricValue::PosInf => f64::INFINITY,
        MetricValue::Absent => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::catalog::CATALOG;

    #[test]
    fn numeric_column_preserves_running_sum_in_row_order() {
        let mut column = NumericColumn::new();
        column.push_f64(1.0);
        column.push_absent();
        column.push_f64(3.5);
        assert_eq!(column.running_sum(), 4.5);
        assert_eq!(column.present_count(), 2);
        assert_eq!(column.get(1), None);
    }

    #[test]
    fn replacing_a_value_updates_running_aggregates() {
        let mut column = NumericColumn::new();
        column.push_f64(1.0);
        column.push_absent();
        column.set_f64(0, 4.0);
        column.set_f64(1, 2.0);
        assert_eq!(column.running_sum(), 6.0);
        assert_eq!(column.present_count(), 2);
    }

    #[test]
    fn ragged_series_resets_grouped_cumsum_at_record_boundaries() {
        let mut series = RaggedSeries::new();
        series.add_for_record(0, &[1.0, 2.0, 3.0]);
        series.add_for_record(1, &[10.0, 20.0]);
        assert_eq!(series.grouped_cumsum(), vec![1.0, 3.0, 6.0, 10.0, 30.0]);
        assert_eq!(series.values_for_record(1), Some(&[10.0, 20.0][..]));
    }

    #[test]
    fn ragged_series_preserves_out_of_order_rows_masks_and_shifted_merge() {
        let mut series = RaggedSeries::new();
        series.prepare_rows(3);
        series.add_for_record(2, &[20.0, 21.0]);
        series.add_for_record(0, &[1.0]);

        assert_eq!(series.values(), &[20.0, 21.0, 1.0]);
        assert_eq!(series.values_for_record(0), Some(&[1.0][..]));
        assert_eq!(series.values_for_record(1), None);
        assert_eq!(series.values_for_record(2), Some(&[20.0, 21.0][..]));
        assert_eq!(series.record_indices().collect::<Vec<_>>(), vec![2, 2, 0]);
        assert_eq!(
            series.values_for_mask(&[true, false, true]),
            vec![20.0, 21.0, 1.0]
        );
        let replay = series.replay().unwrap();
        assert_eq!(replay.offsets, &[2, 0, 0]);
        assert_eq!(replay.lengths, &[1, 0, 2]);
        assert_eq!(replay.append_order, &[2, 0]);

        let mut merged = RaggedSeries::new();
        merged.add_for_record(0, &[7.0]);
        merged.append_shifted(&series, 1, 3);
        assert_eq!(merged.values_for_record(1), Some(&[1.0][..]));
        assert_eq!(merged.values_for_record(2), None);
        assert_eq!(merged.values_for_record(3), Some(&[20.0, 21.0][..]));
        assert_eq!(
            merged.record_indices().collect::<Vec<_>>(),
            vec![0, 1, 3, 3]
        );
    }

    #[test]
    fn categorical_codes_follow_first_appearance() {
        let mut interner = CategoryInterner::default();
        assert_eq!(interner.intern("b".to_string()), 0);
        assert_eq!(interner.intern("a".to_string()), 1);
        assert_eq!(interner.intern("b".to_string()), 0);
        assert_eq!(interner.values(), &["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn borrowed_interning_clones_new_values_once_and_existing_values_never() {
        #[derive(Debug)]
        struct CloneCounted {
            identity: &'static str,
            clones: std::rc::Rc<std::cell::Cell<usize>>,
        }

        impl Clone for CloneCounted {
            fn clone(&self) -> Self {
                self.clones.set(self.clones.get() + 1);
                Self {
                    identity: self.identity,
                    clones: std::rc::Rc::clone(&self.clones),
                }
            }
        }

        impl PartialEq for CloneCounted {
            fn eq(&self, other: &Self) -> bool {
                self.identity == other.identity
            }
        }

        impl Eq for CloneCounted {}

        impl Hash for CloneCounted {
            fn hash<H: Hasher>(&self, state: &mut H) {
                0_u8.hash(state);
            }
        }

        let existing_clones = std::rc::Rc::new(std::cell::Cell::new(0));
        let existing = CloneCounted {
            identity: "existing",
            clones: std::rc::Rc::clone(&existing_clones),
        };
        let new_clones = std::rc::Rc::new(std::cell::Cell::new(0));
        let new = CloneCounted {
            identity: "new",
            clones: std::rc::Rc::clone(&new_clones),
        };
        let mut interner = CategoryInterner::default();

        assert_eq!(interner.intern_ref(&existing), 0);
        assert_eq!(existing_clones.get(), 1);
        assert_eq!(interner.intern_ref(&existing), 0);
        assert_eq!(existing_clones.get(), 1);
        assert_eq!(interner.intern_ref(&new), 1);
        assert_eq!(new_clones.get(), 1);
        assert_eq!(interner.intern_ref(&existing), 0);
        assert_eq!(interner.intern_ref(&new), 1);
        assert_eq!((existing_clones.get(), new_clones.get()), (1, 1));
    }

    #[test]
    fn record_metadata_dimensions_are_row_aligned_and_queryable() {
        let mut store = ColumnStore::new();
        let mut first = RecordIngest::minimal(10, 30, Phase::Profiling);
        first.session_num = 7;
        first.turn_index = 0;
        first.worker_id = Some("worker-b".to_string());
        first.conversation_id = Some("conversation-1".to_string());
        first.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-b/v1/chat/completions".to_string()),
            model: Some("model-b".to_string()),
        };
        let mut second = RecordIngest::minimal(40, 60, Phase::Profiling);
        second.session_num = 7;
        second.turn_index = 1;
        second.worker_id = Some("worker-a".to_string());
        second.conversation_id = Some("conversation-1".to_string());
        let third = RecordIngest::minimal(70, 90, Phase::Warmup);
        store.push_record(&first);
        store.push_record(&second);
        store.push_record(&third);

        assert_eq!(store.session_nums(), &[7, 7, 0]);
        assert_eq!(store.turn_indices(), &[0, 1, 0]);
        assert_eq!(store.worker_ids(), &["worker-b", "worker-a"]);
        assert_eq!(store.conversation_ids(), &["conversation-1"]);
        assert_eq!(store.inference_dimensions()[0], first.dimensions);
        assert_eq!(
            store.mask_for_inference_dimensions(&first.dimensions),
            vec![true, false, false]
        );
        assert_eq!(store.mask_for_session(7), vec![true, true, false]);
        assert_eq!(store.mask_for_turn(1), vec![false, true, false]);
        assert_eq!(store.mask_for_worker("worker-a"), vec![false, true, false]);
        assert_eq!(
            store.mask_for_conversation("conversation-1"),
            vec![true, true, false]
        );
        assert_eq!(store.mask_for_worker("missing"), vec![false, false, false]);
    }

    #[test]
    fn absolute_request_indices_select_column_slots_without_append_order() {
        let mut store = ColumnStore::new();
        let late = RecordIngest::minimal(30, 40, Phase::Profiling);
        let early = RecordIngest::minimal(10, 20, Phase::Profiling);
        store.insert_record_at(2, &late);
        store.insert_record_at(0, &early);

        assert_eq!(store.row_count(), 3);
        assert_eq!(store.record_count(), 2);
        assert_eq!(store.start_ns()[0], 10.0);
        assert!(store.start_ns()[1].is_nan());
        assert_eq!(store.start_ns()[2], 30.0);
        assert_eq!(
            store.mask_for(&ExportContext::phase(Phase::Profiling)),
            vec![true, false, true]
        );
    }

    #[test]
    fn records_are_append_only_and_columns_stay_aligned() {
        let mut store = ColumnStore::new();
        let first = store.push_record(&RecordIngest::minimal(10, 30, Phase::Warmup));
        let second = store.push_record(&RecordIngest::minimal(40, 60, Phase::Profiling));
        assert_eq!((first, second), (0, 1));
        assert!(store.numeric_tags().all(|tag| {
            store
                .numeric_column(tag)
                .is_some_and(|column| column.len() == 2)
        }));
    }

    #[test]
    fn worker_stores_merge_with_numeric_categorical_and_ragged_alignment() {
        let mut left = ColumnStore::new();
        let mut left_record = RecordIngest::minimal(10, 30, Phase::Warmup);
        left_record.worker_id = Some("worker-0".to_string());
        left_record.token_arrival_ns = vec![20, 25];
        left.push_record(&left_record);

        let mut right = ColumnStore::new();
        let mut right_record = RecordIngest::minimal(40, 70, Phase::Profiling);
        right_record.worker_id = Some("worker-1".to_string());
        right_record.dimensions = InferenceDimensions {
            endpoint_url: Some("https://endpoint-a/v1/chat/completions".to_string()),
            model: Some("model-a".to_string()),
        };
        right_record.turn_index = 2;
        right_record.token_arrival_ns = vec![50, 55, 63];
        right.push_record(&right_record);

        left.append_store(&right);
        assert_eq!(left.row_count(), 2);
        assert_eq!(left.mask_for_worker("worker-1"), vec![false, true]);
        assert_eq!(
            left.mask_for_inference_dimensions(&right_record.dimensions),
            vec![false, true]
        );
        assert_eq!(left.turn_indices(), &[0, 2]);
        assert!(left.numeric_tags().all(|tag| {
            left.numeric_column(tag)
                .is_some_and(|column| column.len() == 2)
        }));
        let replay = left.inter_chunk_latency_replay().unwrap();
        assert_eq!(replay.values, &[5.0, 5.0, 8.0]);
        assert_eq!(replay.record_indices().collect::<Vec<_>>(), vec![0, 1, 1]);
        assert_eq!(replay.offsets, &[0, 1]);
    }

    #[test]
    fn clean_and_error_counters_preserve_zero_error_absence() {
        let mut store = ColumnStore::new();
        store.push_record(&RecordIngest::minimal(10, 30, Phase::Profiling));
        assert_eq!(
            store
                .numeric_column(MetricTag::RequestCount)
                .and_then(|column| column.get(0)),
            Some(1.0)
        );
        assert!(store.numeric_column(MetricTag::ErrorRequestCount).is_none());

        let mut failed = RecordIngest::minimal(40, 60, Phase::Profiling);
        failed.errored = true;
        store.push_record(&failed);
        assert_eq!(
            store
                .numeric_column(MetricTag::ErrorRequestCount)
                .and_then(|column| column.get(1)),
            Some(1.0)
        );
    }

    #[test]
    fn phase_mask_is_authoritative_over_time_bounds() {
        let mut store = ColumnStore::new();
        store.push_record(&RecordIngest::minimal(100, 200, Phase::Warmup));
        let context = ExportContext {
            start_ns: Some(500),
            end_ns: Some(600),
            phase: Some(Phase::Warmup),
        };
        assert_eq!(store.mask_for(&context), vec![true]);
    }

    #[test]
    fn started_in_and_overlap_masks_have_distinct_boundaries() {
        let mut store = ColumnStore::new();
        store.push_record(&RecordIngest::minimal(100, 250, Phase::Warmup));
        store.push_record(&RecordIngest::minimal(200, 300, Phase::Profiling));
        assert_eq!(
            store.mask_started_in(Some(100), Some(200)),
            vec![true, false]
        );
        assert_eq!(store.mask_overlaps(200, 210), vec![true, true]);
    }

    #[test]
    fn ttst_is_second_minus_first_not_second_minus_start() {
        let mut store = ColumnStore::new();
        let mut record = RecordIngest::minimal(10, 50, Phase::Profiling);
        record.first_token_ns = Some(20);
        record.second_token_ns = Some(27);
        store.push_record(&record);
        assert_eq!(
            store
                .numeric_column(MetricTag::TimeToSecondToken)
                .and_then(|column| column.get(0)),
            Some(7.0)
        );
    }

    #[test]
    fn inter_chunk_latency_is_retained_for_exact_replay() {
        let mut store = ColumnStore::new();
        let mut record = RecordIngest::minimal(10, 50, Phase::Profiling);
        record.token_arrival_ns = vec![20, 25, 33];
        store.push_record(&record);
        let replay = store.inter_chunk_latency_replay().unwrap();
        assert_eq!(replay.values, &[5.0, 8.0]);
        assert_eq!(replay.record_indices().collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(replay.offsets, &[0]);
    }

    #[test]
    fn explicit_override_replaces_builtin_without_corrupting_sum() {
        let mut store = ColumnStore::new();
        let mut record = RecordIngest::minimal(10, 30, Phase::Profiling);
        record
            .metric_overrides
            .push((MetricTag::RequestLatency, MetricValue::Finite(123.0)));
        store.push_record(&record);
        let column = store.numeric_column(MetricTag::RequestLatency).unwrap();
        assert_eq!(column.get(0), Some(123.0));
        assert_eq!(column.running_sum(), 123.0);
        assert_eq!(column.present_count(), 1);
    }

    #[test]
    fn non_finite_ingest_values_are_absent_from_metric_accessors() {
        let mut store = ColumnStore::new();
        let mut record = RecordIngest::minimal(10, 30, Phase::Profiling);
        record.usage.prompt_audio_seconds = Some(f64::NAN);
        record
            .metric_overrides
            .push((MetricTag::AudioDuration, MetricValue::PosInf));
        store.push_record(&record);
        assert!(
            store
                .metric_f64(0, MetricTag::UsagePromptAudioSeconds)
                .is_none()
        );
        assert!(store.metric_f64(0, MetricTag::AudioDuration).is_none());
    }

    #[test]
    fn catalog_rows_are_not_eagerly_allocated() {
        let store = ColumnStore::<RaggedSeries>::new();
        assert_eq!(store.numeric_tags().count(), 0);
        assert!(!CATALOG.is_empty());
    }
}
