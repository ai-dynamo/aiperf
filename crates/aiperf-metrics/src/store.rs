// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Append-only columnar storage for inference metric records.
//!
//! Numeric columns keep index alignment with NaN absence sentinels and O(1)
//! running sum/count side channels. Metadata dimensions are stored separately and
//! categorical values receive dense first-appearance codes. The sparse-column
//! and query semantics port `src/aiperf/metrics/column_store.py:59-503`; the
//! exact CSR list replay ports `src/aiperf/metrics/ragged_series.py:13-107`.

use crate::catalog::MetricTag;
use crate::ingest::{HttpTrace, RecordIngest, UsageMetrics};
use crate::value::MetricValue;
use crate::window::{ExportContext, Phase};
use rustc_hash::FxHashMap;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;

/// A NaN-sparse numeric column with stable running aggregates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumericColumn {
    values: Vec<f64>,
    running_sum: f64,
    present_count: usize,
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
            running_sum: 0.0,
            present_count: 0,
        }
    }

    /// Appends a raw value; NaN denotes absence while infinity remains present.
    pub fn push_f64(&mut self, value: f64) {
        self.values.push(value);
        if !value.is_nan() {
            self.running_sum += value;
            self.present_count += 1;
        }
    }

    /// Appends an explicit absent value.
    pub fn push_absent(&mut self) {
        self.values.push(f64::NAN);
    }

    /// Appends a boundary-safe metric value.
    pub fn push_metric_value(&mut self, value: MetricValue) {
        self.push_f64(raw_metric_value(value));
    }

    /// Replaces one row while preserving the running sum/count invariant.
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the column.
    pub fn set_f64(&mut self, row: usize, value: f64) {
        let old = self.values[row];
        if !old.is_nan() {
            self.running_sum -= old;
            self.present_count -= 1;
        }
        self.values[row] = value;
        if !value.is_nan() {
            self.running_sum += value;
            self.present_count += 1;
        }
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

    /// Returns the stable insertion-order sum of present values.
    pub fn running_sum(&self) -> f64 {
        self.running_sum
    }

    /// Returns the number of present values.
    pub fn present_count(&self) -> usize {
        self.present_count
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

    /// Returns the insertion-order sum and count for selected present rows.
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
}

/// Borrowed exact replay data exposed by a list-metric backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaggedReplay<'a> {
    /// Concatenated list values.
    pub values: &'a [f64],
    /// Owning record index for every value.
    pub record_indices: &'a [usize],
    /// First flat-value offset for every record; unused rows contain zero.
    pub offsets: &'a [usize],
}

/// Extension seam for exact or bounded-memory list-valued metric storage.
pub trait ListMetricBackend: Debug + Default {
    /// Appends one record's list values.
    fn add_for_record(&mut self, row: usize, values: &[f64]);

    /// Returns all values selected by a record mask.
    fn values_for_mask(&self, record_mask: &[bool]) -> Vec<f64>;

    /// Returns exact per-record replay data when retained by this backend.
    fn replay(&self) -> Option<RaggedReplay<'_>>;

    /// Appends another backend after shifting its record indices by `row_offset`.
    /// `other_rows` includes rows with no list value so later appends stay aligned.
    fn append_shifted(&mut self, other: &Self, row_offset: usize, other_rows: usize);
}

/// Exact CSR-style list backend used for ICL distributions and sweep replay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RaggedSeries {
    values: Vec<f64>,
    record_indices: Vec<usize>,
    offsets: Vec<usize>,
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

    /// Returns the owning record index for each flat value.
    pub fn record_indices(&self) -> &[usize] {
        &self.record_indices
    }

    /// Returns one start offset per record; absent rows contain zero.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns the values stored for one record.
    pub fn values_for_record(&self, row: usize) -> Option<&[f64]> {
        if !self.present.get(row).copied().unwrap_or(false) {
            return None;
        }
        let start = self.offsets[row];
        let end = self.record_indices.partition_point(|record| *record <= row);
        Some(&self.values[start..end])
    }

    /// Computes cumulative sums that reset at each record boundary.
    pub fn grouped_cumsum(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.values.len());
        let mut current_record = None;
        let mut cumulative = 0.0;
        for (&value, &record) in self.values.iter().zip(&self.record_indices) {
            if current_record != Some(record) {
                current_record = Some(record);
                cumulative = 0.0;
            }
            cumulative += value;
            result.push(cumulative);
        }
        result
    }
}

impl ListMetricBackend for RaggedSeries {
    fn add_for_record(&mut self, row: usize, values: &[f64]) {
        if self.offsets.len() <= row {
            self.offsets.resize(row + 1, 0);
            self.present.resize(row + 1, false);
        }
        assert!(!self.present[row], "a ragged row may only be appended once");
        if values.is_empty() {
            return;
        }
        self.offsets[row] = self.values.len();
        self.present[row] = true;
        self.values.extend_from_slice(values);
        self.record_indices
            .extend(std::iter::repeat_n(row, values.len()));
    }

    fn values_for_mask(&self, record_mask: &[bool]) -> Vec<f64> {
        self.values
            .iter()
            .zip(&self.record_indices)
            .filter_map(|(value, record)| {
                record_mask
                    .get(*record)
                    .copied()
                    .unwrap_or(false)
                    .then_some(*value)
            })
            .collect()
    }

    fn replay(&self) -> Option<RaggedReplay<'_>> {
        Some(RaggedReplay {
            values: &self.values,
            record_indices: &self.record_indices,
            offsets: &self.offsets,
        })
    }

    fn append_shifted(&mut self, other: &Self, row_offset: usize, other_rows: usize) {
        if self.offsets.len() < row_offset {
            self.offsets.resize(row_offset, 0);
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
            self.present.resize(row_offset + other_rows, false);
        }
    }
}

/// Dense, first-appearance categorical interner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoryInterner<T: Eq + Hash> {
    by_value: FxHashMap<T, u32>,
    values: Vec<T>,
}

impl<T: Eq + Hash> Default for CategoryInterner<T> {
    fn default() -> Self {
        Self {
            by_value: FxHashMap::default(),
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
        if let Some(code) = self.by_value.get(&value) {
            return *code;
        }
        let code = u32::try_from(self.values.len()).expect("category cardinality exceeds u32");
        self.values.push(value.clone());
        self.by_value.insert(value, code);
        code
    }

    /// Looks up a previously interned value.
    pub fn code<Q>(&self, value: &Q) -> Option<u32>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.by_value.get(value).copied()
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
}

/// Append-only, row-aligned metric and metadata columns.
#[derive(Debug)]
pub struct ColumnStore<B: ListMetricBackend = RaggedSeries> {
    start_ns: Vec<f64>,
    end_ns: Vec<f64>,
    generation_start_ns: Vec<f64>,
    session_nums: Vec<u64>,
    turn_indices: Vec<u32>,
    phase_codes: Vec<u32>,
    correlation_codes: Vec<u32>,
    worker_codes: Vec<Option<u32>>,
    conversation_codes: Vec<Option<u32>>,
    errored: Vec<bool>,
    canceled: Vec<bool>,
    phases: CategoryInterner<Phase>,
    correlations: CategoryInterner<String>,
    workers: CategoryInterner<String>,
    conversations: CategoryInterner<String>,
    numeric: FxHashMap<MetricTag, NumericColumn>,
    ragged: FxHashMap<MetricTag, B>,
}

impl<B: ListMetricBackend> Default for ColumnStore<B> {
    fn default() -> Self {
        Self {
            start_ns: Vec::new(),
            end_ns: Vec::new(),
            generation_start_ns: Vec::new(),
            session_nums: Vec::new(),
            turn_indices: Vec::new(),
            phase_codes: Vec::new(),
            correlation_codes: Vec::new(),
            worker_codes: Vec::new(),
            conversation_codes: Vec::new(),
            errored: Vec::new(),
            canceled: Vec::new(),
            phases: CategoryInterner::default(),
            correlations: CategoryInterner::default(),
            workers: CategoryInterner::default(),
            conversations: CategoryInterner::default(),
            numeric: FxHashMap::default(),
            ragged: FxHashMap::default(),
        }
    }
}

impl<B: ListMetricBackend> ColumnStore<B> {
    /// Builds an empty column store with the selected list backend.
    pub fn with_list_backend() -> Self {
        Self::default()
    }

    /// Appends one raw record, populating directly observable catalog metrics.
    pub fn push_record(&mut self, record: &RecordIngest) -> usize {
        let row = self.append_dimensions(record);
        self.populate_raw_metrics(row, record);
        for (tag, value) in &record.metric_overrides {
            self.set_metric_value(row, *tag, *value);
        }
        row
    }

    /// Appends every row and allocated metric column from another worker store.
    ///
    /// Categorical codes are re-interned in append order, numeric absence remains
    /// index-aligned, and list-valued record indices are shifted exactly once.
    pub fn append_store(&mut self, other: &Self) {
        let row_offset = self.row_count();
        let other_rows = other.row_count();
        if other_rows == 0 {
            return;
        }

        for row in 0..other_rows {
            self.start_ns.push(other.start_ns[row]);
            self.end_ns.push(other.end_ns[row]);
            self.generation_start_ns
                .push(other.generation_start_ns[row]);
            self.session_nums.push(other.session_nums[row]);
            self.turn_indices.push(other.turn_indices[row]);

            let phase = other
                .phases
                .value(other.phase_codes[row])
                .copied()
                .expect("phase codes must resolve");
            self.phase_codes.push(self.phases.intern(phase));
            let correlation = other
                .correlations
                .value(other.correlation_codes[row])
                .cloned()
                .expect("correlation codes must resolve");
            self.correlation_codes
                .push(self.correlations.intern(correlation));
            self.worker_codes.push(other.worker_codes[row].map(|code| {
                let worker = other
                    .workers
                    .value(code)
                    .cloned()
                    .expect("worker codes must resolve");
                self.workers.intern(worker)
            }));
            self.conversation_codes
                .push(other.conversation_codes[row].map(|code| {
                    let conversation = other
                        .conversations
                        .value(code)
                        .cloned()
                        .expect("conversation codes must resolve");
                    self.conversations.intern(conversation)
                }));
            self.errored.push(other.errored[row]);
            self.canceled.push(other.canceled[row]);
        }

        for column in self.numeric.values_mut() {
            for _ in 0..other_rows {
                column.push_absent();
            }
        }
        for (tag, other_column) in &other.numeric {
            let column = self
                .numeric
                .entry(*tag)
                .or_insert_with(|| NumericColumn::with_absent_rows(row_offset + other_rows));
            for (row, value) in other_column.values().iter().copied().enumerate() {
                if !value.is_nan() {
                    column.set_f64(row_offset + row, value);
                }
            }
        }
        for (tag, other_backend) in &other.ragged {
            self.ragged.entry(*tag).or_default().append_shifted(
                other_backend,
                row_offset,
                other_rows,
            );
        }
    }

    /// Returns the number of append-only rows.
    pub fn row_count(&self) -> usize {
        self.start_ns.len()
    }

    /// Returns true when no record has been appended.
    pub fn is_empty(&self) -> bool {
        self.start_ns.is_empty()
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
        self.numeric.get(&tag)
    }

    /// Returns one present numeric value by row and tag.
    pub fn metric_f64(&self, row: usize, tag: MetricTag) -> Option<f64> {
        self.numeric_column(tag)?.get(row)
    }

    /// Returns a mutable numeric metric column.
    pub fn numeric_column_mut(&mut self, tag: MetricTag) -> Option<&mut NumericColumn> {
        self.numeric.get_mut(&tag)
    }

    /// Returns a list-valued metric backend.
    pub fn ragged_column(&self, tag: MetricTag) -> Option<&B> {
        self.ragged.get(&tag)
    }

    /// Returns all numeric metric tags currently allocated.
    pub fn numeric_tags(&self) -> impl Iterator<Item = MetricTag> + '_ {
        self.numeric.keys().copied()
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
        assert!(row < self.row_count());
        let rows = self.row_count();
        self.numeric
            .entry(tag)
            .or_insert_with(|| NumericColumn::with_absent_rows(rows))
            .set_f64(row, value);
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
        assert!(row < self.row_count());
        self.ragged
            .entry(tag)
            .or_default()
            .add_for_record(row, values);
    }

    /// Builds the phase-authoritative or phase-less half-open start-time mask.
    pub fn mask_for(&self, context: &ExportContext) -> Vec<bool> {
        if let Some(phase) = context.phase {
            let expected = self.phase_code(phase);
            return self
                .phase_codes
                .iter()
                .map(|code| Some(*code) == expected)
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
            .map(|code| Some(*code) == expected)
            .collect()
    }

    /// Selects all rows belonging to one session number.
    pub fn mask_for_session(&self, session_num: u64) -> Vec<bool> {
        self.session_nums
            .iter()
            .map(|value| *value == session_num)
            .collect()
    }

    /// Selects all rows with one zero-based turn index.
    pub fn mask_for_turn(&self, turn_index: u32) -> Vec<bool> {
        self.turn_indices
            .iter()
            .map(|value| *value == turn_index)
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

    fn append_dimensions(&mut self, record: &RecordIngest) -> usize {
        let row = self.row_count();
        self.start_ns.push(record.start_ns as f64);
        self.end_ns.push(record.end_ns as f64);
        self.generation_start_ns
            .push(record.first_token_ns.map_or(f64::NAN, |value| value as f64));
        self.session_nums.push(record.session_num);
        self.turn_indices.push(record.turn_index);
        let phase = self.phases.intern(record.phase);
        self.phase_codes.push(phase);
        let correlation = self.correlations.intern(record.correlation_id.clone());
        self.correlation_codes.push(correlation);
        self.worker_codes.push(
            record
                .worker_id
                .as_ref()
                .map(|worker| self.workers.intern(worker.clone())),
        );
        self.conversation_codes.push(
            record
                .conversation_id
                .as_ref()
                .map(|conversation| self.conversations.intern(conversation.clone())),
        );
        self.errored.push(record.errored);
        self.canceled.push(record.canceled);
        for column in self.numeric.values_mut() {
            column.push_absent();
        }
        row
    }

    fn populate_raw_metrics(&mut self, row: usize, record: &RecordIngest) {
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
            let icl = record
                .inter_chunk_latencies_ns()
                .into_iter()
                .map(|value| value as f64)
                .collect::<Vec<_>>();
            if icl.iter().all(|value| *value >= 0.0) {
                self.set_ragged_values(row, MetricTag::InterChunkLatency, &icl);
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

    fn populate_http_metrics(&mut self, row: usize, trace: HttpTrace) {
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
    use crate::catalog::CATALOG;

    #[test]
    fn numeric_column_preserves_running_sum_in_insertion_order() {
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
    fn categorical_codes_follow_first_appearance() {
        let mut interner = CategoryInterner::default();
        assert_eq!(interner.intern("b".to_string()), 0);
        assert_eq!(interner.intern("a".to_string()), 1);
        assert_eq!(interner.intern("b".to_string()), 0);
        assert_eq!(interner.values(), &["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn record_metadata_dimensions_are_row_aligned_and_queryable() {
        let mut store = ColumnStore::new();
        let mut first = RecordIngest::minimal(10, 30, Phase::Profiling);
        first.session_num = 7;
        first.turn_index = 0;
        first.worker_id = Some("worker-b".to_string());
        first.conversation_id = Some("conversation-1".to_string());
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
        right_record.turn_index = 2;
        right_record.token_arrival_ns = vec![50, 55, 63];
        right.push_record(&right_record);

        left.append_store(&right);
        assert_eq!(left.row_count(), 2);
        assert_eq!(left.mask_for_worker("worker-1"), vec![false, true]);
        assert_eq!(left.turn_indices(), &[0, 2]);
        assert!(left.numeric_tags().all(|tag| {
            left.numeric_column(tag)
                .is_some_and(|column| column.len() == 2)
        }));
        let replay = left.inter_chunk_latency_replay().unwrap();
        assert_eq!(replay.values, &[5.0, 5.0, 8.0]);
        assert_eq!(replay.record_indices, &[0, 1, 1]);
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
        assert_eq!(replay.record_indices, &[0, 0]);
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
