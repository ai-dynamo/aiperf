// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic sweep-line curves for concurrency, throughput, and KV-cache load.
//!
//! Curves are right-continuous step functions with deterministic event ordering,
//! floating-point cancellation, decode-token accounting, and active-only
//! statistics.

mod kv_cache;
mod stats;

use crate::metrics_core::{MetricConsoleGroup, MetricValue};
use rayon::prelude::*;
use std::cmp::Ordering;

pub use kv_cache::{
    IclSeries, throughput_sweep_line_icl, tokens_in_flight_sweep_line,
    tokens_in_flight_sweep_line_icl,
};
pub use stats::{
    ClippedSegment, SweepLineStats, build_clipped_segments, compute_active_weighted_stats,
    compute_divided_active_weighted_stats, compute_divided_time_weighted_stats,
    compute_divided_weighted_stats, compute_time_weighted_stats,
};

/// Nanoseconds per second, used to convert token/ns curves at the report boundary.
pub const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

const PARALLEL_SWEEP_MIN_ROWS: usize = 4_096;
const PARALLEL_EVENT_SORT_MIN_EVENTS: usize = 262_144;

/// One timestamped change applied by the sweep-line cumulative sum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepEvent {
    /// Event timestamp in nanoseconds.
    pub timestamp_ns: f64,
    /// Signed contribution added at the timestamp.
    pub delta: f64,
}

impl SweepEvent {
    /// Builds a sweep event.
    pub const fn new(timestamp_ns: f64, delta: f64) -> Self {
        Self {
            timestamp_ns,
            delta,
        }
    }
}

/// One sweep event shared by adjacent bit-identical request trajectories.
///
/// Reduction replays `delta` instead of multiplying it so compression cannot
/// change floating-point accumulation order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RepeatedSweepEvent {
    timestamp_ns: f64,
    delta: f64,
    repetitions: usize,
}

impl RepeatedSweepEvent {
    pub(super) const fn new(timestamp_ns: f64, delta: f64, repetitions: usize) -> Self {
        Self {
            timestamp_ns,
            delta,
            repetitions,
        }
    }
}

/// A right-continuous step function stored at its event boundaries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepFn {
    timestamps_ns: Vec<f64>,
    values: Vec<f64>,
}

impl StepFn {
    /// Builds a step function from sorted, index-aligned vectors.
    ///
    /// # Panics
    ///
    /// Panics when the vectors have different lengths or timestamps are not sorted.
    pub fn new(timestamps_ns: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(timestamps_ns.len(), values.len());
        assert!(
            timestamps_ns
                .windows(2)
                .all(|pair| pair[0].total_cmp(&pair[1]) != Ordering::Greater)
        );
        Self {
            timestamps_ns,
            values,
        }
    }

    /// Builds an empty step function whose value is zero everywhere.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns the sorted event timestamps.
    pub fn timestamps_ns(&self) -> &[f64] {
        &self.timestamps_ns
    }

    /// Returns the values held after the corresponding event timestamps.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the number of event boundaries.
    pub fn len(&self) -> usize {
        self.timestamps_ns.len()
    }

    /// Returns true when the curve contains no events.
    pub fn is_empty(&self) -> bool {
        self.timestamps_ns.is_empty()
    }

    /// Looks up the value held at `query_ns`, returning zero before the first event.
    ///
    /// This is the `searchsorted(..., side="right") - 1` behavior.
    pub fn value_at(&self, query_ns: f64) -> f64 {
        let upper = upper_bound(&self.timestamps_ns, query_ns);
        upper
            .checked_sub(1)
            .and_then(|index| self.values.get(index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns the final cumulative value, or zero for an empty curve.
    pub fn final_value(&self) -> f64 {
        self.values.last().copied().unwrap_or(0.0)
    }

    /// Adds this curve to another curve on their merged timestamp grid.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        self.combine_on_merged_grid(other, |left, right| left + right)
    }

    /// Divides this curve by another curve, yielding zero where the denominator is non-positive.
    pub fn divide(&self, denominator: &Self) -> Self {
        if self.is_empty() || denominator.is_empty() {
            return Self::empty();
        }
        self.combine_on_merged_grid(denominator, |numerator, denominator| {
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
    }

    fn combine_on_merged_grid(
        &self,
        other: &Self,
        mut combine: impl FnMut(f64, f64) -> f64,
    ) -> Self {
        let timestamps_ns = merged_timestamps(self, other);
        let mut left_index = 0;
        let mut right_index = 0;
        let mut left_value = 0.0;
        let mut right_value = 0.0;
        let mut values = Vec::with_capacity(timestamps_ns.len());
        for &timestamp in &timestamps_ns {
            while left_index < self.len()
                && self.timestamps_ns[left_index].total_cmp(&timestamp) != Ordering::Greater
            {
                left_value = self.values[left_index];
                left_index += 1;
            }
            while right_index < other.len()
                && other.timestamps_ns[right_index].total_cmp(&timestamp) != Ordering::Greater
            {
                right_value = other.values[right_index];
                right_index += 1;
            }
            values.push(combine(left_value, right_value));
        }
        Self::new(timestamps_ns, values)
    }
}

/// Sorts events, applies the end-before-start tie-break, and cumulatively sums deltas.
///
/// Events with non-positive deltas sort before positive deltas at equal timestamps.
/// Residuals below `1e-9 * max_abs` are snapped to zero after the sum.
pub fn sweep_line_cumsum(mut events: Vec<SweepEvent>) -> StepFn {
    sort_sweep_events(&mut events);

    let mut timestamps_ns = Vec::with_capacity(events.len());
    let mut values = Vec::with_capacity(events.len());
    let mut current = 0.0;
    for event in events {
        current += event.delta;
        timestamps_ns.push(event.timestamp_ns);
        values.push(current);
    }

    let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if max_abs > 0.0 {
        snap_small_residuals(&mut values, 1e-9 * max_abs);
    }
    StepFn::new(timestamps_ns, values)
}

fn sweep_line_cumsum_compact(mut events: Vec<SweepEvent>) -> StepFn {
    sort_sweep_events(&mut events);
    if events.is_empty() {
        return StepFn::empty();
    }

    let mut unique_timestamps = 1_usize;
    let mut max_abs = 0.0_f64;
    let mut current = 0.0;
    for (index, event) in events.iter().enumerate() {
        current += event.delta;
        max_abs = max_abs.max(current.abs());
        if index > 0 && !same_timestamp(events[index - 1].timestamp_ns, event.timestamp_ns) {
            unique_timestamps += 1;
        }
    }

    let mut timestamps_ns = Vec::with_capacity(unique_timestamps);
    let mut values = Vec::with_capacity(unique_timestamps);
    current = 0.0;
    for (index, event) in events.iter().enumerate() {
        current += event.delta;
        let finishes_timestamp = events
            .get(index + 1)
            .is_none_or(|next| !same_timestamp(event.timestamp_ns, next.timestamp_ns));
        if finishes_timestamp {
            timestamps_ns.push(event.timestamp_ns);
            values.push(current);
        }
    }
    if max_abs > 0.0 {
        snap_small_residuals(&mut values, 1e-9 * max_abs);
    }
    StepFn::new(timestamps_ns, values)
}

pub(super) fn sweep_line_cumsum_repeated(mut events: Vec<RepeatedSweepEvent>) -> StepFn {
    sort_repeated_sweep_events(&mut events);
    if events.is_empty() {
        return StepFn::empty();
    }

    let mut unique_timestamps = 1_usize;
    let mut max_abs = 0.0_f64;
    let mut current = 0.0;
    for (index, event) in events.iter().enumerate() {
        for _ in 0..event.repetitions {
            current += event.delta;
            max_abs = max_abs.max(current.abs());
        }
        if index > 0 && !same_timestamp(events[index - 1].timestamp_ns, event.timestamp_ns) {
            unique_timestamps += 1;
        }
    }

    let mut timestamps_ns = Vec::with_capacity(unique_timestamps);
    let mut values = Vec::with_capacity(unique_timestamps);
    current = 0.0;
    for (index, event) in events.iter().enumerate() {
        for _ in 0..event.repetitions {
            current += event.delta;
        }
        let finishes_timestamp = events
            .get(index + 1)
            .is_none_or(|next| !same_timestamp(event.timestamp_ns, next.timestamp_ns));
        if finishes_timestamp {
            timestamps_ns.push(event.timestamp_ns);
            values.push(current);
        }
    }
    if max_abs > 0.0 {
        snap_small_residuals(&mut values, 1e-9 * max_abs);
    }
    StepFn::new(timestamps_ns, values)
}

// Sorting by `(timestamp asc, delta asc)` places end deltas before start deltas
// at equal timestamps. A stable timestamp radix plus an in-run delta tie-break
// preserves that ordering without comparison-sorting the full event set.

fn sort_sweep_events(events: &mut [SweepEvent]) {
    sort_by_timestamp_then_delta(events, |event| event.timestamp_ns, |event| event.delta);
}

fn sort_repeated_sweep_events(events: &mut [RepeatedSweepEvent]) {
    sort_by_timestamp_then_delta(events, |event| event.timestamp_ns, |event| event.delta);
}

/// Maps an `f64` to the `u64` whose unsigned order is IEEE total order — i.e.
/// `radix_key(a) <= radix_key(b)` iff `a.total_cmp(&b) != Greater`.
#[inline(always)]
fn radix_key(x: f64) -> u64 {
    let bits = x.to_bits();
    bits ^ ((((bits as i64) >> 63) as u64) | (1u64 << 63))
}

/// Sorts `items` by `(timestamp, delta)` total order in place, byte-for-byte
/// identical to sorting by `(timestamp, delta)` with `total_cmp`.
///
/// A stable LSD radix on `radix_key(timestamp)` orders by timestamp (ties keep
/// input order); each equal-timestamp run is then sorted by `delta` so the
/// cumulative sum accumulates in the same order — and therefore to the same
/// `f64` bits — as the comparison sort. Parallel above the event threshold.
fn sort_by_timestamp_then_delta<T, Ts, Delta>(items: &mut [T], timestamp: Ts, delta: Delta)
where
    T: Copy + Send,
    Ts: Fn(&T) -> f64 + Sync,
    Delta: Fn(&T) -> f64,
{
    let n = items.len();
    if n < 2 {
        return;
    }
    let keys: Vec<u64> = items
        .iter()
        .map(|item| radix_key(timestamp(item)))
        .collect();
    // Single-threaded: the seven curves are already fanned out across rayon in
    // `SweepLineCurves::compute`, so an inner parallel sort only nests rayon
    // regions and multiplies epoch/work-steal coordination (which profiles as
    // the dominant export cost). `radix_argsort_mt` stays for callers that sort
    // a single curve with no outer parallelism.
    let permutation = radix_argsort_st(&keys);
    let sorted: Vec<T> = permutation
        .iter()
        .map(|&index| items[index as usize])
        .collect();
    items.copy_from_slice(&sorted);

    let mut run_start = 0usize;
    while run_start < n {
        let mut run_end = run_start + 1;
        while run_end < n
            && same_timestamp(timestamp(&items[run_start]), timestamp(&items[run_end]))
        {
            run_end += 1;
        }
        if run_end - run_start > 1 {
            items[run_start..run_end]
                .sort_unstable_by(|left, right| delta(left).total_cmp(&delta(right)));
        }
        run_start = run_end;
    }
}

/// Stable single-threaded LSD radix argsort of `keys` (ascending). Adaptive:
/// constant key bytes are skipped, so integer-nanosecond timestamps (which sit
/// in the low bytes) cost only the passes that vary.
fn radix_argsort_st(keys: &[u64]) -> Vec<u32> {
    let n = keys.len();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    if n < 2 {
        return indices;
    }
    let mut source = keys.to_vec();
    let (mut or_all, mut and_all) = (0u64, !0u64);
    for &key in &source {
        or_all |= key;
        and_all &= key;
    }
    let vary = or_all & !and_all;

    let mut key_scratch = vec![0u64; n];
    let mut index_scratch = vec![0u32; n];
    for byte in 0..8 {
        let shift = byte * 8;
        if (vary >> shift) & 0xFF == 0 {
            continue;
        }
        let mut counts = [0usize; 256];
        for &key in &source {
            counts[((key >> shift) & 0xFF) as usize] += 1;
        }
        let mut offset = 0usize;
        for count in counts.iter_mut() {
            let bucket = *count;
            *count = offset;
            offset += bucket;
        }
        for i in 0..n {
            let bucket = ((source[i] >> shift) & 0xFF) as usize;
            let position = counts[bucket];
            counts[bucket] += 1;
            key_scratch[position] = source[i];
            index_scratch[position] = indices[i];
        }
        std::mem::swap(&mut source, &mut key_scratch);
        std::mem::swap(&mut indices, &mut index_scratch);
    }
    indices
}

/// Stable parallel LSD radix argsort: per-chunk histograms, exclusive per-chunk
/// bucket offsets, then a contention-free scatter into disjoint output ranges.
///
/// The sweep-curve bundle does not call this path because it already fans out
/// across curves and must avoid nested Rayon regions.
#[allow(dead_code)]
fn radix_argsort_mt(keys: &[u64]) -> Vec<u32> {
    let n = keys.len();
    let mut source = keys.to_vec();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let (or_all, and_all) = source
        .par_iter()
        .fold(|| (0u64, !0u64), |(o, a), &k| (o | k, a & k))
        .reduce(|| (0u64, !0u64), |(o1, a1), (o2, a2)| (o1 | o2, a1 & a2));
    let vary = or_all & !and_all;

    let mut key_scratch = vec![0u64; n];
    let mut index_scratch = vec![0u32; n];
    let threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(threads);

    for byte in 0..8 {
        let shift = byte * 8;
        if (vary >> shift) & 0xFF == 0 {
            continue;
        }
        let histograms: Vec<[usize; 256]> = source
            .par_chunks(chunk)
            .map(|chunk_keys| {
                let mut histogram = [0usize; 256];
                for &key in chunk_keys {
                    histogram[((key >> shift) & 0xFF) as usize] += 1;
                }
                histogram
            })
            .collect();
        let chunk_count = histograms.len();

        let mut bucket_total = [0usize; 256];
        for histogram in &histograms {
            for bucket in 0..256 {
                bucket_total[bucket] += histogram[bucket];
            }
        }
        let mut running = [0usize; 256];
        let mut offset = 0usize;
        for bucket in 0..256 {
            running[bucket] = offset;
            offset += bucket_total[bucket];
        }
        let mut starts = vec![[0usize; 256]; chunk_count];
        for (c, start) in starts.iter_mut().enumerate() {
            start.copy_from_slice(&running);
            for bucket in 0..256 {
                running[bucket] += histograms[c][bucket];
            }
        }

        let key_ptr = key_scratch.as_mut_ptr() as usize;
        let index_ptr = index_scratch.as_mut_ptr() as usize;
        source
            .par_chunks(chunk)
            .zip(indices.par_chunks(chunk))
            .enumerate()
            .for_each(|(c, (chunk_keys, chunk_indices))| {
                let mut offset = starts[c];
                let key_out = key_ptr as *mut u64;
                let index_out = index_ptr as *mut u32;
                for i in 0..chunk_keys.len() {
                    let bucket = ((chunk_keys[i] >> shift) & 0xFF) as usize;
                    let position = offset[bucket];
                    offset[bucket] += 1;
                    // Disjoint output ranges per chunk+bucket: no aliasing.
                    unsafe {
                        *key_out.add(position) = chunk_keys[i];
                        *index_out.add(position) = chunk_indices[i];
                    }
                }
            });
        std::mem::swap(&mut source, &mut key_scratch);
        std::mem::swap(&mut indices, &mut index_scratch);
    }
    indices
}

fn snap_small_residuals(values: &mut [f64], threshold: f64) {
    let mut chunks = values.chunks_exact_mut(8);
    for values in chunks.by_ref() {
        values[0] = snapped_residual(values[0], threshold);
        values[1] = snapped_residual(values[1], threshold);
        values[2] = snapped_residual(values[2], threshold);
        values[3] = snapped_residual(values[3], threshold);
        values[4] = snapped_residual(values[4], threshold);
        values[5] = snapped_residual(values[5], threshold);
        values[6] = snapped_residual(values[6], threshold);
        values[7] = snapped_residual(values[7], threshold);
    }
    for value in chunks.into_remainder() {
        *value = snapped_residual(*value, threshold);
    }
}

#[inline(always)]
fn snapped_residual(value: f64, threshold: f64) -> f64 {
    let magnitude = value.to_bits() & !(1_u64 << 63);
    let keep = u64::from(magnitude >= threshold.to_bits());
    f64::from_bits(value.to_bits() & 0_u64.wrapping_sub(keep))
}

/// Computes exact request concurrency from aligned start/end columns.
///
/// NaN rows are absent.
pub fn concurrency_sweep_line(start_ns: &[f64], end_ns: &[f64]) -> StepFn {
    assert_aligned(start_ns.len(), &[end_ns.len()]);
    let mut events = Vec::with_capacity(start_ns.len() * 2);
    for (&start, &end) in start_ns.iter().zip(end_ns) {
        if !start.is_nan() && !end.is_nan() {
            events.push(SweepEvent::new(start, 1.0));
            events.push(SweepEvent::new(end, -1.0));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes weighted concurrency, such as request-level tokens in flight.
///
/// NaN rows are absent.
pub fn weighted_concurrency_sweep_line(
    start_ns: &[f64],
    end_ns: &[f64],
    weights: &[f64],
) -> StepFn {
    assert_aligned(start_ns.len(), &[end_ns.len(), weights.len()]);
    let mut events = Vec::with_capacity(start_ns.len() * 2);
    for ((&start, &end), &weight) in start_ns.iter().zip(end_ns).zip(weights) {
        if !start.is_nan() && !end.is_nan() && !weight.is_nan() {
            events.push(SweepEvent::new(start, weight));
            events.push(SweepEvent::new(end, -weight));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes uniform decode throughput in tokens/ns.
///
/// Each valid request contributes `(output_tokens - 1) / generation_duration` over
/// `[generation_start, end)`. The first token is not a decode step.
pub fn throughput_sweep_line(
    generation_start_ns: &[f64],
    end_ns: &[f64],
    output_tokens: &[f64],
) -> StepFn {
    assert_aligned(
        generation_start_ns.len(),
        &[end_ns.len(), output_tokens.len()],
    );
    let mut events = Vec::with_capacity(generation_start_ns.len() * 2);
    for ((&start, &end), &tokens) in generation_start_ns.iter().zip(end_ns).zip(output_tokens) {
        let duration = end - start;
        if !start.is_nan() && !tokens.is_nan() && duration > 0.0 && tokens >= 1.0 {
            let rate = (tokens - 1.0) / duration;
            events.push(SweepEvent::new(start, rate));
            events.push(SweepEvent::new(end, -rate));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes uniform prefill throughput in tokens/ns.
///
/// Each valid request contributes `input_tokens / prefill_duration` over
/// `[start, generation_start)`, with no token subtraction.
pub fn prefill_throughput_sweep_line(
    start_ns: &[f64],
    generation_start_ns: &[f64],
    input_tokens: &[f64],
) -> StepFn {
    assert_aligned(
        start_ns.len(),
        &[generation_start_ns.len(), input_tokens.len()],
    );
    let mut events = Vec::with_capacity(start_ns.len() * 2);
    for ((&start, &generation_start), &tokens) in
        start_ns.iter().zip(generation_start_ns).zip(input_tokens)
    {
        let duration = generation_start - start;
        if !start.is_nan() && !generation_start.is_nan() && !tokens.is_nan() && duration > 0.0 {
            let rate = tokens / duration;
            events.push(SweepEvent::new(start, rate));
            events.push(SweepEvent::new(generation_start, -rate));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes combined prefill and decode throughput in one sweep pass.
///
/// Each phase applies its own validity predicate; the decode phase subtracts `1`
/// from the output-token count so the first token is not counted as a decode step.
pub fn total_throughput_sweep_line(
    start_ns: &[f64],
    generation_start_ns: &[f64],
    end_ns: &[f64],
    input_tokens: &[f64],
    output_tokens: &[f64],
) -> StepFn {
    assert_aligned(
        start_ns.len(),
        &[
            generation_start_ns.len(),
            end_ns.len(),
            input_tokens.len(),
            output_tokens.len(),
        ],
    );
    let mut events = Vec::with_capacity(start_ns.len() * 4);
    for ((((&start, &generation_start), &end), &input), &output) in start_ns
        .iter()
        .zip(generation_start_ns)
        .zip(end_ns)
        .zip(input_tokens)
        .zip(output_tokens)
    {
        let prefill_duration = generation_start - start;
        if !start.is_nan()
            && !generation_start.is_nan()
            && !input.is_nan()
            && prefill_duration > 0.0
        {
            let rate = input / prefill_duration;
            events.push(SweepEvent::new(start, rate));
            events.push(SweepEvent::new(generation_start, -rate));
        }

        let generation_duration = end - generation_start;
        if !generation_start.is_nan()
            && !output.is_nan()
            && generation_duration > 0.0
            && output >= 1.0
        {
            let rate = (output - 1.0) / generation_duration;
            events.push(SweepEvent::new(generation_start, rate));
            events.push(SweepEvent::new(end, -rate));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes uniform image-sample throughput in samples/ns.
///
/// Each request submits `num_images` samples that resolve at `end`, so the whole
/// batch is spread uniformly over `[start, end)` as `num_images / duration`.
/// Unlike decode throughput no count is subtracted: every image is a sample.
pub fn sample_throughput_sweep_line(
    start_ns: &[f64],
    end_ns: &[f64],
    num_images: &[f64],
) -> StepFn {
    assert_aligned(start_ns.len(), &[end_ns.len(), num_images.len()]);
    let mut events = Vec::with_capacity(start_ns.len() * 2);
    for ((&start, &end), &images) in start_ns.iter().zip(end_ns).zip(num_images) {
        let duration = end - start;
        if !start.is_nan() && !images.is_nan() && duration > 0.0 && images >= 1.0 {
            let rate = images / duration;
            events.push(SweepEvent::new(start, rate));
            events.push(SweepEvent::new(end, -rate));
        }
    }
    sweep_line_cumsum_compact(events)
}

/// All request-derived sweep curves, computed once and re-windowed for summaries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SweepLineCurves {
    /// Whole-request concurrency.
    pub concurrency: StepFn,
    /// Decode throughput in tokens/ns.
    pub decode_throughput: StepFn,
    /// Prefill throughput in tokens/ns.
    pub prefill_throughput: StepFn,
    /// Decode-phase concurrency.
    pub decode_concurrency: StepFn,
    /// Prefill-phase concurrency.
    pub prefill_concurrency: StepFn,
    /// Combined prefill and decode throughput in tokens/ns.
    pub total_throughput: StepFn,
    /// KV-cache tokens held across active requests.
    pub tokens_in_flight: StepFn,
    /// Image-sample throughput in samples/ns.
    pub sample_throughput: StepFn,
}

impl SweepLineCurves {
    /// Computes the full curve bundle from aligned request columns.
    ///
    /// When a non-empty ICL series is supplied, decode throughput and tokens in flight
    /// use chunk boundaries; otherwise request-level uniform curves are used.
    pub fn compute(
        start_ns: &[f64],
        generation_start_ns: &[f64],
        end_ns: &[f64],
        input_tokens: &[f64],
        output_tokens: &[f64],
        num_images: &[f64],
        icl: Option<IclSeries<'_>>,
    ) -> Self {
        assert_aligned(
            start_ns.len(),
            &[
                generation_start_ns.len(),
                end_ns.len(),
                input_tokens.len(),
                output_tokens.len(),
                num_images.len(),
            ],
        );
        // Image-sample throughput is a single cheap sweep pass independent of the
        // prefill/decode phase split, so it is computed outside the parallel bundle.
        let sample_throughput = sample_throughput_sweep_line(start_ns, end_ns, num_images);
        let compute_decode_throughput = || match icl.filter(|series| !series.is_empty()) {
            Some(series) => throughput_sweep_line_icl(generation_start_ns, output_tokens, series),
            None => throughput_sweep_line(generation_start_ns, end_ns, output_tokens),
        };
        let compute_tokens_in_flight = || match icl.filter(|series| !series.is_empty()) {
            Some(series) => tokens_in_flight_sweep_line_icl(
                start_ns,
                generation_start_ns,
                end_ns,
                input_tokens,
                output_tokens,
                series,
            ),
            None => tokens_in_flight_sweep_line(
                start_ns,
                generation_start_ns,
                end_ns,
                input_tokens,
                output_tokens,
            ),
        };
        let (
            concurrency,
            decode_concurrency,
            prefill_concurrency,
            decode_throughput,
            prefill_throughput,
            total_throughput,
            tokens_in_flight,
        ) = if start_ns.len() >= PARALLEL_SWEEP_MIN_ROWS && rayon::current_num_threads() > 1 {
            let (small_curves, (decode_throughput, tokens_in_flight)) = rayon::join(
                || {
                    let (concurrency, rest) = rayon::join(
                        || concurrency_sweep_line(start_ns, end_ns),
                        || {
                            let (decode_concurrency, rest) = rayon::join(
                                || concurrency_sweep_line(generation_start_ns, end_ns),
                                || {
                                    let (prefill_concurrency, rest) = rayon::join(
                                        || concurrency_sweep_line(start_ns, generation_start_ns),
                                        || {
                                            rayon::join(
                                                || {
                                                    prefill_throughput_sweep_line(
                                                        start_ns,
                                                        generation_start_ns,
                                                        input_tokens,
                                                    )
                                                },
                                                || {
                                                    total_throughput_sweep_line(
                                                        start_ns,
                                                        generation_start_ns,
                                                        end_ns,
                                                        input_tokens,
                                                        output_tokens,
                                                    )
                                                },
                                            )
                                        },
                                    );
                                    (prefill_concurrency, rest)
                                },
                            );
                            (decode_concurrency, rest)
                        },
                    );
                    let (decode_concurrency, (prefill_concurrency, rest)) = rest;
                    let (prefill_throughput, total_throughput) = rest;
                    (
                        concurrency,
                        decode_concurrency,
                        prefill_concurrency,
                        prefill_throughput,
                        total_throughput,
                    )
                },
                || {
                    // These ICL-aware curves own the two largest temporary event
                    // buffers. Running them serially lets each parallel sort use
                    // the whole pool without making both peak allocations live.
                    let decode_throughput = compute_decode_throughput();
                    let tokens_in_flight = compute_tokens_in_flight();
                    (decode_throughput, tokens_in_flight)
                },
            );
            let (
                concurrency,
                decode_concurrency,
                prefill_concurrency,
                prefill_throughput,
                total_throughput,
            ) = small_curves;
            (
                concurrency,
                decode_concurrency,
                prefill_concurrency,
                decode_throughput,
                prefill_throughput,
                total_throughput,
                tokens_in_flight,
            )
        } else {
            (
                concurrency_sweep_line(start_ns, end_ns),
                concurrency_sweep_line(generation_start_ns, end_ns),
                concurrency_sweep_line(start_ns, generation_start_ns),
                compute_decode_throughput(),
                prefill_throughput_sweep_line(start_ns, generation_start_ns, input_tokens),
                total_throughput_sweep_line(
                    start_ns,
                    generation_start_ns,
                    end_ns,
                    input_tokens,
                    output_tokens,
                ),
                compute_tokens_in_flight(),
            )
        };
        Self {
            concurrency,
            decode_throughput,
            prefill_throughput,
            decode_concurrency,
            prefill_concurrency,
            total_throughput,
            tokens_in_flight,
            sample_throughput,
        }
    }

    /// Computes the nine effective and five active metric results for a window.
    pub fn compute_metrics(
        &self,
        window_start_ns: f64,
        window_end_ns: f64,
    ) -> Vec<SweepMetricResult> {
        // Effective and active per-user statistics share ratio boundaries and value
        // ordering, so each pair is computed once without retaining a ratio curve.
        let decode_per_user = compute_divided_weighted_stats(
            &self.decode_throughput,
            &self.decode_concurrency,
            window_start_ns,
            window_end_ns,
        );
        let prefill_per_user = compute_divided_weighted_stats(
            &self.prefill_throughput,
            &self.prefill_concurrency,
            window_start_ns,
            window_end_ns,
        );
        // Image samples are modeled over the full request lifetime, so the per-user
        // rate divides by overall request concurrency (not a phase concurrency).
        let sample_per_user = compute_divided_weighted_stats(
            &self.sample_throughput,
            &self.concurrency,
            window_start_ns,
            window_end_ns,
        );
        let compute_effective = || {
            let mut results = Vec::with_capacity(EFFECTIVE_METRIC_SPECS.len());
            for (spec, curve) in EFFECTIVE_METRIC_SPECS[..6].iter().copied().zip([
                &self.concurrency,
                &self.decode_throughput,
                &self.prefill_throughput,
                &self.decode_concurrency,
                &self.prefill_concurrency,
                &self.total_throughput,
            ]) {
                results.push(SweepMetricResult::from_stats(
                    spec,
                    compute_time_weighted_stats(curve, window_start_ns, window_end_ns),
                ));
            }
            results.push(SweepMetricResult::from_stats(
                EFFECTIVE_METRIC_SPECS[6],
                decode_per_user.0,
            ));
            results.push(SweepMetricResult::from_stats(
                EFFECTIVE_METRIC_SPECS[7],
                prefill_per_user.0,
            ));
            results.push(SweepMetricResult::from_stats(
                EFFECTIVE_METRIC_SPECS[8],
                compute_time_weighted_stats(&self.tokens_in_flight, window_start_ns, window_end_ns),
            ));
            results.push(SweepMetricResult::from_stats(
                EFFECTIVE_METRIC_SPECS[9],
                compute_time_weighted_stats(
                    &self.sample_throughput,
                    window_start_ns,
                    window_end_ns,
                ),
            ));
            results.push(SweepMetricResult::from_stats(
                EFFECTIVE_METRIC_SPECS[10],
                sample_per_user.0,
            ));
            results
        };
        let compute_active = || {
            let mut results = Vec::with_capacity(ACTIVE_METRIC_SPECS.len());
            for (spec, rate, mask) in [
                (
                    ACTIVE_METRIC_SPECS[0],
                    &self.decode_throughput,
                    &self.decode_concurrency,
                ),
                (
                    ACTIVE_METRIC_SPECS[1],
                    &self.prefill_throughput,
                    &self.prefill_concurrency,
                ),
            ] {
                results.push(SweepMetricResult::from_stats(
                    spec,
                    compute_active_weighted_stats(rate, mask, window_start_ns, window_end_ns),
                ));
            }
            results.push(SweepMetricResult::from_stats(
                ACTIVE_METRIC_SPECS[2],
                decode_per_user.1,
            ));
            results.push(SweepMetricResult::from_stats(
                ACTIVE_METRIC_SPECS[3],
                prefill_per_user.1,
            ));
            results.push(SweepMetricResult::from_stats(
                ACTIVE_METRIC_SPECS[4],
                compute_active_weighted_stats(
                    &self.total_throughput,
                    &self.concurrency,
                    window_start_ns,
                    window_end_ns,
                ),
            ));
            results.push(SweepMetricResult::from_stats(
                ACTIVE_METRIC_SPECS[5],
                compute_active_weighted_stats(
                    &self.sample_throughput,
                    &self.concurrency,
                    window_start_ns,
                    window_end_ns,
                ),
            ));
            results
        };
        if self
            .tokens_in_flight
            .len()
            .max(self.decode_throughput.len())
            >= PARALLEL_SWEEP_MIN_ROWS
            && rayon::current_num_threads() > 1
        {
            let (mut effective, active) = rayon::join(compute_effective, compute_active);
            effective.extend(active);
            return effective;
        }
        let mut results = compute_effective();
        results.extend(compute_active());
        results
    }
}

/// Identity and presentation metadata for a sweep-line result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepMetricSpec {
    /// Stable report tag.
    pub tag: &'static str,
    /// Human-readable header.
    pub header: &'static str,
    /// Output unit spelling.
    pub unit: &'static str,
    /// Scale applied after time-weighted statistics are computed.
    pub scale: f64,
    /// Console section for the result.
    pub console_group: MetricConsoleGroup,
}

impl SweepMetricSpec {
    const fn new(
        tag: &'static str,
        header: &'static str,
        unit: &'static str,
        scale: f64,
        console_group: MetricConsoleGroup,
    ) -> Self {
        Self {
            tag,
            header,
            unit,
            scale,
            console_group,
        }
    }
}

const EFFECTIVE_METRIC_SPECS: [SweepMetricSpec; 11] = [
    SweepMetricSpec::new(
        "effective_concurrency",
        "Effective Concurrency",
        "requests",
        1.0,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_decode_throughput",
        "Effective Decode Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_prefill_throughput",
        "Effective Prefill Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_decode_concurrency",
        "Effective Decode Concurrency",
        "requests",
        1.0,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_prefill_concurrency",
        "Effective Prefill Concurrency",
        "requests",
        1.0,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_total_throughput",
        "Effective Total Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_decode_throughput_per_user",
        "Effective Decode Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_prefill_throughput_per_user",
        "Effective Prefill Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "tokens_in_flight",
        "Tokens In Flight",
        "tokens",
        1.0,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_image_samples_per_second",
        "Effective Image Samples Per Second",
        "images/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
    SweepMetricSpec::new(
        "effective_image_samples_per_second_per_user",
        "Effective Image Samples Per Second Per User",
        "images/sec/user",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Effective,
    ),
];

const ACTIVE_METRIC_SPECS: [SweepMetricSpec; 6] = [
    SweepMetricSpec::new(
        "active_decode_throughput",
        "Active Decode Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
    SweepMetricSpec::new(
        "active_prefill_throughput",
        "Active Prefill Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
    SweepMetricSpec::new(
        "active_decode_throughput_per_user",
        "Active Decode Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
    SweepMetricSpec::new(
        "active_prefill_throughput_per_user",
        "Active Prefill Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
    SweepMetricSpec::new(
        "active_total_throughput",
        "Active Total Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
    SweepMetricSpec::new(
        "active_image_samples_per_second",
        "Active Image Samples Per Second",
        "images/sec",
        NANOS_PER_SECOND,
        MetricConsoleGroup::Active,
    ),
];

/// Boundary-safe time-weighted result for one sweep-line curve.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepMetricResult {
    /// Stable report tag.
    pub tag: &'static str,
    /// Human-readable header.
    pub header: &'static str,
    /// Output unit spelling.
    pub unit: &'static str,
    /// Duration-weighted average.
    pub avg: MetricValue,
    /// Minimum observed step value.
    pub min: MetricValue,
    /// Maximum observed step value.
    pub max: MetricValue,
    /// Duration-weighted median.
    pub p50: MetricValue,
    /// Duration-weighted p90.
    pub p90: MetricValue,
    /// Duration-weighted p95.
    pub p95: MetricValue,
    /// Duration-weighted p99.
    pub p99: MetricValue,
    /// Duration-weighted population standard deviation.
    pub std: Option<f64>,
    /// Console section for the result.
    pub console_group: MetricConsoleGroup,
}

impl SweepMetricResult {
    fn from_stats(spec: SweepMetricSpec, stats: SweepLineStats) -> Self {
        let scaled = |value: f64| MetricValue::from_f64(value * spec.scale, false);
        Self {
            tag: spec.tag,
            header: spec.header,
            unit: spec.unit,
            avg: scaled(stats.avg),
            min: scaled(stats.min),
            max: scaled(stats.max),
            p50: scaled(stats.p50),
            p90: scaled(stats.p90),
            p95: scaled(stats.p95),
            p99: scaled(stats.p99),
            std: (stats.std * spec.scale)
                .is_finite()
                .then_some(stats.std * spec.scale),
            console_group: spec.console_group,
        }
    }
}

fn merged_timestamps(left: &StepFn, right: &StepFn) -> Vec<f64> {
    let mut timestamps = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() || right_index < right.len() {
        let take_left = right_index == right.len()
            || (left_index < left.len()
                && left.timestamps_ns()[left_index].total_cmp(&right.timestamps_ns()[right_index])
                    != Ordering::Greater);
        let timestamp = if take_left {
            let timestamp = left.timestamps_ns()[left_index];
            left_index += 1;
            timestamp
        } else {
            let timestamp = right.timestamps_ns()[right_index];
            right_index += 1;
            timestamp
        };
        if timestamps
            .last()
            .is_none_or(|previous| !same_timestamp(*previous, timestamp))
        {
            timestamps.push(timestamp);
        }
    }
    timestamps
}

fn same_timestamp(left: f64, right: f64) -> bool {
    left == right || (left.is_nan() && right.is_nan())
}

pub(crate) fn lower_bound(values: &[f64], query: f64) -> usize {
    values.partition_point(|value| value.total_cmp(&query) == Ordering::Less)
}

pub(crate) fn upper_bound(values: &[f64], query: f64) -> usize {
    values.partition_point(|value| value.total_cmp(&query) != Ordering::Greater)
}

pub(crate) fn assert_aligned(expected: usize, actual: &[usize]) {
    assert!(actual.iter().all(|length| *length == expected));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_sort_before_starts_at_equal_timestamps() {
        let curve = sweep_line_cumsum(vec![
            SweepEvent::new(0.0, 1.0),
            SweepEvent::new(10.0, -1.0),
            SweepEvent::new(10.0, 1.0),
            SweepEvent::new(20.0, -1.0),
        ]);
        assert_eq!(curve.timestamps_ns(), &[0.0, 10.0, 10.0, 20.0]);
        assert_eq!(curve.values(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(curve.value_at(10.0), 1.0);
    }

    #[test]
    fn single_thread_pool_large_event_sort_matches_sequential_reference() {
        let mut events = (0..PARALLEL_EVENT_SORT_MIN_EVENTS)
            .map(|index| {
                let timestamp_ns = ((index * 17) % 4_096) as f64;
                let magnitude = ((index / 2) % 7 + 1) as f64;
                let delta = if index % 2 == 0 {
                    -magnitude
                } else {
                    magnitude
                };
                SweepEvent::new(timestamp_ns, delta)
            })
            .collect::<Vec<_>>();
        let mut expected = events.clone();
        expected.sort_unstable_by(|left, right| {
            left.timestamp_ns
                .total_cmp(&right.timestamp_ns)
                .then_with(|| (left.delta > 0.0).cmp(&(right.delta > 0.0)))
                .then_with(|| left.delta.total_cmp(&right.delta))
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            sort_sweep_events(&mut events);
        });

        assert_eq!(events, expected);
    }

    #[test]
    fn multi_thread_large_event_sort_matches_sequential_reference() {
        // Exercises the parallel radix scatter (unsafe disjoint writes) at the
        // threshold with dense duplicate timestamps and mixed-sign deltas, and
        // pins it byte-for-byte to the `(timestamp, delta)` comparator order.
        let mut events = (0..PARALLEL_EVENT_SORT_MIN_EVENTS * 3)
            .map(|index| {
                let timestamp_ns = ((index * 31) % 2_048) as f64;
                let magnitude = ((index / 3) % 9 + 1) as f64;
                let delta = if index % 2 == 0 {
                    -magnitude
                } else {
                    magnitude
                };
                SweepEvent::new(timestamp_ns, delta)
            })
            .collect::<Vec<_>>();
        let mut expected = events.clone();
        expected.sort_unstable_by(|left, right| {
            left.timestamp_ns
                .total_cmp(&right.timestamp_ns)
                .then_with(|| (left.delta > 0.0).cmp(&(right.delta > 0.0)))
                .then_with(|| left.delta.total_cmp(&right.delta))
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| {
            assert!(rayon::current_num_threads() > 1);
            sort_sweep_events(&mut events);
        });
        assert_eq!(events, expected);
    }

    #[test]
    fn compact_curve_keeps_the_right_continuous_same_timestamp_value() {
        let curve = sweep_line_cumsum_compact(vec![
            SweepEvent::new(0.0, 1.0),
            SweepEvent::new(10.0, -1.0),
            SweepEvent::new(10.0, 1.0),
            SweepEvent::new(20.0, -1.0),
        ]);
        assert_eq!(curve.timestamps_ns(), &[0.0, 10.0, 20.0]);
        assert_eq!(curve.values(), &[1.0, 1.0, 0.0]);
        assert_eq!(curve.value_at(10.0), 1.0);
    }

    #[test]
    fn fp_roundoff_residual_is_snapped_to_zero() {
        let curve = sweep_line_cumsum(vec![
            SweepEvent::new(0.0, 1e12),
            SweepEvent::new(1.0, 1.0),
            SweepEvent::new(2.0, -1e12),
            SweepEvent::new(3.0, -1.0),
        ]);
        assert_eq!(curve.final_value(), 0.0);
    }

    #[test]
    fn decode_rate_excludes_the_ttft_token() {
        let curve = throughput_sweep_line(&[0.0], &[100.0], &[101.0]);
        assert_eq!(curve.values(), &[1.0, 0.0]);
    }

    #[test]
    fn zero_output_tokens_never_produce_negative_rate() {
        assert!(throughput_sweep_line(&[0.0], &[100.0], &[0.0]).is_empty());
    }

    #[test]
    fn prefill_rate_uses_every_input_token() {
        let curve = prefill_throughput_sweep_line(&[0.0], &[50.0], &[100.0]);
        assert_eq!(curve.values(), &[2.0, 0.0]);
    }

    #[test]
    fn total_curve_matches_separate_curve_addition() {
        let starts = [0.0, 10.0, 20.0];
        let first = [50.0, 60.0, 70.0];
        let ends = [150.0, 160.0, 170.0];
        let input = [100.0, 200.0, 150.0];
        let output = [101.0, 51.0, 76.0];
        let total = total_throughput_sweep_line(&starts, &first, &ends, &input, &output);
        let separate = prefill_throughput_sweep_line(&starts, &first, &input)
            .add(&throughput_sweep_line(&first, &ends, &output));
        let total_stats = compute_time_weighted_stats(&total, 0.0, 170.0);
        let separate_stats = compute_time_weighted_stats(&separate, 0.0, 170.0);
        assert!((total_stats.avg - separate_stats.avg).abs() < 1e-12);
        assert!((total_stats.max - separate_stats.max).abs() < 1e-12);
    }

    #[test]
    fn safe_division_yields_zero_for_zero_denominator() {
        let numerator = StepFn::new(vec![0.0, 50.0], vec![10.0, 0.0]);
        let denominator = StepFn::new(vec![0.0, 50.0], vec![0.0, 0.0]);
        assert_eq!(numerator.divide(&denominator).values(), &[0.0, 0.0]);
    }

    #[test]
    fn full_bundle_emits_eleven_effective_and_six_active_metrics() {
        let curves =
            SweepLineCurves::compute(&[0.0], &[10.0], &[110.0], &[100.0], &[11.0], &[4.0], None);
        let metrics = curves.compute_metrics(0.0, 110.0);
        assert_eq!(metrics.len(), 17);
        assert_eq!(metrics[0].tag, "effective_concurrency");
        assert_eq!(metrics[9].tag, "effective_image_samples_per_second");
        assert_eq!(
            metrics[10].tag,
            "effective_image_samples_per_second_per_user"
        );
        assert_eq!(metrics[15].tag, "active_total_throughput");
        assert_eq!(metrics[16].tag, "active_image_samples_per_second");
        // 4 images spread over [0, 110) ns, duration-weighted over the same window
        // and scaled to per-second: 4 / 110ns * 1e9 = 36_363_636.36… images/sec.
        let effective = metrics[9].avg.as_f64().expect("sample rate is finite");
        assert!((effective - 4.0 / 110.0 * NANOS_PER_SECOND).abs() < 1e-6);
        // The single request is in flight over the whole [0, 110) span, so the
        // active-masked rate equals the effective rate here.
        let active = metrics[16]
            .avg
            .as_f64()
            .expect("active sample rate is finite");
        assert!((active - 4.0 / 110.0 * NANOS_PER_SECOND).abs() < 1e-6);
        // Concurrency is exactly 1 the whole span, so the per-user rate (÷ overall
        // concurrency) equals the aggregate effective rate here.
        let per_user = metrics[10].avg.as_f64().expect("per-user rate is finite");
        assert!((per_user - 4.0 / 110.0 * NANOS_PER_SECOND).abs() < 1e-6);
    }

    #[test]
    fn metric_specs_preserve_all_seventeen_exact_identities() {
        let observed = EFFECTIVE_METRIC_SPECS
            .into_iter()
            .chain(ACTIVE_METRIC_SPECS)
            .map(|spec| {
                (
                    spec.tag,
                    spec.header,
                    spec.unit,
                    spec.scale,
                    spec.console_group,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            observed,
            vec![
                (
                    "effective_concurrency",
                    "Effective Concurrency",
                    "requests",
                    1.0,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_decode_throughput",
                    "Effective Decode Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_prefill_throughput",
                    "Effective Prefill Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_decode_concurrency",
                    "Effective Decode Concurrency",
                    "requests",
                    1.0,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_prefill_concurrency",
                    "Effective Prefill Concurrency",
                    "requests",
                    1.0,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_total_throughput",
                    "Effective Total Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_decode_throughput_per_user",
                    "Effective Decode Throughput Per User",
                    "tokens/sec/user",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_prefill_throughput_per_user",
                    "Effective Prefill Throughput Per User",
                    "tokens/sec/user",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "tokens_in_flight",
                    "Tokens In Flight",
                    "tokens",
                    1.0,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_image_samples_per_second",
                    "Effective Image Samples Per Second",
                    "images/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "effective_image_samples_per_second_per_user",
                    "Effective Image Samples Per Second Per User",
                    "images/sec/user",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Effective,
                ),
                (
                    "active_decode_throughput",
                    "Active Decode Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
                (
                    "active_prefill_throughput",
                    "Active Prefill Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
                (
                    "active_decode_throughput_per_user",
                    "Active Decode Throughput Per User",
                    "tokens/sec/user",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
                (
                    "active_prefill_throughput_per_user",
                    "Active Prefill Throughput Per User",
                    "tokens/sec/user",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
                (
                    "active_total_throughput",
                    "Active Total Throughput",
                    "tokens/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
                (
                    "active_image_samples_per_second",
                    "Active Image Samples Per Second",
                    "images/sec",
                    NANOS_PER_SECOND,
                    MetricConsoleGroup::Active,
                ),
            ]
        );
    }

    #[test]
    fn randomized_closed_intervals_remain_nonnegative_and_balanced() {
        let mut state = 0x8f3d_9a17_4c2b_6105_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for size in 1..=128 {
            let mut starts = Vec::with_capacity(size);
            let mut ends = Vec::with_capacity(size);
            let mut weights = Vec::with_capacity(size);
            for _ in 0..size {
                let start = (next() % 10_000) as f64;
                starts.push(start);
                ends.push(start + (next() % 1_000 + 1) as f64);
                weights.push((next() % 10_000 + 1) as f64 / 10.0);
            }
            for curve in [
                concurrency_sweep_line(&starts, &ends),
                weighted_concurrency_sweep_line(&starts, &ends, &weights),
            ] {
                assert_eq!(curve.final_value(), 0.0);
                assert!(curve.values().iter().all(|value| *value >= 0.0));
            }
        }
    }

    #[test]
    fn duration_weighted_percentiles_are_monotone() {
        let curve = StepFn::new(
            vec![0.0, 5.0, 15.0, 40.0, 90.0, 100.0],
            vec![10.0, 1.0, 100.0, 5.0, 50.0, 0.0],
        );
        let stats = compute_time_weighted_stats(&curve, 0.0, 100.0);
        assert!(stats.min <= stats.p50);
        assert!(stats.p50 <= stats.p90);
        assert!(stats.p90 <= stats.p95);
        assert!(stats.p95 <= stats.p99);
        assert!(stats.p99 <= stats.max);
    }
}
