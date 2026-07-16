// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-cache tokens-in-flight and ICL-aware decode curves.

use super::{
    RepeatedSweepEvent, StepFn, SweepEvent, assert_aligned, sweep_line_cumsum_compact,
    sweep_line_cumsum_repeated,
};

/// Borrowed CSR-style inter-chunk-latency series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IclSeries<'a> {
    values_ns: &'a [f64],
    offsets: &'a [usize],
    lengths: &'a [usize],
    append_order: &'a [usize],
}

impl<'a> IclSeries<'a> {
    /// Builds an ICL series from flat values and absolute record slices.
    ///
    /// # Panics
    ///
    /// Panics when row metadata is misaligned or a slice exceeds the values array.
    pub fn new(
        values_ns: &'a [f64],
        offsets: &'a [usize],
        lengths: &'a [usize],
        append_order: &'a [usize],
    ) -> Self {
        assert_eq!(offsets.len(), lengths.len());
        assert!(offsets.iter().all(|offset| *offset <= values_ns.len()));
        assert!(append_order.iter().all(|record| *record < offsets.len()));
        Self {
            values_ns,
            offsets,
            lengths,
            append_order,
        }
    }

    /// Returns true when no ICL observations are retained.
    pub fn is_empty(self) -> bool {
        self.values_ns.is_empty()
    }

    /// Returns the flat ICL values in nanoseconds.
    pub fn values_ns(self) -> &'a [f64] {
        self.values_ns
    }

    /// Returns each record's start offset in the flat ICL array.
    pub fn offsets(self) -> &'a [usize] {
        self.offsets
    }

    /// Returns each record's ICL value count.
    pub fn lengths(self) -> &'a [usize] {
        self.lengths
    }

    /// Returns records in flat-value append order.
    pub fn append_order(self) -> &'a [usize] {
        self.append_order
    }

    fn values_for_record(self, record: usize) -> &'a [f64] {
        let start = self.offsets[record];
        let end = start
            .checked_add(self.lengths[record])
            .expect("ICL record range overflow");
        assert!(end <= self.values_ns.len());
        &self.values_ns[start..end]
    }
}

/// Computes coarse tokens in flight for prefill and generation phases.
///
/// Input tokens enter at request start and remain until request end. Output tokens
/// enter together at generation start.
pub fn tokens_in_flight_sweep_line(
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
    let mut events = Vec::with_capacity(start_ns.len() * 3);
    for ((((&start, &generation_start), &end), &input), &output) in start_ns
        .iter()
        .zip(generation_start_ns)
        .zip(end_ns)
        .zip(input_tokens)
        .zip(output_tokens)
    {
        let prefill_valid = !start.is_nan()
            && !input.is_nan()
            && !generation_start.is_nan()
            && generation_start > start;
        let generation_duration = end - generation_start;
        let generation_valid =
            !generation_start.is_nan() && !output.is_nan() && generation_duration > 0.0;
        let has_end = !end.is_nan();

        if prefill_valid {
            events.push(SweepEvent::new(start, input));
        }
        if generation_valid {
            events.push(SweepEvent::new(generation_start, output));
        }
        match (prefill_valid && has_end, generation_valid && has_end) {
            (true, true) => events.push(SweepEvent::new(end, -(input + output))),
            (true, false) => events.push(SweepEvent::new(end, -input)),
            (false, true) => events.push(SweepEvent::new(end, -output)),
            (false, false) => {}
        }
    }
    sweep_line_cumsum_compact(events)
}

/// Computes ICL-aware tokens in flight, ramping output tokens at chunk boundaries.
///
/// One token enters at TTFT. The remaining `osl - 1` tokens are distributed across
/// all non-negative ICL entries, including zero-duration back-to-back chunks. Chunk
/// timestamps at or after `end_ns` are clamped to the adjacent representable float
/// below the end so the end-before-start tie-break cannot unbalance the curve.
pub fn tokens_in_flight_sweep_line_icl(
    start_ns: &[f64],
    generation_start_ns: &[f64],
    end_ns: &[f64],
    input_tokens: &[f64],
    output_tokens: &[f64],
    icl: IclSeries<'_>,
) -> StepFn {
    assert_aligned(
        start_ns.len(),
        &[
            generation_start_ns.len(),
            end_ns.len(),
            input_tokens.len(),
            output_tokens.len(),
            icl.offsets.len(),
        ],
    );
    if icl.is_empty() {
        return tokens_in_flight_sweep_line(
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens,
            output_tokens,
        );
    }

    validate_record_slices(icl, output_tokens.len());
    let initial_capacity = icl
        .values_ns
        .len()
        .min(icl.append_order.len().saturating_mul(8));
    let mut events = Vec::with_capacity(initial_capacity);
    let mut position = 0;
    while position < icl.append_order.len() {
        let record = icl.append_order[position];
        let repetitions = repeated_record_count(
            icl,
            position,
            &[
                start_ns,
                generation_start_ns,
                end_ns,
                input_tokens,
                output_tokens,
            ],
        );
        let values = icl.values_for_record(record);
        let count = values.len();
        let mut cumulative = 0.0;
        let mut has_valid_chunks = false;
        for &value in values {
            cumulative += value;
            let generation_start = generation_start_ns[record];
            let tokens = output_tokens[record];
            if generation_start.is_nan() || value < 0.0 || tokens.is_nan() || tokens < 1.0 {
                continue;
            }
            let mut timestamp = generation_start + cumulative;
            if !end_ns[record].is_nan() && timestamp >= end_ns[record] {
                timestamp = next_down(end_ns[record]);
            }
            events.push(RepeatedSweepEvent::new(
                timestamp,
                (tokens - 1.0) / count as f64,
                repetitions,
            ));
            has_valid_chunks = true;
        }
        if has_valid_chunks {
            events.push(RepeatedSweepEvent::new(
                generation_start_ns[record],
                1.0,
                repetitions,
            ));
        }

        let prefill_valid = !start_ns[record].is_nan()
            && !input_tokens[record].is_nan()
            && !generation_start_ns[record].is_nan()
            && generation_start_ns[record] > start_ns[record];
        if prefill_valid {
            events.push(RepeatedSweepEvent::new(
                start_ns[record],
                input_tokens[record],
                repetitions,
            ));
        }
        if !end_ns[record].is_nan() {
            let delta = match (prefill_valid, has_valid_chunks) {
                (true, true) => Some(-(input_tokens[record] + output_tokens[record])),
                (true, false) => Some(-input_tokens[record]),
                (false, true) => Some(-output_tokens[record]),
                (false, false) => None,
            };
            if let Some(delta) = delta {
                events.push(RepeatedSweepEvent::new(end_ns[record], delta, repetitions));
            }
        }
        position += repetitions;
    }
    sweep_line_cumsum_repeated(events)
}

/// Computes ICL-aware decode throughput at each positive-duration chunk interval.
///
/// Zero ICLs cannot carry a rate and are excluded from both the event set and the
/// per-record divisor. The remaining `osl - 1` tokens are spread across positive
/// intervals.
pub fn throughput_sweep_line_icl(
    generation_start_ns: &[f64],
    output_tokens: &[f64],
    icl: IclSeries<'_>,
) -> StepFn {
    assert_aligned(
        generation_start_ns.len(),
        &[output_tokens.len(), icl.offsets.len()],
    );
    if icl.is_empty() {
        return StepFn::empty();
    }
    validate_record_slices(icl, output_tokens.len());

    let initial_capacity = icl
        .values_ns
        .len()
        .saturating_mul(2)
        .min(icl.append_order.len().saturating_mul(8));
    let mut events = Vec::with_capacity(initial_capacity);
    let mut position = 0;
    while position < icl.append_order.len() {
        let record = icl.append_order[position];
        let repetitions =
            repeated_record_count(icl, position, &[generation_start_ns, output_tokens]);
        let values = icl.values_for_record(record);
        let count = values.iter().filter(|value| **value > 0.0).count();
        let mut relative_end = 0.0;
        for &value in values {
            relative_end += value;
            let generation_start = generation_start_ns[record];
            let tokens = output_tokens[record];
            if generation_start.is_nan() || value <= 0.0 || tokens.is_nan() || tokens < 1.0 {
                continue;
            }
            if count == 0 {
                continue;
            }
            let interval_end = generation_start + relative_end;
            let interval_start = interval_end - value;
            let rate = ((tokens - 1.0) / count as f64) / value;
            events.push(RepeatedSweepEvent::new(interval_start, rate, repetitions));
            events.push(RepeatedSweepEvent::new(interval_end, -rate, repetitions));
        }
        position += repetitions;
    }
    sweep_line_cumsum_repeated(events)
}

fn repeated_record_count(icl: IclSeries<'_>, position: usize, columns: &[&[f64]]) -> usize {
    let record = icl.append_order[position];
    let values = icl.values_for_record(record);
    let mut repetitions = 1;
    for &candidate in &icl.append_order[position + 1..] {
        if columns
            .iter()
            .any(|column| column[record].to_bits() != column[candidate].to_bits())
            || values.len() != icl.lengths[candidate]
            || !values
                .iter()
                .zip(icl.values_for_record(candidate))
                .all(|(left, right)| left.to_bits() == right.to_bits())
        {
            break;
        }
        repetitions += 1;
    }
    repetitions
}

fn validate_record_slices(icl: IclSeries<'_>, record_count: usize) {
    assert_eq!(icl.offsets.len(), record_count);
    assert_eq!(icl.lengths.len(), record_count);
    let mut seen = vec![false; record_count];
    let mut cursor = 0_usize;
    for &record in icl.append_order {
        assert!(!seen[record], "ICL append order contains a duplicate row");
        seen[record] = true;
        assert_eq!(
            icl.offsets[record], cursor,
            "ICL slices must follow flat append order"
        );
        cursor = cursor
            .checked_add(icl.lengths[record])
            .expect("ICL flat length overflow");
        assert!(cursor <= icl.values_ns.len());
    }
    assert_eq!(cursor, icl.values_ns.len());
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(curve: &StepFn, timestamp: f64) -> f64 {
        curve.value_at(timestamp)
    }

    #[test]
    fn coarse_tokens_in_flight_holds_input_through_generation() {
        let curve = tokens_in_flight_sweep_line(&[0.0], &[10.0], &[60.0], &[100.0], &[50.0]);
        assert_eq!(lookup(&curve, 5.0), 100.0);
        assert_eq!(lookup(&curve, 30.0), 150.0);
        assert_eq!(curve.final_value(), 0.0);
    }

    #[test]
    fn icl_throughput_integrates_to_osl_minus_one() {
        let values = [10.0, 10.0, 10.0];
        let offsets = [0];
        let lengths = [3];
        let append_order = [0];
        let curve = throughput_sweep_line_icl(
            &[0.0],
            &[6.0],
            IclSeries::new(&values, &offsets, &lengths, &append_order),
        );
        let integral = curve
            .timestamps_ns()
            .windows(2)
            .zip(curve.values())
            .map(|(timestamps, value)| (timestamps[1] - timestamps[0]) * value)
            .sum::<f64>();
        assert!((integral - 5.0).abs() < 1e-12);
    }

    #[test]
    fn zero_icl_is_counted_for_tokens_but_not_rate() {
        let values = [0.0, 10.0];
        let offsets = [0];
        let lengths = [2];
        let append_order = [0];
        let series = IclSeries::new(&values, &offsets, &lengths, &append_order);
        let tokens =
            tokens_in_flight_sweep_line_icl(&[0.0], &[10.0], &[21.0], &[100.0], &[5.0], series);
        let rate = throughput_sweep_line_icl(&[10.0], &[5.0], series);
        assert_eq!(tokens.final_value(), 0.0);
        assert!(rate.values().iter().all(|value| value.is_finite()));
        assert!((rate.values().iter().copied().fold(0.0, f64::max) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn chunk_at_end_is_clamped_to_previous_float() {
        let values = [1_024.0];
        let offsets = [0];
        let lengths = [1];
        let append_order = [0];
        let end = 1.7e18;
        let curve = tokens_in_flight_sweep_line_icl(
            &[end - 2_048.0],
            &[end - 1_024.0],
            &[end],
            &[10.0],
            &[2.0],
            IclSeries::new(&values, &offsets, &lengths, &append_order),
        );
        let chunk_timestamp = next_down(end);
        assert!(curve.timestamps_ns().contains(&chunk_timestamp));
        assert!(chunk_timestamp < end);
        assert_eq!(curve.final_value(), 0.0);
        assert!(curve.values().iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn invalid_chunk_record_does_not_take_end_subtraction_path() {
        let values = [50.0, 50.0, 30.0, 30.0];
        let offsets = [0, 2];
        let lengths = [2, 2];
        let append_order = [0, 1];
        let curve = tokens_in_flight_sweep_line_icl(
            &[0.0, f64::NAN],
            &[10.0, f64::NAN],
            &[110.0, 120.0],
            &[100.0, f64::NAN],
            &[20.0, 40.0],
            IclSeries::new(&values, &offsets, &lengths, &append_order),
        );
        assert_eq!(curve.final_value(), 0.0);
        assert!(
            curve
                .values()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }

    #[test]
    fn output_token_nan_cannot_poison_curve() {
        let values = [50.0, 50.0, 30.0, 30.0];
        let offsets = [0, 2];
        let lengths = [2, 2];
        let append_order = [0, 1];
        let curve = tokens_in_flight_sweep_line_icl(
            &[0.0, 5.0],
            &[10.0, 15.0],
            &[110.0, 120.0],
            &[100.0, 50.0],
            &[20.0, f64::NAN],
            IclSeries::new(&values, &offsets, &lengths, &append_order),
        );
        assert_eq!(curve.final_value(), 0.0);
        assert!(curve.values().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn multi_record_icl_curves_match_independent_reference_enumeration() {
        // This is the native twin of the brute-force oracle. It deliberately
        // uses overlapping records, unequal gaps, and a zero-duration gap so the
        // reference does not merely restate the event-generation loop.
        let start = [0.0, 17.0, 53.0];
        let generation_start = [10.0, 30.0, 70.0];
        let end = [70.0, 90.0, 140.0];
        let input = [100.0, 50.0, 80.0];
        let output = [5.0, 7.0, 4.0];
        let icl_values = [
            10.0, 0.0, 20.0, 30.0, // record 0
            5.0, 15.0, 10.0, 30.0, // record 1
            20.0, 20.0, 20.0, 10.0, // record 2
        ];
        let offsets = [0, 4, 8];
        let lengths = [4, 4, 4];
        let append_order = [0, 1, 2];
        let series = IclSeries::new(&icl_values, &offsets, &lengths, &append_order);

        let tokens = tokens_in_flight_sweep_line_icl(
            &start,
            &generation_start,
            &end,
            &input,
            &output,
            series,
        );
        let throughput = throughput_sweep_line_icl(&generation_start, &output, series);

        for sample in (0..320).map(|index| f64::from(index) * 0.5 + 0.25) {
            let expected_tokens = reference_tokens_in_flight(
                sample,
                &start,
                &generation_start,
                &end,
                &input,
                &output,
                &icl_values,
                &offsets,
            );
            let expected_throughput =
                reference_throughput(sample, &generation_start, &output, &icl_values, &offsets);
            assert!((tokens.value_at(sample) - expected_tokens).abs() < 1e-10);
            assert!((throughput.value_at(sample) - expected_throughput).abs() < 1e-12);
        }

        let integral = throughput
            .timestamps_ns()
            .windows(2)
            .zip(throughput.values())
            .map(|(timestamps, value)| (timestamps[1] - timestamps[0]) * value)
            .sum::<f64>();
        let expected_integral = output.iter().map(|tokens| tokens - 1.0).sum::<f64>();
        assert!((integral - expected_integral).abs() < 1e-10);
        assert_eq!(tokens.final_value(), 0.0);
        assert!(tokens.values().iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn icl_curves_are_invariant_to_record_append_order() {
        let start = [0.0, 1_000.0, 2_000.0];
        let generation_start = [100.0, 1_100.0, 2_100.0];
        let end = [400.0, 1_500.0, 2_600.0];
        let input = [10.0, 20.0, 30.0];
        let output = [4.0, 3.0, 5.0];
        let lengths = [3, 2, 2];

        let canonical_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let canonical_offsets = [0, 3, 5];
        let canonical_order = [0, 1, 2];
        let canonical = IclSeries::new(
            &canonical_values,
            &canonical_offsets,
            &lengths,
            &canonical_order,
        );

        let reordered_values = [60.0, 70.0, 10.0, 20.0, 30.0, 40.0, 50.0];
        let reordered_offsets = [2, 5, 0];
        let reordered_order = [2, 0, 1];
        let reordered = IclSeries::new(
            &reordered_values,
            &reordered_offsets,
            &lengths,
            &reordered_order,
        );

        let canonical_tokens = tokens_in_flight_sweep_line_icl(
            &start,
            &generation_start,
            &end,
            &input,
            &output,
            canonical,
        );
        let reordered_tokens = tokens_in_flight_sweep_line_icl(
            &start,
            &generation_start,
            &end,
            &input,
            &output,
            reordered,
        );
        assert_eq!(canonical_tokens, reordered_tokens);

        let canonical_throughput = throughput_sweep_line_icl(&generation_start, &output, canonical);
        let reordered_throughput = throughput_sweep_line_icl(&generation_start, &output, reordered);
        assert_eq!(canonical_throughput, reordered_throughput);
    }

    #[test]
    fn adjacent_identical_icl_trajectories_match_uncompressed_order_bit_exactly() {
        let start = [0.0, 0.0, 50.0];
        let generation_start = [10.0, 10.0, 60.0];
        let end = [40.0, 40.0, 100.0];
        let input = [100.0, 100.0, 80.0];
        let output = [3.0, 3.0, 3.0];
        let lengths = [2, 2, 2];

        let grouped_values = [10.0, 20.0, 10.0, 20.0, 15.0, 25.0];
        let grouped_offsets = [0, 2, 4];
        let grouped_order = [0, 1, 2];
        let grouped = IclSeries::new(&grouped_values, &grouped_offsets, &lengths, &grouped_order);

        // Separating the identical rows prevents run compression while
        // retaining the same request facts and final sweep-event multiset.
        let ungrouped_values = [10.0, 20.0, 15.0, 25.0, 10.0, 20.0];
        let ungrouped_offsets = [0, 4, 2];
        let ungrouped_order = [0, 2, 1];
        let ungrouped = IclSeries::new(
            &ungrouped_values,
            &ungrouped_offsets,
            &lengths,
            &ungrouped_order,
        );

        assert_eq!(
            tokens_in_flight_sweep_line_icl(
                &start,
                &generation_start,
                &end,
                &input,
                &output,
                grouped,
            ),
            tokens_in_flight_sweep_line_icl(
                &start,
                &generation_start,
                &end,
                &input,
                &output,
                ungrouped,
            )
        );
        assert_eq!(
            throughput_sweep_line_icl(&generation_start, &output, grouped),
            throughput_sweep_line_icl(&generation_start, &output, ungrouped)
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_tokens_in_flight(
        timestamp: f64,
        start: &[f64],
        generation_start: &[f64],
        end: &[f64],
        input: &[f64],
        output: &[f64],
        icl_values: &[f64],
        offsets: &[usize],
    ) -> f64 {
        (0..start.len())
            .map(|record| {
                if timestamp < start[record] || timestamp >= end[record] {
                    return 0.0;
                }
                if timestamp < generation_start[record] {
                    return input[record];
                }

                let lo = offsets[record];
                let hi = offsets.get(record + 1).copied().unwrap_or(icl_values.len());
                let mut relative_arrival = 0.0;
                let landed = icl_values[lo..hi]
                    .iter()
                    .filter(|gap| {
                        relative_arrival += **gap;
                        timestamp >= generation_start[record] + relative_arrival
                    })
                    .count();
                input[record] + 1.0 + landed as f64 * (output[record] - 1.0) / (hi - lo) as f64
            })
            .sum()
    }

    fn reference_throughput(
        timestamp: f64,
        generation_start: &[f64],
        output: &[f64],
        icl_values: &[f64],
        offsets: &[usize],
    ) -> f64 {
        (0..generation_start.len())
            .map(|record| {
                let lo = offsets[record];
                let hi = offsets.get(record + 1).copied().unwrap_or(icl_values.len());
                let positive_count = icl_values[lo..hi].iter().filter(|gap| **gap > 0.0).count();
                if positive_count == 0 {
                    return 0.0;
                }
                let tokens_per_interval = (output[record] - 1.0) / positive_count as f64;
                let mut interval_start = generation_start[record];
                icl_values[lo..hi]
                    .iter()
                    .map(|gap| {
                        let interval_end = interval_start + gap;
                        let contribution = if *gap > 0.0
                            && interval_start <= timestamp
                            && timestamp < interval_end
                        {
                            tokens_per_interval / gap
                        } else {
                            0.0
                        };
                        interval_start = interval_end;
                        contribution
                    })
                    .sum::<f64>()
            })
            .sum()
    }
}
