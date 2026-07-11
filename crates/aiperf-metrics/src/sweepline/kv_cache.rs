// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-cache tokens-in-flight and ICL-aware decode curves.

use super::{StepFn, SweepEvent, assert_aligned, sweep_line_cumsum};

/// Borrowed CSR-style inter-chunk-latency series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IclSeries<'a> {
    values_ns: &'a [f64],
    record_indices: &'a [usize],
    offsets: &'a [usize],
}

impl<'a> IclSeries<'a> {
    /// Builds an ICL series from flat values, a record index per value, and one
    /// start offset per record.
    ///
    /// # Panics
    ///
    /// Panics when flat arrays differ in length or an offset exceeds the values array.
    pub fn new(values_ns: &'a [f64], record_indices: &'a [usize], offsets: &'a [usize]) -> Self {
        assert_eq!(values_ns.len(), record_indices.len());
        assert!(offsets.iter().all(|offset| *offset <= values_ns.len()));
        Self {
            values_ns,
            record_indices,
            offsets,
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

    /// Returns the owning record index for each flat ICL value.
    pub fn record_indices(self) -> &'a [usize] {
        self.record_indices
    }

    /// Returns each record's start offset in the flat ICL array.
    pub fn offsets(self) -> &'a [usize] {
        self.offsets
    }
}

/// Computes coarse tokens in flight for prefill and generation phases.
///
/// Input tokens enter at request start and remain until request end. Output tokens
/// enter together at generation start. Addition/subtraction masks follow
/// `src/aiperf/analysis/sweepline_kv_cache.py:18-97`.
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
    sweep_line_cumsum(events)
}

/// Computes ICL-aware tokens in flight, ramping output tokens at chunk boundaries.
///
/// One token enters at TTFT. The remaining `osl - 1` tokens are distributed across
/// all non-negative ICL entries, including zero-duration back-to-back chunks. Chunk
/// timestamps at or after `end_ns` are clamped to the adjacent representable float
/// below the end so the end-before-start tie-break cannot unbalance the curve. This
/// ports `src/aiperf/analysis/sweepline_kv_cache.py:100-265`.
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

    let chunks = build_chunk_events(generation_start_ns, end_ns, output_tokens, icl);
    let mut events = Vec::with_capacity(start_ns.len() * 3 + chunks.events.len());
    for (record, has_chunks) in chunks.has_valid_chunks.iter().copied().enumerate() {
        if has_chunks
            && !generation_start_ns[record].is_nan()
            && !output_tokens[record].is_nan()
            && output_tokens[record] >= 1.0
        {
            events.push(SweepEvent::new(generation_start_ns[record], 1.0));
        }
    }
    events.extend(chunks.events);

    for record in 0..start_ns.len() {
        let prefill_valid = !start_ns[record].is_nan()
            && !input_tokens[record].is_nan()
            && !generation_start_ns[record].is_nan()
            && generation_start_ns[record] > start_ns[record];
        if prefill_valid {
            events.push(SweepEvent::new(start_ns[record], input_tokens[record]));
        }

        if end_ns[record].is_nan() {
            continue;
        }
        match (prefill_valid, chunks.has_valid_chunks[record]) {
            (true, true) => events.push(SweepEvent::new(
                end_ns[record],
                -(input_tokens[record] + output_tokens[record]),
            )),
            (true, false) => {
                events.push(SweepEvent::new(end_ns[record], -input_tokens[record]));
            }
            (false, true) => {
                events.push(SweepEvent::new(end_ns[record], -output_tokens[record]));
            }
            (false, false) => {}
        }
    }
    sweep_line_cumsum(events)
}

/// Computes ICL-aware decode throughput at each positive-duration chunk interval.
///
/// Zero ICLs cannot carry a rate and are excluded from both the event set and the
/// per-record divisor. The remaining `osl - 1` tokens are spread across positive
/// intervals, matching `src/aiperf/analysis/sweepline_kv_cache.py:268-353`.
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
    validate_record_indices(icl, output_tokens.len());

    let mut positive_counts = vec![0_usize; output_tokens.len()];
    for (&value, &record) in icl.values_ns.iter().zip(icl.record_indices) {
        if value > 0.0 {
            positive_counts[record] += 1;
        }
    }
    let relative_cumsum = relative_cumsums(icl);
    let mut events = Vec::with_capacity(icl.values_ns.len() * 2);
    for (index, &relative_end) in relative_cumsum.iter().enumerate() {
        let record = icl.record_indices[index];
        let value = icl.values_ns[index];
        let generation_start = generation_start_ns[record];
        let tokens = output_tokens[record];
        if generation_start.is_nan() || value <= 0.0 || tokens.is_nan() || tokens < 1.0 {
            continue;
        }
        let count = positive_counts[record];
        if count == 0 {
            continue;
        }
        let interval_end = generation_start + relative_end;
        let interval_start = interval_end - value;
        let rate = ((tokens - 1.0) / count as f64) / value;
        events.push(SweepEvent::new(interval_start, rate));
        events.push(SweepEvent::new(interval_end, -rate));
    }
    sweep_line_cumsum(events)
}

struct ChunkEvents {
    events: Vec<SweepEvent>,
    has_valid_chunks: Vec<bool>,
}

fn build_chunk_events(
    generation_start_ns: &[f64],
    end_ns: &[f64],
    output_tokens: &[f64],
    icl: IclSeries<'_>,
) -> ChunkEvents {
    validate_record_indices(icl, output_tokens.len());
    let mut counts = vec![0_usize; output_tokens.len()];
    for &record in icl.record_indices {
        counts[record] += 1;
    }
    let relative_cumsum = relative_cumsums(icl);
    let mut events = Vec::with_capacity(icl.values_ns.len());
    let mut has_valid_chunks = vec![false; output_tokens.len()];
    for (index, &relative_end) in relative_cumsum.iter().enumerate() {
        let record = icl.record_indices[index];
        let value = icl.values_ns[index];
        let generation_start = generation_start_ns[record];
        let tokens = output_tokens[record];
        if generation_start.is_nan() || value < 0.0 || tokens.is_nan() || tokens < 1.0 {
            continue;
        }
        let count = counts[record];
        if count == 0 {
            continue;
        }
        let mut timestamp = generation_start + relative_end;
        if !end_ns[record].is_nan() && timestamp >= end_ns[record] {
            timestamp = next_down(end_ns[record]);
        }
        events.push(SweepEvent::new(timestamp, (tokens - 1.0) / count as f64));
        has_valid_chunks[record] = true;
    }
    ChunkEvents {
        events,
        has_valid_chunks,
    }
}

fn relative_cumsums(icl: IclSeries<'_>) -> Vec<f64> {
    let mut global = Vec::with_capacity(icl.values_ns.len());
    let mut cumulative = 0.0;
    for value in icl.values_ns {
        cumulative += *value;
        global.push(cumulative);
    }
    icl.record_indices
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let offset = icl.offsets[*record];
            assert!(
                offset <= index,
                "ICL offsets must point to the record's first value"
            );
            let before = offset
                .checked_sub(1)
                .and_then(|previous| global.get(previous))
                .copied()
                .unwrap_or(0.0);
            global[index] - before
        })
        .collect()
}

fn validate_record_indices(icl: IclSeries<'_>, record_count: usize) {
    assert!(
        icl.record_indices
            .iter()
            .all(|record| *record < record_count)
    );
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
        let records = [0, 0, 0];
        let offsets = [0];
        let curve =
            throughput_sweep_line_icl(&[0.0], &[6.0], IclSeries::new(&values, &records, &offsets));
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
        let records = [0, 0];
        let offsets = [0];
        let series = IclSeries::new(&values, &records, &offsets);
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
        let records = [0];
        let offsets = [0];
        let end = 1.7e18;
        let curve = tokens_in_flight_sweep_line_icl(
            &[end - 2_048.0],
            &[end - 1_024.0],
            &[end],
            &[10.0],
            &[2.0],
            IclSeries::new(&values, &records, &offsets),
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
        let records = [0, 0, 1, 1];
        let offsets = [0, 2];
        let curve = tokens_in_flight_sweep_line_icl(
            &[0.0, f64::NAN],
            &[10.0, f64::NAN],
            &[110.0, 120.0],
            &[100.0, f64::NAN],
            &[20.0, 40.0],
            IclSeries::new(&values, &records, &offsets),
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
        let records = [0, 0, 1, 1];
        let offsets = [0, 2];
        let curve = tokens_in_flight_sweep_line_icl(
            &[0.0, 5.0],
            &[10.0, 15.0],
            &[110.0, 120.0],
            &[100.0, 50.0],
            &[20.0, f64::NAN],
            IclSeries::new(&values, &records, &offsets),
        );
        assert_eq!(curve.final_value(), 0.0);
        assert!(curve.values().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn multi_record_icl_curves_match_independent_reference_enumeration() {
        // This is the native twin of the brute-force oracle in
        // `tests/unit/analysis/test_sweep.py:1389-1622`. It deliberately uses
        // overlapping records, unequal gaps, and a zero-duration gap so the
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
        let record_indices = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let offsets = [0, 4, 8];
        let series = IclSeries::new(&icl_values, &record_indices, &offsets);

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
