// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral inputs and the statistics primitive for dataset analysis.
//!
//! This module holds the pure-logic foundation for the `--dry-run` dataset
//! report: neutral per-record and per-turn observation structs, plus
//! [`stat_summary`], a distribution summarizer used across the report sections.

/// A single observed request record, expressed in transport-neutral terms.
#[derive(Debug, Clone)]
pub struct AnalyzedRecord {
    /// Conversation the record belongs to.
    pub conversation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Request start time in nanoseconds.
    pub start_ns: i64,
    /// Request end time in nanoseconds.
    pub end_ns: i64,
    /// Admission time in nanoseconds, when available.
    pub admit_ns: Option<i64>,
    /// First-token time in nanoseconds, when available.
    pub first_token_ns: Option<i64>,
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Number of output tokens.
    pub output_tokens: u64,
    /// Per-token arrival times in nanoseconds.
    pub token_arrival_ns: Vec<i64>,
}

/// A single planned turn, expressed in transport-neutral terms.
#[derive(Debug, Clone)]
pub struct AnalyzedTurn {
    /// Conversation the turn belongs to.
    pub conversation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Maximum number of output tokens requested.
    pub max_output_tokens: u64,
    /// Inter-turn delay in milliseconds, when specified.
    pub delay_ms: Option<f64>,
    /// Content-block identifiers, when available.
    pub block_ids: Option<Vec<i64>>,
    /// System-prompt handle, when available.
    pub system_handle: Option<u64>,
}

/// Summary statistics over a set of finite samples.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StatSummary {
    /// Number of samples.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation (divide by N).
    pub std: f64,
    /// Minimum sample.
    pub min: f64,
    /// Maximum sample.
    pub max: f64,
    /// Sum of all samples.
    pub sum: f64,
    /// 1st percentile.
    pub p1: f64,
    /// 5th percentile.
    pub p5: f64,
    /// 10th percentile.
    pub p10: f64,
    /// 25th percentile.
    pub p25: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

/// Compute summary statistics over `values`. Percentiles use linear
/// interpolation between closest ranks on the sorted samples. Returns `None`
/// when `values` is empty. `values` must be finite.
pub fn stat_summary(values: &[f64]) -> Option<StatSummary> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / n as f64;
    let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let pct = |p: f64| -> f64 {
        if n == 1 {
            return sorted[0];
        }
        let rank = p * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
    };
    Some(StatSummary {
        count: n as u64,
        mean,
        std: var.sqrt(),
        min: sorted[0],
        max: sorted[n - 1],
        sum,
        p1: pct(0.01),
        p5: pct(0.05),
        p10: pct(0.10),
        p25: pct(0.25),
        p50: pct(0.50),
        p75: pct(0.75),
        p90: pct(0.90),
        p95: pct(0.95),
        p99: pct(0.99),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_summary_matches_hand_computed() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = stat_summary(&v).expect("non-empty");
        assert_eq!(s.count, 5);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(s.sum, 15.0);
        assert_eq!(s.mean, 3.0);
        assert_eq!(s.p50, 3.0);
        // population variance of 1..5 = 2.0 → std = sqrt(2)
        assert!((s.std - 2.0_f64.sqrt()).abs() < 1e-9);
        // p90 by linear interpolation on rank: idx = 0.90*(5-1)=3.6 → 4.0 + 0.6*(5-4)=4.6
        assert!((s.p90 - 4.6).abs() < 1e-9);
    }

    #[test]
    fn stat_summary_empty_is_none() {
        assert!(stat_summary(&[]).is_none());
    }
}
