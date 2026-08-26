// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-only filter-then-cap trace selection, ported from
//! `src/aiperf/dataset/loader/selection.py`.
//!
//! * FILTER: drop a candidate whose peak context exceeds `max_context_length`.
//! * CAP: keep the first `num_dataset_entries` *eligible* candidates.
//!
//! Filtering happens before the cap counts a slot: an oversized row is skipped
//! without consuming an entry, so the loaded pool always reaches the requested
//! entry count when enough eligible candidates exist.

/// Tally of one filter-then-cap selection pass (Python `SelectionStats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectionStats {
    /// Candidates examined.
    pub scanned: u64,
    /// Candidates dropped for exceeding `max_context_length`.
    pub rejected_by_maxctx: u64,
    /// Largest peak context observed across all scanned candidates.
    pub largest_observed: i64,
    /// Smallest peak context observed across all scanned candidates.
    pub smallest_observed: Option<i64>,
    /// Candidates that passed the filter.
    pub eligible: u64,
    /// Candidates actually kept (`== eligible` unless the cap truncated).
    pub loaded: u64,
}

/// Filter `candidates` (each `(item, peak_context)`, in deterministic scan
/// order) by peak context, then cap to the first `num_dataset_entries` eligible
/// items. Scanning stops once the cap is filled (mirrors the lazy Python
/// iterator). Returns the kept items and the pass stats.
pub fn filter_then_cap<T>(
    candidates: impl IntoIterator<Item = (T, i64)>,
    num_dataset_entries: Option<usize>,
    max_context_length: Option<i64>,
) -> (Vec<T>, SelectionStats) {
    let mut stats = SelectionStats::default();
    let mut kept: Vec<T> = Vec::new();
    for (item, peak) in candidates {
        stats.scanned += 1;
        if peak > stats.largest_observed {
            stats.largest_observed = peak;
        }
        stats.smallest_observed = Some(
            stats
                .smallest_observed
                .map_or(peak, |smallest| smallest.min(peak)),
        );
        if let Some(max) = max_context_length
            && peak > max
        {
            stats.rejected_by_maxctx += 1;
            continue;
        }
        stats.eligible += 1;
        kept.push(item);
        if let Some(n) = num_dataset_entries
            && kept.len() >= n
        {
            break;
        }
    }
    stats.loaded = kept.len() as u64;
    (kept, stats)
}

/// Format the all-rejected context-cap error from one completed selection pass.
pub(crate) fn no_eligible_traces_error(
    source: &str,
    stats: SelectionStats,
    num_dataset_entries: Option<usize>,
    max_context_length: Option<i64>,
) -> String {
    let max_context_length = max_context_length
        .map(|value| value.to_string())
        .unwrap_or_else(|| "None".to_string());
    let num_dataset_entries = num_dataset_entries
        .map(|value| value.to_string())
        .unwrap_or_else(|| "None".to_string());
    let mut message = format!(
        "No eligible traces in {source} after filter-then-cap (scanned {}, --max-context-length={max_context_length}, --num-dataset-entries={num_dataset_entries}).",
        stats.scanned,
    );
    if let Some(smallest) = stats.smallest_observed {
        message.push_str(&format!(
            " Smallest trace requires {smallest} tokens; raise --max-context-length to at least that (e.g. --max-context-length {smallest}) to admit any trace."
        ));
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_drops_oversized_without_consuming_cap() {
        // peaks: 10, 100, 20, 30; max=50 drops the 100; cap=2 keeps first two eligible.
        let cands = vec![("a", 10), ("b", 100), ("c", 20), ("d", 30)];
        let (kept, stats) = filter_then_cap(cands, Some(2), Some(50));
        assert_eq!(kept, vec!["a", "c"]);
        assert_eq!(stats.scanned, 3); // stops after filling the cap at "c"
        assert_eq!(stats.rejected_by_maxctx, 1);
        assert_eq!(stats.largest_observed, 100);
        assert_eq!(stats.eligible, 2);
        assert_eq!(stats.loaded, 2);
    }

    #[test]
    fn no_limits_keeps_all() {
        let cands = vec![("a", 1), ("b", 2)];
        let (kept, stats) = filter_then_cap(cands, None, None);
        assert_eq!(kept, vec!["a", "b"]);
        assert_eq!(stats.loaded, 2);
        assert_eq!(stats.rejected_by_maxctx, 0);
    }

    #[test]
    fn records_the_smallest_observed_peak_including_zero() {
        let (_, stats) = filter_then_cap(vec![("zero", 0), ("high", 12), ("low", 8)], None, None);
        assert_eq!(stats.smallest_observed, Some(0));
    }
}
