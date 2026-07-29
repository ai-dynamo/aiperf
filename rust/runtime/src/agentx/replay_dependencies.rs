// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recorded interval-order dependency inference, ported from
//! `src/aiperf/timing/replay_dependencies.py`.
//!
//! Per-stream ordering is owned by normal conversation replay; this infers the
//! cross-stream completion frontier each request must join. For every OTHER
//! stream, a request depends on that stream's latest request known to have
//! completed by its recorded start. Overlapping intervals create no edge; this
//! captures transitive overlap precisely. The stateful `ReplayBarrierCoordinator`
//! (Credit/TurnToSend-coupled) is not ported here — this is the pure core.

use std::collections::HashMap;

/// Stable dataset identity for one replayed request (Python `ReplayTurnKey`).
/// Ordered by `(conversation_id, turn_index)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReplayTurnKey {
    /// Template conversation id this request belongs to.
    pub conversation_id: String,
    /// Zero-based turn position within the conversation.
    pub turn_index: i64,
}

/// Completed prefix of one replay stream at a phase boundary
/// (Python `ReplayResumeBoundary`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReplayResumeBoundary {
    /// Template conversation id of the replay stream.
    pub conversation_id: String,
    /// Index of the first turn not yet completed at the phase boundary.
    pub next_turn_index: i64,
}

/// One request interval on a logical replay stream (Python `RecordedTurnInterval`).
#[derive(Debug, Clone)]
pub struct RecordedTurnInterval {
    /// Dataset identity (conversation + turn).
    pub key: ReplayTurnKey,
    /// Logical stream this interval belongs to (root or subagent chain).
    pub stream_id: String,
    /// Recorded wall-clock start offset in ms; `None` when unknown.
    pub start_ms: Option<f64>,
    /// Recorded server processing duration in ms; `None` when unknown.
    pub api_time_ms: Option<f64>,
}

impl RecordedTurnInterval {
    /// `[start, end]` using the Weka duration fallback (missing/negative/non-finite
    /// duration → zero-width); `None` when the start is missing/non-finite.
    pub fn normalized_interval(&self) -> Option<(f64, f64)> {
        let start = self.start_ms?;
        if !start.is_finite() {
            return None;
        }
        let duration = match self.api_time_ms {
            Some(d) if d.is_finite() && d >= 0.0 => d,
            _ => 0.0,
        };
        Some((start, start + duration))
    }
}

/// Infer the recorded completion frontier each request must join (Python
/// `infer_cross_stream_predecessors`).
pub fn infer_cross_stream_predecessors(
    intervals: &[RecordedTurnInterval],
) -> HashMap<ReplayTurnKey, Vec<ReplayTurnKey>> {
    // (interval index -> (start, end)); group by stream.
    let mut by_stream: HashMap<String, Vec<(usize, f64, f64)>> = HashMap::new();
    for (i, interval) in intervals.iter().enumerate() {
        if let Some((s, e)) = interval.normalized_interval() {
            by_stream
                .entry(interval.stream_id.clone())
                .or_default()
                .push((i, s, e));
        }
    }

    let mut deps: HashMap<ReplayTurnKey, Vec<ReplayTurnKey>> = HashMap::new();
    for target in intervals {
        let target_start = match target.normalized_interval() {
            Some((s, _)) => s,
            None => {
                deps.insert(target.key.clone(), Vec::new());
                continue;
            }
        };
        // Frontier: latest completed interval on each OTHER stream.
        let mut frontier: Vec<(usize, f64, f64)> = Vec::new();
        for (stream_id, candidates) in &by_stream {
            if *stream_id == target.stream_id {
                continue;
            }
            let latest = candidates
                .iter()
                .filter(|(_, s, e)| *s < target_start && *e <= target_start)
                .max_by(|a, b| {
                    // (end, start, key) ascending.
                    (a.2, a.1)
                        .partial_cmp(&(b.2, b.1))
                        .unwrap()
                        .then_with(|| intervals[a.0].key.cmp(&intervals[b.0].key))
                });
            if let Some(&l) = latest {
                frontier.push(l);
            }
        }
        // Drop a frontier member fully completed-before another frontier member.
        let mut preds: Vec<ReplayTurnKey> = Vec::new();
        for (i, c) in frontier.iter().enumerate() {
            let dominated = frontier
                .iter()
                .enumerate()
                .any(|(j, later)| j != i && c.1 < later.1 && c.2 <= later.1);
            if !dominated {
                preds.push(intervals[c.0].key.clone());
            }
        }
        preds.sort();
        deps.insert(target.key.clone(), preds);
    }
    deps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(conv: &str, ti: i64, sid: &str, start: f64, api: f64) -> RecordedTurnInterval {
        RecordedTurnInterval {
            key: ReplayTurnKey {
                conversation_id: conv.into(),
                turn_index: ti,
            },
            stream_id: sid.into(),
            start_ms: Some(start),
            api_time_ms: Some(api),
        }
    }
    fn k(conv: &str, ti: i64) -> ReplayTurnKey {
        ReplayTurnKey {
            conversation_id: conv.into(),
            turn_index: ti,
        }
    }

    #[test]
    fn matches_python_golden() {
        // A: a0[0,5], a1[5,10]; B: b0[10,20]; C: c0[6,7].
        let ivs = vec![
            iv("A", 0, "A", 0.0, 5.0),
            iv("A", 1, "A", 5.0, 5.0),
            iv("B", 0, "B", 10.0, 10.0),
            iv("C", 0, "C", 6.0, 1.0),
        ];
        let d = infer_cross_stream_predecessors(&ivs);
        assert_eq!(d[&k("A", 0)], vec![]);
        assert_eq!(d[&k("A", 1)], vec![]);
        assert_eq!(d[&k("B", 0)], vec![k("A", 1), k("C", 0)]);
        assert_eq!(d[&k("C", 0)], vec![k("A", 0)]);
    }

    #[test]
    fn missing_start_yields_no_deps() {
        let ivs = vec![RecordedTurnInterval {
            key: k("A", 0),
            stream_id: "A".into(),
            start_ms: None,
            api_time_ms: Some(5.0),
        }];
        let d = infer_cross_stream_predecessors(&ivs);
        assert_eq!(d[&k("A", 0)], vec![]);
    }
}
