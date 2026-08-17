// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active-idle warp and interval-order dependency derivation.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::graph::model::{START_NODE_ID, StaticEdge};

use super::TrieNode;

/// How idle gaps are measured before capping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdleWarpMode {
    /// Gap between a request start and the running max end of all prior requests
    /// (busy-period aware). Default for dynamo/aiperf recorded traces.
    BusyPeriod,
    #[cfg(test)]
    /// Gap between consecutive request *starts*, ignoring durations. Byte-exact
    /// match for the Python WEKA oracle's `_IdleGapTimeWarp`.
    StartToStart,
}

pub(super) fn apply_idle_warp(nodes: &mut [TrieNode], cap: Option<f64>, mode: IdleWarpMode) {
    let Some(cap) = cap else {
        return;
    };
    let mut intervals = nodes
        .iter()
        .map(|node| (node.request.start_seconds, node.request.raw_end()))
        .collect::<Vec<_>>();
    intervals.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let mut cuts = Vec::new();
    if let Some(first) = intervals.first().copied() {
        let mut running_end = first.1;
        #[cfg(test)]
        let mut prev_start = first.0;
        let mut cumulative = 0.0;
        for (start, end) in intervals.into_iter().skip(1) {
            let idle = match mode {
                IdleWarpMode::BusyPeriod => start - running_end,
                #[cfg(test)]
                IdleWarpMode::StartToStart => start - prev_start,
            };
            if idle > cap {
                cumulative += idle - cap;
                cuts.push((start, cumulative));
            }
            running_end = running_end.max(end);
            #[cfg(test)]
            let () = {
                prev_start = start;
            };
        }
    }
    for node in nodes {
        let shift = cuts
            .iter()
            .take_while(|(start, _)| node.request.start_seconds >= *start)
            .last()
            .map_or(0.0, |(_, shift)| *shift);
        node.warped_start = node.request.start_seconds - shift;
    }
}

pub(super) fn compute_ranks(nodes: &mut [TrieNode]) {
    let mut indices = (0..nodes.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| compare_nodes(&nodes[*left], &nodes[*right]));
    for (rank, index) in indices.into_iter().enumerate() {
        nodes[index].rank = rank;
    }
}

fn compare_nodes(left: &TrieNode, right: &TrieNode) -> Ordering {
    left.warped_start
        .total_cmp(&right.warped_start)
        .then_with(|| left.end().total_cmp(&right.end()))
        .then_with(|| left.request.node_id.cmp(&right.request.node_id))
}

fn async_excluded(candidate: &TrieNode, target: &TrieNode) -> bool {
    !candidate
        .request
        .async_ancestors
        .is_subset(&target.request.async_ancestors)
}

pub(super) fn build_interval_edges(nodes: &[TrieNode]) -> HashMap<String, Vec<StaticEdge>> {
    let mut by_rank = (0..nodes.len()).collect::<Vec<_>>();
    by_rank.sort_by_key(|index| nodes[*index].rank);
    let mut output = HashMap::new();
    for (target_index, target) in nodes.iter().enumerate() {
        let candidates = by_rank
            .iter()
            .copied()
            .filter(|candidate_index| {
                *candidate_index != target_index
                    && nodes[*candidate_index].rank < target.rank
                    && nodes[*candidate_index].request.raw_end() <= target.request.start_seconds
                    && !async_excluded(&nodes[*candidate_index], target)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            output.insert(
                target.request.node_id.clone(),
                vec![StaticEdge {
                    source: START_NODE_ID.into(),
                    target: target.request.node_id.clone(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: Some(target.warped_start * 1_000_000.0),
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                }],
            );
            continue;
        }
        let frontier = candidates
            .iter()
            .enumerate()
            .filter_map(|(position, candidate_index)| {
                let candidate = &nodes[*candidate_index];
                let covered = candidates[position + 1..].iter().any(|later_index| {
                    let later = &nodes[*later_index];
                    candidate.request.raw_end() <= later.request.start_seconds
                        && !async_excluded(candidate, later)
                });
                (!covered).then_some(*candidate_index)
            })
            .collect::<Vec<_>>();
        let mut binding = frontier[0];
        for index in frontier.iter().copied().skip(1) {
            if nodes[index].end() > nodes[binding].end() {
                binding = index;
            }
        }
        output.insert(
            target.request.node_id.clone(),
            frontier
                .into_iter()
                .map(|index| {
                    let predecessor = &nodes[index];
                    StaticEdge {
                        source: predecessor.request.node_id.clone(),
                        target: target.request.node_id.clone(),
                        delay_after_predecessor_us: Some(if index == binding {
                            (target.warped_start - predecessor.end()).max(0.0) * 1_000_000.0
                        } else {
                            0.0
                        }),
                        min_start_delay_us: None,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    }
                })
                .collect(),
        );
    }
    output
}

pub(super) fn apply_start_anchors(
    nodes: &[TrieNode],
    edges: &mut HashMap<String, Vec<StaticEdge>>,
) {
    let by_id = nodes
        .iter()
        .map(|node| (node.request.node_id.as_str(), node))
        .collect::<HashMap<_, _>>();
    for node in nodes {
        let Some(parent_id) = node.request.causal_parent_id.as_deref() else {
            continue;
        };
        let Some(parent) = by_id.get(parent_id).copied() else {
            continue;
        };
        if !(parent.request.start_seconds <= node.request.start_seconds
            && node.request.start_seconds < parent.request.raw_end())
        {
            continue;
        }
        let delay_us = (node.warped_start - parent.warped_start).max(0.0) * 1_000_000.0;
        let first_token_delay = parent.request.ttft_seconds.and_then(|ttft| {
            (node.request.start_seconds - parent.request.start_seconds >= ttft)
                .then_some((delay_us - ttft * 1_000_000.0).max(0.0))
        });
        edges.insert(
            node.request.node_id.clone(),
            vec![StaticEdge {
                source: parent.request.node_id.clone(),
                target: node.request.node_id.clone(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: Some(delay_us),
                delay_after_predecessor_first_token_us: first_token_delay,
            }],
        );
    }
}

#[cfg(test)]
mod parity_tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::graph::recorded::trie::RecordedRequest;

    fn node(id: &str, order: usize, start: f64, duration: f64) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "chain".into(),
                turn_index: order,
                order,
                hash_ids: vec![(order + 1) as i128],
                input_tokens: 16,
                output_tokens: 1,
                start_seconds: start,
                duration_seconds: duration,
                model: None,
                streaming: false,
                ttft_seconds: None,
                causal_parent_id: None,
                async_ancestors: HashSet::new(),
                max_tokens: 1,
                extra_headers: BTreeMap::new(),
                adapter_metadata: BTreeMap::new(),
                explicit_tags: None,
                block_lens: None,
            },
            content_parent: None,
            warped_start: start,
            rank: 0,
        }
    }

    #[test]
    fn idle_warp_cuts_only_union_idle_and_accumulates_multiple_gaps() {
        let mut nodes = vec![
            node("a", 0, 0.0, 5.0),
            node("overlap", 1, 2.0, 1.0),
            node("c", 2, 100.0, 1.0),
            node("d", 3, 200.0, 1.0),
        ];
        apply_idle_warp(&mut nodes, Some(10.0), IdleWarpMode::BusyPeriod);
        assert_eq!(nodes[0].warped_start, 0.0);
        assert_eq!(nodes[1].warped_start, 2.0);
        assert_eq!(nodes[2].warped_start, 15.0);
        assert_eq!(nodes[3].warped_start, 26.0);
        assert_eq!(nodes[0].end(), 5.0, "active duration is never compressed");
    }

    #[test]
    fn start_to_start_mode_ignores_durations_matching_python_weka_oracle() {
        // Same nodes as the busy-period test. StartToStart measures gaps between
        // consecutive request *starts* (not the running max end), so the active
        // duration of `a` (end 5.0) does not shrink the 0->2 or 2->100 gap.
        //   starts: 0, 2, 100, 200
        //   0->2   : gap 2   (<=10, no cut)
        //   2->100 : gap 98  (excess 88, cumulative 88, cut at 100)
        //   100->200: gap 100 (excess 90, cumulative 178, cut at 200)
        let mut nodes = vec![
            node("a", 0, 0.0, 5.0),
            node("overlap", 1, 2.0, 1.0),
            node("c", 2, 100.0, 1.0),
            node("d", 3, 200.0, 1.0),
        ];
        apply_idle_warp(&mut nodes, Some(10.0), IdleWarpMode::StartToStart);
        assert_eq!(nodes[0].warped_start, 0.0);
        assert_eq!(nodes[1].warped_start, 2.0);
        assert_eq!(nodes[2].warped_start, 12.0);
        assert_eq!(nodes[3].warped_start, 22.0);
    }

    #[test]
    fn overlapping_causal_child_gets_dispatch_and_first_token_anchors() {
        let mut parent = node("parent", 0, 0.0, 5.0);
        parent.request.streaming = true;
        parent.request.ttft_seconds = Some(1.0);
        let mut child = node("child", 1, 2.0, 1.0);
        child.request.causal_parent_id = Some("parent".into());
        let mut nodes = vec![parent, child];
        compute_ranks(&mut nodes);
        let mut edges = build_interval_edges(&nodes);
        apply_start_anchors(&nodes, &mut edges);
        let edge = &edges["child"][0];
        assert_eq!(edge.source, "parent");
        assert_eq!(edge.delay_after_predecessor_start_us, Some(2_000_000.0));
        assert_eq!(
            edge.delay_after_predecessor_first_token_us,
            Some(1_000_000.0)
        );
        assert_eq!(edge.delay_after_predecessor_us, None);
    }

    #[test]
    fn interval_frontier_preserves_and_fan_in_and_async_exclusion() {
        let a = node("a", 0, 0.0, 4.0);
        let b = node("b", 1, 1.0, 2.0);
        let target = node("target", 2, 6.0, 1.0);
        let mut nodes = vec![a, b, target];
        compute_ranks(&mut nodes);
        let edges = build_interval_edges(&nodes);
        let incoming = &edges["target"];
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].source, "a");
        assert_eq!(incoming[0].delay_after_predecessor_us, Some(2_000_000.0));
        assert_eq!(incoming[1].source, "b");
        assert_eq!(incoming[1].delay_after_predecessor_us, Some(0.0));

        nodes[0]
            .request
            .async_ancestors
            .insert("fire-and-forget".into());
        let edges = build_interval_edges(&nodes);
        assert_eq!(
            edges["target"]
                .iter()
                .map(|edge| edge.source.as_str())
                .collect::<Vec<_>>(),
            ["b"]
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::graph::recorded::trie::RecordedRequest;

    fn node(id: &str, order: usize, start: f64, duration: f64) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "chain".into(),
                turn_index: order,
                order,
                hash_ids: vec![order as i128],
                input_tokens: 1,
                output_tokens: 1,
                start_seconds: start,
                duration_seconds: duration,
                model: None,
                streaming: false,
                ttft_seconds: None,
                causal_parent_id: None,
                async_ancestors: HashSet::new(),
                max_tokens: 1,
                extra_headers: BTreeMap::new(),
                adapter_metadata: BTreeMap::new(),
                explicit_tags: None,
                block_lens: None,
            },
            content_parent: None,
            warped_start: start,
            rank: 0,
        }
    }

    #[test]
    fn idle_warp_caps_only_dead_time_after_the_active_interval() {
        let mut nodes = vec![node("a", 0, 0.0, 2.0), node("b", 1, 137_124.0, 1.0)];
        apply_idle_warp(&mut nodes, Some(60.0), IdleWarpMode::BusyPeriod);
        assert_eq!(nodes[0].warped_start, 0.0);
        assert_eq!(nodes[1].warped_start, 62.0);
        assert_eq!(nodes[0].end(), 2.0);
        assert_eq!(nodes[1].end(), 63.0);
    }

    // `None` disables warping, an
    // over-cap idle gap collapses to the cap, and a sub-cap gap is untouched.
    #[test]
    fn idle_warp_tri_state_matches_python_cap_semantics() {
        // Over-cap gap: intervals (0,2),(100,101); idle 100-2=98 > 60 cap,
        // excess 98-60=38, so b warps 100-38=62.
        let mut over = vec![node("a", 0, 0.0, 2.0), node("b", 1, 100.0, 1.0)];
        apply_idle_warp(&mut over, Some(60.0), IdleWarpMode::BusyPeriod);
        assert_eq!(over[0].warped_start, 0.0);
        assert_eq!(over[1].warped_start, 62.0);

        // `None` leaves warped_start at the raw start_seconds.
        let mut disabled = vec![node("a", 0, 0.0, 2.0), node("b", 1, 100.0, 1.0)];
        apply_idle_warp(&mut disabled, None, IdleWarpMode::BusyPeriod);
        assert_eq!(disabled[0].warped_start, 0.0);
        assert_eq!(disabled[1].warped_start, 100.0);

        // Sub-cap gap: idle 50-2=48 <= 60 cap → no cut, raw passthrough.
        let mut under = vec![node("a", 0, 0.0, 2.0), node("b", 1, 50.0, 1.0)];
        apply_idle_warp(&mut under, Some(60.0), IdleWarpMode::BusyPeriod);
        assert_eq!(under[0].warped_start, 0.0);
        assert_eq!(under[1].warped_start, 50.0);
    }

    #[test]
    fn interval_frontier_keeps_concurrent_predecessors_and_binds_latest_end() {
        let mut nodes = vec![
            node("root", 0, 0.0, 1.0),
            node("left", 1, 1.2, 2.8),
            node("right", 2, 1.3, 3.7),
            node("join", 3, 5.2, 1.0),
        ];
        compute_ranks(&mut nodes);
        let incoming = build_interval_edges(&nodes);
        let join = &incoming["join"];
        assert_eq!(join.len(), 2);
        let left = join.iter().find(|edge| edge.source == "left").unwrap();
        let right = join.iter().find(|edge| edge.source == "right").unwrap();
        assert_eq!(left.delay_after_predecessor_us, Some(0.0));
        assert!((right.delay_after_predecessor_us.unwrap() - 200_000.0).abs() < 1.0e-6);
    }

    #[test]
    fn async_subtree_completion_does_not_join_the_launching_scope() {
        let root = node("root", 0, 0.0, 0.5);
        let mut child = node("async", 1, 0.6, 0.2);
        child.request.async_ancestors.insert("agent".into());
        let resume = node("resume", 2, 3.0, 0.5);
        let mut nodes = vec![root, child, resume];
        compute_ranks(&mut nodes);
        let incoming = build_interval_edges(&nodes);
        assert!(incoming["resume"].iter().all(|edge| edge.source != "async"));
    }

    #[test]
    fn overlapping_streaming_cause_gets_dispatch_and_first_token_anchors() {
        let mut parent = node("parent", 0, 0.0, 8.0);
        parent.request.streaming = true;
        parent.request.ttft_seconds = Some(1.5);
        let mut child = node("child", 1, 2.5, 1.0);
        child.request.causal_parent_id = Some("parent".into());
        let mut nodes = vec![parent, child];
        compute_ranks(&mut nodes);
        let mut incoming = build_interval_edges(&nodes);
        apply_start_anchors(&nodes, &mut incoming);
        let [edge] = incoming["child"].as_slice() else {
            panic!("overlapping cause must collapse to one start anchor")
        };
        assert_eq!(edge.source, "parent");
        assert_eq!(edge.delay_after_predecessor_start_us, Some(2_500_000.0));
        assert_eq!(
            edge.delay_after_predecessor_first_token_us,
            Some(1_000_000.0)
        );
        assert_eq!(edge.delay_after_predecessor_us, None);
    }
}
