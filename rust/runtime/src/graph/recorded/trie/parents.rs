// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Linear-time longest-prefix content-parent resolution.

use std::collections::HashMap;

use super::TrieNode;
use crate::graph::recorded::BlockHash;

#[derive(Default)]
struct State {
    children: HashMap<BlockHash, usize>,
    latest_terminal: Option<usize>,
    earliest_passer: Option<usize>,
}

pub(super) fn resolve_content_parents(nodes: &mut [TrieNode]) {
    let mut states = vec![State::default()];
    for (node_index, node) in nodes.iter_mut().enumerate() {
        let hashes = node.request.hash_ids.clone();
        let mut state = 0_usize;
        let mut matched = 0_usize;
        let mut best_full = None;
        let mut best_partial = None;
        for hash in &hashes {
            let Some(next) = states[state].children.get(hash).copied() else {
                break;
            };
            state = next;
            matched += 1;
            if let Some(owner) = states[state].latest_terminal {
                best_full = Some(owner);
            }
            if let Some(owner) = states[state].earliest_passer {
                best_partial = Some(owner);
            }
        }
        node.content_parent = best_full.or(best_partial);
        if hashes.is_empty() {
            continue;
        }
        for hash in hashes.into_iter().skip(matched) {
            let next = if let Some(next) = states[state].children.get(&hash).copied() {
                next
            } else {
                let next = states.len();
                states.push(State::default());
                states[state].children.insert(hash, next);
                next
            };
            state = next;
            if states[state].earliest_passer.is_none() {
                states[state].earliest_passer = Some(node_index);
            }
        }
        states[state].latest_terminal = Some(node_index);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::graph::recorded::trie::RecordedRequest;

    fn node(id: &str, order: usize, hashes: &[i64]) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "c".into(),
                turn_index: order,
                order,
                hash_ids: hashes.iter().map(|value| i128::from(*value)).collect(),
                input_tokens: hashes.len(),
                output_tokens: 1,
                start_seconds: order as f64,
                duration_seconds: 1.0,
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
            warped_start: order as f64,
            rank: 0,
        }
    }

    #[test]
    fn full_prefix_prefers_latest_and_partial_prefers_earliest() {
        let mut nodes = vec![
            node("a", 0, &[1, 2]),
            node("b", 1, &[1, 2]),
            node("full", 2, &[1, 2, 3]),
            node("partial", 3, &[1, 9]),
        ];
        resolve_content_parents(&mut nodes);
        assert_eq!(nodes[2].content_parent, Some(1));
        assert_eq!(nodes[3].content_parent, Some(0));
    }
}
