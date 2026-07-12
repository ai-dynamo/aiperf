// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen block-role planning and content-addressed message emission.

use std::collections::HashMap;

use aiperf_dataset::{Handle, SegmentPool};
use serde::Serialize;

use super::TrieNode;
use crate::recorded::RecordedTraceError;
use crate::recorded::content::RecordedContentSynthesizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Assistant,
    User,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Assistant => "assistant",
            Self::User => "user",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BlockTag {
    role: Role,
    starts_message: bool,
}

#[derive(Debug)]
struct Geometry {
    lcp: usize,
    covered: usize,
    missing_full_tokens: usize,
}

fn geometry(previous: &TrieNode, current: &TrieNode, block_size: usize) -> Geometry {
    geometry_from_hashes(
        &previous.request.hash_ids,
        &current.request.hash_ids,
        current.request.input_tokens,
        block_size,
    )
}

fn geometry_from_hashes(
    previous: &[crate::recorded::BlockHash],
    current: &[crate::recorded::BlockHash],
    input_tokens: usize,
    block_size: usize,
) -> Geometry {
    let lcp = previous
        .iter()
        .zip(current)
        .take_while(|(left, right)| left == right)
        .count();
    let full = input_tokens / block_size;
    Geometry {
        lcp,
        covered: current.len().min(full),
        missing_full_tokens: full
            .saturating_sub(current.len())
            .saturating_mul(block_size),
    }
}

pub(super) fn compute_assistant_caps(
    nodes: &[TrieNode],
    block_size: usize,
) -> HashMap<usize, usize> {
    let mut caps = HashMap::<usize, usize>::new();
    let mut tiles = vec![Vec::<usize>::new(); nodes.len()];
    let mut effective = vec![0_usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let Some(parent_index) = node.content_parent else {
            let geo = geometry_from_hashes(
                &[],
                &node.request.hash_ids,
                node.request.input_tokens,
                block_size,
            );
            tiles[index] = vec![index; geo.covered];
            continue;
        };
        let parent_tiles = &tiles[parent_index];
        let geo = geometry(&nodes[parent_index], node, block_size);
        let inherited = geo.lcp.min(parent_tiles.len()).min(geo.covered);
        effective[index] = inherited;
        let new_count = geo.covered.saturating_sub(inherited);
        if new_count == 0 && geo.missing_full_tokens == 0 && inherited >= 1 {
            let owner = parent_tiles[inherited - 1];
            if nodes[owner].content_parent.is_some() {
                let bound = (inherited - 1).saturating_sub(effective[owner]);
                caps.entry(owner)
                    .and_modify(|current| *current = (*current).min(bound))
                    .or_insert(bound);
            }
        }
        let mut node_tiles = parent_tiles[..inherited].to_vec();
        node_tiles.extend(std::iter::repeat_n(index, new_count));
        tiles[index] = node_tiles;
    }
    caps
}

pub(super) fn assign_block_tags(
    nodes: &[TrieNode],
    block_size: usize,
    caps: &HashMap<usize, usize>,
) -> Result<(Vec<Vec<BlockTag>>, Vec<usize>), RecordedTraceError> {
    let mut all_tags: Vec<Vec<BlockTag>> = vec![Vec::new(); nodes.len()];
    let mut inherited_by_node = vec![0_usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let (parent_hashes, parent_output, parent_tags) = match node.content_parent {
            Some(parent) => (
                nodes[parent].request.hash_ids.as_slice(),
                nodes[parent].request.output_tokens,
                all_tags[parent].as_slice(),
            ),
            None => (&[][..], 0, &[][..]),
        };
        let geo = geometry_from_hashes(
            parent_hashes,
            &node.request.hash_ids,
            node.request.input_tokens,
            block_size,
        );
        let inherited = geo.lcp.min(parent_tags.len()).min(geo.covered);
        inherited_by_node[index] = inherited;
        let parent_has_user = parent_tags[..inherited]
            .iter()
            .any(|tag| tag.role == Role::User);
        let new_count = geo.covered.saturating_sub(inherited);
        let mut assistant = if parent_has_user && parent_output > 0 {
            parent_output.div_ceil(block_size).min(new_count)
        } else {
            0
        };
        if let Some(cap) = caps.get(&index) {
            assistant = assistant.min(*cap);
        }
        if assistant == new_count && assistant > 0 {
            assistant -= 1;
        }
        let mut tags = parent_tags[..inherited].to_vec();
        for offset in 0..new_count {
            let role = if offset < assistant {
                Role::Assistant
            } else {
                Role::User
            };
            tags.push(BlockTag {
                role,
                starts_message: offset == 0
                    || (offset > 0 && tags.last().is_some_and(|previous| previous.role != role)),
            });
        }
        if tags.len() != geo.covered {
            return Err(RecordedTraceError(format!(
                "node {:?}: frozen tag count {} differs from covered block count {}",
                node.request.node_id,
                tags.len(),
                geo.covered
            )));
        }
        all_tags[index] = tags;
    }
    Ok((all_tags, inherited_by_node))
}

pub(super) fn emit_prompt(
    node: &TrieNode,
    tags: &[BlockTag],
    block_size: usize,
    hash_scope: Option<&str>,
    tail_scope: &str,
    content: &mut dyn RecordedContentSynthesizer,
    pool: &mut SegmentPool,
) -> Result<Vec<Handle>, RecordedTraceError> {
    if tags.is_empty() && node.request.input_tokens > 0 {
        let tokens = content.tail_tokens(
            node.request.input_tokens,
            &format!("{tail_scope}:{}:tiny", node.request.node_id),
        );
        let text = content.decode(&tokens)?;
        return Ok(vec![intern_message(pool, None, "user", &text, &tokens)?]);
    }

    let mut groups = Vec::<(Role, Vec<usize>)>::new();
    for (index, tag) in tags.iter().copied().enumerate() {
        if tag.starts_message || groups.is_empty() {
            groups.push((tag.role, vec![index]));
        } else {
            groups
                .last_mut()
                .expect("non-empty message groups")
                .1
                .push(index);
        }
    }
    let mut handles = Vec::with_capacity(groups.len());
    let mut parent = None;
    let mut assembled = 0_usize;
    for (role, blocks) in groups {
        let mut tokens = Vec::with_capacity(blocks.len().saturating_mul(block_size));
        for block in blocks {
            tokens.extend(content.block_tokens(
                &node.request.hash_ids[block..block + 1],
                block_size,
                hash_scope,
            )?);
        }
        assembled = assembled.saturating_add(tokens.len());
        let text = content.decode(&tokens)?;
        parent = Some(intern_message(pool, parent, role.as_str(), &text, &tokens)?);
        handles.push(parent.expect("message handle assigned"));
    }
    let expected = tags.len().saturating_mul(block_size);
    if assembled != expected {
        return Err(RecordedTraceError(format!(
            "node {:?}: reconstructed {assembled} tokens, expected covered-count {expected}",
            node.request.node_id
        )));
    }
    Ok(handles)
}

#[derive(Serialize)]
struct MessageWire<'a> {
    role: &'a str,
    content: &'a str,
}

pub(super) fn intern_message(
    pool: &mut SegmentPool,
    parent: Option<Handle>,
    role: &str,
    content: &str,
    tokens: &[u32],
) -> Result<Handle, RecordedTraceError> {
    let wire = serde_json::to_vec(&MessageWire { role, content })?;
    pool.intern_message(parent, role, wire, tokens.to_vec().into_boxed_slice())
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use num_bigint::BigInt;

    use super::*;
    use crate::recorded::trie::RecordedRequest;

    fn node(id: &str, order: usize, hashes: &[i64], input: usize, output: usize) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "chain".into(),
                turn_index: order,
                order,
                hash_ids: hashes.iter().copied().map(BigInt::from).collect(),
                input_tokens: input,
                output_tokens: output,
                start_seconds: order as f64,
                duration_seconds: 1.0,
                model: None,
                streaming: false,
                ttft_seconds: None,
                causal_parent_id: None,
                async_ancestors: HashSet::new(),
                max_tokens: output.max(1),
                extra_headers: BTreeMap::new(),
                adapter_metadata: BTreeMap::new(),
            },
            content_parent: None,
            warped_start: order as f64,
            rank: 0,
        }
    }

    #[test]
    fn geometry_clamps_covered_blocks_and_counts_only_missing_whole_blocks() {
        let previous = [BigInt::from(1)];
        let current = [BigInt::from(1), BigInt::from(2), BigInt::from(3)];
        let over_shared = geometry_from_hashes(&previous, &current, 5, 2);
        assert_eq!(over_shared.lcp, 1);
        assert_eq!(over_shared.covered, 2);
        assert_eq!(over_shared.missing_full_tokens, 0);

        let under_covered = geometry_from_hashes(&[], &current[..2], 6, 2);
        assert_eq!(under_covered.covered, 2);
        assert_eq!(under_covered.missing_full_tokens, 2);
    }

    #[test]
    fn newly_created_all_assistant_region_keeps_a_trailing_user_block() {
        let mut nodes = vec![
            node("parent", 0, &[1], 2, 6),
            node("child", 1, &[1, 2, 3], 6, 1),
        ];
        nodes[1].content_parent = Some(0);
        let caps = compute_assistant_caps(&nodes, 2);
        let (tags, inherited) = assign_block_tags(&nodes, 2, &caps).unwrap();
        assert_eq!(inherited[1], 1);
        assert_eq!(tags[1][1].role, Role::Assistant);
        assert_eq!(tags[1][2].role, Role::User);
        assert!(tags[1][1].starts_message);
        assert!(tags[1][2].starts_message);
    }

    #[test]
    fn undercovered_parent_exposes_shared_but_unfrozen_blocks_as_new() {
        let mut nodes = vec![
            node("parent", 0, &[1, 2, 3, 4], 4, 2),
            node("child", 1, &[1, 2, 3, 4, 5], 10, 1),
        ];
        nodes[1].content_parent = Some(0);
        let caps = compute_assistant_caps(&nodes, 2);
        let (tags, inherited) = assign_block_tags(&nodes, 2, &caps).unwrap();
        assert_eq!(tags[0].len(), 2);
        assert_eq!(inherited[1], 2);
        assert_eq!(tags[1].len(), 5);
    }
}
