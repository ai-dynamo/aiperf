// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen block-role planning and content-addressed message emission.

use std::collections::HashMap;

use crate::dataset::{Handle, SegmentPool};
use serde::Serialize;

use super::TrieNode;
use crate::graph::recorded::RecordedTraceError;
use crate::graph::recorded::content::RecordedContentSynthesizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Role {
    Assistant,
    User,
    System,
    Tool,
    Tools,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Assistant => "assistant",
            Self::User => "user",
            Self::System => "system",
            Self::Tool => "tool",
            Self::Tools => "tools",
        }
    }

    /// Map an authored role string to a `Role`. Unknown roles fall back to
    /// `User`. Used by the `aiperf_trace` adapter's explicit-tag path; the
    /// WEKA/Dynamo heuristic only ever produces `User`/`Assistant`.
    fn from_authored(role: &str) -> Self {
        match role {
            "assistant" => Self::Assistant,
            "system" => Self::System,
            "tool" => Self::Tool,
            "tools" | "tool_defs" => Self::Tools,
            _ => Self::User,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockTag {
    role: Role,
    starts_message: bool,
}

impl BlockTag {
    /// Construct a ground-truth tag from an authored role string (adapter side).
    pub(crate) fn from_authored(role: &str, starts_message: bool) -> Self {
        Self {
            role: Role::from_authored(role),
            starts_message,
        }
    }
}

/// De-duplication key for a single prompt message within one trace's lowering.
///
/// A message is fully determined by its parent segment, role, and the recorded
/// block hashes it covers — block content is a pure function of `(hash, scope,
/// block_size)`, and the scope is fixed for the whole build. Interning the same
/// tuple is a `SegmentPool` dedup no-op, so caching the resulting handle lets
/// every shared-prefix message reuse the content-parent's segment verbatim
/// instead of re-decoding (tokenizer) and re-hashing (blake3) it per node. This
/// preserves byte-identical pool output in linear time.
#[derive(PartialEq, Eq, Hash)]
pub(super) struct PromptMessageKey {
    parent: Option<u32>,
    role: Role,
    hashes: Box<[crate::graph::recorded::BlockHash]>,
}

/// Per-trace prompt-message reuse cache consumed by [`emit_prompt`].
pub(super) type PromptMessageCache = rustc_hash::FxHashMap<PromptMessageKey, Handle>;

#[derive(Debug)]
struct Geometry {
    lcp: usize,
    covered: usize,
    missing_full_tokens: usize,
}

fn geometry_from_hashes(
    previous: &[crate::graph::recorded::BlockHash],
    current: &[crate::graph::recorded::BlockHash],
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

/// One role-tagged segment of the sequential reconstruction buffer.
///
/// Only the role and block count affect per-block tag emission; synth-tail
/// overhang is emitted outside the trie.
#[derive(Clone, Copy)]
struct PlanSegment {
    role: Role,
    block_count: usize,
}

fn segment_block_total(segments: &[PlanSegment]) -> usize {
    segments.iter().map(|segment| segment.block_count).sum()
}

/// Truncate the segment buffer in place so its cumulative block count == `target`.
///
/// Block-level equivalent of `weka_synth_buf.truncate_synth_buf_at_block`: every
/// segment is block-aligned, so keeping the first `target` blocks either drops
/// whole trailing segments (boundary/at-start cuts) or slices the straddling
/// segment down to its kept blocks (mid-segment cut). Segment role and the start
/// boundary of every surviving segment are preserved, which is what freezes a
/// block's `(role, starts_message)` across every later turn that inherits it.
fn truncate_segments(segments: &mut Vec<PlanSegment>, target: usize) {
    if target == 0 {
        segments.clear();
        return;
    }
    let mut cursor = 0_usize;
    let mut kept = Vec::with_capacity(segments.len());
    for segment in segments.iter() {
        if cursor >= target {
            break;
        }
        let take = segment.block_count.min(target - cursor);
        kept.push(PlanSegment {
            role: segment.role,
            block_count: take,
        });
        cursor += take;
    }
    *segments = kept;
}

/// Per-chain assistant block caps.
///
/// `turns` is one `(hash_ids, input_tokens)` pair per turn of a single chain, in
/// turn order. `caps[k]` bounds the assistant block count turn `k` may attribute
/// so a future block-aligned pull-back truncates onto a user block. Pure and
/// role-independent, so it is stable on first emission and never relabeled —
/// preserving cross-turn KV-cache reuse.
fn chain_assistant_caps(
    turns: &[(&[crate::graph::recorded::BlockHash], usize)],
    block_size: usize,
) -> Vec<Option<usize>> {
    let bs = block_size;
    let count = turns.len();
    let mut caps = vec![None; count];
    if count == 0 {
        return caps;
    }
    let mut tile: Vec<usize> = Vec::new();
    let mut effective = vec![0_usize; count];
    for turn in 0..count {
        let (hashes, input_tokens) = turns[turn];
        let previous = if turn > 0 { turns[turn - 1].0 } else { &[][..] };
        let geo = geometry_from_hashes(previous, hashes, input_tokens, bs);
        if turn == 0 {
            effective[0] = 0;
            tile = vec![0; geo.covered];
            continue;
        }
        let eff_lcp = geo.lcp.min(tile.len());
        effective[turn] = eff_lcp;
        let new_count = geo.covered.saturating_sub(eff_lcp);
        let synth_tail = input_tokens % bs + geo.missing_full_tokens;
        if new_count == 0 && synth_tail == 0 {
            let target = eff_lcp;
            if target >= 1 {
                let owner = tile[target - 1];
                // Turn-0 owners carry no assistant to cap.
                if owner != 0
                    && let Some(bound) = (target - 1).checked_sub(effective[owner])
                {
                    caps[owner] =
                        Some(caps[owner].map_or(bound, |current: usize| current.min(bound)));
                }
            }
        }
        tile.truncate(eff_lcp);
        tile.extend(std::iter::repeat_n(turn, new_count));
    }
    caps
}

/// Plan the heuristic `(role, starts_message)` tags for one chain by replaying
/// `weka_synth_buf.ConversationReconstructor` sequentially over its turns.
///
/// Role and message boundaries derive from the immediately preceding turn of the
/// same chain, not the
/// LCP-trie `content_parent` (which is chosen for content dedup and may point at
/// an earlier turn on a pull-back). On a pull-back turn the two disagree: the
/// trie parent's larger recorded output would size a longer assistant run.
/// Planning against turn `k-1` preserves exact per-block roles and
/// segment boundaries (including two adjacent same-role messages), which the
/// server's chat template + cross-boundary BPE weight into the ISL.
fn plan_chain_tags(
    chain_indices: &[usize],
    nodes: &[TrieNode],
    block_size: usize,
    all_tags: &mut [Vec<BlockTag>],
    inherited_by_node: &mut [usize],
) {
    let bs = block_size;
    let turns: Vec<(&[crate::graph::recorded::BlockHash], usize)> = chain_indices
        .iter()
        .map(|&index| {
            (
                nodes[index].request.hash_ids.as_slice(),
                nodes[index].request.input_tokens,
            )
        })
        .collect();
    let caps = chain_assistant_caps(&turns, bs);
    let mut segments: Vec<PlanSegment> = Vec::new();
    for (turn, &node_index) in chain_indices.iter().enumerate() {
        let node = &nodes[node_index];
        let input_tokens = node.request.input_tokens;
        let hashes = node.request.hash_ids.as_slice();
        let covered = hashes.len().min(input_tokens / bs);
        if turn == 0 {
            segments.clear();
            // Turn 0: a merged tool+system prefix (`weka_synth_buf.init_turn_0`)
            // becomes one `system` segment of `ceil((tool+system)/bs)` blocks
            // (clamped to covered); the remainder is the user segment. The WEKA
            // adapter supplies zero per-trace tool/system token counts, collapsing
            // this to a single user segment.
            let tool_system_tokens = 0_usize;
            let prefix_blocks = if tool_system_tokens > 0 {
                tool_system_tokens.div_ceil(bs).min(covered)
            } else {
                0
            };
            if prefix_blocks > 0 {
                segments.push(PlanSegment {
                    role: Role::System,
                    block_count: prefix_blocks,
                });
            }
            let user_blocks = covered - prefix_blocks;
            if user_blocks > 0 || segments.is_empty() {
                segments.push(PlanSegment {
                    role: Role::User,
                    block_count: user_blocks,
                });
            }
            inherited_by_node[node_index] = 0;
        } else {
            let previous = &nodes[chain_indices[turn - 1]];
            let geo = geometry_from_hashes(&previous.request.hash_ids, hashes, input_tokens, bs);
            let synth_tail = input_tokens % bs + geo.missing_full_tokens;
            let eff_lcp = geo.lcp.min(covered).min(segment_block_total(&segments));
            truncate_segments(&mut segments, eff_lcp);
            inherited_by_node[node_index] = eff_lcp;
            let new_count = covered.saturating_sub(eff_lcp);
            let previous_output = previous.request.output_tokens;
            let mut assistant = if previous_output > 0 {
                previous_output.div_ceil(bs)
            } else {
                0
            };
            // Context-loss rule: if the truncation removed every user segment the
            // conversation resumes at a user turn, so the whole new region is user.
            if !segments.iter().any(|segment| segment.role == Role::User) {
                assistant = 0;
            }
            assistant = assistant.min(new_count);
            if let Some(Some(cap)) = caps.get(turn) {
                assistant = assistant.min(*cap);
            }
            // Trailing-user invariant: only hand the final new block back to the
            // user when there is no synth tail to seed a trailing user segment
            // (`synth_tail == 0`). This is the guard the trie planner was missing.
            if synth_tail == 0 && assistant == new_count && assistant > 0 {
                assistant -= 1;
            }
            if assistant > 0 {
                segments.push(PlanSegment {
                    role: Role::Assistant,
                    block_count: assistant,
                });
            }
            let user_blocks = new_count - assistant;
            if user_blocks * bs + synth_tail > 0 {
                segments.push(PlanSegment {
                    role: Role::User,
                    block_count: user_blocks,
                });
            }
        }
        let mut tags = Vec::with_capacity(covered);
        for segment in &segments {
            for block in 0..segment.block_count {
                tags.push(BlockTag {
                    role: segment.role,
                    starts_message: block == 0,
                });
            }
        }
        all_tags[node_index] = tags;
    }
}

pub(super) fn assign_block_tags(
    nodes: &[TrieNode],
    block_size: usize,
) -> Result<(Vec<Vec<BlockTag>>, Vec<usize>), RecordedTraceError> {
    let mut all_tags: Vec<Vec<BlockTag>> = vec![Vec::new(); nodes.len()];
    let mut inherited_by_node = vec![0_usize; nodes.len()];
    // Chains whose blocks are planned by the sequential reconstructor. Grouped in
    // flatten order, which is turn order within each chain.
    let mut heuristic_chains: Vec<(String, Vec<usize>)> = Vec::new();
    let mut chain_slot: HashMap<&str, usize> = HashMap::new();
    for (index, node) in nodes.iter().enumerate() {
        // Ground-truth path: the `aiperf_trace` adapter supplies exact per-block
        // `(role, starts_message)` tags from real message boundaries, so we skip
        // the token-geometry heuristic entirely and validate the covered count.
        if let Some(explicit) = &node.request.explicit_tags {
            // Explicit tags cover *every* block, including a message's partial-tail
            // final block. The geometry heuristic's covered count is
            // `input_tokens / block_size` (a floor), which under-counts the moment
            // a segment ends on a partial tail — so validate against the true block
            // count instead. Prefix inheritance still comes from the block-id LCP.
            if explicit.len() != node.request.hash_ids.len() {
                return Err(RecordedTraceError(format!(
                    "node {:?}: explicit tag count {} differs from block count {}",
                    node.request.node_id,
                    explicit.len(),
                    node.request.hash_ids.len()
                )));
            }
            let parent_hashes = node
                .content_parent
                .map_or(&[][..], |parent| nodes[parent].request.hash_ids.as_slice());
            let lcp = parent_hashes
                .iter()
                .zip(&node.request.hash_ids)
                .take_while(|(left, right)| left == right)
                .count();
            inherited_by_node[index] = lcp.min(explicit.len());
            all_tags[index] = explicit.clone();
            continue;
        }
        let chain = node.request.chain_id.as_str();
        let slot = *chain_slot.entry(chain).or_insert_with(|| {
            heuristic_chains.push((node.request.chain_id.clone(), Vec::new()));
            heuristic_chains.len() - 1
        });
        heuristic_chains[slot].1.push(index);
    }
    for (_chain_id, chain_indices) in &heuristic_chains {
        plan_chain_tags(
            chain_indices,
            nodes,
            block_size,
            &mut all_tags,
            &mut inherited_by_node,
        );
    }
    Ok((all_tags, inherited_by_node))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_prompt(
    node: &TrieNode,
    tags: &[BlockTag],
    block_size: usize,
    hash_scope: Option<&str>,
    tail_scope: &str,
    content: &mut dyn RecordedContentSynthesizer,
    pool: &mut SegmentPool,
    cache: &mut PromptMessageCache,
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
    // Per-block token lengths: `block_size` for every block by default, or the
    // adapter's ground-truth lengths when supplied — where a message's final
    // block carries its exact partial-tail length. Truncating the deterministic
    // full block to that length is prefix-stable (a given block id always resolves
    // to the same first-N tokens), so shared-prefix messages stay byte-identical
    // and reuse the cache.
    let block_lens = node.request.block_lens.as_deref();
    let block_len = |block: usize| block_lens.map_or(block_size, |lens| lens[block]);
    let mut handles = Vec::with_capacity(groups.len());
    let mut parent: Option<Handle> = None;
    let mut assembled = 0_usize;
    for (role, blocks) in groups {
        let group_tokens: usize = blocks.iter().map(|block| block_len(*block)).sum();
        assembled = assembled.saturating_add(group_tokens);
        let key = PromptMessageKey {
            parent: parent.map(|handle| handle.index()),
            role,
            hashes: blocks
                .iter()
                .map(|block| node.request.hash_ids[*block])
                .collect(),
        };
        let handle = if let Some(cached) = cache.get(&key) {
            *cached
        } else {
            let mut tokens = Vec::with_capacity(group_tokens);
            for block in &blocks {
                let full = content.block_tokens(
                    &node.request.hash_ids[*block..*block + 1],
                    block_size,
                    hash_scope,
                )?;
                let len = block_len(*block).min(full.len());
                tokens.extend_from_slice(&full[..len]);
            }
            let text = content.decode(&tokens)?;
            let handle = intern_message(pool, parent, role.as_str(), &text, &tokens)?;
            cache.insert(key, handle);
            handle
        };
        parent = Some(handle);
        handles.push(handle);
    }
    // With ground-truth `block_lens`, the reconstruction must total the real input
    // length (`Σ block_lens == input_tokens`); otherwise it totals the covered
    // block count times `block_size` (the WEKA/Dynamo tail lives outside the trie).
    let expected = if block_lens.is_some() {
        node.request.input_tokens
    } else {
        tags.len().saturating_mul(block_size)
    };
    if assembled != expected {
        return Err(RecordedTraceError(format!(
            "node {:?}: reconstructed {assembled} tokens, expected {expected}",
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
mod parity_tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::graph::recorded::BlockHash;
    use crate::graph::recorded::content::RecordedContentSynthesizer;
    use crate::graph::recorded::trie::RecordedRequest;

    struct FixedContent;

    impl RecordedContentSynthesizer for FixedContent {
        fn block_tokens(
            &mut self,
            hashes: &[BlockHash],
            block_size: usize,
            _trace_scope: Option<&str>,
        ) -> Result<Vec<u32>, RecordedTraceError> {
            let mut output = Vec::new();
            for hash in hashes {
                let token = hash.to_string().parse::<u32>().unwrap();
                output.extend(std::iter::repeat_n(token, block_size));
            }
            Ok(output)
        }

        fn tail_tokens(&self, count: usize, _seed: &str) -> Vec<u32> {
            vec![999; count]
        }

        fn decode(&self, tokens: &[u32]) -> Result<String, RecordedTraceError> {
            Ok(tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","))
        }
    }

    fn node(id: &str, order: usize, hashes: &[i64], input: usize, output: usize) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "chain".into(),
                turn_index: order,
                order,
                hash_ids: hashes.iter().copied().map(i128::from).collect(),
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
                explicit_tags: None,
                block_lens: None,
            },
            content_parent: None,
            warped_start: order as f64,
            rank: order,
        }
    }

    #[test]
    fn inherited_tags_freeze_shared_prefix_and_split_assistant_then_user() {
        let parent = node("parent", 0, &[1, 2], 32, 16);
        let mut child = node("child", 1, &[1, 2, 3, 4], 64, 1);
        child.content_parent = Some(0);
        let nodes = vec![parent, child];
        let (tags, inherited) = assign_block_tags(&nodes, 16).unwrap();
        assert_eq!(inherited, [0, 2]);
        assert_eq!(tags[0].len(), 2);
        assert!(tags[0].iter().all(|tag| tag.role == Role::User));
        assert_eq!(
            tags[1][..2].iter().map(|tag| tag.role).collect::<Vec<_>>(),
            [Role::User, Role::User]
        );
        assert_eq!(tags[1][2].role, Role::Assistant);
        assert_eq!(tags[1][3].role, Role::User);
        assert!(tags[1][2].starts_message);
        assert!(tags[1][3].starts_message);

        let mut content = FixedContent;
        let mut pool = SegmentPool::new();
        let mut cache = PromptMessageCache::default();
        let parent_path = emit_prompt(
            &nodes[0],
            &tags[0],
            16,
            None,
            "trace",
            &mut content,
            &mut pool,
            &mut cache,
        )
        .unwrap();
        let child_path = emit_prompt(
            &nodes[1],
            &tags[1],
            16,
            None,
            "trace",
            &mut content,
            &mut pool,
            &mut cache,
        )
        .unwrap();
        assert_eq!(parent_path.len(), 1);
        assert_eq!(child_path.len(), 3);
        assert_eq!(parent_path[0], child_path[0]);
    }

    #[test]
    fn covered_count_zero_uses_one_real_user_message() {
        let node = node("tiny", 0, &[], 7, 1);
        let mut content = FixedContent;
        let mut pool = SegmentPool::new();
        let mut cache = PromptMessageCache::default();
        let path = emit_prompt(
            &node,
            &[],
            16,
            None,
            "trace",
            &mut content,
            &mut pool,
            &mut cache,
        )
        .unwrap();
        assert_eq!(path.len(), 1);
        let payload = pool.freeze();
        let crate::dataset::Payload::Message { role, tokens, .. } =
            crate::dataset::SegmentStore::get(&payload, path[0]).unwrap()
        else {
            panic!("tiny fallback message")
        };
        assert_eq!(role.as_str(), "user");
        assert_eq!(tokens.as_ref(), &[999; 7]);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;
    use crate::graph::recorded::BlockHash;
    use crate::graph::recorded::trie::RecordedRequest;

    fn node(id: &str, order: usize, hashes: &[i64], input: usize, output: usize) -> TrieNode {
        TrieNode {
            request: RecordedRequest {
                node_id: id.into(),
                chain_id: "chain".into(),
                turn_index: order,
                order,
                hash_ids: hashes.iter().copied().map(i128::from).collect(),
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
                explicit_tags: None,
                block_lens: None,
            },
            content_parent: None,
            warped_start: order as f64,
            rank: 0,
        }
    }

    #[test]
    fn geometry_clamps_covered_blocks_and_counts_only_missing_whole_blocks() {
        let previous: [BlockHash; 1] = [1];
        let current: [BlockHash; 3] = [1, 2, 3];
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
        let (tags, inherited) = assign_block_tags(&nodes, 2).unwrap();
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
        let (tags, inherited) = assign_block_tags(&nodes, 2).unwrap();
        assert_eq!(tags[0].len(), 2);
        assert_eq!(inherited[1], 2);
        assert_eq!(tags[1].len(), 5);
    }
}
