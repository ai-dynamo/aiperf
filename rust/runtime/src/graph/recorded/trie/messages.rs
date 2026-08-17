// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen block-role planning and content-addressed message emission.

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

/// Split a turn's new blocks into per-block `assistant`/`user` roles relative to
/// its content parent. Byte-exact port of the graph-ir Python
/// `segment_ir.trie_content.block_role_split`.
///
/// Returns `(inherited, roles)` where `inherited` is the block count carried over
/// from the content parent (`min(lcp, prev_len, m_curr_covered, parent_covered)`)
/// and `roles` is the creation-time role of each new block: a leading assistant
/// run of `ceil(prev_out / bs)` blocks (the parent turn's response), clamped to
/// the new-block width, the `max_asst_blocks` cap, and zeroed when the parent
/// carries no user context (context-loss branch); the remainder is user. A turn
/// whose new region is all-assistant flips its OWN last new block to user so the
/// frozen boundary lands on a user block (inherited verbatim by every descendant).
struct BlockRoleSplitArgs<'a> {
    prev_hash_ids: &'a [crate::graph::recorded::BlockHash],
    curr_hash_ids: &'a [crate::graph::recorded::BlockHash],
    curr_in_tokens: usize,
    prev_out_tokens: usize,
    block_size: usize,
    max_asst_blocks: Option<usize>,
    parent_has_user: bool,
    parent_covered_blocks: usize,
}

fn block_role_split(args: BlockRoleSplitArgs<'_>) -> (usize, Vec<Role>) {
    let geo = geometry_from_hashes(
        args.prev_hash_ids,
        args.curr_hash_ids,
        args.curr_in_tokens,
        args.block_size,
    );
    let inherited = geo
        .lcp
        .min(args.prev_hash_ids.len())
        .min(geo.covered)
        .min(args.parent_covered_blocks);
    let new_n = geo.covered.saturating_sub(inherited);
    let mut asst = if args.prev_out_tokens > 0 {
        args.prev_out_tokens.div_ceil(args.block_size)
    } else {
        0
    };
    if !args.parent_has_user {
        // Context-loss branch: the parent holds no user context, so the wire
        // cannot present assistant output before any user input.
        asst = 0;
    }
    asst = asst.min(new_n);
    if let Some(cap) = args.max_asst_blocks {
        asst = asst.min(cap);
    }
    if asst == new_n && asst > 0 {
        // Trailing-user (frozen at creation): a node whose new region is all
        // assistant flips its own last new block to user.
        asst -= 1;
    }
    let mut roles = Vec::with_capacity(new_n);
    roles.extend(std::iter::repeat_n(Role::Assistant, asst));
    roles.extend(std::iter::repeat_n(Role::User, new_n - asst));
    (inherited, roles)
}

/// Pass-1 trailing-user planner over the GLOBAL `content_parent` tree. Byte-exact
/// port of the graph-ir Python `segment_ir.trie_content.compute_asst_caps`.
///
/// A degenerate pull-back (`new_blocks_count == 0` and `synth_tail_n == 0`) at
/// `eff >= 1` re-exposes block `eff - 1` of the parent lineage; to keep the frozen
/// boundary on a user block, the owning ancestor of that block is capped. Owners
/// are tracked per node as a "tile" (block index -> owning node index). Root
/// owners (`content_parent is None`) are never capped. `caps[node]` bounds that
/// node's assistant block count. Requires `resolve_content_parents` to have run.
fn compute_asst_caps_tree(nodes: &[TrieNode], block_size: usize) -> Vec<Option<usize>> {
    let bs = block_size;
    let n = nodes.len();
    let mut caps: Vec<Option<usize>> = vec![None; n];
    let mut tiles: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut eff = vec![0_usize; n];
    for index in 0..n {
        let node = &nodes[index];
        let curr = node.request.hash_ids.as_slice();
        let input_tokens = node.request.input_tokens;
        let Some(parent) = node.content_parent else {
            let geo = geometry_from_hashes(&[], curr, input_tokens, bs);
            tiles[index] = vec![index; geo.covered];
            eff[index] = 0;
            continue;
        };
        let geo = geometry_from_hashes(&nodes[parent].request.hash_ids, curr, input_tokens, bs);
        let parent_tile_len = tiles[parent].len();
        let e = geo.lcp.min(parent_tile_len).min(geo.covered);
        eff[index] = e;
        let new_count = geo.covered.saturating_sub(e);
        let synth_tail = input_tokens % bs + geo.missing_full_tokens;
        if new_count == 0 && synth_tail == 0 && e >= 1 {
            let owner = tiles[parent][e - 1];
            // Skip a root owner (no assistant to cap), matching agentx's guard.
            if nodes[owner].content_parent.is_some()
                && let Some(bound) = (e - 1).checked_sub(eff[owner])
            {
                caps[owner] = Some(caps[owner].map_or(bound, |current: usize| current.min(bound)));
            }
        }
        let mut tile = tiles[parent][..e].to_vec();
        tile.extend(std::iter::repeat_n(index, new_count));
        tiles[index] = tile;
    }
    caps
}

pub(super) fn assign_block_tags(
    nodes: &[TrieNode],
    block_size: usize,
) -> Result<(Vec<Vec<BlockTag>>, Vec<usize>), RecordedTraceError> {
    let mut all_tags: Vec<Vec<BlockTag>> = vec![Vec::new(); nodes.len()];
    let mut inherited_by_node = vec![0_usize; nodes.len()];
    // Pass-1 trailing-user caps over the content-parent tree (only the heuristic
    // path consults them; the ground-truth `aiperf_trace` path ignores caps).
    let caps = compute_asst_caps_tree(nodes, block_size);
    // Nodes are processed in recorded order so a content-parent is tagged before
    // any child reads its frozen tags (`resolve_content_parents` guarantees a
    // parent's index precedes its children's). Byte-exact port of the graph-ir
    // Python `segment_ir.trie_content._assign_block_tags_and_inheritance`.
    for index in 0..nodes.len() {
        let node = &nodes[index];
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
        // Heuristic path: inherit the content-parent's frozen tags for the shared
        // prefix, then append this turn's new blocks. Message boundaries thus flow
        // through the content-parent tree, NOT a linear per-chain accumulator — a
        // branch/reset turn whose parent is an earlier node never collapses the
        // main lineage's per-turn boundaries.
        let curr = node.request.hash_ids.as_slice();
        let input_tokens = node.request.input_tokens;
        let (prev_hash, prev_out): (&[crate::graph::recorded::BlockHash], usize) =
            match node.content_parent {
                Some(parent) => (
                    nodes[parent].request.hash_ids.as_slice(),
                    nodes[parent].request.output_tokens,
                ),
                None => (&[], 0),
            };
        let parent_tags: Vec<BlockTag> = node
            .content_parent
            .map_or_else(Vec::new, |parent| all_tags[parent].clone());
        let geo = geometry_from_hashes(prev_hash, curr, input_tokens, block_size);
        let inherited = geo.lcp.min(parent_tags.len()).min(geo.covered);
        // Context-loss rule is decided over the INHERITED PREFIX ONLY.
        let parent_has_user = parent_tags[..inherited]
            .iter()
            .any(|tag| tag.role == Role::User);
        let (inh2, new_roles) = block_role_split(BlockRoleSplitArgs {
            prev_hash_ids: prev_hash,
            curr_hash_ids: curr,
            curr_in_tokens: input_tokens,
            prev_out_tokens: prev_out,
            block_size,
            max_asst_blocks: caps[index],
            parent_has_user,
            parent_covered_blocks: parent_tags.len(),
        });
        debug_assert_eq!(inh2, inherited, "block-tag/geometry inherited disagreement");
        let mut tags: Vec<BlockTag> = parent_tags[..inherited].to_vec();
        for (j, role) in new_roles.iter().copied().enumerate() {
            // A new recorded turn always opens a message (even continuing the
            // parent's tail role, preserving contiguous same-role turns); otherwise
            // a message starts only at a role transition within the new region.
            let starts_message = j == 0 || role != new_roles[j - 1];
            tags.push(BlockTag {
                role,
                starts_message,
            });
        }
        all_tags[index] = tags;
        inherited_by_node[index] = inherited;
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
