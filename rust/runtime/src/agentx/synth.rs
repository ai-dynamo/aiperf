// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LCP-driven conversation reconstructor for byte-exact Weka trace replay,
//! ported from `src/aiperf/dataset/loader/weka_synth_buf.py`.
//!
//! The "synthesis buffer" is the multi-segment in-progress chat tile maintained
//! across turns. [`ConversationReconstructor`] walks a conversation's turns,
//! truncating at the longest common prefix of hash ids and re-attributing the
//! new region to assistant/user [`RoleSegment`]s, preserving two invariants:
//! `sum(seg.tokens) == in_tokens` for each turn, and byte-identical token
//! content for any given `hash_id` across every segment it appears in.
//!
//! Token generation is injected through [`TokenSynth`] (three callbacks in the
//! Python original) so this module holds only reconstruction state and never a
//! tokenizer directly.

/// Token-generation seam for the reconstructor (Python's three callables).
///
/// - `decode_block_tokens(hash_ids)` returns the deterministic token sequence
///   for the given blocks (exactly `hash_ids.len() * block_size` tokens).
/// - `sample_partial_tail_tokens(n, seed)` returns `n` deterministic tokens
///   from a position-keyed seed.
/// - `decode_tokens_to_text(tokens)` decodes tokens to text with no
///   special-token insertion.
pub trait TokenSynth {
    /// Deterministic token sequence for `hash_ids` (`len * block_size` tokens).
    fn decode_block_tokens(&mut self, hash_ids: &[i64]) -> Vec<u32>;
    /// `n` deterministic tokens from the position-keyed `seed`.
    fn sample_partial_tail_tokens(&mut self, n: usize, seed: &str) -> Vec<u32>;
    /// Decode tokens to text (no special tokens).
    fn decode_tokens_to_text(&self, tokens: &[u32]) -> String;
}

/// A conversation role. Serializes to the OpenAI wire strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `"system"`.
    System,
    /// `"user"`.
    User,
    /// `"assistant"`.
    Assistant,
}

impl Role {
    /// The OpenAI wire string for this role.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One role-tagged segment of the reconstructed conversation.
///
/// Block ranges of adjacent segments form a contiguous tile of `[0, M_curr)`.
/// Only the final segment may carry a partial tail beyond its block range
/// (encoded into `tokens` but not `block_count`). `tokens` is the canonical size
/// source; `content == decode_tokens_to_text(tokens)` at emission time.
#[derive(Debug, Clone)]
pub struct RoleSegment {
    /// Segment role.
    pub role: Role,
    /// First block index this segment covers.
    pub block_start: i64,
    /// Number of full blocks this segment covers (excludes any partial tail).
    pub block_count: i64,
    /// Exact token IDs for this segment.
    pub tokens: Vec<u32>,
    /// Decoded text of `tokens`.
    pub content: String,
    /// Set on a user segment whose content is the tool output answering the
    /// immediately-preceding assistant tool call; holds the turn index the
    /// segment was created on (keys the deterministic synthetic call id).
    pub tool_result_turn: Option<i64>,
}

/// One OpenAI-style chat message `{role, content}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Role wire string.
    pub role: String,
    /// Message content.
    pub content: String,
}

/// Per-turn emission for delta-encoded reconstruction (Python `TurnDelta`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnDelta {
    /// Messages to emit for this turn.
    pub delta_messages: Vec<ChatMessage>,
    /// Whether the emission resets the previously-sent context.
    pub reset_context: bool,
}

/// Raised by [`ConversationReconstructor::init_turn_0`] when the recorded
/// hash-id prefix is too truncated to reconstruct the system/tool prefix
/// without faking cache structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTooTruncated {
    /// Blocks the system prefix requires.
    pub required_blocks: i64,
    /// Blocks actually recorded.
    pub recorded_blocks: i64,
}

impl std::fmt::Display for PrefixTooTruncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "weka trace turn-0 system prefix requires {} hash blocks but only {} were recorded; \
             the hash_ids list is too truncated to reconstruct the prefix",
            self.required_blocks, self.recorded_blocks
        )
    }
}

impl std::error::Error for PrefixTooTruncated {}

/// Walks a conversation's turns, maintaining synth_buf segments. Byte-exact port
/// of Python `ConversationReconstructor`.
pub struct ConversationReconstructor {
    block_size: i64,
    tool_shaped_messages: bool,
    segments: Vec<RoleSegment>,
    emitted_segment_count: usize,
    last_disturbance_at: Option<usize>,
    turn_index: i64,
    trailing_non_user_turns: Vec<i64>,
}

impl ConversationReconstructor {
    /// Construct with the trace's block size. `tool_shaped_messages` mirrors the
    /// Python flag (OpenAI tool-call wire shaping in `turn_delta`).
    pub fn new(block_size: i64, tool_shaped_messages: bool) -> Self {
        Self {
            block_size,
            tool_shaped_messages,
            segments: Vec::new(),
            emitted_segment_count: 0,
            last_disturbance_at: None,
            turn_index: 0,
            trailing_non_user_turns: Vec::new(),
        }
    }

    /// Current segments (read-only), for the loader/snapshot paths.
    pub fn segments(&self) -> &[RoleSegment] {
        &self.segments
    }

    /// Turn indices that could not be made to end with a user segment.
    pub fn trailing_non_user_turns(&self) -> &[i64] {
        &self.trailing_non_user_turns
    }

    /// Initialize segments for turn 0 from a tool+system / user prefix split
    /// (spec §4.3). Returns `Err` only when even the system/tool prefix cannot
    /// be filled from `hash_ids`.
    #[allow(clippy::too_many_arguments)]
    pub fn init_turn_0(
        &mut self,
        synth: &mut dyn TokenSynth,
        hash_ids: &[i64],
        in_tokens: i64,
        tool_tokens: i64,
        system_tokens: i64,
        seed: &str,
    ) -> Result<(), PrefixTooTruncated> {
        let bs = self.block_size;
        let m_full = in_tokens / bs;
        let partial_tail_tokens_n = in_tokens - m_full * bs;
        let covered_blocks = m_full.min(hash_ids.len() as i64);
        let missing_block_tokens = (m_full - covered_blocks) * bs;

        let mut cursor: i64 = 0;
        let mut segs: Vec<RoleSegment> = Vec::new();

        if tool_tokens > 0 || system_tokens > 0 {
            let prefix_tokens = tool_tokens + system_tokens;
            let mut prefix_blocks = div_ceil(prefix_tokens, bs);
            if prefix_blocks > 0 {
                if prefix_blocks > hash_ids.len() as i64 {
                    return Err(PrefixTooTruncated {
                        required_blocks: prefix_blocks,
                        recorded_blocks: hash_ids.len() as i64,
                    });
                }
                // Clamp to the prompt's own covered-block count (see Python note).
                prefix_blocks = prefix_blocks.min(covered_blocks);
            }
            if prefix_blocks > 0 {
                let slice = &hash_ids[cursor as usize..(cursor + prefix_blocks) as usize];
                let seg_tokens = synth.decode_block_tokens(slice);
                let content = synth.decode_tokens_to_text(&seg_tokens);
                segs.push(RoleSegment {
                    role: Role::System,
                    block_start: cursor,
                    block_count: prefix_blocks,
                    tokens: seg_tokens,
                    content,
                    tool_result_turn: None,
                });
                cursor += prefix_blocks;
            }
        }

        let user_blocks = covered_blocks - cursor;
        let user_slice = &hash_ids[cursor as usize..(cursor + user_blocks) as usize];
        let mut user_tokens = synth.decode_block_tokens(user_slice);
        let synth_tail_n = missing_block_tokens + partial_tail_tokens_n;
        if synth_tail_n > 0 {
            user_tokens.extend(synth.sample_partial_tail_tokens(synth_tail_n as usize, seed));
        }
        if !user_tokens.is_empty() || segs.is_empty() {
            let content = synth.decode_tokens_to_text(&user_tokens);
            segs.push(RoleSegment {
                role: Role::User,
                block_start: cursor,
                block_count: user_blocks,
                tokens: user_tokens,
                content,
                tool_result_turn: None,
            });
        }

        self.turn_index = 0;
        self.segments = segs;
        self.emitted_segment_count = 0;
        self.last_disturbance_at = None;
        self.assert_trailing_user();
        Ok(())
    }

    /// Advance synth_buf to turn `k` via LCP-driven symmetric attribution
    /// (spec §4.4). `max_asst_blocks` is the optional Pass-1 planning cap.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_turn(
        &mut self,
        synth: &mut dyn TokenSynth,
        prev_hash_ids: &[i64],
        prev_out_tokens: i64,
        curr_hash_ids: &[i64],
        curr_in_tokens: i64,
        seed: &str,
        is_tool_result: bool,
        max_asst_blocks: Option<i64>,
    ) {
        let bs = self.block_size;
        let geo = compute_turn_block_geometry(prev_hash_ids, curr_hash_ids, curr_in_tokens, bs);
        let lcp = geo.lcp;
        let m_curr_covered = geo.m_curr_covered;
        let synth_tail_n = geo.synth_tail_n;
        let new_blocks_count = geo.new_blocks_count;

        self.last_disturbance_at = truncate_synth_buf_at_block(&mut self.segments, lcp, bs, synth);

        let new_blocks = &curr_hash_ids[lcp as usize..m_curr_covered as usize];
        let mut new_region_tokens = synth.decode_block_tokens(new_blocks);
        if synth_tail_n > 0 {
            new_region_tokens.extend(synth.sample_partial_tail_tokens(synth_tail_n as usize, seed));
        }

        self.turn_index += 1;
        let mut asst_blocks_target = if prev_out_tokens > 0 {
            div_ceil(prev_out_tokens, bs)
        } else {
            0
        };
        if !self.segments.iter().any(|s| s.role == Role::User) {
            // Context-loss rule: resume at a user turn (no assistant before any user).
            asst_blocks_target = 0;
        }
        let mut asst_blocks = asst_blocks_target.min(new_blocks_count);
        if let Some(cap) = max_asst_blocks {
            asst_blocks = asst_blocks.min(cap);
        }
        if synth_tail_n == 0 && asst_blocks == new_blocks_count && asst_blocks > 0 {
            // Wire invariant: a turn must end with a user message. Hand the final
            // new block back to the user (relabel only; byte-exactness holds).
            asst_blocks -= 1;
        }
        let asst_emit_size = asst_blocks * bs;

        let mut cursor = lcp;
        if asst_blocks > 0 {
            let asst_tokens = new_region_tokens[..asst_emit_size as usize].to_vec();
            let content = synth.decode_tokens_to_text(&asst_tokens);
            self.segments.push(RoleSegment {
                role: Role::Assistant,
                block_start: cursor,
                block_count: asst_blocks,
                tokens: asst_tokens,
                content,
                tool_result_turn: None,
            });
            cursor += asst_blocks;
        }

        let user_blocks = new_blocks_count - asst_blocks;
        let user_tokens = new_region_tokens[asst_emit_size as usize..].to_vec();
        if !user_tokens.is_empty() {
            let content = synth.decode_tokens_to_text(&user_tokens);
            self.segments.push(RoleSegment {
                role: Role::User,
                block_start: cursor,
                block_count: user_blocks,
                tokens: user_tokens,
                content,
                tool_result_turn: if is_tool_result {
                    Some(self.turn_index)
                } else {
                    None
                },
            });
        }

        self.assert_trailing_user();
    }

    /// Record (and note) when the segment list does not end with a user segment.
    fn assert_trailing_user(&mut self) {
        match self.segments.last() {
            None => {}
            Some(seg) if seg.role == Role::User => {}
            Some(_) => self.trailing_non_user_turns.push(self.turn_index),
        }
    }

    /// Compute the messages to emit for the just-completed turn (Python
    /// `turn_delta`). Tool-shaping is applied when `tool_shaped_messages` is set.
    pub fn turn_delta(&mut self) -> TurnDelta {
        let disturbed_emitted = matches!(
            self.last_disturbance_at,
            Some(d) if d < self.emitted_segment_count
        );
        let (source, reset): (&[RoleSegment], bool) =
            if self.emitted_segment_count == 0 || disturbed_emitted {
                (
                    &self.segments[..],
                    self.emitted_segment_count != 0 && disturbed_emitted,
                )
            } else {
                (&self.segments[self.emitted_segment_count..], false)
            };

        let messages: Vec<ChatMessage> = source
            .iter()
            .map(|s| ChatMessage {
                role: s.role.as_str().to_string(),
                content: s.content.clone(),
            })
            .collect();
        assert!(
            !self.tool_shaped_messages,
            "agentx: tool_shaped_messages not yet ported (weka_tool_shape.rs pending)"
        );

        self.emitted_segment_count = self.segments.len();
        self.last_disturbance_at = None;
        TurnDelta {
            delta_messages: messages,
            reset_context: reset,
        }
    }

    /// The current synth_buf as OpenAI-style chat messages (full prefix).
    pub fn snapshot_messages(&self) -> Vec<ChatMessage> {
        self.segments
            .iter()
            .map(|s| ChatMessage {
                role: s.role.as_str().to_string(),
                content: s.content.clone(),
            })
            .collect()
    }
}

/// Ceiling division for non-negative `n` by positive `d` (matches Python
/// `math.ceil(n / d)` on the integer domain used here).
fn div_ceil(n: i64, d: i64) -> i64 {
    if n <= 0 {
        0
    } else {
        (n + d - 1) / d
    }
}

/// Index of the first differing element of the two sequences (Python
/// `longest_common_prefix`).
pub fn longest_common_prefix(prev_hash_ids: &[i64], curr_hash_ids: &[i64]) -> i64 {
    let n = prev_hash_ids.len().min(curr_hash_ids.len());
    for i in 0..n {
        if prev_hash_ids[i] != curr_hash_ids[i] {
            return i as i64;
        }
    }
    n as i64
}

/// Role-independent block accounting for one turn transition (Python
/// `TurnBlockGeometry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnBlockGeometry {
    /// Longest common prefix (in blocks) of prev/curr hash ids.
    pub lcp: i64,
    /// Covered block count `min(len(curr_hash_ids), curr_in_tokens // bs)`.
    pub m_curr_covered: i64,
    /// Full blocks appended after truncating to `lcp`.
    pub new_blocks_count: i64,
    /// Synthesized trailing tokens (partial tail + missing-block region).
    pub synth_tail_n: i64,
}

/// Compute the role-independent block geometry for one turn transition.
pub fn compute_turn_block_geometry(
    prev_hash_ids: &[i64],
    curr_hash_ids: &[i64],
    curr_in_tokens: i64,
    block_size: i64,
) -> TurnBlockGeometry {
    let bs = block_size;
    let m_curr = curr_hash_ids.len() as i64;
    let m_curr_full = curr_in_tokens / bs;
    let m_curr_covered = m_curr.min(m_curr_full);
    let missing_block_tokens = ((m_curr_full - m_curr) * bs).max(0);
    let lcp = longest_common_prefix(prev_hash_ids, curr_hash_ids);
    TurnBlockGeometry {
        lcp,
        m_curr_covered,
        new_blocks_count: (m_curr_covered - lcp).max(0),
        synth_tail_n: curr_in_tokens % bs + missing_block_tokens,
    }
}

/// Pass-1 planner: per-turn upper bounds on assistant block count (Python
/// `compute_asst_block_caps`). `turns` is one `(hash_ids, in_tokens)` per turn.
pub fn compute_asst_block_caps(turns: &[(Vec<i64>, i64)], block_size: i64) -> Vec<Option<i64>> {
    let n_turns = turns.len();
    let mut caps: Vec<Option<i64>> = vec![None; n_turns];
    if n_turns == 0 {
        return caps;
    }
    let bs = block_size;
    let mut tile: Vec<i64> = Vec::new();
    let mut eff_lcp_per_turn: Vec<i64> = vec![0; n_turns];

    for k in 0..n_turns {
        let (hash_ids, in_tokens) = &turns[k];
        let empty: Vec<i64> = Vec::new();
        let prev_hash_ids = if k > 0 { &turns[k - 1].0 } else { &empty };
        let geo = compute_turn_block_geometry(prev_hash_ids, hash_ids, *in_tokens, bs);
        if k == 0 {
            eff_lcp_per_turn[0] = 0;
            tile = vec![0; geo.m_curr_covered.max(0) as usize];
            continue;
        }

        let eff_lcp = geo.lcp.min(tile.len() as i64);
        eff_lcp_per_turn[k] = eff_lcp;
        let new_blocks_count = (geo.m_curr_covered - eff_lcp).max(0);
        let synth_tail_n = geo.synth_tail_n;

        if new_blocks_count == 0 && synth_tail_n == 0 {
            let target = eff_lcp;
            if target >= 1 {
                let owner = tile[(target - 1) as usize];
                if owner != 0 {
                    let bound = (target - 1) - eff_lcp_per_turn[owner as usize];
                    if bound >= 0 {
                        caps[owner as usize] = Some(match caps[owner as usize] {
                            None => bound,
                            Some(prev) => prev.min(bound),
                        });
                    }
                }
            }
        }

        tile.truncate(eff_lcp as usize);
        tile.extend(std::iter::repeat(k as i64).take(new_blocks_count as usize));
    }

    caps
}

/// Truncate `segments` in place so cumulative `block_count == target_blocks`,
/// returning the earliest disturbed segment index (Python
/// `truncate_synth_buf_at_block`). `synth` re-derives `content` from surviving
/// tokens.
pub fn truncate_synth_buf_at_block(
    segments: &mut Vec<RoleSegment>,
    target_blocks: i64,
    block_size: i64,
    synth: &dyn TokenSynth,
) -> Option<usize> {
    if target_blocks <= 0 {
        let had_segments = !segments.is_empty();
        segments.clear();
        return if had_segments { Some(0) } else { None };
    }

    let mut cursor: i64 = 0;
    let n = segments.len();
    for i in 0..n {
        let bc = segments[i].block_count;
        if cursor + bc < target_blocks {
            cursor += bc;
            continue;
        }
        if cursor + bc == target_blocks {
            // Boundary cut: strip this segment's overhang past its block range,
            // then drop any segments past the boundary.
            let mut disturbed: Option<usize> = None;
            let overhang = segments[i].tokens.len() as i64 - segments[i].block_count * block_size;
            if overhang > 0 {
                let keep = segments[i].tokens.len() - overhang as usize;
                segments[i].tokens.truncate(keep);
                segments[i].content = synth.decode_tokens_to_text(&segments[i].tokens);
                disturbed = Some(i);
            }
            let deleted_past_boundary = i + 1 < segments.len();
            segments.truncate(i + 1);
            if disturbed.is_none() && deleted_past_boundary {
                disturbed = Some(i + 1);
            }
            return disturbed;
        }
        if cursor == target_blocks {
            // Cut lands exactly at the start of segment i.
            segments.truncate(i);
            return Some(i);
        }
        // Mid-segment cut on a guaranteed block boundary.
        let kept_blocks = target_blocks - cursor;
        let kept_tokens_n = (segments[i].tokens.len() as i64).min(kept_blocks * block_size);
        segments[i].block_count = kept_blocks;
        segments[i].tokens.truncate(kept_tokens_n as usize);
        segments[i].content = synth.decode_tokens_to_text(&segments[i].tokens);
        segments.truncate(i + 1);
        return Some(i);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stub: block `h` decodes to `bs` tokens `[h*1000 .. h*1000+bs)`,
    /// tail sampler emits `[900000 + i]`, text is space-joined token ids.
    struct StubSynth {
        bs: i64,
    }
    impl TokenSynth for StubSynth {
        fn decode_block_tokens(&mut self, hash_ids: &[i64]) -> Vec<u32> {
            hash_ids
                .iter()
                .flat_map(|&h| (0..self.bs).map(move |i| (h as u32) * 1000 + i as u32))
                .collect()
        }
        fn sample_partial_tail_tokens(&mut self, n: usize, _seed: &str) -> Vec<u32> {
            (0..n as u32).map(|i| 900_000 + i).collect()
        }
        fn decode_tokens_to_text(&self, tokens: &[u32]) -> String {
            tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    #[test]
    fn lcp_basic() {
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(longest_common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(longest_common_prefix(&[9], &[1]), 0);
    }

    #[test]
    fn geometry_partial_last_block() {
        // bs=64, in=250 -> m_full=3; hash_ids len 4 (one partial) -> covered=3.
        let geo = compute_turn_block_geometry(&[], &[0, 1, 2, 3], 250, 64);
        assert_eq!(geo.m_curr_covered, 3);
        assert_eq!(geo.lcp, 0);
        assert_eq!(geo.new_blocks_count, 3);
        assert_eq!(geo.synth_tail_n, 250 % 64); // 58
    }

    #[test]
    fn turn0_invariant_sum_tokens_equals_in() {
        let mut s = StubSynth { bs: 4 };
        let mut r = ConversationReconstructor::new(4, false);
        // in=10, bs=4 -> 2 full blocks + tail 2; hash_ids cover 2 blocks.
        r.init_turn_0(&mut s, &[0, 1], 10, 0, 0, "seed").unwrap();
        let total: usize = r.segments().iter().map(|seg| seg.tokens.len()).sum();
        assert_eq!(total, 10);
        assert_eq!(r.segments().last().unwrap().role, Role::User);
    }

    #[test]
    fn append_only_turn_keeps_invariant_and_trailing_user() {
        let mut s = StubSynth { bs: 4 };
        let mut r = ConversationReconstructor::new(4, false);
        r.init_turn_0(&mut s, &[0, 1], 8, 0, 0, "s0").unwrap();
        let _ = r.turn_delta();
        // Turn 1: prev out 4 tokens (1 block), curr shares prefix [0,1] then adds [2,3].
        r.advance_turn(&mut s, &[0, 1], 4, &[0, 1, 2, 3], 16, "s1", false, None);
        let total: usize = r.segments().iter().map(|seg| seg.tokens.len()).sum();
        assert_eq!(total, 16);
        assert_eq!(r.segments().last().unwrap().role, Role::User);
        assert!(r.trailing_non_user_turns().is_empty());
    }

    #[test]
    fn truncate_boundary_strips_overhang() {
        // Build a single trailing user segment with an overhang tail, truncate to
        // its block boundary, expect the tail stripped and disturbance at 0.
        let s = StubSynth { bs: 4 };
        let mut segs = vec![RoleSegment {
            role: Role::User,
            block_start: 0,
            block_count: 2,
            tokens: (0..10).collect(), // 8 block tokens + 2 overhang
            content: String::new(),
            tool_result_turn: None,
        }];
        let d = truncate_synth_buf_at_block(&mut segs, 2, 4, &s);
        assert_eq!(d, Some(0));
        assert_eq!(segs[0].tokens.len(), 8);
    }

    #[test]
    fn asst_caps_no_panic_on_simple_sequence() {
        let turns = vec![(vec![0, 1], 8i64), (vec![0, 1, 2], 12), (vec![0], 4)];
        let caps = compute_asst_block_caps(&turns, 4);
        assert_eq!(caps.len(), 3);
    }
}
