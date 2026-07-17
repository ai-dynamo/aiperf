// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Deterministic hash allocation for the KV-cache prefix model.
//!
//! Deterministic hash-id allocation — no RNG, pure arithmetic. Layout:
//! `L1: [0..L1_blocks)`, `L1.5: [L1_blocks + group*MAX_GROUP_BLOCKS ..]`,
//! `L2+L3: [session_region_base + session_index*MAX_SESSION_BLOCKS ..]`.

use crate::synthesize::config::CacheLayerConfig;

const MAX_GROUP_BLOCKS: i64 = 200;
const MAX_GROUPS: i64 = 1_000;
const MAX_SESSION_BLOCKS: i64 = 4_000;

/// Allocates deterministic hash IDs for the layered prefix model.
pub struct PrefixAllocator {
    block_size: i64,
    l1_blocks: i64,
    l15_blocks: i64,
    prefix_blocks: i64,
    session_region_base: i64,
}

impl PrefixAllocator {
    pub fn new(config: &CacheLayerConfig, block_size: i64) -> anyhow::Result<Self> {
        if block_size <= 0 {
            anyhow::bail!("block_size must be > 0");
        }
        let l1_blocks = div_ceil(config.layer1_tokens, block_size);
        let l15_blocks = div_ceil(config.layer1_5_tokens, block_size);
        let num_groups = config.layer1_5_groups.num_groups as i64;
        if num_groups > MAX_GROUPS {
            anyhow::bail!("num_groups={num_groups} exceeds MAX_GROUPS={MAX_GROUPS}");
        }
        if l15_blocks > MAX_GROUP_BLOCKS {
            anyhow::bail!(
                "layer1_5_tokens={} needs {l15_blocks} blocks, exceeding MAX_GROUP_BLOCKS={MAX_GROUP_BLOCKS}",
                config.layer1_5_tokens
            );
        }
        Ok(Self {
            block_size,
            l1_blocks,
            l15_blocks,
            prefix_blocks: l1_blocks + l15_blocks,
            session_region_base: l1_blocks + MAX_GROUPS * MAX_GROUP_BLOCKS,
        })
    }

    fn group_base(&self, group_id: i64) -> i64 {
        self.l1_blocks + group_id * MAX_GROUP_BLOCKS
    }

    fn session_base(&self, session_index: i64) -> i64 {
        self.session_region_base + session_index * MAX_SESSION_BLOCKS
    }

    fn l1_ids(&self, used: i64) -> Vec<i64> {
        (0..used).collect()
    }

    fn l15_ids(&self, group_id: i64, used: i64) -> Vec<i64> {
        let base = self.group_base(group_id);
        (base..base + used).collect()
    }

    /// Generate the full `hash_ids` array for a turn.
    pub fn turn_hash_ids(
        &self,
        session_index: i64,
        group_id: i64,
        input_length: i64,
        prev_session_ids: Option<&[i64]>,
    ) -> anyhow::Result<Vec<i64>> {
        let total_blocks = if input_length > 0 {
            div_ceil(input_length, self.block_size)
        } else {
            0
        };
        let l1_used = self.l1_blocks.min(total_blocks);
        let mut result = self.l1_ids(l1_used);

        let remaining = total_blocks - l1_used;
        if remaining <= 0 {
            return Ok(result);
        }

        let l15_used = self.l15_blocks.min(remaining);
        let l15 = self.l15_ids(group_id, l15_used);
        result.extend_from_slice(&l15);

        let session_needed = (remaining - l15_used).max(0);
        if session_needed == 0 {
            return Ok(result);
        }

        if session_needed > MAX_SESSION_BLOCKS {
            anyhow::bail!(
                "input_length={input_length} needs {session_needed} session blocks, exceeding MAX_SESSION_BLOCKS={MAX_SESSION_BLOCKS}"
            );
        }
        let base = self.session_base(session_index);

        let session: Vec<i64> = match prev_session_ids {
            None => (base..base + session_needed).collect(),
            Some(prev) => {
                let new_blocks = session_needed - prev.len() as i64;
                if new_blocks > 0 {
                    let next_id = if prev.is_empty() {
                        base
                    } else {
                        prev[prev.len() - 1] + 1
                    };
                    let mut s = prev.to_vec();
                    s.extend(next_id..next_id + new_blocks);
                    s
                } else {
                    prev[..session_needed as usize].to_vec()
                }
            }
        };
        result.extend_from_slice(&session);
        Ok(result)
    }

    /// Extract session-owned IDs after the shared L1 and L1.5 prefixes.
    pub fn extract_session_ids(&self, hash_ids: &[i64]) -> Vec<i64> {
        if (hash_ids.len() as i64) <= self.prefix_blocks {
            return Vec::new();
        }
        hash_ids[self.prefix_blocks as usize..].to_vec()
    }
}

fn div_ceil(a: i64, b: i64) -> i64 {
    if a <= 0 { 0 } else { (a + b - 1) / b }
}
