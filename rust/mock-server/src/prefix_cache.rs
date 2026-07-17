// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-addressed block-level prefix (KV-cache) reuse model.
//!
//! A prompt is split into fixed-size blocks and chain-hashed so each hash
//! folds in every preceding block. An identical prefix yields identical block
//! hashes, and the longest run of
//! leading blocks already resident in the cache is served from cache — those
//! tokens skip prefill, lowering TTFT. After lookup the request's blocks are
//! inserted into a capacity-bounded cache, so an aged-out prefix goes cold and
//! stops hitting. The cached token count is reported back as
//! `usage.prompt_tokens_details.cached_tokens`.
//!
//! `--prefix-cache-eviction-policy` uses SGLang
//! `--radix-eviction-policy` semantics. See [`EvictionPolicy`].
//!
//! `--prefix-cache-hit-rate > 0` bypasses content addressing and forces a fixed
//! cached fraction on every request — a workload-agnostic "what if N% hit"
//! study for traffic that has no natural prefix sharing.

use std::collections::{BTreeMap, HashMap};

use blake2::{Blake2s256, Digest};
use clap::ValueEnum;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::config::MockServerConfig;

/// SGLang-compatible `--radix-eviction-policy` semantics.
///
/// Capacity pressure removes the block with the smallest eviction key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
pub enum EvictionPolicy {
    /// Least-recently-used (SGLang default): evict the oldest-accessed block.
    #[default]
    Lru,
    /// Least-frequently-used; ties broken by least-recently-used.
    Lfu,
    /// First-in-first-out: evict the earliest-inserted block.
    Fifo,
    /// Most-recently-used: evict the newest-accessed block.
    Mru,
    /// First-in-last-out: evict the latest-inserted block.
    Filo,
    /// Priority-aware: lowest request priority evicted first, ties by LRU. Needs
    /// a per-request `priority` in the payload; with none supplied every block
    /// shares priority 0 and this reduces to LRU (as in SGLang).
    Priority,
    /// Segmented LRU: probationary blocks (reused fewer than 2x) are evicted
    /// before protected ones; LRU within each segment.
    Slru,
}

/// Reuse count at/above which an `slru` block is "protected" (SGLang default).
const SLRU_PROTECTED_THRESHOLD: u64 = 2;

/// Per-block bookkeeping for eviction ordering. The `clock` fields draw from a
/// single monotonic counter, so any eviction key that includes `last_access` or
/// `creation` is unique across distinct resident blocks.
#[derive(Clone, Copy)]
struct BlockMeta {
    creation: u64,
    last_access: u64,
    hit_count: u64,
    priority: i64,
}

impl BlockMeta {
    /// Smaller keys are evicted first.
    fn evict_key(&self, policy: EvictionPolicy) -> (i64, i64, i64) {
        let la = self.last_access as i64;
        let cr = self.creation as i64;
        let hc = self.hit_count as i64;
        match policy {
            EvictionPolicy::Lru => (la, 0, 0),
            EvictionPolicy::Lfu => (hc, la, 0),
            EvictionPolicy::Fifo => (cr, 0, 0),
            EvictionPolicy::Mru => (-la, 0, 0),
            EvictionPolicy::Filo => (-cr, 0, 0),
            EvictionPolicy::Priority => (self.priority, la, 0),
            EvictionPolicy::Slru => ((self.hit_count >= SLRU_PROTECTED_THRESHOLD) as i64, la, 0),
        }
    }
}

/// Capacity-bounded block store with a configurable eviction policy. A `HashMap`
/// gives O(1) membership + metadata; a `BTreeMap` keyed by the policy's eviction
/// key gives O(log n) selection of the next victim.
///
/// Space is freed before insertion, so the incoming in-flight block cannot evict
/// itself. Earlier blocks remain unpinned, allowing a long prompt to evict its own
/// prior blocks under tight capacity.
struct BlockCache {
    policy: EvictionPolicy,
    blocks: HashMap<u64, BlockMeta>,
    order: BTreeMap<(i64, i64, i64), u64>, // eviction key -> hash
    clock: u64,
    capacity: usize,
}

impl BlockCache {
    fn new(policy: EvictionPolicy, capacity: usize) -> Self {
        Self {
            policy,
            blocks: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            capacity: capacity.max(1),
        }
    }

    fn contains(&self, hash: u64) -> bool {
        self.blocks.contains_key(&hash)
    }

    /// Insert a new block or refresh an existing one as just-accessed, evicting
    /// down to capacity per the policy. `priority` is the issuing request's
    /// priority (0 when unspecified).
    fn touch(&mut self, hash: u64, priority: i64) {
        self.clock += 1;
        let now = self.clock;
        if let Some(mut m) = self.blocks.get(&hash).copied() {
            self.order.remove(&m.evict_key(self.policy));
            m.last_access = now;
            m.hit_count += 1;
            m.priority = priority;
            self.order.insert(m.evict_key(self.policy), hash);
            self.blocks.insert(hash, m);
            return;
        }
        // Evict before insertion so the incoming block cannot select itself.
        while self.blocks.len() >= self.capacity {
            match self.order.pop_first() {
                Some((_, h)) => {
                    self.blocks.remove(&h);
                }
                None => break,
            }
        }
        let m = BlockMeta {
            creation: now,
            last_access: now,
            hit_count: 1,
            priority,
        };
        self.order.insert(m.evict_key(self.policy), hash);
        self.blocks.insert(hash, m);
    }
}

/// Upper bound on blocks hashed per request. With `block_tokens=1` (SGLang's
/// default page_size) a 100k-token prompt would otherwise need 100k hashes; the
/// cap keeps cost bounded by coarsening the effective block size for very long
/// prompts while leaving normal-length prompts token-granular.
const MAX_BLOCKS_PER_REQUEST: usize = 4096;

/// KV-cache prefix-reuse model. Shared across requests behind a `Mutex`.
pub struct PrefixCache {
    enabled: bool,
    block_tokens: usize,
    hit_rate: f64,
    cache: Mutex<BlockCache>,
}

impl PrefixCache {
    /// Returns `None` when both cache operation and synthetic hits are disabled.
    pub fn from_config(cfg: &MockServerConfig) -> Option<Self> {
        if cfg.disable_prefix_cache && cfg.prefix_cache_hit_rate <= 0.0 {
            return None;
        }
        Some(Self {
            enabled: !cfg.disable_prefix_cache,
            block_tokens: cfg.prefix_cache_block_tokens.max(1),
            hit_rate: cfg.prefix_cache_hit_rate.clamp(0.0, 1.0),
            cache: Mutex::new(BlockCache::new(
                cfg.prefix_cache_eviction_policy,
                cfg.prefix_cache_capacity_blocks,
            )),
        })
    }

    /// Number of prompt tokens served from cache for this request. Always
    /// `< prompt_tokens` so prefill does at least one block of real work, and
    /// the cache is updated with this prompt's blocks. `priority` is the issuing
    /// request's priority, used only by the `priority` eviction policy. The
    /// hit-rate override is purely synthetic (does not touch the content-
    /// addressed cache).
    pub fn cached_tokens(&self, prompt_text: &str, prompt_tokens: usize, priority: i64) -> usize {
        if prompt_tokens == 0 {
            return 0;
        }
        if self.hit_rate > 0.0 {
            let c = (prompt_tokens as f64 * self.hit_rate).round() as usize;
            return c.min(prompt_tokens - 1);
        }
        if !self.enabled {
            return 0;
        }
        let bytes = prompt_text.as_bytes();
        if bytes.is_empty() {
            return 0;
        }
        // Bound hashing cost while preserving the cached-token fraction.
        let target_blocks = prompt_tokens
            .div_ceil(self.block_tokens)
            .clamp(1, MAX_BLOCKS_PER_REQUEST);
        let block_bytes = bytes.len().div_ceil(target_blocks).max(1);

        // Hashing is independent and expensive; keep it outside the shared cache
        // lock. Membership and touch remain interleaved to preserve self-eviction
        // under tight capacity.
        let mut chained: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis, just a seed
        let mut hashes: Vec<u64> = Vec::with_capacity(target_blocks);
        let mut lo = 0usize;
        while lo < bytes.len() {
            let hi = (lo + block_bytes).min(bytes.len());
            let mut hasher = Blake2s256::new();
            hasher.update(chained.to_le_bytes());
            hasher.update(&bytes[lo..hi]);
            let digest = hasher.finalize();
            chained = u64::from_le_bytes(digest[..8].try_into().unwrap());
            hashes.push(chained);
            lo = hi;
        }

        let mut matched = 0usize;
        let mut still_matching = true;
        let total = hashes.len();
        let mut cache = self.cache.lock();
        for &hash in &hashes {
            if still_matching && cache.contains(hash) {
                matched += 1;
            } else {
                still_matching = false;
            }
            cache.touch(hash, priority);
        }
        let cached = prompt_tokens * matched / total.max(1);
        cached.min(prompt_tokens - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(block_tokens: usize, capacity: usize) -> PrefixCache {
        let cfg = MockServerConfig {
            prefix_cache_block_tokens: block_tokens,
            prefix_cache_capacity_blocks: capacity,
            ..MockServerConfig::default()
        };
        PrefixCache::from_config(&cfg).unwrap()
    }

    #[test]
    fn cold_prompt_misses_then_warm_repeat_hits() {
        let pc = cache(4, 10_000);
        let text = "the quick brown fox jumps over the lazy dog and keeps running far away";
        assert_eq!(pc.cached_tokens(text, 64, 0), 0);
        let again = pc.cached_tokens(text, 64, 0);
        assert!(
            (60..64).contains(&again),
            "warm repeat should be near-full, got {again}"
        );
    }

    #[test]
    fn shared_prefix_partial_hit() {
        let pc = cache(4, 10_000);
        let shared = "SYSTEM PROMPT: you are a helpful assistant. ".repeat(4);
        pc.cached_tokens(&format!("{shared}question one about apples"), 80, 0);
        let cached = pc.cached_tokens(
            &format!("{shared}a totally different unique tail here"),
            80,
            0,
        );
        assert!(cached > 0, "shared prefix should hit");
        assert!(cached < 80, "unique tail must not be cached");
    }

    #[test]
    fn unique_prompts_never_hit() {
        let pc = cache(4, 10_000);
        assert_eq!(
            pc.cached_tokens("alpha beta gamma delta epsilon zeta", 48, 0),
            0
        );
        assert_eq!(
            pc.cached_tokens("completely different words here now ok", 48, 0),
            0
        );
    }

    #[test]
    fn capacity_eviction_goes_cold() {
        let pc = cache(4, 2);
        let a = "first prompt aaaa bbbb cccc dddd";
        pc.cached_tokens(a, 64, 0);
        for i in 0..50 {
            pc.cached_tokens(&format!("flood traffic number {i} xxxx yyyy"), 64, 0);
        }
        assert_eq!(
            pc.cached_tokens(a, 64, 0),
            0,
            "evicted prefix should be cold again"
        );
    }

    #[test]
    fn hit_rate_override_forces_fraction_without_content() {
        let cfg = MockServerConfig {
            prefix_cache_hit_rate: 0.6,
            ..MockServerConfig::default()
        };
        let pc = PrefixCache::from_config(&cfg).unwrap();
        assert_eq!(pc.cached_tokens("anything at all goes here", 100, 0), 60);
        assert_eq!(pc.cached_tokens("totally different text", 100, 0), 60);
        assert_eq!(pc.cached_tokens("x", 1, 0), 0);
    }

    #[test]
    fn lru_evicts_least_recently_accessed() {
        let mut c = BlockCache::new(EvictionPolicy::Lru, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0);
        c.touch(3, 0);
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn lfu_evicts_least_frequently_used() {
        let mut c = BlockCache::new(EvictionPolicy::Lfu, 2);
        c.touch(1, 0);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(3, 0);
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn fifo_evicts_earliest_inserted() {
        let mut c = BlockCache::new(EvictionPolicy::Fifo, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0);
        c.touch(3, 0);
        assert!(c.contains(2) && c.contains(3) && !c.contains(1));
    }

    #[test]
    fn mru_evicts_most_recently_accessed() {
        let mut c = BlockCache::new(EvictionPolicy::Mru, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0);
        c.touch(3, 0);
        assert!(c.contains(2) && c.contains(3) && !c.contains(1));
    }

    #[test]
    fn filo_evicts_latest_inserted() {
        let mut c = BlockCache::new(EvictionPolicy::Filo, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0);
        c.touch(3, 0);
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn priority_evicts_lowest_priority_first() {
        let mut c = BlockCache::new(EvictionPolicy::Priority, 2);
        c.touch(1, 5);
        c.touch(2, 1);
        c.touch(3, 9);
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn slru_evicts_probationary_before_protected() {
        let mut c = BlockCache::new(EvictionPolicy::Slru, 2);
        c.touch(1, 0);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(3, 0);
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }
}
