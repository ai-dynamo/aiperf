// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-addressed block-level prefix (KV-cache) reuse model.
//!
//! Mirrors what vLLM/SGLang do: a prompt is split into fixed-size blocks, each
//! block is chain-hashed (its hash folds in every preceding block, so an
//! identical prefix yields identical block hashes), and the longest run of
//! leading blocks already resident in the cache is served from cache — those
//! tokens skip prefill, lowering TTFT. After lookup the request's blocks are
//! inserted into a capacity-bounded cache, so an aged-out prefix goes cold and
//! stops hitting. The cached token count is reported back as
//! `usage.prompt_tokens_details.cached_tokens`.
//!
//! The eviction policy is configurable (`--prefix-cache-eviction-policy`),
//! mirroring SGLang's `--radix-eviction-policy`. See [`EvictionPolicy`].
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

/// KV-cache eviction policy, mirroring SGLang's `--radix-eviction-policy`
/// (`srt/mem_cache/evict_policy.py`). Under capacity pressure the block with the
/// smallest "eviction key" is removed first; each variant matches the
/// corresponding SGLang `EvictionStrategy.get_priority`. `lru` is the default.
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
    /// Eviction key (smaller is evicted first), mirroring SGLang's
    /// `EvictionStrategy.get_priority` for each policy.
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
/// Fidelity note: a new block frees space *before* it is inserted, so the
/// incoming (in-flight) block is never its own eviction victim — the mock's
/// stand-in for SGLang reference-counting the running request's nodes. Blocks of
/// *prior* requests are not pinned, so a very long prompt under tight capacity
/// can still evict its own earlier blocks mid-scan (as a real undersized cache
/// would).
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
            // Existing block: re-key as just-accessed without changing residency.
            self.order.remove(&m.evict_key(self.policy));
            m.last_access = now;
            m.hit_count += 1;
            m.priority = priority;
            self.order.insert(m.evict_key(self.policy), hash);
            self.blocks.insert(hash, m);
            return;
        }
        // New block: free space first so the incoming block is never the victim.
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
    /// Build from config; `None` when prefix caching is disabled and no hit-rate
    /// override is requested (so the hot path stays a plain `Option`).
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
        // Block count tracks token count, so matched_blocks / total_blocks is a
        // faithful cached-token fraction regardless of bytes-per-token. Capped so
        // token-granular (block_tokens=1) matching stays cheap on huge prompts.
        let target_blocks = prompt_tokens
            .div_ceil(self.block_tokens)
            .clamp(1, MAX_BLOCKS_PER_REQUEST);
        let block_bytes = bytes.len().div_ceil(target_blocks).max(1);

        // Chain-hash every block up front, outside the cache lock: blake2 is the
        // expensive per-request work and needs no shared state, so serializing it
        // under the lock would bottleneck all requests. Only the cheap
        // contains/touch bookkeeping below holds the lock, and it stays interleaved
        // to preserve mid-scan self-eviction fidelity under tight capacity.
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
        }; // prefix cache on by default
        PrefixCache::from_config(&cfg).unwrap()
    }

    #[test]
    fn cold_prompt_misses_then_warm_repeat_hits() {
        let pc = cache(4, 10_000);
        let text = "the quick brown fox jumps over the lazy dog and keeps running far away";
        // First sight: nothing cached yet.
        assert_eq!(pc.cached_tokens(text, 64, 0), 0);
        // Exact repeat: almost everything cached (all but the final block of work).
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
        // A different request reusing the same leading prefix hits on the prefix
        // only — not on its unique tail.
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
        let pc = cache(4, 2); // room for ~2 blocks only
        let a = "first prompt aaaa bbbb cccc dddd";
        pc.cached_tokens(a, 64, 0); // warm a (many blocks -> immediately over capacity)
        // Flood with unrelated traffic to evict a's blocks.
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
        // 60% of 100 tokens cached, regardless of (unique) content.
        assert_eq!(pc.cached_tokens("anything at all goes here", 100, 0), 60);
        assert_eq!(pc.cached_tokens("totally different text", 100, 0), 60);
        // Never caches the whole prompt.
        assert_eq!(pc.cached_tokens("x", 1, 0), 0);
    }

    // --- Eviction-policy unit tests --------------------------------------------
    // Each drives the same access sequence through a capacity-2 cache and asserts
    // which block the policy evicts when a third block arrives. last_access order
    // after the shared prelude is block 2 (oldest), then block 1 (re-accessed).

    #[test]
    fn lru_evicts_least_recently_accessed() {
        let mut c = BlockCache::new(EvictionPolicy::Lru, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0); // re-access 1 -> 2 is now least-recently-used
        c.touch(3, 0); // evicts 2
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn lfu_evicts_least_frequently_used() {
        let mut c = BlockCache::new(EvictionPolicy::Lfu, 2);
        c.touch(1, 0);
        c.touch(1, 0); // hit_count(1) = 2
        c.touch(2, 0); // hit_count(2) = 1
        c.touch(3, 0); // evicts the least-frequent block 2, not the recent-but-frequent 1
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn fifo_evicts_earliest_inserted() {
        let mut c = BlockCache::new(EvictionPolicy::Fifo, 2);
        c.touch(1, 0); // created first
        c.touch(2, 0);
        c.touch(1, 0); // re-access does not change creation order
        c.touch(3, 0); // evicts the earliest-created block 1
        assert!(c.contains(2) && c.contains(3) && !c.contains(1));
    }

    #[test]
    fn mru_evicts_most_recently_accessed() {
        let mut c = BlockCache::new(EvictionPolicy::Mru, 2);
        c.touch(1, 0);
        c.touch(2, 0);
        c.touch(1, 0); // 1 is now most-recently-used (and the incoming 3 is pinned)
        c.touch(3, 0); // evicts the most-recently-used resident block 1
        assert!(c.contains(2) && c.contains(3) && !c.contains(1));
    }

    #[test]
    fn filo_evicts_latest_inserted() {
        let mut c = BlockCache::new(EvictionPolicy::Filo, 2);
        c.touch(1, 0);
        c.touch(2, 0); // 2 is the latest-inserted resident block
        c.touch(1, 0); // re-access does not change creation order
        c.touch(3, 0); // evicts the latest-created resident block 2
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn priority_evicts_lowest_priority_first() {
        let mut c = BlockCache::new(EvictionPolicy::Priority, 2);
        c.touch(1, 5);
        c.touch(2, 1); // lowest priority
        c.touch(3, 9); // evicts the lowest-priority block 2
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }

    #[test]
    fn slru_evicts_probationary_before_protected() {
        let mut c = BlockCache::new(EvictionPolicy::Slru, 2);
        c.touch(1, 0);
        c.touch(1, 0); // hit_count(1) = 2 -> protected
        c.touch(2, 0); // hit_count(2) = 1 -> probationary, and more recent than 1
        c.touch(3, 0); // evicts the probationary block 2 despite it being newer
        assert!(c.contains(1) && c.contains(3) && !c.contains(2));
    }
}
