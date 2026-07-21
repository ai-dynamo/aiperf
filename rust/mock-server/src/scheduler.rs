// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Step-based batched scheduler for the mock server.
//!
//! Models the dominant first-order behavior of a continuous-batching LLM
//! server: a global decode loop ticking every `step_ms`, admitting up to
//! `max_batch_size` decoders per step, plus a separate prefill chunk pool with
//! bounded `max_prefill_chunks_per_step`. Produces a real throughput-vs-
//! concurrency saturation knee at concurrency ~= `max_batch_size`, and (when
//! goodput-collapse is enabled) an actual tok/s *decrease* past the knee.
//!
//! Each waiter is a Tokio `oneshot`; admission sends the step index, while
//! cancellation or shutdown wakes the waiter by dropping its sender.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use aiperf_runtime::rng::RustRandomGenerator;
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::MockServerConfig;

/// Mean-1 lognormal multiplier with coefficient of variation `cv` (`cv<=0` -> 1.0).
fn lognormal(rng: &mut RustRandomGenerator, cv: f64) -> f64 {
    if cv <= 0.0 {
        return 1.0;
    }
    let sigma = (1.0 + cv * cv).ln().sqrt();
    let u1: f64 = rng.random().max(1e-12);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (sigma * z - 0.5 * sigma * sigma).exp()
}

struct Waiter {
    request_id: String,
    tx: oneshot::Sender<u64>,
}

#[derive(Default)]
struct Queues {
    decode: VecDeque<Waiter>,
    prefill: VecDeque<Waiter>,
}

/// Step-based scheduler. Owns one background tick task that wakes queued
/// decode/prefill waiters on each step.
pub struct BatchScheduler {
    enabled: bool,
    step_ms: f64,
    max_batch: usize,
    max_prefill_chunks: usize,
    prefill_chunk_tokens: usize,
    fixed_chunks: usize,
    work_cv: f64,
    admit_jitter_cv: f64,
    prefill_tput_exponent: f64,
    prefill_tput_ref: usize,
    goodput_enabled: bool,
    goodput_threshold: f64,
    goodput_slope: f64,
    goodput_floor: f64,
    queues: Mutex<Queues>,
    step_index: AtomicU64,
    stopped: AtomicBool,
    tick: Mutex<Option<JoinHandle<()>>>,
    admit_rng: Mutex<RustRandomGenerator>,
}

impl BatchScheduler {
    pub fn new(cfg: &MockServerConfig) -> Arc<Self> {
        Arc::new(Self {
            enabled: cfg.scheduler_enabled,
            step_ms: cfg.scheduler_step_ms,
            max_batch: cfg.scheduler_max_batch_size.max(1),
            max_prefill_chunks: cfg.scheduler_max_prefill_chunks_per_step.max(1),
            prefill_chunk_tokens: cfg.scheduler_prefill_chunk_tokens.max(1),
            fixed_chunks: cfg.scheduler_prefill_chunks_per_request,
            work_cv: cfg.scheduler_prefill_work_cv,
            admit_jitter_cv: cfg.scheduler_admit_jitter_cv,
            prefill_tput_exponent: cfg.scheduler_prefill_throughput_exponent,
            prefill_tput_ref: cfg.scheduler_prefill_throughput_ref,
            goodput_enabled: cfg.scheduler_goodput_collapse_enabled,
            goodput_threshold: cfg.scheduler_goodput_collapse_threshold,
            goodput_slope: cfg.scheduler_goodput_collapse_slope,
            goodput_floor: cfg.scheduler_goodput_collapse_floor,
            queues: Mutex::new(Queues::default()),
            step_index: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
            tick: Mutex::new(None),
            admit_rng: Mutex::new(match cfg.random_seed {
                Some(s) => RustRandomGenerator::from_seed(Some(s ^ 0x5c4ed_u64)),
                None => RustRandomGenerator::from_seed(None),
            }),
        })
    }

    /// Starts one tick task when enabled and running inside Tokio.
    pub fn start(self: &Arc<Self>) {
        if !self.enabled || tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let mut guard = self.tick.lock();
        if guard.is_some() {
            return;
        }
        let me = Arc::clone(self);
        *guard = Some(tokio::spawn(async move { me.tick_loop().await }));
        tracing::info!(
            step_ms = self.step_ms,
            max_batch = self.max_batch,
            max_prefill_chunks = self.max_prefill_chunks,
            prefill_chunk_tokens = self.prefill_chunk_tokens,
            goodput_collapse = self.goodput_enabled,
            "BatchScheduler started"
        );
    }

    /// Stop the tick loop and wake every queued waiter.
    pub async fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        let handle = self.tick.lock().take();
        if let Some(h) = handle {
            h.abort();
            let _ = h.await;
        }
        let mut q = self.queues.lock();
        q.decode.clear();
        q.prefill.clear();
    }

    pub fn step_index(&self) -> u64 {
        self.step_index.load(Ordering::Relaxed)
    }

    /// Block until this request's next decode token is admitted; returns the
    /// admitted step index.
    pub async fn next_decode_step(&self, request_id: &str) -> u64 {
        if !self.enabled || self.stopped.load(Ordering::Relaxed) {
            return self.step_index();
        }
        let (tx, rx) = oneshot::channel();
        self.queues.lock().decode.push_back(Waiter {
            request_id: request_id.to_string(),
            tx,
        });
        rx.await.unwrap_or_else(|_| self.step_index())
    }

    /// Block until all prefill chunks for this prompt have been admitted;
    /// returns the chunk count. The count is `scheduler_prefill_chunks_per_request`
    /// when set (ISL-independent), else `ceil(prompt_tokens / chunk_tokens)`, then
    /// scaled down by the prefix-cache hit fraction (`cached_tokens` skip prefill,
    /// floored at one block of real work), and finally scaled by a per-request
    /// mean-1 lognormal (seeded by `seed`) when `scheduler_prefill_work_cv > 0` so
    /// queue-wait/TTFT spreads request-to-request.
    pub async fn run_prefill(
        &self,
        request_id: &str,
        prompt_tokens: usize,
        cached_tokens: usize,
        seed: u64,
    ) -> usize {
        if !self.enabled || self.stopped.load(Ordering::Relaxed) {
            return 0;
        }
        let uncached = if prompt_tokens > 0 {
            (prompt_tokens.saturating_sub(cached_tokens) as f64 / prompt_tokens as f64)
                .clamp(0.0, 1.0)
        } else {
            1.0
        };
        let base = if self.fixed_chunks > 0 {
            ((self.fixed_chunks as f64 * uncached).round() as usize).max(1)
        } else if prompt_tokens == 0 {
            return 0;
        } else {
            let eff = ((prompt_tokens as f64 * uncached).round() as usize).max(1);
            eff.div_ceil(self.prefill_chunk_tokens).max(1)
        };
        let chunks = if self.work_cv > 0.0 {
            let mut rng = RustRandomGenerator::from_seed(Some(seed ^ 0x9E37_79B9_7F4A_7C15));
            ((base as f64 * lognormal(&mut rng, self.work_cv)).round() as usize).max(1)
        } else {
            base
        };
        for _ in 0..chunks {
            if self.stopped.load(Ordering::Relaxed) {
                break;
            }
            let (tx, rx) = oneshot::channel();
            self.queues.lock().prefill.push_back(Waiter {
                request_id: request_id.to_string(),
                tx,
            });
            let _ = rx.await;
        }
        chunks
    }

    /// Drop all pending waiters for a request (client disconnect). Dropping the
    /// sender wakes any currently-awaiting coroutine; removing the waiter keeps
    /// it from inflating queue depth (which would skew goodput accounting).
    pub fn cancel(&self, request_id: &str) {
        let mut q = self.queues.lock();
        q.decode.retain(|w| w.request_id != request_id);
        q.prefill.retain(|w| w.request_id != request_id);
    }

    async fn tick_loop(self: Arc<Self>) {
        // RealClock preserves sub-millisecond scheduler steps.
        let step_ns = ((self.step_ms * 1_000_000.0).max(0.0)) as i64;
        loop {
            aiperf_runtime::clock::sleep_ns(step_ns).await;
            if self.stopped.load(Ordering::Relaxed) {
                break;
            }
            self.step_index.fetch_add(1, Ordering::Relaxed);
            self.admit_prefill();
            self.admit_decode();
        }
    }

    fn jitter_budget(&self, base: usize) -> usize {
        if self.admit_jitter_cv <= 0.0 || base == 0 {
            return base;
        }
        let f = lognormal(&mut self.admit_rng.lock(), self.admit_jitter_cv);
        ((base as f64 * f).round() as usize).max(1)
    }

    /// Per-step prefill chunk budget after the sublinear throughput scaling.
    /// `occupancy` is the live in-flight count (prefill + decode waiters), which
    /// in a closed loop tracks client concurrency (each request holds exactly
    /// one waiter at a time). The budget grows as `(occupancy / ref)^exponent`,
    /// floored at the base so prefill never slows below nominal at low load.
    /// With `exponent` in (0,1) this makes TTFT grow as `C^(1 - exponent)`.
    fn effective_prefill_budget(&self, occupancy: usize) -> usize {
        if self.prefill_tput_exponent <= 0.0 {
            return self.max_prefill_chunks;
        }
        let occ_ref = if self.prefill_tput_ref > 0 {
            self.prefill_tput_ref
        } else {
            self.max_batch
        };
        let ratio = occupancy.max(1) as f64 / occ_ref.max(1) as f64;
        let scale = ratio.max(1.0).powf(self.prefill_tput_exponent);
        ((self.max_prefill_chunks as f64 * scale).round() as usize).max(self.max_prefill_chunks)
    }

    fn admit_prefill(&self) {
        let step = self.step_index();
        let mut q = self.queues.lock();
        let occupancy = q.prefill.len() + q.decode.len();
        let mut budget = self.jitter_budget(self.effective_prefill_budget(occupancy));
        while budget > 0 {
            match q.prefill.pop_front() {
                Some(w) => {
                    let _ = w.tx.send(step);
                    budget -= 1;
                }
                None => break,
            }
        }
    }

    fn admit_decode(&self) {
        let step = self.step_index();
        let mut q = self.queues.lock();
        let mut budget = self.jitter_budget(self.effective_decode_budget(q.decode.len()));
        while budget > 0 {
            match q.decode.pop_front() {
                Some(w) => {
                    let _ = w.tx.send(step);
                    budget -= 1;
                }
                None => break,
            }
        }
    }

    /// Per-step decode admit budget after goodput-collapse adjustment. When the
    /// decode queue grows past `threshold * max_batch`, the budget shrinks
    /// linearly toward `floor * max_batch` (>= 1).
    fn effective_decode_budget(&self, queue_len: usize) -> usize {
        if !self.goodput_enabled {
            return self.max_batch;
        }
        let ratio = queue_len as f64 / self.max_batch as f64;
        let overload = ratio - self.goodput_threshold;
        if overload <= 0.0 {
            return self.max_batch;
        }
        let shrink = (overload * self.goodput_slope).min(1.0 - self.goodput_floor);
        ((self.max_batch as f64 * (1.0 - shrink)) as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_batch: usize, step_ms: f64) -> MockServerConfig {
        MockServerConfig {
            scheduler_enabled: true,
            scheduler_step_ms: step_ms,
            scheduler_max_batch_size: max_batch,
            scheduler_max_prefill_chunks_per_step: 64,
            scheduler_prefill_chunk_tokens: 512,
            ..MockServerConfig::default()
        }
    }

    #[tokio::test]
    async fn single_request_admitted_on_next_step() {
        let sched = BatchScheduler::new(&cfg(4, 5.0));
        sched.start();
        let step = sched.next_decode_step("req-1").await;
        assert!(step >= 1, "first admitted decode step should be >= 1");
        sched.stop().await;
    }

    #[tokio::test]
    async fn oversubscription_serializes_admission() {
        let sched = BatchScheduler::new(&cfg(4, 2.0));
        sched.start();
        let futs: Vec<_> = (0..8)
            .map(|i| {
                let s = Arc::clone(&sched);
                async move { s.next_decode_step(&format!("r{i}")).await }
            })
            .collect();
        let steps = futures::future::join_all(futs).await;
        let min = *steps.iter().min().unwrap();
        let max = *steps.iter().max().unwrap();
        assert_eq!(max - min, 1, "admission spans exactly two steps");
        assert_eq!(steps.iter().filter(|&&s| s == min).count(), 4);
        assert_eq!(steps.iter().filter(|&&s| s == max).count(), 4);
        sched.stop().await;
    }

    #[tokio::test]
    async fn prefill_chunks_split_long_prompts() {
        let sched = BatchScheduler::new(&cfg(64, 1.0));
        sched.start();
        let chunks = sched.run_prefill("req-long", 1500, 0, 0).await;
        assert_eq!(chunks, 3, "ceil(1500 / 512) == 3 chunks");
        sched.stop().await;
    }

    #[tokio::test]
    async fn cached_prefix_reduces_prefill_chunks() {
        let mut c = cfg(64, 1.0);
        c.scheduler_prefill_chunks_per_request = 20;
        let sched = BatchScheduler::new(&c);
        sched.start();
        assert_eq!(sched.run_prefill("a", 1000, 0, 0).await, 20);
        assert_eq!(sched.run_prefill("b", 1000, 750, 0).await, 5);
        assert_eq!(sched.run_prefill("c", 1000, 1000, 0).await, 1);
        sched.stop().await;
    }

    #[tokio::test]
    async fn goodput_collapse_shrinks_budget() {
        let mut c = cfg(10, 5.0);
        c.scheduler_goodput_collapse_enabled = true;
        c.scheduler_goodput_collapse_threshold = 1.5;
        c.scheduler_goodput_collapse_slope = 0.5;
        c.scheduler_goodput_collapse_floor = 0.3;
        let sched = BatchScheduler::new(&c);
        assert_eq!(sched.effective_decode_budget(10), 10);
        assert_eq!(sched.effective_decode_budget(25), 5);
        assert_eq!(sched.effective_decode_budget(1000), 3);
    }

    #[tokio::test]
    async fn prefill_throughput_scales_sublinearly_with_occupancy() {
        let mut c = cfg(64, 5.0);
        c.scheduler_max_prefill_chunks_per_step = 4;
        c.scheduler_prefill_throughput_exponent = 0.585;
        c.scheduler_prefill_throughput_ref = 512;
        let sched = BatchScheduler::new(&c);
        assert_eq!(sched.effective_prefill_budget(256), 4);
        assert_eq!(sched.effective_prefill_budget(512), 4);
        assert_eq!(sched.effective_prefill_budget(1024), 6);
        c.scheduler_prefill_throughput_exponent = 0.0;
        assert_eq!(BatchScheduler::new(&c).effective_prefill_budget(1024), 4);
    }

    #[tokio::test]
    async fn disabled_passthrough() {
        let sched = BatchScheduler::new(&MockServerConfig::default());
        sched.start();
        assert_eq!(sched.run_prefill("r", 4000, 0, 0).await, 0);
        let _ = sched.next_decode_step("r").await;
    }
}
