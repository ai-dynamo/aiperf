// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request latency simulator.
//!
//! Two models, selected by `--scheduler-enabled`:
//!
//! * **Analytic** (default): an effective TTFT/ITL computed once per request
//!   from the base values plus ISL/OSL and concurrency terms, with optional
//!   lognormal jitter. Token `i` is scheduled at `start + ttft + itl * i` — no
//!   interior mutable state, so each streaming request owns its simulator.
//! * **Scheduler**: the first token blocks on the batched scheduler's prefill
//!   admission and later tokens on per-step decode admission, so latency
//!   emerges from batch contention (plus a small positive jitter).
//!
//! Timing runs on the `aiperf` [`RealClock`](aiperf_runtime::clock::RealClock) backend:
//! `now` is read off a shared [`RealClockAnchor`] and every wait uses
//! [`aiperf_runtime::clock::sleep_ns`] — the RealClock `timerfd` primitive with
//! nanosecond resolution, instead of `tokio::time`'s 1 ms wheel (which would
//! quantize a 5 ms ITL by ~20%). The `Clock` trait's own `sleep` is `!Send` /
//! `Rc`-based and cannot cross this crate's multi-threaded axum handler
//! boundary, so we use the anchor + the `Send` `sleep_ns` form of the same
//! primitive.

use std::sync::Arc;
use std::time::{Duration, Instant};

use aiperf_runtime::clock::{RealClockAnchor, sleep_ns};
use aiperf_runtime::rng::RandomGenerator;

use crate::config::MockServerConfig;
use crate::scheduler::BatchScheduler;

/// Milliseconds → integer nanoseconds, floored at 0.
#[inline]
fn ms_to_ns(ms: f64) -> i64 {
    (ms * 1_000_000.0).max(0.0) as i64
}

/// Lognormal multiplier with mean ~= 1.0 and coefficient of variation `cv`
/// (stddev/mean). `cv <= 0` returns 1.0. The `-sigma^2/2` term keeps the mean
/// at 1.0 so a base latency can be multiplied without bias.
fn lognormal_jitter(rng: &mut RandomGenerator, cv: f64) -> f64 {
    if cv <= 0.0 {
        return 1.0;
    }
    let sigma = (1.0 + cv * cv).ln().sqrt();
    // Box-Muller: z ~ N(0, 1) from two uniforms.
    let u1: f64 = rng.random().max(1e-12);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (sigma * z - 0.5 * sigma * sigma).exp()
}

/// Extra (>= 0) seconds to add as jitter on top of `base_ms`. Used in scheduler
/// mode, where the admit floor means we can only ever delay, not accelerate, so
/// faster-than-nominal samples contribute 0.
fn positive_jitter_extra_secs(rng: &mut RandomGenerator, base_ms: f64, cv: f64) -> f64 {
    if cv <= 0.0 || base_ms <= 0.0 {
        return 0.0;
    }
    let factor = lognormal_jitter(rng, cv);
    if factor <= 1.0 {
        return 0.0;
    }
    (factor - 1.0) * base_ms * 0.001
}

fn seeded_rng(seed: u64, salt: u64) -> RandomGenerator {
    RandomGenerator::from_seed(Some(seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

/// Per-request latency scheduler. Owned exclusively by one request.
pub struct LatencySimulator {
    /// Shared `RealClock` timeline; `now_ns` reads and target math run on it.
    anchor: RealClockAnchor,
    /// Request start, in ns on `anchor`'s timeline.
    start_ns: i64,
    /// Effective analytic TTFT (base + ISL + concurrency terms, jittered once), ns.
    ttft_ns: i64,
    /// Effective analytic ITL (base + OSL + concurrency terms, jittered once), ns.
    itl_ns: i64,
    /// Set in scheduler mode; admission drives timing instead of the delays above.
    sched: Option<Arc<BatchScheduler>>,
    request_key: String,
    isl: usize,
    /// Prompt tokens served from the KV cache; they skip prefill work.
    cached_tokens: usize,
    // Base values + CVs for scheduler-mode positive jitter.
    ttft_base_ms: f64,
    itl_base_ms: f64,
    ttft_cv: f64,
    itl_cv: f64,
    seed: u64,
}

impl LatencySimulator {
    /// Build the simulator for one request. `active_inflight` is the live
    /// in-flight request count used by the concurrency terms; `sched` is `Some`
    /// only when the batched scheduler is enabled.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        anchor: RealClockAnchor,
        cfg: &MockServerConfig,
        isl: usize,
        osl: usize,
        active_inflight: usize,
        sched: Option<Arc<BatchScheduler>>,
        request_key: String,
        cached_tokens: usize,
    ) -> Self {
        // Deterministic per-request jitter seed: FNV-1a hash of the unique key.
        let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
        for b in request_key.as_bytes() {
            seed ^= *b as u64;
            seed = seed.wrapping_mul(0x0000_0100_0000_01b3);
        }

        let mut rng = seeded_rng(seed, 1);
        let active = active_inflight as f64;
        // Prefix-cached tokens skip prefill, so the per-ISL-token TTFT term
        // scales with the uncached suffix only.
        let eff_isl = isl.saturating_sub(cached_tokens);
        let ttft_ms = (cfg.ttft
            + cfg.ttft_per_isl_token_ms * eff_isl as f64
            + cfg.ttft_concurrency_quad_ms * active * active)
            * lognormal_jitter(&mut rng, cfg.ttft_jitter_cv);
        // ITL jitter is applied once per request here (the Python model
        // resamples per token); identical unless itl_jitter_cv is set with the
        // scheduler off, and it preserves the lock-free cumulative-target design.
        let itl_ms =
            (cfg.itl + cfg.itl_per_osl_token_ms * osl as f64 + cfg.itl_concurrency_lin_ms * active)
                * lognormal_jitter(&mut rng, cfg.itl_jitter_cv);

        Self {
            anchor,
            start_ns: anchor.now_ns(),
            ttft_ns: ms_to_ns(ttft_ms),
            itl_ns: ms_to_ns(itl_ms),
            sched,
            request_key,
            isl,
            cached_tokens,
            ttft_base_ms: cfg.ttft,
            itl_base_ms: cfg.itl,
            ttft_cv: cfg.ttft_jitter_cv,
            itl_cv: cfg.itl_jitter_cv,
            seed,
        }
    }

    /// Zero-latency only when there's no scheduler and both delays are zero.
    #[inline]
    pub fn is_fast(&self) -> bool {
        self.sched.is_none() && self.ttft_ns == 0 && self.itl_ns == 0
    }

    /// Elapsed wall time since this request started, as a `Duration`, read off
    /// the `RealClock` timeline.
    #[inline]
    fn elapsed(&self) -> Duration {
        Duration::from_nanos((self.anchor.now_ns() - self.start_ns).max(0) as u64)
    }

    /// Wait until the emission time for token `index`. Returns the resume instant
    /// so the caller can compute TTFT/ITL.
    pub async fn wait_for_index(&self, index: usize) -> Instant {
        if let Some(sched) = &self.sched {
            if index == 0 {
                sched
                    .run_prefill(
                        &self.request_key,
                        self.isl.max(1),
                        self.cached_tokens,
                        self.seed,
                    )
                    .await;
                self.sleep_jitter_extra(self.ttft_base_ms, self.ttft_cv, 0)
                    .await;
            } else {
                sched.next_decode_step(&self.request_key).await;
                self.sleep_jitter_extra(self.itl_base_ms, self.itl_cv, index as u64)
                    .await;
            }
            return Instant::now();
        }
        if self.is_fast() {
            return Instant::now();
        }
        let target_ns = self.start_ns + self.ttft_ns + self.itl_ns * index as i64;
        let delta_ns = target_ns - self.anchor.now_ns();
        if delta_ns > 0 {
            sleep_ns(delta_ns).await;
        }
        Instant::now()
    }

    /// Wait for an entire (non-streaming) completion. Returns
    /// (measured_ttft, measured_decode) for metrics recording.
    pub async fn wait_for_tokens(&self, num_tokens: usize) -> (Duration, Duration) {
        if let Some(sched) = &self.sched {
            sched
                .run_prefill(
                    &self.request_key,
                    self.isl.max(1),
                    self.cached_tokens,
                    self.seed,
                )
                .await;
            self.sleep_jitter_extra(self.ttft_base_ms, self.ttft_cv, 0)
                .await;
            let measured_ttft = self.elapsed();
            for i in 0..num_tokens {
                sched.next_decode_step(&self.request_key).await;
                self.sleep_jitter_extra(self.itl_base_ms, self.itl_cv, (i + 1) as u64)
                    .await;
            }
            let total = self.elapsed();
            return (measured_ttft, total.saturating_sub(measured_ttft));
        }
        if self.is_fast() {
            return (Duration::ZERO, Duration::ZERO);
        }
        let ttft_target_ns = self.start_ns + self.ttft_ns;
        let delta_ns = ttft_target_ns - self.anchor.now_ns();
        if delta_ns > 0 {
            sleep_ns(delta_ns).await;
        }
        let measured_ttft = self.elapsed();

        let decode_target_ns = ttft_target_ns + self.itl_ns * num_tokens as i64;
        let delta_ns = decode_target_ns - self.anchor.now_ns();
        if delta_ns > 0 {
            sleep_ns(delta_ns).await;
        }
        let total = self.elapsed();
        (measured_ttft, total.saturating_sub(measured_ttft))
    }

    async fn sleep_jitter_extra(&self, base_ms: f64, cv: f64, salt: u64) {
        if cv <= 0.0 || base_ms <= 0.0 {
            return;
        }
        let extra_secs = {
            let mut rng = seeded_rng(self.seed, salt.wrapping_add(0xABCD));
            positive_jitter_extra_secs(&mut rng, base_ms, cv)
        };
        if extra_secs > 0.0 {
            sleep_ns((extra_secs * 1_000_000_000.0) as i64).await;
        }
    }
}

pub async fn wait_for_processing(base_ms: f64, per_unit_ms: f64, units: usize) {
    let total_ms = base_ms + per_unit_ms * (units as f64);
    if total_ms > 0.0 {
        sleep_ns(ms_to_ns(total_ms)).await;
    }
}
