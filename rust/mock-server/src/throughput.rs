// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lock-free token-throughput tracker used to auto-scale DCGM load.
//!
//! Hot path (`record_tokens`) is a single `fetch_add` on an `AtomicU64`.
//! A background tokio task samples the counter on a fixed cadence, maintains
//! the sliding-window view, tracks peak throughput, and invokes the
//! registered DCGM load callback.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

pub type LoadCallback = Arc<dyn Fn(f64) + Send + Sync + 'static>;

/// Sampling cadence. At 100 Hz we see <1 % error on a 1 s window and don't
/// burn significant CPU.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(10);

pub struct Throughput {
    running_total: AtomicU64,
    max_observed_bits: AtomicU64,
    min_throughput_baseline: AtomicU32,
    window_ms: AtomicU32,
    inner: Mutex<Inner>,
    callback: Mutex<Option<LoadCallback>>,
}

struct Inner {
    samples: VecDeque<(Instant, u64)>,
    sampler_running: bool,
}

impl Throughput {
    pub fn new() -> Self {
        Self {
            running_total: AtomicU64::new(0),
            max_observed_bits: AtomicU64::new(0),
            min_throughput_baseline: AtomicU32::new(100),
            window_ms: AtomicU32::new(1_000),
            inner: Mutex::new(Inner {
                samples: VecDeque::new(),
                sampler_running: false,
            }),
            callback: Mutex::new(None),
        }
    }

    /// Register a callback and start the sampler. Safe to call multiple times —
    /// only the first call spawns the sampler task.
    pub fn register_callback(
        self: &Arc<Self>,
        cb: LoadCallback,
        min_throughput: u32,
        window_sec: f64,
    ) {
        *self.callback.lock() = Some(cb);
        self.min_throughput_baseline
            .store(min_throughput, Ordering::Relaxed);
        self.window_ms
            .store(((window_sec * 1000.0).max(1.0)) as u32, Ordering::Relaxed);

        let start_sampler = {
            let mut inner = self.inner.lock();
            if inner.sampler_running {
                false
            } else {
                inner.sampler_running = true;
                true
            }
        };
        if start_sampler {
            let this = Arc::clone(self);
            tokio::spawn(async move {
                this.sampler_loop().await;
            });
        }
    }

    async fn sampler_loop(self: Arc<Self>) {
        let mut tick = tokio::time::interval(SAMPLE_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            self.sample();
        }
    }

    fn sample(&self) {
        let now = Instant::now();
        let total = self.running_total.load(Ordering::Relaxed);
        let window_ms = self.window_ms.load(Ordering::Relaxed) as u64;

        let mut inner = self.inner.lock();
        inner.samples.push_back((now, total));
        let cutoff = now - Duration::from_millis(window_ms);
        while let Some(&(t, _)) = inner.samples.front() {
            if t < cutoff {
                inner.samples.pop_front();
            } else {
                break;
            }
        }

        let (throughput, _) = compute_throughput(&inner.samples, window_ms);
        drop(inner);

        loop {
            let cur_bits = self.max_observed_bits.load(Ordering::Relaxed);
            let cur = f64::from_bits(cur_bits);
            if throughput <= cur {
                break;
            }
            let new_bits = throughput.to_bits();
            if self
                .max_observed_bits
                .compare_exchange_weak(cur_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Callbacks may perform arbitrary work and must not hold the sample lock.
        if let Some(cb) = self.callback.lock().as_ref().cloned() {
            let peak = f64::from_bits(self.max_observed_bits.load(Ordering::Relaxed));
            let floor = self.min_throughput_baseline.load(Ordering::Relaxed) as f64;
            let effective = peak.max(floor);
            let load = (throughput / effective).clamp(0.0, 1.0);
            cb(load);
        }
    }

    /// Hot-path token counter — single relaxed atomic increment.
    #[inline]
    pub fn record_tokens(&self, count: u64) {
        self.running_total.fetch_add(count, Ordering::Relaxed);
    }

    /// Trigger an immediate sample (used in tests).
    pub fn flush_now(&self) {
        self.sample();
    }

    pub fn current_throughput(&self) -> f64 {
        let window_ms = self.window_ms.load(Ordering::Relaxed) as u64;
        let inner = self.inner.lock();
        compute_throughput(&inner.samples, window_ms).0
    }
}

fn compute_throughput(samples: &VecDeque<(Instant, u64)>, window_ms: u64) -> (f64, u64) {
    match (samples.front(), samples.back()) {
        (Some(&(_, first_total)), Some(&(_, last_total))) if samples.len() >= 2 => {
            let delta = last_total.saturating_sub(first_total);
            let tput = (delta as f64) * 1000.0 / (window_ms as f64);
            (tput, delta)
        }
        _ => (0.0, 0),
    }
}

impl Default for Throughput {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn callback_fires_via_sampler() {
        let tp = Arc::new(Throughput::new());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        tp.register_callback(
            Arc::new(move |_load: f64| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
            10,
            1.0,
        );
        tp.record_tokens(5);
        // Wait a few sample ticks (30 ms covers 3× 10ms sampling intervals).
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(counter.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn record_is_lock_free_and_additive() {
        let tp = Throughput::new();
        for _ in 0..1000 {
            tp.record_tokens(1);
        }
        assert_eq!(tp.running_total.load(Ordering::Relaxed), 1000);
    }
}
