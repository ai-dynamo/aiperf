// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared application state held inside axum's `State<Arc<AppState>>`.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use crate::config::MockServerConfig;
use crate::dcgm::{DcgmFaker, DcgmPool};
use crate::metrics::MetricRecorder;
use crate::prefix_cache::PrefixCache;
use crate::scheduler::BatchScheduler;

pub struct AppState {
    pub config: MockServerConfig,
    pub recorder: MetricRecorder,
    pub dcgm: DcgmPool,
    pub start_instant: Instant,
    pub start_wallclock: std::time::SystemTime,
    pub error_rng: Mutex<ErrorRng>,
    /// Step-based batched scheduler, present only when `--scheduler-enabled`.
    /// When set, the latency model is driven by scheduler admission instead of
    /// the closed-form analytic delays.
    pub scheduler: Option<Arc<BatchScheduler>>,
    /// KV-cache prefix-reuse model, present when `--prefix-cache-enabled` or a
    /// `--prefix-cache-hit-rate` override is set. Cached prefix tokens skip
    /// prefill (lower TTFT) and are reported in usage.
    pub prefix_cache: Option<Arc<PrefixCache>>,
}

pub struct ErrorRng {
    rng: rand::rngs::SmallRng,
}

impl ErrorRng {
    /// Build the error-injection RNG. When a root seed is provided, the
    /// actual seed is derived from the canonical `mock.errors` namespace.
    /// When `None`, we seed from OS entropy.
    pub fn new(seed: Option<u64>) -> Self {
        use rand::SeedableRng;
        let rng = match seed {
            Some(root) => rand::rngs::SmallRng::seed_from_u64(
                aiperf_rng::RngRoot::new(Some(root))
                    .derive_seed(aiperf_rng::namespace::MOCK_ERRORS)
                    .expect("a seeded RNG root always derives a seed"),
            ),
            None => rand::rngs::SmallRng::from_entropy(),
        };
        Self { rng }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f64 {
        use rand::Rng;
        self.rng.r#gen()
    }
}

impl AppState {
    pub fn build(config: MockServerConfig) -> Arc<Self> {
        let recorder = MetricRecorder::new();
        let dcgm_pool = Self::build_dcgm_pool(&config);
        // Create the scheduler when enabled and start its tick task (start is a
        // no-op outside a tokio runtime, so non-async test builds are safe).
        let scheduler = if config.scheduler_enabled {
            let s = BatchScheduler::new(&config);
            s.start();
            Some(s)
        } else {
            None
        };
        let prefix_cache = PrefixCache::from_config(&config).map(Arc::new);
        let state = Arc::new(AppState {
            error_rng: Mutex::new(ErrorRng::new(config.random_seed)),
            config: config.clone(),
            recorder,
            dcgm: dcgm_pool,
            start_instant: Instant::now(),
            start_wallclock: std::time::SystemTime::now(),
            scheduler,
            prefix_cache,
        });

        if config.dcgm_auto_load {
            let state_weak = Arc::downgrade(&state);
            state.recorder.throughput.register_callback(
                std::sync::Arc::new(move |load: f64| {
                    if let Some(s) = state_weak.upgrade() {
                        s.dcgm.set_load(load);
                    }
                }),
                config.dcgm_min_throughput,
                config.dcgm_window_sec,
            );
        }

        state
    }

    fn build_dcgm_pool(config: &MockServerConfig) -> DcgmPool {
        // Resolve per-faker DCGM seeds:
        // - Explicit `--dcgm-seed` wins (legacy behavior: seed + faker_idx).
        // - Otherwise derive from `--random-seed` via the RNG engine under
        //   the indexed `"mock.dcgm"` namespace. This keeps all mock-server RNGs under a
        //   single reproducibility root.
        let faker_seed = |idx: u64| -> Option<u64> {
            if let Some(s) = config.dcgm_seed {
                Some(s + idx)
            } else {
                config.random_seed.and_then(|root| {
                    aiperf_rng::RngRoot::new(Some(root))
                        .derive_indexed_seed(aiperf_rng::namespace::MOCK_DCGM, idx)
                })
            }
        };
        // Two fakers by default - matches Python behavior.
        let fakers = vec![
            DcgmFaker::new(
                &config.dcgm_gpu_name,
                config.dcgm_num_gpus,
                faker_seed(0),
                &config.dcgm_hostname,
            )
            .expect("dcgm faker 1"),
            DcgmFaker::new(
                &config.dcgm_gpu_name,
                config.dcgm_num_gpus,
                faker_seed(1),
                &config.dcgm_hostname,
            )
            .expect("dcgm faker 2"),
        ];
        DcgmPool::new(fakers)
    }

    pub fn uptime_secs(&self) -> f64 {
        self.start_wallclock
            .elapsed()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn inject_error(&self) -> bool {
        let rate = self.config.error_rate;
        if rate <= 0.0 {
            return false;
        }
        let v = self.error_rng.lock().next();
        v * 100.0 < rate
    }
}
