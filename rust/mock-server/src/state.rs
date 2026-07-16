// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared application state held inside axum's `State<Arc<AppState>>`.

use std::sync::Arc;
use std::time::Instant;

use aiperf::clock::RealClockAnchor;
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
    /// Monotonic `RealClock` timeline anchor for this process. Latency injection
    /// reads `now_ns` off this anchor and sleeps via the RealClock `timerfd`
    /// primitive ([`aiperf::clock::sleep_ns`]), so the mock's TTFT/ITL pacing has
    /// nanosecond resolution instead of `tokio::time`'s 1 ms wheel.
    pub clock_anchor: RealClockAnchor,
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
    /// Ground-truth-aware response mode, present when `--accuracy-dataset` is
    /// set. Requests whose prompt matches a dataset row return the (seeded)
    /// correct-or-wrong answer formatted for the benchmark grader.
    pub accuracy: Option<Arc<crate::accuracy::AccuracyDataset>>,
    /// Live tally of what the mock has actually answered correctly this run,
    /// exposed at `GET /accuracy` and on the Prometheus `/metrics` scrape.
    pub accuracy_live: crate::accuracy::AccuracyLive,
}

pub struct ErrorRng {
    rng: aiperf::rng::RandomGenerator,
}

impl ErrorRng {
    /// Build the error-injection RNG. When a root seed is provided, the
    /// actual stream is derived from the canonical `mock.errors` namespace.
    /// When `None`, the derived stream seeds from OS entropy.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = aiperf::rng::RngRoot::new(seed).derive(aiperf::rng::namespace::MOCK_ERRORS);
        Self { rng }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f64 {
        self.rng.random()
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
        let accuracy = config.accuracy_dataset.as_ref().map(|path| {
            match crate::accuracy::AccuracyDataset::load(std::path::Path::new(path), &config) {
                Ok(ds) => {
                    tracing::info!("Accuracy dataset loaded: {} rows from {}", ds.len(), path);
                    Arc::new(ds)
                }
                // A misconfigured accuracy dataset is a hard startup error: the
                // whole point of the run is ground-truth-aware responses.
                Err(e) => panic!("failed to load accuracy dataset: {e}"),
            }
        });
        let state = Arc::new(AppState {
            error_rng: Mutex::new(ErrorRng::new(config.random_seed)),
            config: config.clone(),
            recorder,
            dcgm: dcgm_pool,
            start_instant: Instant::now(),
            clock_anchor: RealClockAnchor::now(),
            start_wallclock: std::time::SystemTime::now(),
            scheduler,
            prefix_cache,
            accuracy,
            accuracy_live: crate::accuracy::AccuracyLive::default(),
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
                    aiperf::rng::RngRoot::new(Some(root))
                        .derive_indexed_seed(aiperf::rng::namespace::MOCK_DCGM, idx)
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
