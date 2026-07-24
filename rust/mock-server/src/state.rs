// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared application state held inside axum's `State<Arc<AppState>>`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use aiperf_runtime::clock::RealClockAnchor;
use bytes::Bytes;
use http_body_util::Empty;
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
    /// primitive ([`aiperf_runtime::clock::sleep_ns`]), so the mock's TTFT/ITL pacing has
    /// nanosecond resolution instead of `tokio::time`'s 1 ms wheel.
    pub clock_anchor: RealClockAnchor,
    pub start_wallclock: std::time::SystemTime,
    pub error_rng: Mutex<ErrorRng>,
    /// Dedicated seeded stream that cannot perturb `mock.errors` draw order.
    pub tool_call_rng: Mutex<ToolCallRng>,
    /// Scheduler admission drives latency when enabled.
    pub scheduler: Option<Arc<BatchScheduler>>,
    /// KV-cache prefix-reuse model. Disabled only when `--disable-prefix-cache`
    /// is set without a positive `--prefix-cache-hit-rate` override.
    pub prefix_cache: Option<Arc<PrefixCache>>,
    /// Control-plane profiler lifecycle counters exposed to focused tests.
    pub profiler_state: ProfilerState,
    /// Monotone reset generation incremented by `/reset_prefix_cache`.
    pub prefix_cache_generation: AtomicU64,
    /// Seeded benchmark answers loaded by `--accuracy-dataset`.
    pub accuracy: Option<Arc<crate::accuracy::AccuracyDataset>>,
    /// Live accuracy exposed by `GET /accuracy` and Prometheus metrics.
    pub accuracy_live: crate::accuracy::AccuracyLive,
    /// HTTP client for fetching `image_url`/`video_url` content when
    /// `--fetch-content-urls` is enabled. `None` disables fetching. `HttpConnector`
    /// speaks plain HTTP only (the content server serves plain HTTP) and ignores
    /// any ambient `HTTP_PROXY`, so localhost content URLs are reached directly.
    pub content_fetch_client: Option<ContentFetchClient>,
}

/// Pooled plain-HTTP client used to fetch content URLs. `Empty<Bytes>` is the
/// request body (GET carries none); responses are drained via `BodyExt::collect`.
pub type ContentFetchClient = hyper_util::client::legacy::Client<
    hyper_util::client::legacy::connect::HttpConnector,
    Empty<Bytes>,
>;

pub struct ErrorRng {
    rng: aiperf_runtime::rng::RustRandomGenerator,
}

impl ErrorRng {
    /// Build the error-injection RNG. When a root seed is provided, the
    /// actual stream is derived from the canonical `mock.errors` namespace.
    /// When `None`, the derived stream seeds from OS entropy.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = aiperf_runtime::rng::RngRoot::new(seed)
            .derive(aiperf_runtime::rng::namespace::MOCK_ERRORS);
        Self { rng }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f64 {
        self.rng.random()
    }
}

/// Seeded RNG for the per-request tool-call decision. Kept on a dedicated
/// `mock.tool_calls` stream (derived off the same `--random-seed` root) so its
/// draws are reproducible and independent of the error stream.
pub struct ToolCallRng {
    rng: aiperf_runtime::rng::RustRandomGenerator,
}

/// Server-local profiler lifecycle counters for control-route testing.
#[derive(Debug, Default)]
pub struct ProfilerState {
    starts: AtomicU64,
    stops: AtomicU64,
}

impl ProfilerState {
    /// Record one profiler-start control action.
    pub fn note_start(&self) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one profiler-stop control action.
    pub fn note_stop(&self) {
        self.stops.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the number of profiler starts observed so far.
    pub fn starts(&self) -> u64 {
        self.starts.load(Ordering::Relaxed)
    }

    /// Snapshot the number of profiler stops observed so far.
    pub fn stops(&self) -> u64 {
        self.stops.load(Ordering::Relaxed)
    }
}

impl ToolCallRng {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = aiperf_runtime::rng::RngRoot::new(seed).derive("mock.tool_calls");
        Self { rng }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f64 {
        self.rng.random()
    }
}

impl AppState {
    pub fn build(config: MockServerConfig) -> Arc<Self> {
        let recorder = MetricRecorder::with_enabled(!config.no_metrics);
        let dcgm_pool = Self::build_dcgm_pool(&config);
        let scheduler = if config.scheduler_enabled {
            let s = BatchScheduler::new(&config);
            s.start();
            Some(s)
        } else {
            None
        };
        let prefix_cache = PrefixCache::from_config(&config).map(Arc::new);
        let content_fetch_client = config.fetch_content_urls.then(|| {
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http()
        });
        let accuracy = config.accuracy_dataset.as_ref().map(|path| {
            match crate::accuracy::AccuracyDataset::load(std::path::Path::new(path), &config) {
                Ok(ds) => {
                    tracing::info!("Accuracy dataset loaded: {} rows from {}", ds.len(), path);
                    Arc::new(ds)
                }
                // Accuracy mode cannot satisfy its contract without the dataset.
                Err(e) => panic!("failed to load accuracy dataset: {e}"),
            }
        });
        let state = Arc::new(AppState {
            error_rng: Mutex::new(ErrorRng::new(config.random_seed)),
            tool_call_rng: Mutex::new(ToolCallRng::new(config.random_seed)),
            config: config.clone(),
            recorder,
            dcgm: dcgm_pool,
            start_instant: Instant::now(),
            clock_anchor: RealClockAnchor::now(),
            start_wallclock: std::time::SystemTime::now(),
            scheduler,
            prefix_cache,
            profiler_state: ProfilerState::default(),
            prefix_cache_generation: AtomicU64::new(0),
            accuracy,
            accuracy_live: crate::accuracy::AccuracyLive::default(),
            content_fetch_client,
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
        // An explicit `--dcgm-seed` takes precedence; otherwise each faker uses
        // an indexed `mock.dcgm` stream under `--random-seed`.
        let faker_seed = |idx: u64| -> Option<u64> {
            if let Some(s) = config.dcgm_seed {
                Some(s + idx)
            } else {
                config.random_seed.and_then(|root| {
                    aiperf_runtime::rng::RngRoot::new(Some(root))
                        .derive_indexed_seed(aiperf_runtime::rng::namespace::MOCK_DCGM, idx)
                })
            }
        };
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

    /// Decide whether to inject a pre-stream error for this request, and if so
    /// which HTTP status code to return.
    ///
    /// Both draws use `mock.errors`; the status is uniform over
    /// `--error-status-codes`, whose default is `[500]`.
    pub fn inject_error_status(&self) -> Option<u16> {
        let rate = self.config.error_rate;
        if rate <= 0.0 {
            return None;
        }
        let mut rng = self.error_rng.lock();
        if rng.next() * 100.0 >= rate {
            return None;
        }
        let codes = &self.config.error_status_codes;
        if codes.is_empty() {
            return Some(500);
        }
        let idx = ((rng.next() * codes.len() as f64) as usize).min(codes.len() - 1);
        Some(codes[idx])
    }

    /// Seeded decision whether a *streaming* request should fail mid-stream
    /// (emit a few token frames, then a terminal `event: error` SSE frame). The
    /// draw comes from the same `mock.errors` stream, so it is reproducible
    /// under `--random-seed`. `--error-midstream-rate` is a 0.0–1.0 probability.
    pub fn inject_midstream(&self) -> bool {
        let rate = self.config.error_midstream_rate;
        if rate <= 0.0 {
            return false;
        }
        self.error_rng.lock().next() < rate
    }

    /// Seeded decision whether this chat request should answer with a function
    /// tool call instead of a plain assistant turn. The draw comes from the
    /// dedicated `mock.tool_calls` stream, so it is reproducible under
    /// `--random-seed`. `--tool-call-rate` is a 0.0–1.0 probability.
    pub fn inject_tool_call(&self) -> bool {
        let rate = self.config.tool_call_rate;
        if rate <= 0.0 {
            return false;
        }
        self.tool_call_rng.lock().next() < rate
    }

    /// Borrow the profiler lifecycle counters for assertions.
    pub fn profiler_state(&self) -> &ProfilerState {
        &self.profiler_state
    }

    /// Count successful prefix-cache reset control requests.
    pub fn prefix_cache_generation(&self) -> u64 {
        self.prefix_cache_generation.load(Ordering::Relaxed)
    }

    /// Reset prefix-cache state and advance the observable generation counter.
    pub fn reset_prefix_cache(&self) {
        if let Some(prefix_cache) = &self.prefix_cache {
            prefix_cache.clear();
        }
        self.prefix_cache_generation.fetch_add(1, Ordering::Relaxed);
    }
}
