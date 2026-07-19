// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Characterization tests for `workers > 1` execution.
//!
//! Rate-based phases partition request budgets; `user_centric` and
//! `fixed_schedule` partition conversations. Static-accuracy evaluators remain
//! coordinator-local while dispatch and capture are sharded.
//!
//! Multi-worker execution uses real clocks because `SimClock` advances only its
//! own reactor. These tests therefore assert exact data and record-count
//! invariants while checking timing only for presence and positivity.

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::io::Read;
    use std::net::TcpListener as StdTcpListener;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use futures::stream;
    use http_body::Frame;
    use http_body_util::StreamBody;
    use hyper::service::service_fn;
    use hyper::{Request, Response as HttpResponse};
    use hyper_util::rt::TokioIo;
    use serde_json::{Value, json};

    use crate::endpoints::{EndpointId, RawEndpointConfig};
    use crate::engine::execute::{
        NativeDatasetPlan, NativeEndpointPlan, NativeRunSpec, NativeSidecarPlan,
        build_synthetic_dataset, execute_prepared_native_plan_uncommitted_selected,
    };
    use crate::engine::execution_factories::native_execution_factories;
    use crate::engine::protocol::{
        ArtifactSpec, MetricsSpec, ModelsSpec, PhaseSpec, TokenizerSpec,
    };
    use crate::engine::registry::ValidatedEndpointProfileV2;
    use crate::engine::sidecar_input::{
        BuiltinRunnerSidecarInputAdapterResolver, PreparedSidecarInputs,
        SidecarInputAdapterResolver,
    };
    use crate::engine::turn_execution::{HttpExecutionFactory, RequestExecutorFactory};
    use crate::extensions::AIPerfRegistry;
    use crate::rng::RngRoot;
    use crate::transport::core::ConnectionReuseStrategy;

    const FIXED_ISL: u64 = 12;
    const FIXED_OSL: usize = 6;
    const MOCK_TTFT_MS: u64 = 8;
    const MOCK_ITL_MS: u64 = 2;

    /// Fixed-latency OpenAI-compatible SSE server.
    ///
    /// A multi-threaded runtime accepts concurrent worker connections. Responses
    /// contain `FIXED_OSL` timed chunks, authoritative usage, and `[DONE]`.
    ///
    /// Tracks the peak number of requests concurrently mid-response (from first
    /// scheduled byte through `[DONE]`) via `current`/`peak`, so a test can
    /// prove the actually-observed concurrency at the wire never exceeded an
    /// admission cap — independent of what the client-side admission bookkeeping
    /// itself claims.
    struct FixedMock {
        base_url: String,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
        // Held only so `Drop`/reuse keeps a live handle; the increment/decrement
        // pair in `serve_sse` is the actual instrumentation, `peak` is what
        // tests read.
        #[allow(dead_code)]
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl FixedMock {
        fn spawn() -> Self {
            let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let base_url = format!("http://{addr}");
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
            let current = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let thread = std::thread::Builder::new()
                .name("fixed-mock".into())
                .spawn({
                    let current = current.clone();
                    let peak = peak.clone();
                    move || {
                        let runtime = tokio::runtime::Builder::new_multi_thread()
                            .worker_threads(2)
                            .enable_all()
                            .build()
                            .unwrap();
                        runtime.block_on(async move {
                            let listener = tokio::net::TcpListener::from_std(
                                std::net::TcpListener::from(listener),
                            )
                            .unwrap();
                            let mut shutdown_rx = shutdown_rx;
                            loop {
                                tokio::select! {
                                    _ = &mut shutdown_rx => break,
                                    accepted = listener.accept() => {
                                        let Ok((stream, _)) = accepted else { continue };
                                        let current = current.clone();
                                        let peak = peak.clone();
                                        tokio::spawn(async move {
                                            let service = service_fn(move |request| {
                                                serve_sse(request, current.clone(), peak.clone())
                                            });
                                            let _ = hyper::server::conn::http1::Builder::new()
                                                .serve_connection(TokioIo::new(stream), service)
                                                .await;
                                        });
                                    }
                                }
                            }
                        });
                    }
                })
                .unwrap();
            Self {
                base_url,
                shutdown: Some(shutdown_tx),
                thread: Some(thread),
                current,
                peak,
            }
        }

        /// The maximum number of requests observed concurrently mid-response
        /// (server-side, wire-observed — not the client's own admission
        /// bookkeeping) since this mock was spawned or last reset.
        fn peak_concurrent(&self) -> usize {
            self.peak.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// Reset the peak-concurrency high-water mark so the same mock (and its
        /// already-warm connection pool) can be reused for a second run without
        /// carrying over the first run's peak.
        fn reset_peak(&self) {
            self.peak.store(0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl Drop for FixedMock {
        fn drop(&mut self) {
            if let Some(tx) = self.shutdown.take() {
                let _ = tx.send(());
            }
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }

    async fn serve_sse(
        _request: Request<hyper::body::Incoming>,
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<
        HttpResponse<StreamBody<impl stream::Stream<Item = Result<Frame<Bytes>, Infallible>>>>,
        Infallible,
    > {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Frame<Bytes>, Infallible>>();
        tokio::spawn(async move {
            let now = current.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            peak.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(MOCK_TTFT_MS)).await;
            for index in 0..FIXED_OSL {
                if index > 0 {
                    tokio::time::sleep(Duration::from_millis(MOCK_ITL_MS)).await;
                }
                let chunk = json!({
                    "id": "chatcmpl-fixed",
                    "object": "chat.completion.chunk",
                    "model": "mock-model",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "x"},
                        "finish_reason": Value::Null,
                    }],
                });
                let frame = format!("data: {chunk}\n\n");
                if tx.send(Ok(Frame::data(Bytes::from(frame)))).is_err() {
                    current.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    return;
                }
            }
            let usage = json!({
                "id": "chatcmpl-fixed",
                "object": "chat.completion.chunk",
                "model": "mock-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": FIXED_ISL,
                    "completion_tokens": FIXED_OSL,
                    "total_tokens": FIXED_ISL + FIXED_OSL as u64,
                },
            });
            let _ = tx.send(Ok(Frame::data(Bytes::from(format!("data: {usage}\n\n")))));
            let _ = tx.send(Ok(Frame::data(Bytes::from_static(b"data: [DONE]\n\n"))));
            current.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        });
        let body = StreamBody::new(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        Ok(HttpResponse::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(body)
            .unwrap())
    }

    fn models() -> ModelsSpec {
        serde_json::from_value(json!({
            "strategy": "round_robin",
            "items": [{"name": "mock-model"}],
        }))
        .unwrap()
    }

    fn synthetic_spec(
        entries: usize,
        turns: usize,
    ) -> crate::engine::dataset_input::SyntheticDatasetSpec {
        let mut spec = json!({
            "entries": entries,
            "random_seed": 7,
            "sampling": "sequential",
            "prompts": {
                "batch_size": 1,
                "sequence_distribution": [{
                    "isl": {"value": FIXED_ISL as f64},
                    "osl": {"value": FIXED_OSL as f64},
                    "probability": 100.0,
                }],
            },
        });
        if turns > 1 {
            spec["turns"] = json!({"value": turns as f64});
            spec["turn_delay_ms"] = json!({"value": 0.0});
        }
        serde_json::from_value(spec).unwrap()
    }

    fn block_on_local<F: std::future::Future>(fut: F) -> F::Output {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, fut)
    }

    fn build_dataset(
        registry: &AIPerfRegistry,
        entries: usize,
        turns: usize,
    ) -> crate::engine::dataset_input::PreparedDatasetInput {
        let spec = synthetic_spec(entries, turns);
        let tokenizer = crate::dataset::tokenizer::TiktokenTokenizer::builtin();
        let dataset = block_on_local(build_synthetic_dataset(
            registry,
            &spec,
            &models(),
            RngRoot::new(Some(7)),
            &tokenizer,
            false,
            Arc::new(crate::dataset::NativeSyntheticMediaGeneratorFactory::default()),
            false,
        ))
        .unwrap();
        crate::engine::dataset_input::PreparedDatasetInput {
            dataset,
            random_seed: Some(7),
            default_output_tokens: FIXED_OSL,
        }
    }

    fn build_mooncake_dataset(
        registry: &AIPerfRegistry,
        entries: usize,
    ) -> crate::engine::dataset_input::PreparedDatasetInput {
        let rows: Vec<Value> = (0..entries)
            .map(|index| {
                json!({
                    "timestamp": (index as f64) * 5.0,
                    "input_length": FIXED_ISL,
                    "output_length": FIXED_OSL,
                    "session_id": format!("conv-{index}"),
                })
            })
            .collect();
        let tokenizer = crate::dataset::tokenizer::TiktokenTokenizer::builtin();
        let mut compose =
            crate::dataset::compose::ComposeConfig::new("mock-model", RngRoot::new(Some(7)));
        compose.models = vec![crate::dataset::model::ModelId::from("mock-model")];
        let mut load = crate::dataset::loader::LoadConfig::new(
            crate::dataset::loader::DatasetSource::Inline(json!(rows)),
        );
        load.sampling_strategy = Some("sequential".into());
        let dataset = block_on_local(registry.dataset_formats().build_dataset(
            Some("mooncake_trace"),
            &load,
            &compose,
            &tokenizer,
        ))
        .unwrap();
        crate::engine::dataset_input::PreparedDatasetInput {
            dataset,
            random_seed: Some(7),
            default_output_tokens: FIXED_OSL,
        }
    }

    fn empty_sidecars() -> PreparedSidecarInputs {
        BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&[])
            .unwrap()
    }

    fn endpoint_profile(base_url: &str) -> Arc<Vec<ValidatedEndpointProfileV2>> {
        Arc::new(vec![ValidatedEndpointProfileV2 {
            profile_id: "default".into(),
            endpoint_id: EndpointId::new("chat").unwrap(),
            config: RawEndpointConfig {
                urls: vec![base_url.to_string()],
                streaming: true,
                ..RawEndpointConfig::default()
            },
            connection_reuse: ConnectionReuseStrategy::Pooled,
            client: Default::default(),
            session_header: None,
        }])
    }

    fn plan(
        base_url: &str,
        artifact_dir: &Path,
        workers: usize,
        dataset: crate::engine::dataset_input::PreparedDatasetInput,
        phase: PhaseSpec,
    ) -> NativeRunSpec {
        plan_with_dispatch(
            base_url,
            artifact_dir,
            workers,
            dataset,
            phase,
            crate::engine::protocol::DispatchMode::Sharded,
        )
    }

    fn plan_with_dispatch(
        base_url: &str,
        artifact_dir: &Path,
        workers: usize,
        dataset: crate::engine::dataset_input::PreparedDatasetInput,
        phase: PhaseSpec,
        dispatch_mode: crate::engine::protocol::DispatchMode,
    ) -> NativeRunSpec {
        NativeRunSpec {
            benchmark_id: "characterization".into(),
            random_seed: Some(7),
            workers,
            artifact_dir: artifact_dir.to_path_buf(),
            models: models(),
            endpoint: NativeEndpointPlan::Prepared(endpoint_profile(base_url)),
            dataset: NativeDatasetPlan::PreparedLinear(dataset),
            tokenizer: TokenizerSpec {
                name: "builtin".into(),
                apply_chat_template: false,
                server_url: None,
            },
            phases: vec![phase],
            metrics: MetricsSpec::default(),
            artifacts: ArtifactSpec {
                records_path: Some("profile_export.jsonl".into()),
                ..ArtifactSpec::default()
            },
            sidecars: NativeSidecarPlan::Prepared(Arc::new(empty_sidecars())),
            user_files: Vec::new(),
            failure_policy: None,
            native_otel_enabled: false,
            transport: None,
            dispatch_mode,
        }
    }

    fn run_and_read_records(
        registry: &AIPerfRegistry,
        mock: &FixedMock,
        workers: usize,
        dataset: crate::engine::dataset_input::PreparedDatasetInput,
        phase: PhaseSpec,
    ) -> Vec<Value> {
        let artifact_dir = tempfile::tempdir().unwrap();
        let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
        let factories = native_execution_factories();
        let spec = plan(&mock.base_url, artifact_dir.path(), workers, dataset, phase);
        let report = execute_prepared_native_plan_uncommitted_selected(
            spec,
            request_executor,
            &factories,
            registry,
            None,
        )
        .expect("native run must complete");
        assert!(
            report_error_count(&report) == 0,
            "expected zero profiling errors, report: {report:?}"
        );
        read_records(artifact_dir.path())
    }

    fn report_error_count<T: serde::Serialize>(report: &T) -> usize {
        // Structural access avoids depending on the private report type.
        let value = serde_json::to_value(report).unwrap();
        value
            .get("errors")
            .and_then(Value::as_array)
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(|error| error.get("count").and_then(Value::as_u64))
                    .sum::<u64>() as usize
            })
            .unwrap_or(0)
    }

    fn read_records(artifact_dir: &Path) -> Vec<Value> {
        let path = artifact_dir.join("profile_export.jsonl");
        let mut contents = String::new();
        std::fs::File::open(&path)
            .unwrap_or_else(|error| panic!("opening {}: {error}", path.display()))
            .read_to_string(&mut contents)
            .unwrap();
        contents
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|row| {
                row.get("metadata")
                    .and_then(|m| m.get("benchmark_phase"))
                    .map(|phase| phase != "warmup" && phase != "Warmup")
                    .unwrap_or(true)
            })
            .collect()
    }

    fn metric(row: &Value, tag: &str) -> Option<f64> {
        row.get("metrics")
            .and_then(|metrics| metrics.get(tag))
            .and_then(|metric| metric.get("value"))
            .and_then(Value::as_f64)
    }

    fn data_key(row: &Value) -> (i64, i64) {
        (
            metric(row, "input_sequence_length").unwrap().round() as i64,
            metric(row, "output_sequence_length").unwrap().round() as i64,
        )
    }

    fn assert_pinned_records(rows: &[Value], expected_count: usize, expect_fixed_isl: bool) {
        assert_eq!(
            rows.len(),
            expected_count,
            "profiling record count must equal the dispatched request budget"
        );
        for row in rows {
            assert!(
                row.get("error").map(Value::is_null).unwrap_or(true),
                "no record may carry an error: {row}"
            );
            assert_eq!(
                metric(row, "output_sequence_length").unwrap().round() as usize,
                FIXED_OSL,
                "OSL is reconciled to the mock's fixed completion_tokens"
            );
            let isl = metric(row, "input_sequence_length").expect("input_sequence_length present");
            if expect_fixed_isl {
                assert_eq!(
                    isl.round() as u64,
                    FIXED_ISL,
                    "single-turn ISL is the fixed synthetic input length"
                );
            } else {
                assert!(isl > 0.0, "ISL must be positive: {isl}");
            }
            let latency = metric(row, "request_latency").expect("request_latency present");
            assert!(latency > 0.0, "request_latency must be positive: {latency}");
            assert!(
                metric(row, "inter_token_latency").is_some(),
                "inter_token_latency present for multi-token records"
            );
            assert!(
                metric(row, "time_to_first_token")
                    .map(|t| t > 0.0)
                    .unwrap_or(false),
                "time_to_first_token present and positive"
            );
        }
    }

    fn sorted_data_keys(rows: &[Value]) -> Vec<(i64, i64)> {
        let mut keys: Vec<(i64, i64)> = rows.iter().map(data_key).collect();
        keys.sort_unstable();
        keys
    }

    fn distinct_data_keys(rows: &[Value]) -> Vec<(i64, i64)> {
        let mut keys: Vec<(i64, i64)> = rows.iter().map(data_key).collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Proves the whole point of `DispatchMode::Global`: with a concurrency cap
    /// that does NOT evenly divide `workers` (`concurrency = 3`, `workers = 4`),
    /// `Sharded` mode's per-thread `owned_positions(cap, t, workers).max(1)`
    /// floor over-subscribes — one thread's share rounds down to `0`, which the
    /// `.max(1)` floor bumps back up to `1`, so `Sharded`'s aggregate cap across
    /// all 4 threads is `4`, not the authored `3` (see
    /// `sharded_scheduled::slice_phase_for_thread`'s "Admission caps are floored
    /// to one" doc comment). `Global` mode instead admits every thread from the
    /// SAME shared `GlobalSlotPool`, so the wire-observed peak concurrency
    /// across all worker threads combined never exceeds the authored cap of 3,
    /// even though each thread issues without any local partition of it.
    ///
    /// Peak concurrency is measured server-side (`FixedMock::peak_concurrent`),
    /// not from client-side admission bookkeeping, so this is a true
    /// end-to-end proof of the aggregate cap actually enforced on the wire —
    /// the same kind of proof `GlobalSlotPool`'s own cross-OS-thread test uses,
    /// applied here at the full `ScheduledRuntime`/`ShardedShared` integration
    /// level via `aiperf-mock-server`-shaped HTTP execution.
    #[test]
    fn global_dispatch_enforces_true_aggregate_concurrency_cap_sharded_does_not() {
        // Proves the whole point of `DispatchMode::Global`: with a concurrency
        // cap that does NOT evenly divide `workers` (`concurrency = 3`,
        // `workers = 4`), `Sharded` mode's per-thread
        // `owned_positions(cap, t, workers).max(1)` floor over-subscribes —
        // one thread's share rounds down to `0`, which the `.max(1)` floor
        // bumps back up to `1`, so `Sharded`'s aggregate cap across all 4
        // threads is `4`, not the authored `3` (see
        // `sharded_scheduled::slice_phase_for_thread`'s "Admission caps are
        // floored to one" doc comment). `Global` mode instead admits every
        // thread from the SAME shared `GlobalSlotPool`, so the wire-observed
        // peak concurrency across all worker threads combined never exceeds
        // the authored cap of 3, even though each thread issues without any
        // local partition of it.
        //
        // Peak concurrency is measured server-side
        // (`FixedMock::peak_concurrent`), not from client-side admission
        // bookkeeping, so this is a true end-to-end proof of the aggregate
        // cap actually enforced on the wire — the same kind of proof
        // `GlobalSlotPool`'s own cross-OS-thread test uses, applied here at
        // the full `ScheduledRuntime`/`ShardedShared` integration level via
        // HTTP execution against a real mock server.
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 24;
        let requests = 24u64;
        let phase = |concurrency: usize| -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": requests,
                "concurrency": concurrency,
            }))
            .unwrap()
        };
        let run = |dispatch_mode: crate::engine::protocol::DispatchMode| {
            let artifact_dir = tempfile::tempdir().unwrap();
            let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
            let factories = native_execution_factories();
            let spec = plan_with_dispatch(
                &mock.base_url,
                artifact_dir.path(),
                4,
                build_dataset(&registry, entries, 1),
                phase(3),
                dispatch_mode,
            );
            let report = execute_prepared_native_plan_uncommitted_selected(
                spec,
                request_executor,
                &factories,
                &registry,
                None,
            )
            .expect("native run must complete");
            assert!(
                report_error_count(&report) == 0,
                "expected zero profiling errors, report: {report:?}"
            );
        };

        run(crate::engine::protocol::DispatchMode::Sharded);
        let sharded_peak = mock.peak_concurrent();
        assert_eq!(
            sharded_peak, 4,
            "Sharded mode's per-thread floor-to-1 admission cap is EXPECTED to \
             over-subscribe the authored cap of 3 up to the worker count of 4 \
             (documented, accepted trade — not the bug Global mode fixes); got {sharded_peak}"
        );

        mock.reset_peak();
        run(crate::engine::protocol::DispatchMode::Global);
        let global_peak = mock.peak_concurrent();
        assert!(
            global_peak <= 3,
            "Global dispatch must enforce the true aggregate concurrency cap (3) \
             across all 4 worker threads combined, never exceeding it \
             (observed wire-side peak concurrent requests: {global_peak})"
        );
        assert_eq!(
            global_peak, 3,
            "the cap must actually be reached (proves this test exercises real \
             contention across all 4 worker threads, not an under-subscribed run); \
             got {global_peak}"
        );
    }

    #[test]
    fn concurrency_workers_gt_1_is_sharded_and_data_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 16;
        let requests = 16u64;
        let phase = |concurrency: usize| -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": requests,
                "concurrency": concurrency,
            }))
            .unwrap()
        };

        let baseline = run_and_read_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(4),
        );
        assert_pinned_records(&baseline, requests as usize, true);

        let sharded = run_and_read_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(4),
        );
        assert_pinned_records(&sharded, requests as usize, true);

        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&sharded),
            "sharded workers>1 must be DATA-identical to the single-thread baseline"
        );
    }

    #[test]
    fn poisson_workers_gt_1_is_sharded_and_data_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 12;
        let requests = 12u64;
        let phase = || -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "poisson",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": requests,
                "rate": 200.0,
            }))
            .unwrap()
        };

        let baseline = run_and_read_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(),
        );
        assert_pinned_records(&baseline, requests as usize, true);

        let sharded = run_and_read_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(),
        );
        assert_pinned_records(&sharded, requests as usize, true);

        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&sharded),
            "sharded Poisson workers>1 must be DATA-identical to the baseline"
        );
    }

    #[test]
    fn user_centric_workers_gt_1_thread_per_core_data_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 12;
        let requests = 12u64;
        let phase = || -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "user_centric",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": requests,
                "rate": 200.0,
                "users": 4,
            }))
            .unwrap()
        };
        let dataset = || build_dataset(&registry, entries, 2);

        let baseline = run_and_read_records(&registry, &mock, 1, dataset(), phase());
        assert_pinned_records(&baseline, requests as usize, false);

        let threaded = run_and_read_records(&registry, &mock, 4, dataset(), phase());
        assert_pinned_records(&threaded, requests as usize, false);

        // Real-clock churn makes the per-shape multiset timing-dependent; the
        // stable invariant is the set of turn shapes.
        assert_eq!(
            distinct_data_keys(&baseline),
            distinct_data_keys(&threaded),
            "user_centric workers>1 must draw from the same turn-shape universe"
        );
    }

    #[test]
    fn fixed_schedule_workers_gt_1_thread_per_core_data_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 10;
        let phase = || -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "fixed_schedule",
                "name": "profiling",
                "exclude_from_results": false,
            }))
            .unwrap()
        };

        let baseline = run_and_read_records(
            &registry,
            &mock,
            1,
            build_mooncake_dataset(&registry, entries),
            phase(),
        );
        let threaded = run_and_read_records(
            &registry,
            &mock,
            4,
            build_mooncake_dataset(&registry, entries),
            phase(),
        );

        assert_eq!(
            baseline.len(),
            entries,
            "fixed_schedule dispatches one first turn per conversation"
        );
        assert_pinned_records(&baseline, entries, true);
        assert_pinned_records(&threaded, entries, true);
        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&threaded),
            "ThreadPerCore fixed_schedule workers>1 must be DATA-identical to the baseline"
        );
    }

    use crate::accuracy_core::{
        AccuracyEvaluator, EvaluatorDatasetIdentity, EvaluatorGenerationConfig, EvaluatorGrade,
        EvaluatorGradeBatch, EvaluatorGradeItem, EvaluatorIdentity, EvaluatorLoadConfig,
        EvaluatorLoadResult, EvaluatorMessage, EvaluatorProblem, EvaluatorProblemPage,
        EvaluatorWorkerError, ProblemId,
    };
    use crate::engine::execute::{
        NativeStaticAccuracyPlan, StaticAccuracyEvaluatorFactory,
        StaticAccuracyEvaluatorProcessSpec,
    };
    use async_trait::async_trait;

    const ACCURACY_PROBLEMS: usize = 4;

    fn accuracy_identity() -> EvaluatorIdentity {
        EvaluatorIdentity {
            protocol: 1,
            worker_version: "fixture-worker".into(),
            python_version: "3.fixture".into(),
            python_executable: "/fixture/python".into(),
            packages: std::collections::BTreeMap::from([(
                "lighteval".to_string(),
                Some("fixture".to_string()),
            )]),
            worker_source_sha256: "fixture-source".into(),
            dependency_lock_sha256: Some("fixture-lock".into()),
            container_digest: Some("sha256:fixture".into()),
            capabilities: vec!["grade_batch".into()],
        }
    }

    fn accuracy_loaded() -> EvaluatorLoadResult {
        EvaluatorLoadResult {
            benchmark: "fixture-bench".into(),
            problem_count: ACCURACY_PROBLEMS,
            dataset: EvaluatorDatasetIdentity {
                provider: "fixture".into(),
                benchmark: None,
                repository: Some("fixture/repo".into()),
                subset: Some("default".into()),
                revision: Some("fixture-revision".into()),
                evaluation_splits: vec!["test".into()],
                task_version: Some(1),
            },
            grader: "fixture grader".into(),
        }
    }

    struct FixtureEvaluator {
        identity: EvaluatorIdentity,
        loaded: EvaluatorLoadResult,
        problems: Vec<EvaluatorProblem>,
    }

    impl FixtureEvaluator {
        fn new() -> Self {
            let problems = (0..ACCURACY_PROBLEMS)
                .map(|index| {
                    let prompt = format!("fixture problem {index}");
                    EvaluatorProblem {
                        problem_id: ProblemId::new(format!("prob-{index}")).unwrap(),
                        task: "demo".into(),
                        prompt: prompt.clone(),
                        messages: vec![EvaluatorMessage {
                            role: "user".into(),
                            content: Value::String(prompt),
                        }],
                        generation: EvaluatorGenerationConfig {
                            max_tokens: 16,
                            temperature: 0.0,
                            top_p: 1.0,
                            stop: Vec::new(),
                        },
                    }
                })
                .collect();
            Self {
                identity: accuracy_identity(),
                loaded: accuracy_loaded(),
                problems,
            }
        }
    }

    fn fixture_is_correct(problem_id: &str) -> bool {
        problem_id
            .rsplit('-')
            .next()
            .and_then(|n| n.parse::<usize>().ok())
            .map(|n| n % 2 == 0)
            .unwrap_or(false)
    }

    #[async_trait(?Send)]
    impl AccuracyEvaluator for FixtureEvaluator {
        fn identity(&self) -> &EvaluatorIdentity {
            &self.identity
        }

        async fn load(
            &mut self,
            _benchmark: &str,
            _config: &EvaluatorLoadConfig,
        ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError> {
            Ok(self.loaded.clone())
        }

        async fn next_problems(
            &mut self,
            offset: usize,
            limit: usize,
        ) -> Result<EvaluatorProblemPage, EvaluatorWorkerError> {
            let end = offset.saturating_add(limit).min(self.problems.len());
            Ok(EvaluatorProblemPage {
                items: self.problems[offset..end].to_vec(),
                next_offset: end,
                done: end == self.problems.len(),
            })
        }

        async fn grade_batch(
            &mut self,
            items: &[EvaluatorGradeItem],
        ) -> Result<EvaluatorGradeBatch, EvaluatorWorkerError> {
            Ok(EvaluatorGradeBatch {
                items: items
                    .iter()
                    .map(|item| {
                        let correct = fixture_is_correct(item.problem_id.as_str());
                        EvaluatorGrade {
                            problem_id: item.problem_id.clone(),
                            task: "demo".into(),
                            correct,
                            unparsed: false,
                            confidence: if correct { 1.0 } else { 0.0 },
                            reasoning: "fixture grade".into(),
                            extracted_answer: Some(item.response.clone()),
                        }
                    })
                    .collect(),
            })
        }

        async fn shutdown(&mut self) -> Result<(), EvaluatorWorkerError> {
            Ok(())
        }
    }

    struct FixtureEvaluatorFactory;

    #[async_trait(?Send)]
    impl StaticAccuracyEvaluatorFactory for FixtureEvaluatorFactory {
        async fn spawn(
            &self,
            _process: &StaticAccuracyEvaluatorProcessSpec,
        ) -> anyhow::Result<Box<dyn AccuracyEvaluator>> {
            Ok(Box::new(FixtureEvaluator::new()))
        }
    }

    fn accuracy_plan(
        base_url: &str,
        artifact_dir: &Path,
        workers: usize,
        requests: u64,
    ) -> NativeRunSpec {
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": requests,
            "concurrency": 4,
        }))
        .unwrap();
        NativeRunSpec {
            benchmark_id: "characterization-accuracy".into(),
            random_seed: Some(7),
            workers,
            artifact_dir: artifact_dir.to_path_buf(),
            models: models(),
            endpoint: NativeEndpointPlan::Prepared(endpoint_profile(base_url)),
            dataset: NativeDatasetPlan::StaticAccuracy(NativeStaticAccuracyPlan {
                benchmark: "fixture-bench".into(),
                tasks: None,
                n_shots: None,
                enable_cot: None,
                grader: None,
                system_prompt: None,
                process: StaticAccuracyEvaluatorProcessSpec {
                    python_executable: "/usr/bin/python3".into(),
                    worker_module: "fixture".into(),
                },
                evaluator_factory: Arc::new(FixtureEvaluatorFactory),
            }),
            tokenizer: TokenizerSpec {
                name: "builtin".into(),
                apply_chat_template: false,
                server_url: None,
            },
            phases: vec![phase],
            metrics: MetricsSpec::default(),
            artifacts: ArtifactSpec::default(),
            sidecars: NativeSidecarPlan::Prepared(Arc::new(empty_sidecars())),
            user_files: Vec::new(),
            failure_policy: None,
            native_otel_enabled: false,
            transport: None,
            dispatch_mode: crate::engine::protocol::DispatchMode::Sharded,
        }
    }

    fn run_accuracy_tally(
        registry: &AIPerfRegistry,
        mock: &FixedMock,
        workers: usize,
        requests: u64,
    ) -> (usize, usize) {
        let artifact_dir = tempfile::tempdir().unwrap();
        let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
        let factories = native_execution_factories();
        let spec = accuracy_plan(&mock.base_url, artifact_dir.path(), workers, requests);
        let report = execute_prepared_native_plan_uncommitted_selected(
            spec,
            request_executor,
            &factories,
            registry,
            None,
        )
        .expect("static-accuracy run must complete");
        let value = serde_json::to_value(&report).unwrap();
        let records = value
            .get("accuracy_records")
            .and_then(Value::as_array)
            .expect("report carries accuracy_records");
        let total = records.len();
        let correct = records
            .iter()
            .filter(|record| {
                record
                    .pointer("/result/correct")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            })
            .count();
        (total, correct)
    }

    #[test]
    fn static_accuracy_workers_gt_1_shards_and_tally_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let requests = 12u64;

        let (baseline_total, baseline_correct) = run_accuracy_tally(&registry, &mock, 1, requests);
        assert_eq!(
            baseline_total, requests as usize,
            "single-worker accuracy grades one record per dispatched request"
        );
        assert_eq!(
            baseline_correct,
            requests as usize / 2,
            "half the dispatched problems (even index) grade correct"
        );

        let (sharded_total, sharded_correct) = run_accuracy_tally(&registry, &mock, 4, requests);
        assert_eq!(
            (sharded_total, sharded_correct),
            (baseline_total, baseline_correct),
            "sharded workers>1 static-accuracy tally must equal the single-thread baseline"
        );
    }
}
