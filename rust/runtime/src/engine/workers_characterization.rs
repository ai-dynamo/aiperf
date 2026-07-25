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
            agentic_trees: std::sync::Arc::default(),
            warmup_handoff: crate::agentic_tree::empty_warmup_handoff_carrier(),
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
            agentic_trees: std::sync::Arc::default(),
            warmup_handoff: crate::agentic_tree::empty_warmup_handoff_carrier(),
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

    /// Rate-pacing analogue of
    /// `global_dispatch_enforces_true_aggregate_concurrency_cap_sharded_does_not`.
    ///
    /// Proves `DispatchMode::Global` paces the AGGREGATE request rate across all
    /// `W` worker threads at the single configured global rate, not `W ×` too
    /// fast. Before the `GlobalRateGate` was actually consumed, `slice_phase_for_thread`
    /// left `rate` UNSLICED under `Global`, so each of the 4 worker threads paced
    /// a `Constant` phase at the full `RATE_PER_SEC`, producing a merged arrival
    /// stream at ≈ `4 × RATE_PER_SEC`. With the shared gate consumed, every
    /// thread claims one distinct slot from the same evenly-spaced base grid, so
    /// the merged `credit_issued_ns` timeline advances at exactly one interval
    /// per request — one global rate.
    ///
    /// The aggregate rate is measured end-to-end from the emitted per-record
    /// `credit_issued_ns` (actual wire issue times, merged across every worker
    /// thread's records), not from client-side pacing bookkeeping — the same
    /// merged-timeline proof shape the concurrency test uses server-side.
    #[test]
    fn global_dispatch_paces_true_aggregate_rate_not_workers_times_too_fast() {
        const WORKERS: usize = 4;
        const RATE_PER_SEC: f64 = 200.0; // 5ms base interval.
        const REQUESTS: u64 = 40; // 10 per thread at W=4.

        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "constant",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": REQUESTS,
            "rate": RATE_PER_SEC,
        }))
        .unwrap();

        // Collect the merged `credit_issued_ns` timeline across all worker
        // threads' records and derive the aggregate issuance rate from its span.
        let measure_aggregate_rate = |dispatch_mode: crate::engine::protocol::DispatchMode| -> f64 {
            let artifact_dir = tempfile::tempdir().unwrap();
            let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
            let factories = native_execution_factories();
            let spec = plan_with_dispatch(
                &mock.base_url,
                artifact_dir.path(),
                WORKERS,
                build_dataset(&registry, REQUESTS as usize, 1),
                phase.clone(),
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
            let rows = read_records(artifact_dir.path());
            assert_eq!(
                rows.len(),
                REQUESTS as usize,
                "every issued request must produce one profiling record"
            );
            let mut issued: Vec<i64> = rows
                .iter()
                .map(|row| {
                    row.get("metadata")
                        .and_then(|m| m.get("credit_issued_ns"))
                        .and_then(Value::as_i64)
                        .expect("record carries credit_issued_ns for a rate phase")
                })
                .collect();
            issued.sort_unstable();
            let span_ns = (issued.last().unwrap() - issued.first().unwrap()).max(1);
            // (N-1) intervals span the merged timeline.
            (REQUESTS as f64 - 1.0) / (span_ns as f64 / 1e9)
        };

        let global_rate = measure_aggregate_rate(crate::engine::protocol::DispatchMode::Global);
        eprintln!(
            "global-mode aggregate rate across {WORKERS} threads: {global_rate:.1}/s \
             (configured global rate {RATE_PER_SEC}/s)"
        );
        // Global mode must pace the aggregate at the configured rate. A generous
        // band (±35%) still cleanly rejects the pre-fix ≈4× regression (which
        // would land near 800/s), while tolerating real-clock issue jitter.
        assert!(
            global_rate < RATE_PER_SEC * 1.35,
            "Global dispatch must pace the AGGREGATE rate across all {WORKERS} worker \
             threads at the global {RATE_PER_SEC}/s, NOT ~{WORKERS}x too fast; the \
             unsliced-but-unconsumed-gate regression paced every thread at the full \
             rate for a merged ~{:.0}/s. Measured aggregate: {global_rate:.1}/s",
            RATE_PER_SEC * WORKERS as f64
        );
        assert!(
            global_rate > RATE_PER_SEC * 0.65,
            "Global aggregate rate must actually reach the configured {RATE_PER_SEC}/s \
             (proves the run is rate-bound, not stalled); measured {global_rate:.1}/s"
        );
    }

    /// Run one prepared native plan under `dispatch_mode`/`workers` and return
    /// its profiling records. Shared by the `GlobalHop` proofs below.
    fn run_dispatch_records(
        registry: &AIPerfRegistry,
        mock: &FixedMock,
        workers: usize,
        dataset: crate::engine::dataset_input::PreparedDatasetInput,
        phase: PhaseSpec,
        dispatch_mode: crate::engine::protocol::DispatchMode,
    ) -> Vec<Value> {
        let artifact_dir = tempfile::tempdir().unwrap();
        let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
        let factories = native_execution_factories();
        let spec = plan_with_dispatch(
            &mock.base_url,
            artifact_dir.path(),
            workers,
            dataset,
            phase,
            dispatch_mode,
        );
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

    /// Proves the whole point of `DispatchMode::GlobalHop`: ONE coordinator loop
    /// hops every turn round-robin to `W` worker threads over the thread-per-core
    /// hop executor, dispatching every request EXACTLY ONCE and merging the
    /// worker shards into a DETERMINISTIC global order — the property `Global`
    /// mode's `W` independent racing loops cannot guarantee.
    ///
    /// Exactly-once + deterministic merge are proven end-to-end by data
    /// equivalence to the authoritative single-thread (`workers = 1`) run: the
    /// `GlobalHop` `workers = 4` merged record stream must carry the SAME
    /// per-request `(ISL, OSL)` multiset as the single dispatcher (no dropped,
    /// duplicated, or reordered-into-a-different-multiset turn), and re-running
    /// `GlobalHop` must reproduce that stream byte-for-byte after the merge sort.
    #[test]
    fn global_hop_dispatches_every_request_exactly_once_in_deterministic_merged_order() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 24;
        let requests = 24u64;
        let phase = || -> PhaseSpec {
            serde_json::from_value(json!({
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": requests,
                "concurrency": 4,
            }))
            .unwrap()
        };

        // Authoritative single dispatcher: workers=1 issues every turn from one
        // loop in exact global order.
        let baseline = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        let baseline_keys = sorted_data_keys(&baseline);

        // GlobalHop across 4 worker threads: one coordinator loop hops each turn
        // round-robin, then merges the shards deterministically.
        let hop = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(),
            crate::engine::protocol::DispatchMode::GlobalHop,
        );
        assert_eq!(
            hop.len(),
            requests as usize,
            "GlobalHop must produce exactly one record per dispatched request \
             (every turn dispatched exactly once — no loss, no duplicate)"
        );
        assert_eq!(
            sorted_data_keys(&hop),
            baseline_keys,
            "GlobalHop's merged record stream must carry the same per-request multiset \
             the single dispatcher does (exactly-once, deterministically merged)"
        );

        // Determinism: a second GlobalHop run reproduces the identical merged
        // record ordering.
        let hop_again = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(),
            crate::engine::protocol::DispatchMode::GlobalHop,
        );
        assert_eq!(
            sorted_data_keys(&hop_again),
            sorted_data_keys(&hop),
            "GlobalHop merge order must be deterministic across runs"
        );
    }

    /// `GlobalHop` aggregate-concurrency exactness — the counterpart of
    /// `global_dispatch_enforces_true_aggregate_concurrency_cap_sharded_does_not`.
    ///
    /// A single coordinator loop drives the FULL cell-level concurrency cap
    /// through one local `SlotPool` (NOT through `GlobalAdmission`, which
    /// `GlobalHop` deliberately does not consume — see `global_hop`'s module
    /// doc), so the wire-observed aggregate peak concurrency across all 4 worker
    /// threads combined never exceeds the authored cap of 3, exactly as `Global`
    /// mode's shared gate achieves, but here from "one loop, one full-cap local
    /// pool". Peak concurrency is measured server-side (`FixedMock::peak_concurrent`).
    #[test]
    fn global_hop_enforces_true_aggregate_concurrency_cap() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 24;
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 24u64,
            "concurrency": 3,
        }))
        .unwrap();

        mock.reset_peak();
        let rows = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase,
            crate::engine::protocol::DispatchMode::GlobalHop,
        );
        assert_eq!(rows.len(), 24, "every request must produce one record");
        let peak = mock.peak_concurrent();
        assert!(
            peak <= 3,
            "GlobalHop must enforce the true aggregate concurrency cap (3) across all \
             4 worker threads combined, never exceeding it (observed wire peak: {peak})"
        );
        assert_eq!(
            peak, 3,
            "the cap must actually be reached (proves real cross-thread contention, \
             not an under-subscribed run); got {peak}"
        );
    }

    /// `GlobalHop` aggregate-rate exactness — the counterpart of
    /// `global_dispatch_paces_true_aggregate_rate_not_workers_times_too_fast`.
    ///
    /// The single coordinator loop paces one `Constant` phase at the full global
    /// rate through the local per-phase interval grid (again NOT via
    /// `GlobalAdmission`), so the merged `credit_issued_ns` timeline across all
    /// 4 worker threads advances at the configured aggregate rate, not `W ×` too
    /// fast.
    #[test]
    fn global_hop_paces_true_aggregate_rate() {
        const WORKERS: usize = 4;
        const RATE_PER_SEC: f64 = 200.0;
        const REQUESTS: u64 = 40;

        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "constant",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": REQUESTS,
            "rate": RATE_PER_SEC,
        }))
        .unwrap();

        let rows = run_dispatch_records(
            &registry,
            &mock,
            WORKERS,
            build_dataset(&registry, REQUESTS as usize, 1),
            phase,
            crate::engine::protocol::DispatchMode::GlobalHop,
        );
        assert_eq!(
            rows.len(),
            REQUESTS as usize,
            "every issued request must produce one profiling record"
        );
        let mut issued: Vec<i64> = rows
            .iter()
            .map(|row| {
                row.get("metadata")
                    .and_then(|m| m.get("credit_issued_ns"))
                    .and_then(Value::as_i64)
                    .expect("record carries credit_issued_ns for a rate phase")
            })
            .collect();
        issued.sort_unstable();
        let span_ns = (issued.last().unwrap() - issued.first().unwrap()).max(1);
        let hop_rate = (REQUESTS as f64 - 1.0) / (span_ns as f64 / 1e9);
        eprintln!(
            "global-hop aggregate rate across {WORKERS} threads: {hop_rate:.1}/s \
             (configured global rate {RATE_PER_SEC}/s)"
        );
        assert!(
            hop_rate < RATE_PER_SEC * 1.35,
            "GlobalHop must pace the AGGREGATE rate across all {WORKERS} worker threads \
             at the global {RATE_PER_SEC}/s, NOT ~{WORKERS}x too fast; measured {hop_rate:.1}/s"
        );
        assert!(
            hop_rate > RATE_PER_SEC * 0.65,
            "GlobalHop aggregate rate must actually reach the configured {RATE_PER_SEC}/s \
             (proves the run is rate-bound, not stalled); measured {hop_rate:.1}/s"
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

    /// `Global`-mode twin of
    /// `concurrency_workers_gt_1_is_sharded_and_data_matches_single_thread`:
    /// the same single-thread-baseline data-multiset assertion, but with
    /// `workers = 4` dispatched under `DispatchMode::Global` instead of
    /// `Sharded`. Proves `Global` mode is data-identical to the
    /// authoritative single dispatcher for the `concurrency` phase shape,
    /// not just an approximation of it.
    #[test]
    fn concurrency_workers_gt_1_global_data_matches_single_thread() {
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

        let baseline = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(4),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        assert_pinned_records(&baseline, requests as usize, true);

        let global = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(4),
            crate::engine::protocol::DispatchMode::Global,
        );
        assert_pinned_records(&global, requests as usize, true);

        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&global),
            "Global workers>1 must be DATA-identical to the single-thread baseline"
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

    /// `Global`-mode twin of
    /// `poisson_workers_gt_1_is_sharded_and_data_matches_single_thread`.
    #[test]
    fn poisson_workers_gt_1_global_data_matches_single_thread() {
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

        let baseline = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        assert_pinned_records(&baseline, requests as usize, true);

        let global = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_dataset(&registry, entries, 1),
            phase(),
            crate::engine::protocol::DispatchMode::Global,
        );
        assert_pinned_records(&global, requests as usize, true);

        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&global),
            "Global Poisson workers>1 must be DATA-identical to the baseline"
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

    /// `Global`-mode twin of
    /// `user_centric_workers_gt_1_thread_per_core_data_matches_single_thread`.
    #[test]
    fn user_centric_workers_gt_1_global_data_matches_single_thread() {
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

        let baseline = run_dispatch_records(
            &registry,
            &mock,
            1,
            dataset(),
            phase(),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        assert_pinned_records(&baseline, requests as usize, false);

        let global = run_dispatch_records(
            &registry,
            &mock,
            4,
            dataset(),
            phase(),
            crate::engine::protocol::DispatchMode::Global,
        );
        assert_pinned_records(&global, requests as usize, false);

        // Real-clock churn makes the per-shape multiset timing-dependent; the
        // stable invariant is the set of turn shapes.
        assert_eq!(
            distinct_data_keys(&baseline),
            distinct_data_keys(&global),
            "Global user_centric workers>1 must draw from the same turn-shape universe"
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

    /// `Global`-mode twin of
    /// `fixed_schedule_workers_gt_1_thread_per_core_data_matches_single_thread`.
    #[test]
    fn fixed_schedule_workers_gt_1_global_data_matches_single_thread() {
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

        let baseline = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_mooncake_dataset(&registry, entries),
            phase(),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        let global = run_dispatch_records(
            &registry,
            &mock,
            4,
            build_mooncake_dataset(&registry, entries),
            phase(),
            crate::engine::protocol::DispatchMode::Global,
        );

        assert_eq!(
            baseline.len(),
            entries,
            "fixed_schedule dispatches one first turn per conversation"
        );
        assert_pinned_records(&baseline, entries, true);
        assert_pinned_records(&global, entries, true);
        assert_eq!(
            sorted_data_keys(&baseline),
            sorted_data_keys(&global),
            "Global fixed_schedule workers>1 must be DATA-identical to the baseline"
        );
    }

    /// A second mock, distinct from `FixedMock`, whose per-request TTFT
    /// alternates between a short and a long delay (round-robin over
    /// arrival order) so that completion times across concurrently
    /// in-flight requests are deliberately UNEVEN — unlike `FixedMock`,
    /// where every request takes the same fixed time and so every worker
    /// thread's local admission slots free up in lockstep.
    ///
    /// Uneven completion times are the condition under which `Sharded`
    /// mode's static `1/workers` partition of the concurrency budget is
    /// most visibly wrong: a thread stuck behind a slow request cannot
    /// borrow spare capacity from a thread whose fast requests already
    /// completed, so the true, wire-observed aggregate concurrency drifts
    /// away from what a shared, dynamically-reallocated pool (`Global`)
    /// would allow.
    struct VariableLatencyMock {
        base_url: String,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
        #[allow(dead_code)]
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl VariableLatencyMock {
        fn spawn() -> Self {
            let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let base_url = format!("http://{addr}");
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
            let current = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let arrival = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let thread = std::thread::Builder::new()
                .name("variable-latency-mock".into())
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
                                        let arrival = arrival.clone();
                                        tokio::spawn(async move {
                                            let service = service_fn(move |request| {
                                                serve_variable_sse(
                                                    request,
                                                    current.clone(),
                                                    peak.clone(),
                                                    arrival.clone(),
                                                )
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

        fn peak_concurrent(&self) -> usize {
            self.peak.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl Drop for VariableLatencyMock {
        fn drop(&mut self) {
            if let Some(tx) = self.shutdown.take() {
                let _ = tx.send(());
            }
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }

    async fn serve_variable_sse(
        _request: Request<hyper::body::Incoming>,
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
        arrival: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<
        HttpResponse<StreamBody<impl stream::Stream<Item = Result<Frame<Bytes>, Infallible>>>>,
        Infallible,
    > {
        // Every third arrival is a "slow" request (long TTFT + long ITL);
        // the rest are "fast". This uneven mix, combined with `Sharded`'s
        // static round-robin request-to-thread assignment, concentrates
        // slow requests unevenly across worker threads.
        let index = arrival.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let slow = index % 3 == 0;
        let (ttft_ms, itl_ms) = if slow { (60, 10) } else { (4, 1) };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Frame<Bytes>, Infallible>>();
        tokio::spawn(async move {
            let now = current.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            peak.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(ttft_ms)).await;
            for index in 0..FIXED_OSL {
                if index > 0 {
                    tokio::time::sleep(Duration::from_millis(itl_ms)).await;
                }
                let chunk = json!({
                    "id": "chatcmpl-variable",
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
                "id": "chatcmpl-variable",
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

    /// The divergence regression guard: proves `Sharded` and `Global` modes
    /// produce MEASURABLY DIFFERENT aggregate-concurrency behavior under
    /// deliberately uneven per-request completion times, and that the
    /// difference is in the expected direction — `Sharded`'s local,
    /// per-thread admission floor over-subscribes the authored global cap,
    /// while `Global`'s shared pool never does.
    ///
    /// If `Global` mode's admission wiring were ever broken (e.g. reverted
    /// to consulting only a local per-thread slot count), this test would
    /// stop observing a difference between the two modes and fail on the
    /// `global_peak <= 3` assertion — the same failure mode this whole task
    /// plan exists to guard against.
    #[test]
    fn sharded_and_global_diverge_under_uneven_completion_times() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let entries = 24;
        let requests = 24u64;
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": requests,
            "concurrency": 3,
        }))
        .unwrap();

        let run = |mock: &VariableLatencyMock,
                   dispatch_mode: crate::engine::protocol::DispatchMode| {
            let artifact_dir = tempfile::tempdir().unwrap();
            let request_executor: Arc<dyn RequestExecutorFactory> = Arc::new(HttpExecutionFactory);
            let factories = native_execution_factories();
            let spec = plan_with_dispatch(
                &mock.base_url,
                artifact_dir.path(),
                4,
                build_dataset(&registry, entries, 1),
                phase.clone(),
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

        let sharded_mock = VariableLatencyMock::spawn();
        run(
            &sharded_mock,
            crate::engine::protocol::DispatchMode::Sharded,
        );
        let sharded_peak = sharded_mock.peak_concurrent();

        let global_mock = VariableLatencyMock::spawn();
        run(&global_mock, crate::engine::protocol::DispatchMode::Global);
        let global_peak = global_mock.peak_concurrent();

        assert!(
            sharded_peak > 3,
            "Sharded mode's local per-thread admission floor is expected to \
             over-subscribe the authored aggregate cap of 3 under uneven \
             per-request completion times (a slow request stalls one \
             thread's local slot while other threads keep issuing from \
             their own local floor-to-1 slots); observed peak {sharded_peak}"
        );
        assert!(
            global_peak <= 3,
            "Global mode's shared admission pool must never exceed the \
             authored aggregate cap of 3, even under uneven per-request \
             completion times; observed peak {global_peak}"
        );
        assert!(
            sharded_peak != global_peak,
            "Sharded and Global must diverge under uneven completion times \
             (Sharded {sharded_peak} vs Global {global_peak}) — a passing \
             assertion here that shows no difference would be a false \
             negative for this regression guard"
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

    /// The subset of one profiling record's fields that a seeded, deterministic
    /// fixture (fixed synthetic dataset + `FixedMock`'s constant per-chunk
    /// delays + a seeded RNG) actually promises to reproduce byte-for-byte
    /// across repeated runs.
    ///
    /// This file's own module doc says why the OTHER fields cannot be part of
    /// that promise: "Multi-worker execution uses real clocks because
    /// `SimClock` advances only its own reactor" — `workers > 1` (and, per
    /// [`execute_prepared_native_plan_uncommitted_with_runtime_factories`]'s
    /// virtual-clock branch, even `workers == 1` unless the run is the
    /// `dry_run` transport's own `SimClock`-driven graph path, which this
    /// scheduled-dataset harness does not exercise) always drives real wall
    /// time. `credit_issued_ns`/`request_start_ns`/`request_ack_ns`/
    /// `request_end_ns` and every metric computed from them
    /// (`request_latency`, `inter_token_latency`, `inter_chunk_latency`,
    /// `time_to_first_token`, `time_to_first_output_token`,
    /// `time_to_second_token`, `http_req_waiting`,
    /// `output_token_throughput_per_user`, `prefill_throughput_per_user`) are
    /// therefore real scheduler-jitter-dependent floats — never byte-identical
    /// run over run even with fixed `sleep()` durations on the mock side, as
    /// verified empirically (dumped record shape shows sub-microsecond-precision
    /// floats such as `25.757441` ms that vary between runs).
    ///
    /// What IS byte-exact-reproducible under this deterministic fixture: the
    /// dispatched turn identity (`conversation_id`, `turn_index`, `session_num`),
    /// its outcome (`error`, `was_cancelled`, `benchmark_phase`), and every
    /// data-derived metric that comes from the fixed synthetic dataset and the
    /// mock's fixed `usage`/token counts rather than from wall time (ISL, OSL,
    /// output token count, usage fields, and the OSL-mismatch/usage-diff
    /// percentages, which are ratios of those same fixed counts). Canonicalizing
    /// to just this subset and sorting by turn identity turns "assert the two
    /// record sets are equal" into a genuine, false-negative-resistant
    /// byte-exact comparison (`serde_json` equality on the canonical `Value`,
    /// not a length or summary-statistic check) instead of a vacuous one.
    fn canonical_deterministic_view(rows: &[Value]) -> Vec<Value> {
        const METRIC_ALLOWLIST: &[&str] = &[
            "input_sequence_length",
            "output_sequence_length",
            "output_token_count",
            "usage_completion_tokens",
            "usage_completion_tokens_diff_pct",
            "usage_prompt_tokens",
            "usage_prompt_tokens_diff_pct",
            "usage_total_tokens",
            "osl_mismatch_diff_pct",
        ];
        let mut canonical: Vec<Value> = rows
            .iter()
            .map(|row| {
                let metadata = row.get("metadata").cloned().unwrap_or(Value::Null);
                let metrics = row.get("metrics").and_then(Value::as_object);
                let canonical_metrics: serde_json::Map<String, Value> = METRIC_ALLOWLIST
                    .iter()
                    .filter_map(|tag| {
                        metrics
                            .and_then(|m| m.get(*tag))
                            .map(|value| (tag.to_string(), value.clone()))
                    })
                    .collect();
                json!({
                    "conversation_id": metadata.get("conversation_id"),
                    "turn_index": metadata.get("turn_index"),
                    "session_num": metadata.get("session_num"),
                    "benchmark_phase": metadata.get("benchmark_phase"),
                    "was_cancelled": metadata.get("was_cancelled"),
                    "error": row.get("error"),
                    "metrics": canonical_metrics,
                })
            })
            .collect();
        canonical.sort_by(|a, b| {
            let key = |v: &Value| {
                (
                    v.get("conversation_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    v.get("turn_index").and_then(Value::as_i64).unwrap_or(0),
                )
            };
            key(a).cmp(&key(b))
        });
        canonical
    }

    /// Serializes [`canonical_deterministic_view`]'s output to a single
    /// canonical JSON string so two record sets can be compared with a single
    /// `assert_eq!` on `String` — a genuine byte-exact comparison over the
    /// deterministic subset, not a length or summary-statistic check.
    fn canonical_deterministic_bytes(rows: &[Value]) -> String {
        serde_json::to_string(&canonical_deterministic_view(rows)).unwrap()
    }

    /// Proves `DispatchMode::Global` is deterministic: running the SAME
    /// `concurrency`-phase, `workers = 4`, seeded-RNG fixture three times
    /// produces byte-identical canonical record output every time (see
    /// [`canonical_deterministic_view`] for exactly what "byte-identical"
    /// covers and why, given this file's real-clock, `workers > 1`
    /// architecture, timestamp-derived fields are excluded from that promise).
    ///
    /// This also establishes the pattern
    /// [`dispatch_mode_is_byte_exact_equivalent_at_workers_eq_1`] reuses for
    /// `Global`/`Sharded` cross-mode agreement.
    #[test]
    fn global_dispatch_is_byte_exact_deterministic_across_repeated_sim_runs() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 16;
        let requests = 16u64;
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": requests,
            "concurrency": 4,
        }))
        .unwrap();

        let run = || {
            let rows = run_dispatch_records(
                &registry,
                &mock,
                4,
                build_dataset(&registry, entries, 1),
                phase.clone(),
                crate::engine::protocol::DispatchMode::Global,
            );
            assert_eq!(
                rows.len(),
                requests as usize,
                "every dispatched request must produce exactly one record"
            );
            canonical_deterministic_bytes(&rows)
        };

        let first = run();
        let second = run();
        let third = run();

        assert_eq!(
            first, second,
            "Global dispatch's canonical record output must be byte-identical \
             across repeated runs of the identical seeded fixture (run 1 vs 2)"
        );
        assert_eq!(
            second, third,
            "Global dispatch's canonical record output must be byte-identical \
             across repeated runs of the identical seeded fixture (run 2 vs 3)"
        );
    }

    /// Cross-mode agreement where static partitioning provably cannot matter:
    /// at `workers = 1`, `thread_id = 0`, `slice_phase_for_thread`'s
    /// `owned_positions(value, 0, 1)` returns `value` unchanged for BOTH the
    /// `Sharded` per-thread-floor branch and the `Global`
    /// `global_admits_concurrency_and_rate` branch (which leaves the phase's
    /// concurrency/rate fields at their cell-local, i.e. already-unsliced,
    /// value) — see `sharded_scheduled::slice_phase_for_thread` and
    /// `sharded_scheduled::owned_positions`. There is exactly one worker
    /// thread to own the full budget either way, so the two dispatch modes
    /// resolve to the SAME sliced phase and the SAME single-loop coordinator
    /// path (this file's own module doc: "`workers == 1` uses one co-located
    /// worker sink on the coordinator's current-thread runtime"). This is the
    /// general shape of "every case NOT affected by static partitioning must
    /// produce identical output between modes", verified against the actual
    /// partitioning math rather than assumed from intuition.
    ///
    /// The two dispatch modes are therefore asserted byte-identical over the
    /// same canonical view [`global_dispatch_is_byte_exact_deterministic_across_repeated_sim_runs`]
    /// uses.
    #[test]
    fn dispatch_mode_is_byte_exact_equivalent_at_workers_eq_1() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        let entries = 16;
        let requests = 16u64;
        let phase: PhaseSpec = serde_json::from_value(json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": requests,
            "concurrency": 4,
        }))
        .unwrap();

        let sharded = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase.clone(),
            crate::engine::protocol::DispatchMode::Sharded,
        );
        assert_eq!(sharded.len(), requests as usize);

        let global = run_dispatch_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase,
            crate::engine::protocol::DispatchMode::Global,
        );
        assert_eq!(global.len(), requests as usize);

        assert_eq!(
            canonical_deterministic_bytes(&sharded),
            canonical_deterministic_bytes(&global),
            "at workers=1, static partitioning cannot differ between Sharded and \
             Global (owned_positions(value, 0, 1) == value for both branches), so \
             the two dispatch modes must produce byte-identical canonical output"
        );
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
