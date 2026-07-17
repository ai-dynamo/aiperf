// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-1 characterization tests for the `workers > 1` execution paths
//! (Graham-King verified-port standard).
//!
//! [`execute_native_inner`](crate::engine::execute) now runs ONE thread-per-core
//! execution model for a scheduled run:
//!
//! - **sharded** ([`crate::engine::sharded_scheduled`]) — chosen for EVERY
//!   `workers > 1` run, whatever the phase shape, INCLUDING static accuracy. Each
//!   OS thread is a self-contained sub-cell with a co-located transport; there is
//!   no per-request cross-thread hop. Rate-based phases (Concurrency/Poisson/Gamma/
//!   Constant) partition the request budget; the trace-driven
//!   `user_centric`/`fixed_schedule` phases partition per conversation (each
//!   sub-cell owns a disjoint conversation subset via its two-level partition). A
//!   static-accuracy run shards its dispatch+capture too: the `!Send`
//!   evaluator/grader stays on the coordinator, but each shard owns an
//!   `AccuracyRecordProcessor` over the shared read-only associations and the
//!   disjoint `Send` captures concatenate at the coordinator for a single grade.
//! - **co-located single worker** — `workers <= 1` only.
//!
//! Phase 2 unified these: the sharded model covers ALL `workers > 1` shapes and the
//! old cross-thread `ThreadPerCoreExecutor` hop is deleted. The last accuracy clamp
//! (workers forced to 1 for static accuracy) is removed. These tests pin the
//! captured per-record output and record counts so that unification stays
//! data-equivalent (exact multiset parity for rate-based and fixed_schedule,
//! aggregate parity for open-loop user_centric), and the static-accuracy tally
//! identical between `workers == 1` and `workers == 4`.
//!
//! # Why the assertions are DATA-level, not ns-level
//!
//! `workers > 1` is inherently a **real-clock** run: a `SimClock` can only advance
//! the single reactor its idle-pump drives, so
//! [`execute_prepared_native_plan_uncommitted_with_runtime_factories`](crate::engine::execute)
//! forces `workers = 1` under the virtual clock (the thread-per-core workers own
//! private reactors the pump cannot reach). There is therefore NO deterministic-ns
//! lib-level path for `workers > 1`; the only reproducible fixture is a
//! fixed-latency mock with tolerance on timing and exactness on data — exactly the
//! `CLAUDE.md` feature-complete recipe. These tests pin the reproducible facts:
//! record count, OSL (generated-token count), ISL (fixed synthetic input length),
//! status, and cross-model DATA parity between `workers = 1` and `workers = 4`
//! (the invariant Phase 2 must preserve), and assert timing only for presence and
//! positivity.

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

    /// Fixed synthetic input sequence length every conversation carries.
    const FIXED_ISL: u64 = 12;
    /// Fixed generated-token count the mock returns for every request (== OSL).
    const FIXED_OSL: usize = 6;
    /// Mock time-to-first-token, milliseconds.
    const MOCK_TTFT_MS: u64 = 8;
    /// Mock inter-token latency, milliseconds.
    const MOCK_ITL_MS: u64 = 2;

    /// A fixed-latency OpenAI-shaped SSE chat-completions mock.
    ///
    /// Runs on its own dedicated OS thread with a multi-threaded runtime so every
    /// thread-per-core worker (each on a private `current_thread` reactor) can
    /// connect independently. Every request — regardless of path or body — gets an
    /// identical deterministic reply: after `MOCK_TTFT_MS`, `FIXED_OSL`
    /// content-delta chunks each `MOCK_ITL_MS` apart (fixed content `"x"`), then a
    /// usage frame with `completion_tokens == FIXED_OSL`, then `[DONE]`.
    struct FixedMock {
        base_url: String,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl FixedMock {
        fn spawn() -> Self {
            let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let base_url = format!("http://{addr}");
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
            let thread = std::thread::Builder::new()
                .name("fixed-mock".into())
                .spawn(move || {
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
                                    tokio::spawn(async move {
                                        let service = service_fn(serve_sse);
                                        let _ = hyper::server::conn::http1::Builder::new()
                                            .serve_connection(TokioIo::new(stream), service)
                                            .await;
                                    });
                                }
                            }
                        }
                    });
                })
                .unwrap();
            Self {
                base_url,
                shutdown: Some(shutdown_tx),
                thread: Some(thread),
            }
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

    /// One connection's handler: stream the fixed SSE reply with real inter-token
    /// delays (an mpsc channel drained as an HTTP body stream).
    async fn serve_sse(
        _request: Request<hyper::body::Incoming>,
    ) -> Result<
        HttpResponse<StreamBody<impl stream::Stream<Item = Result<Frame<Bytes>, Infallible>>>>,
        Infallible,
    > {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Frame<Bytes>, Infallible>>();
        tokio::spawn(async move {
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
                    return;
                }
            }
            // Terminal usage frame (empty choices) — authoritative completion count.
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
        // Fixed ISL, zero stddev, no OSL distribution: every first turn is exactly
        // FIXED_ISL input tokens. `turns > 1` yields multi-turn conversations
        // (required by user_centric); later turns accumulate context so only the
        // first-turn ISL is FIXED_ISL. Deterministic.
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

    /// Build the canonical synthetic dataset on a throwaway current-thread runtime.
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

    /// Build a single-turn `mooncake_trace` dataset whose every conversation
    /// carries a first-turn `timestamp_ms` — the shape `fixed_schedule` requires
    /// (it bails on a first turn missing `timestamp_ms`). Fixed `input_length ==
    /// FIXED_ISL`, `output_length == FIXED_OSL`; ascending timestamps.
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

    /// Assemble one `NativeRunSpec` for a fixed profiling `phase`, pointed at the
    /// mock, writing per-record JSONL into `artifact_dir`.
    fn plan(
        base_url: &str,
        artifact_dir: &Path,
        workers: usize,
        dataset: crate::engine::dataset_input::PreparedDatasetInput,
        phase: PhaseSpec,
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
        }
    }

    /// Drive one run to completion against `mock` and return the parsed per-record
    /// JSONL rows (each row's `metadata` + `metrics` map).
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
        // No profiling-phase errors: the report groups them into `errors`.
        assert!(
            report_error_count(&report) == 0,
            "expected zero profiling errors, report: {report:?}"
        );
        read_records(artifact_dir.path())
    }

    fn report_error_count<T: serde::Serialize>(report: &T) -> usize {
        // The report serializes its aggregate; the errors array is the grouped
        // profiling terminal errors. Read it structurally so the assertion does not
        // depend on private fields (and so the test never names the `NativeReport`
        // type, whose re-export is private).
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
                // Single profiling phase only; guard anyway.
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

    /// The exact DATA facts Phase 2 must preserve for one record, order-independent.
    fn data_key(row: &Value) -> (i64, i64) {
        (
            metric(row, "input_sequence_length").unwrap().round() as i64,
            metric(row, "output_sequence_length").unwrap().round() as i64,
        )
    }

    /// Assert every record carries the pinned fixed OSL, a present positive
    /// latency, and no error — the deterministic per-record facts. When
    /// `expect_fixed_isl` is set (single-turn datasets), every ISL must equal
    /// `FIXED_ISL`; multi-turn datasets only require a positive ISL (later turns
    /// accumulate context).
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
            // With OSL >= 2 the reconciled inter-token latency is defined.
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

    /// The distinct set of turn shapes present, order-independent.
    fn distinct_data_keys(rows: &[Value]) -> Vec<(i64, i64)> {
        let mut keys: Vec<(i64, i64)> = rows.iter().map(data_key).collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    // ============================ concurrency ============================

    /// `workers > 1` + concurrency is the SHARDED path today
    /// (`sharded_scheduled::run_sharded_scheduled`). Pin its per-record output and
    /// prove it is DATA-identical to the single-thread `workers == 1` baseline.
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

        // Baseline: single-thread coordinator (workers == 1, NOT sharded).
        let baseline = run_and_read_records(
            &registry,
            &mock,
            1,
            build_dataset(&registry, entries, 1),
            phase(4),
        );
        assert_pinned_records(&baseline, requests as usize, true);

        // workers == 4 selects the sharded path (rate-based + accuracy.is_none()).
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

    /// `workers > 1` + Poisson request-rate also shards. Pin it.
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

    // ============================ user_centric ============================

    /// `workers > 1` + user_centric now shards per conversation (each sub-cell owns
    /// a disjoint conversation subset and runs `1/W` of the users/rate/budget). Pin
    /// the captured output and prove `workers == 4` draws from the same turn-shape
    /// universe and dispatches the same total count as `workers == 1` — the
    /// aggregate invariants an open-loop workload preserves under sharding.
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
        // user_centric requires multi-turn conversations (average turns >= 2).
        let dataset = || build_dataset(&registry, entries, 2);

        let baseline = run_and_read_records(&registry, &mock, 1, dataset(), phase());
        assert_pinned_records(&baseline, requests as usize, false);

        let threaded = run_and_read_records(&registry, &mock, 4, dataset(), phase());
        assert_pinned_records(&threaded, requests as usize, false);

        // CHARACTERIZATION FINDING: user_centric is OPEN-LOOP churn — the exact
        // first-turn/second-turn split under a `requests` budget is timing-sensitive
        // (how many users reached their 2nd turn before the budget filled), so it
        // differs between `workers == 1` and `workers == 4` under the real clock. The
        // deterministic, Phase-2-preservable invariants are therefore the total
        // record count (== the budget) and the SET of turn shapes dispatched (the
        // same conversations feed both), NOT the per-shape multiset. Phase 2 parity
        // for user_centric must be asserted at this aggregate level, not per-turn.
        assert_eq!(
            distinct_data_keys(&baseline),
            distinct_data_keys(&threaded),
            "user_centric workers>1 must draw from the same turn-shape universe"
        );
    }

    // ============================ fixed_schedule ============================

    /// `workers > 1` + fixed_schedule now shards per conversation: each sub-cell
    /// owns a disjoint conversation subset (its two-level partition) and replays
    /// that subset's full authored schedule, so the W threads tile the trace
    /// exactly. Pin the captured output and prove `workers == 4` is DATA-identical
    /// (exact multiset parity) to `workers == 1`.
    #[test]
    fn fixed_schedule_workers_gt_1_thread_per_core_data_matches_single_thread() {
        let registry = AIPerfRegistry::builtin().unwrap();
        let mock = FixedMock::spawn();
        // Fixed-schedule dispatches one first turn per conversation, so the record
        // count equals the conversation count, not a `requests` budget.
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

        // Fixed-schedule dispatches one first turn per conversation.
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

    // ============================ static-accuracy ============================

    // A static-accuracy run NOW shards (the last clamp removed). Its `!Send`
    // evaluator/grader stays on the coordinator, but the per-record CAPTURE is pure
    // `Send` data — each shard owns an `AccuracyRecordProcessor` over the shared
    // read-only associations, and the disjoint captures concatenate at the
    // coordinator for a single main-thread grade (keyed by `problem_id`, so the
    // merged set is order-independent).
    //
    // The full production evaluator is a pinned Python subprocess (lighteval/harness)
    // a pure-Rust `--lib` test cannot spawn, so the evaluator is injected here through
    // its trait seam (`StaticAccuracyEvaluatorFactory`). The evaluator produces the
    // problems and grades the captured responses; the FixedMock is the inference
    // target. The invariant: the graded tally is IDENTICAL for `workers == 1` and
    // `workers == 4` (the sharded partition dispatches the same problem_id multiset).

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

    /// Number of evaluator-authored problems in the accuracy fixture.
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

    /// A fixture evaluator producing `ACCURACY_PROBLEMS` single-turn chat problems
    /// and grading each captured response by its problem index parity — a
    /// deterministic, response-independent verdict so the tally is a pure function of
    /// the captured problem_id multiset (identical across worker counts).
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

    /// `prob-N` is graded correct when `N` is even — response-independent.
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

    /// Trait-injected fixture factory: returns a fresh in-process evaluator (no
    /// Python subprocess), so the whole static-accuracy path is driven at lib level.
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

    /// Assemble a static-accuracy `NativeRunSpec` (fixture evaluator) for a fixed
    /// concurrency profiling phase pointed at the mock.
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
            },
            phases: vec![phase],
            metrics: MetricsSpec::default(),
            artifacts: ArtifactSpec::default(),
            sidecars: NativeSidecarPlan::Prepared(Arc::new(empty_sidecars())),
            user_files: Vec::new(),
            failure_policy: None,
            native_otel_enabled: false,
            transport: None,
        }
    }

    /// Run one static-accuracy benchmark to completion and return `(total, correct)`
    /// from the graded `accuracy_records` in the report.
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

    /// The last executor-unification clamp removed: a static-accuracy run at
    /// `workers == 4` shards its dispatch+capture and grades the concatenated
    /// captures to the SAME tally as the single-worker baseline. Each of the 4
    /// problems is dispatched `requests / 4` times; problems `prob-0`/`prob-2` grade
    /// correct, so exactly half the records are correct at either worker count.
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
