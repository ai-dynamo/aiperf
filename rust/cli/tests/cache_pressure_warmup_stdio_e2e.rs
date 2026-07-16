// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-E capstone: process-level proof that the full cache-pressure warmup
//! flow works end-to-end through the runner's product protocol-v2 stdio path.
//!
//! Drives ONE run in which the WARMUP phase carries
//! `agentic_cache_warmup_duration` (the C2-lowered cache-pressure window) over a
//! recorded WEKA trace with an active trajectory-start (`t*`) window, so:
//!
//!   * E3b's [`GraphPressureRecycle`] recycles the warmup corpus for the pressure
//!     duration (more than one corpus pass), rather than the single-pass workload
//!     an ordinary warmup runs;
//!   * E3c stashes a `GraphWarmupHandoff` and PROFILING resumes each lane from its
//!     recorded frontier, replaying the post-`t*` remainder;
//!   * the run reaches a valid native report with no hang and no abort.
//!
//! The recycle count is proved from the run's own native report (the WARMUP
//! accumulator's `request_count` counter total exceeds one corpus pass), not by
//! racing the mock, so the assertion is deterministic. The pressure loop is
//! Clock-duration-bounded and the child is wrapped in a wall-clock timeout, so
//! the test cannot hang. Mirrors the harness in `warmup_abort_stdio_e2e.rs` and
//! the `t*` midpoint trace it pins.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{Router, body::Bytes, extract::State, response::IntoResponse, routing::post};
use serde_json::{Value, json};

/// Shared mock state: total node dispatches observed across warmup + profiling.
#[derive(Default)]
struct MockState {
    requests: AtomicU64,
}

/// Every dispatched node returns one streamed token and a terminal usage frame,
/// so both the recycled warmup instances and the resumed profiling frontier
/// produce complete records.
async fn chat(State(state): State<Arc<MockState>>, _body: Bytes) -> impl IntoResponse {
    state.requests.fetch_add(1, Ordering::SeqCst);
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":16,\"completion_tokens\":1}}\n\n\
         data: [DONE]\n\n",
    )
        .into_response()
}

fn benchmark_run(legacy: Value) -> Value {
    let mut endpoint = legacy["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    let cfg = json!({
        "models": legacy["resources"]["models"],
        "endpoint": endpoint,
        "datasets": [legacy["workload"]["config"]["dataset"]],
        "phases": legacy["workload"]["config"]["phases"],
        "tokenizer": legacy["workload"]["config"]["tokenizer"],
        "transport": {"type": legacy["transport"]["type"]},
        "runtime": {"workers": legacy["workload"]["config"]["worker_count"]}
    });
    json!({
        "benchmark_id": legacy["identity"]["benchmark_id"],
        "artifact_dir": legacy["artifact_target"],
        "random_seed": legacy["identity"]["random_seed"],
        "cfg": cfg
    })
}

fn run_child(mut request: Value) -> Output {
    request["run"] = benchmark_run(request["run"].take());
    let bytes = serde_json::to_vec(&request["run"]).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}

/// A `t*` window pinned at the midpoint so WARMUP primes the pre-`t*` boundary
/// turn and PROFILING replays the post-`t*` frontier — the active window is what
/// engages the warmup/profiling snapshot split and the cache-pressure priming.
fn synthesis() -> Value {
    json!({
        "speedup_ratio": 1.0,
        "prefix_len_multiplier": 1.0,
        "prefix_root_multiplier": 1,
        "prompt_len_multiplier": 1.0,
        "output_len_multiplier": 1.0,
        "idle_gap_cap_seconds": 60.0,
        "corpus": "sonnet",
        "trajectory_start_min_ratio": 0.5,
        "trajectory_start_max_ratio": 0.5,
        "t_star_random_seed": 0
    })
}

/// One WEKA session with three time-spaced requests so the lowered chain
/// straddles the midpoint `t*`: exactly one node is the pre-`t*` warmup boundary
/// (the corpus is a single template that dispatches ONE node per pass) and the
/// post-`t*` remainder is the profiling frontier. Identical trace shape to
/// `warmup_abort_stdio_e2e.rs`, whose clean case proves warmup dispatches exactly
/// one node without recycle — so any warmup `request_count > 1` here is recycle.
fn weka_dataset() -> Value {
    let request = |t: f64| {
        json!({
            "t": t,
            "type": "s",
            "model": "recorded-model",
            "in": 16,
            "out": 1,
            "hash_ids": [123456789],
            "api_time": 0.2,
            "ttft": 0.1
        })
    };
    json!({
        "type": "file",
        "format": "weka_trace",
        "sampling": "sequential",
        "synthesis": synthesis(),
        "records": [{
            "id": "pressure_trace",
            "models": ["recorded-model"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [request(0.0), request(0.02), request(0.04)]
        }]
    })
}

/// A warmup + profiling request whose WARMUP phase carries a small
/// `agentic_cache_warmup_duration` (the C2 knob), engaging the in-runtime
/// cache-pressure recycle. The duration is tiny (localhost dispatches are
/// sub-millisecond) so the recycle loop turns hundreds of passes within the
/// budget while the whole run finishes in a few seconds.
fn request(
    endpoint: &str,
    artifact_target: &std::path::Path,
    benchmark_id: &str,
    warmup_duration_s: f64,
) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "identity": {"benchmark_id": benchmark_id, "random_seed": 20_260_715_u64},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"strategy": "round_robin", "items": [{"name": "recorded-model"}]},
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "chat",
                    "urls": [endpoint],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                }]}
            },
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "graph", "config": {
                "worker_count": 1,
                "dataset": weka_dataset(),
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [
                    {
                        // The auto cache-pressure warmup carries NO explicit
                        // sessions/requests/duration stop condition: the pressure
                        // duration IS its deadline. Adding one would end the phase
                        // (and cancel the recycle) after the first trace completes.
                        "type": "concurrency",
                        "name": "warmup",
                        "exclude_from_results": true,
                        "concurrency": 1,
                        "agentic_cache_warmup_duration": warmup_duration_s
                    },
                    {
                        "type": "concurrency",
                        "name": "profiling",
                        "exclude_from_results": false,
                        "sessions": 1,
                        "concurrency": 1
                    }
                ]
            }}
        }
    })
}

async fn spawn_mock() -> (String, Arc<MockState>) {
    let state = Arc::new(MockState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), state)
}

/// The `request_count` counter total for a metric map (warmup or profiling): the
/// number of records the phase's accumulator ingested. `request_count` is an
/// Aggregate/Sum metric, serialized as a `counter` with a `total`
/// (`metrics_core::report::report_stats`).
fn request_count_total(metrics: &Value) -> f64 {
    metrics["request_count"]["series"][0]["stats"]["total"]
        .as_f64()
        .unwrap_or_else(|| panic!("request_count.total missing: {metrics}"))
}

/// The full cache-pressure warmup -> handoff -> profiling-resume flow completes
/// through the product stdio path: warmup recycles the corpus more than once,
/// profiling produces records for the resumed post-frontier nodes, and the run
/// reaches a valid report with no hang and no abort.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_pressure_warmup_recycles_then_profiling_resumes() {
    let (endpoint, state) = spawn_mock().await;
    let temporary = tempfile::tempdir().unwrap();
    let artifact_target = temporary.path().join("cache-pressure-warmup");
    // A tiny but comfortably-larger-than-one-dispatch pressure budget: localhost
    // dispatches are sub-millisecond, so 250 ms turns many recycle passes while
    // the whole run stays fast. The recycle loop is Clock-deadline-bounded.
    let request = request(&endpoint, &artifact_target, "cache-pressure-warmup", 0.3);

    // Wall-clock guard: the run is duration-bounded and the mock always answers,
    // so this only ever trips on a genuine hang (never on the happy path).
    let output: Output = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::task::spawn_blocking(move || run_child(request)),
    )
    .await
    .expect("cache-pressure warmup run hung past the timeout budget")
    .unwrap();

    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "non-JSON terminal line: {error}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    // The run completes cleanly: no warmup abort, no failure envelope.
    assert_eq!(terminal["event"], "run_terminal", "{terminal}");
    assert_eq!(terminal["success"], true, "{terminal}");

    // A valid native report was written.
    let report_path = terminal["report_path"].as_str().unwrap();
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0", "{report}");

    // Recycle proof: the WARMUP accumulator ingested strictly more than one
    // corpus pass. The single-template corpus dispatches exactly one node per
    // pass (a plain warmup over this trace dispatches one — see the sibling
    // abort test), so a warmup `request_count > 1` can only come from E3b's
    // pressure recycle turning additional corpus passes.
    let warmup = &report["warmup_metrics"];
    assert!(warmup.is_object(), "warmup_metrics missing: {report}");
    let warmup_requests = request_count_total(warmup);
    assert!(
        warmup_requests > 1.0,
        "expected warmup cache-pressure recycle (>1 corpus pass), saw request_count={warmup_requests}"
    );

    // Profiling produced records for the resumed post-frontier nodes.
    let profiling_requests = request_count_total(&report["metrics"]);
    assert!(
        profiling_requests >= 1.0,
        "expected profiling records for the resumed frontier, saw request_count={profiling_requests}"
    );

    // Corroboration at the wire: total dispatches exceed a single warmup pass
    // plus the profiling frontier, i.e. the recycle actually hit the backend.
    let total_dispatches = state.requests.load(Ordering::SeqCst);
    assert!(
        total_dispatches >= 3,
        "expected recycle + profiling dispatches at the backend, saw {total_dispatches}"
    );
}
