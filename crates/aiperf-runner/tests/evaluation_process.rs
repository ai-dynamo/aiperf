// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product-process proofs for the provider-neutral evaluation pair.

use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

const PROVIDER_ROOTS_ENV: &str = "AIPERF_EVALUATOR_PROVIDER_ROOTS";
const UPSTREAM_KEY_SENTINEL: &str = "upstream-api-key-sentinel-evaluation-process";
const FORBIDDEN_URL_SENTINEL: &str = "https://upstream-url-sentinel.invalid/v1";
const RAW_SSE_SENTINEL: &str = "raw-upstream-sse-must-not-cross-provider-boundary";
const HIDDEN_ANSWER_SENTINEL: &str = "#### 18";
const HIDDEN_QUESTION_SENTINEL: &str = "Janet’s ducks lay 16 eggs per day";
const PUBLIC_SCORE_SCHEMA_SHA256: &str =
    "d156e6577305139bac7f48946996fa35d489a381a87bce4c58d18c47d8d9eeb5";
const GSM8K_SOURCE: &str = "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866";

#[derive(Debug)]
struct CapturedRequest {
    headers: HeaderMap,
    body: Vec<u8>,
}

#[derive(Debug, Default)]
struct ModelServerState {
    readiness_calls: AtomicUsize,
    readiness_artifact_absence: Mutex<Vec<bool>>,
    expected_artifact_target: Mutex<Option<PathBuf>>,
    kserve_chat: Mutex<Vec<CapturedRequest>>,
    failing_chat: Mutex<Vec<CapturedRequest>>,
    messages: Mutex<Vec<CapturedRequest>>,
}

async fn kserve_readiness(State(state): State<Arc<ModelServerState>>) -> impl IntoResponse {
    state.readiness_calls.fetch_add(1, Ordering::SeqCst);
    let artifact_absent = state
        .expected_artifact_target
        .lock()
        .unwrap()
        .as_ref()
        .is_none_or(|path| !path.exists());
    state
        .readiness_artifact_absence
        .lock()
        .unwrap()
        .push(artifact_absent);
    Json(json!({"data": [{"id": "candidate"}]}))
}

async fn kserve_chat(
    State(state): State<Arc<ModelServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    state.kserve_chat.lock().unwrap().push(CapturedRequest {
        headers,
        body: body.to_vec(),
    });
    let response = concat!(
        "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"candidate\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"candidate\",\"raw_sse_sentinel\":\"raw-upstream-sse-must-not-cross-provider-boundary\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"18\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"candidate\",\"choices\":[],\"usage\":{\"prompt_tokens\":31,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], response)
}

async fn anthropic_messages(
    State(state): State<Arc<ModelServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    state.messages.lock().unwrap().push(CapturedRequest {
        headers,
        body: body.to_vec(),
    });
    let response = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"candidate\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":31,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"raw_sse_sentinel\":\"raw-upstream-sse-must-not-cross-provider-boundary\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer: 18\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], response)
}

async fn failing_chat(
    State(state): State<Arc<ModelServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    state.failing_chat.lock().unwrap().push(CapturedRequest {
        headers,
        body: body.to_vec(),
    });
    Json(json!({
        "id": "chat-fatal",
        "object": "chat.completion",
        "created": 0,
        "model": "candidate",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": FORBIDDEN_URL_SENTINEL}
                }]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 31,
            "completion_tokens": 1,
            "total_tokens": 32
        }
    }))
}

fn run_runner(request: &Value, provider_roots: Option<&OsStr>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"));
    if let Some(roots) = provider_roots {
        child.env(PROVIDER_ROOTS_ENV, roots);
    } else {
        child.env_remove(PROVIDER_ROOTS_ENV);
    }
    let mut child = child
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(request).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn capabilities(provider_roots: Option<&OsStr>) -> Value {
    let mut command = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"));
    command.arg("--capabilities");
    if let Some(roots) = provider_roots {
        command.env(PROVIDER_ROOTS_ENV, roots);
    } else {
        command.env_remove(PROVIDER_ROOTS_ENV);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    one_json_line(&output.stdout)
}

fn one_json_line(stdout: &[u8]) -> Value {
    let lines = stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1, "stdout={}", String::from_utf8_lossy(stdout));
    serde_json::from_slice(lines[0]).unwrap()
}

fn provider_roots() -> OsString {
    std::env::var_os(PROVIDER_ROOTS_ENV)
        .expect("ignored stock-provider proof requires AIPERF_EVALUATOR_PROVIDER_ROOTS")
}

fn evaluation_request(
    distribution_id: &Value,
    artifact_target: &Path,
    address: std::net::SocketAddr,
    provider: &str,
) -> Value {
    let (distribution, provider_config, endpoint_type, endpoint_path, readiness) = match provider {
        "nemo_evaluator" => (
            "nvidia_nemo_evaluator_0_4_locked",
            json!({
                "environment": "gsm8k",
                "solver": "chat",
                "solver_config": {"max_tokens": 64, "temperature": 0.0},
                "selection": {"limit": 1, "seed": 0}
            }),
            "kserve_chat",
            "/openai/v1/chat/completions",
            json!({
                "wait_for_model_timeout": 5.0,
                "wait_for_model_interval": 0.01,
                "wait_for_model_mode": "models"
            }),
        ),
        "openbench" => (
            "groq_openbench_0_5_3_inspect_0_3_141_locked",
            json!({"task": "gsm8k", "task_args": {}, "epochs": 1, "limit": 1}),
            "messages",
            "/v1/messages",
            json!({
                "wait_for_model_timeout": 0.0,
                "wait_for_model_interval": 5.0,
                "wait_for_model_mode": "inference"
            }),
        ),
        _ => panic!("unknown stock provider"),
    };
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": distribution_id,
        "run": {
            "identity": {"benchmark_id": format!("evaluation-{provider}-process-proof")},
            "artifact_target": artifact_target,
            "resources": {
                "models": {"items": [{"name": "candidate"}]},
                "endpoints": {"profiles": [{
                    "id": "candidate",
                    "type": endpoint_type,
                    "urls": [format!("http://{address}")],
                    "path": endpoint_path,
                    "streaming": true,
                    "use_server_token_count": true,
                    "api_key": UPSTREAM_KEY_SENTINEL,
                    "wait_for_model_timeout": readiness["wait_for_model_timeout"],
                    "wait_for_model_interval": readiness["wait_for_model_interval"],
                    "wait_for_model_mode": readiness["wait_for_model_mode"]
                }]}
            },
            "backend": {"type": "online_http", "config": {}},
            "workload": {"type": "evaluation", "config": {
                "provider": {"type": provider, "distribution": distribution},
                "evaluation": provider_config,
                "routes": {
                    "candidate": {
                        "model": "candidate",
                        "endpoint_profile": "candidate",
                        "purpose": "primary"
                    }
                },
                "resources": {},
                "unit_concurrency": 1
            }}
        }
    })
}

fn assert_successful_evaluation(
    output: &Output,
    artifact_target: &Path,
    provider: &str,
    upstream_url: &str,
) -> Value {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal = one_json_line(&output.stdout);
    assert_eq!(terminal["success"], true, "{terminal}");
    assert_eq!(terminal["provenance"]["provider"], provider);
    assert_public_sentinels_absent(&serde_json::to_vec(&terminal).unwrap(), upstream_url);
    let report_path = artifact_target.join("native-v2.json");
    let report_bytes = std::fs::read(&report_path).unwrap();
    let report: Value = serde_json::from_slice(&report_bytes).unwrap();
    assert_eq!(report["evaluation"]["case_count"], 1);
    assert_eq!(report["evaluation"]["completed_count"], 1);
    assert_eq!(report["evaluation"]["infrastructure_error_count"], 0);
    assert_eq!(report["evaluation"]["cancelled_count"], 0);
    assert_eq!(report["evaluation"]["routes"][0]["service_id"], "candidate");
    assert_eq!(
        report["evaluation"]["route_summaries"]["candidate"]["logical_operations"],
        1
    );
    assert_public_evaluation_result(&report, provider);
    let artifact_entries = report["evaluation"]["artifacts"].as_array().unwrap();
    assert!(
        artifact_entries
            .iter()
            .all(|artifact| artifact["visibility"] == "restricted")
    );
    let bundle_entries = artifact_entries
        .iter()
        .filter(|artifact| artifact.get("normalized_result_sha256").is_some())
        .collect::<Vec<_>>();
    assert_eq!(bundle_entries.len(), 1, "{artifact_entries:?}");
    assert!(
        bundle_entries[0].get("path").is_none() && bundle_entries[0].get("media_type").is_none(),
        "restricted provider bundle exposed filesystem metadata"
    );
    for artifact in artifact_entries {
        assert_public_sentinels_absent(&serde_json::to_vec(artifact).unwrap(), upstream_url);
    }
    assert_no_worker_root(artifact_target);
    assert!(
        !artifact_target
            .join("evaluation/evaluator-proxy.sock")
            .exists()
    );
    assert_public_sentinels_absent(&output.stdout, upstream_url);
    assert_public_sentinels_absent(&output.stderr, upstream_url);
    assert_public_sentinels_absent(&report_bytes, upstream_url);
    for artifact in report["evaluation"]["artifacts"].as_array().unwrap() {
        if artifact["visibility"] == "public"
            && let Some(path) = artifact["path"].as_str()
        {
            let bytes =
                std::fs::read(artifact_target.join("evaluation/artifacts").join(path)).unwrap();
            assert_bytes_absent(&bytes, HIDDEN_ANSWER_SENTINEL.as_bytes());
            assert_bytes_absent(&bytes, HIDDEN_QUESTION_SENTINEL.as_bytes());
        }
    }
    for path in regular_files(artifact_target) {
        let bytes = std::fs::read(&path).unwrap();
        assert_bytes_absent(&bytes, UPSTREAM_KEY_SENTINEL.as_bytes());
        assert_bytes_absent(&bytes, FORBIDDEN_URL_SENTINEL.as_bytes());
        assert_bytes_absent(&bytes, upstream_url.as_bytes());
        assert_bytes_absent(&bytes, RAW_SSE_SENTINEL.as_bytes());
    }
    assert_tree_metadata_sentinels_absent(artifact_target, upstream_url);
    report
}

fn assert_public_evaluation_result(report: &Value, provider: &str) {
    let evaluation = &report["evaluation"];
    assert_eq!(evaluation["identity"]["provider"], provider);
    let case = &evaluation["cases"][0];
    assert_eq!(case["task"], "gsm8k");
    assert_eq!(case["source"], GSM8K_SOURCE);
    assert_eq!(case["outcome"], "completed");
    let (score_name, expected_config, definition) = match provider {
        "nemo_evaluator" => (
            "reward",
            json!({
                "environment": "gsm8k",
                "environment_config": {},
                "selection": {"limit": 1, "seed": 0},
                "solver": "chat",
                "solver_config": {"max_tokens": 64, "temperature": 0.0},
            }),
            json!({"exclude_cancelled": true, "exclude_infrastructure": true}),
        ),
        "openbench" => (
            "grade_school_math_scorer",
            json!({"epochs": 1, "limit": 1, "task": "gsm8k", "task_args": {}}),
            json!({
                "metric_params": {},
                "params": {},
                "score_name": "grade_school_math_scorer",
            }),
        ),
        _ => panic!("unknown stock provider"),
    };
    assert_eq!(evaluation["config"], expected_config);
    assert_eq!(case["primary_score"], score_name);
    assert_eq!(
        case["scores"][score_name]["projection_schema"],
        PUBLIC_SCORE_SCHEMA_SHA256
    );
    let public_score = case["scores"][score_name]["value"].as_object().unwrap();
    assert_eq!(public_score.len(), 1);
    assert_eq!(public_score["value"].as_f64(), Some(1.0));
    assert_eq!(case["numeric_metrics"]["accuracy"], 1.0);
    assert_eq!(case["numeric_metrics"].as_object().unwrap().len(), 1);
    let aggregates = evaluation["aggregates"].as_array().unwrap();
    assert_eq!(aggregates.len(), 1, "{aggregates:?}");
    let aggregate = &aggregates[0];
    assert_eq!(aggregate["scorer"], "accuracy");
    assert_eq!(aggregate["reducer"], "mean");
    assert_eq!(aggregate["metric"], "accuracy");
    assert_eq!(aggregate["value"], 1.0);
    assert_eq!(aggregate["scored_count"], 1);
    assert_eq!(aggregate["unscored_count"], 0);
    assert_eq!(aggregate["definition"], definition);
}

fn assert_no_worker_root(artifact_target: &Path) {
    let evaluation_root = artifact_target.join("evaluation");
    assert!(
        std::fs::read_dir(&evaluation_root)
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("worker-root-")),
        "materialized provider root survived report commit"
    );
}

fn regular_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.file_type().unwrap();
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                files.push(entry.path());
            } else {
                panic!(
                    "artifact tree contained a special file: {}",
                    entry.path().display()
                );
            }
        }
    }
    files
}

fn assert_tree_metadata_sentinels_absent(root: &Path, upstream_url: &str) {
    if !root.is_dir() {
        return;
    }
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            assert_public_sentinels_absent(
                entry.path().as_os_str().as_encoded_bytes(),
                upstream_url,
            );
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            }
        }
    }
}

fn assert_bytes_absent(haystack: &[u8], needle: &[u8]) {
    assert!(
        !haystack
            .windows(needle.len())
            .any(|window| window == needle),
        "secret sentinel appeared in public/process output"
    );
}

fn assert_public_sentinels_absent(bytes: &[u8], upstream_url: &str) {
    for sentinel in [
        UPSTREAM_KEY_SENTINEL,
        FORBIDDEN_URL_SENTINEL,
        RAW_SSE_SENTINEL,
        HIDDEN_ANSWER_SENTINEL,
        HIDDEN_QUESTION_SENTINEL,
        upstream_url,
    ] {
        assert_bytes_absent(bytes, sentinel.as_bytes());
    }
}

fn assert_no_process_mentions(path: &Path) {
    let needle = path.as_os_str().as_encoded_bytes();
    for entry in std::fs::read_dir("/proc").unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().parse::<u32>().is_err() {
            continue;
        }
        let Ok(command) = std::fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        assert_bytes_absent(&command, needle);
    }
}

#[test]
fn capabilities_omit_evaluation_without_deployment_owned_provider_roots() {
    let capabilities = capabilities(None);
    assert!(
        !capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["online_http", "evaluation"]))
    );
    assert!(
        capabilities["evaluation_providers"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        capabilities["supported_evaluation_combinations"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires two exact provider-specific roots in AIPERF_EVALUATOR_PROVIDER_ROOTS"]
async fn both_stock_providers_execute_through_rust_owned_http_and_sse() {
    let roots = provider_roots();
    let capabilities = capabilities(Some(&roots));
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["online_http", "evaluation"]))
    );
    let combinations = capabilities["supported_evaluation_combinations"]
        .as_array()
        .unwrap();
    assert_eq!(combinations.len(), 2, "{capabilities}");
    assert_eq!(
        combinations
            .iter()
            .map(|item| item["provider"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["nemo_evaluator", "openbench"]
    );
    assert!(
        combinations
            .iter()
            .all(|item| item["operations"] == json!(["model.generate"]))
    );

    let state = Arc::new(ModelServerState::default());
    let app = Router::new()
        .route("/openai/v1/models", get(kserve_readiness))
        .route("/openai/v1/chat/completions", post(kserve_chat))
        .route("/openai/v1/chat/fatal", post(failing_chat))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let artifacts = tempfile::tempdir().unwrap();
    let nemo_target = artifacts.path().join("nemo");
    *state.expected_artifact_target.lock().unwrap() = Some(nemo_target.clone());
    let nemo_request = evaluation_request(
        &capabilities["distribution_id"],
        &nemo_target,
        address,
        "nemo_evaluator",
    );
    let nemo_output = tokio::task::spawn_blocking({
        let roots = roots.clone();
        move || run_runner(&nemo_request, Some(&roots))
    })
    .await
    .unwrap();
    let upstream_url = format!("http://{address}");
    let nemo_report =
        assert_successful_evaluation(&nemo_output, &nemo_target, "nemo_evaluator", &upstream_url);
    assert!(state.readiness_calls.load(Ordering::SeqCst) > 0);
    assert!(
        state
            .readiness_artifact_absence
            .lock()
            .unwrap()
            .iter()
            .all(|absent| *absent),
        "evaluation created artifacts before readiness completed"
    );
    assert!(
        nemo_report["metrics"]["time_to_first_token"]
            .get("series")
            .is_some(),
        "real upstream SSE did not produce native timing metrics"
    );
    let chat = state.kserve_chat.lock().unwrap();
    assert_eq!(chat.len(), 1);
    assert_eq!(
        chat[0]
            .headers
            .get(header::AUTHORIZATION)
            .unwrap()
            .to_str()
            .unwrap(),
        format!("Bearer {UPSTREAM_KEY_SENTINEL}")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&chat[0].body).unwrap()["stream"],
        true
    );
    drop(chat);

    let openbench_target = artifacts.path().join("openbench");
    *state.expected_artifact_target.lock().unwrap() = Some(openbench_target.clone());
    let openbench_request = evaluation_request(
        &capabilities["distribution_id"],
        &openbench_target,
        address,
        "openbench",
    );
    let openbench_output = tokio::task::spawn_blocking({
        let roots = roots.clone();
        move || run_runner(&openbench_request, Some(&roots))
    })
    .await
    .unwrap();
    let openbench_report = assert_successful_evaluation(
        &openbench_output,
        &openbench_target,
        "openbench",
        &upstream_url,
    );
    assert_eq!(
        openbench_report["run"]["evaluation_compatibility"]["dialect_ids"],
        json!(["openai_chat_completions"])
    );
    let messages = state.messages.lock().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0]
            .headers
            .get("x-api-key")
            .unwrap()
            .to_str()
            .unwrap(),
        UPSTREAM_KEY_SENTINEL
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&messages[0].body).unwrap()["stream"],
        true
    );
    drop(messages);

    let abort_target = artifacts.path().join("launched-provider-abort");
    let mut abort_request = evaluation_request(
        &capabilities["distribution_id"],
        &abort_target,
        address,
        "openbench",
    );
    let endpoint = &mut abort_request["run"]["resources"]["endpoints"]["profiles"][0];
    endpoint["type"] = Value::String("chat".to_string());
    endpoint["path"] = Value::String("/openai/v1/chat/fatal".to_string());
    endpoint["streaming"] = Value::Bool(false);
    let abort_output = tokio::task::spawn_blocking({
        let roots = roots.clone();
        move || run_runner(&abort_request, Some(&roots))
    })
    .await
    .unwrap();
    assert_eq!(state.failing_chat.lock().unwrap().len(), 1);
    assert!(!abort_output.status.success());
    let abort_terminal = one_json_line(&abort_output.stdout);
    assert_eq!(abort_terminal["success"], false, "{abort_terminal}");
    assert_public_sentinels_absent(&serde_json::to_vec(&abort_terminal).unwrap(), &upstream_url);
    assert_public_sentinels_absent(&abort_output.stdout, &upstream_url);
    assert_public_sentinels_absent(&abort_output.stderr, &upstream_url);
    assert_bytes_absent(&abort_output.stdout, FORBIDDEN_URL_SENTINEL.as_bytes());
    assert_bytes_absent(&abort_output.stderr, FORBIDDEN_URL_SENTINEL.as_bytes());
    assert_bytes_absent(&abort_output.stdout, UPSTREAM_KEY_SENTINEL.as_bytes());
    assert_bytes_absent(&abort_output.stderr, UPSTREAM_KEY_SENTINEL.as_bytes());
    assert!(!abort_target.join("native-v2.json").exists());
    assert!(
        !abort_target
            .join("evaluation/evaluator-proxy.sock")
            .exists()
    );
    if abort_target.join("evaluation").is_dir() {
        assert_no_worker_root(&abort_target);
    }
    assert_tree_metadata_sentinels_absent(&abort_target, &upstream_url);
    assert_no_process_mentions(&abort_target);

    let failure_target = artifacts.path().join("failure-must-not-exist");
    let mut failure = evaluation_request(
        &capabilities["distribution_id"],
        &failure_target,
        address,
        "nemo_evaluator",
    );
    failure["run"]["workload"]["config"]["evaluation"]["base_url"] =
        Value::String(FORBIDDEN_URL_SENTINEL.to_owned());
    let failure_output = tokio::task::spawn_blocking({
        let roots = roots.clone();
        move || run_runner(&failure, Some(&roots))
    })
    .await
    .unwrap();
    assert!(!failure_output.status.success());
    let failure_terminal = one_json_line(&failure_output.stdout);
    assert_eq!(failure_terminal["success"], false);
    assert_public_sentinels_absent(
        &serde_json::to_vec(&failure_terminal).unwrap(),
        &upstream_url,
    );
    assert_public_sentinels_absent(&failure_output.stdout, &upstream_url);
    assert_public_sentinels_absent(&failure_output.stderr, &upstream_url);
    assert_bytes_absent(&failure_output.stdout, FORBIDDEN_URL_SENTINEL.as_bytes());
    assert_bytes_absent(&failure_output.stderr, FORBIDDEN_URL_SENTINEL.as_bytes());
    assert_bytes_absent(&failure_output.stdout, UPSTREAM_KEY_SENTINEL.as_bytes());
    assert_bytes_absent(&failure_output.stderr, UPSTREAM_KEY_SENTINEL.as_bytes());
    assert_bytes_absent(&failure_output.stdout, upstream_url.as_bytes());
    assert_bytes_absent(&failure_output.stderr, upstream_url.as_bytes());
    assert_tree_metadata_sentinels_absent(&failure_target, &upstream_url);
    assert!(!failure_target.exists());
}
