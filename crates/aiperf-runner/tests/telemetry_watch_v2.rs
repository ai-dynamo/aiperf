// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real subprocess proofs for standalone telemetry watch and source isolation.

use std::io::Write;
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use aiperf_telemetry_archive::{
    ArchiveId, LocalArchiveRepository, NoDurabilityFaults, ProjectionCoverageV1, QualifiedSpool,
    RemoteLatestV1, TableId,
};
use axum::Router;
use axum::body::Body;
use axum::http::{Response, StatusCode, header};
use axum::routing::get;
use serde_json::{Value, json};

fn capabilities() -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

fn run_child(request: Value) -> Output {
    spawn_child(request).wait_with_output().unwrap()
}

fn spawn_child(request: Value) -> Child {
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_aiperf-runner"))
        .env(
            "AIPERF_ARCHIVE_KEY_ARCHIVE_IDENTITY",
            "0707070707070707070707070707070707070707070707070707070707070707",
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child
}

fn watch_request(
    capabilities: &Value,
    root: &std::path::Path,
    source_url: String,
    benchmark_id: &str,
) -> (
    Value,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let artifact = root.join(format!("{benchmark_id}-artifacts"));
    let spool = root.join(format!("{benchmark_id}-spool"));
    let remote = root.join(format!("{benchmark_id}-remote"));
    let target = url::Url::from_directory_path(&remote).unwrap().to_string();
    (
        json!({
            "protocol_version": 2,
            "operation": "execute",
            "expected_distribution_id": capabilities["distribution_id"],
            "run": {
                "identity": {"benchmark_id": benchmark_id},
                "artifact_target": artifact,
                "transport": {"type": "http", "config": {}},
                "workload": {"type": "telemetry_watch", "config": {
                    "mode": "collect",
                    "duration_ns": 150_000_000_i64,
                    "shutdown_timeout_ns": 2_000_000_000_i64,
                    "sources": [{
                        "id": "node-a",
                        "type": "prometheus_http",
                        "interval_ns": 20_000_000_i64,
                        "request_timeout_ns": 15_000_000_i64,
                        "config": {
                            "url": source_url,
                            "connect_timeout_ns": 10_000_000_i64,
                            "redirects": "disabled",
                            "proxy": "disabled",
                            "accepted_formats": ["prometheus_text_0_0_4"],
                            "max_compressed_bytes": 4096,
                            "max_decompressed_bytes": 16384
                        },
                        "attributes": {"role": "node"}
                    }],
                    "archive": {
                        "target": target,
                        "local_spool": spool,
                        "spool_quota_bytes": 4_294_967_296_u64,
                        "spool_quota_files": 10_000_u64,
                        "required": true,
                        "writer": {"type": "parquet_archive_v1", "config": {}},
                        "store_access": {"type": "local_filesystem", "config": {}},
                        "rotation": {"type": "rows_bytes_age", "config": {}},
                        "admission": {"type": "primary_durable", "config": {}},
                        "recovery": {"type": "create_new", "config": {}},
                        "archive_key": {
                            "type": "secret_provider",
                            "config": {"id": "archive-identity"}
                        },
                        "enrichers": [],
                        "sanitizers": [],
                        "raw_body": {"type": "none", "config": {}}
                    }
                }},
                "resources": {}
            }
        }),
        artifact,
        remote,
        spool,
    )
}

fn assert_success(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], true, "{terminal}");
    terminal
}

fn sync_request(
    capabilities: &Value,
    artifact: &std::path::Path,
    spool: &std::path::Path,
    remote: &std::path::Path,
    archive_id: &str,
) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": capabilities["distribution_id"],
        "run": {
            "identity": {"benchmark_id": "telemetry-watch-sync"},
            "artifact_target": artifact,
            "transport": {"type": "http", "config": {}},
            "workload": {"type": "telemetry_watch", "config": {
                "mode": "finalize_remote",
                "shutdown_timeout_ns": 2_000_000_000_i64,
                "archive": {
                    "archive_id": archive_id,
                    "target": url::Url::from_directory_path(remote).unwrap().to_string(),
                    "local_spool": spool,
                    "store_access": {"type": "local_filesystem", "config": {}},
                    "recovery": {"type": "finalize_remote", "config": {}},
                    "archive_key": {
                        "type": "secret_provider",
                        "config": {"id": "archive-identity"}
                    }
                }
            }},
            "resources": {}
        }
    })
}

async fn metrics_response(counter: Arc<AtomicUsize>) -> Response<Body> {
    counter.fetch_add(1, Ordering::SeqCst);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(
            "# HELP temperature fixture\n# TYPE temperature gauge\ntemperature{sensor=\"a\"} 42\n",
        ))
        .unwrap()
}

async fn wait_for_path(path: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn recover_coverages(
    spool: &std::path::Path,
    remote: &std::path::Path,
    archive_id: &str,
) -> Vec<ProjectionCoverageV1> {
    let archive_uuid = uuid::Uuid::parse_str(archive_id).unwrap();
    let archive_id = ArchiveId::new(*archive_uuid.as_bytes()).unwrap();
    let target = url::Url::from_directory_path(remote).unwrap().to_string();
    let repository = LocalArchiveRepository::recover_existing(
        QualifiedSpool::open(spool).unwrap(),
        archive_id,
        aiperf_telemetry_archive::manifest::archive_target_digest(&target),
        &NoDurabilityFaults,
    )
    .unwrap();
    repository
        .index()
        .entries()
        .filter_map(|entry| {
            ProjectionCoverageV1::from_canonical_bytes(entry.descriptor_bytes()).ok()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn collect_runs_real_sources_and_commits_local_and_remote_heads() {
    let counter = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/metrics",
        get({
            let counter = counter.clone();
            move || metrics_response(counter.clone())
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    assert!(
        capabilities["supported_pairs"]
            .as_array()
            .unwrap()
            .contains(&json!(["http", "telemetry_watch"]))
    );
    let temporary = tempfile::tempdir().unwrap();
    let (request, artifact, remote, _spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/metrics"),
        "telemetry-watch-success",
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert!(counter.load(Ordering::SeqCst) >= 2);
    assert!(remote.join("LATEST").is_file());
    let report_path = terminal["report_path"].as_str().unwrap();
    assert_eq!(
        std::path::Path::new(report_path),
        artifact.join("native-v2.json")
    );
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["metrics"], json!({}));
    assert_eq!(report["telemetry_archive"]["state"], "remotely_finalized");
    assert_eq!(report["telemetry_archive"]["finalized_local"], true);
    assert_eq!(report["telemetry_archive"]["finalized_remote"], true);
}

async fn status_500_with_metric_body() -> Response<Body> {
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(
            "# TYPE must_not_parse gauge\nmust_not_parse 42\n",
        ))
        .unwrap()
}

#[cfg(unix)]
fn set_tree_read_only(path: &std::path::Path, read_only: bool) {
    use std::os::unix::fs::PermissionsExt;

    if !read_only {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let child = entry.path();
        if child.is_dir() {
            set_tree_read_only(&child, read_only);
        } else {
            std::fs::set_permissions(
                &child,
                std::fs::Permissions::from_mode(if read_only { 0o444 } else { 0o644 }),
            )
            .unwrap();
        }
    }
    if read_only {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o555)).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metric_looking_http_500_is_archived_without_failing_the_watch() {
    let app = Router::new().route("/metrics", get(status_500_with_metric_body));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    let temporary = tempfile::tempdir().unwrap();
    let (request, _artifact, remote, spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/metrics"),
        "telemetry-watch-http-500",
    );
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert!(remote.join("LATEST").is_file());
    let archive_id = terminal["provenance"]["telemetry_archive_id"]
        .as_str()
        .unwrap();
    let coverages = recover_coverages(&spool, &remote, archive_id);
    assert!(
        coverages
            .iter()
            .any(|coverage| coverage.table == TableId::Attempts && coverage.row_count > 0)
    );
    assert!(
        coverages
            .iter()
            .any(|coverage| coverage.table == TableId::Samples)
    );
    assert!(
        coverages
            .iter()
            .filter(|coverage| coverage.table == TableId::Samples)
            .all(|coverage| coverage.row_count == 0),
        "non-2xx metric-looking body produced sample rows"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_cadences_isolate_slow_oversized_and_malformed_sources() {
    let fast_counter = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(
            "/fast",
            get({
                let counter = fast_counter.clone();
                move || metrics_response(counter.clone())
            }),
        )
        .route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
                    .body(Body::from("# TYPE slow gauge\nslow 1\n"))
                    .unwrap()
            }),
        )
        .route(
            "/oversized",
            get(|| async {
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
                    .body(Body::from("x".repeat(16 * 1024)))
                    .unwrap()
            }),
        )
        .route(
            "/malformed",
            get(|| async {
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
                    .body(Body::from(
                        "# TYPE broken gauge\nbroken{label=\"unterminated} 1\n",
                    ))
                    .unwrap()
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    let temporary = tempfile::tempdir().unwrap();
    let (mut request, _artifact, remote, spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/fast"),
        "telemetry-watch-source-isolation",
    );
    request["run"]["workload"]["config"]["duration_ns"] = json!(300_000_000_i64);
    let template = request["run"]["workload"]["config"]["sources"][0].clone();
    let mut sources = Vec::new();
    for (id, route, interval_ns, timeout_ns, compressed_limit) in [
        (
            "node-fast",
            "fast",
            20_000_000_i64,
            15_000_000_i64,
            4096_u64,
        ),
        (
            "node-slow",
            "slow",
            30_000_000_i64,
            20_000_000_i64,
            4096_u64,
        ),
        (
            "node-oversized",
            "oversized",
            40_000_000_i64,
            30_000_000_i64,
            1024_u64,
        ),
        (
            "node-malformed",
            "malformed",
            50_000_000_i64,
            30_000_000_i64,
            4096_u64,
        ),
    ] {
        let mut source = template.clone();
        source["id"] = json!(id);
        source["interval_ns"] = json!(interval_ns);
        source["request_timeout_ns"] = json!(timeout_ns);
        source["config"]["url"] = json!(format!("http://{address}/{route}"));
        source["config"]["max_compressed_bytes"] = json!(compressed_limit);
        source["attributes"] = json!({"role": id});
        sources.push(source);
    }
    request["run"]["workload"]["config"]["sources"] = Value::Array(sources);

    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert!(fast_counter.load(Ordering::SeqCst) >= 2);
    let archive_id = terminal["provenance"]["telemetry_archive_id"]
        .as_str()
        .unwrap();
    let coverages = recover_coverages(&spool, &remote, archive_id);
    for source_id in ["node-fast", "node-slow", "node-oversized", "node-malformed"] {
        assert!(coverages.iter().any(|coverage| {
            coverage.table == TableId::Attempts
                && coverage.source_id.as_deref() == Some(source_id)
                && coverage.row_count > 0
        }));
    }
    assert!(coverages.iter().any(|coverage| {
        coverage.table == TableId::Samples
            && coverage.source_id.as_deref() == Some("node-fast")
            && coverage.row_count > 0
    }));
    for failed_source in ["node-slow", "node-oversized", "node-malformed"] {
        assert!(coverages.iter().any(|coverage| {
            coverage.table == TableId::Samples
                && coverage.source_id.as_deref() == Some(failed_source)
        }));
        assert!(
            coverages
                .iter()
                .filter(|coverage| {
                    coverage.table == TableId::Samples
                        && coverage.source_id.as_deref() == Some(failed_source)
                })
                .all(|coverage| coverage.row_count == 0)
        );
    }
}

#[cfg(unix)]
async fn run_remote_finalization_failure_case(
    required: bool,
) -> (Output, std::path::PathBuf, tempfile::TempDir) {
    let temporary = tempfile::tempdir().unwrap();
    let capabilities = capabilities();
    let remote_holder = Arc::new(std::sync::Mutex::new(None::<std::path::PathBuf>));
    let frozen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let app = Router::new().route(
        "/metrics",
        get({
            let remote_holder = remote_holder.clone();
            let frozen = frozen.clone();
            move || {
                let remote_holder = remote_holder.clone();
                let frozen = frozen.clone();
                async move {
                    if !frozen.swap(true, Ordering::SeqCst) {
                        let remote = remote_holder
                            .lock()
                            .unwrap()
                            .clone()
                            .expect("test remote path must be installed before collection");
                        set_tree_read_only(&remote, true);
                    }
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
                        .body(Body::from("# TYPE live gauge\nlive 1\n"))
                        .unwrap()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (mut request, artifact, remote, _spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/metrics"),
        if required {
            "telemetry-watch-required-remote-failure"
        } else {
            "telemetry-watch-optional-remote-failure"
        },
    );
    request["run"]["workload"]["config"]["archive"]["required"] = json!(required);
    *remote_holder.lock().unwrap() = Some(remote.clone());
    let output = tokio::task::spawn_blocking(move || run_child(request))
        .await
        .unwrap();
    set_tree_read_only(&remote, false);
    (output, artifact, temporary)
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn required_remote_failure_is_diagnostic_only_without_report_path() {
    let (output, artifact, _temporary) = run_remote_finalization_failure_case(true).await;
    assert!(!output.status.success());
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["success"], false);
    assert_eq!(terminal["stage"], "reporting");
    assert!(terminal.get("report_path").is_none());
    assert_eq!(
        terminal["diagnostic_artifacts"].as_array().unwrap().len(),
        1
    );
    let evidence = &terminal["diagnostic_artifacts"][0];
    assert_eq!(evidence["kind"], "archive_failure_diagnostic");
    let path = artifact.join(evidence["relative_path"].as_str().unwrap());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        evidence["content_hash"],
        format!("blake3:{}", blake3::hash(&bytes).to_hex())
    );
    assert!(!artifact.join("native-v2.json").exists());
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn optional_remote_failure_returns_a_locally_finalized_report() {
    let (output, artifact, _temporary) = run_remote_finalization_failure_case(false).await;
    let terminal = assert_success(&output);
    let report: Value =
        serde_json::from_slice(&std::fs::read(terminal["report_path"].as_str().unwrap()).unwrap())
            .unwrap();
    assert_eq!(report["telemetry_archive"]["state"], "locally_finalized");
    assert_eq!(report["telemetry_archive"]["finalized_local"], true);
    assert_eq!(report["telemetry_archive"]["finalized_remote"], false);
    assert!(artifact.join("native-v2.json").is_file());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sigterm_drains_and_finalizes_both_archive_heads() {
    let counter = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/metrics",
        get({
            let counter = counter.clone();
            move || metrics_response(counter.clone())
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    let temporary = tempfile::tempdir().unwrap();
    let (mut request, _artifact, remote, _spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/metrics"),
        "telemetry-watch-sigterm",
    );
    request["run"]["workload"]["config"]["duration_ns"] = Value::Null;
    let mut child = spawn_child(request);
    wait_for_path(&remote.join("LATEST")).await;
    Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status()
        .unwrap();
    let output = tokio::task::spawn_blocking(move || child.wait_with_output().unwrap())
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert_eq!(terminal["success"], true);
    let remote_head =
        RemoteLatestV1::decode(&std::fs::read(remote.join("LATEST")).unwrap()).unwrap();
    assert!(remote_head.writer_claim.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forced_crash_exact_resume_preserves_authority_and_finishes() {
    let counter = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/metrics",
        get({
            let counter = counter.clone();
            move || metrics_response(counter.clone())
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    let temporary = tempfile::tempdir().unwrap();
    let source_url = format!("http://{address}/metrics");
    let (mut first, _first_artifact, remote, spool) = watch_request(
        &capabilities,
        temporary.path(),
        source_url.clone(),
        "telemetry-watch-crash",
    );
    first["run"]["workload"]["config"]["duration_ns"] = json!(60_000_000_000_i64);
    let mut child = spawn_child(first);
    wait_for_path(&remote.join("LATEST")).await;
    while counter.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    child.kill().unwrap();
    let crashed = child.wait_with_output().unwrap();
    assert!(!crashed.status.success());

    let crashed_remote =
        RemoteLatestV1::decode(&std::fs::read(remote.join("LATEST")).unwrap()).unwrap();
    let prior_claim = crashed_remote
        .writer_claim
        .expect("crash must leave the active fence");
    let archive_id = uuid::Uuid::from_bytes(*crashed_remote.head.archive_id.as_bytes());
    let prior_claim_id = prior_claim.claim_id().to_hex();

    let (mut resumed, _artifact, _unused_remote, _unused_spool) = watch_request(
        &capabilities,
        temporary.path(),
        source_url,
        "telemetry-watch-resume",
    );
    resumed["run"]["workload"]["config"]["archive"]["local_spool"] = json!(spool);
    resumed["run"]["workload"]["config"]["archive"]["target"] =
        json!(url::Url::from_directory_path(&remote).unwrap().to_string());
    resumed["run"]["workload"]["config"]["archive"]["recovery"] = json!({
        "type": "exact_resume",
        "config": {
            "archive_id": archive_id,
            "prior_claim_id": prior_claim_id
        }
    });
    let output = tokio::task::spawn_blocking(move || run_child(resumed))
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert_eq!(
        terminal["provenance"]["telemetry_archive_id"],
        archive_id.to_string()
    );
    let finalized = RemoteLatestV1::decode(&std::fs::read(remote.join("LATEST")).unwrap()).unwrap();
    assert!(finalized.writer_claim.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn source_free_sync_uses_no_source_and_reports_no_new_collection_session() {
    let counter = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/metrics",
        get({
            let counter = counter.clone();
            move || metrics_response(counter.clone())
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let capabilities = capabilities();
    let temporary = tempfile::tempdir().unwrap();
    let (collect, _artifact, remote, spool) = watch_request(
        &capabilities,
        temporary.path(),
        format!("http://{address}/metrics"),
        "telemetry-watch-before-sync",
    );
    let first = tokio::task::spawn_blocking(move || run_child(collect))
        .await
        .unwrap();
    let first_terminal = assert_success(&first);
    let archive_id = first_terminal["provenance"]["telemetry_archive_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let attempts_after_collect = counter.load(Ordering::SeqCst);

    std::fs::remove_file(remote.join("LATEST")).unwrap();
    let sync_artifact = temporary.path().join("telemetry-watch-sync-artifacts");
    let sync = sync_request(&capabilities, &sync_artifact, &spool, &remote, &archive_id);
    let output = tokio::task::spawn_blocking(move || run_child(sync))
        .await
        .unwrap();
    let terminal = assert_success(&output);
    assert_eq!(counter.load(Ordering::SeqCst), attempts_after_collect);
    let report: Value =
        serde_json::from_slice(&std::fs::read(terminal["report_path"].as_str().unwrap()).unwrap())
            .unwrap();
    let archive = &report["telemetry_archive"];
    assert!(archive["collection_session_id"].is_null());
    assert!(archive["latest_collection_session_id"].is_string());
    assert_eq!(archive["finalized_remote"], true);
    assert!(remote.join("LATEST").is_file());
}
