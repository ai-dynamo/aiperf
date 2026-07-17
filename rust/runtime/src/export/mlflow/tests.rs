// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract and backend tests for the MLflow exporter.
//!
//! Metric, parameter, and tag payloads are asserted directly. REST behavior uses
//! an in-process recording HTTP server, and FileStore behavior uses a temp directory.

use super::*;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::export::ExportConfig;
use crate::metrics_core::{
    MetricSeries, NativeReport, ReportCounterStats, ReportDistributionStats, ReportScalarStats,
    ReportStats, ReportValue,
};

// --- report construction helpers -------------------------------------------

fn distribution(
    avg: Option<ReportValue>,
    min: Option<f64>,
    max: Option<f64>,
    std: Option<f64>,
    count: Option<usize>,
    percentiles: &[(&str, f64)],
) -> ReportStats {
    ReportStats::Distribution(ReportDistributionStats {
        count,
        avg,
        min: min.map(ReportValue::Finite),
        max: max.map(ReportValue::Finite),
        std: std.map(ReportValue::Finite),
        percentiles: percentiles
            .iter()
            .map(|(k, v)| (k.to_string(), ReportValue::Finite(*v)))
            .collect(),
    })
}

fn entry(metric_type: &'static str, stats: ReportStats) -> MetricEntry {
    MetricEntry {
        metric_type,
        unit: "ms".to_string(),
        group: "default",
        higher_is_better: false,
        series: vec![MetricSeries {
            labels: None,
            endpoint_url: None,
            stats,
            timeslices: Vec::new(),
        }],
    }
}

/// Build a synthetic native report exercising every stat shape.
fn sample_report() -> NativeReport {
    let mut report = NativeReport::new(&crate::metrics_core::AccumulatorSummary::new(), None);
    report.aiperf_version = "9.9.9".to_string();
    report.summary.was_cancelled = false;

    let mut metrics = BTreeMap::new();
    metrics.insert(
        "request_latency".to_string(),
        entry(
            "distribution",
            distribution(
                Some(ReportValue::Finite(100.0)),
                Some(10.0),
                Some(200.0),
                Some(5.0),
                Some(3),
                &[("p50", 90.0), ("p90", 150.0), ("p99", 199.0)],
            ),
        ),
    );
    // A non-finite average is omitted while finite sibling stats remain.
    metrics.insert(
        "adj_request_latency".to_string(),
        entry(
            "distribution",
            distribution(
                Some(ReportValue::NonFinite),
                Some(1.0),
                None,
                None,
                None,
                &[],
            ),
        ),
    );
    metrics.insert(
        "request_count".to_string(),
        entry(
            "counter",
            ReportStats::Counter(ReportCounterStats {
                total: ReportValue::Finite(3.0),
                rate: Some(ReportValue::Finite(1.5)),
            }),
        ),
    );
    metrics.insert(
        "request_throughput".to_string(),
        entry(
            "scalar",
            ReportStats::Scalar(ReportScalarStats {
                value: ReportValue::Finite(1.5),
            }),
        ),
    );
    report.metrics = metrics;
    report
}

fn sample_config(tracking_uri: &str) -> MlflowExportConfig {
    MlflowExportConfig {
        enabled: true,
        tracking_uri: Some(tracking_uri.to_string()),
        experiment: Some("aiperf".to_string()),
        run_name: Some("my-run".to_string()),
        parent_run_id: None,
        tags: BTreeMap::from([("team".to_string(), "perf".to_string())]),
        artifact_globs: Vec::new(),
        benchmark_id: Some("abcdef1234567".to_string()),
        aiperf_version: None,
        params: BTreeMap::from([
            ("endpoint.type".to_string(), "chat".to_string()),
            ("endpoint.models".to_string(), "m".to_string()),
            ("timing.mode".to_string(), "concurrency".to_string()),
        ]),
        total_expected_requests: Some(4.0),
        export_timeout_seconds: Some(10),
    }
}

// --- payload parity tests --------------------------------------------------

#[test]
fn metric_payload_key_scheme_matches_python() {
    // Averages use the bare metric tag; other stats use `tag.stat`.
    let report = sample_report();
    let cfg = sample_config("http://unused");
    let payload = build_metric_payload(&report, &cfg);

    // Distribution: bare tag == avg, plus each present stat suffixed.
    assert_eq!(payload.get("request_latency"), Some(&100.0));
    assert_eq!(payload.get("request_latency.min"), Some(&10.0));
    assert_eq!(payload.get("request_latency.max"), Some(&200.0));
    assert_eq!(payload.get("request_latency.std"), Some(&5.0));
    assert_eq!(payload.get("request_latency.count"), Some(&3.0));
    assert_eq!(payload.get("request_latency.p50"), Some(&90.0));
    assert_eq!(payload.get("request_latency.p90"), Some(&150.0));
    assert_eq!(payload.get("request_latency.p99"), Some(&199.0));

    // Non-finite avg is skipped; its finite tail survives.
    assert!(!payload.contains_key("adj_request_latency"));
    assert_eq!(payload.get("adj_request_latency.min"), Some(&1.0));

    // Counter total -> bare tag; rate is NOT a `_STAT_FIELD` and is never keyed.
    assert_eq!(payload.get("request_count"), Some(&3.0));
    assert!(!payload.contains_key("request_count.rate"));

    // Scalar value -> bare tag.
    assert_eq!(payload.get("request_throughput"), Some(&1.5));

    // Synthetic completed and expected request metrics.
    assert_eq!(payload.get("aiperf.completed_requests"), Some(&3.0));
    assert_eq!(payload.get("aiperf.total_expected_requests"), Some(&4.0));
}

#[test]
fn tag_payload_matches_python() {
    // System tags precede user tags.
    let report = sample_report();
    let cfg = sample_config("http://unused");
    let tags = build_tag_payload(&report, &cfg);
    assert_eq!(tags.get("aiperf.version"), Some(&"9.9.9".to_string()));
    assert_eq!(tags.get("aiperf.was_cancelled"), Some(&"false".to_string()));
    assert_eq!(tags.get("benchmark_id"), Some(&"abcdef1234567".to_string()));
    assert_eq!(tags.get("team"), Some(&"perf".to_string()));
}

#[test]
fn run_name_prefers_cli_then_benchmark_then_epoch() {
    // Run-name precedence is explicit name, benchmark prefix, then epoch.
    let mut cfg = sample_config("http://unused");
    assert_eq!(resolve_run_name(&cfg), "my-run");
    cfg.run_name = None;
    assert_eq!(resolve_run_name(&cfg), "aiperf-abcdef12");
    cfg.benchmark_id = None;
    assert!(resolve_run_name(&cfg).starts_with("aiperf-"));
}

#[test]
fn glob_matcher_handles_defaults() {
    assert!(glob_match("*.json", "profile_export.json"));
    assert!(!glob_match("*.json", "sub/profile_export.json"));
    assert!(glob_match("**/*.png", "plots/a/b.png"));
    assert!(glob_match("**/*.png", "b.png"));
    assert!(glob_match("*_timeslices.*", "latency_timeslices.csv"));
    assert!(!glob_match("*.csv", "a.json"));
}

// --- REST backend test -----------------------------------------------------

#[derive(Clone)]
struct Recorded {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// A tiny recording HTTP server: accepts one request per connection, records
/// method/path/body, and answers with canned MLflow responses. Enough to assert
/// the exact bodies the exporter sends.
struct MockServer {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Whether the mock's `get-by-name` reports the experiment as already existing
/// (200 with an id) or missing (404, forcing the create path).
#[derive(Clone, Copy)]
enum ExperimentState {
    Missing,
    Existing,
}

impl MockServer {
    fn start() -> Self {
        Self::start_with(ExperimentState::Missing)
    }

    fn start_with(state: ExperimentState) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let requests_thread = requests.clone();
        let stop_thread = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).ok();
                        if let Some(rec) = read_request(&mut stream) {
                            let response = canned_response(&rec.path, state);
                            let _ = stream.write_all(response.as_bytes());
                            let _ = stream.flush();
                            requests_thread.lock().unwrap().push(rec);
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        MockServer {
            addr,
            requests,
            stop,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    // Read until end of headers.
    let header_end = loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = header_text.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let content_length = header_text
        .lines()
        .find_map(|l| {
            let lower = l.to_ascii_lowercase();
            lower
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some(Recorded { method, path, body })
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn canned_response(path: &str, state: ExperimentState) -> String {
    let (code, body) = if path.contains("experiments/get-by-name") {
        match state {
            // Force the create path.
            ExperimentState::Missing => (
                404,
                r#"{"error_code":"RESOURCE_DOES_NOT_EXIST","message":"no experiment"}"#.to_string(),
            ),
            // Experiment already exists: the exporter must reuse it, never create.
            ExperimentState::Existing => (
                200,
                r#"{"experiment":{"experiment_id":"7","name":"aiperf"}}"#.to_string(),
            ),
        }
    } else if path.contains("experiments/create") {
        (200, r#"{"experiment_id":"7"}"#.to_string())
    } else if path.contains("runs/create") {
        (
            200,
            r#"{"run":{"info":{"run_id":"run-xyz","run_name":"my-run","artifact_uri":"mlflow-artifacts:/7/run-xyz/artifacts"}}}"#
                .to_string(),
        )
    } else {
        (200, "{}".to_string())
    };
    format!(
        "HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn rest_upload_sends_expected_experiment_run_and_log_batch_bodies() {
    let server = MockServer::start();
    let report = sample_report();
    let cfg = sample_config(&server.addr);
    let export_cfg = ExportConfig {
        mlflow: cfg,
        ..ExportConfig::default()
    };
    // A dedicated, mostly-empty artifact dir with one matching file, so the
    // proxy-upload path runs deterministically (never scanning the whole /tmp).
    let artifact_dir =
        std::env::temp_dir().join(format!("aiperf-mlflow-rest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&artifact_dir);
    std::fs::create_dir_all(&artifact_dir).unwrap();
    std::fs::write(artifact_dir.join("profile_export.json"), b"{}").unwrap();

    MlflowExporter
        .export(&report, &artifact_dir, &export_cfg)
        .expect("REST export should succeed against the mock server");

    let requests = server.requests();
    let paths: Vec<&str> = requests.iter().map(|r| r.path.as_str()).collect();
    assert!(paths.iter().any(|p| p.contains("experiments/get-by-name")));
    assert!(paths.iter().any(|p| p.contains("experiments/create")));
    // `get-by-name` must be a GET (POST returns 405 from a real MLflow server).
    assert!(
        requests
            .iter()
            .any(|r| r.path.contains("experiments/get-by-name") && r.method == "GET")
    );
    // Artifact uploaded through the mlflow-artifacts proxy under exports/.
    assert!(requests.iter().any(|r| {
        r.method == "PUT"
            && r.path.contains(
                "mlflow-artifacts/artifacts/7/run-xyz/artifacts/exports/profile_export.json",
            )
    }));
    let _ = std::fs::remove_dir_all(&artifact_dir);
    assert!(
        requests
            .iter()
            .any(|r| r.path.contains("runs/create") && r.method == "POST")
    );
    assert!(
        requests
            .iter()
            .any(|r| r.path.contains("runs/update") && r.method == "POST")
    );

    // Merge every log-batch body and assert the aggregate metric/param/tag sets.
    let mut metric_keys = std::collections::BTreeSet::new();
    let mut param_keys = std::collections::BTreeSet::new();
    let mut tag_keys = std::collections::BTreeSet::new();
    for req in requests
        .iter()
        .filter(|r| r.path.contains("runs/log-batch"))
    {
        let value: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        for m in value["metrics"].as_array().into_iter().flatten() {
            metric_keys.insert(m["key"].as_str().unwrap().to_string());
        }
        for p in value["params"].as_array().into_iter().flatten() {
            param_keys.insert(p["key"].as_str().unwrap().to_string());
        }
        for t in value["tags"].as_array().into_iter().flatten() {
            tag_keys.insert(t["key"].as_str().unwrap().to_string());
        }
    }

    for key in [
        "request_latency",
        "request_latency.p50",
        "request_latency.min",
        "request_count",
        "request_throughput",
        "aiperf.completed_requests",
        "aiperf.total_expected_requests",
    ] {
        assert!(metric_keys.contains(key), "missing metric key {key}");
    }
    for key in ["endpoint.type", "endpoint.models", "timing.mode"] {
        assert!(param_keys.contains(key), "missing param key {key}");
    }
    for key in [
        "aiperf.version",
        "aiperf.was_cancelled",
        "benchmark_id",
        "team",
    ] {
        assert!(tag_keys.contains(key), "missing tag key {key}");
    }
}

/// Reuses an existing experiment via GET without creating it again.
#[test]
fn rest_upload_reuses_existing_experiment_without_create() {
    let server = MockServer::start_with(ExperimentState::Existing);
    let report = sample_report();
    let cfg = sample_config(&server.addr);
    let export_cfg = ExportConfig {
        mlflow: cfg,
        ..ExportConfig::default()
    };
    let artifact_dir =
        std::env::temp_dir().join(format!("aiperf-mlflow-existing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&artifact_dir);
    std::fs::create_dir_all(&artifact_dir).unwrap();

    MlflowExporter
        .export(&report, &artifact_dir, &export_cfg)
        .expect("REST export should succeed reusing the existing experiment");

    let requests = server.requests();
    let _ = std::fs::remove_dir_all(&artifact_dir);

    assert!(
        requests
            .iter()
            .any(|r| r.path.contains("experiments/get-by-name") && r.method == "GET"),
        "expected a GET experiments/get-by-name"
    );
    assert!(
        !requests
            .iter()
            .any(|r| r.path.contains("experiments/create")),
        "must not create an experiment that already exists"
    );
    assert!(
        requests
            .iter()
            .any(|r| r.path.contains("runs/create") && r.method == "POST"),
        "expected the run to be created"
    );
}

#[test]
fn rest_upload_times_out_on_dead_server() {
    // Bind and drop the listener so connection attempts fail immediately.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let report = sample_report();
    let mut cfg = sample_config(&addr);
    cfg.export_timeout_seconds = Some(2);
    let export_cfg = ExportConfig {
        mlflow: cfg,
        ..ExportConfig::default()
    };
    let result = MlflowExporter.export(&report, &std::env::temp_dir(), &export_cfg);
    assert!(result.is_err(), "dead server must surface an error");
}

// --- FileStore backend test ------------------------------------------------

#[test]
fn file_store_writes_mlruns_layout() {
    let temp = std::env::temp_dir().join(format!("aiperf-mlflow-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let root = temp.join("mlruns");
    let tracking_uri = format!("file://{}", root.display());

    // Seed an artifact so the artifact copy path runs.
    let artifact_dir = temp.join("artifacts_src");
    std::fs::create_dir_all(&artifact_dir).unwrap();
    std::fs::write(artifact_dir.join("profile_export.json"), b"{}").unwrap();

    let report = sample_report();
    let cfg = sample_config(&tracking_uri);
    let export_cfg = ExportConfig {
        mlflow: cfg,
        ..ExportConfig::default()
    };
    MlflowExporter
        .export(&report, &artifact_dir, &export_cfg)
        .expect("file store export should succeed");

    // One experiment dir, one run dir.
    let exp_dir = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_dir() && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
        .expect("experiment dir");
    assert!(exp_dir.join("meta.yaml").exists());

    let run_dir = std::fs::read_dir(&exp_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("run dir");

    // Metrics, parameters, and tags each occupy their FileStore subtree.
    assert!(run_dir.join("metrics/request_latency").exists());
    assert!(run_dir.join("metrics/request_latency.p50").exists());
    assert!(run_dir.join("params/endpoint.type").exists());
    assert_eq!(
        std::fs::read_to_string(run_dir.join("tags/aiperf.version")).unwrap(),
        "9.9.9"
    );
    assert!(run_dir.join("tags/mlflow.runName").exists());

    let meta = std::fs::read_to_string(run_dir.join("meta.yaml")).unwrap();
    assert!(meta.contains("status: 3"), "run should be FINISHED");

    assert!(
        run_dir
            .join("artifacts/exports/profile_export.json")
            .exists()
    );

    let _ = std::fs::remove_dir_all(&temp);
}
