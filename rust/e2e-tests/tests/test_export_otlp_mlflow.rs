// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Payload-content coverage for the two exporters that write off-box.
//!
//! Both sinks were previously only reachable through a run-completes assertion,
//! which cannot fail for the reason it is named for: an exporter that silently
//! wrote nothing would still leave the run green. These tests read what was
//! actually emitted — the OTLP protobuf body as received by a collector, and the
//! MLflow FileStore tree as written to disk.
//!
//! The OTLP body is decoded with `opentelemetry-proto` rather than the sink's own
//! hand-written prost subset (`runtime/src/export/otel.rs:674`), so an encoding bug
//! shared between writer and reader cannot hide.

mod common;
use common::*;

use std::collections::BTreeMap;
use std::net::TcpListener as StdTcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::post;
use axum::{Router, body::Bytes, http::StatusCode};
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value as AnyVal};
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use prost::Message;

/// Requests captured by the stub OTLP collector, in arrival order.
type Captured = Arc<Mutex<Vec<ExportMetricsServiceRequest>>>;

/// A stub OTLP/HTTP collector that decodes and retains every posted body.
///
/// Owns its runtime so the collector outlives the `aiperf` subprocess without
/// depending on the calling test's runtime shape.
struct OtlpCollector {
    url: String,
    captured: Captured,
    _runtime: tokio::runtime::Runtime,
}

impl OtlpCollector {
    fn start() -> Self {
        let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind otlp collector");
        let port = listener.local_addr().expect("collector addr").port();
        listener
            .set_nonblocking(true)
            .expect("set collector nonblocking");

        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("build collector runtime");

        let router = Router::new()
            .route("/v1/metrics", post(receive))
            .with_state(captured.clone());
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener)
                .expect("adopt collector listener into tokio");
            let _ = axum::serve(listener, router).await;
        });

        Self {
            url: format!("http://127.0.0.1:{port}"),
            captured,
            _runtime: runtime,
        }
    }

    /// Every metric across every captured request, as `(name, data)` pairs.
    fn metrics(&self) -> Vec<(String, Data)> {
        self.captured
            .lock()
            .expect("collector mutex")
            .iter()
            .flat_map(|req| req.resource_metrics.clone())
            .flat_map(|rm| rm.scope_metrics)
            .flat_map(|sm| sm.metrics)
            .filter_map(|m| m.data.map(|d| (m.name, d)))
            .collect()
    }

    /// Resource attributes from the first captured request.
    fn resource_attributes(&self) -> BTreeMap<String, String> {
        let guard = self.captured.lock().expect("collector mutex");
        let first = guard.first().expect("no OTLP request captured");
        let resource = first
            .resource_metrics
            .first()
            .expect("no resource_metrics")
            .resource
            .clone()
            .expect("no resource on resource_metrics");
        flatten(&resource.attributes)
    }
}

/// Decode the posted protobuf and retain it; a decode failure is a test failure,
/// surfaced as a 400 so the run's stderr names the collector.
async fn receive(State(captured): State<Captured>, body: Bytes) -> StatusCode {
    match ExportMetricsServiceRequest::decode(body) {
        Ok(req) => {
            captured.lock().expect("collector mutex").push(req);
            StatusCode::OK
        }
        Err(_) => StatusCode::BAD_REQUEST,
    }
}

/// Reduce OTLP `KeyValue`s to a string map, keeping only string-valued entries.
fn flatten(attrs: &[KeyValue]) -> BTreeMap<String, String> {
    attrs
        .iter()
        .filter_map(|kv| match &kv.value {
            Some(AnyValue {
                value: Some(AnyVal::StringValue(s)),
            }) => Some((kv.key.clone(), s.clone())),
            _ => None,
        })
        .collect()
}

/// A streaming run with OTLP export enabled must post decodable GenAI histograms.
///
/// Asserts the semconv metric names, that the duration histogram's `count` matches
/// the request count actually issued, and that the resource carries the model and
/// endpoint type — none of which a no-op exporter could satisfy.
///
/// `--export-level raw` requests a per-record artifact, which disqualifies exact-fold
/// (`compose_sidecars.rs:209`) and so takes the retain path, where the post-run
/// `observe_otel_record` loop fills `report.otel_per_record` from the retained records
/// (`compose_sidecars.rs:960`). Without a per-record artifact the sink falls back to
/// aggregate-only points whose `bucket_counts` are all zero (`otel.rs:464`).
#[tokio::test]
async fn test_otlp_export_posts_genai_histograms_with_populated_buckets() {
    const REQUESTS: u32 = 6;

    let collector = OtlpCollector::start();
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 4 \
         --request-count {REQUESTS} --concurrency 2 --otel-url {} --export-level raw \
         --otel-resource-attributes deployment.environment=e2e --ui none",
        h.mock.url, collector.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);

    let metrics = collector.metrics();
    assert!(
        !metrics.is_empty(),
        "the collector received no OTLP metrics; stderr: {}",
        r.stderr
    );

    let names: Vec<&str> = metrics.iter().map(|(n, _)| n.as_str()).collect();
    for expected in [
        "gen_ai.client.operation.duration",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.token.usage",
    ] {
        assert!(
            names.contains(&expected),
            "missing semconv metric {expected}; got {names:?}"
        );
    }

    let (_, duration) = metrics
        .iter()
        .find(|(n, _)| n == "gen_ai.client.operation.duration")
        .expect("duration metric present");
    let Data::Histogram(hist) = duration else {
        panic!("gen_ai.client.operation.duration must be a Histogram, got {duration:?}");
    };
    let total: u64 = hist.data_points.iter().map(|p| p.count).sum();
    assert_eq!(
        total,
        u64::from(REQUESTS),
        "duration histogram must count every issued request"
    );
    for point in &hist.data_points {
        assert_eq!(
            point.bucket_counts.iter().sum::<u64>(),
            point.count,
            "bucket_counts must sum to the data point count (an aggregate-only \
             fallback leaves them at zero): {point:?}"
        );
        let attrs = flatten(&point.attributes);
        assert_eq!(attrs.get("gen_ai.operation.name").map(String::as_str), Some("chat"));
    }

    let resource = collector.resource_attributes();
    assert_eq!(resource.get("service.name").map(String::as_str), Some("aiperf"));
    assert_eq!(
        resource.get("aiperf.endpoint.type").map(String::as_str),
        Some("chat")
    );
    assert_eq!(
        resource.get("aiperf.model.name").map(String::as_str),
        Some(DEFAULT_MODEL)
    );
    assert_eq!(
        resource.get("deployment.environment").map(String::as_str),
        Some("e2e"),
        "--otel-resource-attributes must reach the wire; got {resource:?}"
    );
}

/// A `file://` tracking URI must produce a FileStore tree MLflow itself can read.
///
/// The layout is asserted structurally — experiment `meta.yaml` naming the
/// experiment, a run dir with `metrics`/`params`/`tags`, a FINISHED run status, the
/// run name recorded as the `mlflow.runName` tag, and the metric-file
/// `<timestamp> <value> <step>` line shape (`runtime/src/export/mlflow.rs:884`).
#[tokio::test]
async fn test_mlflow_file_store_export_writes_a_readable_run() {
    const EXPERIMENT: &str = "aiperf-e2e";
    const RUN_NAME: &str = "e2e-file-store";

    let store = tempfile::tempdir().expect("mlflow store tempdir");
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 4 \
         --request-count 4 --concurrency 1 \
         --mlflow-tracking-uri file://{} --mlflow-experiment {EXPERIMENT} \
         --mlflow-run-name {RUN_NAME} --mlflow-tag source:e2e --ui none",
        h.mock.url,
        store.path().display()
    ));
    assert!(r.success(), "run failed: {}", r.stderr);

    let experiment = single_subdir(store.path(), "experiment");
    let exp_meta = read(&experiment.join("meta.yaml"));
    assert!(
        exp_meta.contains(&format!("name: {EXPERIMENT}")),
        "experiment meta.yaml must name the experiment: {exp_meta}"
    );

    let run = single_subdir(&experiment, "run");
    for sub in ["metrics", "params", "tags", "artifacts"] {
        assert!(
            run.join(sub).is_dir(),
            "FileStore run is missing the {sub}/ directory"
        );
    }

    let run_meta = read(&run.join("meta.yaml"));
    assert!(
        run_meta.contains("status: 3"),
        "run must be marked FINISHED (status 3): {run_meta}"
    );

    assert_eq!(read(&run.join("tags/mlflow.runName")).trim(), RUN_NAME);
    assert_eq!(read(&run.join("tags/source")).trim(), "e2e");

    let metrics: Vec<PathBuf> = std::fs::read_dir(run.join("metrics"))
        .expect("read metrics dir")
        .flatten()
        .map(|e| e.path())
        .collect();
    assert!(
        !metrics.is_empty(),
        "no metric files written; an exporter that ran but emitted nothing would \
         otherwise leave this run green"
    );
    for path in &metrics {
        let line = read(path);
        let fields: Vec<&str> = line.trim().split(' ').collect();
        assert_eq!(
            fields.len(),
            3,
            "metric file {} must hold `<timestamp> <value> <step>`, got {line:?}",
            path.display()
        );
        assert!(
            fields[0].parse::<u64>().is_ok(),
            "metric timestamp must be an integer: {line:?}"
        );
        assert!(
            fields[1].parse::<f64>().is_ok_and(f64::is_finite),
            "metric value must be finite: {line:?}"
        );
    }

    // `params/` is deliberately not asserted non-empty: `MlflowExport::build`
    // hardcodes `params: BTreeMap::new()` (`runtime/src/config/model/export.rs:397`)
    // and nothing ever inserts into it, so the directory is created and left empty for
    // every run. Asserting on it would assert a gap, not a contract.
}

/// The single non-hidden child directory of `parent`, or a failure naming `label`.
fn single_subdir(parent: &Path, label: &str) -> PathBuf {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(parent)
        .unwrap_or_else(|e| panic!("read {} for {label}: {e}", parent.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    dirs.sort();
    assert_eq!(
        dirs.len(),
        1,
        "expected exactly one {label} directory under {}, got {dirs:?}",
        parent.display()
    );
    dirs.pop().expect("one directory")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}
