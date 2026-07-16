// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-cluster Kubernetes reporting for the native `aiperf controller` role.
//!
//! Native port of the Python `aiperf.kubernetes.completion_signal` +
//! `results_sidecar.write_ready_marker`: the controller pod reaches up to its
//! own `AIPerfJob` CR and PUSHES progress/snapshot into `.status` during the run
//! and a completion annotation at the end, using the in-cluster service-account
//! token + the `aiperfjobs/status` RBAC the run pod already carries. The operator
//! reacts via kopf field/annotation handlers — there is no progress service to poll.
//!
//! Off-cluster (no `AIPERF_JOB_ID`/`AIPERF_NAMESPACE`) every method is a no-op, so
//! the same `aiperf controller` binary runs locally unchanged.
//!
//! Ported from `src/aiperf/kubernetes/completion_signal.py` (patch shapes),
//! `cr_refs.py` (group/version/plural), `constants.py::Annotations`
//! (`aiperf.nvidia.com/benchmark-complete`), and `results_sidecar.py`
//! (`.aiperf_results_ready.json`). Best-effort: a transient API error logs and
//! returns without failing the run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

/// AIPerfJob CRD coordinates (mirrors `cr_refs.py`).
const AIPERF_GROUP: &str = "aiperf.nvidia.com";
const AIPERF_VERSION: &str = "v1alpha1";
const AIPERF_PLURAL: &str = "aiperfjobs";
/// Completion annotation key (mirrors `constants.py::Annotations.BENCHMARK_COMPLETE`).
const BENCHMARK_COMPLETE_ANNOTATION: &str = "aiperf.nvidia.com/benchmark-complete";
/// Results-ready marker filename (mirrors `results_sidecar.py::READY_MARKER_NAME`).
const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";

/// Standard in-cluster service-account mount (overridable in tests via
/// [`InClusterConfig::from_parts`]).
const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";

/// The owning AIPerfJob identity + in-cluster API access for this controller pod.
pub struct InClusterConfig {
    /// `KUBERNETES_SERVICE_HOST`.
    host: String,
    /// `KUBERNETES_SERVICE_PORT` (default 443).
    port: u16,
    /// Service-account bearer token.
    token: String,
    /// Cluster CA certificate (PEM).
    ca_pem: Vec<u8>,
    /// `AIPERF_NAMESPACE` — the CR's namespace.
    namespace: String,
    /// `AIPERF_JOB_ID` — the AIPerfJob CR name.
    job_id: String,
}

impl InClusterConfig {
    /// Load from the environment + service-account mount, or `None` off-cluster.
    ///
    /// Returns `None` (a no-op reporter) unless BOTH `AIPERF_JOB_ID` and
    /// `AIPERF_NAMESPACE` are set (the operator sets them via
    /// `jobset_helpers.build_cr_identity_env`) AND the in-cluster API env
    /// (`KUBERNETES_SERVICE_HOST`) + service-account token/CA are present.
    pub fn load() -> Option<Self> {
        let job_id = non_empty_env("AIPERF_JOB_ID")?;
        let namespace = non_empty_env("AIPERF_NAMESPACE")?;
        let host = non_empty_env("KUBERNETES_SERVICE_HOST")?;
        let port = std::env::var("KUBERNETES_SERVICE_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(443);
        let token = std::fs::read_to_string(SA_TOKEN_PATH)
            .ok()?
            .trim()
            .to_string();
        let ca_pem = std::fs::read(SA_CA_PATH).ok()?;
        Some(Self {
            host,
            port,
            token,
            ca_pem,
            namespace,
            job_id,
        })
    }

    /// Construct from explicit parts (tests / non-standard mounts).
    pub fn from_parts(
        host: String,
        port: u16,
        token: String,
        ca_pem: Vec<u8>,
        namespace: String,
        job_id: String,
    ) -> Self {
        Self {
            host,
            port,
            token,
            ca_pem,
            namespace,
            job_id,
        }
    }

    /// PATCH path for the CR `status` subresource.
    fn status_path(&self) -> String {
        format!(
            "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERF_PLURAL}/{}/status",
            self.namespace, self.job_id
        )
    }

    /// PATCH path for the CR object itself (metadata/annotations).
    fn object_path(&self) -> String {
        format!(
            "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERF_PLURAL}/{}",
            self.namespace, self.job_id
        )
    }
}

/// Look up an env var, treating empty/whitespace as absent.
fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// The `.status.phases.<phase>` merge-patch body (mirrors
/// `completion_signal.report_benchmark_progress`).
pub fn progress_body(
    phase: &str,
    requests_completed: u64,
    requests_total: Option<u64>,
    requests_per_second: Option<f64>,
    overall_phase: Option<&str>,
) -> Value {
    let mut phase_stats = json!({ "requestsCompleted": requests_completed });
    if let Some(total) = requests_total {
        phase_stats["requestsTotal"] = json!(total);
        if total > 0 {
            // Round to 1 decimal, matching Python's `round(x, 1)`.
            let pct = (1000.0 * requests_completed as f64 / total as f64).round() / 10.0;
            phase_stats["requestsProgressPercent"] = json!(pct);
        }
    }
    if let Some(rps) = requests_per_second {
        phase_stats["requestsPerSecond"] = json!(rps);
    }
    let mut status = json!({ "phases": { phase: phase_stats } });
    if let Some(overall) = overall_phase {
        status["phase"] = json!(overall);
    }
    json!({ "status": status })
}

/// The `.status.snapshot` merge-patch body (mirrors `report_benchmark_snapshot`).
pub fn snapshot_body(snapshot: Value) -> Value {
    json!({ "status": { "snapshot": snapshot } })
}

/// The completion-annotation merge-patch body (mirrors `signal_benchmark_complete`).
pub fn complete_body() -> Value {
    json!({ "metadata": { "annotations": { BENCHMARK_COMPLETE_ANNOTATION: "true" } } })
}

/// Path of the results-ready marker under `base_dir`.
pub fn ready_marker_path(base_dir: &Path) -> PathBuf {
    base_dir.join(READY_MARKER_NAME)
}

/// Write the results-ready marker after exports complete (mirrors
/// `results_sidecar.write_ready_marker`). The operator/sidecar refuses to serve
/// results until this file exists, so it must be written AFTER the report is
/// committed and BEFORE the completion annotation.
pub fn write_ready_marker(base_dir: &Path, was_cancelled: bool) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(base_dir)?;
    let marker = ready_marker_path(base_dir);
    let body = json!({ "ready": true, "was_cancelled": was_cancelled });
    std::fs::write(&marker, serde_json::to_vec(&body).expect("marker json"))?;
    Ok(marker)
}

/// A best-effort in-cluster CR reporter. Off-cluster ([`InClusterConfig::load`]
/// returned `None`) every method is a silent no-op.
pub struct CrReporter {
    config: Option<InClusterConfig>,
}

impl CrReporter {
    /// Build from the ambient environment; no-op off-cluster.
    pub fn from_env() -> Self {
        Self {
            config: InClusterConfig::load(),
        }
    }

    /// Whether this reporter will actually talk to a cluster.
    pub fn active(&self) -> bool {
        self.config.is_some()
    }

    /// Merge-patch the CR `.status` (progress or snapshot). No-op off-cluster.
    pub fn patch_status(&self, body: &Value) {
        if let Some(cfg) = &self.config {
            let path = cfg.status_path();
            self.send(cfg, &path, body);
        }
    }

    /// Merge-patch the CR object (completion annotation). No-op off-cluster.
    pub fn patch_object(&self, body: &Value) {
        if let Some(cfg) = &self.config {
            let path = cfg.object_path();
            self.send(cfg, &path, body);
        }
    }

    /// Signal completion: mirror the Python ordering write-marker -> annotation
    /// (the marker is written by the caller after export; here we set the
    /// annotation the operator watches). No-op off-cluster.
    pub fn signal_complete(&self) {
        self.patch_object(&complete_body());
    }

    /// Send one merge-patch. Best-effort: logs and swallows transport/API errors
    /// so a reporting hiccup never fails the benchmark.
    fn send(&self, cfg: &InClusterConfig, path: &str, body: &Value) {
        match send_merge_patch(cfg, path, body) {
            Ok(status) if (200..300).contains(&status) => {
                tracing::debug!(path, status, "patched AIPerfJob CR");
            }
            Ok(status) => {
                tracing::warn!(path, status, "AIPerfJob CR patch returned non-2xx");
            }
            Err(error) => {
                tracing::warn!(
                    path,
                    error = format!("{error:#}"),
                    "AIPerfJob CR patch failed"
                );
            }
        }
    }
}

/// Perform one `PATCH <path>` with `Content-Type: application/merge-patch+json`
/// and the service-account bearer token, over TLS trusting the cluster CA.
/// Runs a small current-thread runtime for the single request.
fn send_merge_patch(cfg: &InClusterConfig, path: &str, body: &Value) -> anyhow::Result<u16> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(send_merge_patch_async(cfg, path, body))
}

async fn send_merge_patch_async(
    cfg: &InClusterConfig,
    path: &str,
    body: &Value,
) -> anyhow::Result<u16> {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::Request;

    let payload = serde_json::to_vec(body)?;

    let mut roots = rustls::RootCertStore::empty();
    let certs = rustls_pemfile::certs(&mut cfg.ca_pem.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("failed to parse cluster CA PEM: {e}"))?;
    for cert in certs {
        roots
            .add(cert)
            .map_err(|e| anyhow::anyhow!("failed to add cluster CA: {e}"))?;
    }
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(tls_config));

    let tcp = tokio::net::TcpStream::connect((cfg.host.as_str(), cfg.port)).await?;
    let dnsname = rustls::pki_types::ServerName::try_from(cfg.host.clone())
        .map_err(|e| anyhow::anyhow!("invalid API server name {}: {e}", cfg.host))?;
    let tls = connector.connect(dnsname, tcp).await?;
    let (mut sender, conn) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls)).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let req = Request::builder()
        .method("PATCH")
        .uri(path)
        .header("host", format!("{}:{}", cfg.host, cfg.port))
        .header("authorization", format!("Bearer {}", cfg.token))
        .header("content-type", "application/merge-patch+json")
        .header("accept", "application/json")
        .body(Full::<Bytes>::new(Bytes::from(payload)))?;

    let resp = sender.send_request(req).await?;
    let status = resp.status().as_u16();
    // Drain the body so the connection closes cleanly.
    let _ = resp.into_body().collect().await;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InClusterConfig {
        InClusterConfig::from_parts(
            "10.0.0.1".to_string(),
            6443,
            "tok".to_string(),
            Vec::new(),
            "bench-ns".to_string(),
            "job-42".to_string(),
        )
    }

    #[test]
    fn status_and_object_paths() {
        let c = cfg();
        assert_eq!(
            c.status_path(),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench-ns/aiperfjobs/job-42/status"
        );
        assert_eq!(
            c.object_path(),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench-ns/aiperfjobs/job-42"
        );
    }

    #[test]
    fn progress_body_shape_and_percent() {
        let b = progress_body("profiling", 25, Some(100), Some(12.5), Some("Profiling"));
        let p = &b["status"]["phases"]["profiling"];
        assert_eq!(p["requestsCompleted"], 25);
        assert_eq!(p["requestsTotal"], 100);
        assert_eq!(p["requestsProgressPercent"], 25.0);
        assert_eq!(p["requestsPerSecond"], 12.5);
        assert_eq!(b["status"]["phase"], "Profiling");
    }

    #[test]
    fn progress_body_rounds_to_one_decimal() {
        // 1/3 -> 33.3 (round(33.333..., 1)).
        let b = progress_body("profiling", 1, Some(3), None, None);
        assert_eq!(
            b["status"]["phases"]["profiling"]["requestsProgressPercent"],
            33.3
        );
        // No overall phase key when None.
        assert!(b["status"].get("phase").is_none());
    }

    #[test]
    fn progress_body_omits_percent_without_total() {
        let b = progress_body("warmup", 5, None, None, None);
        let p = &b["status"]["phases"]["warmup"];
        assert_eq!(p["requestsCompleted"], 5);
        assert!(p.get("requestsTotal").is_none());
        assert!(p.get("requestsProgressPercent").is_none());
    }

    #[test]
    fn snapshot_and_complete_bodies() {
        let s = snapshot_body(json!({"ttft": {"avg": 20.0}}));
        assert_eq!(s["status"]["snapshot"]["ttft"]["avg"], 20.0);
        let c = complete_body();
        assert_eq!(
            c["metadata"]["annotations"]["aiperf.nvidia.com/benchmark-complete"],
            "true"
        );
    }

    #[test]
    fn ready_marker_writes_expected_json() {
        let dir = std::env::temp_dir().join(format!("aiperf-k8s-marker-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let marker = write_ready_marker(&dir, false).unwrap();
        assert_eq!(marker.file_name().unwrap(), ".aiperf_results_ready.json");
        let v: Value = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        assert_eq!(v["ready"], true);
        assert_eq!(v["was_cancelled"], false);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reporter_off_cluster_is_noop() {
        // With no config, patch/complete are silent no-ops (no panic, no network).
        let reporter = CrReporter { config: None };
        assert!(!reporter.active());
        reporter.patch_status(&progress_body("profiling", 1, Some(2), None, None));
        reporter.signal_complete();
    }
}
