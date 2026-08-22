// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-cluster AIPerfJob reporting and private result-readiness compatibility.
//!
//! Kubernetes authentication and TLS live in [`crate::kube`]. Reporting is a
//! no-op off-cluster and API failures never fail a benchmark.

use std::path::{Path, PathBuf};
use std::io::Write;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::kube::auth::in_cluster_credentials;
use crate::kube::client::{
    AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, BENCHMARK_COMPLETE_ANNOTATION, KubeClient,
};

const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";
const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";

/// The owning AIPerfJob identity plus shared Kubernetes client.
pub struct InClusterConfig {
    client: KubeClient,
    namespace: String,
    job_id: String,
}

impl InClusterConfig {
    /// Load service-account credentials from the ambient pod environment.
    pub fn load() -> Option<Self> {
        let job_id = non_empty_env("AIPERF_JOB_ID")?;
        let namespace = non_empty_env("AIPERF_NAMESPACE")?;
        let host = non_empty_env("KUBERNETES_SERVICE_HOST")?;
        let port = std::env::var("KUBERNETES_SERVICE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(443);
        let credentials = in_cluster_credentials(host, port, Path::new(SA_TOKEN_PATH), Path::new(SA_CA_PATH)).ok()?;
        let client = KubeClient::from_credentials(credentials).ok()?;
        Some(Self { client, namespace, job_id })
    }

    #[cfg(test)]
    fn from_parts(
        host: String,
        port: u16,
        token: String,
        ca_pem: Vec<u8>,
        namespace: String,
        job_id: String,
    ) -> Self {
        let credentials = crate::kube::auth::KubeCredentials {
            server_name: host.clone(),
            host,
            port,
            token: Some(token),
            client_certificate_pem: None,
            client_key_pem: None,
            ca_pem: Some(ca_pem),
            insecure_skip_tls_verify: false,
        };
        let client = KubeClient::from_credentials(credentials).expect("test credentials are valid");
        Self { client, namespace, job_id }
    }

    fn status_path(&self) -> String {
        format!("/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERF_PLURAL}/{}/status", self.namespace, self.job_id)
    }

    fn object_path(&self) -> String {
        format!("/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERF_PLURAL}/{}", self.namespace, self.job_id)
    }
}

fn non_empty_env(key: &str) -> Option<String> { std::env::var(key).ok().filter(|value| !value.trim().is_empty()) }

/// Build a `.status.phases.<phase>` merge patch.
pub fn progress_body(phase: &str, requests_completed: u64, requests_total: Option<u64>, requests_per_second: Option<f64>, overall_phase: Option<&str>) -> Value {
    let mut phase_stats = json!({ "requestsCompleted": requests_completed });
    if let Some(total) = requests_total {
        phase_stats["requestsTotal"] = json!(total);
        if total > 0 { phase_stats["requestsProgressPercent"] = json!((1000.0 * requests_completed as f64 / total as f64).round() / 10.0); }
    }
    if let Some(rps) = requests_per_second { phase_stats["requestsPerSecond"] = json!(rps); }
    let mut status = json!({ "phases": { phase: phase_stats } });
    if let Some(overall) = overall_phase { status["phase"] = json!(overall); }
    json!({ "status": status })
}

/// Build a `.status.snapshot` merge patch.
pub fn snapshot_body(snapshot: Value) -> Value { json!({ "status": { "snapshot": snapshot } }) }

/// Build the completion-annotation merge patch.
pub fn complete_body() -> Value { json!({ "metadata": { "annotations": { BENCHMARK_COMPLETE_ANNOTATION: "true" } } }) }

/// Path of the private compatibility marker under `base_dir`.
pub fn ready_marker_path(base_dir: &Path) -> PathBuf { base_dir.join(READY_MARKER_NAME) }

/// Atomically publish the public native-k8s/v1 results manifest, then compatibility marker.
pub fn publish_results(base_dir: &Path, run_id: &str, was_cancelled: bool) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(base_dir)?;
    let artifacts = collect_artifacts(base_dir)?;
    let manifest = json!({
        "contractVersion": "native-k8s/v1",
        "runId": run_id,
        "ready": true,
        "wasCancelled": was_cancelled,
        "artifactRoot": base_dir,
        "artifacts": artifacts,
    });
    let manifest_path = base_dir.join("results-manifest.json");
    write_atomic_json(&manifest_path, &manifest)?;
    write_ready_marker(base_dir, was_cancelled)?;
    Ok(manifest_path)
}

fn collect_artifacts(base_dir: &Path) -> std::io::Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut stack = vec![base_dir.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() { stack.push(path); continue; }
            if !kind.is_file() { continue; }
            let relative = path.strip_prefix(base_dir).map_err(std::io::Error::other)?;
            let name = relative.to_string_lossy().replace('\\', "/");
            if name == READY_MARKER_NAME || name == "results-manifest.json" { continue; }
            let bytes = std::fs::read(&path)?;
            out.push(json!({"path": name, "sha256": format!("{:x}", Sha256::digest(&bytes)), "bytes": bytes.len(), "contentType": content_type(&path)}));
        }
    }
    out.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    Ok(out)
}

fn write_atomic_json(path: &Path, value: &Value) -> std::io::Result<()> {
    let temporary = path.with_extension("tmp");
    let bytes = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&temporary, path)?;
    std::fs::File::open(path.parent().unwrap_or(Path::new(".")))?.sync_all()
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("json") => "application/json",
        Some("jsonl") => "application/x-ndjson",
        Some("csv") => "text/csv",
        Some("parquet") => "application/vnd.apache.parquet",
        _ => "application/octet-stream",
    }
}

/// Write the legacy marker only after the public manifest has been fsynced.
pub fn write_ready_marker(base_dir: &Path, was_cancelled: bool) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(base_dir)?;
    let marker = ready_marker_path(base_dir);
    let body = json!({ "ready": true, "was_cancelled": was_cancelled });
    write_atomic_json(&marker, &body)?;
    Ok(marker)
}

/// A best-effort in-cluster CR reporter.
pub struct CrReporter { config: Option<InClusterConfig> }

impl CrReporter {
    /// Build from ambient service-account credentials; inactive off-cluster.
    pub fn from_env() -> Self { Self { config: InClusterConfig::load() } }
    /// Whether this reporter will talk to the Kubernetes API.
    pub fn active(&self) -> bool { self.config.is_some() }
    /// Merge-patch the CR status. This is deliberately best effort.
    pub fn patch_status(&self, body: &Value) { if let Some(config) = &self.config { self.send(config, &config.status_path(), body); } }
    /// Merge-patch the CR object. This is deliberately best effort.
    pub fn patch_object(&self, body: &Value) { if let Some(config) = &self.config { self.send(config, &config.object_path(), body); } }
    /// Mark benchmark completion after the caller publishes final results.
    pub fn signal_complete(&self) { self.patch_object(&complete_body()); }

    fn send(&self, config: &InClusterConfig, path: &str, body: &Value) {
        match config.client.merge_patch(path, body) {
            Ok(status) if (200..300).contains(&status) => tracing::debug!(path, status, "patched AIPerfJob CR"),
            Ok(status) => tracing::warn!(path, status, "AIPerfJob CR patch returned non-2xx"),
            Err(error) => tracing::warn!(path, error = %error, "AIPerfJob CR patch failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::kube::auth::KubeCredentials;
    use crate::kube::client::{KubeRequest, KubeTransport, KubeWatch};

    #[test]
    fn status_and_object_paths() {
        let config = InClusterConfig::from_parts("10.0.0.1".to_string(), 6443, "tok".to_string(), Vec::new(), "bench-ns".to_string(), "job-42".to_string());
        assert_eq!(config.status_path(), "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench-ns/aiperfjobs/job-42/status");
        assert_eq!(config.object_path(), "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench-ns/aiperfjobs/job-42");
    }

    #[test]
    fn progress_body_shape_and_percent() {
        let body = progress_body("profiling", 25, Some(100), Some(12.5), Some("Profiling"));
        assert_eq!(body["status"]["phases"]["profiling"]["requestsProgressPercent"], 25.0);
        assert_eq!(body["status"]["phase"], "Profiling");
    }

    #[test]
    fn ready_marker_writes_expected_json() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let marker = write_ready_marker(dir.path(), false).expect("marker write");
        let value: Value = serde_json::from_slice(&std::fs::read(marker).expect("marker read")).expect("marker JSON");
        assert_eq!(value["ready"], true);
    }

    #[test]
    fn reporter_constructs_status_and_completion_requests() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let credentials = KubeCredentials { host: "api".to_string(), port: 443, server_name: "api".to_string(), token: Some("token".to_string()), client_certificate_pem: None, client_key_pem: None, ca_pem: None, insecure_skip_tls_verify: true };
        let client = KubeClient::with_transport(credentials, Arc::new(RecordingTransport(requests.clone())));
        let reporter = CrReporter { config: Some(InClusterConfig { client, namespace: "bench".to_string(), job_id: "job".to_string() }) };
        reporter.patch_status(&progress_body("profiling", 2, Some(4), None, None));
        reporter.signal_complete();
        let requests = requests.lock().expect("recording lock");
        assert_eq!(requests[0].path, "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job/status");
        assert_eq!(requests[1].path, "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job");
        assert_eq!(requests[1].body, serde_json::to_vec(&complete_body()).expect("completion JSON"));
    }

    struct RecordingTransport(Arc<Mutex<Vec<KubeRequest>>>);
    impl KubeTransport for RecordingTransport {
        fn send(&self, _credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, crate::kube::error::KubeError> { self.0.lock().expect("recording lock").push(request); Ok(200) }
        fn watch(&self, _credentials: &KubeCredentials, _request: KubeRequest) -> Result<KubeWatch, crate::kube::error::KubeError> { Err(crate::kube::error::KubeError::Transport("watch is unavailable in reporter test".to_string())) }
    }

    #[test]
    fn reporter_off_cluster_is_noop() {
        let reporter = CrReporter { config: None };
        assert!(!reporter.active());
        reporter.signal_complete();
    }
}
