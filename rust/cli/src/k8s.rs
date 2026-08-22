// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-cluster AIPerfJob reporting and private result-readiness compatibility.
//!
//! Kubernetes authentication and TLS live in [`crate::kube`]. Reporting is a
//! no-op off-cluster and API failures never fail a benchmark.

use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::kube::auth::in_cluster_credentials;
use crate::kube::client::{AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, KubeClient};

const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";
const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";

/// The owning AIPerfJob identity plus rotating Kubernetes credentials.
pub struct InClusterConfig {
    host: String,
    port: u16,
    token_path: PathBuf,
    ca_path: PathBuf,
    namespace: String,
    job_id: String,
    object_uid: String,
    #[cfg(test)]
    transport: Option<Arc<dyn crate::kube::client::KubeTransport>>,
}

impl InClusterConfig {
    /// Load service-account credentials from the ambient pod environment.
    pub fn load() -> Option<Self> {
        let job_id = non_empty_env("AIPERF_JOB_ID")?;
        let object_uid = non_empty_env("AIPERF_JOB_UID")?;
        let namespace = non_empty_env("AIPERF_NAMESPACE")?;
        let host = non_empty_env("KUBERNETES_SERVICE_HOST")?;
        let port = std::env::var("KUBERNETES_SERVICE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(443);
        let config = Self {
            host,
            port,
            token_path: PathBuf::from(SA_TOKEN_PATH),
            ca_path: PathBuf::from(SA_CA_PATH),
            namespace,
            job_id,
            object_uid,
            #[cfg(test)]
            transport: None,
        };
        config.client().ok()?;
        Some(config)
    }

    fn client(&self) -> Result<KubeClient, crate::kube::error::KubeError> {
        let credentials = in_cluster_credentials(
            self.host.clone(),
            self.port,
            &self.token_path,
            &self.ca_path,
        )?;
        #[cfg(test)]
        if let Some(transport) = &self.transport {
            return Ok(KubeClient::with_transport(credentials, transport.clone()));
        }
        KubeClient::from_credentials(credentials)
    }

    fn status_path(&self) -> String {
        format!(
            "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERF_PLURAL}/{}/status",
            self.namespace, self.job_id
        )
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Build a `.status.phases.<phase>` merge patch.
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
            phase_stats["requestsProgressPercent"] =
                json!((1000.0 * requests_completed as f64 / total as f64).round() / 10.0);
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

/// Build a `.status.snapshot` merge patch.
pub fn snapshot_body(snapshot: Value) -> Value {
    json!({ "status": { "snapshot": snapshot } })
}

/// Build the terminal status merge patch.
pub fn complete_body() -> Value {
    json!({ "status": { "phase": "PublishingResults" } })
}

/// Path of the private compatibility marker under `base_dir`.
pub fn ready_marker_path(base_dir: &Path) -> PathBuf {
    base_dir.join(READY_MARKER_NAME)
}

/// Atomically publish the public native-k8s/v1 results manifest, then compatibility marker.
pub fn publish_results(
    base_dir: &Path,
    run_id: &str,
    was_cancelled: bool,
) -> std::io::Result<PathBuf> {
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
            if kind.is_dir() {
                stack.push(path);
                continue;
            }
            if !kind.is_file() {
                continue;
            }
            let relative = path.strip_prefix(base_dir).map_err(std::io::Error::other)?;
            let name = relative.to_string_lossy().replace('\\', "/");
            if name == READY_MARKER_NAME || name == "results-manifest.json" {
                continue;
            }
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
pub struct CrReporter {
    config: Option<InClusterConfig>,
}

impl CrReporter {
    /// Build from ambient service-account credentials; inactive off-cluster.
    pub fn from_env() -> Self {
        Self {
            config: InClusterConfig::load(),
        }
    }
    /// Whether this reporter will talk to the Kubernetes API.
    pub fn active(&self) -> bool {
        self.config.is_some()
    }
    /// Merge-patch the CR status. This is deliberately best effort.
    pub fn patch_status(&self, body: &Value) {
        if let Some(config) = &self.config {
            self.send(config, &config.status_path(), body);
        }
    }
    /// Mark benchmark completion after the caller publishes final results.
    pub fn signal_complete(&self) {
        self.patch_status(&complete_body());
    }

    fn send(&self, config: &InClusterConfig, path: &str, body: &Value) {
        let mut bound_body = body.clone();
        bound_body["metadata"] = json!({"uid": config.object_uid});
        let response = config
            .client()
            .and_then(|client| client.merge_patch(path, &bound_body));
        match response {
            Ok(status) if (200..300).contains(&status) => {
                tracing::debug!(path, status, "patched AIPerfJob CR")
            }
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
    fn status_path_is_the_only_workload_reporting_target() {
        let config = InClusterConfig {
            host: "10.0.0.1".to_string(),
            port: 6443,
            token_path: PathBuf::new(),
            ca_path: PathBuf::new(),
            namespace: "bench-ns".to_string(),
            job_id: "job-42".to_string(),
            object_uid: "uid-42".to_string(),
            transport: None,
        };
        assert_eq!(
            config.status_path(),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench-ns/aiperfjobs/job-42/status"
        );
    }

    #[test]
    fn progress_body_shape_and_percent() {
        let body = progress_body("profiling", 25, Some(100), Some(12.5), Some("Profiling"));
        assert_eq!(
            body["status"]["phases"]["profiling"]["requestsProgressPercent"],
            25.0
        );
        assert_eq!(body["status"]["phase"], "Profiling");
    }

    #[test]
    fn ready_marker_writes_expected_json() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let marker = write_ready_marker(dir.path(), false).expect("marker write");
        let value: Value = serde_json::from_slice(&std::fs::read(marker).expect("marker read"))
            .expect("marker JSON");
        assert_eq!(value["ready"], true);
    }

    #[test]
    fn reporter_constructs_status_and_completion_requests() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let directory = tempfile::tempdir().expect("temporary directory");
        let token_path = directory.path().join("token");
        let ca_path = directory.path().join("ca.crt");
        std::fs::write(&token_path, "token").expect("test token");
        std::fs::write(&ca_path, []).expect("test CA");
        let reporter = CrReporter {
            config: Some(InClusterConfig {
                host: "api".to_string(),
                port: 443,
                token_path,
                ca_path,
                namespace: "bench".to_string(),
                job_id: "job".to_string(),
                object_uid: "uid-1".to_string(),
                transport: Some(Arc::new(RecordingTransport(requests.clone()))),
            }),
        };
        reporter.patch_status(&progress_body("profiling", 2, Some(4), None, None));
        reporter.signal_complete();
        let requests = requests.lock().expect("recording lock");
        assert_eq!(
            requests[0].path,
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job/status"
        );
        assert_eq!(
            requests[1].path,
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job/status"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&requests[0].body).expect("progress JSON"),
            json!({
                "metadata": {"uid": "uid-1"},
                "status": {"phases": {"profiling": {"requestsCompleted": 2, "requestsTotal": 4, "requestsProgressPercent": 50.0}}}
            })
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&requests[1].body).expect("completion JSON"),
            json!({
                "metadata": {"uid": "uid-1"},
                "status": {"phase": "PublishingResults"}
            })
        );
    }

    #[test]
    fn reporter_reloads_a_rotated_projected_token_for_each_patch() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let token_path = directory.path().join("token");
        let ca_path = directory.path().join("ca.crt");
        std::fs::write(&token_path, "token-1\n").expect("initial token");
        std::fs::write(&ca_path, []).expect("test CA");
        let tokens = Arc::new(Mutex::new(Vec::new()));
        let reporter = CrReporter {
            config: Some(InClusterConfig {
                host: "api".to_string(),
                port: 443,
                token_path,
                ca_path,
                namespace: "bench".to_string(),
                job_id: "job".to_string(),
                object_uid: "uid-1".to_string(),
                transport: Some(Arc::new(TokenRecordingTransport(tokens.clone()))),
            }),
        };

        reporter.patch_status(&snapshot_body(json!({"step": 1})));
        std::fs::write(
            &reporter
                .config
                .as_ref()
                .expect("active reporter")
                .token_path,
            "token-2\n",
        )
        .expect("rotated token");
        reporter.patch_status(&snapshot_body(json!({"step": 2})));

        assert_eq!(
            *tokens.lock().expect("token recording lock"),
            vec!["token-1".to_string(), "token-2".to_string()]
        );
    }

    struct RecordingTransport(Arc<Mutex<Vec<KubeRequest>>>);
    impl KubeTransport for RecordingTransport {
        fn send(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<crate::kube::client::KubeResponse, crate::kube::error::KubeError> {
            self.0.lock().expect("recording lock").push(request);
            Ok(crate::kube::client::KubeResponse {
                status: 200,
                body: Vec::new(),
            })
        }
        fn watch(
            &self,
            _credentials: &KubeCredentials,
            _request: KubeRequest,
        ) -> Result<KubeWatch, crate::kube::error::KubeError> {
            Err(crate::kube::error::KubeError::Transport(
                "watch is unavailable in reporter test".to_string(),
            ))
        }
    }

    struct TokenRecordingTransport(Arc<Mutex<Vec<String>>>);
    impl KubeTransport for TokenRecordingTransport {
        fn send(
            &self,
            credentials: &KubeCredentials,
            _request: KubeRequest,
        ) -> Result<crate::kube::client::KubeResponse, crate::kube::error::KubeError> {
            self.0
                .lock()
                .expect("token recording lock")
                .push(credentials.token.clone().expect("bearer token"));
            Ok(crate::kube::client::KubeResponse {
                status: 200,
                body: Vec::new(),
            })
        }
        fn watch(
            &self,
            _credentials: &KubeCredentials,
            _request: KubeRequest,
        ) -> Result<KubeWatch, crate::kube::error::KubeError> {
            Err(crate::kube::error::KubeError::Transport(
                "watch is unavailable in reporter test".to_string(),
            ))
        }
    }

    #[test]
    fn reporter_off_cluster_is_noop() {
        let reporter = CrReporter { config: None };
        assert!(!reporter.active());
        reporter.signal_complete();
    }
}
