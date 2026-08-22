// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-cluster AIPerfJob reporting and private result-readiness compatibility.
//!
//! Kubernetes authentication and TLS live in [`crate::kube`]. Reporting is a
//! no-op off-cluster and API failures never fail a benchmark.

#[cfg(unix)]
use std::ffi::{CStr, CString, OsStr, OsString};
#[cfg(unix)]
use std::fs::File;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::kube::auth::in_cluster_credentials;
use crate::kube::client::{AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, KubeClient};
use crate::kube::results::MAX_ARTIFACT_BYTES;

const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";
const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";
const ARTIFACT_HASH_BUFFER_BYTES: usize = 64 * 1024;

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
    collect_artifacts_with_directory_opened(base_dir, |_| {})
}

#[cfg(unix)]
fn collect_artifacts_with_directory_opened<F>(
    base_dir: &Path,
    mut directory_opened: F,
) -> std::io::Result<Vec<Value>>
where
    F: FnMut(&Path),
{
    let mut out = Vec::new();
    let root = open_artifact_directory_path(base_dir)?;
    collect_artifacts_in_directory(&root, Path::new(""), &mut out, &mut directory_opened)?;
    out.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    Ok(out)
}

#[cfg(unix)]
fn collect_artifacts_in_directory<F>(
    directory: &File,
    relative_directory: &Path,
    artifacts: &mut Vec<Value>,
    directory_opened: &mut F,
) -> std::io::Result<()>
where
    F: FnMut(&Path),
{
    for name in read_directory_names(directory)? {
        let relative = relative_directory.join(&name);
        match open_artifact_directory_at(directory, &name) {
            Ok(child) => {
                directory_opened(&relative);
                collect_artifacts_in_directory(&child, &relative, artifacts, directory_opened)?;
            }
            Err(error) if error.raw_os_error() == Some(libc::ENOTDIR) => {
                let mut file = open_regular_artifact_at(directory, &name, &relative)?;
                let relative_name = relative.to_string_lossy().replace('\\', "/");
                if relative_name == READY_MARKER_NAME || relative_name == "results-manifest.json" {
                    continue;
                }
                if file.metadata()?.len() > MAX_ARTIFACT_BYTES {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "result artifact {} exceeds {MAX_ARTIFACT_BYTES} bytes",
                            relative.display()
                        ),
                    ));
                }
                let (sha256, bytes) = hash_artifact_reader(&mut file, MAX_ARTIFACT_BYTES)?;
                artifacts.push(json!({"path": relative_name, "sha256": sha256, "bytes": bytes, "contentType": content_type(&relative)}));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn open_artifact_directory_path(path: &Path) -> std::io::Result<File> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut directory = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open("/")?;
    for component in absolute.components() {
        match component {
            std::path::Component::RootDir => continue,
            std::path::Component::Normal(name) => {
                directory = open_artifact_directory_at(&directory, name)?;
            }
            std::path::Component::CurDir => continue,
            std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "artifact root is not canonical",
                ));
            }
        }
    }
    Ok(directory)
}

#[cfg(unix)]
fn component_name(name: &OsStr) -> std::io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifact path component contains NUL",
        )
    })
}

#[cfg(unix)]
fn open_artifact_directory_at(parent: &File, name: &OsStr) -> std::io::Result<File> {
    let name = component_name(name)?;
    // SAFETY: `parent` is an open directory descriptor and `name` is one
    // NUL-terminated directory-entry component. O_NOFOLLOW prevents a rename
    // race from redirecting this traversal through a symlink.
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a successful `openat` creates one owned descriptor.
    let directory = unsafe { File::from_raw_fd(descriptor) };
    if directory.metadata()?.is_dir() {
        Ok(directory)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifact path component is not a directory",
        ))
    }
}

#[cfg(unix)]
fn open_regular_artifact_at(parent: &File, name: &OsStr, relative: &Path) -> std::io::Result<File> {
    let name = component_name(name)?;
    // SAFETY: `parent` is an open directory descriptor and `name` is one
    // NUL-terminated directory-entry component. O_NOFOLLOW confines the leaf
    // to the retained parent directory.
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if descriptor < 0 {
        let error = std::io::Error::last_os_error();
        return Err(match error.raw_os_error() {
            Some(libc::ELOOP | libc::ENOENT | libc::ENOTDIR | libc::ESTALE) => std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "result artifact {} changed or became a symlink",
                    relative.display()
                ),
            ),
            _ => error,
        });
    }
    // SAFETY: a successful `openat` creates one owned descriptor.
    let file = unsafe { File::from_raw_fd(descriptor) };
    if file.metadata()?.is_file() {
        Ok(file)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "result artifact {} is not a regular file",
                relative.display()
            ),
        ))
    }
}

#[cfg(unix)]
struct DirectoryStream(*mut libc::DIR);

#[cfg(unix)]
impl DirectoryStream {
    fn open(directory: &File) -> std::io::Result<Self> {
        let descriptor = directory.try_clone()?.into_raw_fd();
        // SAFETY: `descriptor` is an owned directory descriptor. On success,
        // `fdopendir` takes ownership and this stream closes it on drop.
        let stream = unsafe { libc::fdopendir(descriptor) };
        if stream.is_null() {
            let error = std::io::Error::last_os_error();
            // SAFETY: on `fdopendir` failure the descriptor remains owned here.
            drop(unsafe { File::from_raw_fd(descriptor) });
            return Err(error);
        }
        Ok(Self(stream))
    }

    fn read_names(&mut self) -> std::io::Result<Vec<OsString>> {
        let mut names = Vec::new();
        loop {
            set_errno(0);
            // SAFETY: this stream owns a live `DIR*`; each entry is consumed
            // before the next `readdir` call.
            let entry = unsafe { libc::readdir(self.0) };
            if entry.is_null() {
                let error = current_errno();
                if error == 0 {
                    break;
                }
                return Err(std::io::Error::from_raw_os_error(error));
            }
            // SAFETY: POSIX provides a NUL-terminated `d_name` while this
            // directory entry remains current.
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            names.push(OsString::from_vec(name.to_vec()));
        }
        Ok(names)
    }
}

#[cfg(unix)]
impl Drop for DirectoryStream {
    fn drop(&mut self) {
        // SAFETY: this stream uniquely owns the descriptor held by `DIR*`.
        let _ = unsafe { libc::closedir(self.0) };
    }
}

#[cfg(unix)]
fn read_directory_names(directory: &File) -> std::io::Result<Vec<OsString>> {
    DirectoryStream::open(directory)?.read_names()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc exposes this thread's writable errno location on Linux.
    unsafe { libc::__errno_location() }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd"
))]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc exposes this thread's writable errno location on these targets.
    unsafe { libc::__error() }
}

#[cfg(unix)]
fn set_errno(value: libc::c_int) {
    // SAFETY: `errno_pointer` returns this thread's writable errno cell.
    unsafe { *errno_pointer() = value };
}

#[cfg(unix)]
fn current_errno() -> libc::c_int {
    // SAFETY: `errno_pointer` returns this thread's readable errno cell.
    unsafe { *errno_pointer() }
}

#[cfg(not(unix))]
fn collect_artifacts_with_directory_opened<F>(
    _base_dir: &Path,
    _directory_opened: F,
) -> std::io::Result<Vec<Value>>
where
    F: FnMut(&Path),
{
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "result collection requires POSIX no-follow descriptors",
    ))
}

fn hash_artifact_reader<R: Read>(reader: &mut R, maximum: u64) -> std::io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; ARTIFACT_HASH_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "result artifact is too large",
            )
        })?;
        if bytes > maximum {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("result artifact exceeds {maximum} bytes"),
            ));
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), bytes))
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

    #[cfg(unix)]
    #[test]
    fn publish_results_refuses_a_symlinked_artifact() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let outside = tempfile::NamedTempFile::new().expect("outside artifact");
        symlink(outside.path(), directory.path().join("profile.json")).expect("artifact symlink");

        let error = publish_results(directory.path(), "run-1", false)
            .expect_err("unsafe artifacts must prevent manifest publication");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!directory.path().join("results-manifest.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn nested_artifact_directory_replacement_never_collects_through_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let nested = directory.path().join("nested");
        std::fs::create_dir(&nested).expect("nested directory");
        let inside = b"inside";
        std::fs::write(nested.join("inside.json"), inside).expect("inside artifact");
        let outside = tempfile::tempdir().expect("outside directory");
        std::fs::write(outside.path().join("external.json"), b"external")
            .expect("external artifact");

        let artifacts = collect_artifacts_with_directory_opened(directory.path(), |relative| {
            if relative == Path::new("nested") {
                std::fs::rename(&nested, directory.path().join("nested-original"))
                    .expect("replace nested directory");
                symlink(outside.path(), &nested).expect("nested symlink");
            }
        })
        .expect("retained nested descriptor remains confined");

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0]["path"], "nested/inside.json");
        assert_eq!(
            artifacts[0]["sha256"],
            format!("{:x}", Sha256::digest(inside))
        );
        assert!(
            artifacts
                .iter()
                .all(|artifact| artifact["path"] != "nested/external.json")
        );
    }

    #[test]
    fn publish_results_refuses_an_artifact_over_the_upload_limit() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let artifact =
            std::fs::File::create(directory.path().join("large.bin")).expect("artifact creation");
        artifact
            .set_len(512 * 1024 * 1024 + 1)
            .expect("sparse oversized artifact");

        let error = publish_results(directory.path(), "run-1", false)
            .expect_err("oversized artifacts must prevent manifest publication");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!directory.path().join("results-manifest.json").exists());
    }

    #[test]
    fn artifact_hashing_never_requests_more_than_64_kib() {
        let payload = vec![b'x'; 64 * 1024 + 1];
        let mut reader = RequestedBufferReader {
            source: std::io::Cursor::new(payload.clone()),
            largest_request: 0,
        };

        let (digest, bytes) =
            hash_artifact_reader(&mut reader, payload.len() as u64).expect("bounded artifact hash");

        assert_eq!(digest, format!("{:x}", Sha256::digest(&payload)));
        assert_eq!(bytes, payload.len() as u64);
        assert!(reader.largest_request <= 64 * 1024);
    }

    struct RequestedBufferReader {
        source: std::io::Cursor<Vec<u8>>,
        largest_request: usize,
    }

    impl std::io::Read for RequestedBufferReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.largest_request = self.largest_request.max(buffer.len());
            if buffer.len() > 64 * 1024 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "hash reader requested more than 64 KiB",
                ));
            }
            self.source.read(buffer)
        }
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
