// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf results-sidecar`, the controller pod's durable upload companion.
//!
//! The server exposes only health. Results are validated beneath a retained
//! no-follow root and uploaded to the durable operator API.
//!
//! HTTP contract:
//! - `GET /healthz` -> `{"status":"ok"}`
//!
//! Env: `AIPERF_RESULTS_DIR` (default `/results`) and
//! `AIPERF_RESULTS_SIDECAR_PORT` (default `9091`). Kubernetes also supplies
//! `AIPERF_RESULTS_UPLOAD_URL`, `AIPERF_NAMESPACE`, `AIPERF_JOB_ID`,
//! and `AIPERF_RUN_ID`; after a durable upload is acknowledged, the regular
//! sidecar exits so its Job can finish.

#[cfg(unix)]
use std::ffi::{CString, OsStr};
use std::fs::File;
use std::future::Future;
use std::io::Read;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, ensure};
use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderValue};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::json;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_util::io::ReaderStream;
use url::Url;

const RESULTS_MANIFEST_NAME: &str = "results-manifest.json";
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MANIFEST_WAIT_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const UPLOAD_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
const UPLOAD_RETRY_DEADLINE: Duration = Duration::from_secs(10 * 60);
const UPLOAD_RETRY_DELAY: Duration = Duration::from_secs(1);
const MAX_UPLOAD_ATTEMPTS: usize = 600;
type ResponseBody = BoxBody<Bytes, std::io::Error>;

#[derive(Clone)]
struct ResultsRoot {
    directory: Arc<File>,
}

impl ResultsRoot {
    #[cfg(unix)]
    fn open(path: &Path) -> anyhow::Result<Self> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .context("results root current directory is unavailable")?
                .join(path)
        };
        let mut directory = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open("/")
            .context("results filesystem root is unavailable")?;
        for component in absolute.components() {
            match component {
                Component::RootDir => continue,
                Component::Normal(name) => {
                    directory = open_directory_at(&directory, name)
                        .with_context(|| format!("unsafe results root {}", path.display()))?;
                }
                _ => anyhow::bail!("results root must be a canonical path"),
            }
        }
        Ok(Self {
            directory: Arc::new(directory),
        })
    }

    #[cfg(not(unix))]
    fn open(_path: &Path) -> anyhow::Result<Self> {
        anyhow::bail!("results serving requires POSIX no-follow descriptors")
    }

    #[cfg(unix)]
    fn open_regular(&self, relative: &Path) -> anyhow::Result<File> {
        let mut components = relative.components().peekable();
        let mut directory = self.directory.as_ref().try_clone()?;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                anyhow::bail!("result path is not canonical and relative");
            };
            if components.peek().is_none() {
                return open_regular_at(&directory, name)
                    .context("result leaf is not a no-follow regular file");
            }
            directory = open_directory_at(&directory, name)
                .context("result ancestor is not a no-follow directory")?;
        }
        anyhow::bail!("result path is empty")
    }

    #[cfg(not(unix))]
    fn open_regular(&self, _relative: &Path) -> anyhow::Result<File> {
        anyhow::bail!("results serving requires POSIX no-follow descriptors")
    }

    fn read_bounded(&self, relative: &Path, maximum: u64) -> anyhow::Result<Vec<u8>> {
        let mut source = self.open_regular(relative)?;
        let metadata = source.metadata()?;
        ensure!(
            metadata.len() <= maximum,
            "result regular file exceeds its limit"
        );
        let mut body = Vec::new();
        source.by_ref().take(maximum + 1).read_to_end(&mut body)?;
        ensure!(
            body.len() as u64 <= maximum,
            "result file exceeds its limit"
        );
        Ok(body)
    }

    fn is_regular(&self, relative: &Path) -> bool {
        self.open_regular(relative).is_ok()
    }
}

#[cfg(unix)]
fn component_name(name: &OsStr) -> std::io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result path component contains NUL",
        )
    })
}

#[cfg(unix)]
fn open_directory_at(parent: &File, name: &OsStr) -> std::io::Result<File> {
    let name = component_name(name)?;
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
    let directory = unsafe { File::from_raw_fd(descriptor) };
    ensure_regular_directory(&directory)?;
    Ok(directory)
}

#[cfg(unix)]
fn ensure_regular_directory(directory: &File) -> std::io::Result<()> {
    if directory.metadata()?.is_dir() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result path component is not a directory",
        ))
    }
}

#[cfg(unix)]
fn open_regular_at(parent: &File, name: &OsStr) -> std::io::Result<File> {
    let name = component_name(name)?;
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let file = unsafe { File::from_raw_fd(descriptor) };
    if file.metadata()?.is_file() {
        Ok(file)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result leaf is not a regular file",
        ))
    }
}

#[derive(Clone)]
struct UploadIdentity {
    namespace: String,
    job_id: String,
    run_id: String,
}

#[derive(Clone, Copy)]
struct UploadPolicy {
    manifest_wait_timeout: Duration,
    attempt_timeout: Duration,
    retry_deadline: Duration,
    retry_delay: Duration,
    max_attempts: usize,
}

impl Default for UploadPolicy {
    fn default() -> Self {
        Self {
            manifest_wait_timeout: MANIFEST_WAIT_TIMEOUT,
            attempt_timeout: UPLOAD_ATTEMPT_TIMEOUT,
            retry_deadline: UPLOAD_RETRY_DEADLINE,
            retry_delay: UPLOAD_RETRY_DELAY,
            max_attempts: MAX_UPLOAD_ATTEMPTS,
        }
    }
}

struct UploadConfig {
    base_url: Url,
    identity: UploadIdentity,
    policy: UploadPolicy,
}

impl UploadConfig {
    fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(raw_url) = std::env::var("AIPERF_RESULTS_UPLOAD_URL")
            .ok()
            .filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let base_url = Url::parse(&raw_url).context("invalid AIPERF_RESULTS_UPLOAD_URL")?;
        ensure!(
            base_url.scheme() == "http",
            "AIPERF_RESULTS_UPLOAD_URL must use in-cluster HTTP"
        );
        ensure!(
            base_url.host_str().is_some()
                && base_url.username().is_empty()
                && base_url.password().is_none(),
            "AIPERF_RESULTS_UPLOAD_URL must be an unauthenticated service URL"
        );
        let identity = UploadIdentity {
            namespace: required_env("AIPERF_NAMESPACE")?,
            job_id: required_env("AIPERF_JOB_ID")?,
            run_id: required_env("AIPERF_RUN_ID")?,
        };
        Ok(Some(Self {
            base_url,
            identity,
            policy: UploadPolicy::default(),
        }))
    }
}

/// Run the results HTTP server until the process is terminated.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let results_dir = flag_value(args, "--results-dir")
        .map(PathBuf::from)
        .or_else(|| std::env::var("AIPERF_RESULTS_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/results"));
    let port = flag_value(args, "--port")
        .or_else(|| std::env::var("AIPERF_RESULTS_SIDECAR_PORT").ok())
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(9091);
    let upload = UploadConfig::from_env()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(serve(results_dir, port, upload))?;
    Ok(0)
}

async fn serve(
    results_dir: PathBuf,
    port: u16,
    upload: Option<UploadConfig>,
) -> anyhow::Result<()> {
    let results_root = ResultsRoot::open(&results_dir)?;
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, dir = %results_dir.display(), "results sidecar listening");
    let server = serve_connections(listener);
    if let Some(upload) = upload {
        supervise_server_and_upload(server, upload_when_ready(results_root, upload)).await
    } else {
        server.await
    }
}

async fn serve_connections(listener: tokio::net::TcpListener) -> anyhow::Result<()> {
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(error) => {
                tracing::warn!(error = %error, "results sidecar accept failed");
                continue;
            }
        };
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(handle);
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::debug!(error = %error, "results sidecar connection ended");
            }
        });
    }
}

async fn supervise_server_and_upload<S, U>(server: S, upload: U) -> anyhow::Result<()>
where
    S: Future<Output = anyhow::Result<()>>,
    U: Future<Output = anyhow::Result<()>>,
{
    tokio::pin!(server);
    tokio::pin!(upload);
    tokio::select! {
        result = &mut upload => result,
        result = &mut server => {
            result?;
            anyhow::bail!("results server stopped before durable upload completed")
        }
    }
}

async fn handle(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<ResponseBody>, std::convert::Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    Ok(tokio::task::spawn_blocking(move || route(&method, &path))
        .await
        .unwrap_or_else(|_| {
            json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &json!({"detail": "results request worker failed"}),
            )
        }))
}

fn route(method: &hyper::Method, path: &str) -> Response<ResponseBody> {
    if method != hyper::Method::GET {
        json_response(
            StatusCode::METHOD_NOT_ALLOWED,
            &json!({"detail": "GET only"}),
        )
    } else if path == "/healthz" {
        json_response(StatusCode::OK, &json!({"status": "ok"}))
    } else {
        json_response(StatusCode::NOT_FOUND, &json!({"detail": "not found"}))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResultsManifest {
    contract_version: String,
    run_id: String,
    ready: bool,
    #[serde(rename = "wasCancelled")]
    _was_cancelled: bool,
    artifact_root: String,
    artifacts: Vec<ManifestArtifact>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ManifestArtifact {
    path: String,
    sha256: String,
    bytes: u64,
    content_type: String,
}

fn required_env(name: &str) -> anyhow::Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{name} is required for durable results upload"))
}

enum UploadBody {
    File(PathBuf),
    Bytes(Bytes),
}

struct UploadRequest {
    uri: Uri,
    method: Method,
    kind: &'static str,
    path: String,
    digest: String,
    length: u64,
    body: UploadBody,
}

fn artifact_upload_request(
    config: &UploadConfig,
    artifact: &ManifestArtifact,
    relative: PathBuf,
) -> anyhow::Result<UploadRequest> {
    Ok(UploadRequest {
        uri: upload_uri(config, Some(&artifact.path))?,
        method: Method::PUT,
        kind: "artifact",
        path: artifact.path.clone(),
        digest: artifact.sha256.clone(),
        length: artifact.bytes,
        body: UploadBody::File(relative),
    })
}

fn manifest_upload_request(config: &UploadConfig, body: Vec<u8>) -> anyhow::Result<UploadRequest> {
    Ok(UploadRequest {
        uri: upload_uri(config, None)?,
        method: Method::POST,
        kind: "manifest",
        path: RESULTS_MANIFEST_NAME.to_string(),
        digest: format!("{:x}", Sha256::digest(&body)),
        length: body.len() as u64,
        body: UploadBody::Bytes(Bytes::from(body)),
    })
}

fn upload_uri(config: &UploadConfig, artifact: Option<&str>) -> anyhow::Result<Uri> {
    let mut url = config.base_url.clone();
    url.set_query(None);
    url.set_fragment(None);
    let mut segments = url
        .path_segments_mut()
        .map_err(|()| anyhow::anyhow!("results upload URL cannot be a base URL"))?;
    segments.pop_if_empty();
    segments.extend([
        "api",
        "uploads",
        &config.identity.namespace,
        &config.identity.job_id,
        &config.identity.run_id,
    ]);
    if let Some(path) = artifact {
        segments.push("artifacts");
        segments.extend(path.split('/'));
    } else {
        segments.push("manifest");
    }
    drop(segments);
    url.as_str().parse().context("invalid results upload URI")
}

async fn upload_when_ready(results_root: ResultsRoot, config: UploadConfig) -> anyhow::Result<()> {
    let manifest_path = PathBuf::from(RESULTS_MANIFEST_NAME);
    tokio::time::timeout(config.policy.manifest_wait_timeout, async {
        loop {
            let root = results_root.clone();
            let path = manifest_path.clone();
            if tokio::task::spawn_blocking(move || root.is_regular(&path)).await? {
                break Ok::<(), anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await
    .context("timed out waiting for a durable results manifest")??;
    let root = results_root.clone();
    let run_id = config.identity.run_id.clone();
    let (manifest, manifest_body) =
        tokio::task::spawn_blocking(move || validated_upload_manifest(&root, &run_id))
            .await
            .context("results manifest validation worker failed")??;
    let connector = HttpConnector::new();
    let client: Client<_, ResponseBody> = Client::builder(TokioExecutor::new()).build(connector);
    for artifact in &manifest.artifacts {
        let relative = safe_relative(&artifact.path)
            .ok_or_else(|| anyhow::anyhow!("manifest contains an unsafe artifact path"))?;
        let request = artifact_upload_request(&config, artifact, relative)?;
        upload_until_ack(&client, &results_root, &config, &request).await?;
    }
    let request = manifest_upload_request(&config, manifest_body)?;
    upload_until_ack(&client, &results_root, &config, &request).await
}

fn validated_upload_manifest(
    results_root: &ResultsRoot,
    run_id: &str,
) -> anyhow::Result<(ResultsManifest, Vec<u8>)> {
    let body = results_root
        .read_bounded(Path::new(RESULTS_MANIFEST_NAME), MAX_MANIFEST_BYTES)
        .context("results manifest is not a bounded no-follow regular file")?;
    let manifest: ResultsManifest =
        serde_json::from_slice(&body).context("results manifest is invalid")?;
    ensure!(
        manifest.contract_version == "native-k8s/v1"
            && manifest.ready
            && manifest.run_id == run_id
            && !manifest.artifact_root.is_empty(),
        "results manifest identity is invalid"
    );
    let mut paths = std::collections::HashSet::new();
    for artifact in &manifest.artifacts {
        let relative = safe_relative(&artifact.path)
            .ok_or_else(|| anyhow::anyhow!("manifest contains an unsafe artifact path"))?;
        ensure!(
            posix(&relative) == artifact.path && paths.insert(artifact.path.as_str()),
            "manifest artifact paths must be canonical and unique"
        );
        ensure!(
            artifact.sha256.len() == 64
                && artifact
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')),
            "manifest artifact digest is invalid"
        );
        ensure!(
            HeaderValue::from_str(&artifact.content_type).is_ok(),
            "manifest artifact content type is invalid"
        );
        let mut source = results_root
            .open_regular(&relative)
            .with_context(|| format!("declared artifact is unavailable: {}", artifact.path))?;
        let metadata = source.metadata().with_context(|| {
            format!(
                "declared artifact metadata is unavailable: {}",
                artifact.path
            )
        })?;
        ensure!(
            metadata.is_file(),
            "declared artifact is not a regular file: {}",
            artifact.path
        );
        ensure!(
            metadata.len() == artifact.bytes && artifact.bytes <= MAX_ARTIFACT_BYTES,
            "declared artifact length mismatch: {}",
            artifact.path
        );
        let mut hasher = Sha256::new();
        std::io::copy(&mut source, &mut hasher)
            .with_context(|| format!("declared artifact is unreadable: {}", artifact.path))?;
        ensure!(
            format!("{:x}", hasher.finalize()) == artifact.sha256,
            "declared artifact digest mismatch: {}",
            artifact.path
        );
    }
    Ok((manifest, body))
}

async fn upload_until_ack(
    client: &Client<HttpConnector, ResponseBody>,
    results_root: &ResultsRoot,
    config: &UploadConfig,
    upload: &UploadRequest,
) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + config.policy.retry_deadline;
    for attempt in 1..=config.policy.max_attempts {
        let outcome = tokio::time::timeout(
            config.policy.attempt_timeout,
            send_upload(client, results_root, upload),
        )
        .await;
        match outcome {
            Ok(Ok(StatusCode::OK | StatusCode::CREATED)) => return Ok(()),
            Ok(Ok(status))
                if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS =>
            {
                tracing::debug!(%status, attempt, path = %upload.path, "results upload will retry");
            }
            Ok(Ok(status)) => {
                anyhow::bail!(
                    "operator rejected {} upload for {} with HTTP {status}",
                    upload.kind,
                    upload.path
                );
            }
            Ok(Err(error)) => {
                tracing::debug!(error = %error, attempt, path = %upload.path, "results upload will retry");
            }
            Err(_) => {
                tracing::debug!(attempt, path = %upload.path, "results upload attempt timed out");
            }
        }
        if attempt == config.policy.max_attempts || tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "durable {} upload for {} exhausted its retry budget",
                upload.kind,
                upload.path
            );
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        tokio::time::sleep(config.policy.retry_delay.min(remaining)).await;
    }
    anyhow::bail!("durable results upload exhausted its retry budget")
}

async fn send_upload(
    client: &Client<HttpConnector, ResponseBody>,
    results_root: &ResultsRoot,
    upload: &UploadRequest,
) -> anyhow::Result<StatusCode> {
    let body = match &upload.body {
        UploadBody::File(relative) => {
            let root = results_root.clone();
            let relative = relative.clone();
            let file = tokio::task::spawn_blocking(move || root.open_regular(&relative))
                .await
                .context("upload artifact open worker failed")??;
            StreamBody::new(
                ReaderStream::new(tokio::fs::File::from_std(file)).map_ok(hyper::body::Frame::data),
            )
            .boxed()
        }
        UploadBody::Bytes(bytes) => full_body(bytes.clone()),
    };
    let request = Request::builder()
        .method(upload.method.clone())
        .uri(upload.uri.clone())
        .header("x-aiperf-content-sha256", &upload.digest)
        .header("x-aiperf-content-length", upload.length)
        .header(CONTENT_LENGTH, upload.length)
        .body(body)
        .context("failed to build results upload request")?;
    Ok(client.request(request).await?.status())
}

/// Lexically validate a request path as a safe relative path under the base:
/// reject absolute paths and any `..` / root / prefix component. Returns the
/// normalized relative path (only `Normal` components) or `None`.
fn safe_relative(filename: &str) -> Option<PathBuf> {
    if filename.chars().any(char::is_control) {
        return None;
    }
    let raw = PathBuf::from(filename);
    let mut rel = PathBuf::new();
    for component in raw.components() {
        match component {
            Component::Normal(part) => rel.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if rel.as_os_str().is_empty() {
        return None;
    }
    Some(rel)
}

fn posix(path: &Path) -> String {
    path.components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

fn json_response(status: StatusCode, body: &serde_json::Value) -> Response<ResponseBody> {
    let bytes = match serde_json::to_vec(body) {
        Ok(bytes) => bytes,
        Err(_) => br#"{"detail":"response serialization failed"}"#.to_vec(),
    };
    response(
        status,
        Some(HeaderValue::from_static("application/json")),
        full_body(Bytes::from(bytes)),
    )
}

fn response(
    status: StatusCode,
    content_type: Option<HeaderValue>,
    body: ResponseBody,
) -> Response<ResponseBody> {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    if let Some(content_type) = content_type {
        response.headers_mut().insert(CONTENT_TYPE, content_type);
    }
    response
}

fn full_body(bytes: Bytes) -> ResponseBody {
    Full::new(bytes).map_err(|never| match never {}).boxed()
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let eq = format!("{flag}=");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(rest) = a.strip_prefix(&eq) {
            return Some(rest.to_string());
        }
        if a == flag {
            return it.next().cloned();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_relative_rejects_traversal_and_absolute() {
        assert!(safe_relative("../secret").is_none());
        assert!(safe_relative("a/../../b").is_none());
        assert!(safe_relative("/etc/passwd").is_none());
        assert!(safe_relative("").is_none());
        assert_eq!(
            safe_relative("aggregate/profile.json"),
            Some(PathBuf::from("aggregate/profile.json"))
        );
        assert_eq!(safe_relative("./x.json"), Some(PathBuf::from("x.json")));
    }

    #[test]
    fn ready_manifest_and_declared_large_artifact_are_valid_for_upload() {
        let directory = tempfile::tempdir().expect("tempdir");
        let payload = vec![b'x'; 9 * 1024 * 1024];
        let digest = format!("{:x}", Sha256::digest(&payload));
        std::fs::write(directory.path().join("large.bin"), &payload).expect("artifact");
        std::fs::write(
            directory.path().join(RESULTS_MANIFEST_NAME),
            format!(
                r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"large.bin","sha256":"{digest}","bytes":{},"contentType":"application/octet-stream"}}]}}"#,
                payload.len()
            ),
        )
        .expect("manifest");
        let root = ResultsRoot::open(directory.path()).expect("results root");

        let (manifest, _) =
            validated_upload_manifest(&root, "run-1").expect("valid upload manifest");
        assert_eq!(manifest.artifacts[0].bytes, payload.len() as u64);
    }

    #[test]
    fn sidecar_refuses_unauthenticated_result_publication_routes() {
        let response = route(&hyper::Method::GET, "/api/results/manifest");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn invalid_manifest_response_metadata_fails_closed_without_panicking() {
        for (filename, content_type) in [
            ("line\nbreak.json", "application/json"),
            ("profile.json", "application/json\r\nx-injected: true"),
        ] {
            let directory = tempfile::tempdir().expect("tempdir");
            let payload = b"{}";
            let digest = format!("{:x}", Sha256::digest(payload));
            let path = directory.path().join(filename);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("artifact parent");
            }
            std::fs::write(&path, payload).expect("artifact");
            let manifest = json!({
                "contractVersion": "native-k8s/v1",
                "runId": "run-1",
                "ready": true,
                "wasCancelled": false,
                "artifactRoot": "/results",
                "artifacts": [{
                    "path": filename,
                    "sha256": digest,
                    "bytes": payload.len(),
                    "contentType": content_type,
                }],
            });
            std::fs::write(
                directory.path().join(RESULTS_MANIFEST_NAME),
                serde_json::to_vec(&manifest).expect("manifest JSON"),
            )
            .expect("manifest");
            let root = ResultsRoot::open(directory.path()).expect("results root");

            let result = std::panic::catch_unwind(|| validated_upload_manifest(&root, "run-1"))
                .expect("invalid response metadata must not panic");
            assert!(result.is_err());
        }
    }

    #[test]
    fn upload_configuration_does_not_require_uid_or_bootstrap() {
        static ENVIRONMENT: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENVIRONMENT.lock().expect("environment lock");
        let names = [
            "AIPERF_RESULTS_UPLOAD_URL",
            "AIPERF_NAMESPACE",
            "AIPERF_JOB_ID",
            "AIPERF_RUN_ID",
            "AIPERF_JOB_UID",
            "AIPERF_ROLE_BOOTSTRAP_FILE",
        ];
        let previous = names.map(|name| (name, std::env::var_os(name)));
        unsafe {
            std::env::set_var("AIPERF_RESULTS_UPLOAD_URL", "http://operator.test:8080");
            std::env::set_var("AIPERF_NAMESPACE", "bench");
            std::env::set_var("AIPERF_JOB_ID", "job-1");
            std::env::set_var("AIPERF_RUN_ID", "run-1");
            std::env::remove_var("AIPERF_JOB_UID");
            std::env::remove_var("AIPERF_ROLE_BOOTSTRAP_FILE");
        }

        let config = UploadConfig::from_env().expect("configuration without result credentials");

        for (name, value) in previous {
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(name, value);
                } else {
                    std::env::remove_var(name);
                }
            }
        }
        let config = config.expect("configured upload");
        assert_eq!(config.identity.namespace, "bench");
        assert_eq!(config.identity.job_id, "job-1");
        assert_eq!(config.identity.run_id, "run-1");
    }

    #[tokio::test]
    async fn upload_request_has_integrity_metadata_but_no_signature() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("connection");
            let service = service_fn(|request| async move {
                assert_eq!(
                    request.headers()["x-aiperf-content-sha256"],
                    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
                );
                assert_eq!(request.headers()["x-aiperf-content-length"], "2");
                assert!(request.headers().get("x-aiperf-signature").is_none());
                Ok::<_, std::convert::Infallible>(
                    Response::builder()
                        .status(StatusCode::CREATED)
                        .body(Full::new(Bytes::new()))
                        .expect("response"),
                )
            });
            hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
                .expect("serve connection");
        });
        let directory = tempfile::tempdir().expect("tempdir");
        let root = ResultsRoot::open(directory.path()).expect("results root");
        let config = UploadConfig {
            base_url: Url::parse(&format!("http://{address}")).expect("URL"),
            identity: UploadIdentity {
                namespace: "bench".to_string(),
                job_id: "job-1".to_string(),
                run_id: "run-1".to_string(),
            },
            policy: UploadPolicy::default(),
        };
        let request = manifest_upload_request(&config, b"{}".to_vec()).expect("request");
        let client: Client<_, ResponseBody> =
            Client::builder(TokioExecutor::new()).build(HttpConnector::new());

        let status = send_upload(&client, &root, &request)
            .await
            .expect("upload response");

        drop(client);
        server.await.expect("server task");
        assert_eq!(status, StatusCode::CREATED);
    }

    #[tokio::test]
    async fn durable_upload_ack_ends_the_regular_sidecar_lifecycle() {
        let server = std::future::pending::<anyhow::Result<()>>();
        let upload = async { Ok(()) };
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            supervise_server_and_upload(server, upload),
        )
        .await
        .expect("durable ACK must end the sidecar")
        .expect("successful durable upload");
    }

    #[tokio::test]
    async fn failed_upload_cannot_report_sidecar_success() {
        let server = std::future::pending::<anyhow::Result<()>>();
        let upload = async { anyhow::bail!("operator rejected upload") };
        let error = supervise_server_and_upload(server, upload)
            .await
            .expect_err("rejected upload must fail the sidecar");
        assert!(error.to_string().contains("operator rejected upload"));
    }

    #[tokio::test]
    async fn missing_manifest_exhausts_bounded_wait_and_fails_sidecar() {
        let directory = tempfile::tempdir().expect("tempdir");
        let root = ResultsRoot::open(directory.path()).expect("results root");
        let config = UploadConfig {
            base_url: Url::parse("http://operator.test").expect("URL"),
            identity: UploadIdentity {
                namespace: "bench".to_string(),
                job_id: "job-1".to_string(),
                run_id: "run-1".to_string(),
            },
            policy: UploadPolicy {
                manifest_wait_timeout: Duration::from_millis(20),
                ..UploadPolicy::default()
            },
        };

        let error = supervise_server_and_upload(
            std::future::pending::<anyhow::Result<()>>(),
            upload_when_ready(root, config),
        )
        .await
        .expect_err("missing manifest must terminate the sidecar with failure");

        assert!(error.to_string().contains("timed out waiting"), "{error:#}");
    }

    #[tokio::test]
    async fn retryable_uploads_stop_at_the_configured_attempt_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let address = listener.local_addr().expect("listener address");
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = attempts.clone();
        let server = tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.expect("connection");
                let observed = observed.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |_request| {
                        observed.fetch_add(1, Ordering::SeqCst);
                        async {
                            Ok::<_, std::convert::Infallible>(
                                Response::builder()
                                    .status(StatusCode::SERVICE_UNAVAILABLE)
                                    .body(Full::new(Bytes::new()))
                                    .expect("response"),
                            )
                        }
                    });
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await
                        .expect("serve connection");
                });
            }
        });
        let directory = tempfile::tempdir().expect("tempdir");
        let root = ResultsRoot::open(directory.path()).expect("results root");
        let config = UploadConfig {
            base_url: Url::parse(&format!("http://{address}")).expect("URL"),
            identity: UploadIdentity {
                namespace: "bench".to_string(),
                job_id: "job-1".to_string(),
                run_id: "run-1".to_string(),
            },
            policy: UploadPolicy {
                manifest_wait_timeout: Duration::from_secs(1),
                attempt_timeout: Duration::from_secs(1),
                retry_deadline: Duration::from_secs(1),
                retry_delay: Duration::from_millis(1),
                max_attempts: 2,
            },
        };
        let request = manifest_upload_request(&config, b"{}".to_vec()).expect("request");
        let client: Client<_, ResponseBody> =
            Client::builder(TokioExecutor::new()).build(HttpConnector::new());

        let error = upload_until_ack(&client, &root, &config, &request)
            .await
            .expect_err("retryable failures must exhaust a finite budget");

        server.abort();
        assert!(error.to_string().contains("exhausted its retry budget"));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retryable_uploads_stop_at_the_configured_deadline() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.expect("connection");
                tokio::spawn(async move {
                    let service = service_fn(|_request| async {
                        Ok::<_, std::convert::Infallible>(
                            Response::builder()
                                .status(StatusCode::SERVICE_UNAVAILABLE)
                                .body(Full::new(Bytes::new()))
                                .expect("response"),
                        )
                    });
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await
                        .expect("serve connection");
                });
            }
        });
        let directory = tempfile::tempdir().expect("tempdir");
        let root = ResultsRoot::open(directory.path()).expect("results root");
        let config = UploadConfig {
            base_url: Url::parse(&format!("http://{address}")).expect("URL"),
            identity: UploadIdentity {
                namespace: "bench".to_string(),
                job_id: "job-1".to_string(),
                run_id: "run-1".to_string(),
            },
            policy: UploadPolicy {
                manifest_wait_timeout: Duration::from_secs(1),
                attempt_timeout: Duration::from_secs(1),
                retry_deadline: Duration::from_millis(20),
                retry_delay: Duration::from_millis(1),
                max_attempts: usize::MAX,
            },
        };
        let request = manifest_upload_request(&config, b"{}".to_vec()).expect("request");
        let client: Client<_, ResponseBody> =
            Client::builder(TokioExecutor::new()).build(HttpConnector::new());

        let error = upload_until_ack(&client, &root, &config, &request)
            .await
            .expect_err("retryable failures must stop at the configured deadline");

        server.abort();
        assert!(error.to_string().contains("exhausted its retry budget"));
    }

    #[cfg(unix)]
    #[test]
    fn upload_refuses_a_manifest_declared_artifact_below_a_symlinked_ancestor() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        let payload = b"private service-account token";
        let digest = format!("{:x}", Sha256::digest(payload));
        std::fs::write(outside.path().join("token"), payload).expect("outside token");
        symlink(outside.path(), directory.path().join("bootstrap")).expect("symlink");
        std::fs::write(
            directory.path().join(RESULTS_MANIFEST_NAME),
            format!(
                r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"bootstrap/token","sha256":"{digest}","bytes":{},"contentType":"application/octet-stream"}}]}}"#,
                payload.len()
            ),
        )
        .expect("manifest");
        let root = ResultsRoot::open(directory.path()).expect("results root");

        validated_upload_manifest(&root, "run-1")
            .expect_err("artifact below a symlinked ancestor must fail closed");
    }

    #[cfg(unix)]
    #[test]
    fn upload_refuses_a_symlinked_manifest() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::NamedTempFile::new().expect("outside");
        std::fs::write(
            outside.path(),
            r#"{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[]}"#,
        )
        .expect("manifest");
        symlink(outside.path(), directory.path().join(RESULTS_MANIFEST_NAME)).expect("symlink");
        let root = ResultsRoot::open(directory.path()).expect("results root");

        validated_upload_manifest(&root, "run-1")
            .expect_err("a symlinked manifest must fail closed");
    }
}
