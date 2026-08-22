// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf results-sidecar`, the controller pod's results server.
//!
//! The server exposes the controller pod's `/results` volume after
//! `.aiperf_results_ready.json` is written. Checkpoint artifacts remain
//! available while the run is in progress.
//!
//! HTTP contract:
//! - `GET /healthz` -> `{"status":"ok"}`
//! - `GET /api/results/list` -> `{"files":[{"name","size"}],"ready","processing"}`
//! - `GET /api/results/files/{path}` -> marker-gated file bytes after lexical path checks
//!
//! Env: `AIPERF_RESULTS_DIR` (default `/results`),
//! `AIPERF_RESULTS_SIDECAR_PORT` (default `9091`). Binds `0.0.0.0:<port>`.

use std::io::{Read, Seek};
use std::path::{Component, Path, PathBuf};

use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use serde_json::json;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_util::io::ReaderStream;

const RESULTS_MANIFEST_NAME: &str = "results-manifest.json";
const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";
const PROCESSING_MARKER_NAME: &str = ".aiperf_results_processing.json";
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
type ResponseBody = BoxBody<Bytes, std::io::Error>;

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

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(serve(results_dir, port))?;
    Ok(0)
}

async fn serve(results_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, dir = %results_dir.display(), "results sidecar listening");
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(error) => {
                tracing::warn!(error = %error, "results sidecar accept failed");
                continue;
            }
        };
        let dir = results_dir.clone();
        tokio::spawn(async move {
            let io = hyper_util::rt::TokioIo::new(stream);
            let service = service_fn(move |req| handle(req, dir.clone()));
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::debug!(error = %error, "results sidecar connection ended");
            }
        });
    }
}

async fn handle(
    req: Request<hyper::body::Incoming>,
    base_dir: PathBuf,
) -> Result<Response<ResponseBody>, std::convert::Infallible> {
    Ok(route(req.method(), req.uri().path(), &base_dir))
}

fn route(method: &hyper::Method, path: &str, base_dir: &Path) -> Response<ResponseBody> {
    if method != hyper::Method::GET {
        json_response(
            StatusCode::METHOD_NOT_ALLOWED,
            &json!({"detail": "GET only"}),
        )
    } else if path == "/healthz" {
        json_response(StatusCode::OK, &json!({"status": "ok"}))
    } else if path == "/api/results/list" {
        json_response(StatusCode::OK, &list_results(&base_dir))
    } else if path == "/api/results/manifest" {
        serve_manifest(&base_dir)
    } else if let Some(rest) = path.strip_prefix("/api/results/files/") {
        serve_file(&base_dir, rest)
    } else {
        json_response(StatusCode::NOT_FOUND, &json!({"detail": "not found"}))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResultsManifest {
    contract_version: String,
    run_id: String,
    ready: bool,
    was_cancelled: bool,
    artifact_root: String,
    artifacts: Vec<ManifestArtifact>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ManifestArtifact {
    path: String,
    sha256: String,
    bytes: u64,
    content_type: String,
}

fn read_manifest(base_dir: &Path) -> Option<ResultsManifest> {
    let manifest: ResultsManifest =
        serde_json::from_slice(&std::fs::read(base_dir.join(RESULTS_MANIFEST_NAME)).ok()?).ok()?;
    if manifest.contract_version != "native-k8s/v1"
        || !manifest.ready
        || manifest.run_id.is_empty()
        || manifest.artifact_root.is_empty()
        || manifest.was_cancelled && manifest.artifacts.is_empty()
    {
        return None;
    }
    let mut paths = std::collections::HashSet::new();
    for artifact in &manifest.artifacts {
        if safe_relative(&artifact.path).is_none()
            || !paths.insert(&artifact.path)
            || artifact.sha256.len() != 64
        {
            return None;
        }
    }
    Some(manifest)
}

fn list_results(base_dir: &Path) -> serde_json::Value {
    let Some(manifest) = read_manifest(base_dir) else {
        return json!({"files": [], "ready": false, "processing": is_processing(base_dir)});
    };
    let files: Vec<_> = manifest.artifacts.iter().map(|artifact| json!({"name": artifact.path, "size": artifact.bytes, "contentType": artifact.content_type})).collect();
    json!({"files": files, "ready": true, "processing": is_processing(base_dir)})
}

/// Serve a marker-gated artifact after rejecting lexical traversal components.
fn serve_manifest(base_dir: &Path) -> Response<ResponseBody> {
    let path = base_dir.join(RESULTS_MANIFEST_NAME);
    if read_manifest(base_dir).is_none() {
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": "results manifest is not ready"}),
        );
    }
    match std::fs::read(path) {
        Ok(body) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(full_body(Bytes::from(body)))
            .expect("response"),
        Err(_) => json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": "results manifest is unavailable"}),
        ),
    }
}

fn serve_file(base_dir: &Path, filename: &str) -> Response<ResponseBody> {
    let Some(rel) = safe_relative(filename) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({"detail": format!("invalid filename {filename:?}: path traversal")}),
        );
    };
    if rel.file_name().and_then(|n| n.to_str()) == Some(READY_MARKER_NAME)
        || rel.file_name().and_then(|n| n.to_str()) == Some(RESULTS_MANIFEST_NAME)
    {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({"detail": "reserved marker name"}),
        );
    }
    let Some(manifest) = read_manifest(base_dir) else {
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": "results manifest is not ready"}),
        );
    };
    let relative = posix(&rel);
    let Some(declared) = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.path == relative)
    else {
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": "artifact is not declared"}),
        );
    };
    let file_path = base_dir.join(&rel);
    match std::fs::File::open(&file_path) {
        Ok(mut file) => {
            let mut hasher = Sha256::new();
            let mut length = 0_u64;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = match file.read(&mut buffer) {
                    Ok(read) => read,
                    Err(_) => {
                        return json_response(
                            StatusCode::NOT_FOUND,
                            &json!({"detail": format!("result file not found: {filename}")}),
                        );
                    }
                };
                if read == 0 {
                    break;
                }
                length += read as u64;
                if length > MAX_ARTIFACT_BYTES {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        &json!({"detail": "artifact exceeds maximum download size"}),
                    );
                }
                hasher.update(&buffer[..read]);
            }
            if length != declared.bytes || format!("{:x}", hasher.finalize()) != declared.sha256 {
                return json_response(
                    StatusCode::NOT_FOUND,
                    &json!({"detail": "artifact digest mismatch"}),
                );
            }
            if file.rewind().is_err() {
                return json_response(
                    StatusCode::NOT_FOUND,
                    &json!({"detail": format!("result file not found: {filename}")}),
                );
            }
            let ct = declared.content_type.as_str();
            let name = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("download");
            let body = StreamBody::new(
                ReaderStream::new(tokio::fs::File::from_std(file)).map_ok(hyper::body::Frame::data),
            )
            .boxed();
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", ct)
                .header(
                    "content-disposition",
                    format!("attachment; filename=\"{name}\""),
                )
                .header("x-filename", name)
                .header("content-length", length)
                .body(body)
                .expect("response")
        }
        Err(_) => json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": format!("result file not found: {filename}")}),
        ),
    }
}

fn is_processing(base_dir: &Path) -> bool {
    base_dir.join(PROCESSING_MARKER_NAME).is_file()
}

/// Lexically validate a request path as a safe relative path under the base:
/// reject absolute paths and any `..` / root / prefix component. Returns the
/// normalized relative path (only `Normal` components) or `None`.
fn safe_relative(filename: &str) -> Option<PathBuf> {
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
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(full_body(Bytes::from(
            serde_json::to_vec(body).expect("json"),
        )))
        .expect("response")
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

    #[tokio::test]
    async fn ready_manifest_and_declared_large_artifact_are_retrievable() {
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

        let manifest = route(
            &hyper::Method::GET,
            "/api/results/manifest",
            directory.path(),
        );
        assert_eq!(manifest.status(), StatusCode::OK);

        let artifact = route(
            &hyper::Method::GET,
            "/api/results/files/large.bin",
            directory.path(),
        );
        assert_eq!(artifact.status(), StatusCode::OK);
        let content_length = artifact
            .headers()
            .get("content-length")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        assert_eq!(content_length, Some(payload.len().to_string()));
        let streamed = artifact
            .into_body()
            .collect()
            .await
            .expect("artifact stream")
            .to_bytes();
        assert_eq!(streamed, payload);
    }

    #[test]
    fn list_requires_a_valid_manifest() {
        let dir = std::env::temp_dir().join(format!("aiperf-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(READY_MARKER_NAME), b"{\"ready\":true}").unwrap();
        assert_eq!(list_results(&dir)["ready"], false);

        std::fs::write(
            dir.join(RESULTS_MANIFEST_NAME),
            r#"{"contractVersion":"native-k8s/v1","runId":"run","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{"path":"profile.json","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":2,"contentType":"application/json"}]}"#,
        )
        .unwrap();
        assert_eq!(list_results(&dir)["files"][0]["name"], "profile.json");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
