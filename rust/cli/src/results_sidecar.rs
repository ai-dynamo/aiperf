// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native `aiperf results-sidecar` — the controller pod's results server.
//!
//! Native port of the Python `aiperf.kubernetes.results_sidecar`: a tiny HTTP
//! server that serves the controller pod's `/results` volume so the operator can
//! recover artifacts even if the main container exits after export. Files are
//! hidden until the controller writes `.aiperf_results_ready.json` (see
//! `crate::k8s::write_ready_marker`), preventing consumers from downloading
//! partial exports; `checkpoints/` artifacts are served unconditionally.
//!
//! Contract (unchanged, so the operator's `_completion_fetch` fetch works
//! against either implementation):
//! - `GET /healthz` -> `{"status":"ok"}`
//! - `GET /api/results/list` -> `{"files":[{"name","size"}],"ready","processing"}`
//! - `GET /api/results/files/{path}` -> the file bytes (marker-gated, traversal-safe)
//!
//! Env: `AIPERF_RESULTS_DIR` (default `/results`),
//! `AIPERF_RESULTS_SIDECAR_PORT` (default `9091`). Binds `0.0.0.0:<port>`.

use std::path::{Component, Path, PathBuf};

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use serde_json::json;

const READY_MARKER_NAME: &str = ".aiperf_results_ready.json";
const PROCESSING_MARKER_NAME: &str = ".aiperf_results_processing.json";
const CHECKPOINTS_DIR_NAME: &str = "checkpoints";

/// `aiperf results-sidecar` entry: run the results HTTP server until killed.
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

/// Bind `0.0.0.0:port` and serve connections until the process is killed.
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

/// Route one request. Only `GET` is served.
async fn handle(
    req: Request<hyper::body::Incoming>,
    base_dir: PathBuf,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let path = req.uri().path().to_string();
    let resp = if req.method() != hyper::Method::GET {
        json_response(
            StatusCode::METHOD_NOT_ALLOWED,
            &json!({"detail": "GET only"}),
        )
    } else if path == "/healthz" {
        json_response(StatusCode::OK, &json!({"status": "ok"}))
    } else if path == "/api/results/list" {
        json_response(StatusCode::OK, &list_results(&base_dir))
    } else if let Some(rest) = path.strip_prefix("/api/results/files/") {
        serve_file(&base_dir, rest)
    } else {
        json_response(StatusCode::NOT_FOUND, &json!({"detail": "not found"}))
    };
    Ok(resp)
}

/// `ResultsListResponse`: files (name+size), plus ready/processing flags.
fn list_results(base_dir: &Path) -> serde_json::Value {
    if !base_dir.is_dir() {
        return json!({"files": [], "ready": false, "processing": false});
    }
    let files: Vec<serde_json::Value> = collect_files(base_dir)
        .into_iter()
        .map(|(name, size)| json!({"name": name, "size": size}))
        .collect();
    json!({
        "files": files,
        "ready": is_ready(base_dir),
        "processing": is_processing(base_dir),
    })
}

/// Enumerate downloadable artifacts: everything under `base_dir` once the ready
/// marker is set (excluding the marker itself and `checkpoints/`), plus every
/// `checkpoints/` file unconditionally. Sorted by relative posix name.
fn collect_files(base_dir: &Path) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = Vec::new();
    let checkpoints = base_dir.join(CHECKPOINTS_DIR_NAME);

    if is_ready(base_dir) {
        for entry in walk_files(base_dir) {
            if entry.starts_with(&checkpoints) {
                continue;
            }
            if entry.file_name().and_then(|n| n.to_str()) == Some(READY_MARKER_NAME) {
                continue;
            }
            if let (Ok(rel), Ok(meta)) = (entry.strip_prefix(base_dir), entry.metadata()) {
                out.push((posix(rel), meta.len()));
            }
        }
    }
    if checkpoints.is_dir() {
        for entry in walk_files(&checkpoints) {
            if let (Ok(rel), Ok(meta)) = (entry.strip_prefix(base_dir), entry.metadata()) {
                out.push((posix(rel), meta.len()));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Serve one file: reject traversal / the reserved marker (400); gate on the
/// ready marker unless it is a `checkpoints/` path (404); stream the bytes.
fn serve_file(base_dir: &Path, filename: &str) -> Response<Full<Bytes>> {
    let Some(rel) = safe_relative(filename) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({"detail": format!("invalid filename {filename:?}: path traversal")}),
        );
    };
    if rel.file_name().and_then(|n| n.to_str()) == Some(READY_MARKER_NAME) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({"detail": "reserved marker name"}),
        );
    }
    let is_checkpoint =
        rel.components().next().and_then(|c| c.as_os_str().to_str()) == Some(CHECKPOINTS_DIR_NAME);
    if !is_ready(base_dir) && !is_checkpoint {
        let processing = if is_processing(base_dir) {
            " export still processing;"
        } else {
            ""
        };
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": format!(
                "results not ready;{processing} marker {READY_MARKER_NAME} not present"
            )}),
        );
    }
    let file_path = base_dir.join(&rel);
    match std::fs::read(&file_path) {
        Ok(bytes) => {
            let ct = content_type(&file_path);
            let name = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("download");
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", ct)
                .header(
                    "content-disposition",
                    format!("attachment; filename=\"{name}\""),
                )
                .header("x-filename", name)
                .body(Full::new(Bytes::from(bytes)))
                .expect("response")
        }
        Err(_) => json_response(
            StatusCode::NOT_FOUND,
            &json!({"detail": format!("result file not found: {filename}")}),
        ),
    }
}

fn is_ready(base_dir: &Path) -> bool {
    base_dir.join(READY_MARKER_NAME).is_file()
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
            // Traversal, absolute root, or a Windows prefix — reject outright.
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if rel.as_os_str().is_empty() {
        return None;
    }
    Some(rel)
}

/// Recursively collect regular files under `root` (best-effort; unreadable dirs
/// are skipped). Iterative to avoid unbounded recursion depth.
fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(path),
                Ok(ft) if ft.is_file() => files.push(path),
                _ => {}
            }
        }
    }
    files
}

/// Relative path -> forward-slash string (stable listing keys across platforms).
fn posix(path: &Path) -> String {
    path.components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

/// Content-type by extension (mirrors the Python `_CONTENT_TYPES` map).
fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("jsonl") => "application/x-ndjson",
        Some("csv") => "text/csv",
        Some("parquet") => "application/vnd.apache.parquet",
        Some("txt") => "text/plain",
        _ => "application/octet-stream",
    }
}

fn json_response(status: StatusCode, body: &serde_json::Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(
            serde_json::to_vec(body).expect("json"),
        )))
        .expect("response")
}

/// Scan `args` for `--flag <value>` / `--flag=<value>`.
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
    fn content_type_by_extension() {
        assert_eq!(content_type(Path::new("a.json")), "application/json");
        assert_eq!(content_type(Path::new("a.jsonl")), "application/x-ndjson");
        assert_eq!(content_type(Path::new("a.csv")), "text/csv");
        assert_eq!(
            content_type(Path::new("a.parquet")),
            "application/vnd.apache.parquet"
        );
        assert_eq!(content_type(Path::new("a.bin")), "application/octet-stream");
    }

    #[test]
    fn list_gates_on_ready_marker_but_always_serves_checkpoints() {
        let dir = std::env::temp_dir().join(format!("aiperf-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("checkpoints")).unwrap();
        std::fs::write(dir.join("profile_export_aiperf.json"), b"{}").unwrap();
        std::fs::write(dir.join("checkpoints").join("ckpt_0.parquet"), b"x").unwrap();

        // Not ready: only the checkpoint file is listed.
        let before = collect_files(&dir);
        let names: Vec<&str> = before.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["checkpoints/ckpt_0.parquet"]);

        // Ready: the export surfaces too, marker excluded.
        std::fs::write(dir.join(READY_MARKER_NAME), b"{\"ready\":true}").unwrap();
        let after = collect_files(&dir);
        let names: Vec<&str> = after.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["checkpoints/ckpt_0.parquet", "profile_export_aiperf.json"]
        );
        assert!(!names.iter().any(|n| n.contains(READY_MARKER_NAME)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
