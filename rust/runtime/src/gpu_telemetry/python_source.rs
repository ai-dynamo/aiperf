// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervised Python GPU telemetry source.
//!
//! Rust owns phase barriers, cadence, timestamps, accumulation, and reporting.
//! This source retains canonical Python collectors (PyNVML, AMDSMI, custom
//! DCGM mappings, and registered user extensions) behind one strict JSON-lines
//! request/response process. Python never schedules inference work or computes
//! benchmark timing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::rc::Rc;

use crate::clock::Clock;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::gpu_telemetry::model::{GpuMetadata, GpuScrape, GpuTelemetryRecord};
use crate::gpu_telemetry::source::{GpuScrapeMode, GpuTelemetryError, GpuTelemetrySource};

/// Version of the Python-collector stdio protocol.
pub const PYTHON_GPU_TELEMETRY_PROTOCOL_VERSION: u32 = 1;

const MAX_PROTOCOL_LINE_BYTES: usize = 64 * 1024 * 1024;

/// Fully resolved Python source process and collector configuration.
#[derive(Debug, Clone)]
pub struct PythonGpuTelemetryConfig {
    /// Absolute Python interpreter path selected by Config v2.
    pub python_executable: PathBuf,
    /// Importable worker module.
    pub worker_module: String,
    /// Registered collector name.
    pub collector: String,
    /// Optional remote DCGM endpoint.
    pub url: Option<String>,
    /// Optional custom DCGM metrics CSV.
    pub metrics_file: Option<PathBuf>,
    /// Reachability/connect timeout passed to HTTP-backed collectors.
    pub request_timeout_seconds: f64,
}

/// Python-backed implementation of the ordinary telemetry-source trait.
pub struct PythonGpuTelemetrySource {
    clock: Rc<dyn Clock>,
    endpoint_url: String,
    worker: Mutex<PythonWorker>,
}

impl PythonGpuTelemetrySource {
    /// Spawns, negotiates, configures, and probes one Python collector.
    pub async fn spawn(
        clock: Rc<dyn Clock>,
        config: PythonGpuTelemetryConfig,
    ) -> Result<Self, GpuTelemetryError> {
        if !config.python_executable.is_absolute() {
            return Err(GpuTelemetryError::Worker(
                "python_executable must be absolute".to_string(),
            ));
        }
        if config.worker_module.trim().is_empty() || config.collector.trim().is_empty() {
            return Err(GpuTelemetryError::Worker(
                "worker_module and collector must be non-empty".to_string(),
            ));
        }
        if !config.request_timeout_seconds.is_finite() || config.request_timeout_seconds <= 0.0 {
            return Err(GpuTelemetryError::Worker(
                "request_timeout_seconds must be finite and positive".to_string(),
            ));
        }

        let mut command = Command::new(&config.python_executable);
        command
            .arg("-u")
            .arg("-m")
            .arg(&config.worker_module)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| {
            GpuTelemetryError::Worker(format!(
                "spawning {}: {error}",
                config.python_executable.display()
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| GpuTelemetryError::Worker("worker stdin is unavailable".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| GpuTelemetryError::Worker("worker stdout is unavailable".to_string()))?;
        let mut worker = PythonWorker {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            next_id: 1,
            shutdown: false,
        };
        let hello_id = worker.take_id()?;
        let hello: HelloResult = worker
            .request(WorkerRequest::Hello {
                id: hello_id,
                protocol: PYTHON_GPU_TELEMETRY_PROTOCOL_VERSION,
            })
            .await?;
        if hello.protocol != PYTHON_GPU_TELEMETRY_PROTOCOL_VERSION {
            return Err(GpuTelemetryError::Worker(format!(
                "worker negotiated protocol {}, expected {}",
                hello.protocol, PYTHON_GPU_TELEMETRY_PROTOCOL_VERSION
            )));
        }
        for capability in ["configure", "scrape", "shutdown"] {
            if !hello.capabilities.iter().any(|item| item == capability) {
                return Err(GpuTelemetryError::Worker(format!(
                    "worker omitted required capability {capability:?}"
                )));
            }
        }

        let configure_id = worker.take_id()?;
        let configured: ConfigureResult = worker
            .request(WorkerRequest::Configure {
                id: configure_id,
                collector: &config.collector,
                url: config.url.as_deref(),
                metrics_file: config.metrics_file.as_deref(),
                request_timeout_seconds: config.request_timeout_seconds,
            })
            .await?;
        if !configured.reachable {
            let reason = configured
                .reason
                .unwrap_or_else(|| "collector source is unavailable".to_string());
            let _ = worker.shutdown().await;
            return Err(GpuTelemetryError::Worker(reason));
        }
        if configured.endpoint_url.trim().is_empty() {
            return Err(GpuTelemetryError::Worker(
                "worker configured an empty endpoint_url".to_string(),
            ));
        }
        Ok(Self {
            clock,
            endpoint_url: configured.endpoint_url,
            worker: Mutex::new(worker),
        })
    }
}

#[async_trait(?Send)]
impl GpuTelemetrySource for PythonGpuTelemetrySource {
    fn endpoint_url(&self) -> &str {
        &self.endpoint_url
    }

    async fn scrape(&self, mode: GpuScrapeMode) -> Result<Option<GpuScrape>, GpuTelemetryError> {
        let mut worker = self.worker.lock().await;
        let id = worker.take_id()?;
        let result: ScrapeResult = worker
            .request(WorkerRequest::Scrape {
                id,
                boundary: mode == GpuScrapeMode::Boundary,
            })
            .await?;
        if result.duplicate && mode == GpuScrapeMode::Continuous {
            return Ok(None);
        }
        if result.endpoint_url != self.endpoint_url {
            return Err(GpuTelemetryError::Worker(format!(
                "worker changed endpoint identity from {:?} to {:?}",
                self.endpoint_url, result.endpoint_url
            )));
        }
        let timestamp_ns = self.clock.now_ns();
        let records = result
            .records
            .into_iter()
            .map(|record| record.into_native(timestamp_ns))
            .collect();
        Ok(Some(GpuScrape {
            timestamp_ns,
            endpoint_url: self.endpoint_url.clone(),
            records,
        }))
    }

    async fn shutdown(&self) -> Result<(), GpuTelemetryError> {
        self.worker.lock().await.shutdown().await
    }
}

struct PythonWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    shutdown: bool,
}

impl PythonWorker {
    fn take_id(&mut self) -> Result<u64, GpuTelemetryError> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| GpuTelemetryError::Worker("worker request id overflow".to_string()))?;
        Ok(id)
    }

    async fn request<T: for<'de> Deserialize<'de>>(
        &mut self,
        request: WorkerRequest<'_>,
    ) -> Result<T, GpuTelemetryError> {
        if self.shutdown {
            return Err(GpuTelemetryError::Worker(
                "worker request attempted after shutdown".to_string(),
            ));
        }
        let expected_id = request.id();
        let encoded = serde_json::to_vec(&request)
            .map_err(|error| GpuTelemetryError::Worker(format!("encoding request: {error}")))?;
        self.stdin.write_all(&encoded).await.map_err(worker_io)?;
        self.stdin.write_all(b"\n").await.map_err(worker_io)?;
        self.stdin.flush().await.map_err(worker_io)?;

        // Cap the read at the reader level (MAX+1 bytes) before allocating:
        // read_line grows the buffer to the full line first and only checks the
        // limit afterward, so a hostile/oversized worker line could OOM us. A
        // line that fills the +1 slack byte is over the limit and rejected.
        let mut line = Vec::new();
        let bytes = (&mut self.stdout)
            .take(MAX_PROTOCOL_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)
            .await
            .map_err(worker_io)?;
        if bytes == 0 {
            let status = self
                .child
                .wait()
                .await
                .map(|status| status.to_string())
                .unwrap_or_else(|error| format!("unknown status: {error}"));
            return Err(GpuTelemetryError::Worker(format!(
                "worker closed stdout ({status})"
            )));
        }
        if bytes > MAX_PROTOCOL_LINE_BYTES {
            return Err(GpuTelemetryError::Worker(format!(
                "worker response exceeded {MAX_PROTOCOL_LINE_BYTES} bytes"
            )));
        }
        let response: WorkerResponse = serde_json::from_slice(&line)
            .map_err(|error| GpuTelemetryError::Worker(format!("decoding response: {error}")))?;
        if response.id != expected_id {
            return Err(GpuTelemetryError::Worker(format!(
                "worker response id {} did not match request {expected_id}",
                response.id
            )));
        }
        if !response.ok {
            return Err(GpuTelemetryError::Worker(response.error.unwrap_or_else(
                || "worker returned an unspecified error".to_string(),
            )));
        }
        serde_json::from_value(response.result.ok_or_else(|| {
            GpuTelemetryError::Worker("successful worker response omitted result".to_string())
        })?)
        .map_err(|error| GpuTelemetryError::Worker(format!("decoding result: {error}")))
    }

    async fn shutdown(&mut self) -> Result<(), GpuTelemetryError> {
        if self.shutdown {
            return Ok(());
        }
        let id = self.take_id()?;
        let result: ShutdownResult = self.request(WorkerRequest::Shutdown { id }).await?;
        if !result.shutdown {
            return Err(GpuTelemetryError::Worker(
                "worker did not acknowledge shutdown".to_string(),
            ));
        }
        self.shutdown = true;
        let status = self.child.wait().await.map_err(worker_io)?;
        if !status.success() {
            return Err(GpuTelemetryError::Worker(format!(
                "worker exited with {status}"
            )));
        }
        Ok(())
    }
}

fn worker_io(error: std::io::Error) -> GpuTelemetryError {
    GpuTelemetryError::Worker(error.to_string())
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum WorkerRequest<'a> {
    Hello {
        id: u64,
        protocol: u32,
    },
    Configure {
        id: u64,
        collector: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metrics_file: Option<&'a Path>,
        request_timeout_seconds: f64,
    },
    Scrape {
        id: u64,
        boundary: bool,
    },
    Shutdown {
        id: u64,
    },
}

impl WorkerRequest<'_> {
    fn id(&self) -> u64 {
        match self {
            Self::Hello { id, .. }
            | Self::Configure { id, .. }
            | Self::Scrape { id, .. }
            | Self::Shutdown { id } => *id,
        }
    }
}

#[derive(Deserialize)]
struct WorkerResponse {
    id: u64,
    ok: bool,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct HelloResult {
    protocol: u32,
    capabilities: Vec<String>,
}

#[derive(Deserialize)]
struct ConfigureResult {
    endpoint_url: String,
    reachable: bool,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Deserialize)]
struct ScrapeResult {
    endpoint_url: String,
    duplicate: bool,
    records: Vec<PythonTelemetryRecord>,
}

#[derive(Deserialize)]
struct PythonTelemetryRecord {
    gpu_index: i32,
    gpu_uuid: String,
    gpu_model_name: String,
    #[serde(default)]
    pci_bus_id: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    pod_name: Option<String>,
    dcgm_url: String,
    telemetry_data: BTreeMap<String, f64>,
}

impl PythonTelemetryRecord {
    fn into_native(self, timestamp_ns: i64) -> GpuTelemetryRecord {
        GpuTelemetryRecord {
            timestamp_ns,
            endpoint_url: self.dcgm_url,
            metadata: GpuMetadata {
                gpu_index: self.gpu_index,
                gpu_uuid: self.gpu_uuid,
                gpu_model_name: self.gpu_model_name,
                pci_bus_id: self.pci_bus_id,
                device: self.device,
                hostname: self.hostname,
                namespace: self.namespace,
                pod_name: self.pod_name,
            },
            metrics: self
                .telemetry_data
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct ShutdownResult {
    shutdown: bool,
}
