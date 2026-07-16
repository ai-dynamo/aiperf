// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded live-results bridge into canonical Python OTel/MLflow extensions.
//!
//! Rust computes every request metric and phase timestamp. A supervised Python
//! child receives only versioned terminal/progress facts over stdio and invokes
//! the existing extension libraries. Queue overflow drops the oldest telemetry
//! event and can never backpressure request dispatch.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::rc::Rc;
use std::time::Duration;

use crate::clock::Clock;
use crate::metrics_core::MetricsConfig;
use crate::timing::{PhaseBranchStats, PhaseConfig, PhaseObserver, PhaseStats};
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdout, Command};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::engine::execute::{NativeEndpointPlan, NativeRunSpec};
use crate::engine::records::{CapturedRecord, record_json_value};

const WORKER_CONTROL_TIMEOUT: Duration = Duration::from_secs(30);
const LIVE_STREAMING_PROTOCOL_VERSION: u32 = 1;

/// Terminal/progress consumer seam for optional live result extensions.
pub(crate) trait LiveResultsSink {
    /// Publish one Rust-computed terminal request record without blocking.
    fn emit_record(&self, record: &CapturedRecord);

    /// Publish one exact native phase snapshot without blocking.
    fn emit_phase(&self, stats: &PhaseStats, observed_at_ns: i64);
}

struct QueueState {
    pending: VecDeque<Vec<u8>>,
    closed: bool,
    dropped_events: u64,
}

struct PythonLiveResultsSink {
    active: Cell<bool>,
    capacity: usize,
    metrics_config: MetricsConfig,
    queue: Rc<RefCell<QueueState>>,
    wake: Rc<Notify>,
}

impl PythonLiveResultsSink {
    fn enqueue(&self, value: &impl Serialize) {
        if !self.active.get() {
            return;
        }
        let mut line = match serde_json::to_vec(value) {
            Ok(line) => line,
            Err(error) => {
                self.queue.borrow_mut().dropped_events += 1;
                tracing::warn!(error = %error, "failed to serialize live telemetry event");
                return;
            }
        };
        line.push(b'\n');
        let mut queue = self.queue.borrow_mut();
        if queue.closed {
            queue.dropped_events += 1;
            return;
        }
        if queue.pending.len() == self.capacity {
            queue.pending.pop_front();
            queue.dropped_events += 1;
        }
        queue.pending.push_back(line);
        drop(queue);
        self.wake.notify_one();
    }

    fn close(&self) {
        self.queue.borrow_mut().closed = true;
        self.wake.notify_one();
    }
}

impl LiveResultsSink for PythonLiveResultsSink {
    fn emit_record(&self, record: &CapturedRecord) {
        match record_json_value(record, &self.metrics_config, false) {
            Ok(record) => self.enqueue(&MetricRecordEvent {
                protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
                event: "metric_record",
                record,
            }),
            Err(error) => {
                self.queue.borrow_mut().dropped_events += 1;
                tracing::warn!(
                    error = format!("{error:#}"),
                    "failed to project live native metric record"
                );
            }
        }
    }

    fn emit_phase(&self, stats: &PhaseStats, observed_at_ns: i64) {
        self.enqueue(&PhaseStatsEvent {
            protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
            event: "phase_stats",
            observed_at_ns,
            stats,
        });
    }
}

/// One supervised Python streaming worker for the lifetime of a native run.
pub(crate) struct PythonLiveStreamingRun {
    sink: Rc<PythonLiveResultsSink>,
    child: Child,
    stdout: BufReader<ChildStdout>,
    stdin: Option<tokio::process::ChildStdin>,
    writer: Option<JoinHandle<Result<u64>>>,
}

impl PythonLiveStreamingRun {
    /// Spawn and side-effect-free prepare the strict extension worker.
    ///
    /// Preparation may validate imports and construct the Python processor, but
    /// the worker cannot start an exporter process or touch the artifact target
    /// until [`Self::activate`] sends the explicit post-commit barrier.
    pub(crate) async fn spawn(run: &NativeRunSpec, metrics_config: MetricsConfig) -> Result<Self> {
        let spec = run
            .sidecars
            .live_streaming()?
            .ok_or_else(|| anyhow!("live streaming spec is absent"))?;
        ensure!(
            spec.python_executable.is_absolute(),
            "live streaming python_executable must be absolute"
        );
        ensure!(
            !spec.worker_module.trim().is_empty(),
            "live streaming worker_module cannot be empty"
        );
        ensure!(
            spec.buffer_capacity > 0,
            "live streaming buffer_capacity must be positive"
        );
        let endpoint = live_endpoint_config(&run.endpoint)?;

        let mut child = Command::new(&spec.python_executable)
            .arg("-u")
            .arg("-m")
            .arg(&spec.worker_module)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "starting live telemetry worker {} with {}",
                    spec.worker_module,
                    spec.python_executable.display()
                )
            })?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("live telemetry worker stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("live telemetry worker stdout was not piped"))?;
        let mut stdout = BufReader::new(stdout);

        let initialize = InitializeEvent {
            protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
            event: "initialize",
            benchmark_id: &run.benchmark_id,
            config: WorkerConfig {
                models: run
                    .models
                    .items
                    .iter()
                    .map(|item| item.name.as_str())
                    .collect(),
                endpoint_type: endpoint.endpoint_id,
                endpoint_urls: endpoint.urls,
                streaming: endpoint.streaming,
                artifact_dir: &run.artifact_dir,
                otel: &spec.otel,
                mlflow: &spec.mlflow,
            },
        };
        write_json_line(&mut stdin, &initialize)
            .await
            .context("initializing live telemetry worker")?;
        let prepared_value = read_json_value(&mut stdout, "prepared").await?;
        let prepared: WorkerPrepared = serde_json::from_value(prepared_value)
            .context("validating live telemetry worker prepared response")?;
        ensure!(
            prepared.protocol_version == LIVE_STREAMING_PROTOCOL_VERSION
                && prepared.event == "prepared",
            "live telemetry worker returned an incompatible prepared response"
        );
        if !prepared.active {
            tracing::warn!(
                reason = prepared
                    .disabled_reason
                    .as_deref()
                    .unwrap_or("no reason supplied"),
                "live telemetry extension disabled itself"
            );
        }

        let queue = Rc::new(RefCell::new(QueueState {
            pending: VecDeque::with_capacity(spec.buffer_capacity),
            closed: false,
            dropped_events: 0,
        }));
        let wake = Rc::new(Notify::new());
        let sink = Rc::new(PythonLiveResultsSink {
            active: Cell::new(false),
            capacity: spec.buffer_capacity,
            metrics_config,
            queue,
            wake,
        });
        Ok(Self {
            sink,
            child,
            stdout,
            stdin: Some(stdin),
            writer: None,
        })
    }

    /// Cross the artifact-ownership barrier and start optional exporters.
    pub(crate) async fn activate(&mut self) -> Result<()> {
        ensure!(
            self.writer.is_none(),
            "live telemetry worker activated twice"
        );
        let mut stdin = self
            .stdin
            .take()
            .ok_or_else(|| anyhow!("live telemetry worker activation stdin is unavailable"))?;
        write_json_line(
            &mut stdin,
            &ActivateEvent {
                protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
                event: "activate",
            },
        )
        .await
        .context("activating live telemetry worker")?;
        let ready_value = read_json_value(&mut self.stdout, "ready").await?;
        let ready: WorkerReady = serde_json::from_value(ready_value)
            .context("validating live telemetry worker ready response")?;
        ensure!(
            ready.protocol_version == LIVE_STREAMING_PROTOCOL_VERSION && ready.event == "ready",
            "live telemetry worker returned an incompatible ready response"
        );
        self.sink.active.set(ready.active);
        if !ready.active {
            tracing::warn!(
                reason = ready
                    .disabled_reason
                    .as_deref()
                    .unwrap_or("no reason supplied"),
                "live telemetry extension did not activate"
            );
        }
        self.writer = Some(tokio::task::spawn_local(pump_worker_stdin(
            stdin,
            self.sink.queue.clone(),
            self.sink.wake.clone(),
        )));
        Ok(())
    }

    /// Clone the local nonblocking event sink.
    pub(crate) fn sink(&self) -> Rc<dyn LiveResultsSink> {
        self.sink.clone()
    }

    /// Drain queued facts, flush the canonical processor, and reap the child.
    pub(crate) async fn shutdown(mut self) -> Result<()> {
        self.sink.close();
        let dropped_events = match self.writer.take() {
            Some(writer) => tokio::time::timeout(WORKER_CONTROL_TIMEOUT, writer)
                .await
                .context("live telemetry stdin drain timed out")?
                .context("live telemetry stdin pump task failed")??,
            None => {
                let mut stdin = self.stdin.take().ok_or_else(|| {
                    anyhow!("live telemetry worker shutdown stdin is unavailable")
                })?;
                write_json_line(
                    &mut stdin,
                    &ShutdownEvent {
                        protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
                        event: "shutdown",
                        dropped_events: 0,
                    },
                )
                .await
                .context("shutting down prepared live telemetry worker")?;
                stdin
                    .shutdown()
                    .await
                    .context("closing prepared live telemetry stdin")?;
                0
            }
        };
        let terminal_value = read_json_value(&mut self.stdout, "terminal").await?;
        let terminal: WorkerTerminal = serde_json::from_value(terminal_value)
            .context("validating live telemetry worker terminal response")?;
        ensure!(
            terminal.protocol_version == LIVE_STREAMING_PROTOCOL_VERSION
                && terminal.event == "terminal",
            "live telemetry worker returned an incompatible terminal response"
        );
        ensure!(
            terminal.success,
            "live telemetry worker failed: {}",
            terminal.error.as_deref().unwrap_or("no diagnostic")
        );
        let status = tokio::time::timeout(WORKER_CONTROL_TIMEOUT, self.child.wait())
            .await
            .context("live telemetry worker exit timed out")?
            .context("waiting for live telemetry worker")?;
        ensure!(
            status.success(),
            "live telemetry worker exited with status {status}"
        );
        if dropped_events > 0 || terminal.dropped_events > 0 {
            tracing::warn!(
                rust = dropped_events,
                worker_acknowledged = terminal.dropped_events,
                delivered_records = terminal.metric_records,
                delivered_phase_events = terminal.phase_events,
                "live telemetry dropped events"
            );
        }
        if terminal.processing_errors > 0 {
            tracing::warn!(
                rejected_events = terminal.processing_errors,
                delivered_records = terminal.metric_records,
                delivered_phase_events = terminal.phase_events,
                "live telemetry processor rejected events"
            );
        }
        Ok(())
    }
}

struct LiveEndpointConfig<'a> {
    endpoint_id: &'a str,
    urls: &'a [String],
    streaming: bool,
}

fn live_endpoint_config(endpoint: &NativeEndpointPlan) -> Result<LiveEndpointConfig<'_>> {
    let NativeEndpointPlan::Prepared(profiles) = endpoint;
    let profile = crate::engine::execute::default_prepared_endpoint_profile(profiles)?;
    Ok(LiveEndpointConfig {
        endpoint_id: profile.endpoint_id.as_str(),
        urls: &profile.config.urls,
        streaming: profile.config.streaming,
    })
}

/// Build a phase observer over an injected live-results sink and Clock.
pub(crate) fn live_phase_observer(
    sink: Rc<dyn LiveResultsSink>,
    clock: Rc<dyn Clock>,
) -> Rc<dyn PhaseObserver> {
    Rc::new(LivePhaseObserver { sink, clock })
}

struct LivePhaseObserver {
    sink: Rc<dyn LiveResultsSink>,
    clock: Rc<dyn Clock>,
}

impl LivePhaseObserver {
    fn emit(&self, stats: PhaseStats) {
        self.sink.emit_phase(&stats, self.clock.now_ns());
    }
}

impl PhaseObserver for LivePhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, stats: PhaseStats) {
        self.emit(stats);
    }

    fn on_progress(&self, stats: PhaseStats) {
        self.emit(stats);
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        self.emit(stats);
    }

    fn on_phase_complete(&self, stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {
        self.emit(stats);
    }
}

async fn pump_worker_stdin(
    stdin: tokio::process::ChildStdin,
    queue: Rc<RefCell<QueueState>>,
    wake: Rc<Notify>,
) -> Result<u64> {
    // Coalesce per-event writes: under bursts many queued events share a single
    // syscall. Flushing whenever the queue drains keeps steady-state latency
    // identical to the unbuffered path, and the auto-flush at buffer capacity
    // bounds how long any event can sit unflushed.
    let mut stdin = BufWriter::new(stdin);
    loop {
        let (next, closed) = {
            let mut state = queue.borrow_mut();
            (state.pending.pop_front(), state.closed)
        };
        if let Some(line) = next {
            stdin
                .write_all(&line)
                .await
                .context("writing live telemetry event")?;
            continue;
        }
        if closed {
            break;
        }
        stdin
            .flush()
            .await
            .context("flushing live telemetry events")?;
        wake.notified().await;
    }
    let dropped_events = queue.borrow().dropped_events;
    write_json_line(
        &mut stdin,
        &ShutdownEvent {
            protocol_version: LIVE_STREAMING_PROTOCOL_VERSION,
            event: "shutdown",
            dropped_events,
        },
    )
    .await
    .context("shutting down live telemetry worker")?;
    stdin
        .shutdown()
        .await
        .context("closing live telemetry stdin")?;
    Ok(dropped_events)
}

async fn write_json_line<W: AsyncWrite + Unpin>(
    stdin: &mut W,
    value: &impl Serialize,
) -> Result<()> {
    let mut line = serde_json::to_vec(value).context("serializing worker protocol event")?;
    line.push(b'\n');
    stdin
        .write_all(&line)
        .await
        .context("writing worker event")?;
    stdin.flush().await.context("flushing worker event")
}

async fn read_json_value(stdout: &mut BufReader<ChildStdout>, label: &str) -> Result<Value> {
    let mut line = String::new();
    let bytes = tokio::time::timeout(WORKER_CONTROL_TIMEOUT, stdout.read_line(&mut line))
        .await
        .with_context(|| format!("live telemetry worker {label} response timed out"))?
        .with_context(|| format!("reading live telemetry worker {label} response"))?;
    ensure!(
        bytes > 0,
        "live telemetry worker exited before its {label} response"
    );
    serde_json::from_str(&line)
        .with_context(|| format!("parsing live telemetry worker {label} response"))
}

#[derive(Serialize)]
struct InitializeEvent<'a> {
    protocol_version: u32,
    event: &'static str,
    benchmark_id: &'a str,
    config: WorkerConfig<'a>,
}

#[derive(Serialize)]
struct WorkerConfig<'a> {
    models: Vec<&'a str>,
    endpoint_type: &'a str,
    endpoint_urls: &'a [String],
    streaming: bool,
    artifact_dir: &'a Path,
    otel: &'a crate::engine::protocol::OTelStreamingSpec,
    mlflow: &'a crate::engine::protocol::MLflowStreamingSpec,
}

#[derive(Serialize)]
struct ActivateEvent {
    protocol_version: u32,
    event: &'static str,
}

#[derive(Serialize)]
struct MetricRecordEvent {
    protocol_version: u32,
    event: &'static str,
    record: Value,
}

#[derive(Serialize)]
struct PhaseStatsEvent<'a> {
    protocol_version: u32,
    event: &'static str,
    observed_at_ns: i64,
    stats: &'a PhaseStats,
}

#[derive(Serialize)]
struct ShutdownEvent {
    protocol_version: u32,
    event: &'static str,
    dropped_events: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerPrepared {
    protocol_version: u32,
    event: String,
    active: bool,
    #[serde(default)]
    disabled_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReady {
    protocol_version: u32,
    event: String,
    active: bool,
    #[serde(default)]
    disabled_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerTerminal {
    protocol_version: u32,
    event: String,
    success: bool,
    #[serde(default)]
    metric_records: u64,
    #[serde(default)]
    phase_events: u64,
    #[serde(default)]
    processing_errors: u64,
    #[serde(default)]
    dropped_events: u64,
    #[serde(default)]
    error: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn bounded_queue_drops_oldest_event() {
        let queue = Rc::new(RefCell::new(QueueState {
            pending: VecDeque::new(),
            closed: false,
            dropped_events: 0,
        }));
        let sink = PythonLiveResultsSink {
            active: Cell::new(true),
            capacity: 2,
            metrics_config: MetricsConfig::default(),
            queue: queue.clone(),
            wake: Rc::new(Notify::new()),
        };
        sink.enqueue(&serde_json::json!({"index": 1}));
        sink.enqueue(&serde_json::json!({"index": 2}));
        sink.enqueue(&serde_json::json!({"index": 3}));

        let state = queue.borrow();
        assert_eq!(state.dropped_events, 1);
        assert_eq!(state.pending.len(), 2);
        assert_eq!(
            serde_json::from_slice::<Value>(&state.pending[0]).unwrap()["index"],
            2
        );
    }

    #[test]
    fn prepared_endpoint_projects_open_identity_without_legacy_conversion() {
        let endpoint = NativeEndpointPlan::Prepared(Arc::new(vec![
            crate::engine::registry::ValidatedEndpointProfileV2 {
                profile_id: "default".into(),
                endpoint_id: crate::endpoints::EndpointId::new("extension_chat").unwrap(),
                config: crate::endpoints::RawEndpointConfig {
                    urls: vec!["http://example.test/v1".into()],
                    streaming: false,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
                connection_reuse: crate::transport::core::ConnectionReuseStrategy::default(),
                client: Default::default(),
                session_header: None,
            },
        ]));

        let projected = live_endpoint_config(&endpoint).unwrap();
        assert_eq!(projected.endpoint_id, "extension_chat");
        assert_eq!(projected.urls, ["http://example.test/v1"]);
        assert!(!projected.streaming);
    }
}
