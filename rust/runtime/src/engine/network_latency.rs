// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profiling-bounded network RTT calibration for one native run.
//!
//! The scheduled phase supplies the start/end barriers. Interval probes are
//! fire-and-forget, while the final barrier tops every target up to the
//! configured successful-sample floor before the metrics accumulator is
//! summarized.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

use crate::clock::Clock;
use crate::network_latency::{
    NetworkLatencyAccumulator, NetworkLatencyProbe, NetworkLatencyTarget, TcpConnectProbe,
};
use crate::phase_runtime::ScheduledPhaseSidecar;
use anyhow::{Context, Result, ensure};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::engine::protocol::{NetworkLatencyProbeSpec, NetworkLatencySpec};

/// Run-owned fixed or actively measured RTT calibration.
pub(crate) struct NetworkLatencyRun {
    fixed_mean_rtt_ns: Option<f64>,
    sidecar: Option<Rc<NetworkLatencySidecar>>,
}

impl NetworkLatencyRun {
    /// Validate Config-v2 lowering and construct unique endpoint probes.
    pub(crate) fn new(
        benchmark_id: &str,
        spec: &NetworkLatencySpec,
        endpoint_urls: &[String],
        clock: Rc<dyn Clock>,
    ) -> Result<Self> {
        ensure!(
            spec.mean_rtt_ns.is_some() ^ spec.probe.is_some(),
            "network latency requires exactly one of mean_rtt_ns or probe"
        );
        if let Some(mean_rtt_ns) = spec.mean_rtt_ns {
            ensure!(
                mean_rtt_ns.is_finite() && mean_rtt_ns >= 0.0,
                "network latency mean_rtt_ns must be finite and non-negative"
            );
            return Ok(Self {
                fixed_mean_rtt_ns: Some(mean_rtt_ns),
                sidecar: None,
            });
        }

        let probe_spec = spec
            .probe
            .as_ref()
            .expect("exclusive network latency mode validated");
        validate_probe_spec(probe_spec)?;
        let mut target_keys = BTreeSet::new();
        let mut targets = Vec::new();
        for endpoint_url in endpoint_urls {
            if let Some(target) = NetworkLatencyTarget::from_endpoint_url(endpoint_url)?
                && target_keys.insert(target.key())
            {
                targets.push(target);
            }
        }
        let probes = targets
            .into_iter()
            .map(|target| {
                Rc::new(TcpConnectProbe::new(clock.clone(), target)) as Rc<dyn NetworkLatencyProbe>
            })
            .collect();
        Ok(Self {
            fixed_mean_rtt_ns: None,
            sidecar: Some(Rc::new(NetworkLatencySidecar::new(
                benchmark_id,
                clock,
                probe_spec,
                probes,
            ))),
        })
    }

    /// Object-safe lifecycle adapter, absent for a fixed mean.
    pub(crate) fn sidecar(&self) -> Option<Rc<dyn ScheduledPhaseSidecar>> {
        self.sidecar
            .as_ref()
            .map(|sidecar| sidecar.clone() as Rc<dyn ScheduledPhaseSidecar>)
    }

    /// Strictly positive fixed/measured RTT delivered before metric export.
    pub(crate) fn mean_rtt_ns(&self) -> Option<f64> {
        self.fixed_mean_rtt_ns
            .or_else(|| {
                self.sidecar
                    .as_ref()
                    .and_then(|sidecar| sidecar.state.accumulator.borrow().mean_rtt_ns())
            })
            .filter(|value| value.is_finite() && *value > 0.0)
    }

    /// Whether this retained resource owns active probes rather than a fixed RTT.
    pub(crate) fn is_active_probe(&self) -> bool {
        self.sidecar.is_some()
    }

    /// Write every active-probe sample in the Python compatibility shape.
    pub(crate) fn write_records_jsonl(&self, path: &Path) -> Result<()> {
        let Some(sidecar) = &self.sidecar else {
            return Ok(());
        };
        JsonlNetworkLatencyArtifactSink.write(path, sidecar.state.accumulator.borrow().samples())
    }
}

fn validate_probe_spec(spec: &NetworkLatencyProbeSpec) -> Result<()> {
    ensure!(
        spec.ping_interval_ns > 0,
        "network latency ping_interval_ns must be positive"
    );
    ensure!(
        spec.connect_timeout_ns > 0,
        "network latency connect_timeout_ns must be positive"
    );
    ensure!(
        spec.complete_topup_timeout_ns >= 0,
        "network latency complete_topup_timeout_ns cannot be negative"
    );
    ensure!(
        spec.min_successful_samples > 0,
        "network latency min_successful_samples must be positive"
    );
    Ok(())
}

trait NetworkLatencyArtifactSink {
    fn write(
        &self,
        path: &Path,
        samples: &[crate::network_latency::NetworkLatencySample],
    ) -> Result<()>;
}

struct JsonlNetworkLatencyArtifactSink;

impl NetworkLatencyArtifactSink for JsonlNetworkLatencyArtifactSink {
    fn write(
        &self,
        path: &Path,
        samples: &[crate::network_latency::NetworkLatencySample],
    ) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating network latency export directory {}",
                    parent.display()
                )
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("creating network latency export {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        for sample in samples {
            serde_json::to_writer(&mut writer, sample).with_context(|| {
                format!("serializing network latency export {}", path.display())
            })?;
            writer
                .write_all(b"\n")
                .with_context(|| format!("writing network latency export {}", path.display()))?;
        }
        writer
            .flush()
            .with_context(|| format!("flushing network latency export {}", path.display()))
    }
}

struct NetworkLatencySidecar {
    state: Rc<NetworkLatencyState>,
}

struct NetworkLatencyState {
    clock: Rc<dyn Clock>,
    ping_interval_ns: i64,
    connect_timeout_ns: i64,
    complete_topup_timeout_ns: i64,
    min_successful_samples: usize,
    probes: Vec<Rc<dyn NetworkLatencyProbe>>,
    accumulator: Rc<RefCell<NetworkLatencyAccumulator>>,
    stop: Rc<Notify>,
    cadence_task: RefCell<Option<JoinHandle<()>>>,
    probe_tasks: RefCell<Vec<JoinHandle<()>>>,
    started: Cell<bool>,
    finished: Cell<bool>,
}

impl NetworkLatencySidecar {
    fn new(
        benchmark_id: &str,
        clock: Rc<dyn Clock>,
        spec: &NetworkLatencyProbeSpec,
        probes: Vec<Rc<dyn NetworkLatencyProbe>>,
    ) -> Self {
        Self {
            state: Rc::new(NetworkLatencyState {
                clock,
                ping_interval_ns: spec.ping_interval_ns,
                connect_timeout_ns: spec.connect_timeout_ns,
                complete_topup_timeout_ns: spec.complete_topup_timeout_ns,
                min_successful_samples: spec.min_successful_samples,
                probes,
                accumulator: Rc::new(RefCell::new(NetworkLatencyAccumulator::new(Some(
                    benchmark_id.to_string(),
                )))),
                stop: Rc::new(Notify::new()),
                cadence_task: RefCell::new(None),
                probe_tasks: RefCell::new(Vec::new()),
                started: Cell::new(false),
                finished: Cell::new(false),
            }),
        }
    }
}

impl ScheduledPhaseSidecar for NetworkLatencySidecar {
    fn start(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
        let state = self.state.clone();
        Box::pin(async move { state.start().await })
    }

    fn finish(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
        let state = self.state.clone();
        Box::pin(async move { state.finish().await })
    }
}

impl NetworkLatencyState {
    async fn start(self: &Rc<Self>) -> Result<()> {
        if self.started.replace(true) {
            return Ok(());
        }
        for probe in &self.probes {
            if let Err(error) = probe.resolve().await {
                tracing::warn!(
                    target = %probe.target().key(),
                    error = %error,
                    "network latency DNS pre-resolution failed; probes will resolve per connect"
                );
            }
            self.spawn_probe(probe.clone(), self.connect_timeout_ns);
        }
        if self.probes.is_empty() {
            tracing::warn!("network latency enabled but no TCP endpoint targets were discovered");
            return Ok(());
        }

        let state = self.clone();
        *self.cadence_task.borrow_mut() = Some(tokio::task::spawn_local(async move {
            state.collect_on_cadence().await;
        }));
        Ok(())
    }

    async fn collect_on_cadence(self: Rc<Self>) {
        loop {
            let sleep = self.clock.clone().sleep(self.ping_interval_ns);
            let stopped = self.stop.notified();
            tokio::pin!(sleep);
            tokio::pin!(stopped);
            tokio::select! {
                biased;
                () = &mut stopped => return,
                () = &mut sleep => {}
            }
            for probe in &self.probes {
                self.spawn_probe(probe.clone(), self.connect_timeout_ns);
            }
        }
    }

    fn spawn_probe(self: &Rc<Self>, probe: Rc<dyn NetworkLatencyProbe>, timeout_ns: i64) {
        self.probe_tasks
            .borrow_mut()
            .retain(|task| !task.is_finished());
        let state = self.clone();
        self.probe_tasks
            .borrow_mut()
            .push(tokio::task::spawn_local(async move {
                let sample = probe.probe_once(timeout_ns).await;
                state.accumulator.borrow_mut().add_sample(sample);
            }));
    }

    async fn finish(self: &Rc<Self>) -> Result<()> {
        if self.finished.replace(true) {
            return Ok(());
        }
        self.stop.notify_one();
        let cadence_task = self.cadence_task.borrow_mut().take();
        if let Some(cadence_task) = cadence_task
            && let Err(error) = cadence_task.await
        {
            tracing::warn!(error = %error, "network latency cadence task failed");
        }

        let probe_tasks = std::mem::take(&mut *self.probe_tasks.borrow_mut());
        for task in &probe_tasks {
            if !task.is_finished() {
                task.abort();
            }
        }
        for task in probe_tasks {
            let _ = task.await;
        }

        let topup_started_ns = self.clock.now_ns();
        let topup_deadline_ns = topup_started_ns.saturating_add(self.complete_topup_timeout_ns);
        for probe in &self.probes {
            let key = probe.target().key();
            let mut attempts = 0usize;
            let maximum_attempts = self.min_successful_samples.saturating_mul(2);
            while self.accumulator.borrow().successful_samples_for(&key)
                < self.min_successful_samples
                && attempts < maximum_attempts
            {
                let remaining_ns = topup_deadline_ns.saturating_sub(self.clock.now_ns());
                if remaining_ns <= 0 {
                    break;
                }
                attempts += 1;
                let sample = probe
                    .probe_once(self.connect_timeout_ns.min(remaining_ns))
                    .await;
                self.accumulator.borrow_mut().add_sample(sample);
            }
            if self.clock.now_ns() >= topup_deadline_ns {
                break;
            }
        }
        Ok(())
    }
}
