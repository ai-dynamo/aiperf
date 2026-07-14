// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cellular controller — the Phase-2 multi-process topology.
//!
//! When a run requests `cfg.runtime.cells > 1`, the receiving runner becomes the
//! controller rather than executing in-process. It partitions the request budget by
//! `(cell_id, cell_count)`, spawns one `aiperf-runner --cell` child per cell (each a
//! separate OS process, wired with the autonomous issuer and per-cell sampler),
//! serves the [`transport`](aiperf::cellular::transport) endpoint the cells ship
//! their records-shard partitions and heartbeats back over, merges every cell's
//! records in global dispatch-ordinal order into the single authoritative
//! `native-v2.json`, and fails the run loudly if any cell exits non-zero. To the
//! Python orchestrator this is still one run behind one v2 request.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aiperf::cellular::{
    CellMessage, ControllerTransport, MetricsHeartbeat, RecordsShardPartition,
    TcpControllerTransport, merge_records_in_global_order,
};
use aiperf::metrics_core::report::NativeReport;
use aiperf::metrics_core::{ExportContext, MetricsConfig};
use anyhow::{Context, Result, bail, ensure};

use crate::cellular_cell::CellLaunchSpec;

/// The outcome of a cellular run: the merged report path plus a live view of the
/// last heartbeat each cell reported (for diagnostics/logging).
pub struct CellularRunOutcome {
    /// Path of the written merged `native-v2.json`.
    pub report_path: PathBuf,
    /// The number of cells the run was partitioned across.
    pub cell_count: u32,
    /// The merged record count across all cells.
    pub record_count: usize,
}

/// Runs one benchmark across `cell_count` cell subprocesses and writes the merged
/// report to `report_path`. Blocks until every cell finishes.
pub fn run_cellular(
    envelope: &serde_json::Value,
    cell_count: u32,
    report_path: &Path,
    metrics_config: MetricsConfig,
) -> Result<CellularRunOutcome> {
    ensure!(cell_count >= 1, "cell_count must be at least 1");
    let total_requests = profiling_request_budget(envelope)?;
    ensure!(
        total_requests >= cell_count as u64,
        "cellular runs need at least one request per cell ({total_requests} requests, {cell_count} cells)"
    );

    // The controller is off the per-request hot path; a small multi-thread runtime
    // drives the transport accept/read tasks and the child processes concurrently.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("building controller runtime")?;

    runtime.block_on(async move {
        let mut transport = TcpControllerTransport::bind("127.0.0.1:0", cell_count as usize)
            .await
            .context("binding controller transport")?;
        let controller_addr = transport.local_addr().to_string();

        let temp_root =
            std::env::temp_dir().join(format!("aiperf-cellular-{}", std::process::id()));
        let mut children = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let cell_dir = temp_root.join(format!("cell-{cell_id}"));
            std::fs::create_dir_all(&cell_dir)
                .with_context(|| format!("creating cell {cell_id} artifact dir"))?;
            let cell_envelope =
                build_cell_envelope(envelope, cell_id, cell_count, total_requests, &cell_dir)?;
            let spec = CellLaunchSpec {
                cell_id,
                cell_count,
                controller_addr: controller_addr.clone(),
                envelope: cell_envelope,
            };
            children.push(spawn_cell(&spec).await?);
        }

        // Collect each cell's partition + latest heartbeat; a cell reporting failure
        // aborts the run. The stream ends once every cell has closed its connection.
        let mut partitions: Vec<RecordsShardPartition> = Vec::with_capacity(cell_count as usize);
        let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
        while let Some(message) = transport.recv().await.context("receiving from cell")? {
            match message {
                CellMessage::Hello { .. } => {}
                CellMessage::Heartbeat { cell_id, heartbeat } => {
                    heartbeats.insert(cell_id, *heartbeat);
                }
                CellMessage::Partition(partition) => partitions.push(partition),
                CellMessage::Done {
                    cell_id,
                    ok: false,
                    error,
                } => bail!(
                    "cell {cell_id} reported failure: {}",
                    error.unwrap_or_else(|| "unknown".to_owned())
                ),
                CellMessage::Done { ok: true, .. } => {}
            }
        }

        // Every cell process must exit successfully.
        for (cell_id, mut child) in children.into_iter().enumerate() {
            let status = child
                .wait()
                .await
                .with_context(|| format!("waiting on cell {cell_id}"))?;
            ensure!(status.success(), "cell {cell_id} exited with {status}");
        }

        ensure!(
            partitions.len() == cell_count as usize,
            "expected {cell_count} cell partitions, received {}",
            partitions.len()
        );

        // Records-first merge in global dispatch-ordinal order → the single report.
        let merged = merge_records_in_global_order(metrics_config, partitions)
            .context("merging cell partitions")?;
        let record_count = merged.record_count();
        let summary = merged.export_results(&ExportContext::phase(
            aiperf::metrics_core::Phase::Profiling,
        ));
        let report = NativeReport::new(&summary, None);
        let json = serde_json::to_string_pretty(&report).context("serializing merged report")?;
        std::fs::write(report_path, json)
            .with_context(|| format!("writing merged report to {}", report_path.display()))?;

        let _ = std::fs::remove_dir_all(&temp_root);
        Ok(CellularRunOutcome {
            report_path: report_path.to_path_buf(),
            cell_count,
            record_count,
        })
    })
}

/// Spawns one `aiperf-runner --cell` child and pipes its [`CellLaunchSpec`] to stdin.
async fn spawn_cell(spec: &CellLaunchSpec) -> Result<tokio::process::Child> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;
    let exe = std::env::current_exe().context("resolving the runner executable for a cell")?;
    let spec_bytes = serde_json::to_vec(spec).context("serializing cell launch spec")?;
    let mut child = tokio::process::Command::new(exe)
        .arg("--cell")
        .stdin(Stdio::piped())
        // A cell's records flow over the transport; keep stderr for its diagnostics
        // and drop stdout (its would-be terminal envelope is unused by the controller).
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning cell {}", spec.cell_id))?;
    let mut stdin = child.stdin.take().context("cell child stdin unavailable")?;
    stdin
        .write_all(&spec_bytes)
        .await
        .with_context(|| format!("piping launch spec to cell {}", spec.cell_id))?;
    stdin.shutdown().await.ok();
    Ok(child)
}

/// Builds the protocol-v2 envelope for one cell: the same run with its phase
/// budgets sliced to the cell's owned share and its own scratch artifact dir. The
/// dataset and seed are unchanged — the cell's `PartitionedSampler` selects its
/// owned instances from the shared space.
fn build_cell_envelope(
    envelope: &serde_json::Value,
    cell_id: u32,
    cell_count: u32,
    total_requests: u64,
    cell_dir: &Path,
) -> Result<serde_json::Value> {
    let mut cell = envelope.clone();
    let run = cell
        .get_mut("run")
        .and_then(serde_json::Value::as_object_mut)
        .context("envelope has no run object")?;
    run.insert(
        "artifact_dir".to_owned(),
        serde_json::Value::String(cell_dir.to_string_lossy().into_owned()),
    );
    // The cell runs single-process (its slice); its autonomous behaviour comes from
    // the env vars the controller sets, not from re-entering the controller path.
    if let Some(runtime) = run
        .get_mut("cfg")
        .and_then(|cfg| cfg.get_mut("runtime"))
        .and_then(serde_json::Value::as_object_mut)
    {
        runtime.insert("cells".to_owned(), serde_json::Value::from(1));
    }
    let phases = run
        .get_mut("cfg")
        .and_then(|cfg| cfg.get_mut("phases"))
        .and_then(serde_json::Value::as_array_mut)
        .context("run cfg has no phases array")?;
    for phase in phases.iter_mut() {
        let Some(phase) = phase.as_object_mut() else {
            continue;
        };
        if let Some(requests) = phase.get("requests").and_then(serde_json::Value::as_u64) {
            let owned = owned_share(requests, cell_id, cell_count);
            phase.insert("requests".to_owned(), serde_json::Value::from(owned));
        }
        if let Some(concurrency) = phase.get("concurrency").and_then(serde_json::Value::as_u64) {
            let cell_concurrency = (concurrency / cell_count as u64).max(1);
            phase.insert(
                "concurrency".to_owned(),
                serde_json::Value::from(cell_concurrency),
            );
        }
    }
    let _ = total_requests;
    Ok(cell)
}

/// The number of instances cell `k` owns of `total` under round-robin ownership —
/// `ceil((total - k) / count)` — so the cells' shares sum to `total` and their
/// global ordinals tile `0..total`.
fn owned_share(total: u64, cell_id: u32, cell_count: u32) -> u64 {
    let count = cell_count as u64;
    let k = cell_id as u64;
    if k >= total {
        return 0;
    }
    (total - k).div_ceil(count)
}

/// The profiling phase's request budget from the v2 envelope.
fn profiling_request_budget(envelope: &serde_json::Value) -> Result<u64> {
    let phases = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no phases array")?;
    for phase in phases {
        let is_profiling = phase
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(|name| name == "profiling")
            .unwrap_or(false);
        if is_profiling
            && let Some(requests) = phase.get("requests").and_then(serde_json::Value::as_u64)
        {
            return Ok(requests);
        }
    }
    bail!("cellular runs require a profiling phase with a request budget")
}

/// Reads `cfg.runtime.cells` from a v2 envelope, defaulting to 1 (single process).
pub fn cell_count_from_envelope(envelope: &serde_json::Value) -> u32 {
    envelope
        .pointer("/run/cfg/runtime/cells")
        .and_then(serde_json::Value::as_u64)
        .map(|cells| cells.clamp(1, 1024) as u32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_shares_sum_to_total_and_tile() {
        for total in [1_u64, 7, 100, 500, 501] {
            for count in 1..=8u32 {
                let sum: u64 = (0..count).map(|k| owned_share(total, k, count)).sum();
                assert_eq!(sum, total, "total {total} count {count}");
            }
        }
    }

    #[test]
    fn cell_count_reads_runtime_cells() {
        let envelope = serde_json::json!({"run": {"cfg": {"runtime": {"cells": 4}}}});
        assert_eq!(cell_count_from_envelope(&envelope), 4);
        let single = serde_json::json!({"run": {"cfg": {"runtime": {"workers": 1}}}});
        assert_eq!(cell_count_from_envelope(&single), 1);
    }
}
