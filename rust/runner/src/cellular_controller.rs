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
    CellMessage, ControllerTransport, MetricsHeartbeat, RecordsShardPartition, TDigest,
    TcpControllerTransport, merge_records_in_global_order,
};
use aiperf::metrics_core::report::NativeReport;
use aiperf::metrics_core::{ExportContext, MetricsConfig, PERCENTILES};
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

/// Removes the controller's per-cell scratch tree on any exit path — a normal
/// return or a bailed run — so a failed cellular run never leaks a `/tmp` tree.
struct ScratchTreeGuard(PathBuf);

impl Drop for ScratchTreeGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
    validate_cellular_phase_budgets(envelope)?;
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
        // Cleans the scratch tree on every exit path, including a bail; the
        // kill_on_drop cells stop before this runs when the runtime unwinds.
        let _scratch = ScratchTreeGuard(temp_root.clone());
        let mut children = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let cell_dir = temp_root.join(format!("cell-{cell_id}"));
            std::fs::create_dir_all(&cell_dir)
                .with_context(|| format!("creating cell {cell_id} artifact dir"))?;
            let cell_envelope = build_cell_envelope(envelope, cell_id, cell_count, &cell_dir)?;
            let spec = CellLaunchSpec {
                cell_id,
                cell_count,
                controller_addr: controller_addr.clone(),
                envelope: cell_envelope,
            };
            children.push(spawn_cell(&spec).await?);
        }

        // Watch each cell in a background task and forward the first hard failure, so
        // a cell that dies BEFORE connecting aborts the run rather than hanging the
        // transport's accept loop (which awaits `expected_cells` connections).
        let (failure_tx, mut failure_rx) =
            tokio::sync::mpsc::channel::<String>(cell_count as usize);
        for (cell_id, mut child) in children.into_iter().enumerate() {
            let failure_tx = failure_tx.clone();
            tokio::spawn(async move {
                let report = match child.wait().await {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(format!("cell {cell_id} exited with {status}")),
                    Err(error) => Some(format!("cell {cell_id} could not be waited on: {error}")),
                };
                if let Some(report) = report {
                    let _ = failure_tx.send(report).await;
                }
            });
        }
        drop(failure_tx);

        // Collect exactly one partition per cell (plus the latest heartbeat). A cell
        // failure or an early transport close aborts; a cell that shipped its records
        // and then failed post-ship does not — its records are already authoritative.
        // A cell that connects but hangs indefinitely without shipping is NOT covered
        // (no per-cell deadline yet — the failure watcher only fires on a cell exit);
        // that bound belongs with the cross-host transport work.
        let mut partitions: Vec<RecordsShardPartition> = Vec::with_capacity(cell_count as usize);
        let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
        while partitions.len() < cell_count as usize {
            tokio::select! {
                message = transport.recv() => match message.context("receiving from cell")? {
                    Some(CellMessage::Partition(partition)) => partitions.push(partition),
                    Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                        heartbeats.insert(cell_id, *heartbeat);
                    }
                    None => bail!(
                        "transport closed with {} of {cell_count} cell partitions",
                        partitions.len()
                    ),
                },
                Some(failure) = failure_rx.recv() => bail!("{failure}"),
            }
        }

        // Records-first merge in global dispatch-ordinal order → the single report.
        let merged = merge_records_in_global_order(metrics_config, partitions)
            .context("merging cell partitions")?;
        let record_count = merged.record_count();
        let summary = merged.export_results(&ExportContext::phase(
            aiperf::metrics_core::Phase::Profiling,
        ));
        // Assemble the report so its metric data matches a 1-cell run: the profiling
        // metrics, the warmup section (carried only when a warmup phase actually ran,
        // so a profiling-only run stays byte-identical to the plain builder), plus the
        // run mode/model and configured endpoints. `was_cancelled` is left false — the
        // controller has no cross-cell cancellation.
        //
        // Two blocks a 1-cell report carries are intentionally NOT reproduced here:
        // (1) the coordinator's finalize_run provenance (distribution_id / workload /
        //     alias-resolved endpoint_profiles / extensions) — the controller carries
        //     transport/workload/cells/record_count in its terminal envelope instead
        //     of replaying the coordinator's alias resolution; and
        // (2) the grouped per-error detail — cells ship metric records (with the
        //     error/cancel flags, so error COUNTS are in the metrics) but not the
        //     messages group_record_errors needs, and a cross-cell regroup could not
        //     reproduce the single-cell error-list order.
        let warmup =
            merged.export_results(&ExportContext::phase(aiperf::metrics_core::Phase::Warmup));
        let outcome = aiperf::metrics_core::report::RunOutcome {
            run: aiperf::metrics_core::report::ReportRunInfo {
                mode: Some("online".to_owned()),
                model: cellular_model_name(envelope),
            },
            summary: aiperf::metrics_core::report::ReportSummary {
                endpoints_configured: cellular_endpoint_urls(envelope),
                ..Default::default()
            },
            warmup: (!warmup.result_map().is_empty()).then_some(warmup),
            ..Default::default()
        };
        let report = NativeReport::from_outcome(&summary, &outcome);
        let json = serde_json::to_string_pretty(&report).context("serializing merged report")?;
        std::fs::write(report_path, json)
            .with_context(|| format!("writing merged report to {}", report_path.display()))?;

        // Aggregate the cells' final heartbeats (counters summed, sketches t-digest
        // merged) into one run-wide view written beside the report. The exact report
        // stays authoritative from S2; this is the cross-cell live-lane aggregate.
        let mut aggregate = heartbeats.into_values();
        if let Some(mut merged_heartbeat) = aggregate.next() {
            for heartbeat in aggregate {
                merged_heartbeat.merge(&heartbeat);
            }
            write_heartbeat_sidecar(report_path, &merged_heartbeat)
                .context("writing merged cellular heartbeat")?;
        }

        // `_scratch` removes `temp_root` on drop.
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
        // On any controller abort the runtime is dropped, cancelling the watcher
        // tasks that own these children; kill_on_drop then SIGKILLs each cell so a
        // failed run never leaves cells generating load against the target.
        .kill_on_drop(true)
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
///
/// Each cell's j-th dispatched turn sits at global dispatch stream position
/// `j*cell_count + cell_id` (the sampler yields owned positions `{k, k+C, …}` in
/// order across *all* phases, and the issuer stamps that same value). The stream is
/// the phases concatenated in execution order, so cell `k`'s slice of a phase
/// covering `[base, base+total)` is the count of owned positions in that half-open
/// window — `owned_positions(base+total) − owned_positions(base)` — NOT
/// `owned_positions(total)`. Slicing each phase independently would make the
/// per-cell counts over-count by the warmup phase's `base mod cell_count` remainder,
/// pushing a profiling ordinal past `total` and failing the merge's permutation
/// check. The running `base` keeps the per-cell phase counts telescoping to exactly
/// `owned_positions(grand_total)`, so the global ordinals tile `0..grand_total`.
fn build_cell_envelope(
    envelope: &serde_json::Value,
    cell_id: u32,
    cell_count: u32,
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
    // Cumulative base of the current phase in the concatenated global dispatch
    // stream (phases execute in array order: warmup, then profiling).
    let mut base: u64 = 0;
    for phase in phases.iter_mut() {
        let Some(phase) = phase.as_object_mut() else {
            continue;
        };
        if let Some(requests) = phase.get("requests").and_then(serde_json::Value::as_u64) {
            let owned = owned_positions(base + requests, cell_id, cell_count)
                - owned_positions(base, cell_id, cell_count);
            phase.insert("requests".to_owned(), serde_json::Value::from(owned));
            base += requests;
        }
        // Split the global concurrency cap by the same round-robin share as the
        // request budget so the cells' caps sum to the requested aggregate in-flight.
        // Concurrency is a per-phase saturation cap, not a budget that tiles the
        // stream, so it needs no base offset. `.max(1)` keeps every cell able to make
        // progress when `concurrency < cell_count`, a bounded over-subscription.
        if let Some(concurrency) = phase.get("concurrency").and_then(serde_json::Value::as_u64) {
            let cell_concurrency = owned_positions(concurrency, cell_id, cell_count).max(1);
            phase.insert(
                "concurrency".to_owned(),
                serde_json::Value::from(cell_concurrency),
            );
        }
    }
    Ok(cell)
}

/// Rejects a cellular run whose phases are not exactly request-bounded. The
/// dense-ordinal tiling requires every phase's actual dispatch count to equal its
/// sliced `requests` budget, so a phase that lacks `requests`, or carries a
/// `duration`/`sessions` bound that can stop it early, would mis-partition (run
/// unpartitioned and abort the merge). Fail closed rather than silently corrupt.
fn validate_cellular_phase_budgets(envelope: &serde_json::Value) -> Result<()> {
    let phases = envelope
        .pointer("/run/cfg/phases")
        .and_then(serde_json::Value::as_array)
        .context("run cfg has no phases array")?;
    for phase in phases {
        let name = phase
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        ensure!(
            phase
                .get("requests")
                .and_then(serde_json::Value::as_u64)
                .is_some(),
            "cellular runs require every phase to be request-bounded; phase {name:?} has no `requests` budget"
        );
        ensure!(
            phase.get("duration").is_none() && phase.get("sessions").is_none(),
            "cellular runs do not support a phase with a `duration`/`sessions` bound that can stop before its request budget; phase {name:?}"
        );
    }
    Ok(())
}

/// The number of dispatch-stream positions in `[0, total)` that cell `k` owns under
/// round-robin ownership (`position % cell_count == cell_id`) — `ceil((total-k)/C)`.
/// A phase's per-cell slice is the difference of this over the phase's `[base,
/// base+len)` window (see [`build_cell_envelope`]); over a single phase (`base=0`)
/// it is just each cell's share, summing to `total`.
fn owned_positions(total: u64, cell_id: u32, cell_count: u32) -> u64 {
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

/// The primary model name from the v2 envelope's model list, for the merged
/// report's run info (matching the single-process report).
fn cellular_model_name(envelope: &serde_json::Value) -> Option<String> {
    envelope
        .pointer("/run/cfg/models/items/0/name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// The configured endpoint URLs from the v2 envelope, for the merged report's
/// summary (matching the single-process report).
fn cellular_endpoint_urls(envelope: &serde_json::Value) -> Vec<String> {
    envelope
        .pointer("/run/cfg/endpoint/urls")
        .and_then(serde_json::Value::as_array)
        .map(|urls| {
            urls.iter()
                .filter_map(|url| url.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Reads `cfg.runtime.cells` from a v2 envelope, defaulting to 1 (single process).
pub fn cell_count_from_envelope(envelope: &serde_json::Value) -> u32 {
    envelope
        .pointer("/run/cfg/runtime/cells")
        .and_then(serde_json::Value::as_u64)
        .map(|cells| cells.clamp(1, 1024) as u32)
        .unwrap_or(1)
}

/// Writes the merged cross-cell live heartbeat beside the report as a JSON-safe
/// percentile projection (a raw t-digest anchors `min = +inf`, which JSON cannot
/// encode). The exact report percentiles stay authoritative from S2.
fn write_heartbeat_sidecar(report_path: &Path, heartbeat: &MetricsHeartbeat) -> Result<()> {
    let quantiles: Vec<f64> = PERCENTILES.iter().map(|&p| p as f64 / 100.0).collect();
    let project = |sketch: &TDigest| -> serde_json::Value {
        let percentiles: serde_json::Map<String, serde_json::Value> = PERCENTILES
            .iter()
            .zip(sketch.quantiles(&quantiles))
            .filter_map(|(&p, value)| {
                value.map(|value| (format!("p{p}"), serde_json::json!(value)))
            })
            .collect();
        serde_json::json!({ "count": sketch.count(), "percentiles": percentiles })
    };
    let document = serde_json::json!({
        "event": "cellular_heartbeat_merged",
        "counters": {
            "issued": heartbeat.counters.issued,
            "completed": heartbeat.counters.completed,
            "errored": heartbeat.counters.errored,
        },
        "ttft_ms": project(&heartbeat.ttft_ms),
        "itl_ms": project(&heartbeat.itl_ms),
        "latency_ms": project(&heartbeat.latency_ms),
    });
    let path = report_path.with_file_name("cellular-heartbeat.json");
    std::fs::write(&path, serde_json::to_string_pretty(&document)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_request_bounded_phases() {
        // Request-bounded phases pass.
        let ok = serde_json::json!({"run": {"cfg": {"phases": [
            {"name": "warmup", "requests": 10},
            {"name": "profiling", "requests": 100},
        ]}}});
        assert!(validate_cellular_phase_budgets(&ok).is_ok());
        // A phase lacking `requests`, or carrying a duration/sessions bound, fails closed.
        for bad in [
            serde_json::json!({"run": {"cfg": {"phases": [{"name": "profiling"}]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"name": "profiling", "requests": 100, "duration": 5.0},
            ]}}}),
            serde_json::json!({"run": {"cfg": {"phases": [
                {"name": "profiling", "requests": 100, "sessions": 3},
            ]}}}),
        ] {
            assert!(
                validate_cellular_phase_budgets(&bad).is_err(),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn owned_positions_sum_to_total_and_tile() {
        for total in [1_u64, 7, 100, 500, 501] {
            for count in 1..=8u32 {
                let sum: u64 = (0..count).map(|k| owned_positions(total, k, count)).sum();
                assert_eq!(sum, total, "total {total} count {count}");
            }
        }
    }

    #[test]
    fn multi_phase_global_ordinals_tile_densely() {
        // Regression for the multi-phase merge blocker: with warmup+profiling, each
        // cell's dispatched turns must map under the issuer's `j*count + cell_id` to a
        // dense permutation of `0..(warmup+profiling)`. Slicing each phase with an
        // independent `owned_positions(len)` over-counts by warmup's `base%count`
        // remainder and pushes a profiling ordinal past the total (the merge's
        // `OrdinalOutOfRange`). (100,1000,3) is the exact case from the review.
        let dir = Path::new("/tmp/aiperf-cellular-envelope-test");
        for (warmup, profiling) in [(100u64, 1000u64), (3, 3), (1, 7), (50, 50), (7, 13)] {
            for count in 1..=5u32 {
                let grand_total = (warmup + profiling) as usize;
                let mut seen = vec![false; grand_total];
                for cell_id in 0..count {
                    let envelope = serde_json::json!({"run": {"cfg": {"phases": [
                        {"name": "warmup", "requests": warmup},
                        {"name": "profiling", "requests": profiling},
                    ]}}});
                    let cell = build_cell_envelope(&envelope, cell_id, count, dir).unwrap();
                    let local_count: u64 = cell
                        .pointer("/run/cfg/phases")
                        .and_then(serde_json::Value::as_array)
                        .unwrap()
                        .iter()
                        .filter_map(|p| p.get("requests").and_then(serde_json::Value::as_u64))
                        .sum();
                    for j in 0..local_count {
                        let ordinal = (j * count as u64 + cell_id as u64) as usize;
                        assert!(
                            ordinal < grand_total,
                            "ordinal {ordinal} >= total {grand_total} (w{warmup} p{profiling} c{count} cell{cell_id})"
                        );
                        assert!(!seen[ordinal], "duplicate ordinal {ordinal}");
                        seen[ordinal] = true;
                    }
                }
                assert!(
                    seen.iter().all(|&s| s),
                    "ordinals did not cover 0..{grand_total} (w{warmup} p{profiling} c{count})"
                );
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
