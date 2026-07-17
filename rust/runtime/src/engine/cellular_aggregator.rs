// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier-T2 hierarchical merge — the aggregator role.
//!
//! An *aggregator* is an `aiperf --aggregator` process placed between the
//! cells and the controller to lift the single-controller fan-in ceiling. Instead of
//! all `N` cells shipping their folded stores to one controller (a star: the
//! controller merges `O(N)` partitions and receives `O(N)` shipments on one NIC), the
//! controller launches `M = ceil(N / fanout)` aggregators; each cell ships to its
//! assigned aggregator, each aggregator merges its `~fanout` children's stores and
//! ships **one** merged store up to the controller, and the controller merges only
//! `M`. Because the fold-mode store merge (`merge_store_partitions` →
//! `ColumnStore::append_store` → t-digest merge) is **associative and
//! deterministic-at-topology**, the tree-merged report equals the flat-star report
//! within the same tolerance (counts/sums/extrema exact; percentiles t-digest).
//!
//! An aggregator therefore does exactly the *collect + merge* half of the controller
//! (it reuses [`VeloControllerTransport`] to receive its children's partitions) and
//! the *ship* half of a cell ([`CellRecordsShipper`] to send its one merged store
//! up). It never dispatches load, serves envelopes, or triggers START — the cells
//! still fetch their envelope and await START from the real controller, so a cell's
//! partition/issuer/sampler behaviour is byte-identical to the flat topology; only the
//! terminal ship target moves from the controller to the aggregator.
//!
//! This is the **fold-only** path: only a `StorePartition` (sketch or exact-fold)
//! merges associatively, so an aggregator rejects a raw `Partition` (the byte-exact
//! retain path keeps its star topology, which needs the global dispatch order).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::cellular::transport::connect::{BindSpec, build_velo, parse_endpoint};
use crate::cellular::{
    CellMessage, ColumnStorePartition, ControllerTransport, HeartbeatCounters, MetricsHeartbeat,
    SpecFor, VeloControllerTransport, merge_store_partitions,
};
use crate::engine::cellular_cell::{CELL_CONTROLLER_ADDR_ENV, CellRecordsShipper};

/// Env var carrying this aggregator's id (`0..M`). Set by the controller on each
/// spawned `--aggregator` child; also orders the controller's `merge_store_partitions`.
pub const AGG_ID_ENV: &str = "AIPERF_AGG_ID";

/// Env var carrying the velo coordinate this aggregator binds (`tcp://HOST:PORT`), a
/// fixed loopback port the controller assigned so the aggregator's cells can find it.
pub const AGG_BIND_ENV: &str = "AIPERF_AGG_BIND";

/// Env var carrying how many cells ship to this aggregator (its collect barrier).
pub const AGG_CHILD_COUNT_ENV: &str = "AIPERF_AGG_CHILD_COUNT";

/// Env var (on the run) selecting the aggregator fan-out: the max number of cells one
/// aggregator collects. Unset or `>= cells` keeps the flat star topology. Set to a
/// smaller value to insert `ceil(cells / fanout)` aggregators (tier T2).
pub const CELL_AGG_FANOUT_ENV: &str = "AIPERF_CELL_AGG_FANOUT";

/// Env var overriding the base loopback port aggregators bind (`base + agg_id`);
/// default [`DEFAULT_AGG_BASE_PORT`]. Local execution only — k8s aggregators would use
/// the operator's DNS/ports (a follow-on, like the controller's own k8s bind).
pub const CELL_AGG_BASE_PORT_ENV: &str = "AIPERF_CELL_AGG_BASE_PORT";

/// The default base loopback port for local aggregators.
pub const DEFAULT_AGG_BASE_PORT: u16 = 9700;

/// Env var the **operator** sets on the k8s controller pod to signal that it created
/// the aggregator tier as pods (and injected each cell's ship-DNS). Its presence is
/// the gate that lets the controller take the k8s "expect, don't spawn" path (§3.2):
/// the controller then does NOT spawn aggregator subprocesses and does NOT inject a
/// loopback ship address — it only sizes `expected_partitions = M` and collects the M
/// merged stores the operator-created aggregator pods ship up. Absent on a k8s run,
/// a set [`CELL_AGG_FANOUT_ENV`] fails closed to the flat star (cells would otherwise
/// ship into a void). The value is the operator's DNS template for the aggregator pods
/// (`{jobset}-aggregators-{id}-0.{jobset}.{ns}.svc.cluster.local:{port}`) — carried for
/// observability/validation; the controller consumes only its presence because the
/// operator injects the concrete ship coordinate into each cell pod directly.
pub const AGG_DNS_TEMPLATE_ENV: &str = "AIPERF_CELL_AGG_DNS_TEMPLATE";

/// The number of aggregators for `cells` at `fanout`, or `None` for the flat star
/// topology (fanout unset, `< 1`, or `>= cells` — one aggregator per cell or fewer is
/// pointless). Read from [`CELL_AGG_FANOUT_ENV`].
pub fn aggregator_count(cell_count: u32) -> Option<u32> {
    let fanout: u32 = std::env::var(CELL_AGG_FANOUT_ENV).ok()?.parse().ok()?;
    if fanout < 1 || fanout >= cell_count {
        return None;
    }
    Some(cell_count.div_ceil(fanout))
}

/// Resolve the effective aggregator count for the deployment. `requested` is what the
/// fanout asks for ([`aggregator_count`]); the result is the tier the controller will
/// actually build. Off k8s the request stands (the controller spawns local aggregator
/// subprocesses). On k8s a request only stands when the operator signalled it wired the
/// aggregator tier (`k8s_wired`, from [`AGG_DNS_TEMPLATE_ENV`]); otherwise it falls
/// closed to the flat star (`None`) so cells never ship into a void. Pure so the k8s
/// "expect, don't spawn" gate is unit-testable without a velo runtime.
pub fn effective_aggregator_count(
    is_k8s: bool,
    k8s_wired: bool,
    requested: Option<u32>,
) -> Option<u32> {
    match (is_k8s, requested) {
        (true, Some(_)) if !k8s_wired => None,
        (_, other) => other,
    }
}

/// The base loopback port for aggregators, from [`CELL_AGG_BASE_PORT_ENV`].
pub fn aggregator_base_port() -> u16 {
    std::env::var(CELL_AGG_BASE_PORT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_AGG_BASE_PORT)
}

/// The aggregator a cell ships to under round-robin assignment (`cell_id % agg_count`),
/// as a `tcp://127.0.0.1:PORT` coordinate.
pub fn ship_coordinate(cell_id: u32, agg_count: u32, base_port: u16) -> String {
    let agg = cell_id % agg_count;
    format!("tcp://127.0.0.1:{}", base_port + agg as u16)
}

/// The number of cells assigned to aggregator `agg_id` under round-robin over
/// `cell_count` cells and `agg_count` aggregators: `ceil((cell_count - agg_id) / agg_count)`.
pub fn children_of(agg_id: u32, agg_count: u32, cell_count: u32) -> u32 {
    if agg_id >= cell_count {
        return 0;
    }
    (cell_count - agg_id).div_ceil(agg_count)
}

/// Runs this process as a tier-T2 aggregator: bind at the controller-assigned fixed
/// loopback port, collect its children's folded stores, merge them associatively, and
/// ship the one merged store up to the controller. `envelope` is the run envelope the
/// controller piped on stdin (used only for the merge `MetricsConfig`).
#[cfg(feature = "cellular")]
pub async fn run_aggregator(envelope: &serde_json::Value) -> Result<()> {
    let agg_id: u32 = std::env::var(AGG_ID_ENV)
        .context("AIPERF_AGG_ID not set")?
        .parse()
        .context("parsing AIPERF_AGG_ID")?;
    let bind_coordinate = std::env::var(AGG_BIND_ENV).context("AIPERF_AGG_BIND not set")?;
    // The collect barrier: how many cells ship to this aggregator. Same-host, the
    // controller sets the exact [`AGG_CHILD_COUNT_ENV`] per spawned child. On k8s the
    // aggregators are an *indexed* JobSet replicatedJob sharing one env template, so a
    // per-agg static value cannot express an uneven round-robin split (cells=7, M=3 →
    // 3,2,2). When AGG_CHILD_COUNT is absent, derive it from this pod's AGG_ID and the
    // static cell-count + fanout the operator injects, reusing [`children_of`] so the
    // operator and aggregator can never disagree.
    let child_count: u32 = match std::env::var(AGG_CHILD_COUNT_ENV) {
        Ok(value) => value.parse().context("parsing AIPERF_AGG_CHILD_COUNT")?,
        Err(_) => {
            let cell_count: u32 = std::env::var(crate::cellular::partition::CELL_COUNT_ENV)
                .context(
                    "AIPERF_AGG_CHILD_COUNT unset and AIPERF_CELL_COUNT missing (k8s \
                     aggregator needs the cell count to derive its collect barrier)",
                )?
                .parse()
                .context("parsing AIPERF_CELL_COUNT")?;
            let agg_count = aggregator_count(cell_count).context(
                "AIPERF_AGG_CHILD_COUNT unset and AIPERF_CELL_AGG_FANOUT does not subdivide \
                 the cells (k8s aggregator cannot size its collect barrier)",
            )?;
            children_of(agg_id, agg_count, cell_count)
        }
    };
    let controller_coordinate =
        std::env::var(CELL_CONTROLLER_ADDR_ENV).context("AIPERF_CELL_CONTROLLER_ADDR not set")?;
    let metrics_config = crate::engine::cellular_controller::cellular_metrics_config(envelope)?;

    // Bind velo at the fixed loopback port the controller assigned (its cells dial it
    // by that coordinate). An aggregator serves no envelopes and no START, so the
    // register-handler spec is a no-op and the START handle is a throwaway.
    let bind = match parse_endpoint(&bind_coordinate)? {
        velo::Endpoint::Tcp(addr) => BindSpec::TcpBind(addr),
        #[cfg(unix)]
        velo::Endpoint::Uds(path) => BindSpec::UdsPath(path),
    };
    let velo = build_velo(bind)
        .await
        .with_context(|| format!("aggregator {agg_id} binding velo at {bind_coordinate}"))?;
    let throwaway_event = velo
        .event_manager()
        .new_event()
        .context("aggregator start event")?;
    let noop_spec: SpecFor = std::sync::Arc::new(|_| None);
    let mut transport = VeloControllerTransport::bind_controller(
        velo,
        noop_spec,
        child_count,
        throwaway_event.handle(),
    )
    .with_context(|| format!("aggregator {agg_id} binding transport"))?;

    // Collect exactly `child_count` folded stores (fold-only path), summing the child
    // heartbeat counters. A raw record `Partition` cannot merge associatively, so it is
    // rejected — the retain path keeps the flat star topology.
    let mut store_partitions: Vec<ColumnStorePartition> = Vec::with_capacity(child_count as usize);
    let mut counters = HeartbeatCounters {
        issued: 0,
        completed: 0,
        errored: 0,
    };
    let mut heartbeats: BTreeMap<u32, MetricsHeartbeat> = BTreeMap::new();
    let deadline = tokio::time::sleep(crate::engine::cellular_controller::collect_timeout());
    tokio::pin!(deadline);
    while (store_partitions.len() as u32) < child_count {
        tokio::select! {
            biased;
            message = transport.recv() => match message.with_context(|| format!("aggregator {agg_id} receiving from cell"))? {
                Some(CellMessage::StorePartition(partition)) => store_partitions.push(*partition),
                Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                    counters.issued = counters.issued.saturating_add(heartbeat.counters.issued);
                    counters.completed = counters.completed.saturating_add(heartbeat.counters.completed);
                    counters.errored = counters.errored.saturating_add(heartbeat.counters.errored);
                    heartbeats.insert(cell_id, *heartbeat);
                }
                Some(CellMessage::Partition(_)) => bail!(
                    "aggregator {agg_id} received a raw record Partition; tier-T2 tree merge is \
                     fold-only (sketch or exact-fold). The byte-exact retain path keeps the flat \
                     star topology — do not set {CELL_AGG_FANOUT_ENV} with AIPERF_RUNTIME_EXACT_FOLD=0"
                ),
                None => bail!(
                    "aggregator {agg_id} transport closed with {} of {child_count} child stores",
                    store_partitions.len()
                ),
            },
            _ = &mut deadline => bail!(
                "aggregator {agg_id} timed out with {} of {child_count} child stores",
                store_partitions.len()
            ),
        }
    }

    // Merge the subtree's stores associatively, then ship the ONE merged store up to
    // the controller under this aggregator's id (which orders the controller's merge).
    let merged = merge_store_partitions(metrics_config, store_partitions);
    let epoch_ns = heartbeats
        .values()
        .map(|heartbeat| heartbeat.observed_at_ns)
        .max()
        .unwrap_or(0);
    tracing::info!(
        agg_id,
        child_count,
        issued = counters.issued,
        "aggregator merged its subtree; shipping one store to the controller"
    );
    CellRecordsShipper::to_coordinate(agg_id, controller_coordinate).ship_store(
        merged.column_store().clone(),
        counters,
        epoch_ns,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_tile_exactly_across_aggregators() {
        // Every cell is assigned to exactly one aggregator (round-robin), so the
        // per-aggregator child counts must sum to the cell count — otherwise the
        // controller's collect barrier (one partition per aggregator) and each
        // aggregator's own barrier (child_count stores) would never both complete.
        for cell_count in [1_u32, 2, 5, 6, 7, 60, 1000] {
            for agg_count in 1..=cell_count.min(16) {
                let sum: u32 = (0..agg_count)
                    .map(|agg_id| children_of(agg_id, agg_count, cell_count))
                    .sum();
                assert_eq!(sum, cell_count, "cells={cell_count} aggs={agg_count}");
                // And every cell maps to exactly one aggregator in range.
                for cell_id in 0..cell_count {
                    assert!(cell_id % agg_count < agg_count);
                }
            }
        }
    }

    #[test]
    fn ship_coordinate_round_robins_over_the_base_port() {
        // Cell k ships to aggregator `k % M` at `base + (k % M)`.
        assert_eq!(ship_coordinate(0, 2, 9700), "tcp://127.0.0.1:9700");
        assert_eq!(ship_coordinate(1, 2, 9700), "tcp://127.0.0.1:9701");
        assert_eq!(ship_coordinate(2, 2, 9700), "tcp://127.0.0.1:9700");
        assert_eq!(ship_coordinate(5, 3, 9800), "tcp://127.0.0.1:9802");
    }

    #[test]
    fn aggregator_count_selects_flat_or_tree() {
        // fanout >= cells or < 1 stays flat (None); a real subdivision yields
        // ceil(cells / fanout) aggregators.
        unsafe {
            std::env::set_var(CELL_AGG_FANOUT_ENV, "3");
        }
        assert_eq!(aggregator_count(6), Some(2));
        assert_eq!(aggregator_count(7), Some(3));
        assert_eq!(aggregator_count(3), None, "fanout >= cells is flat");
        assert_eq!(aggregator_count(2), None);
        unsafe {
            std::env::remove_var(CELL_AGG_FANOUT_ENV);
        }
        assert_eq!(aggregator_count(6), None, "unset fanout is flat");
    }

    #[test]
    fn effective_aggregator_count_gates_k8s_on_operator_signal() {
        // Off k8s the request always stands (the controller spawns local aggregators).
        assert_eq!(effective_aggregator_count(false, false, Some(2)), Some(2));
        assert_eq!(effective_aggregator_count(false, false, None), None);
        // On k8s a request only stands when the operator wired the tier; otherwise it
        // falls closed to the flat star so cells never ship into a void.
        assert_eq!(
            effective_aggregator_count(true, false, Some(2)),
            None,
            "k8s fanout without operator wiring must fall back to flat"
        );
        assert_eq!(
            effective_aggregator_count(true, true, Some(2)),
            Some(2),
            "k8s fanout with operator wiring builds the tree"
        );
        // A flat request stays flat regardless of the k8s signal.
        assert_eq!(effective_aggregator_count(true, true, None), None);
    }
}
