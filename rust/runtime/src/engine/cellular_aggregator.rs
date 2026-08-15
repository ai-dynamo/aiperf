// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hierarchical folded-store aggregation.
//!
//! An *aggregator* is an `aiperf --aggregator` process placed between the
//! cells and the controller to lift the single-controller fan-in ceiling. Instead of
//! all `N` cells shipping their folded stores to one controller (a star: the
//! controller merges `O(N)` partitions and receives `O(N)` shipments on one NIC), the
//! controller launches a **reduction tree** of aggregators; each cell ships to its
//! assigned tier-1 aggregator, each aggregator merges its `~fanout` children's stores
//! and ships **one** merged store up, and the controller merges only the top tier.
//! Because the fold-mode store merge (`merge_store_partitions` →
//! `ColumnStore::append_store` → t-digest merge) is **associative and
//! deterministic-at-topology**, the tree-merged report equals the flat-star report
//! within the same tolerance (counts/sums/extrema exact; percentiles t-digest).
//!
//! The tree can be **multi-tier**: each tier reduces the one below it by `fanout`
//! (tier 1 has `ceil(N / fanout)` nodes, tier 2 `ceil(tier1 / fanout)`, …) until the
//! top tier is `<= fanout` nodes and ships to the controller
//! ([`tier_counts`]). A run whose first reduction already lands `<= fanout` nodes is a
//! single-tier tree — byte-identical to the original 2-level topology. Because an
//! aggregator is **symmetric** (it collects `StorePartition`s and ships one up,
//! exactly like a cell to the tier above it), the wire, merge, and associativity carry
//! over unchanged at every tier; only its ship target differs (a parent aggregator for
//! a lower tier, the controller for the top tier).
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

/// Env var carrying how many children ship to this aggregator (its collect barrier).
/// In a multi-tier tree the children may themselves be lower-tier aggregators, not
/// cells; the barrier is the count either way.
pub const AGG_CHILD_COUNT_ENV: &str = "AIPERF_AGG_CHILD_COUNT";

/// Env var carrying the velo coordinate this aggregator ships its ONE merged store up
/// to. In the single-tier tree (and the top tier of a multi-tier tree) this is unset
/// and the aggregator ships to the controller ([`CELL_CONTROLLER_ADDR_ENV`]); in a
/// lower tier of a multi-tier tree the controller sets it to this node's parent
/// aggregator (`tcp://HOST:PORT`), which merges this tier's subtree further up. Only
/// the ship target moves — the collect+merge half is identical at every tier, because
/// an aggregator is symmetric: it consumes `StorePartition`s and ships one up, exactly
/// like a cell to the tier above it.
pub const AGG_SHIP_ADDR_ENV: &str = "AIPERF_AGG_SHIP_ADDR";

/// Env var (on the run) selecting the aggregator fan-out: the max number of cells one
/// aggregator collects. Unset or `>= cells` keeps the flat star topology. Set to a
/// smaller value to insert `ceil(cells / fanout)` aggregators.
pub const CELL_AGG_FANOUT_ENV: &str = "AIPERF_CELL_AGG_FANOUT";

/// Env var overriding the base loopback port aggregators bind (`base + agg_id`);
/// default [`DEFAULT_AGG_BASE_PORT`]. Local execution only; Kubernetes aggregators use
/// operator-provided DNS and ports.
pub const CELL_AGG_BASE_PORT_ENV: &str = "AIPERF_CELL_AGG_BASE_PORT";

/// The default base loopback port for local aggregators.
pub const DEFAULT_AGG_BASE_PORT: u16 = 9700;

/// Env var the **operator** sets on the k8s controller pod to signal that it created
/// the aggregators as pods and injected each cell's ship DNS. Its presence selects
/// the Kubernetes "expect, don't spawn" path:
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
/// fanout asks for ([`aggregator_count`]); the result is the topology the controller will
/// actually build. Off k8s the request stands (the controller spawns local aggregator
/// subprocesses). On k8s a request only stands when the operator signalled it wired the
/// aggregators (`k8s_wired`, from [`AGG_DNS_TEMPLATE_ENV`]); otherwise it falls
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

/// The number of children assigned to node `agg_id` under round-robin over
/// `child_count` children of the tier below and `agg_count` nodes at this tier:
/// `ceil((child_count - agg_id) / agg_count)`. The children are cells for tier 1 and
/// lower-tier aggregators for higher tiers; the round-robin share is identical either
/// way (`child % agg_count == agg_id`).
pub fn children_of(agg_id: u32, agg_count: u32, child_count: u32) -> u32 {
    if agg_id >= child_count {
        return 0;
    }
    (child_count - agg_id).div_ceil(agg_count)
}

/// The aggregator node counts per tier for `cell_count` cells reduced by `fanout`,
/// from tier 1 (the tier the cells ship to) up to the top tier that ships to the
/// controller. Empty when the fanout does not subdivide the cells (the flat star:
/// `fanout < 2` or `fanout >= cell_count`, since one aggregator per cell or fewer is
/// pointless).
///
/// Each tier reduces the one below by `ceil(prev / fanout)`, stopping once a tier is
/// `<= fanout` (that tier ships straight to the controller). The reduction strictly
/// decreases while `fanout >= 2`, so it always terminates. The first element equals
/// [`aggregator_count`]'s value, so the cell→tier-1 ship wiring is unchanged; a plan of
/// length 1 is the original 2-level tree.
pub fn tier_counts(cell_count: u32, fanout: u32) -> Vec<u32> {
    if fanout < 2 || fanout >= cell_count {
        return Vec::new();
    }
    let mut tiers = Vec::new();
    let mut prev = cell_count;
    loop {
        let count = prev.div_ceil(fanout);
        tiers.push(count);
        if count <= fanout {
            break;
        }
        prev = count;
    }
    tiers
}

/// The tier plan for `cell_count` read from [`CELL_AGG_FANOUT_ENV`], or empty for the
/// flat star (fanout unset, `< 2`, or `>= cell_count`).
pub fn tier_counts_from_env(cell_count: u32) -> Vec<u32> {
    match std::env::var(CELL_AGG_FANOUT_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(fanout) => tier_counts(cell_count, fanout),
        None => Vec::new(),
    }
}

/// Env carrying a k8s aggregator's 0-based tier index within the
/// [`tier_counts`] plan. The operator sets it on each `aggregators-{tier}` pod of a
/// **multi-tier** k8s tree so the pod can locate its tier in the plan and derive both
/// its collect barrier ([`k8s_tier_child_count`]) and its parent
/// ([`k8s_tier_parent_id`]) — a JobSet indexed replicatedJob shares one env template, so
/// per-pod placement must be derived, not injected. Absent for a same-host aggregator
/// (the controller sets [`AGG_CHILD_COUNT_ENV`] + a concrete ship addr per spawned
/// child) and for the single-tier k8s tree (tier 0, derived by default).
pub const AGG_TIER_INDEX_ENV: &str = "AIPERF_AGG_TIER_INDEX";

/// This node's collect barrier in a k8s multi-tier tree: the children assigned to
/// aggregator `agg_id` at `tier_index` of the [`tier_counts`]`(cell_count, fanout)`
/// plan, under the same round-robin [`children_of`] the rest of the tree uses. Tier 0's
/// children are the cells; a higher tier's children are the nodes of the tier below.
/// `None` when the plan has no such tier (a flat star, or an out-of-range index).
pub fn k8s_tier_child_count(
    cell_count: u32,
    fanout: u32,
    tier_index: usize,
    agg_id: u32,
) -> Option<u32> {
    let tiers = tier_counts(cell_count, fanout);
    let this_count = *tiers.get(tier_index)?;
    let child_tier_count = if tier_index == 0 {
        cell_count
    } else {
        tiers[tier_index - 1]
    };
    Some(children_of(agg_id, this_count, child_tier_count))
}

/// The parent aggregator id this node ships its one merged store to in a k8s multi-tier
/// tree: round-robin `agg_id % parent_count` over the tier above (`tier_index + 1`),
/// mirroring [`aggregator_nodes`]' `id % parent_count`. `None` when `tier_index` is the
/// top tier (which ships to the controller, not a parent aggregator) or out of range.
pub fn k8s_tier_parent_id(
    cell_count: u32,
    fanout: u32,
    tier_index: usize,
    agg_id: u32,
) -> Option<u32> {
    let tiers = tier_counts(cell_count, fanout);
    let parent_count = *tiers.get(tier_index + 1)?;
    Some(agg_id % parent_count)
}

/// Where a spawned aggregator ships its one merged store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShipTarget {
    /// The controller (the top tier of the tree).
    Controller,
    /// A parent aggregator at a fixed loopback `base + offset` port (a lower tier).
    Aggregator(u16),
}

/// One aggregator process's placement in a same-host multi-tier tree: its tier and id,
/// its collect barrier, the loopback port it binds, and where it ships its merged
/// store. Pure data so the tree wiring is unit-testable without spawning processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatorNode {
    /// 0-based aggregator tier (tier 0 here is the cells' tier-1 aggregators).
    pub tier: u32,
    /// This node's id within its tier (`0..tier_count`).
    pub id: u32,
    /// How many children (cells or lower-tier aggregators) ship to this node.
    pub child_count: u32,
    /// The loopback port this node binds (its children dial it here).
    pub bind_port: u16,
    /// Where this node ships its one merged store.
    pub ship: ShipTarget,
}

/// The full set of aggregator processes for a same-host multi-tier tree over
/// `cell_count` cells at `fanout`, each at a unique loopback `base_port + global_index`
/// port. Empty for the flat star. Round-robin at every tier: node `i` of a tier ships
/// to parent `i % parent_tier_count`, and collects the `children_of(i, tier_count,
/// child_tier_count)` children whose index is `≡ i` mod the tier count. The tier-1
/// nodes bind `base_port + id` (offset 0), so a same-host single-tier tree is
/// byte-identical to the original topology.
pub fn aggregator_nodes(cell_count: u32, fanout: u32, base_port: u16) -> Vec<AggregatorNode> {
    let tiers = tier_counts(cell_count, fanout);
    if tiers.is_empty() {
        return Vec::new();
    }
    // Prefix-sum port offsets so every node across every tier gets a distinct port.
    let mut offsets = Vec::with_capacity(tiers.len());
    let mut acc = 0u16;
    for &count in &tiers {
        offsets.push(acc);
        acc = acc.saturating_add(count as u16);
    }
    let mut nodes = Vec::new();
    for (tier, &count) in tiers.iter().enumerate() {
        // Children of this tier: the cells (tier 0) or the tier below.
        let child_tier_count = if tier == 0 {
            cell_count
        } else {
            tiers[tier - 1]
        };
        for id in 0..count {
            let ship = if tier + 1 == tiers.len() {
                ShipTarget::Controller
            } else {
                let parent_count = tiers[tier + 1];
                let parent_id = id % parent_count;
                ShipTarget::Aggregator(base_port + offsets[tier + 1] + parent_id as u16)
            };
            nodes.push(AggregatorNode {
                tier: tier as u32,
                id,
                child_count: children_of(id, count, child_tier_count),
                bind_port: base_port + offsets[tier] + id as u16,
                ship,
            });
        }
    }
    nodes
}

/// Runs this process as an aggregator: bind at the controller-assigned fixed
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
    // A k8s aggregator's 0-based tier in the multi-tier plan (default 0: the single-tier
    // tree and every same-host aggregator, which anyway carries an explicit child count).
    let tier_index: usize = std::env::var(AGG_TIER_INDEX_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let child_count: u32 = match std::env::var(AGG_CHILD_COUNT_ENV) {
        Ok(value) => value.parse().context("parsing AIPERF_AGG_CHILD_COUNT")?,
        Err(_) => {
            // k8s: the aggregators are an indexed JobSet replicatedJob sharing one env
            // template, so a per-agg static barrier cannot express an uneven round-robin
            // split. Derive it from AGG_ID + the static cell-count/fanout/tier the
            // operator injects, reusing the shared tree math so operator and aggregator
            // can never disagree. Generalizes over the tier: tier 0 collects cells, a
            // higher tier collects the tier below ([`k8s_tier_child_count`]).
            let cell_count: u32 = std::env::var(crate::cellular::partition::CELL_COUNT_ENV)
                .context(
                    "AIPERF_AGG_CHILD_COUNT unset and AIPERF_CELL_COUNT missing (k8s \
                     aggregator needs the cell count to derive its collect barrier)",
                )?
                .parse()
                .context("parsing AIPERF_CELL_COUNT")?;
            let fanout: u32 = std::env::var(CELL_AGG_FANOUT_ENV)
                .context(
                    "AIPERF_AGG_CHILD_COUNT unset and AIPERF_CELL_AGG_FANOUT missing (k8s \
                     aggregator cannot size its collect barrier)",
                )?
                .parse()
                .context("parsing AIPERF_CELL_AGG_FANOUT")?;
            k8s_tier_child_count(cell_count, fanout, tier_index, agg_id).context(
                "AIPERF_CELL_AGG_FANOUT/AIPERF_AGG_TIER_INDEX do not resolve a tier for this \
                 k8s aggregator (cannot size its collect barrier)",
            )?
        }
    };
    // This node ships its one merged store to its parent: an upper-tier aggregator when
    // [`AGG_SHIP_ADDR_ENV`] is set (a lower tier of a multi-tier tree), else the
    // controller (the top tier / single-tier tree). Only the ship target differs by
    // tier; the collect+merge half below is identical everywhere. On k8s the operator
    // injects a DNS *template* with a `{agg_id}` placeholder (the indexed job shares one
    // env), which this node resolves to its round-robin parent ([`k8s_tier_parent_id`]);
    // a same-host lower tier carries a concrete loopback addr with no placeholder.
    let ship_coordinate = match std::env::var(AGG_SHIP_ADDR_ENV) {
        Ok(addr) if !addr.is_empty() && addr.contains("{agg_id}") => {
            let cell_count: u32 = std::env::var(crate::cellular::partition::CELL_COUNT_ENV)
                .context("AIPERF_AGG_SHIP_ADDR is a template but AIPERF_CELL_COUNT is missing")?
                .parse()
                .context("parsing AIPERF_CELL_COUNT")?;
            let fanout: u32 = std::env::var(CELL_AGG_FANOUT_ENV)
                .context(
                    "AIPERF_AGG_SHIP_ADDR is a template but AIPERF_CELL_AGG_FANOUT is missing",
                )?
                .parse()
                .context("parsing AIPERF_CELL_AGG_FANOUT")?;
            let parent_id = k8s_tier_parent_id(cell_count, fanout, tier_index, agg_id).context(
                "AIPERF_AGG_SHIP_ADDR template set on the top tier (no parent to ship to)",
            )?;
            addr.replace("{agg_id}", &parent_id.to_string())
        }
        Ok(addr) if !addr.is_empty() => addr,
        _ => std::env::var(CELL_CONTROLLER_ADDR_ENV)
            .context("neither AIPERF_AGG_SHIP_ADDR nor AIPERF_CELL_CONTROLLER_ADDR set")?,
    };
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
                Some(CellMessage::Preflight { cell_id, .. }) => bail!(
                    "aggregator {agg_id} received replay preflight from cell {cell_id}; preflight belongs to the controller START barrier"
                ),
                Some(CellMessage::StorePartition(partition)) => store_partitions.push(*partition),
                Some(CellMessage::Heartbeat { cell_id, heartbeat }) => {
                    counters.issued = counters.issued.saturating_add(heartbeat.counters.issued);
                    counters.completed = counters.completed.saturating_add(heartbeat.counters.completed);
                    counters.errored = counters.errored.saturating_add(heartbeat.counters.errored);
                    heartbeats.insert(cell_id, *heartbeat);
                }
                Some(CellMessage::PhaseSignal {
                    cell_id,
                    phase,
                    signal,
                }) => bail!(
                    "aggregator {agg_id} received unexpected phase signal {signal:?} for \
                     phase {phase:?} from cell {cell_id}; controller-owned phase barriers \
                     must bypass aggregator store-merge transport"
                ),
                Some(CellMessage::Partition(_)) => bail!(
                    "aggregator {agg_id} received a raw record Partition; hierarchical merge is \
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
    let replay_cells = store_partitions
        .iter()
        .filter_map(|partition| partition.graph_supplement().cloned())
        .collect::<Vec<_>>();
    let expected = replay_cells
        .iter()
        .flat_map(|cell| cell.traces.iter())
        .map(crate::graph::supplement::ReplayTraceInstance::from)
        .collect();
    let replay_phase =
        crate::graph::supplement::merge_graph_cell_supplements(&expected, replay_cells)
            .context("folding graph replay supplements in cellular aggregator")?;
    let has_replay_supplement = !replay_phase.traces.is_empty();
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
        ship = %ship_coordinate,
        "aggregator merged its subtree; shipping one store up"
    );
    CellRecordsShipper::to_coordinate(agg_id, ship_coordinate).ship_store(
        merged.column_store().clone(),
        counters,
        epoch_ns,
        has_replay_supplement.then(|| {
            crate::graph::supplement::GraphCellSupplement::from_phase(agg_id, replay_phase)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_tile_exactly_across_aggregators() {
        for cell_count in [1_u32, 2, 5, 6, 7, 60, 1000] {
            for agg_count in 1..=cell_count.min(16) {
                let sum: u32 = (0..agg_count)
                    .map(|agg_id| children_of(agg_id, agg_count, cell_count))
                    .sum();
                assert_eq!(sum, cell_count, "cells={cell_count} aggs={agg_count}");
                for cell_id in 0..cell_count {
                    assert!(cell_id % agg_count < agg_count);
                }
            }
        }
    }

    #[test]
    fn ship_coordinate_round_robins_over_the_base_port() {
        assert_eq!(ship_coordinate(0, 2, 9700), "tcp://127.0.0.1:9700");
        assert_eq!(ship_coordinate(1, 2, 9700), "tcp://127.0.0.1:9701");
        assert_eq!(ship_coordinate(2, 2, 9700), "tcp://127.0.0.1:9700");
        assert_eq!(ship_coordinate(5, 3, 9800), "tcp://127.0.0.1:9802");
    }

    #[test]
    fn aggregator_count_selects_flat_or_tree() {
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
    fn tier_counts_reduce_by_fanout_until_top_fits() {
        // Single tier: the first reduction already lands <= fanout (the original 2-level
        // tree). Byte-identical topology, so the plan is length 1.
        assert_eq!(
            tier_counts(6, 3),
            vec![2],
            "6 cells / 3 → 2 aggregators, one tier"
        );
        assert_eq!(tier_counts(60, 8), vec![8], "60/8 = 8 <= 8, one tier");
        // Multi-tier: 8 cells / 2 → 4 → 2 → controller.
        assert_eq!(tier_counts(8, 2), vec![4, 2]);
        // Deeper: 100 cells / 3 → 34 → 12 → 4 → 2 → controller.
        assert_eq!(tier_counts(100, 3), vec![34, 12, 4, 2]);
        // Flat: fanout unset semantics (< 2) or >= cells.
        assert_eq!(
            tier_counts(6, 1),
            Vec::<u32>::new(),
            "fanout 1 is a pointless flat"
        );
        assert_eq!(
            tier_counts(6, 6),
            Vec::<u32>::new(),
            "fanout >= cells is flat"
        );
        assert_eq!(tier_counts(6, 9), Vec::<u32>::new());
    }

    #[test]
    fn tier_counts_first_equals_aggregator_count() {
        // The cell→tier-1 ship wiring keys on aggregator_count; the plan's first tier
        // must equal it so cells and the tree can never disagree on M.
        unsafe { std::env::set_var(CELL_AGG_FANOUT_ENV, "3") };
        for cells in [4u32, 6, 7, 8, 60, 100, 1000] {
            let tiers = tier_counts(cells, 3);
            if let Some(&first) = tiers.first() {
                assert_eq!(Some(first), aggregator_count(cells), "cells={cells}");
            }
            // Each tier strictly reduces and the top fits within the fanout.
            for pair in tiers.windows(2) {
                assert!(pair[1] < pair[0], "tiers must strictly reduce: {tiers:?}");
            }
            if let Some(&top) = tiers.last() {
                assert!(top <= 3, "top tier must fit the fanout: {tiers:?}");
            }
        }
        unsafe { std::env::remove_var(CELL_AGG_FANOUT_ENV) };
    }

    #[test]
    fn aggregator_nodes_wire_a_valid_reduction_tree() {
        // 8 cells, fanout 2 → tier plan [4, 2]. Tier 0 binds base+0..4, tier 1 base+4..6.
        let base = 9800u16;
        let nodes = aggregator_nodes(8, 2, base);
        assert_eq!(nodes.len(), 6, "4 + 2 aggregator processes");

        // Tier 0: four nodes, each collecting its round-robin two cells, shipping to a
        // tier-1 parent at id % 2.
        for id in 0..4u32 {
            let node = &nodes[id as usize];
            assert_eq!(node.tier, 0);
            assert_eq!(node.id, id);
            assert_eq!(node.bind_port, base + id as u16);
            assert_eq!(
                node.child_count,
                children_of(id, 4, 8),
                "8 cells over 4 tier-1 nodes"
            );
            let parent_port = base + 4 + (id % 2) as u16;
            assert_eq!(node.ship, ShipTarget::Aggregator(parent_port));
        }
        // Tier 1: two nodes shipping to the controller, each collecting its two tier-0
        // children (children_of over the 4 tier-0 nodes).
        for id in 0..2u32 {
            let node = &nodes[4 + id as usize];
            assert_eq!(node.tier, 1);
            assert_eq!(node.bind_port, base + 4 + id as u16);
            assert_eq!(node.child_count, children_of(id, 2, 4));
            assert_eq!(node.ship, ShipTarget::Controller);
        }

        // Barriers tile: every cell is collected exactly once by tier 0, every tier-0
        // node exactly once by tier 1.
        let tier0_barrier: u32 = nodes
            .iter()
            .filter(|n| n.tier == 0)
            .map(|n| n.child_count)
            .sum();
        let tier1_barrier: u32 = nodes
            .iter()
            .filter(|n| n.tier == 1)
            .map(|n| n.child_count)
            .sum();
        assert_eq!(tier0_barrier, 8, "tier-0 barriers cover all cells");
        assert_eq!(tier1_barrier, 4, "tier-1 barriers cover all tier-0 nodes");

        // Every bind port is distinct across the whole tree.
        let mut ports: Vec<u16> = nodes.iter().map(|n| n.bind_port).collect();
        ports.sort_unstable();
        ports.dedup();
        assert_eq!(
            ports.len(),
            nodes.len(),
            "aggregator bind ports must be unique"
        );
    }

    #[test]
    fn single_tier_aggregator_nodes_match_the_original_topology() {
        // 6 cells, fanout 3 → one tier of 2 nodes at base+0, base+1, both → controller,
        // each collecting children_of over the 6 cells. This is the pre-multitier layout.
        let base = 9764u16;
        let nodes = aggregator_nodes(6, 3, base);
        assert_eq!(nodes.len(), 2);
        for id in 0..2u32 {
            let node = &nodes[id as usize];
            assert_eq!(node.tier, 0);
            assert_eq!(
                node.bind_port,
                base + id as u16,
                "tier-1 binds base + id, unchanged"
            );
            assert_eq!(node.child_count, children_of(id, 2, 6));
            assert_eq!(
                node.ship,
                ShipTarget::Controller,
                "single tier ships to controller"
            );
        }
    }

    #[test]
    fn k8s_tier_derivations_match_the_same_host_tree() {
        // The k8s per-pod derivations (child barrier + parent id) must equal what
        // `aggregator_nodes` wires same-host, so the operator-built pods and the
        // controller can never disagree with the reference tree math.
        for &(cell_count, fanout) in &[(6u32, 3u32), (8, 2), (7, 3), (100, 3), (60, 8)] {
            let tiers = tier_counts(cell_count, fanout);
            let nodes = aggregator_nodes(cell_count, fanout, 9700);
            for tier_index in 0..tiers.len() {
                let count = tiers[tier_index];
                for agg_id in 0..count {
                    let node = nodes
                        .iter()
                        .find(|n| n.tier as usize == tier_index && n.id == agg_id)
                        .unwrap();
                    assert_eq!(
                        k8s_tier_child_count(cell_count, fanout, tier_index, agg_id),
                        Some(node.child_count),
                        "child_count cells={cell_count} fanout={fanout} tier={tier_index} id={agg_id}"
                    );
                    match node.ship {
                        ShipTarget::Controller => assert_eq!(
                            k8s_tier_parent_id(cell_count, fanout, tier_index, agg_id),
                            None,
                            "top tier ships to controller"
                        ),
                        ShipTarget::Aggregator(_) => {
                            let parent_count = tiers[tier_index + 1];
                            assert_eq!(
                                k8s_tier_parent_id(cell_count, fanout, tier_index, agg_id),
                                Some(agg_id % parent_count),
                            );
                        }
                    }
                }
            }
            // Out-of-range tier is None, not a panic.
            assert_eq!(
                k8s_tier_child_count(cell_count, fanout, tiers.len(), 0),
                None
            );
        }
        // Flat star: no tier resolves.
        assert_eq!(k8s_tier_child_count(6, 1, 0, 0), None);
        assert_eq!(k8s_tier_parent_id(6, 9, 0, 0), None);
    }

    #[test]
    fn effective_aggregator_count_gates_k8s_on_operator_signal() {
        assert_eq!(effective_aggregator_count(false, false, Some(2)), Some(2));
        assert_eq!(effective_aggregator_count(false, false, None), None);
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
        assert_eq!(effective_aggregator_count(true, true, None), None);
    }
}
