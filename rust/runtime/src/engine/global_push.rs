// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credit-routing (`GlobalPush`) execution for `workers > 1`.
//!
//! The Rust analogue of Python's `StickyCreditRouter`. One issuer stamps global
//! order and ROUTES a credit to a specific worker, then returns to its
//! scheduling loop; the worker owns the whole request and returns the credit
//! out of band on a single shared stream that one coordinator loop drains.
//!
//! # What it keeps from [`GlobalHop`]
//!
//! - **Exact aggregate concurrency and rate.** One issuer drives the FULL
//!   (un-thread-sliced) cell-level cap through the ordinary local `SlotPool` and
//!   per-phase rate grid, so no cross-thread [`GlobalAdmission`] gate is needed;
//!   `shared.global_admission` stays `None`. See [`global_hop`]'s module doc,
//!   which this mode shares verbatim.
//! - **Deterministic placement.** This is a router, not a queue workers pull
//!   from: [`pick_worker`] chooses one specific worker (sticky binding, else
//!   least-loaded) exactly as `send_credit` does, and the credit goes there.
//! - **Exactly-once delivery and deterministic merged record order.** Every
//!   credit is routed once and returned once, and the merge sorts by the
//!   coordinator-assigned dispatch ordinal, not by completion timing.
//!
//! # The one behaviour that legitimately differs
//!
//! [`GlobalHop`] holds a worker's in-flight slot from send THROUGH REPLY, since
//! the coordinator future that occupies the slot is the one awaiting the reply.
//! A credit router has no such future, so it releases the slot on CREDIT RETURN
//! instead. The load signal therefore moves later, and
//! [`HopRouting::LeastLoaded`] can break a tie differently. Under
//! [`HopRouting::RoundRobin`] and [`HopRouting::Sticky`] placement is identical.
//!
//! # Why the mode exists
//!
//! [`GlobalHop`]'s coordinator is a serialization point: it holds one future,
//! one reply channel, and one cancellation latch per in-flight request for that
//! request's entire lifetime. Measured on a 144-core box against
//! `aiperf-mock-server --fast` at ISL 550 / OSL 1 / concurrency 512, that capped
//! the run near 52k requests/sec with a single thread at ~1.08 cores and the
//! other 144 idle, while `sharded` and `global` ran the same workload at
//! 281k-309k. Removing the coordinator from the request's lifetime is the only
//! structural fix; the per-request allocations on that path were measured and
//! are noise by comparison.
//!
//! [`GlobalHop`]: crate::engine::protocol::DispatchMode::GlobalHop
//! [`GlobalAdmission`]: super::execute::GlobalAdmission
//! [`global_hop`]: super::global_hop
//! [`pick_worker`]: super::turn_execution::pick_worker
//! [`HopRouting::LeastLoaded`]: crate::engine::protocol::HopRouting::LeastLoaded
//! [`HopRouting::RoundRobin`]: crate::engine::protocol::HopRouting::RoundRobin
//! [`HopRouting::Sticky`]: crate::engine::protocol::HopRouting::Sticky

use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;

use crate::clock::Clock;
use crate::phase_runtime::ScheduledPhaseSidecar;

use super::execute::{ScheduledShardOutcome, ShardedShared};

/// Run the whole cell's schedule from ONE coordinator-owned issuer, routing each
/// issued turn to a worker thread as a credit returned out of band.
///
/// Signature-compatible with
/// [`run_global_hop`](super::global_hop::run_global_hop) and
/// [`run_sharded_scheduled`](super::sharded_scheduled::run_sharded_scheduled) so
/// the caller in `execute.rs` selects purely by `dispatch_mode`. The pipeline
/// body is shared with the hop
/// ([`run_single_coordinator`](super::global_hop::run_single_coordinator)); what
/// makes this mode different is chosen per phase from `shared.dispatch_mode`
/// when the phase plan is built, which is what turns on the phase's
/// credit-return loop.
pub(crate) async fn run_global_push(
    shared: Arc<ShardedShared>,
    profiling_sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    coordinator_clock: Rc<dyn Clock>,
) -> Result<ScheduledShardOutcome> {
    super::global_hop::run_single_coordinator(shared, profiling_sidecars, coordinator_clock).await
}
