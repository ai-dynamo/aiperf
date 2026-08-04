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
//! # Thin credits
//!
//! A routed credit carries only identity -- conversation, session, turn index --
//! and the WORKER builds the request body, exactly as Python's `Credit` does.
//! The issuer's `sampler.next()` still stamps global order. This applies to
//! single-turn sessions: a continuation's body can splice the live model reply,
//! which a worker replaying `build_turn_at` over the dataset cannot reproduce,
//! so multi-turn sessions keep issuer-side materialization and stay
//! byte-identical.
//!
//! # Why the mode exists, and what it is NOT
//!
//! [`GlobalHop`]'s coordinator holds one future, one reply channel, and one
//! cancellation latch per in-flight request for that request's entire lifetime,
//! and materializes every body itself. Measured on a 144-core box against
//! `aiperf-mock-server --fast` at ISL 550 / OSL 1 / concurrency 512, that capped
//! the run near 54k requests/sec with a single thread pegged at ~1.06 cores and
//! the other 144 idle, while `sharded` ran the same workload at 277k.
//!
//! This mode reaches 96k -- +76% over the hop -- but does NOT approach
//! `sharded`, and the profile says why. Removing the coordinator from the
//! request's lifetime was worth only ~6%: the awaited future was never the cost.
//! Routing and enqueue are ~5% of the pegged thread. The cost is that ONE thread
//! does every request's issuance work, and `sharded` is faster because its `W`
//! loops each do a `1/W` share of it. The remaining removable items are the
//! metric fold (~12%, would need per-worker captures merged at drain) and the
//! coordinator's discarded observer (~6%).
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
use crate::endpoints::PreparedEndpointTable;
use crate::engine::turn_execution::CreditMaterializerFactory;
use crate::multiturn::{
    NativeDatasetConversationSource, WorkerMaterializationRecipe, WorkerMaterializer,
};
use crate::phase_runtime::ScheduledPhaseSidecar;

use super::execute::{ScheduledShardOutcome, ShardedShared};

/// Hands every worker its own materializer over the one shared recipe.
///
/// The recipe is `Send + Sync` and shared; the resolver is `Rc` and must be
/// built per worker over the dense-key table that worker was handed.
struct NativeCreditMaterializerFactory {
    recipe: WorkerMaterializationRecipe,
    endpoints: Arc<crate::engine::execute::NativePreparedEndpointTableFactory>,
}

impl CreditMaterializerFactory for NativeCreditMaterializerFactory {
    fn build_worker(&self, table: PreparedEndpointTable) -> Result<WorkerMaterializer> {
        Ok(self.recipe.build(self.endpoints.resolver_over(table)?))
    }
}

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
    // Built once per run over the same partition the coordinator pipeline
    // samples from, so every conversation the issuer can draw is one every
    // worker can rebuild. Constructing a source purely for its recipe costs one
    // `lower_static_messages` pass; `NativeDataset` is handle-only, so the
    // dataset itself is shared, not copied.
    let credit_materializer: Option<Arc<dyn CreditMaterializerFactory>> = if shared.workers > 1 {
        let partition = crate::engine::sharded_scheduled::two_level_partition(
            shared.cell_id,
            shared.cells,
            0,
            1,
        )?;
        let recipe =
            NativeDatasetConversationSource::preferred_with_prepared_resolver_for_partition(
                shared.dataset.clone(),
                shared.primary_model.clone(),
                shared.default_output_tokens,
                shared.rng_root,
                &shared.samplers,
                shared.table_factory.coordinator_resolver()?,
                Some(partition),
            )?
            .with_response_tokenizer(shared.tokenizer.clone())
            .with_input_token_counter(shared.input_token_counter.clone())
            .worker_recipe();
        Some(Arc::new(NativeCreditMaterializerFactory {
            recipe,
            endpoints: shared.table_factory.clone(),
        }))
    } else {
        None
    };
    super::global_hop::run_single_coordinator(
        shared,
        profiling_sidecars,
        coordinator_clock,
        credit_materializer,
    )
    .await
}
