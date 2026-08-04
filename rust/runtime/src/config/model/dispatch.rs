// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Admission-strategy selector shared by the typed config model and runtime.
//!
//! `DispatchMode` is the `runtime.dispatch` selector for `workers>1` scheduled
//! execution. It is defined here in the leaf config crate so both the typed
//! Config-v2 model and `aiperf-runtime` share one serde-stable enum without a
//! dependency cycle.

use serde::{Deserialize, Serialize};

/// Admission strategy for `workers>1` scheduled execution.
///
/// - `Sharded` statically partitions request budget, concurrency, and rate
///   `1/workers`-ways up front, per worker thread.
/// - `Global` (default) admits from one shared per-cell slot pool / rate gate,
///   so aggregate concurrency and rate across all worker threads is byte-exact
///   against a single global limiter.
/// - `GlobalHop` additionally routes every individual request through one
///   coordinator-owned dispatcher, for exact request-to-thread assignment order.
///
/// That single dispatcher is a serialization point: measured against a fast
/// target it saturated near 50k requests/sec, roughly 5.6x below `Sharded` and
/// `Global`, which stayed within ~2% of each other. The ordering guarantee is
/// therefore nearly free below that rate and dominant above it. A target that
/// cannot serve the offered load hides the difference completely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DispatchMode {
    Sharded,
    #[default]
    Global,
    GlobalHop,
    /// One issuer stamps global order and PUSHES to workers without awaiting any
    /// individual request, after the Python `StickyCreditRouter`: credits go out
    /// on a queue and workers report `FirstToken`/`CreditReturn` back
    /// out-of-band.
    ///
    /// Shares `GlobalHop`'s worker selection (sticky session binding, else
    /// least-loaded) and, like it, needs no cross-thread admission gate: a single
    /// issuer enforces the full cell-local cap directly, so aggregate concurrency
    /// and rate stay exact. Placement is as deterministic as `GlobalHop`'s: this
    /// is a router, not a shared queue workers pull from -- the issuer picks a
    /// specific worker and routes to it, exactly as `send_credit` does. The one
    /// behavioural difference is WHEN the load signal moves: `GlobalHop` holds a
    /// worker's in-flight slot from send through reply, while a push issuer
    /// releases it on credit-return, so `LeastLoaded` can break ties
    /// differently. Under `RoundRobin`/`Sticky` the assignment is identical.
    ///
    /// Exists because the hop's coordinator is a serialization point --
    /// measured near 50k requests/sec against a target the other modes push
    /// past 290k on, with one thread at 1.08 cores and the other 144 idle.
    GlobalPush,
}

/// Worker-assignment policy applied at the single [`DispatchMode::GlobalHop`]
/// pick site (`ThreadPerCoreExecutor::execute_command`) when `workers > 1`.
///
/// The hop only chooses *which worker executes an already-issued request*; every
/// global-hop guarantee (exactly-once, deterministic merged record order,
/// aggregate concurrency/rate/arrival pattern) is coordinator-side and unaffected
/// by this choice, so the policy is free to trade placement determinism for
/// per-session connection reuse.
///
/// - `RoundRobin` (default) hops each issued turn to worker `i % workers` in
///   issuance order — deterministic and load-even, but it fragments a session's
///   worker-local sticky connection pool across workers.
/// - `Sticky` maps every turn of a conversation to one worker via a fixed
///   seed-free hash of its `correlation_id`, so the worker-local sticky pool
///   reuses one connection per session; a turn with no `correlation_id` falls
///   back to round-robin.
/// - `LeastLoaded` sends a new session to the worker with the shallowest in-flight
///   count, then binds that `correlation_id` to the chosen worker so its
///   continuations stay sticky.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HopRouting {
    #[default]
    RoundRobin,
    Sticky,
    LeastLoaded,
}

#[cfg(test)]
mod dispatch_mode_tests {
    use super::DispatchMode;

    /// The wire spelling is what users type and what protocol-v2 round-trips.
    /// A rename here silently changes a public CLI surface, so pin all four.
    #[test]
    fn every_mode_round_trips_its_kebab_case_spelling() {
        for (mode, spelling) in [
            (DispatchMode::Sharded, "\"sharded\""),
            (DispatchMode::Global, "\"global\""),
            (DispatchMode::GlobalHop, "\"global-hop\""),
            (DispatchMode::GlobalPush, "\"global-push\""),
        ] {
            let encoded = serde_json::to_string(&mode).expect("mode serializes");
            assert_eq!(encoded, spelling);
            let decoded: DispatchMode =
                serde_json::from_str(spelling).expect("mode deserializes");
            assert_eq!(decoded, mode);
        }
    }

    /// Omitting the selector must keep the parity-preserving default.
    #[test]
    fn default_is_global() {
        assert_eq!(DispatchMode::default(), DispatchMode::Global);
    }
}
