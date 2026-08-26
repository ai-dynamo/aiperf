// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission-strategy selector shared by the typed config model and runtime.
//!
//! `DispatchMode` is the `runtime.dispatch` selector for `workers>1` scheduled
//! execution. It is defined here in the leaf config crate so both the typed
//! Config-v2 model and `aiperf-runtime` share one serde-stable enum without a
//! dependency cycle.
//!
//! Both discriminants are closed sets (no plugin tail), so they stay enums, but
//! their serde goes through the shared [`normalize_ident`] seam: serialization
//! emits the canonical kebab-case spelling (wire-stable), while deserialization
//! accepts any case/separator variant that normalizes to it (`global-hop`,
//! `global_hop`, `GLOBAL_HOP` all decode to [`DispatchMode::GlobalHop`]).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::extensions::normalize_ident;

/// Generate `normalize_ident`-based `Serialize`/`Deserialize` for a closed
/// discriminant enum. Serialization emits the canonical spelling verbatim;
/// deserialization matches `normalize_ident(input)` against the normalized
/// canonical of each variant, so spelling/case variants are accepted without
/// widening the emitted wire form.
macro_rules! normalized_wire_enum {
    ($name:ident { $($variant:ident => $wire:literal),+ $(,)? }) => {
        impl $name {
            /// Canonical wire spelling emitted on serialization.
            pub const fn as_wire(&self) -> &'static str {
                match self { $(Self::$variant => $wire),+ }
            }

            fn from_normalized(value: &str) -> Option<Self> {
                let normalized = normalize_ident(value);
                $(if normalized == normalize_ident($wire) {
                    return Some(Self::$variant);
                })+
                None
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_wire())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = String::deserialize(deserializer)?;
                Self::from_normalized(&value).ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        concat!("unknown ", stringify!($name), " {:?}"),
                        value
                    ))
                })
            }
        }
    };
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DispatchMode {
    Sharded,
    #[default]
    Global,
    GlobalHop,
    /// One issuer stamps global order and ROUTES a credit to a worker without
    /// awaiting any individual request, after the Python `StickyCreditRouter`:
    /// the worker owns the whole round-trip and reports `FirstToken` /
    /// `CreditReturn` back out of band on one shared stream.
    ///
    /// Shares `GlobalHop`'s worker selection (sticky session binding, else
    /// least-loaded) and, like it, needs no cross-thread admission gate: a single
    /// issuer enforces the full cell-local cap directly, so aggregate concurrency
    /// and rate stay exact. Placement is as deterministic as `GlobalHop`'s: this
    /// is a router, not a shared queue workers pull from -- the issuer picks a
    /// specific worker and routes to it, exactly as `send_credit` does. The one
    /// behavioural difference is WHEN the load signal moves: `GlobalHop` holds a
    /// worker's in-flight slot from send through reply, while a credit router
    /// releases it on credit return, so `LeastLoaded` can break ties
    /// differently. Under `RoundRobin`/`Sticky` the assignment is identical.
    ///
    /// The credit itself carries only identity -- conversation, session, turn
    /// index -- and the WORKER builds the request body, exactly as Python's
    /// `Credit` does. This applies to single-turn sessions; a continuation's
    /// body can splice the live model reply, which a worker replaying the
    /// dataset cannot reproduce, so multi-turn sessions keep issuer-side
    /// materialization and stay byte-identical.
    ///
    /// # It still does NOT reach `Sharded`
    ///
    /// Measured on 144 cores against `aiperf-mock-server --fast` at ISL 550 /
    /// OSL 1 / concurrency 512: 95.6k requests/sec against `GlobalHop`'s 54.4k
    /// (+76%) and `Sharded`'s 276.8k. Removing the coordinator from each
    /// request's lifetime accounts for only a sixth of that gain, because the
    /// awaited future was never the cost; the rest came from a profile of the
    /// pegged issuer thread, which attributed its per-request CPU to dataset
    /// sampling and body materialization (~29%, now on the worker), issuance
    /// accounting (~22%, of which routing and enqueue is only ~5%), and the
    /// credit-return drain with its capture bookkeeping and metric fold (~20%).
    /// What remains is per-request work a single issuer must do however requests
    /// reach workers; `Sharded` is faster because its `W` loops each do a `1/W`
    /// share of it.
    ///
    /// Choose this mode for exact global issuance order at a much lower
    /// coordinator cost than the hop, not as a throughput mode.
    GlobalPush,
}

normalized_wire_enum!(DispatchMode {
    Sharded => "sharded",
    Global => "global",
    GlobalHop => "global-hop",
    GlobalPush => "global-push",
});

/// Worker-assignment policy applied by both single-coordinator modes when
/// `workers > 1`: [`DispatchMode::GlobalHop`]
/// (`ThreadPerCoreExecutor::execute_command`) and [`DispatchMode::GlobalPush`]
/// (`ThreadPerCoreExecutor::send_credit`), which share one `pick_worker` seam.
///
/// The choice only decides *which worker executes an already-issued request*;
/// every single-coordinator guarantee (exactly-once, deterministic merged record
/// order, aggregate concurrency/rate/arrival pattern) is coordinator-side and
/// unaffected by it, so the policy is free to trade placement determinism for
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HopRouting {
    #[default]
    RoundRobin,
    Sticky,
    LeastLoaded,
}

normalized_wire_enum!(HopRouting {
    RoundRobin => "round-robin",
    Sticky => "sticky",
    LeastLoaded => "least-loaded",
});

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
            let decoded: DispatchMode = serde_json::from_str(spelling).expect("mode deserializes");
            assert_eq!(decoded, mode);
        }
    }

    /// Omitting the selector must keep the parity-preserving default.
    #[test]
    fn default_is_global() {
        assert_eq!(DispatchMode::default(), DispatchMode::Global);
    }
}

#[cfg(test)]
mod tests {
    use super::{DispatchMode, HopRouting};

    #[test]
    fn dispatch_mode_serializes_to_canonical_kebab() {
        assert_eq!(
            serde_json::to_string(&DispatchMode::GlobalHop).unwrap(),
            "\"global-hop\""
        );
        assert_eq!(
            serde_json::to_string(&DispatchMode::Global).unwrap(),
            "\"global\""
        );
    }

    #[test]
    fn dispatch_mode_accepts_normalized_spellings() {
        for spelling in ["global-hop", "global_hop", "GLOBAL-HOP", "  Global_Hop  "] {
            let decoded: DispatchMode = serde_json::from_str(&format!("\"{spelling}\"")).unwrap();
            assert_eq!(decoded, DispatchMode::GlobalHop, "{spelling}");
        }
    }

    #[test]
    fn hop_routing_normalizes_and_round_trips() {
        for spelling in ["round-robin", "round_robin", "ROUND-ROBIN"] {
            let decoded: HopRouting = serde_json::from_str(&format!("\"{spelling}\"")).unwrap();
            assert_eq!(decoded, HopRouting::RoundRobin, "{spelling}");
        }
        assert_eq!(
            serde_json::to_string(&HopRouting::LeastLoaded).unwrap(),
            "\"least-loaded\""
        );
    }

    #[test]
    fn unknown_variant_is_rejected() {
        assert!(serde_json::from_str::<DispatchMode>("\"bogus\"").is_err());
        assert!(serde_json::from_str::<HopRouting>("\"bogus\"").is_err());
    }
}
