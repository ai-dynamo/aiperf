<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global-hop worker routing strategies

## Purpose

Make the worker-assignment policy of the `global-hop` single-dispatcher pluggable,
so a multi-worker global-hop run can honor per-session connection reuse
(`ConnectionReuseStrategy::StickyUserSessions`) instead of silently fragmenting it.

Today `ThreadPerCoreExecutor::execute_command`
(`rust/runtime/src/engine/turn_execution.rs`, the `next_worker % senders.len()`
pick) hops every issued turn round-robin to worker `i % W`. That is deterministic
and load-even, but it breaks `StickyUserSessions`: the HTTP/gRPC connection pool is
**worker-local** (`transport/http/client/pool.rs` — `Rc`/`RefCell` state) and keys a
sticky connection by `correlation_id` (`pool.rs:191-208`). Round-robin sends a
conversation's successive turns to different workers, so each worker's sticky map
independently mints its own connection for that `correlation_id` — one session ends
up with up to `W` "sticky" connections instead of one, with no error (the
"bound to a different origin" guard is within one worker's map).

## Why worker-assignment determinism is safe to trade

Multi-worker global-hop is **RealClock-only** (thread-per-core requires a real
clock; the SimClock path collapses to `workers==1`). Every guarantee global-hop
asserts is **coordinator-side**, not router-side:

- exactly-once + deterministic merged record order — from the coordinator-assigned
  `request_index` sort (`global_hop.rs` finalize).
- exact aggregate concurrency / rate / arrival pattern — the single coordinator loop
  computes `scheduled_ns` and issues before the hop.
- ISL/OSL/content multiset == `workers==1` — content, worker-independent.

The hop only chooses *which worker executes an already-issued request*, which is not
in the output and, under RealClock, does not affect timing reproducibility. So
changing the router preserves every asserted guarantee; it only changes internal
placement.

## Design

### `HopRouting` strategy (runtime protocol)

```
enum HopRouting { RoundRobin, Sticky, LeastLoaded }
```

Applied at the single pick site in `ThreadPerCoreExecutor::execute_command`. The
`correlation_id` is already available there via
`context.metadata.correlation_id` (`metrics.rs` `RequestMetricMetadata`).

- **`RoundRobin`** — `next_worker % W`, incremented per command (today's behavior).
  Deterministic, load-even. The default when connection reuse is not sticky.
- **`Sticky`** — `stable_hash(correlation_id) % W`. All of a session's turns land on
  one worker, so its worker-local sticky pool reuses the single connection keyed by
  the same `correlation_id`. Deterministic placement (same correlation → same worker
  every run). A turn with no `correlation_id` falls back to round-robin.
- **`LeastLoaded`** — pick the worker with the shallowest in-flight count, then
  **bind** that `correlation_id` to the chosen worker so continuations stay sticky
  (Python `StickyCreditRouter` shape: least-loaded for a new session, sticky for its
  continuations). In-flight counts are per-worker `Cell<usize>` on the coordinator
  (single-threaded): `+1` on send, `-1` on reply. Non-deterministic placement
  (depends on runtime queue depths) — acceptable because worker placement is not
  observable in the output.

`stable_hash` must be a fixed, seed-free hash (e.g. FNV-1a / a pinned hasher) so the
same `correlation_id` maps to the same worker across processes and runs — not
`DefaultHasher` (randomized per process).

### Selection

- CLI `--hop-routing {round-robin|sticky|least-loaded}` → `runtime.hop_routing`.
- **Auto default when unset**: `Sticky` if the resolved
  `connection_reuse == StickyUserSessions`, else `RoundRobin`. This makes the sticky
  gap impossible to misconfigure — a `StickyUserSessions` run never silently
  fragments.
- An explicit flag always overrides the auto default (including forcing
  `round-robin` under `StickyUserSessions`, if the user really wants the old
  fragmenting behavior).
- Only meaningful for `DispatchMode::GlobalHop` with `workers > 1`. Under `Sharded`
  the session→worker partition is already static (sticky is coherent there); under
  `Global` the W loops own their own pools. Ignored (with no effect) outside
  multi-worker global-hop; do not error.

## Testing

- **Unit (each router):** over a fixed sequence of `(correlation_id)` turns and
  `W`, assert the worker pick: RoundRobin = `i%W`; Sticky = `hash%W` and identical
  for repeated correlations (and stable across a re-run); LeastLoaded = shallowest
  queue then sticky-after-first (drive synthetic in-flight counts).
- **Auto-selection:** `connection_reuse=StickyUserSessions` with no flag resolves to
  `Sticky`; with `--hop-routing round-robin` resolves to `RoundRobin`; non-sticky
  reuse resolves to `RoundRobin`.
- **E2e (RealClock, mock server):** a multi-turn workload under `workers>1` global-hop
  with `StickyUserSessions` + `--hop-routing sticky` asserts, from raw records, that
  each conversation reuses **one** connection across its turns (the
  `connection_reused` record field is set on turns after the first for each
  correlation), and that round-robin on the same config does not.

## Source anchors

- Pick site: `rust/runtime/src/engine/turn_execution.rs`
  (`ThreadPerCoreExecutor::execute_command`).
- Sticky connection pool (worker-local, correlation-keyed):
  `rust/runtime/src/transport/http/client/pool.rs`,
  `rust/runtime/src/transport/grpc/models.rs` (`ConnectionReuseStrategy`).
- Correlation at the hop: `rust/runtime/src/metrics.rs` (`RequestMetricMetadata`).
- Dispatch selection / config: `rust/runtime/src/engine/protocol.rs` (`DispatchMode`),
  `rust/runtime/src/engine/protocol_v2.rs` (`parse_dispatch_mode`),
  `rust/cli/src/load.rs` (flag resolution).
- Related record: [agentic-replay-join-gating.md](agentic-replay-join-gating.md)
  (global-hop single-central-driver rationale).
